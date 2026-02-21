"""
Test: Decoder self-attention masking is applied per-batch.

Verifies that the fix to decoder_layer.py correctly separates the combined
(B, 1, L, L) trg_mask into:
  - attn_mask  : (L, L) causal-only mask, same for all batches
  - key_padding_mask : (B, L) per-batch padding mask

Acceptance criteria:
  1. Two samples with DIFFERENT padding patterns produce DIFFERENT
     self-attention outputs (padding positions are masked differently).
  2. The causal constraint is respected: position i cannot attend to j > i.
  3. Existing encoder-decoder cross-attention is unaffected.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.blocks.decoder_layer import DecoderLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAD_IDX = 0


def make_trg_mask(trg, pad_idx, device):
    """Replicates Transformer.make_trg_mask() logic."""
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(3)  # (B, 1, L, 1)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(
        torch.ones(trg_len, trg_len, device=device)
    ).bool()
    return trg_pad_mask & trg_sub_mask  # (B, 1, L, L)


def make_src_mask(src, pad_idx):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Ls)


# ---------------------------------------------------------------------------
# Test 1: Per-batch padding produces different outputs
# ---------------------------------------------------------------------------

def test_per_batch_padding_differs():
    """
    Two samples in the same batch have DIFFERENT padding patterns.
    After the fix, their self-attention outputs at non-padding positions
    should differ because each sample's padding is masked independently.
    Before the fix (using trg_mask[0, 0, :, :] only), sample 1's padding
    token at position 3 would still be visible as a key, leading to
    incorrect aggregation.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    d_model = 32
    ffn_hidden = 64
    n_head = 4
    drop_prob = 0.0  # no dropout so outputs are deterministic
    L = 5  # sequence length

    layer = DecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden,
                         n_head=n_head, drop_prob=drop_prob)
    layer.eval()

    # Batch of 2 sequences with DIFFERENT padding
    # sample 0: all valid tokens [1, 2, 3, 4, 5]  -> no padding
    # sample 1: valid [1, 2, 3], then PAD [0, 0]  -> positions 3,4 are PAD
    trg_tokens = torch.tensor([[1, 2, 3, 4, 5],
                                [1, 2, 3, 0, 0]], dtype=torch.long)

    # Create a random (but shared) embedding so we can forward through layer
    emb = torch.randn(2, L, d_model)
    enc = torch.randn(2, L, d_model)
    src_mask = make_src_mask(torch.ones(2, L, dtype=torch.long), pad_idx=PAD_IDX)
    trg_mask = make_trg_mask(trg_tokens, pad_idx=PAD_IDX, device=device)

    with torch.no_grad():
        out = layer(emb, enc, trg_mask, src_mask)  # (2, L, d_model)

    # --- Assertion 1: padding positions of sample 1 are masked differently ---
    # In sample 1, positions 3 and 4 are PAD and should NOT contribute as keys.
    # Because sample 0 has no padding, if batch-0-mask were reused for sample 1,
    # those positions would NOT be masked as keys, and the output at position 2
    # (last valid token of sample 1) would aggregate contributions from positions
    # 3 and 4 — which is wrong.
    #
    # We verify this indirectly: run each sample ALONE and compare with the
    # batched output.  The single-sample run has the correct individual mask;
    # after the fix, the batched outputs must match the per-sample outputs.

    emb0 = emb[0:1]  # (1, L, d_model)
    emb1 = emb[1:2]
    enc0 = enc[0:1]
    enc1 = enc[1:2]
    trg_tokens0 = trg_tokens[0:1]
    trg_tokens1 = trg_tokens[1:2]
    src_mask0 = make_src_mask(torch.ones(1, L, dtype=torch.long), pad_idx=PAD_IDX)
    src_mask1 = src_mask0.clone()
    trg_mask0 = make_trg_mask(trg_tokens0, pad_idx=PAD_IDX, device=device)
    trg_mask1 = make_trg_mask(trg_tokens1, pad_idx=PAD_IDX, device=device)

    with torch.no_grad():
        out0 = layer(emb0, enc0, trg_mask0, src_mask0)  # (1, L, d_model)
        out1 = layer(emb1, enc1, trg_mask1, src_mask1)  # (1, L, d_model)

    # Batched output for sample 0 should match single-sample output for sample 0
    assert torch.allclose(out[0], out0[0], atol=1e-5), \
        "Batched output for sample 0 differs from single-sample output"

    # Batched output for sample 1 should match single-sample output for sample 1
    assert torch.allclose(out[1], out1[0], atol=1e-5), \
        "Batched output for sample 1 differs from single-sample output — " \
        "per-batch padding masks are NOT being applied correctly"

    print("✓ test_per_batch_padding_differs PASSED")
    print("  Batched outputs match per-sample outputs for both samples.")
    return True


# ---------------------------------------------------------------------------
# Test 2: Causal constraint is respected
# ---------------------------------------------------------------------------

def test_causal_masking():
    """
    Position i must not receive information from position j > i.
    We verify by checking that zeroing out later embedding positions
    does NOT change the output at earlier positions.
    """
    torch.manual_seed(7)
    device = torch.device("cpu")

    d_model = 32
    ffn_hidden = 64
    n_head = 4
    drop_prob = 0.0
    L = 6

    layer = DecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden,
                         n_head=n_head, drop_prob=drop_prob)
    layer.eval()

    trg_tokens = torch.ones(1, L, dtype=torch.long)  # all valid
    emb = torch.randn(1, L, d_model)
    enc = torch.randn(1, L, d_model)
    src_mask = make_src_mask(torch.ones(1, L, dtype=torch.long), pad_idx=PAD_IDX)
    trg_mask = make_trg_mask(trg_tokens, pad_idx=PAD_IDX, device=device)

    # Base output
    with torch.no_grad():
        base_out = layer(emb, enc, trg_mask, src_mask)  # (1, L, d_model)

    # Perturb future positions (positions 3, 4, 5) and check positions 0, 1, 2
    emb_perturbed = emb.clone()
    emb_perturbed[0, 3:, :] = torch.randn(L - 3, d_model)

    with torch.no_grad():
        perturbed_out = layer(emb_perturbed, enc, trg_mask, src_mask)

    assert torch.allclose(base_out[0, :3], perturbed_out[0, :3], atol=1e-5), \
        "Causal masking broken: earlier positions are affected by later positions"

    print("✓ test_causal_masking PASSED")
    print("  Earlier positions are unaffected by changes to future positions.")
    return True


# ---------------------------------------------------------------------------
# Test 3: trg_mask=None does not crash
# ---------------------------------------------------------------------------

def test_no_mask():
    """DecoderLayer forward should work when trg_mask is None."""
    torch.manual_seed(0)
    d_model = 16
    layer = DecoderLayer(d_model=d_model, ffn_hidden=32, n_head=2, drop_prob=0.0)
    layer.eval()

    dec = torch.randn(2, 4, d_model)
    enc = torch.randn(2, 4, d_model)

    with torch.no_grad():
        out = layer(dec, enc, trg_mask=None, src_mask=None)

    assert out.shape == (2, 4, d_model), "Unexpected output shape with trg_mask=None"
    print("✓ test_no_mask PASSED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Decoder Self-Attention Masking Tests")
    print("=" * 70)

    results = []
    for test_fn in [test_per_batch_padding_differs, test_causal_masking, test_no_mask]:
        try:
            ok = test_fn()
            results.append(ok)
        except AssertionError as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            results.append(False)
        except Exception as e:
            print(f"✗ {test_fn.__name__} ERROR: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    if all(results):
        print("✓ All tests PASSED")
        sys.exit(0)
    else:
        print(f"✗ {results.count(False)}/{len(results)} test(s) FAILED")
        sys.exit(1)
