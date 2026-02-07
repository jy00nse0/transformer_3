"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnn852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size,
                 d_model, n_head, max_len, ffn_hidden, n_layers, drop_prob, device,
                 gradient_checkpointing=False,
                 use_custom=False, d_k=None, d_v=None):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        # ================================================================
        # Encoder / Decoder 생성
        # use_custom=True 시 d_k, d_v를 함께 전달
        # ================================================================
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device,
                               gradient_checkpointing=gradient_checkpointing,
                               use_custom=use_custom,
                               d_k=d_k,
                               d_v=d_v)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device,
                               gradient_checkpointing=gradient_checkpointing,
                               use_custom=use_custom,
                               d_k=d_k,
                               d_v=d_v)

        # ================================================================
        # Weight Tying (논문 필수 명세)
        # "We share the same weight matrix between the two embedding
        #  layers and the pre-softmax linear transformation"
        # ================================================================
        self.decoder.emb.tok_emb.weight = self.encoder.emb.tok_emb.weight
        self.decoder.linear.weight = self.encoder.emb.tok_emb.weight

        # ================================================================
        # 초기화 정보 출력
        # ================================================================
        print("=" * 80)
        print("Weight Tying Enabled (Paper Implementation)")
        print("=" * 80)
        print(f"Shared weight shape: {self.encoder.emb.tok_emb.weight.shape}")
        print(f"  - Encoder embedding: shared")
        print(f"  - Decoder embedding: shared")
        print(f"  - Output linear:     shared")

        # ---- Attention 모드 출력 ----
        if use_custom:
            # d_k / d_v가 None이면 실제로 사용되는 기본값을 표시
            effective_d_k = d_k if d_k is not None else (d_model // n_head)
            effective_d_v = d_v if d_v is not None else (d_model // n_head)
            print(f"\nAttention Mode: Custom MultiheadAttention")
            print(f"  - d_k (per head): {effective_d_k}")
            print(f"  - d_v (per head): {effective_d_v}")
            print(f"  - d_k * n_head:   {effective_d_k * n_head}")
            print(f"  - d_v * n_head:   {effective_d_v * n_head}")
        else:
            print(f"\nAttention Mode: Standard nn.MultiheadAttention")
            print(f"  - d_k = d_v = d_model / n_head = {d_model // n_head}")

        # ---- Gradient Checkpointing ----
        if gradient_checkpointing:
            print(f"\nGradient Checkpointing: ENABLED")
            print(f"  - Memory savings: ~50-70%")
            print(f"  - Slight compute overhead: ~10-20%")

        # ---- 파라미터 수 계산 ----
        total_params = sum(p.numel() for p in self.parameters())
        shared_params = enc_voc_size * d_model

        print(f"\nParameter Count:")
        print(f"  Without sharing: ~{(total_params + 2 * shared_params) / 1e6:.1f}M")
        print(f"  With sharing:    ~{total_params / 1e6:.1f}M")
        print(f"  Saved:           ~{(2 * shared_params) / 1e6:.1f}M params")
        print("=" * 80)

    # ================================================================
    # Forward
    # ================================================================
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(
            torch.ones(trg_len, trg_len, device=self.device)
        ).bool()

        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask