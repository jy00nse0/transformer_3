
import torch
import torch.nn as nn
import pickle
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

from models.model.transformer import Transformer

class MockVocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        
    def __len__(self):
        return len(self.stoi)

def load_vocab(vocab_dir):
    with open(Path(vocab_dir) / 'vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
    
    src_vocab = MockVocab(vocab_data['source_stoi'], vocab_data['source_itos'])
    trg_vocab = MockVocab(vocab_data['target_stoi'], vocab_data['target_itos'])
    return src_vocab, trg_vocab

def debug():
    vocab_dir = 'artifacts_pretrained'
    checkpoint_path = 'checkpoints/base/model_step_100000.pt'
    
    print(f"Loading vocab from {vocab_dir}...")
    source_vocab, target_vocab = load_vocab(vocab_dir)
    
    print("\nVocabulary Check:")
    print(f"Source <pad>: {source_vocab.stoi.get('<pad>')}")
    print(f"Source <sos>: {source_vocab.stoi.get('<sos>')}")
    print(f"Source <eos>: {source_vocab.stoi.get('<eos>')}")
    print(f"Target <pad>: {target_vocab.stoi.get('<pad>')}")
    print(f"Target <sos>: {target_vocab.stoi.get('<sos>')}")
    print(f"Target <eos>: {target_vocab.stoi.get('<eos>')}")
    print(f"Target 'das': {target_vocab.stoi.get('das')}")
    print(f"Target 'ist': {target_vocab.stoi.get('ist')}")
    
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    print("\nModel Config:")
    print(f"d_model: {model_config.get('d_model')}")
    print(f"n_layers: {model_config.get('n_layers')}")
    print(f"use_custom: {model_config.get('use_custom')}")
    
    # Initialize model
    model = Transformer(
        src_pad_idx=source_vocab.stoi['<pad>'],
        trg_pad_idx=target_vocab.stoi['<pad>'],
        trg_sos_idx=target_vocab.stoi['<sos>'],
        enc_voc_size=len(source_vocab),
        dec_voc_size=len(target_vocab),
        d_model=model_config['d_model'],
        n_head=model_config['n_head'],
        max_len=model_config['max_len'],
        ffn_hidden=model_config['ffn_hidden'],
        n_layers=model_config['n_layers'],
        drop_prob=0.0,
        device='cpu',
        use_custom=model_config.get('use_custom', False)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nRunning Encoder Test...")
    
    # Get some valid token IDs (skip special tokens 0-3)
    valid_ids = list(source_vocab.stoi.values())
    valid_ids = [idx for idx in valid_ids if idx > 10][:10]
    
    print(f"Testing with valid token IDs: {valid_ids[:4]}")
    
    # Check Embeddings
    print("\nChecking Embeddings...")
    print(f"Source Vocab Size: {len(source_vocab)}")
    
    try:
        tok_emb = model.encoder.emb.tok_emb(torch.tensor(valid_ids[:4]))
        print(f"Token Embedding shape: {tok_emb.shape}")
        print(f"Embedding 1 (first 5): {tok_emb[0, :5].tolist()}")
        print(f"Embedding 2 (first 5): {tok_emb[1, :5].tolist()}")
        
        emb_diff = (tok_emb[0] - tok_emb[1]).abs().sum().item()
        print(f"Embedding Diff (1 vs 2): {emb_diff:.6f}")
        
        if emb_diff < 1e-6:
            print("🚨 CRITICAL: Token Embeddings are identical! Weights might be zero or collapsed.")
            
            # Check raw weights
            weight_mean = model.encoder.emb.tok_emb.weight.mean().item()
            weight_std = model.encoder.emb.tok_emb.weight.std().item()
            print(f"Overall Embedding Weight Mean: {weight_mean:.6f}, Std: {weight_std:.6f}")
        else:
            print("✅ Token Embeddings are distinct.")
            
        # Check Layer 0 Weights (Standard PyTorch MultiheadAttention)
        print("\nChecking Encoder Layer 0 Weights (nn.MultiheadAttention)...")
        # Standard implementation uses in_proj_weight (concatenated Q,K,V)
        if hasattr(model.encoder.layers[0].attention, 'in_proj_weight'):
            attn_weight = model.encoder.layers[0].attention.in_proj_weight
            print(f"Layer 0 In-Proj Weight Shape: {attn_weight.shape}")
            print(f"Layer 0 In-Proj Weight Mean: {attn_weight.mean().item():.6f}")
            print(f"Layer 0 In-Proj Weight Std:  {attn_weight.std().item():.6f}")
            
            if attn_weight.std().item() < 1e-6:
                print("🚨 CRITICAL: Attention weights have collapsed (std ~ 0)!")
            else:
                print("✅ Attention weights look distinct.")
                
            out_weight = model.encoder.layers[0].attention.out_proj.weight
            print(f"Layer 0 Out-Proj Weight Std: {out_weight.std().item():.6f}")
        else:
            print("⚠️ Could not find in_proj_weight. Is it using custom attention?")
            print(f"Attributes: {dir(model.encoder.layers[0].attention)}")
            
    except Exception as e:
        print(f"Error checking embeddings/layers: {e}")
        import traceback
        traceback.print_exc()

    # Test 1
    src1 = torch.tensor([valid_ids[:2]]) # First 2 tokens
    try:
        mask1 = model.make_src_mask(src1)
        
        # Check Full Embedding Output
        full_emb1 = model.encoder.emb(src1)
        print(f"Full Embedding 1 (first 5): {full_emb1[0, 0, :5].tolist()}")
        
        # Check Layer 0 Output
        layer0_out1 = model.encoder.layers[0](full_emb1, mask1)
        print(f"Layer 0 Output 1 (first 5): {layer0_out1[0, 0, :5].tolist()}")
        
        enc1 = model.encoder(src1, mask1)
    except Exception as e:
        print(f"Error in Test 1: {e}")

    # Test 2
    src2 = torch.tensor([valid_ids[2:4]]) # Next 2 tokens
    try:
        mask2 = model.make_src_mask(src2)
        
        # Check Full Embedding Output
        full_emb2 = model.encoder.emb(src2)
        print(f"Full Embedding 2 (first 5): {full_emb2[0, 0, :5].tolist()}")
        
        # Check Layer 0 Output
        layer0_out2 = model.encoder.layers[0](full_emb2, mask2)
        print(f"Layer 0 Output 2 (first 5): {layer0_out2[0, 0, :5].tolist()}")
        
        emb_diff_full = (full_emb1 - full_emb2).abs().sum().item()
        print(f"\nFull Embedding Diff: {emb_diff_full:.6f}")
        
        layer0_diff = (layer0_out1 - layer0_out2).abs().sum().item()
        print(f"Layer 0 Output Diff: {layer0_diff:.6f}")
        
        enc2 = model.encoder(src2, mask2)
        
        diff = (enc1 - enc2).abs().sum().item()
        print(f"\nFinal Encoder Diff: {diff:.6f}")

            
    except Exception as e:
        print(f"Error in Test 2: {e}")

if __name__ == '__main__':
    debug()
