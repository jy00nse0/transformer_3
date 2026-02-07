"""
Custom Encoder/Decoder Layers with configurable d_k and d_v
For Table 3 experiments in "Attention Is All You Need"
"""

import torch
from torch import nn
from models.layers.custom_multihead_attention import CustomMultiheadAttention


class CustomEncoderLayer(nn.Module):
    """
    Encoder Layer with custom d_k and d_v support
    """
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, d_k=None, d_v=None):
        super().__init__()
        
        # Custom MultiheadAttention with configurable d_k and d_v
        self.attention = CustomMultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=drop_prob,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )
        
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
    
    def forward(self, x, src_mask):
        # Self Attention
        residual = x
        
        # Handle mask
        if src_mask is not None:
            # (B, 1, 1, L) → (B, L)
            key_padding_mask = src_mask.squeeze(1).squeeze(1)
            # Invert: Transformer uses True=valid, PyTorch uses True=ignore
            key_padding_mask = ~key_padding_mask
        else:
            key_padding_mask = None
        
        # Attention
        x, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Add & Norm
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        
        # FFN
        residual = x
        x = self.ffn(x)
        
        # Add & Norm
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        
        return x


class CustomDecoderLayer(nn.Module):
    """
    Decoder Layer with custom d_k and d_v support
    """
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, d_k=None, d_v=None):
        super().__init__()
        
        # 1. Masked Self-Attention
        self.self_attention = CustomMultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=drop_prob,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        # 2. Encoder-Decoder Attention
        self.enc_dec_attention = CustomMultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=drop_prob,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        # 3. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)
    
    def forward(self, dec, enc, trg_mask, src_mask):
        # ================================================================
        # 1. Masked Self Attention
        # ================================================================
        residual = dec
        
        # Handle trg_mask (causal mask)
        if trg_mask is not None:
            # (B, 1, L, L) → (L, L)
            attn_mask = trg_mask[0, 0, :, :]
            
            # Boolean → Float
            attn_mask = attn_mask.float()
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, float(0.0))
        else:
            attn_mask = None
        
        # Self Attention
        x, _ = self.self_attention(
            query=dec,
            key=dec,
            value=dec,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        # Add & Norm
        x = self.norm1(residual + self.dropout1(x))
        
        # ================================================================
        # 2. Encoder-Decoder Attention
        # ================================================================
        residual = x
        
        # Handle src_mask (padding mask)
        if src_mask is not None:
            # (B, 1, 1, L) → (B, L)
            key_padding_mask = src_mask.squeeze(1).squeeze(1)
            # Invert
            key_padding_mask = ~key_padding_mask
        else:
            key_padding_mask = None
        
        # Cross Attention
        x, _ = self.enc_dec_attention(
            query=x,
            key=enc,
            value=enc,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Add & Norm
        x = self.norm2(residual + self.dropout2(x))
        
        # ================================================================
        # 3. FFN
        # ================================================================
        residual = x
        x = self.ffn(x)
        
        # Add & Norm
        x = self.norm3(residual + self.dropout3(x))
        
        return x


# ============================================================================
# 사용 예시
# ============================================================================

def example_usage():
    """
    Table 3(B) 실험 예시
    """
    
    batch_size = 32
    seq_len = 64
    d_model = 512
    ffn_hidden = 2048
    n_head = 8
    drop_prob = 0.1
    
    # 입력
    x = torch.randn(batch_size, seq_len, d_model)
    
    # ================================================================
    # Base model (d_k=64, d_v=64)
    # ================================================================
    print("Base Model (d_k=64, d_v=64):")
    encoder_base = CustomEncoderLayer(
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        n_head=n_head,
        drop_prob=drop_prob,
        d_k=64,
        d_v=64
    )
    
    output = encoder_base(x, src_mask=None)
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder_base.parameters()):,}")
    
    # ================================================================
    # Table 3(B) - d_k=16
    # ================================================================
    print("\nTable 3(B) - d_k=16:")
    encoder_dk16 = CustomEncoderLayer(
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        n_head=n_head,
        drop_prob=drop_prob,
        d_k=16,  # ← 변경
        d_v=64
    )
    
    output = encoder_dk16(x, src_mask=None)
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder_dk16.parameters()):,}")
    
    # ================================================================
    # Table 3(B) - d_k=32
    # ================================================================
    print("\nTable 3(B) - d_k=32:")
    encoder_dk32 = CustomEncoderLayer(
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        n_head=n_head,
        drop_prob=drop_prob,
        d_k=32,  # ← 변경
        d_v=64
    )
    
    output = encoder_dk32(x, src_mask=None)
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder_dk32.parameters()):,}")


if __name__ == '__main__':
    example_usage()
