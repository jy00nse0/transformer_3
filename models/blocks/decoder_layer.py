import torch
from torch import nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, kdim=None):
        super(DecoderLayer, self).__init__()
        
        # 1. Masked Self-Attention (디코더 스스로를 쳐다봄)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_head, 
            dropout=drop_prob, 
            batch_first=True,
            kdim=kdim,
            vdim=kdim
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # 2. Encoder-Decoder Attention (인코더의 정보를 가져옴)
        self.enc_dec_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_head, 
            dropout=drop_prob, 
            batch_first=True,
            kdim=kdim,
            vdim=kdim
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # 3. Positionwise Feed Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # =================================================================
        # 1. Masked Self Attention (Look-ahead masking)
        # =================================================================
        residual = dec
        
        # trg_mask 처리
        # Transformer.make_trg_mask()는 (B, 1, L, L) 형태
        # - Causal mask (look-ahead 방지) + Padding mask 결합
        # PyTorch MultiheadAttention의 attn_mask는:
        #   - 형태: (L, L) 또는 (B*num_heads, L, L)
        #   - 타입: Float (0.0 = 허용, -inf = 무시)
        
        if trg_mask is not None:
            # (B, 1, L, L) -> (L, L)
            # 모든 배치에 대해 동일한 causal mask 사용
            attn_mask = trg_mask[0, 0, :, :]  # (L, L)
            
            # Boolean -> Float 변환
            # True (허용) -> 0.0, False (무시) -> -inf
            attn_mask = attn_mask.float()
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, float(0.0))
        else:
            attn_mask = None
        
        # Self Attention 수행
        x, _ = self.self_attention(
            query=dec, 
            key=dec, 
            value=dec, 
            attn_mask=attn_mask,
            need_weights=False
        )
        
        # Add & Norm
        x = self.norm1(residual + self.dropout1(x))

        # =================================================================
        # 2. Encoder-Decoder Attention (Cross Attention)
        # =================================================================
        residual = x
        
        # src_mask 처리
        # Transformer.make_src_mask()는 (B, 1, 1, L) 형태
        # PyTorch MultiheadAttention의 key_padding_mask는:
        #   - 형태: (B, L)
        #   - 타입: Boolean (True = 무시, False = 허용)
        
        if src_mask is not None:
            # (B, 1, 1, L) -> (B, L)
            key_padding_mask = src_mask.squeeze(1).squeeze(1)
            
            # 의미 반전 필요!
            # Transformer: True = 유효, False = 패딩
            # PyTorch: True = 패딩(무시), False = 유효
            key_padding_mask = ~key_padding_mask
        else:
            key_padding_mask = None
        
        # Encoder-Decoder Attention 수행
        # Query: 디코더 출력 (x)
        # Key, Value: 인코더 출력 (enc)
        x, _ = self.enc_dec_attention(
            query=x, 
            key=enc, 
            value=enc, 
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Add & Norm
        x = self.norm2(residual + self.dropout2(x))

        # =================================================================
        # 3. Positionwise Feed Forward
        # =================================================================
        residual = x
        x = self.ffn(x)
        
        # Add & Norm
        x = self.norm3(residual + self.dropout3(x))
        
        return x
