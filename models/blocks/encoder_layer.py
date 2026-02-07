"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, kdim=None):
        super(EncoderLayer, self).__init__()
        
        # 1. 파이토치 내장 MultiheadAttention
        # batch_first=True로 설정하면 (batch, seq, feature) 순서를 유지할 수 있습니다.
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_head, 
            dropout=drop_prob, 
            batch_first=True,
            kdim=kdim,
            vdim=kdim
        )
        
        # 2. 파이토치 내장 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. 파이토치 모듈 조합으로 만든 FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )
        
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # =================================================================
        # 1. Self Attention
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
        
        # Self Attention 수행
        x, _ = self.attention(
            query=x, 
            key=x, 
            value=x, 
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # =================================================================
        # 2. Add & Norm
        # =================================================================
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        
        # =================================================================
        # 3. Feed Forward
        # =================================================================
        residual = x
        x = self.ffn(x)
        
        # =================================================================
        # 4. Add & Norm
        # =================================================================
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        
        return x
