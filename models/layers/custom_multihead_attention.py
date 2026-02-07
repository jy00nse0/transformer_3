"""
Custom MultiheadAttention for Table 3 Experiments
Allows independent control of d_k and d_v (key/value dimensions per head)

Based on PyTorch's nn.MultiheadAttention but modified to support:
- Custom d_k (key dimension per head)
- Custom d_v (value dimension per head)
- Table 3(B) experiments: varying d_k and d_v independently
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_


class CustomMultiheadAttention(nn.Module):
    """
    Custom Multi-Head Attention with configurable d_k and d_v
    
    Key differences from PyTorch's implementation:
    1. d_k and d_v can be set independently
    2. Projection matrices sized based on custom dimensions
    3. Supports Table 3(B) experiments from "Attention Is All You Need"
    
    Args:
        embed_dim: Total dimension of the model (d_model)
        num_heads: Number of parallel attention heads (h)
        d_k: Dimension per head for keys and queries (default: embed_dim // num_heads)
        d_v: Dimension per head for values (default: embed_dim // num_heads)
        dropout: Dropout probability
        bias: Whether to use bias in projections
        batch_first: If True, input/output is (batch, seq, feature)
    """
    
    def __init__(
        self,
        embed_dim,
        num_heads,
        d_k=None,
        d_v=None,
        dropout=0.0,
        bias=True,
        batch_first=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.dropout = dropout
        
        # Set d_k and d_v (논문의 key/value dimension per head)
        self.d_k = d_k if d_k is not None else embed_dim // num_heads
        self.d_v = d_v if d_v is not None else embed_dim // num_heads
        
        # Total dimensions after multi-head projection
        self.d_k_total = self.d_k * num_heads
        self.d_v_total = self.d_v * num_heads
        
        # ================================================================
        # Projection Matrices
        # ================================================================
        
        # Q projection: embed_dim → d_k * num_heads
        self.q_proj = nn.Linear(embed_dim, self.d_k_total, bias=bias, **factory_kwargs)
        
        # K projection: embed_dim → d_k * num_heads
        self.k_proj = nn.Linear(embed_dim, self.d_k_total, bias=bias, **factory_kwargs)
        
        # V projection: embed_dim → d_v * num_heads
        self.v_proj = nn.Linear(embed_dim, self.d_v_total, bias=bias, **factory_kwargs)
        
        # Output projection: d_v * num_heads → embed_dim
        self.out_proj = nn.Linear(self.d_v_total, embed_dim, bias=bias, **factory_kwargs)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform"""
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
            constant_(self.k_proj.bias, 0.0)
            constant_(self.v_proj.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
    
    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
    ):
        """
        Forward pass
        
        Args:
            query: (batch, seq_len, embed_dim) if batch_first=True
            key: (batch, seq_len, embed_dim) if batch_first=True
            value: (batch, seq_len, embed_dim) if batch_first=True
            key_padding_mask: (batch, seq_len) - True means ignore
            need_weights: Whether to return attention weights
            attn_mask: (seq_len, seq_len) or (batch*num_heads, seq_len, seq_len)
        
        Returns:
            attn_output: (batch, seq_len, embed_dim)
            attn_weights: (batch, num_heads, seq_len, seq_len) if need_weights
        """
        
        # Handle batch_first and extract sequence lengths for each input
        # Important: In cross-attention, query/key/value can have different seq lengths
        if self.batch_first:
            # Input: (batch, seq, feature)
            batch_size, q_seq_len, _ = query.shape
            _, k_seq_len, _ = key.shape
            _, v_seq_len, _ = value.shape
        else:
            # Input: (seq, batch, feature) → transpose
            q_seq_len, batch_size, _ = query.shape
            k_seq_len, _, _ = key.shape
            v_seq_len, _, _ = value.shape
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # ================================================================
        # 1. Linear Projections
        # ================================================================
        
        # Q: (batch, seq_len, d_k * num_heads)
        Q = self.q_proj(query)
        # K: (batch, seq_len, d_k * num_heads)
        K = self.k_proj(key)
        # V: (batch, seq_len, d_v * num_heads)
        V = self.v_proj(value)
        
        # ================================================================
        # 2. Reshape for Multi-Head
        # ================================================================
        
        # Q: (batch, q_seq_len, num_heads, d_k) → (batch, num_heads, q_seq_len, d_k)
        Q = Q.view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # K: (batch, k_seq_len, num_heads, d_k) → (batch, num_heads, k_seq_len, d_k)
        K = K.view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # V: (batch, v_seq_len, num_heads, d_v) → (batch, num_heads, v_seq_len, d_v)
        V = V.view(batch_size, v_seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # ================================================================
        # 3. Scaled Dot-Product Attention
        # ================================================================
        
        # Attention scores: Q @ K^T / sqrt(d_k)
        # (batch, num_heads, seq_len, d_k) @ (batch, num_heads, d_k, seq_len)
        # → (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply attention mask (causal mask for decoder)
        if attn_mask is not None:
            # attn_mask: (seq_len, seq_len) - float mask
            # Expand to (batch, num_heads, seq_len, seq_len)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_mask = attn_mask.expand(batch_size, self.num_heads, -1, -1)
            
            scores = scores + attn_mask
        
        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq_len) - True means ignore
            # Expand to (batch, 1, 1, seq_len)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            
            # Set ignored positions to -inf
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, d_v)
        # → (batch, num_heads, seq_len, d_v)
        attn_output = torch.matmul(attn_weights, V)
        
        # ================================================================
        # 4. Concatenate Heads
        # ================================================================
        
        # (batch, num_heads, seq_len, d_v) → (batch, seq_len, num_heads, d_v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # (batch, q_seq_len, num_heads, d_v) → (batch, q_seq_len, d_v * num_heads)
        attn_output = attn_output.view(batch_size, q_seq_len, self.d_v_total)
        
        # ================================================================
        # 5. Output Projection
        # ================================================================
        
        # (batch, seq_len, d_v * num_heads) → (batch, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        # Handle batch_first
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        # Return
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output, None


# ============================================================================
# 테스트 코드
# ============================================================================

def test_custom_attention():
    """커스텀 attention 테스트"""
    
    print("="*80)
    print("Testing CustomMultiheadAttention")
    print("="*80)
    
    batch_size = 4
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    
    # 입력 생성
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # ================================================================
    # Test 1: 기본 설정 (d_k = d_v = 64)
    # ================================================================
    print("\nTest 1: Default (d_k=64, d_v=64)")
    attn_default = CustomMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True
    )
    
    output, _ = attn_default(x, x, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    print("✓ Test 1 passed")
    
    # ================================================================
    # Test 2: Table 3(B) - d_k=16, d_v=64
    # ================================================================
    print("\nTest 2: Table 3(B) - d_k=16, d_v=64")
    attn_table3b_16 = CustomMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        d_k=16,
        d_v=64,
        batch_first=True
    )
    
    output, _ = attn_table3b_16(x, x, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    print("✓ Test 2 passed")
    
    # ================================================================
    # Test 3: Table 3(B) - d_k=32, d_v=64
    # ================================================================
    print("\nTest 3: Table 3(B) - d_k=32, d_v=64")
    attn_table3b_32 = CustomMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        d_k=32,
        d_v=64,
        batch_first=True
    )
    
    output, _ = attn_table3b_32(x, x, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    print("✓ Test 3 passed")
    
    # ================================================================
    # Test 4: Mask 테스트
    # ================================================================
    print("\nTest 4: With attention mask")
    
    # Causal mask
    attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    output, attn_weights = attn_default(x, x, x, attn_mask=attn_mask, need_weights=True)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print("✓ Test 4 passed")
    
    # ================================================================
    # Test 5: Parameter count
    # ================================================================
    print("\nTest 5: Parameter count comparison")
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    params_default = count_params(attn_default)
    params_16 = count_params(attn_table3b_16)
    params_32 = count_params(attn_table3b_32)
    
    print(f"Default (d_k=64, d_v=64): {params_default:,} params")
    print(f"Table3B (d_k=16, d_v=64): {params_16:,} params")
    print(f"Table3B (d_k=32, d_v=64): {params_32:,} params")
    print(f"Reduction (d_k=16): {(1 - params_16/params_default)*100:.1f}%")
    print(f"Reduction (d_k=32): {(1 - params_32/params_default)*100:.1f}%")
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)


if __name__ == '__main__':
    test_custom_attention()
