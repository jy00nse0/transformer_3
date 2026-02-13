"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    """
    Xavier weight initialization for Transformer model
    Following "Attention is All You Need" paper specifications
    
    Rules:
    - nn.Linear: Xavier Uniform (prevents variance explosion in QK^T and FFN)
    - nn.Embedding: Normal with std = d_model^(-0.5) as per paper
    - nn.LayerNorm: Weight=1, Bias=0 (default, DO NOT change)
    - nn.MultiheadAttention: Xavier for internal weights
    """
    from torch import nn
    
    # 1. Linear Layer (Attention의 Q,K,V,O 프로젝션 및 FFN)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    # 2. Embedding Layer (Source & Target Token Embeddings)
    elif isinstance(m, nn.Embedding):
        # 논문에서는 std = d_model ** -0.5 로 초기화 (Xavier와 유사)
        nn.init.normal_(m.weight, mean=0, std=m.embedding_dim ** -0.5)
        # 만약 Padding Index가 있다면 0으로 고정
        if m.padding_idx is not None:
            nn.init.constant_(m.weight[m.padding_idx], 0)
    
    # 3. LayerNorm (절대 건드리면 안 되는 곳, 혹은 1/0으로 강제)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)  # Gamma
        nn.init.constant_(m.bias, 0.0)    # Beta
    
    # 4. MultiheadAttention (PyTorch 내장 모듈 사용 시 필수)
    elif isinstance(m, nn.MultiheadAttention):
        # Q, K, V가 하나로 뭉쳐있는 in_proj_weight 처리
        if m.in_proj_weight is not None:
            nn.init.xavier_uniform_(m.in_proj_weight)
        
        # Output Projection 처리
        if m.out_proj is not None:
            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.constant_(m.out_proj.bias, 0)

