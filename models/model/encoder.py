"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnn852
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.embedding import Embedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers,
                 drop_prob, device, gradient_checkpointing=False,
                 use_custom=False, d_k=None, d_v=None):
        super().__init__()
        self.emb = Embedding(d_model=d_model,
                             max_len=max_len,
                             vocab_size=enc_voc_size,
                             drop_prob=drop_prob,
                             device=device)

        # ================================================================
        # use_custom 플래그에 따라 레이어 클래스를 분기
        # ================================================================
        if use_custom:
            from models.blocks.custom_layers import CustomEncoderLayer
            self.layers = nn.ModuleList([
                CustomEncoderLayer(d_model=d_model,
                                   ffn_hidden=ffn_hidden,
                                   n_head=n_head,
                                   drop_prob=drop_prob,
                                   d_k=d_k,
                                   d_v=d_v)
                for _ in range(n_layers)
            ])
        else:
            # 기존 standard layer (kdim 파라미터 제거)
            self.layers = nn.ModuleList([
                EncoderLayer(d_model=d_model,
                             ffn_hidden=ffn_hidden,
                             n_head=n_head,
                             drop_prob=drop_prob)
                for _ in range(n_layers)
            ])

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, src_mask, use_reentrant=False)
            else:
                x = layer(x, src_mask)

        return x