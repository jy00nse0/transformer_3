"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnn852
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.embedding import Embedding


class Decoder(nn.Module):

    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers,
                 drop_prob, device, gradient_checkpointing=False,
                 use_custom=False, d_k=None, d_v=None):
        super().__init__()
        self.emb = Embedding(d_model=d_model,
                             drop_prob=drop_prob,
                             max_len=max_len,
                             vocab_size=dec_voc_size,
                             device=device)

        # ================================================================
        # use_custom 플래그에 따라 레이어 클래스를 분기
        # ================================================================
        if use_custom:
            from models.blocks.custom_layers import CustomDecoderLayer
            self.layers = nn.ModuleList([
                CustomDecoderLayer(d_model=d_model,
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
                DecoderLayer(d_model=d_model,
                             ffn_hidden=ffn_hidden,
                             n_head=n_head,
                             drop_prob=drop_prob)
                for _ in range(n_layers)
            ])

        self.linear = nn.Linear(d_model, dec_voc_size)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                trg = checkpoint(layer, trg, enc_src, trg_mask, src_mask, use_reentrant=False)
            else:
                trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output