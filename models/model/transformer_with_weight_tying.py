"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        
        # ================================================================
        # Weight Tying (논문 필수 명세)
        # "We share the same weight matrix between the two embedding 
        #  layers and the pre-softmax linear transformation"
        # ================================================================
        
        # 1. Decoder embedding weight를 Encoder와 공유
        self.decoder.emb.tok_emb.weight = self.encoder.emb.tok_emb.weight
        
        # 2. Output linear layer weight를 Embedding과 공유
        self.decoder.linear.weight = self.encoder.emb.tok_emb.weight
        
        print("="*80)
        print("Weight Tying Enabled (Paper Implementation)")
        print("="*80)
        print(f"Shared weight shape: {self.encoder.emb.tok_emb.weight.shape}")
        print(f"  - Encoder embedding: shared")
        print(f"  - Decoder embedding: shared")
        print(f"  - Output linear: shared")
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.parameters())
        shared_params = enc_voc_size * d_model
        
        print(f"\nParameter Reduction:")
        print(f"  Without sharing: ~{(total_params + 2 * shared_params) / 1e6:.1f}M")
        print(f"  With sharing: ~{total_params / 1e6:.1f}M")
        print(f"  Saved: ~{(2 * shared_params) / 1e6:.1f}M params")
        print("="*80)

    def forward(self, src, trg):
        # 마스크 먼저 만듬
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
        
        # ================================================================
        # 수정: torch.ByteTensor 대신 device-aware 방식 사용
        # ================================================================
        # 기존 (오류):
        # trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        
        # 수정 1: 직접 device 지정
        trg_sub_mask = torch.tril(
            torch.ones(trg_len, trg_len, device=self.device)
        ).bool()
        
        # 또는 수정 2: trg와 같은 device 사용
        # trg_sub_mask = torch.tril(
        #     torch.ones(trg_len, trg_len, device=trg.device)
        # ).bool()
        
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask