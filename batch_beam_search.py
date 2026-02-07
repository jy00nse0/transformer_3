"""
Batch Beam Search Implementation
배치 단위 Beam Search 디코더

Following "Attention Is All You Need" paper specifications:
- Beam size: 4
- Length penalty: α = 0.6
- Max output length: input_length + 50
- Early termination when possible
"""

import torch
import torch.nn.functional as F


class BatchBeamSearch:
    """
    배치 단위 Beam Search Decoder
    
    핵심 개선:
    - 배치 내 모든 샘플을 동시에 처리
    - Encoder는 1회만 실행 (배치 단위)
    - Decoder도 배치 단위로 실행
    - GPU 활용도 극대화
    """
    
    def __init__(self, model, beam_size=4, max_len=None, length_penalty=0.6,
                 sos_idx=2, eos_idx=3, pad_idx=1):
        """
        Args:
            model: Transformer model
            beam_size: Beam width (default: 4, paper setting)
            max_len: Maximum output length (default: input_len + 50)
            length_penalty: Length penalty alpha (default: 0.6, paper setting)
            sos_idx: Start of sentence token index
            eos_idx: End of sentence token index
            pad_idx: Padding token index
        """
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.device = model.device
    
    def search(self, src, src_lengths=None):
        """
        배치 단위 Beam Search
        
        Args:
            src: (batch_size, src_len) source sequences
            src_lengths: (batch_size,) actual lengths (optional)
        
        Returns:
            best_sequences: (batch_size, max_out_len) best decoded sequences
            best_scores: (batch_size,) scores of best sequences
        """
        batch_size = src.size(0)
        
        # ================================================================
        # 1. Encoder (배치 처리 - 1회만 실행!)
        # ================================================================
        src_mask = self.model.make_src_mask(src)
        enc_src = self.model.encoder(src, src_mask)
        # enc_src: (batch_size, src_len, d_model)
        
        # ================================================================
        # 2. Max output length 결정
        # ================================================================
        if src_lengths is not None:
            max_src_len = src_lengths.max().item()
        else:
            max_src_len = src.size(1)
        
        # max_len is treated as OFFSET (how much to add to input length)
        offset = self.max_len if self.max_len is not None else 50
        max_output_len = max_src_len + offset
        
        # ================================================================
        # 3. Beam 초기화 (모든 배치에 대해)
        # ================================================================
        # beams: (batch_size, beam_size, current_len)
        beams = torch.full(
            (batch_size, self.beam_size, 1),
            self.sos_idx,
            dtype=torch.long,
            device=self.device
        )
        
        # Beam scores: (batch_size, beam_size)
        beam_scores = torch.zeros(batch_size, self.beam_size, device=self.device)
        beam_scores[:, 1:] = -1e9  # 첫 번째 beam만 활성화
        
        # Finished flags: (batch_size, beam_size)
        is_finished = torch.zeros(
            batch_size, self.beam_size,
            dtype=torch.bool,
            device=self.device
        )
        
        # ================================================================
        # 4. Beam Search Loop
        # ================================================================
        for step in range(max_output_len - 1):
            # 모든 배치의 모든 beam이 끝났는지 확인
            if is_finished.all():
                break
            
            current_len = beams.size(2)
            
            # ---- 4.1. Decoder Input 준비 ----
            # beams: (batch_size, beam_size, current_len)
            # → (batch_size * beam_size, current_len)
            decoder_input = beams.view(batch_size * self.beam_size, current_len)
            
            # ---- 4.2. Encoder output 확장 ----
            # enc_src: (batch_size, src_len, d_model)
            # → (batch_size, beam_size, src_len, d_model)
            # → (batch_size * beam_size, src_len, d_model)
            enc_src_expanded = enc_src.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
            enc_src_expanded = enc_src_expanded.view(
                batch_size * self.beam_size,
                enc_src.size(1),
                enc_src.size(2)
            )
            
            # ---- 4.3. Source mask 확장 ----
            # src_mask: (batch_size, 1, 1, src_len)
            # → (batch_size * beam_size, 1, 1, src_len)
            src_mask_expanded = src_mask.unsqueeze(1).repeat(1, self.beam_size, 1, 1, 1)
            src_mask_expanded = src_mask_expanded.view(
                batch_size * self.beam_size,
                src_mask.size(1),
                src_mask.size(2),
                src_mask.size(3)
            )
            
            # ---- 4.4. Target mask 생성 ----
            trg_mask = self.model.make_trg_mask(decoder_input)
            
            # ---- 4.5. Decoder Forward (배치 처리!) ----
            output = self.model.decoder(
                decoder_input,
                enc_src_expanded,
                trg_mask,
                src_mask_expanded
            )
            # output: (batch_size * beam_size, current_len, vocab_size)
            
            # ---- 4.6. Next token probabilities ----
            next_token_logits = output[:, -1, :]  # (batch_size * beam_size, vocab_size)
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            # Reshape: (batch_size, beam_size, vocab_size)
            next_token_log_probs = next_token_log_probs.view(
                batch_size, self.beam_size, -1
            )
            
            vocab_size = next_token_log_probs.size(-1)
            
            # ---- 4.7. Finished beam 처리 ----
            # Finished beams는 pad token만 허용
            next_token_log_probs[is_finished, :] = -1e9
            next_token_log_probs[is_finished, self.pad_idx] = 0
            
            # ---- 4.8. Candidate scores 계산 ----
            # beam_scores: (batch_size, beam_size) → (batch_size, beam_size, 1)
            # next_token_log_probs: (batch_size, beam_size, vocab_size)
            candidate_scores = beam_scores.unsqueeze(2) + next_token_log_probs
            # → (batch_size, beam_size, vocab_size)
            
            # ---- 4.9. Top-k 선택 ----
            # Flatten: (batch_size, beam_size * vocab_size)
            candidate_scores_flat = candidate_scores.view(batch_size, -1)
            
            # 각 배치에서 top beam_size 선택
            top_scores, top_indices = torch.topk(
                candidate_scores_flat, self.beam_size, dim=1
            )
            # top_scores: (batch_size, beam_size)
            # top_indices: (batch_size, beam_size)
            
            # ---- 4.10. Beam indices & Token indices ----
            beam_indices = top_indices // vocab_size  # (batch_size, beam_size)
            token_indices = top_indices % vocab_size  # (batch_size, beam_size)
            
            # ---- 4.11. Beams 업데이트 ----
            # 이전 beams에서 선택된 beam 가져오기
            beam_indices_expanded = beam_indices.unsqueeze(2).expand(
                batch_size, self.beam_size, current_len
            )
            selected_beams = torch.gather(beams, 1, beam_indices_expanded)
            # selected_beams: (batch_size, beam_size, current_len)
            
            # 새 토큰 추가
            new_tokens = token_indices.unsqueeze(2)  # (batch_size, beam_size, 1)
            beams = torch.cat([selected_beams, new_tokens], dim=2)
            # beams: (batch_size, beam_size, current_len + 1)
            
            # ---- 4.12. Scores & Finished 업데이트 ----
            beam_scores = top_scores
            
            # Finished 상태 업데이트
            prev_finished = torch.gather(is_finished, 1, beam_indices)
            new_finished = prev_finished | (token_indices == self.eos_idx)
            is_finished = new_finished
        
        # ================================================================
        # 5. Best beam 선택 (length penalty 적용)
        # ================================================================
        # Length penalty: ((5 + len) / 6)^α
        lengths = (beams != self.pad_idx).sum(dim=2).float()
        # lengths: (batch_size, beam_size)
        
        length_penalties = ((5 + lengths) / 6) ** self.length_penalty
        normalized_scores = beam_scores / length_penalties
        
        # 각 배치에서 최고 점수 beam 선택
        best_beam_indices = normalized_scores.argmax(dim=1)  # (batch_size,)
        
        # Best sequences 가져오기
        best_beam_indices_expanded = best_beam_indices.unsqueeze(1).unsqueeze(2).expand(
            batch_size, 1, beams.size(2)
        )
        best_sequences = torch.gather(beams, 1, best_beam_indices_expanded).squeeze(1)
        # best_sequences: (batch_size, final_len)
        
        # Best scores 가져오기
        best_beam_indices_for_scores = best_beam_indices.unsqueeze(1)
        best_scores = torch.gather(beam_scores, 1, best_beam_indices_for_scores).squeeze(1)
        # best_scores: (batch_size,)
        
        return best_sequences, best_scores
    
    def translate(self, src, src_lengths=None, return_scores=False):
        """
        편의 메서드 (기존 API 호환)
        
        Args:
            src: (batch_size, src_len)
            src_lengths: (batch_size,)
            return_scores: whether to return scores
        
        Returns:
            translations: (batch_size, max_len)
            scores (optional): (batch_size,)
        """
        self.model.eval()
        with torch.no_grad():
            sequences, scores = self.search(src, src_lengths)
        
        if return_scores:
            return sequences, scores
        return sequences


def greedy_decode_batch(model, src, src_lengths=None, max_len=None, 
                       sos_idx=2, eos_idx=3, pad_idx=1):
    """
    배치 단위 Greedy Decoding (baseline)
    
    Args:
        model: Transformer model
        src: (batch_size, src_len)
        src_lengths: (batch_size,)
        max_len: maximum output length
        sos_idx, eos_idx, pad_idx: special token indices
    
    Returns:
        decoded: (batch_size, max_len)
    """
    batch_size = src.size(0)
    device = src.device
    
    # Encode (배치 처리)
    src_mask = model.make_src_mask(src)
    enc_src = model.encoder(src, src_mask)
    
    # Max length
    if src_lengths is not None:
        max_src_len = src_lengths.max().item()
    else:
        max_src_len = src.size(1)
    
    if max_len is None:
        max_len = max_src_len + 50
    
    # Initialize with SOS token
    decoded = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    
    # Track which sequences have finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len - 1):
        # All finished?
        if finished.all():
            break
        
        # Create target mask
        trg_mask = model.make_trg_mask(decoded)
        
        # Forward
        output = model.decoder(decoded, enc_src, trg_mask, src_mask)
        
        # Get next token (greedy)
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        # next_token: (batch_size, 1)
        
        # For finished sequences, use pad token
        next_token[finished] = pad_idx
        
        # Append
        decoded = torch.cat([decoded, next_token], dim=1)
        
        # Update finished status
        finished = finished | (next_token.squeeze(1) == eos_idx)
    
    return decoded
