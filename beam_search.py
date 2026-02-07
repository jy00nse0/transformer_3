"""
Beam Search Implementation for Transformer
Following "Attention Is All You Need" paper specifications
"""

import torch
import torch.nn.functional as F

class BeamSearch:
    """
    Beam Search Decoder
    
    Paper specifications:
    - Beam size: 4
    - Length penalty: α = 0.6
    - Max output length: input_length + 50
    - Early termination when possible
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
    
    def search(self, src):
        """
        Perform beam search
        
        Args:
            src: Source sequence (batch_size, src_len)
        
        Returns:
            best_sequences: Best decoded sequences (batch_size, max_len)
            best_scores: Scores of best sequences (batch_size,)
        """
        batch_size = src.size(0)
        
        # Encode source
        src_mask = self.model.make_src_mask(src)
        enc_src = self.model.encoder(src, src_mask)
        
        # Determine max output length (paper: input_length + 50)
        src_len = src.size(1)
        if self.max_len is None:
            max_output_len = src_len + 50
        else:
            max_output_len = self.max_len
        
        # Process each sentence in batch independently
        all_best_sequences = []
        all_best_scores = []
        
        for batch_idx in range(batch_size):
            # Get encoder output for this sentence
            enc_src_single = enc_src[batch_idx:batch_idx+1]  # (1, src_len, d_model)
            src_mask_single = src_mask[batch_idx:batch_idx+1]  # (1, 1, 1, src_len)
            
            # Initialize beams
            # Shape: (beam_size, current_len)
            beams = torch.full((self.beam_size, 1), self.sos_idx, 
                             dtype=torch.long, device=self.device)
            
            # Beam scores (log probabilities)
            beam_scores = torch.zeros(self.beam_size, device=self.device)
            beam_scores[1:] = -1e9  # Only first beam is active initially
            
            # Track which beams have ended
            is_finished = torch.zeros(self.beam_size, dtype=torch.bool, device=self.device)
            
            # Beam search loop
            for step in range(max_output_len - 1):
                # All beams finished?
                if is_finished.all():
                    break
                
                # Prepare decoder input
                # Expand encoder output for all beams
                enc_src_expanded = enc_src_single.expand(self.beam_size, -1, -1)
                src_mask_expanded = src_mask_single.expand(self.beam_size, -1, -1, -1)
                
                # Create target mask
                trg_mask = self.model.make_trg_mask(beams)
                
                # Forward pass through decoder
                output = self.model.decoder(beams, enc_src_expanded, trg_mask, src_mask_expanded)
                
                # Get next token probabilities
                # output: (beam_size, current_len, vocab_size)
                next_token_logits = output[:, -1, :]  # (beam_size, vocab_size)
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Compute scores for all possible next tokens
                # beam_scores: (beam_size,) -> (beam_size, 1)
                # next_token_log_probs: (beam_size, vocab_size)
                vocab_size = next_token_log_probs.size(-1)
                
                # For finished beams, only allow pad token
                next_token_log_probs[is_finished, :] = -1e9
                next_token_log_probs[is_finished, self.pad_idx] = 0
                
                # Compute candidate scores
                candidate_scores = beam_scores.unsqueeze(1) + next_token_log_probs
                # Shape: (beam_size, vocab_size)
                
                # Flatten to select top-k across all beams
                candidate_scores = candidate_scores.view(-1)  # (beam_size * vocab_size,)
                
                # Select top beam_size candidates
                top_scores, top_indices = torch.topk(candidate_scores, self.beam_size)
                
                # Convert flat indices to (beam_idx, token_idx)
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                
                # Update beams
                new_beams = []
                new_scores = []
                new_is_finished = []
                
                for i in range(self.beam_size):
                    beam_idx = beam_indices[i]
                    token_idx = token_indices[i]
                    score = top_scores[i]
                    
                    # Append token to beam
                    new_beam = torch.cat([beams[beam_idx], token_idx.unsqueeze(0)])
                    new_beams.append(new_beam)
                    new_scores.append(score)
                    
                    # Check if finished (EOS token)
                    finished = is_finished[beam_idx] or (token_idx == self.eos_idx)
                    new_is_finished.append(finished)
                
                # Stack new beams
                beams = torch.stack(new_beams)
                beam_scores = torch.tensor(new_scores, device=self.device)
                is_finished = torch.tensor(new_is_finished, device=self.device)
            
            # Select best beam with length penalty
            # Length penalty: ((5 + len) / 6)^α (paper: α=0.6)
            lengths = (beams != self.pad_idx).sum(dim=1).float()
            length_penalties = ((5 + lengths) / 6) ** self.length_penalty
            normalized_scores = beam_scores / length_penalties
            
            best_beam_idx = normalized_scores.argmax()
            best_sequence = beams[best_beam_idx]
            best_score = beam_scores[best_beam_idx]
            
            all_best_sequences.append(best_sequence)
            all_best_scores.append(best_score)
        
        # Pad sequences to same length
        max_seq_len = max(seq.size(0) for seq in all_best_sequences)
        padded_sequences = []
        
        for seq in all_best_sequences:
            if seq.size(0) < max_seq_len:
                padding = torch.full((max_seq_len - seq.size(0),), self.pad_idx, 
                                   dtype=torch.long, device=self.device)
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)
        
        best_sequences = torch.stack(padded_sequences)
        best_scores = torch.tensor(all_best_scores, device=self.device)
        
        return best_sequences, best_scores
    
    def translate(self, src, return_scores=False):
        """
        Convenience method for translation
        
        Args:
            src: Source sequence (batch_size, src_len)
            return_scores: Whether to return scores
        
        Returns:
            translations: Translated sequences (batch_size, max_len)
            scores (optional): Translation scores
        """
        self.model.eval()
        with torch.no_grad():
            sequences, scores = self.search(src)
        
        if return_scores:
            return sequences, scores
        return sequences


def greedy_decode(model, src, max_len=None, sos_idx=2, eos_idx=3):
    """
    Greedy decoding (baseline, not used in paper)
    
    Args:
        model: Transformer model
        src: Source sequence (batch_size, src_len)
        max_len: Maximum output length
        sos_idx: Start of sentence token
        eos_idx: End of sentence token
    
    Returns:
        decoded: Decoded sequences (batch_size, max_len)
    """
    batch_size = src.size(0)
    device = src.device
    
    # Encode
    src_mask = model.make_src_mask(src)
    enc_src = model.encoder(src, src_mask)
    
    # Max length
    if max_len is None:
        max_len = src.size(1) + 50
    
    # Initialize with SOS token
    decoded = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    
    for _ in range(max_len - 1):
        # Create target mask
        trg_mask = model.make_trg_mask(decoded)
        
        # Forward
        output = model.decoder(decoded, enc_src, trg_mask, src_mask)
        
        # Get next token
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # Append
        decoded = torch.cat([decoded, next_token], dim=1)
        
        # Check if all sequences have EOS
        if (next_token == eos_idx).all():
            break
    
    return decoded
