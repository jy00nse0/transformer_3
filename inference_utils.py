"""
Inference Batch Utilities
배치 추론을 위한 유틸리티 함수들
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_inference_batch(batch, source_vocab, target_vocab, source_tokenizer=None, target_tokenizer=None, source_lang='en', target_lang='de'):
    """
    추론용 배치 collate function
    
    Args:
        batch: list of dict from HuggingFace dataset
              Each dict has 'translation' key with {source_lang: text, target_lang: text}
        source_vocab: source vocabulary object with .stoi and .itos
        target_vocab: target vocabulary object with .stoi and .itos
        source_tokenizer: optional tokenizer for source text
        target_tokenizer: optional tokenizer for target text
        source_lang: source language key (default: 'en')
        target_lang: target language key (default: 'de')
    
    Returns:
        dict containing:
            - src: (batch_size, max_src_len) padded tensor
            - src_lengths: (batch_size,) actual lengths
            - references: list of reference texts for BLEU computation
    """
    # ================================================================
    # HuggingFace Dataset 처리
    # ================================================================
    src_batch = []
    references = []
    
    for example in batch:
        # HuggingFace dataset format: {'translation': {'en': '...', 'de': '...'}}
        if isinstance(example, dict) and 'translation' in example:
            src_text = example['translation'][source_lang]
            trg_text = example['translation'][target_lang]
        else:
            # Fallback for other formats
            src_text = example.src if hasattr(example, 'src') else str(example)
            trg_text = example.trg if hasattr(example, 'trg') else ''
        
        # Tokenize
        if source_tokenizer:
            src_tokens = source_tokenizer.tokenize(src_text.lower())
        else:
            src_tokens = src_text.split()
        
        # Convert to indices
        src_indices = [source_vocab.stoi.get(token, source_vocab.stoi.get('<unk>', 1)) 
                      for token in src_tokens]
        
        # Add SOS/EOS
        sos_idx = source_vocab.stoi.get('<sos>', 2)
        eos_idx = source_vocab.stoi.get('<eos>', 3)
        src_indices = [sos_idx] + src_indices + [eos_idx]
        
        # Convert to tensor
        src_batch.append(torch.tensor(src_indices, dtype=torch.long))
        references.append(trg_text)
    
    # ================================================================
    # Source 패딩
    # ================================================================
    src_padded = pad_sequence(
        src_batch, 
        batch_first=True,
        padding_value=source_vocab.stoi.get('<pad>', 0)
    )
    
    # ================================================================
    # 실제 길이 계산 (패딩 제외)
    # ================================================================
    src_lengths = torch.tensor([len(src) for src in src_batch], dtype=torch.long)
    
    return {
        'src': src_padded,
        'src_lengths': src_lengths,
        'references': references
    }


def indices_to_text(indices, itos, sos_idx, eos_idx, pad_idx, tokenizer=None):
    """
    인덱스 시퀀스를 텍스트로 변환
    
    CRITICAL: 모델 output indices는 CUSTOM vocabulary 인덱스!
             tokenizer.decode(indices) 직접 호출하면 안됨 (GPT-2 vocab 기대)
    
    Args:
        indices: (seq_len,) tensor or list of CUSTOM vocab indices  
        itos: index to string dictionary (custom vocab)
        sos_idx, eos_idx, pad_idx: special token indices
        tokenizer: Tokenizer for BPE detokenization
    
    Returns:
        str: decoded text
    """
    # Tensor를 list로 변환
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().tolist()
    
    # ================================================================
    # Step 1: Custom vocab indices → BPE token strings (using itos)
    # ================================================================
    tokens = []
    for idx in indices:
        if idx not in [sos_idx, eos_idx, pad_idx]:
            # itos: custom vocab index → BPE token string
            # 예: idx=100 → "Ġw", idx=101 → "ird"
            token = itos.get(idx, '<unk>')
            tokens.append(token)
    
    # ================================================================
    # Step 2: BPE token strings → Clean readable text
    # ================================================================
    if tokenizer is not None and len(tokens) > 0:
        # convert_tokens_to_string: BPE 토큰 리스트 → 정상 텍스트
        # 예: ["Ġw", "ird"] → "wird"
        #     ["Ã¶"] → "ö"
        try:
            decoded_text = tokenizer.convert_tokens_to_string(tokens)
            return decoded_text
        except Exception as e:
            # Fallback if conversion fails
            print(f"Warning: tokenizer.convert_tokens_to_string failed: {e}")
            return ' '.join(tokens)
    else:
        # No tokenizer: just join tokens
        return ' '.join(tokens)


def batch_indices_to_texts(batch_indices, itos, sos_idx, eos_idx, pad_idx, tokenizer=None):
    """
    배치 인덱스를 텍스트 리스트로 변환
    
    Args:
        batch_indices: (batch_size, seq_len) tensor
        itos: index to string dictionary
        sos_idx, eos_idx, pad_idx: special token indices
        tokenizer: (CRITICAL!) Tokenizer for proper BPE detokenization
    
    Returns:
        list of str: properly decoded texts
    """
    batch_size = batch_indices.size(0)
    texts = []
    
    for i in range(batch_size):
        text = indices_to_text(
            batch_indices[i], 
            itos, 
            sos_idx, 
            eos_idx, 
            pad_idx,
            tokenizer=tokenizer  # Pass tokenizer!
        )
        texts.append(text)
    
    return texts


def create_inference_dataloader(dataset, batch_size=32, max_tokens=4096, 
                                num_workers=2, device='cuda'):
    """
    추론용 DataLoader 생성
    
    Args:
        dataset: TranslationDataset 객체
        batch_size: 최대 배치 크기
        max_tokens: 배치당 최대 토큰 수
        num_workers: 데이터 로딩 워커 수
        device: 디바이스
    
    Returns:
        DataLoader 객체
    """
    from torch.utils.data import DataLoader as TorchDataLoader
    from util.data_loader import TokenBucketSampler
    
    # TokenBucketSampler 생성 (기존 코드 재사용)
    sampler = TokenBucketSampler(
        lengths=dataset.lengths,  # TranslationDataset이 이미 계산
        max_tokens=max_tokens,
        shuffle=False,            # 추론은 순서 유지
        drop_last=False           # 모든 샘플 번역
    )
    
    # DataLoader 생성
    dataloader = TorchDataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
    
    print(f"Created inference DataLoader:")
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Total batches: {len(sampler):,}")
    print(f"  Avg batch size: {len(dataset) / len(sampler):.1f}")
    print(f"  Max tokens per batch: {max_tokens:,}")
    
    return dataloader, sampler
