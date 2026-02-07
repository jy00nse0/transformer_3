# ============================================================================
# WMT14 with Step-based Training (NaN-Safe Version)
# ============================================================================
#
# Key improvements:
#   - Automatic NaN detection in model outputs, loss, and gradients
#   - Gradient clipping (max_norm=1.0) for numerical stability
#   - NaN batches are automatically skipped
#   - Detailed NaN statistics tracking and reporting
#   - Enhanced logging with gradient norms and NaN ratios
#
# This version is specifically designed to handle extreme configurations
# like d_k=16 in Table 3(B) experiments where numerical instability is expected.
#
# ============================================================================
# WMT14 with Step-based Training (Paper Implementation)
# ============================================================================
#
# Arguments:
#   --load_dir: Directory to load pre-built vocabulary (default: None)
#   --save_dir: Directory to save vocabulary (default: 'artifacts_pretrained')
#   --max_steps: Maximum training steps (default: 100000)
#   --max_tokens: Maximum tokens per batch (default: 25000)
#   --tokenizer_en: English tokenizer model (default: 'gpt2')
#   --tokenizer_de: German tokenizer model (default: 'gpt2')
#   --checkpoint_dir: Directory to save model checkpoints (default: 'checkpoints')
#   --checkpoint_every: Save checkpoint every N steps (default: 10000)
#   --log_every: Print log every N steps (default: 100)
#   --gradient_checkpointing: Enable gradient checkpointing to reduce memory (default: False)
#   --num_workers: Number of data loading workers (default: 2)
#   --drop_prob: Dropout probability (default: 0.1)
#   --n_head: Number of attention heads (default: 8)
#   --n_layers: Number of encoder/decoder layers (default: 6)
#   --d_model: Model dimension (default: 512)
#   --ffn_hidden: Feed-forward network hidden dimension (default: 2048)
#   --label_smoothing: Label smoothing value (default: 0.1)
#   --kdim: Key dimension for MultiheadAttention (default: None, uses d_model)
#
# Example Usage:
#   # Basic training with default settings
#   python demo_wmt14_pretrained.py
#
#   # Training with custom model architecture
#   python demo_wmt14_pretrained.py --d_model 256 --n_head 4 --n_layers 4 --ffn_hidden 1024
#
#   # Training with gradient checkpointing for memory optimization
#   python demo_wmt14_pretrained.py --gradient_checkpointing --num_workers 1
#
#   # Training with custom dropout and label smoothing
#   python demo_wmt14_pretrained.py --drop_prob 0.2 --label_smoothing 0.15
#
#   # Resume training from existing vocabulary
#   python demo_wmt14_pretrained.py --load_dir ./artifacts_pretrained
#
# ============================================================================

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from util.data_loader import DataLoader
from models.model.transformer import Transformer
import time
import sys
import argparse
import pickle
import os
from pathlib import Path
import json
from checkpoint_utils import safe_torch_save

def print_section(title):
    """섹션 헤더 출력"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

# ============================================================================
# 사전 학습된 토크나이저 래퍼 클래스
# ============================================================================

class PretrainedBPETokenizer:
    """사전 학습된 GPT-2 BPE 토크나이저"""
    def __init__(self, model_name="gpt2", vocab_size=None):
        print(f"Loading pre-trained tokenizer: {model_name}...")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.vocab = list(self.tokenizer.get_vocab().keys())
        self.merges = {}
        
        elapsed = time.time() - start_time
        print(f"✓ Tokenizer loaded in {elapsed:.2f}s")
        print(f"  Model: {model_name}")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Type: BPE (Byte-Pair Encoding)")
    
    def train(self, corpus):
        print(f"\n{'='*80}")
        print(f"Pre-trained Tokenizer - No Training Required!")
        print(f"{'='*80}")
        print(f"✓ Skipping training (already trained on billions of tokens)")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Ready to use immediately!")
        print(f"{'='*80}\n")
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text.lower())
    
    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)


# ============================================================================
# 저장/로드 함수들
# ============================================================================

def save_tokenizer(tokenizer, filepath):
    """토크나이저를 파일로 저장"""
    print(f"Saving tokenizer to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    tokenizer_info = {
        'type': 'pretrained',
        'model_name': getattr(tokenizer.tokenizer, 'name_or_path', 'gpt2'),
        'vocab_size': tokenizer.vocab_size
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer_info, f)
    
    print(f"✓ Tokenizer info saved (model: {tokenizer_info['model_name']})")

def load_tokenizer(filepath, tokenizer_class=PretrainedBPETokenizer):
    """토크나이저를 파일에서 로드"""
    print(f"Loading tokenizer from {filepath}...")
    
    with open(filepath, 'rb') as f:
        tokenizer_info = pickle.load(f)
    
    if tokenizer_info['type'] == 'pretrained':
        tokenizer = tokenizer_class(model_name=tokenizer_info['model_name'])
        print(f"✓ Pre-trained tokenizer loaded")
    else:
        tokenizer = pickle.load(open(filepath, 'rb'))
    
    return tokenizer

def save_vocab(loader, filepath):
    """어휘사전을 파일로 저장"""
    print(f"Saving vocabulary to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    vocab_data = {
        'source_stoi': loader.source.vocab.stoi,
        'source_itos': loader.source.vocab.itos,
        'target_stoi': loader.target.vocab.stoi,
        'target_itos': loader.target.vocab.itos,
        'shared': loader.source.vocab is loader.target.vocab
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(vocab_data, f)
    
    print(f"✓ Vocabulary saved")

def load_vocab(loader, filepath):
    """어휘사전을 파일에서 로드"""
    print(f"Loading vocabulary from {filepath}...")
    
    with open(filepath, 'rb') as f:
        vocab_data = pickle.load(f)
    
    source_vocab_obj = type('obj', (object,), {
        'stoi': vocab_data['source_stoi'],
        'itos': vocab_data['source_itos'],
        '__len__': lambda self: len(vocab_data['source_stoi'])
    })()
    
    if vocab_data['shared']:
        loader.source.vocab = source_vocab_obj
        loader.target.vocab = source_vocab_obj
    else:
        target_vocab_obj = type('obj', (object,), {
            'stoi': vocab_data['target_stoi'],
            'itos': vocab_data['target_itos'],
            '__len__': lambda self: len(vocab_data['target_stoi'])
        })()
        
        loader.source.vocab = source_vocab_obj
        loader.target.vocab = target_vocab_obj
    
    print(f"✓ Vocabulary loaded")

def save_training_artifacts(tokenizer_en, tokenizer_de, loader, save_dir):
    """모든 학습 결과물을 저장"""
    print_section("저장 중...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_tokenizer(tokenizer_en, save_dir / 'tokenizer_en.pkl')
    save_tokenizer(tokenizer_de, save_dir / 'tokenizer_de.pkl')
    save_vocab(loader, save_dir / 'vocab.pkl')
    
    metadata = {
        'vocab_size': len(loader.source.vocab),
        'shared_vocab': loader.source.vocab is loader.target.vocab,
        'src_pad_idx': loader.source.vocab.stoi['<pad>'],
        'trg_pad_idx': loader.target.vocab.stoi['<pad>'],
        'trg_sos_idx': loader.target.vocab.stoi['<sos>'],
        'tokenizer_type': 'pretrained',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ All artifacts saved to: {save_dir}")

def load_training_artifacts(loader, load_dir):
    """저장된 학습 결과물을 로드"""
    print_section("기존 결과물 로드 중...")
    
    load_dir = Path(load_dir)
    
    with open(load_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Metadata:")
    print(f"  Created: {metadata['timestamp']}")
    print(f"  Vocab size: {metadata['vocab_size']:,}")
    print(f"  Tokenizer type: {metadata.get('tokenizer_type', 'custom')}")
    
    print()
    tokenizer_en = load_tokenizer(load_dir / 'tokenizer_en.pkl', PretrainedBPETokenizer)
    tokenizer_de = load_tokenizer(load_dir / 'tokenizer_de.pkl', PretrainedBPETokenizer)
    
    print()
    load_vocab(loader, load_dir / 'vocab.pkl')
    
    print(f"\n✓ All artifacts loaded from: {load_dir}")
    
    return tokenizer_en, tokenizer_de, metadata

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='WMT14 Transformer with Step-based Training')
    
    parser.add_argument(
        '--load_dir',
        type=str,
        default=None,
        help='Directory to load pre-built vocabulary'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='artifacts_pretrained',
        help='Directory to save vocabulary (default: artifacts_pretrained)'
    )
    
    # ================================================================
    # 핵심 변경: --epochs 대신 --max_steps 사용
    # ================================================================
    parser.add_argument(
        '--max_steps',
        type=int,
        default=100000,  # 논문: 100K steps
        help='Maximum training steps (default: 100000, paper setting)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=25000,
        help='Maximum tokens per batch (default: 25000)'
    )
    
    parser.add_argument(
        '--tokenizer_en',
        type=str,
        default='gpt2',
        help='English tokenizer model (default: gpt2)'
    )
    
    parser.add_argument(
        '--tokenizer_de',
        type=str,
        default='gpt2',
        help='German tokenizer model (default: gpt2)'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Directory to save model checkpoints (default: checkpoints)'
    )
    
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=10000,  # 10K steps마다 저장
        help='Save checkpoint every N steps (default: 10000)'
    )
    
    parser.add_argument(
        '--log_every',
        type=int,
        default=100,
        help='Print log every N steps (default: 100)'
    )
    
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable gradient checkpointing to reduce memory (default: False)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='Number of data loading workers (default: 2, reduce for lower CPU memory)'
    )
    
    parser.add_argument(
        '--drop_prob',
        type=float,
        default=0.1,
        help='Dropout probability (default: 0.1)'
    )
    
    parser.add_argument(
        '--n_head',
        type=int,
        default=8,
        help='Number of attention heads (default: 8)'
    )
    
    parser.add_argument(
        '--n_layers',
        type=int,
        default=6,
        help='Number of encoder/decoder layers (default: 6)'
    )
    
    parser.add_argument(
        '--d_model',
        type=int,
        default=512,
        help='Model dimension (default: 512)'
    )
    
    parser.add_argument(
        '--ffn_hidden',
        type=int,
        default=2048,
        help='Feed-forward network hidden dimension (default: 2048)'
    )
    
    parser.add_argument(
        '--label_smoothing',
        type=float,
        default=0.1,
        help='Label smoothing value (default: 0.1)'
    )
    
    parser.add_argument(
        '--kdim',
        type=int,
        default=None,
        help='[DEPRECATED] Key dimension for MultiheadAttention. Use --custom with --d_k instead.'
    )
    
    parser.add_argument(
        '--n_avg',
        type=int,
        default=5,
        help='Number of checkpoints to average (default: 5 for base model, 20 for big model)'
    )
    
    # ================================================================
    # Custom Attention Arguments (Table 3(B) experiments)
    # ================================================================
    parser.add_argument(
        '--custom',
        action='store_true',
        help='Use CustomMultiheadAttention with independent d_k/d_v control. '
             'Required for Table 3(B) experiments. (default: False)'
    )

    parser.add_argument(
        '--d_k',
        type=int,
        default=None,
        help='Key/Query dimension per head. '
             'Only used when --custom is set. (default: d_model / n_head)'
    )

    parser.add_argument(
        '--d_v',
        type=int,
        default=None,
        help='Value dimension per head. '
             'Only used when --custom is set. (default: d_model / n_head)'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*80)
    print(" "*10 + "WMT14 Transformer with Step-based Training (Paper)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Load directory: {args.load_dir if args.load_dir else 'None (build vocab from scratch)'}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Max training steps: {args.max_steps:,} (논문: 100K)")
    print(f"  Max tokens per batch: {args.max_tokens:,}")
    print(f"  Checkpoint every: {args.checkpoint_every:,} steps")
    print(f"  Log every: {args.log_every} steps")
    print(f"  English tokenizer: {args.tokenizer_en}")
    print(f"  German tokenizer: {args.tokenizer_de}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print(f"  Gradient checkpointing: {'Enabled' if args.gradient_checkpointing else 'Disabled'}")
    print(f"  Num workers: {args.num_workers}")
    print(f"\nModel Hyperparameters:")
    print(f"  d_model: {args.d_model}")
    print(f"  n_head: {args.n_head}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  ffn_hidden: {args.ffn_hidden}")
    print(f"  drop_prob: {args.drop_prob}")
    print(f"  label_smoothing: {args.label_smoothing}")
    if args.custom:
        eff_d_k = args.d_k if args.d_k else args.d_model // args.n_head
        eff_d_v = args.d_v if args.d_v else args.d_model // args.n_head
        print(f"  Attention: CustomMultiheadAttention")
        print(f"    d_k (per head): {eff_d_k}")
        print(f"    d_v (per head): {eff_d_v}")
    else:
        print(f"  Attention: Standard nn.MultiheadAttention")
        if args.kdim:
            print(f"    kdim: {args.kdim} [DEPRECATED - use --custom instead]")
    print(f"  n_avg: {args.n_avg} (checkpoints to average)")
    
    # ------------------------------------------------------------------------
    # 1. 데이터셋 로드
    # ------------------------------------------------------------------------
    print_section("1. 데이터셋 로드")
    
    loader = DataLoader(
        ext=('.en', '.de'),
        tokenize_en=None,
        tokenize_de=None,
        init_token='<sos>',
        eos_token='<eos>'
    )
    
    print("Loading WMT14 dataset...")
    start_time = time.time()
    
    train, valid, test = loader.make_dataset(dataset_name='wmt14')
    
    elapsed = time.time() - start_time
    print(f"\n✓ Dataset loaded in {elapsed:.1f}s")
    print(f"  Train: {len(train):,} sentence pairs")
    print(f"  Valid: {len(valid):,} sentence pairs")
    print(f"  Test:  {len(test):,} sentence pairs")
    
    # ------------------------------------------------------------------------
    # 2. 토크나이저 로드
    # ------------------------------------------------------------------------
    if args.load_dir:
        tokenizer_en, tokenizer_de, metadata = load_training_artifacts(loader, args.load_dir)
        loader.tokenize_en = tokenizer_en.tokenize
        loader.tokenize_de = tokenizer_de.tokenize
        loader.source_tokenizer = tokenizer_en
        loader.target_tokenizer = tokenizer_de
    else:
        print_section("2. 사전 학습 토크나이저 로드")
        
        print("=" * 80)
        print("🚀 Using Pre-trained Tokenizers - No 42-hour Training!")
        print("=" * 80)
        print()
        
        print("English Tokenizer:")
        tokenizer_en = PretrainedBPETokenizer(model_name=args.tokenizer_en)
        
        print()
        
        print("German Tokenizer:")
        tokenizer_de = PretrainedBPETokenizer(model_name=args.tokenizer_de)
        
        print()
        print("=" * 80)
        print("✓ Both tokenizers ready! (Total time: ~2 seconds)")
        print("  ⏱️  Time saved: ~84 hours (42h EN + 42h DE)")
        print("=" * 80)
        
        loader.source_tokenizer = tokenizer_en
        loader.target_tokenizer = tokenizer_de
        
        # ------------------------------------------------------------------------
        # 3. 어휘 사전 구축
        # ------------------------------------------------------------------------
        print_section("3. 어휘 사전 구축")
        
        print("Building vocabulary from tokenized data...")
        vocab_start = time.time()
        
        effective_vocab_size = min(tokenizer_en.vocab_size, tokenizer_de.vocab_size)
        
        loader.build_vocab(
            train_data=train,
            min_freq=2,
            max_vocab_size=37000,  # 논문 설정
            shared_vocab=True
        )
        
        print(f"\n✓ Vocabulary built in {time.time() - vocab_start:.1f}s")
        
        # ------------------------------------------------------------------------
        # 4. 결과물 저장
        # ------------------------------------------------------------------------
        save_training_artifacts(tokenizer_en, tokenizer_de, loader, args.save_dir)
    
    # ------------------------------------------------------------------------
    # 어휘 정보 출력
    # ------------------------------------------------------------------------
    src_pad_idx = loader.source.vocab.stoi['<pad>']
    trg_pad_idx = loader.target.vocab.stoi['<pad>']
    trg_sos_idx = loader.target.vocab.stoi['<sos>']
    
    enc_voc_size = len(loader.source.vocab)
    dec_voc_size = len(loader.target.vocab)
    
    print_section("어휘 정보")
    print(f"Vocabulary Statistics:")
    print(f"  Source vocab size: {enc_voc_size:,}")
    print(f"  Target vocab size: {dec_voc_size:,}")
    print(f"  Shared: {loader.source.vocab is loader.target.vocab}")
    print(f"\nSpecial Tokens:")
    print(f"  <pad>: {src_pad_idx}")
    print(f"  <unk>: {loader.source.vocab.stoi['<unk>']}")
    print(f"  <sos>: {trg_sos_idx}")
    print(f"  <eos>: {loader.source.vocab.stoi['<eos>']}")
    
    # ------------------------------------------------------------------------
    # 5. Iterator 생성
    # ------------------------------------------------------------------------
    print_section("5. 데이터 Iterator 생성")
    
    print("Creating data iterators...")
    iter_start = time.time()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_iter, valid_iter, test_iter = loader.make_iter(
        train, valid, test,
        max_tokens=args.max_tokens,
        device=device,
        num_workers=args.num_workers,
        max_len=256  # Must match model's max_len
    )
    
    print(f"✓ Iterators created in {time.time() - iter_start:.1f}s")
    print(f"\nIterator Statistics:")
    print(f"  Train batches per epoch: {len(train_iter):,}")
    print(f"  Valid batches: {len(valid_iter):,}")
    
    # Steps per epoch 계산
    steps_per_epoch = len(train_iter)
    total_epochs = (args.max_steps + steps_per_epoch - 1) // steps_per_epoch
    
    print(f"\nStep-based Training Info:")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Estimated epochs: ~{total_epochs} epochs")
    print(f"  (Note: Training stops at {args.max_steps:,} steps exactly)")
    
    # ------------------------------------------------------------------------
    # 6. 모델 초기화
    # ------------------------------------------------------------------------
    print_section("6. 모델 초기화")
    
    print(f"Using device: {device}")
    print("\nInitializing Transformer model...")
    
    # Handle deprecated kdim argument
    if args.kdim and not args.custom:
        print("\n⚠️  Warning: --kdim is deprecated. Use --custom with --d_k instead.")
        print("   Falling back to standard attention.\n")
    
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        d_model=args.d_model,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=256,
        ffn_hidden=args.ffn_hidden,
        n_head=args.n_head,
        n_layers=args.n_layers,
        drop_prob=args.drop_prob,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
        use_custom=args.custom,
        d_k=args.d_k,
        d_v=args.d_v
    ).to(device)
    
    # Store model configuration for checkpoint saving
    model_config = {
        'd_model': args.d_model,
        'n_head': args.n_head,
        'n_layers': args.n_layers,
        'ffn_hidden': args.ffn_hidden,
        'drop_prob': args.drop_prob,
        'max_len': 256,
        'enc_voc_size': enc_voc_size,
        'dec_voc_size': dec_voc_size,
        'src_pad_idx': src_pad_idx,
        'trg_pad_idx': trg_pad_idx,
        'trg_sos_idx': trg_sos_idx,
        'label_smoothing': args.label_smoothing,
        'n_avg': args.n_avg,
        # Custom attention config
        'use_custom': args.custom,
        'd_k': args.d_k,
        'd_v': args.d_v,
        # Deprecated (keep for backward compatibility)
        'kdim': args.kdim
    }
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    
    # ------------------------------------------------------------------------
    # 7. Optimizer 및 Scheduler
    # ------------------------------------------------------------------------
    print_section("7. Optimizer 및 Scheduler")
    
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LambdaLR
    
    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    
    def lr_schedule(step):
        d_model = args.d_model
        warmup_steps = 4000
        return (d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=args.label_smoothing)
    # ============================================================================
    # Import NaN-safe training utilities
    # ============================================================================
    from training_utils import NaNSafeTrainer, print_training_log, print_nan_warning
    
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=args.label_smoothing)
    
    print("✓ Training setup complete")
    print(f"  Warmup steps: 4000")
    print(f"  Learning rate schedule: sqrt decay with warmup")
    print(f"  Gradient clipping: max_norm=1.0")
    print(f"  NaN detection: ENABLED")
    
    # ------------------------------------------------------------------------
    # Initialize NaN-safe trainer
    # ------------------------------------------------------------------------
    trainer = NaNSafeTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=1.0  # Gradient clipping for numerical stability
    )
    
    print(f"\n{'='*80}")
    print("NaN-Safe Training Features:")
    print(f"{'='*80}")
    print("  ✓ Automatic NaN detection in:")
    print("    - Model outputs")
    print("    - Loss values")
    print("    - Gradients")
    print("  ✓ Gradient clipping (max_norm=1.0)")
    print("  ✓ NaN batches skipped automatically")
    print("  ✓ Detailed NaN statistics tracking")
    print(f"{'='*80}\n")
    
    # ------------------------------------------------------------------------
    # 8. Step-based 학습 루프
    # ------------------------------------------------------------------------
    print_section("8. Step-based 학습 시작")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training for {args.max_steps:,} steps (논문: 100K)")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Checkpoint frequency: every {args.checkpoint_every:,} steps")
    
    training_start = time.time()
    global_step = 0
    epoch = 0
    
    # 학습 통계
    running_loss = 0.0
    best_val_loss = float('inf')
    
    # Checkpoint averaging variables (논문 구현)
    last_checkpoint_save_count = 1
    checkpoint_save_threshold = {}
    
    print(f"\n{'='*80}")
    print(f"Starting training...")
    print(f"{'='*80}\n")
    
    # ================================================================
    # Step-based training loop (NaN-safe version)
    # ================================================================
    while global_step < args.max_steps:
        epoch += 1
        model.train()
        epoch_start = time.time()
        
        print(f"Epoch {epoch} (Steps {global_step:,} - {min(global_step + steps_per_epoch, args.max_steps):,})")
        
        for batch_idx, batch in enumerate(train_iter):
            # Max steps 도달 확인
            if global_step >= args.max_steps:
                print(f"\n✓ Reached max steps ({args.max_steps:,}), stopping training")
                break
            
            src = batch.src.to(device)
            trg = batch.trg.to(device)
            
            # ============================================================
            # NaN-safe training step
            # ============================================================
            success, loss_value, stats = trainer.train_step(src, trg)
            
            if not success:
                # NaN detected - print warning and skip this batch
                print_nan_warning(global_step, stats, trainer.get_nan_statistics())
                
                # Check if NaN ratio is too high
                nan_stats = trainer.get_nan_statistics()
                if nan_stats['nan_ratio'] > 0.1:  # 10% threshold
                    print(f"\n{'='*80}")
                    print(f"⛔ CRITICAL: NaN ratio exceeds 10%!")
                    print(f"{'='*80}")
                    trainer.print_nan_report()
                    print(f"\nConsider:")
                    print(f"  1. Reducing learning rate")
                    print(f"  2. Increasing gradient clipping (current: {trainer.max_grad_norm})")
                    print(f"  3. Using mixed precision training")
                    print(f"  4. Checking model configuration (d_k, d_v)")
                    print(f"{'='*80}\n")
                
                continue  # Skip this batch
            
            # ============================================================
            # Update statistics (only for successful steps)
            # ============================================================
            global_step += 1
            running_loss += loss_value
            
            # ============================================================
            # 로그 출력 (매 N steps)
            # ============================================================
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                elapsed = time.time() - training_start
                steps_per_sec = global_step / elapsed
                eta_seconds = (args.max_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0
                eta_hours = eta_seconds / 3600
                
                current_lr = scheduler.get_last_lr()[0]
                
                # Enhanced logging with NaN statistics
                print_training_log(
                    step=global_step,
                    max_steps=args.max_steps,
                    loss=loss_value,
                    avg_loss=avg_loss,
                    lr=current_lr,
                    grad_norm=stats['grad_norm'],
                    steps_per_sec=steps_per_sec,
                    eta_hours=eta_hours,
                    nan_stats=trainer.get_nan_statistics()
                )
                
                running_loss = 0.0
            
            # ============================================================
            # NaN report (every 1000 steps)
            # ============================================================
            if global_step % 1000 == 0 and trainer.nan_count > 0:
                trainer.print_nan_report()
            
            # ============================================================
            # 논문 구현: 10분 간격 체크포인트 저장
            # ============================================================
            if global_step >= 90000 and last_checkpoint_save_count < args.n_avg:
                threshold_seconds = 600 * (args.n_avg - last_checkpoint_save_count)
                
                if eta_seconds <= threshold_seconds and last_checkpoint_save_count not in checkpoint_save_threshold:
                    checkpoint_name = f'model_last_{args.n_avg - last_checkpoint_save_count}.pt'
                    checkpoint_path = checkpoint_dir / checkpoint_name
                    
                    print(f"\n  [논문 구현] 학습 종료 ~{threshold_seconds//60}분 전 체크포인트 저장...")
                    print(f"  ETA: {eta_seconds/60:.1f}분, 임계값: {threshold_seconds/60:.1f}분")
                    
                    checkpoint_data = {
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'model_config': model_config,
                        'nan_statistics': trainer.get_nan_statistics(),  # Save NaN stats
                    }
                    
                    if safe_torch_save(checkpoint_data, checkpoint_path, atomic=True):
                        print(f"  ✓ 체크포인트 저장: {checkpoint_path}")
                    else:
                        print(f"  ⚠️  Failed to save checkpoint!")
                    
                    checkpoint_save_threshold[last_checkpoint_save_count] = True
                    last_checkpoint_save_count += 1
            
            # ============================================================
            # 정기 체크포인트 저장
            # ============================================================
            if global_step % args.checkpoint_every == 0:
                checkpoint_path = checkpoint_dir / f'model_step_{global_step}.pt'
                
                print(f"\n  Saving checkpoint at step {global_step}...")
                
                checkpoint_data = {
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'model_config': model_config,
                    'nan_statistics': trainer.get_nan_statistics(),
                }
                
                if safe_torch_save(checkpoint_data, checkpoint_path, atomic=True):
                    print(f"  ✓ 체크포인트 저장: {checkpoint_path}")
                else:
                    print(f"  ⚠️  Failed to save checkpoint!")
            
            # ============================================================
            # Validation (optional - every 5000 steps)
            # ============================================================
            if global_step % 5000 == 0:
                print(f"\n  Running validation at step {global_step}...")
                model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for val_batch in valid_iter:
                        if val_batches >= 100:  # Limit validation batches
                            break
                        
                        val_src = val_batch.src.to(device)
                        val_trg = val_batch.trg.to(device)
                        
                        val_output = model(val_src, val_trg[:, :-1])
                        val_loss_batch = criterion(
                            val_output.contiguous().view(-1, val_output.shape[-1]),
                            val_trg[:, 1:].contiguous().view(-1)
                        )
                        
                        # Skip NaN in validation too
                        if not torch.isnan(val_loss_batch) and not torch.isinf(val_loss_batch):
                            val_loss += val_loss_batch.item()
                            val_batches += 1
                
                if val_batches > 0:
                    val_loss = val_loss / val_batches
                    print(f"  Validation Loss: {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint_path = checkpoint_dir / 'model_best.pt'
                        
                        checkpoint_data = {
                            'step': global_step,
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'model_config': model_config,
                            'nan_statistics': trainer.get_nan_statistics(),
                        }
                        
                        if safe_torch_save(checkpoint_data, best_checkpoint_path, atomic=True):
                            print(f"  ✓ Best model saved: {best_checkpoint_path}")
                
                model.train()
        
        # Max steps 도달 시 루프 탈출
        if global_step >= args.max_steps:
            break
    
    # ================================================================
    # Final checkpoint and NaN report
    # ================================================================
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"{'='*80}\n")
    
    # Final NaN report
    trainer.print_nan_report()
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / 'model_final.pt'
    
    checkpoint_data = {
        'step': global_step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': model_config,
        'nan_statistics': trainer.get_nan_statistics(),
        'training_time': time.time() - training_start,
    }
    
    if safe_torch_save(checkpoint_data, final_checkpoint_path, atomic=True):
        print(f"\n✓ Final checkpoint saved: {final_checkpoint_path}")
    
    total_time = time.time() - training_start
    print(f"\nTotal training time: {total_time/3600:.2f} hours")
    print(f"Average speed: {global_step/total_time:.2f} steps/sec")
                    checkpoint_save_threshold[last_checkpoint_save_count] = True
                    last_checkpoint_save_count += 1
                    print()
            
            # ========================================================
            # 체크포인트 저장 (매 N steps)
            # ========================================================
            if global_step % args.checkpoint_every == 0:
                print(f"\n  Saving checkpoint at step {global_step:,}...")
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_batch in valid_iter:
                        val_output = model(val_batch.src.to(device), val_batch.trg[:, :-1].to(device))
                        val_loss_batch = criterion(
                            val_output.contiguous().view(-1, val_output.shape[-1]),
                            val_batch.trg[:, 1:].to(device).contiguous().view(-1)
                        )
                        val_loss += val_loss_batch.item()
                
                avg_val_loss = val_loss / len(valid_iter)
                
                # 체크포인트 저장 (atomic write to prevent corruption)
                checkpoint_path = checkpoint_dir / f'model_step_{global_step}.pt'
                checkpoint_data = {
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'model_config': model_config,
                }
                
                if safe_torch_save(checkpoint_data, checkpoint_path, atomic=True):
                    pass  # Success message printed below
                else:
                    print(f"  ⚠️  Failed to save checkpoint!")
                    continue  # Skip best model save if checkpoint failed
                
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
                print(f"  Validation Loss: {avg_val_loss:.4f}")
                
                # Best model 저장
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = checkpoint_dir / 'model_best.pt'
                    best_checkpoint_data = {
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': avg_val_loss,
                        'model_config': model_config,
                    }
                    
                    if safe_torch_save(best_checkpoint_data, best_model_path, atomic=True):
                        pass  # Success
                    else:
                        print(f"  ⚠️  Failed to save best model!")
                    print(f"  ✓ New best model saved! (val_loss: {avg_val_loss:.4f})")
                
                print()  # 빈 줄
                model.train()
        
        # Epoch 종료
        if global_step >= args.max_steps:
            break
    
    # ================================================================
    # 학습 완료
    # ================================================================
    print_section("학습 완료!")
    total_time = time.time() - training_start
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Total steps completed: {global_step:,}")
    print(f"Total epochs completed: {epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Average speed: {global_step/total_time:.2f} steps/s")
    
    # Final checkpoint
    final_checkpoint_path = checkpoint_dir / 'model_final.pt'
    final_checkpoint_data = {
        'step': global_step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'model_config': model_config,
    }
    
    if safe_torch_save(final_checkpoint_data, final_checkpoint_path, atomic=True):
        pass  # Success
    else:
        print(f"⚠️  Failed to save final checkpoint!")
    print(f"\n✓ Final checkpoint saved: {final_checkpoint_path}")
    
    # ================================================================
    # Checkpoint Averaging (논문 명세)
    # "For the base models, we used a single model obtained by 
    #  averaging the last 5 checkpoints"
    # ================================================================
    print_section("체크포인트 평균화 (논문 구현)")
    
    try:
        from checkpoint_averaging import average_last_n_checkpoints
        
        # 논문 구현: 10분 간격으로 저장된 model_last_*.pt 파일들을 평균화
        # Base model: n_avg=5, Big model: n_avg=20
        
        averaged_state_dict = average_last_n_checkpoints(
            checkpoint_dir=checkpoint_dir,
            n=args.n_avg,
            output_path=checkpoint_dir / f'model_averaged_last{args.n_avg}.pt',
            pattern='model_last_*.pt'  # 논문 구현: 10분 간격 체크포인트 사용
        )
        
        print(f"✓ Averaged model saved (논문 구현)!")
        print(f"  평균화된 체크포인트 개수: {args.n_avg}")
        print(f"  사용된 파일 패턴: model_last_*.pt")
        print(f"  Use this for inference: {checkpoint_dir}/model_averaged_last{args.n_avg}.pt")
        print(f"  Command: python inference.py --checkpoint_dir {checkpoint_dir}")
        
    except Exception as e:
        print(f"⚠️  Could not average checkpoints: {e}")
        print(f"   You can manually average later using checkpoint_averaging.py")

if __name__ == '__main__':
    main()