# ============================================================================
# WMT14 + DataLoader with Save/Load Functionality
# ============================================================================

import torch
import torch.nn as nn
from util.tokenizer_with_progress import BPETokenizer
from util.data_loader import DataLoader
from models.model.transformer import Transformer
import time
import sys
import argparse
import pickle
import os
from pathlib import Path
import json

def print_section(title):
    """섹션 헤더 출력"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

def save_tokenizer(tokenizer, filepath):
    """토크나이저를 파일로 저장"""
    print(f"Saving tokenizer to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✓ Tokenizer saved")

def load_tokenizer(filepath):
    """토크나이저를 파일에서 로드"""
    print(f"Loading tokenizer from {filepath}...")
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer.vocab):,})")
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
    print(f"  Source vocab size: {len(vocab_data['source_stoi']):,}")
    print(f"  Target vocab size: {len(vocab_data['target_stoi']):,}")
    print(f"  Shared: {vocab_data['shared']}")

def load_vocab(loader, filepath):
    """어휘사전을 파일에서 로드"""
    print(f"Loading vocabulary from {filepath}...")
    
    with open(filepath, 'rb') as f:
        vocab_data = pickle.load(f)
    
    # vocab 객체 생성
    source_vocab_obj = type('obj', (object,), {
        'stoi': vocab_data['source_stoi'],
        'itos': vocab_data['source_itos'],
        '__len__': lambda self: len(vocab_data['source_stoi'])
    })()
    
    if vocab_data['shared']:
        # 공유 어휘인 경우
        loader.source.vocab = source_vocab_obj
        loader.target.vocab = source_vocab_obj
    else:
        # 별도 어휘인 경우
        target_vocab_obj = type('obj', (object,), {
            'stoi': vocab_data['target_stoi'],
            'itos': vocab_data['target_itos'],
            '__len__': lambda self: len(vocab_data['target_stoi'])
        })()
        
        loader.source.vocab = source_vocab_obj
        loader.target.vocab = target_vocab_obj
    
    print(f"✓ Vocabulary loaded")
    print(f"  Source vocab size: {len(loader.source.vocab):,}")
    print(f"  Target vocab size: {len(loader.target.vocab):,}")
    print(f"  Shared: {vocab_data['shared']}")

def save_training_artifacts(tokenizer_en, tokenizer_de, loader, save_dir):
    """모든 학습 결과물을 저장"""
    print_section("저장 중...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 토크나이저 저장
    save_tokenizer(tokenizer_en, save_dir / 'tokenizer_en.pkl')
    save_tokenizer(tokenizer_de, save_dir / 'tokenizer_de.pkl')
    
    # 어휘사전 저장
    save_vocab(loader, save_dir / 'vocab.pkl')
    
    # 메타데이터 저장
    metadata = {
        'vocab_size': len(loader.source.vocab),
        'shared_vocab': loader.source.vocab is loader.target.vocab,
        'src_pad_idx': loader.source.vocab.stoi['<pad>'],
        'trg_pad_idx': loader.target.vocab.stoi['<pad>'],
        'trg_sos_idx': loader.target.vocab.stoi['<sos>'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ All artifacts saved to: {save_dir}")
    print(f"  - tokenizer_en.pkl")
    print(f"  - tokenizer_de.pkl")
    print(f"  - vocab.pkl")
    print(f"  - metadata.json")

def load_training_artifacts(loader, load_dir):
    """저장된 학습 결과물을 로드"""
    print_section("기존 결과물 로드 중...")
    
    load_dir = Path(load_dir)
    
    # 필수 파일 확인
    required_files = ['tokenizer_en.pkl', 'tokenizer_de.pkl', 'vocab.pkl', 'metadata.json']
    missing_files = [f for f in required_files if not (load_dir / f).exists()]
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    # 메타데이터 로드
    with open(load_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Metadata:")
    print(f"  Created: {metadata['timestamp']}")
    print(f"  Vocab size: {metadata['vocab_size']:,}")
    print(f"  Shared vocab: {metadata['shared_vocab']}")
    
    # 토크나이저 로드
    print()
    tokenizer_en = load_tokenizer(load_dir / 'tokenizer_en.pkl')
    tokenizer_de = load_tokenizer(load_dir / 'tokenizer_de.pkl')
    
    # 어휘사전 로드
    print()
    load_vocab(loader, load_dir / 'vocab.pkl')
    
    print(f"\n✓ All artifacts loaded from: {load_dir}")
    
    return tokenizer_en, tokenizer_de, metadata

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='WMT14 Transformer Training with Save/Load')
    
    parser.add_argument(
        '--load_dir',
        type=str,
        default=None,
        help='Directory to load pre-trained tokenizers and vocabulary (skip training)'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='artifacts',
        help='Directory to save tokenizers and vocabulary (default: artifacts)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=25000,
        help='Maximum tokens per batch (default: 25000)'
    )
    
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=37000,
        help='Vocabulary size (default: 37000)'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints_test',
        help='Directory to save model checkpoints (default: checkpoints)'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*80)
    print(" "*20 + "WMT14 Transformer Training")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Load directory: {args.load_dir if args.load_dir else 'None (train from scratch)'}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Max tokens per batch: {args.max_tokens:,}")
    print(f"  Vocabulary size: {args.vocab_size:,}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    
    # ------------------------------------------------------------------------
    # 1. 데이터셋 로드
    # ------------------------------------------------------------------------
    print_section("1. 데이터셋 로드")
    
    loader = DataLoader(
        ext=('.en', '.de'),
        tokenize_en=None,  # 아래에서 설정
        tokenize_de=None,
        init_token='<sos>',
        eos_token='<eos>'
    )
    
    print("Loading WMT14 dataset...")
    start_time = time.time()
    
    train, valid, test = loader.make_dataset(dataset_name='wmt14')
    
    elapsed = time.time() - start_time
    print(f"\n✓ Dataset loaded in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Train: {len(train):,} sentence pairs")
    print(f"  Valid: {len(valid):,} sentence pairs")
    print(f"  Test:  {len(test):,} sentence pairs")
    
    # ------------------------------------------------------------------------
    # 2. 토크나이저 및 어휘사전 로드 또는 학습
    # ------------------------------------------------------------------------
    if args.load_dir:
        # 기존 결과물 로드
        tokenizer_en, tokenizer_de, metadata = load_training_artifacts(loader, args.load_dir)
        
        # DataLoader에 토크나이저 설정
        loader.tokenize_en = tokenizer_en.tokenize
        loader.tokenize_de = tokenizer_de.tokenize
        
    else:
        # 새로 학습
        print_section("2. BPE 토크나이저 훈련")
        
        # 코퍼스 추출
        print("Extracting corpus from dataset...")
        corpus_start = time.time()
        en_corpus = [item['translation']['en'] for item in train]
        de_corpus = [item['translation']['de'] for item in train]
        print(f"✓ Corpus extracted in {time.time() - corpus_start:.1f}s")
        print(f"  English corpus: {len(en_corpus):,} sentences")
        print(f"  German corpus: {len(de_corpus):,} sentences")
        
        # 영어 토크나이저 훈련
        print("\n" + "-"*80)
        print("Training English BPE Tokenizer")
        print("-"*80)
        tokenizer_en = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer_en.train(en_corpus)
        
        # 독일어 토크나이저 훈련
        print("\n" + "-"*80)
        print("Training German BPE Tokenizer")
        print("-"*80)
        tokenizer_de = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer_de.train(de_corpus)
        
        print("\n✓ Both tokenizers trained successfully!")
        
        # DataLoader에 토크나이저 설정
        loader.tokenize_en = tokenizer_en.tokenize
        loader.tokenize_de = tokenizer_de.tokenize
        
        # ------------------------------------------------------------------------
        # 3. 어휘 사전 구축
        # ------------------------------------------------------------------------
        print_section("3. 어휘 사전 구축")
        
        print(f"Building vocabulary (shared, max_size={args.vocab_size:,})...")
        vocab_start = time.time()
        
        loader.build_vocab(
            train_data=train,
            min_freq=2,
            max_vocab_size=args.vocab_size,
            shared_vocab=True
        )
        
        print(f"\n✓ Vocabulary built in {time.time() - vocab_start:.1f}s")
        
        # ------------------------------------------------------------------------
        # 4. 학습 결과물 저장
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
    # 5. 데이터 Iterator 생성
    # ------------------------------------------------------------------------
    print_section("5. 데이터 Iterator 생성")
    
    print("Creating data iterators (token-based batching)...")
    iter_start = time.time()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_iter, valid_iter, test_iter = loader.make_iter(
        train, valid, test,
        max_tokens=args.max_tokens,
        device=device,
        num_workers=4
    )
    
    print(f"✓ Iterators created in {time.time() - iter_start:.1f}s")
    print(f"\nIterator Statistics:")
    print(f"  Train batches: {len(train_iter):,}")
    print(f"  Valid batches: {len(valid_iter):,}")
    print(f"  Test batches:  {len(test_iter):,}")
    
    # 첫 번째 배치 정보
    print("\nFirst batch info:")
    for batch in train_iter:
        total_tokens = batch.src.shape[0] * batch.src.shape[1]
        print(f"  Batch size (sentences): {batch.src.shape[0]}")
        print(f"  Max sequence length: {batch.src.shape[1]}")
        print(f"  Total tokens: {total_tokens:,}")
        break
    
    # ------------------------------------------------------------------------
    # 6. 모델 초기화
    # ------------------------------------------------------------------------
    print_section("6. 모델 초기화")
    
    print(f"Using device: {device}")
    
    print("\nInitializing Transformer model...")
    model_start = time.time()
    
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        d_model=512,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=512,
        ffn_hidden=2048,
        n_head=8,
        n_layers=6,
        drop_prob=0.1,
        device=device
    ).to(device)
    
    print(f"✓ Model initialized in {time.time() - model_start:.1f}s")
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / (1024**2):.1f} MB (fp32)")
    
    # ------------------------------------------------------------------------
    # 7. Optimizer 및 Scheduler
    # ------------------------------------------------------------------------
    print_section("7. Optimizer 및 Scheduler")
    
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LambdaLR
    
    optimizer = Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    def lr_schedule(step):
        d_model = 512
        warmup_steps = 4000
        return (d_model ** -0.5) * min(
            (step + 1) ** -0.5,
            (step + 1) * (warmup_steps ** -1.5)
        )
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    
    print("✓ Optimizer: Adam (β1=0.9, β2=0.98, ε=1e-9)")
    print("✓ Scheduler: Custom warmup schedule (warmup_steps=4000)")
    
    # ------------------------------------------------------------------------
    # 8. Loss Function
    # ------------------------------------------------------------------------
    print_section("8. Loss Function")
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=trg_pad_idx,
        label_smoothing=0.1
    )
    
    print("✓ Loss: CrossEntropyLoss with label smoothing (ε=0.1)")
    
    # ------------------------------------------------------------------------
    # 9. 학습 루프
    # ------------------------------------------------------------------------
    print_section("9. 학습 시작")
    
    print("Training configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batches per epoch: {len(train_iter):,}")
    print(f"  Total steps: {args.epochs * len(train_iter):,}")
    print(f"  Device: {device}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    input("\nPress Enter to start training...")
    
    training_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # ====================================================================
        # Training
        # ====================================================================
        model.train()
        epoch_loss = 0
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        for batch_idx, batch in enumerate(train_iter):
            src = batch.src
            trg = batch.trg
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, trg[:, :-1])
            
            # Compute loss
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_reshape = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output_reshape, trg_reshape)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            # 진행률 표시
            if batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start
                progress = (batch_idx + 1) / len(train_iter) * 100
                batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(train_iter) - batch_idx - 1) / batches_per_sec if batches_per_sec > 0 else 0
                
                lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / (batch_idx + 1)
                
                print(f"  Batch {batch_idx + 1:,}/{len(train_iter):,} ({progress:.1f}%) | "
                      f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                      f"LR: {lr:.6f} | "
                      f"Speed: {batches_per_sec:.1f} batch/s | "
                      f"ETA: {eta:.0f}s", end='\r')
        
        train_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / len(train_iter)
        
        print(f"  Batch {len(train_iter):,}/{len(train_iter):,} (100.0%) | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Time: {train_time:.1f}s                    ")
        
        # ====================================================================
        # Validation
        # ====================================================================
        model.eval()
        val_loss = 0
        
        print(f"\n  Validating...", end='')
        sys.stdout.flush()
        
        val_start = time.time()
        
        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                trg = batch.trg
                
                output = model(src, trg[:, :-1])
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg_reshape = trg[:, 1:].contiguous().view(-1)
                
                loss = criterion(output_reshape, trg_reshape)
                val_loss += loss.item()
        
        val_time = time.time() - val_start
        avg_val_loss = val_loss / len(valid_iter)
        
        print(f"\r  Validation complete in {val_time:.1f}s")
        
        # ====================================================================
        # Epoch Summary
        # ====================================================================
        total_epoch_time = time.time() - epoch_start
        
        print(f"\n  {'─'*76}")
        print(f"  Epoch {epoch + 1} Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f} | Train PPL: {torch.exp(torch.tensor(avg_train_loss)):.2f}")
        print(f"    Val Loss:   {avg_val_loss:.4f} | Val PPL:   {torch.exp(torch.tensor(avg_val_loss)):.2f}")
        print(f"    Epoch Time: {total_epoch_time:.1f}s ({total_epoch_time/60:.1f} min)")
        print(f"    Total Time: {(time.time() - training_start)/3600:.2f} hours")
        print(f"  {'─'*76}")
        
        # 모델 저장 (10 에폭마다)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"\n  ✓ Checkpoint saved: {checkpoint_path}")
    
    # ========================================================================
    # 학습 완료
    # ========================================================================
    total_training_time = time.time() - training_start
    
    print_section("학습 완료!")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} min)")
    
    final_model_path = checkpoint_dir / 'model_final.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
