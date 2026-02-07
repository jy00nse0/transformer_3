# ============================================================================
# 완전한 WMT14 + DataLoader 사용 예시 (진행률 표시 추가)
# ============================================================================

import torch
import torch.nn as nn
from util.tokenizer_with_progress import BPETokenizer
from util.data_loader import DataLoader
from models.model.transformer import Transformer
import time
import sys
def print_section(title):
    """섹션 헤더 출력"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

# ----------------------------------------------------------------------------
# 1. 데이터셋 로드
# ----------------------------------------------------------------------------
print_section("1. 데이터셋 로드")

loader = DataLoader(
    ext=('.en', '.de'),
    tokenize_en=None,  # 아래에서 설정
    tokenize_de=None,
    init_token='<sos>',
    eos_token='<eos>'
)

print("Loading WMT14 dataset (this may take a few minutes)...")
start_time = time.time()

# WMT14 로드 (논문의 실제 데이터셋)
train, valid, test = loader.make_dataset(dataset_name='wmt14')

elapsed = time.time() - start_time
print(f"\n✓ Dataset loaded in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"  Train: {len(train):,} sentence pairs")
print(f"  Valid: {len(valid):,} sentence pairs")
print(f"  Test:  {len(test):,} sentence pairs")

# ----------------------------------------------------------------------------
# 2. BPE 토크나이저 훈련
# ----------------------------------------------------------------------------
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
bpe_en = BPETokenizer(vocab_size=37000)
bpe_en.train(en_corpus)

# 독일어 토크나이저 훈련
print("\n" + "-"*80)
print("Training German BPE Tokenizer")
print("-"*80)
bpe_de = BPETokenizer(vocab_size=37000)
bpe_de.train(de_corpus)

# DataLoader에 토크나이저 설정
loader.tokenize_en = bpe_en.tokenize
loader.tokenize_de = bpe_de.tokenize

print("✓ Both tokenizers trained successfully!")

# ----------------------------------------------------------------------------
# 3. 어휘 사전 구축 (논문 준수)
# ----------------------------------------------------------------------------
print_section("3. 어휘 사전 구축")

print("Building vocabulary (shared, max_size=37000)...")
vocab_start = time.time()

loader.build_vocab(
    train_data=train,
    min_freq=2,              # 최소 빈도
    max_vocab_size=37000,    # 논문: EN-DE shared 37k tokens
    shared_vocab=True        # 논문: shared vocabulary
)

print(f"\n✓ Vocabulary built in {time.time() - vocab_start:.1f}s")

src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)

print(f"\nVocabulary Statistics:")
print(f"  Source vocab size: {enc_voc_size:,}")
print(f"  Target vocab size: {dec_voc_size:,}")
print(f"  Shared: {loader.source.vocab is loader.target.vocab}")
print(f"\nSpecial Tokens:")
print(f"  <pad>: {src_pad_idx}")
print(f"  <unk>: {loader.source.vocab.stoi['<unk>']}")
print(f"  <sos>: {trg_sos_idx}")
print(f"  <eos>: {loader.source.vocab.stoi['<eos>']}")

# ----------------------------------------------------------------------------
# 4. 데이터 Iterator 생성 (논문 준수)
# ----------------------------------------------------------------------------
print_section("4. 데이터 Iterator 생성")

print("Creating data iterators (token-based batching)...")
iter_start = time.time()

train_iter, valid_iter, test_iter = loader.make_iter(
    train, valid, test,
    max_tokens=25000,  # 논문: ~25k tokens per batch
    device='cuda' if torch.cuda.is_available() else 'cpu',
    num_workers=4      # 멀티프로세싱
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

# ----------------------------------------------------------------------------
# 5. 모델 초기화
# ----------------------------------------------------------------------------
print_section("5. 모델 초기화")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("\nInitializing Transformer model...")
model_start = time.time()

model = Transformer(
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    trg_sos_idx=trg_sos_idx,
    d_model=512,              # 논문: base model
    enc_voc_size=enc_voc_size,
    dec_voc_size=dec_voc_size,
    max_len=512,
    ffn_hidden=2048,          # 논문: d_ff
    n_head=8,                 # 논문: h=8
    n_layers=6,               # 논문: N=6
    drop_prob=0.1,            # 논문: P_drop=0.1
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

# ----------------------------------------------------------------------------
# 6. Optimizer 및 Scheduler (논문 준수)
# ----------------------------------------------------------------------------
print_section("6. Optimizer 및 Scheduler")

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

optimizer = Adam(
    model.parameters(),
    lr=1.0,              # learning rate schedule로 조정됨
    betas=(0.9, 0.98),   # 논문: β_1=0.9, β_2=0.98
    eps=1e-9             # 논문: ε=10^-9
)

def lr_schedule(step):
    # 논문: lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    d_model = 512
    warmup_steps = 4000
    return (d_model ** -0.5) * min(
        (step + 1) ** -0.5,
        (step + 1) * (warmup_steps ** -1.5)
    )

scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

print("✓ Optimizer: Adam (β1=0.9, β2=0.98, ε=1e-9)")
print("✓ Scheduler: Custom warmup schedule (warmup_steps=4000)")

# ----------------------------------------------------------------------------
# 7. Loss Function (Label Smoothing)
# ----------------------------------------------------------------------------
print_section("7. Loss Function")

criterion = nn.CrossEntropyLoss(
    ignore_index=trg_pad_idx,
    label_smoothing=0.1  # 논문: ε_ls=0.1
)

print("✓ Loss: CrossEntropyLoss with label smoothing (ε=0.1)")

# ----------------------------------------------------------------------------
# 8. 학습 루프
# ----------------------------------------------------------------------------
print_section("8. 학습 시작")

print("Training configuration:")
print(f"  Epochs: 100")
print(f"  Batches per epoch: {len(train_iter):,}")
print(f"  Total steps: {100 * len(train_iter):,}")
print(f"  Device: {device}")

input("\nPress Enter to start training...")

training_start = time.time()

for epoch in range(100):
    epoch_start = time.time()
    
    # ========================================================================
    # Training
    # ========================================================================
    model.train()
    epoch_loss = 0
    
    print(f"\n{'='*80}")
    print(f"Epoch {epoch + 1}/100")
    print(f"{'='*80}")
    
    for batch_idx, batch in enumerate(train_iter):
        batch_start = time.time()
        
        src = batch.src  # [variable_batch_size, src_len]
        trg = batch.trg  # [variable_batch_size, trg_len]
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, trg[:, :-1])  # Teacher forcing
        
        # Compute loss
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg_reshape = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output_reshape, trg_reshape)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        
        # 진행률 표시 (매 100 배치마다)
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
    
    # ========================================================================
    # Validation
    # ========================================================================
    model.eval()
    val_loss = 0
    
    print(f"\n  Validating...", end='')
    sys.stdout.flush()
    
    val_start = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_iter):
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
    
    # ========================================================================
    # Epoch Summary
    # ========================================================================
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
        checkpoint_path = f'saved/model_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"\n  ✓ Checkpoint saved: {checkpoint_path}")

# ============================================================================
# 학습 완료
# ============================================================================
total_training_time = time.time() - training_start

print_section("학습 완료!")
print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} min)")
print(f"Final model saved to: saved/model_final.pt")

torch.save(model.state_dict(), 'saved/model_final.pt')

print("\n" + "="*80)
