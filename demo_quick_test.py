# ============================================================================
# WMT14 DataLoader 빠른 테스트 (일부 데이터만 사용)
# ============================================================================

import torch
import torch.nn as nn
from tokenizer_with_progress import BPETokenizer
from data_loader_improved import DataLoader
from transformer import Transformer
import time

def print_section(title):
    """섹션 헤더 출력"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

# 설정
SAMPLE_SIZE = 10000  # 전체 450만개 중 1만개만 사용
VOCAB_SIZE = 5000    # 작은 어휘 크기
EPOCHS = 3           # 짧은 학습

print_section("빠른 테스트 모드")
print(f"설정:")
print(f"  샘플 크기: {SAMPLE_SIZE:,} (전체의 {SAMPLE_SIZE/4508785*100:.2f}%)")
print(f"  어휘 크기: {VOCAB_SIZE:,}")
print(f"  에폭 수: {EPOCHS}")

# ----------------------------------------------------------------------------
# 1. 데이터셋 로드
# ----------------------------------------------------------------------------
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

# WMT14 로드
full_dataset = loader.make_dataset(dataset_name='wmt14')

# 샘플링
train = type('obj', (object,), {
    'data': [full_dataset[0][i] for i in range(min(SAMPLE_SIZE, len(full_dataset[0])))],
    '__len__': lambda self: len(self.data),
    '__getitem__': lambda self, idx: self.data[idx]
})()

valid = type('obj', (object,), {
    'data': [full_dataset[1][i] for i in range(min(1000, len(full_dataset[1])))],
    '__len__': lambda self: len(self.data),
    '__getitem__': lambda self, idx: self.data[idx]
})()

test = full_dataset[2]

elapsed = time.time() - start_time
print(f"\n✓ Dataset loaded in {elapsed:.1f}s")
print(f"  Train: {len(train):,} sentence pairs")
print(f"  Valid: {len(valid):,} sentence pairs")

# ----------------------------------------------------------------------------
# 2. BPE 토크나이저 훈련
# ----------------------------------------------------------------------------
print_section("2. BPE 토크나이저 훈련")

print("Extracting corpus...")
en_corpus = [item['translation']['en'] for item in train]
de_corpus = [item['translation']['de'] for item in train]

print(f"\n영어 토크나이저 훈련 ({VOCAB_SIZE:,} vocab)...")
bpe_en = BPETokenizer(vocab_size=VOCAB_SIZE)
bpe_en.train(en_corpus)

print(f"\n독일어 토크나이저 훈련 ({VOCAB_SIZE:,} vocab)...")
bpe_de = BPETokenizer(vocab_size=VOCAB_SIZE)
bpe_de.train(de_corpus)

loader.tokenize_en = bpe_en.tokenize
loader.tokenize_de = bpe_de.tokenize

# ----------------------------------------------------------------------------
# 3. 어휘 사전 구축
# ----------------------------------------------------------------------------
print_section("3. 어휘 사전 구축")

loader.build_vocab(
    train_data=train,
    min_freq=2,
    max_vocab_size=VOCAB_SIZE,
    shared_vocab=True
)

src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)

print(f"Vocabulary size: {enc_voc_size:,}")

# ----------------------------------------------------------------------------
# 4. Iterator 생성
# ----------------------------------------------------------------------------
print_section("4. Iterator 생성")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_iter, valid_iter, test_iter = loader.make_iter(
    train, valid, test,
    max_tokens=5000,  # 작게 설정
    device=device,
    num_workers=2
)

print(f"Train batches: {len(train_iter):,}")
print(f"Valid batches: {len(valid_iter):,}")

# ----------------------------------------------------------------------------
# 5. 모델 초기화
# ----------------------------------------------------------------------------
print_section("5. 모델 초기화")

model = Transformer(
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    trg_sos_idx=trg_sos_idx,
    d_model=256,        # 작게
    enc_voc_size=enc_voc_size,
    dec_voc_size=dec_voc_size,
    max_len=256,
    ffn_hidden=512,     # 작게
    n_head=4,           # 작게
    n_layers=3,         # 작게
    drop_prob=0.1,
    device=device
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ----------------------------------------------------------------------------
# 6. Optimizer
# ----------------------------------------------------------------------------
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

def lr_schedule(step):
    d_model = 256
    warmup_steps = 1000
    return (d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))

scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)

# ----------------------------------------------------------------------------
# 7. 학습
# ----------------------------------------------------------------------------
print_section("7. 학습 시작")

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 80)
    
    # Training
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(train_iter):
        src, trg = batch.src, batch.trg
        
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        
        loss = criterion(
            output.contiguous().view(-1, output.shape[-1]),
            trg[:, 1:].contiguous().view(-1)
        )
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 10 == 0:
            progress = (batch_idx + 1) / len(train_iter) * 100
            avg_loss = epoch_loss / (batch_idx + 1)
            lr = scheduler.get_last_lr()[0]
            
            print(f"  Batch {batch_idx + 1:>4}/{len(train_iter)} ({progress:>5.1f}%) | "
                  f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                  f"LR: {lr:.6f}", end='\r')
    
    train_time = time.time() - epoch_start
    avg_train_loss = epoch_loss / len(train_iter)
    
    print(f"  Batch {len(train_iter):>4}/{len(train_iter)} (100.0%) | "
          f"Loss: {avg_train_loss:.4f} | "
          f"Time: {train_time:.1f}s         ")
    
    # Validation
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in valid_iter:
            src, trg = batch.src, batch.trg
            output = model(src, trg[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, output.shape[-1]),
                trg[:, 1:].contiguous().view(-1)
            )
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(valid_iter)
    
    print(f"\n  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

print_section("테스트 완료!")
print("✓ DataLoader가 정상적으로 작동합니다.")
print("✓ 토크나이저가 정상적으로 작동합니다.")
print("✓ 모델 학습이 정상적으로 진행됩니다.")
print("\n전체 데이터셋으로 학습하려면 demo_wmt14_with_progress.py를 사용하세요.")
