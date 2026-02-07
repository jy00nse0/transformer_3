# ============================================================================
# 완전한 WMT14 + DataLoader 사용 예시
# ============================================================================

import torch
from tokenizer import BPETokenizer
from data_loader_improved import DataLoader
from transformer import Transformer

# ----------------------------------------------------------------------------
# 1. 데이터셋 로드
# ----------------------------------------------------------------------------
loader = DataLoader(
    ext=('.en', '.de'),
    tokenize_en=None,  # 아래에서 설정
    tokenize_de=None,
    init_token='<sos>',
    eos_token='<eos>'
)

# WMT14 로드 (논문의 실제 데이터셋)
train, valid, test = loader.make_dataset(dataset_name='wmt14')
# Train: ~4.5M sentence pairs

# ----------------------------------------------------------------------------
# 2. BPE 토크나이저 훈련
# ----------------------------------------------------------------------------
print("Training BPE tokenizers...")

# 코퍼스 추출
en_corpus = [item['translation']['en'] for item in train]
de_corpus = [item['translation']['de'] for item in train]

# 토크나이저 훈련 (논문: vocab_size=37000)
bpe_en = BPETokenizer(vocab_size=37000)
bpe_de = BPETokenizer(vocab_size=37000)

bpe_en.train(en_corpus)
bpe_de.train(de_corpus)

# DataLoader에 토크나이저 설정
loader.tokenize_en = bpe_en.tokenize
loader.tokenize_de = bpe_de.tokenize

# ----------------------------------------------------------------------------
# 3. 어휘 사전 구축 (논문 준수)
# ----------------------------------------------------------------------------
loader.build_vocab(
    train_data=train,
    min_freq=2,              # 최소 빈도
    max_vocab_size=37000,    # 논문: EN-DE shared 37k tokens
    shared_vocab=True        # 논문: shared vocabulary
)

src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)

print(f"Vocabulary size: {enc_voc_size}")

# ----------------------------------------------------------------------------
# 4. 데이터 Iterator 생성 (논문 준수)
# ----------------------------------------------------------------------------
train_iter, valid_iter, test_iter = loader.make_iter(
    train, valid, test,
    max_tokens=25000,  # 논문: ~25k tokens per batch
    device='cuda',
    num_workers=4      # 멀티프로세싱
)

print(f"Train batches: {len(train_iter)}")
print(f"Valid batches: {len(valid_iter)}")

# ----------------------------------------------------------------------------
# 5. 모델 초기화
# ----------------------------------------------------------------------------
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
    device='cuda'
).to('cuda')

# ----------------------------------------------------------------------------
# 6. Optimizer 및 Scheduler (논문 준수)
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# 7. Loss Function (Label Smoothing)
# ----------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss(
    ignore_index=trg_pad_idx,
    label_smoothing=0.1  # 논문: ε_ls=0.1
)

# ----------------------------------------------------------------------------
# 8. 학습 루프
# ----------------------------------------------------------------------------
print("Starting training...")

for epoch in range(100):
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(train_iter):
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
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in valid_iter:
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            # ... compute validation loss
    
    print(f"Epoch {epoch}: Train Loss={epoch_loss/len(train_iter):.4f}, "
          f"Val Loss={val_loss/len(valid_iter):.4f}")

# ============================================================================
# 끝
# ============================================================================
