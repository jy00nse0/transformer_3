"""
DataLoader 사용 예시 - 순수 Python 버전
실제 환경에서는 torch, datasets, transformers가 필요합니다.
"""

# 샘플 데이터셋
class SampleDataset:
    """WMT14를 모방한 샘플 데이터"""
    def __init__(self, size=100):
        self.data = []
        
        # 영어-독일어 예문
        samples = [
            ("The quick brown fox jumps over the lazy dog.", 
             "Der schnelle braune Fuchs springt über den faulen Hund."),
            ("Machine translation is a challenging task.", 
             "Maschinelle Übersetzung ist eine herausfordernde Aufgabe."),
            ("Attention is all you need.", 
             "Aufmerksamkeit ist alles, was Sie brauchen."),
            ("Neural networks learn from data.", 
             "Neuronale Netze lernen aus Daten."),
            ("The model achieves state-of-the-art results.", 
             "Das Modell erzielt hochmoderne Ergebnisse."),
        ]
        
        for i in range(size):
            en, de = samples[i % len(samples)]
            self.data.append({'translation': {'en': en, 'de': de}})
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def demo_1_data_loading():
    """1단계: 데이터 로딩"""
    print("="*80)
    print("STEP 1: 데이터 로딩")
    print("="*80)
    
    print("\n실제 코드:")
    print("-"*80)
    print("""
from datasets import load_dataset

# Multi30k 로드 (작은 데이터셋)
dataset = load_dataset('bentrevett/multi30k')

# 또는 WMT14 로드 (논문의 데이터셋, 4.5M 문장)
dataset = load_dataset('wmt14', 'de-en')

train = dataset['train']
valid = dataset['validation']
test = dataset['test']
    """)
    
    print("\n데모 (샘플 데이터):")
    print("-"*80)
    train = SampleDataset(100)
    valid = SampleDataset(20)
    test = SampleDataset(20)
    
    print(f"✓ Train: {len(train)} samples")
    print(f"✓ Valid: {len(valid)} samples")
    print(f"✓ Test:  {len(test)} samples")
    
    sample = train[0]
    print(f"\n샘플 데이터:")
    print(f"  English: {sample['translation']['en']}")
    print(f"  German:  {sample['translation']['de']}")
    
    return train, valid, test


def demo_2_tokenizer_training(train):
    """2단계: 토크나이저 훈련"""
    print("\n" + "="*80)
    print("STEP 2: BPE 토크나이저 훈련")
    print("="*80)
    
    print("\n실제 코드:")
    print("-"*80)
    print("""
from tokenizer import BPETokenizer

# 코퍼스 추출
english_corpus = [item['translation']['en'] for item in train]
german_corpus = [item['translation']['de'] for item in train]

# BPE 토크나이저 훈련
bpe_en = BPETokenizer(vocab_size=37000)  # 논문: 37k tokens
bpe_en.train(english_corpus)

bpe_de = BPETokenizer(vocab_size=37000)
bpe_de.train(german_corpus)

# 테스트
tokens = bpe_en.tokenize("Hello world")
print(tokens)  # ['Hello', 'Ġworld']
    """)
    
    print("\n데모:")
    print("-"*80)
    
    # 간단한 토크나이저 (공백 분할)
    class SimpleTokenizer:
        def tokenize(self, text):
            return text.lower().split()
    
    tokenizer_en = SimpleTokenizer()
    tokenizer_de = SimpleTokenizer()
    
    test_text = "The quick brown fox"
    tokens = tokenizer_en.tokenize(test_text)
    
    print(f"Text:   '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"✓ 토크나이저 준비 완료")
    
    return tokenizer_en, tokenizer_de


def demo_3_dataloader_usage(train, valid, test, tokenizer_en, tokenizer_de):
    """3단계: DataLoader 사용"""
    print("\n" + "="*80)
    print("STEP 3: DataLoader 초기화 및 사용")
    print("="*80)
    
    print("\n실제 코드:")
    print("-"*80)
    print("""
from data_loader_improved import DataLoader

# 1. DataLoader 초기화
loader = DataLoader(
    ext=('.en', '.de'),
    tokenize_en=bpe_en.tokenize,
    tokenize_de=bpe_de.tokenize,
    init_token='<sos>',
    eos_token='<eos>'
)

# 2. 데이터셋 로드
train, valid, test = loader.make_dataset(dataset_name='wmt14')
# 또는 Multi30k: loader.make_dataset(dataset_name='bentrevett/multi30k')

# 3. 어휘 사전 구축 (논문 준수)
loader.build_vocab(
    train_data=train,
    min_freq=2,              # 최소 빈도
    max_vocab_size=37000,    # 논문: EN-DE 37k tokens
    shared_vocab=True        # 논문: source-target 공유
)

# 4. Iterator 생성 (논문 준수)
train_iter, valid_iter, test_iter = loader.make_iter(
    train, valid, test,
    max_tokens=25000,  # 논문: ~25k tokens per batch
    device='cuda',
    num_workers=4
)

# 5. 학습 루프
for batch in train_iter:
    src = batch.src  # [batch_size, src_len]
    trg = batch.trg  # [batch_size, trg_len]
    
    # 모델 학습
    output = model(src, trg[:, :-1])
    loss = criterion(output, trg[:, 1:])
    """)
    
    print("\n주요 특징:")
    print("-"*80)
    print("✓ 토큰 수 기반 배치 생성 (논문: 배치당 ~25,000 토큰)")
    print("✓ 길이 기반 정렬 (패딩 최소화)")
    print("✓ 공유 어휘 지원 (논문: EN-DE 37k shared vocab)")
    print("✓ 어휘 크기 제한 (재현성)")
    print("✓ 기존 코드와 100% 호환")


def demo_4_vocab_details():
    """4단계: 어휘 사전 상세"""
    print("\n" + "="*80)
    print("STEP 4: 어휘 사전 구축 상세")
    print("="*80)
    
    print("\n[공유 어휘 vs 별도 어휘]")
    print("-"*80)
    
    print("\n1. 공유 어휘 (Shared Vocabulary) - 논문의 EN-DE 방식")
    print("""
loader.build_vocab(
    train_data=train,
    min_freq=2,
    max_vocab_size=37000,
    shared_vocab=True  # ← 핵심
)

결과:
- source와 target이 같은 어휘 사전 공유
- 논문: "byte-pair encoding with shared source-target vocabulary of ~37,000 tokens"
- 장점: 메모리 효율적, 언어 간 토큰 일치
    """)
    
    print("\n2. 별도 어휘 (Separate Vocabularies)")
    print("""
loader.build_vocab(
    train_data=train,
    min_freq=2,
    max_vocab_size=32000,  # EN-FR 방식
    shared_vocab=False
)

결과:
- source와 target이 별도 어휘 사전
- 논문의 EN-FR: 32,000 word-piece tokens
- 장점: 언어별 최적화
    """)
    
    print("\n[어휘 접근 방법]")
    print("-"*80)
    print("""
# 특수 토큰 인덱스
pad_idx = loader.source.vocab.stoi['<pad>']  # 0
unk_idx = loader.source.vocab.stoi['<unk>']  # 1
sos_idx = loader.source.vocab.stoi['<sos>']  # 2
eos_idx = loader.source.vocab.stoi['<eos>']  # 3

# 어휘 크기
vocab_size = len(loader.source.vocab)

# 토큰 -> 인덱스
idx = loader.source.vocab.stoi['hello']

# 인덱스 -> 토큰
token = loader.source.vocab.itos[42]
    """)


def demo_5_batch_mechanism():
    """5단계: 배치 생성 메커니즘"""
    print("\n" + "="*80)
    print("STEP 5: 토큰 기반 배치 생성 (핵심 메커니즘)")
    print("="*80)
    
    print("\n[기존 방식 vs 논문 방식]")
    print("-"*80)
    
    print("\n❌ 기존 방식 (고정 문장 개수):")
    print("""
batch_size = 128  # 항상 128개 문장

문제점:
- 짧은 문장 배치: 128 × 20토큰 = 2,560 토큰 (GPU 저활용)
- 긴 문장 배치: 128 × 500토큰 = 64,000 토큰 (OOM 위험)
- 메모리 사용량 불규칙
    """)
    
    print("\n✅ 논문 방식 (토큰 수 기반):")
    print("""
max_tokens = 25000  # 배치당 약 25,000 토큰

장점:
- 짧은 문장: ~125개 문장 × 200토큰 = 25,000 토큰
- 긴 문장: ~50개 문장 × 500토큰 = 25,000 토큰
- 메모리 사용량 일정
- GPU 활용도 최대화
    """)
    
    print("\n[TokenBucketSampler 작동 원리]")
    print("-"*80)
    
    sentences = [
        ("짧은1", 10),
        ("짧은2", 12),
        ("짧은3", 8),
        ("긴1", 50),
        ("긴2", 52),
        ("긴3", 48),
    ]
    
    print("\n1. 문장들을 길이로 정렬:")
    sorted_sents = sorted(sentences, key=lambda x: x[1])
    for name, length in sorted_sents:
        print(f"   {name}: {length} 토큰")
    
    print("\n2. 토큰 수 기준으로 배치 생성 (max_tokens=100):")
    print("""
   Batch 1: [짧은3(8), 짧은1(10), 짧은2(12)]
   - 최대 길이: 12
   - 문장 수: 3
   - 총 토큰: 12 × 3 = 36 토큰
   
   Batch 2: [긴3(48), 긴1(50)]
   - 최대 길이: 50
   - 문장 수: 2
   - 총 토큰: 50 × 2 = 100 토큰
   
   → 각 배치가 약 100 토큰 유지
    """)
    
    print("\n3. 패딩 최소화:")
    print("""
   비슷한 길이끼리 묶여서 패딩 감소
   - 랜덤 배치: [10, 50, 12, 48] → 패딩 45%
   - 정렬 배치: [10, 12, 8] → 패딩 20%
    """)


def demo_6_complete_example():
    """6단계: 완전한 사용 예시"""
    print("\n" + "="*80)
    print("STEP 6: 완전한 사용 예시")
    print("="*80)
    
    print("""
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
    """)


def main():
    """전체 데모 실행"""
    print("\n" + "="*80)
    print(" "*25 + "DataLoader 사용 가이드")
    print("="*80)
    
    # 1. 데이터 로딩
    train, valid, test = demo_1_data_loading()
    
    # 2. 토크나이저
    tokenizer_en, tokenizer_de = demo_2_tokenizer_training(train)
    
    # 3. DataLoader 사용
    demo_3_dataloader_usage(train, valid, test, tokenizer_en, tokenizer_de)
    
    # 4. 어휘 상세
    demo_4_vocab_details()
    
    # 5. 배치 메커니즘
    demo_5_batch_mechanism()
    
    # 6. 완전한 예시
    demo_6_complete_example()
    
    print("\n" + "="*80)
    print("데모 완료!")
    print("="*80)
    print("\n실제 사용 시 필요한 패키지:")
    print("  pip install torch datasets transformers")
    print("\n제공된 파일:")
    print("  - data_loader_improved.py: 논문 준수 DataLoader")
    print("  - tokenizer.py: BPE/WordPiece 토크나이저")
    print("  - transformer.py: Transformer 모델")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
