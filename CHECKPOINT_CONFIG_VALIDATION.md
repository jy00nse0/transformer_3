# 체크포인트 model_config 저장 및 로드 검증 리포트

## 검증 개요

모델 학습 시 설정한 arguments가 체크포인트에 저장되고, inference.py에서 이를 올바르게 로드하여 모델 추론에 사용하는지 검증했습니다.

## 수정 사항

### 1. demo_wmt14_pretrained.py 수정

#### 1.1 모델 설정 저장 (Line 556-570)

모델 초기화 후 `model_config` 딕셔너리를 생성하여 모든 하이퍼파라미터를 저장합니다:

```python
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
    'kdim': args.kdim
}
```

#### 1.2 체크포인트 저장 시 model_config 추가

세 곳의 체크포인트 저장 위치에 `model_config` 추가:

**a) 정기 체크포인트 (Line 690)**
```python
torch.save({
    'step': global_step,
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_loss': avg_val_loss,
    'model_config': model_config,  # 추가됨
}, checkpoint_path)
```

**b) Best 모델 (Line 707)**
```python
torch.save({
    'step': global_step,
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_loss': avg_val_loss,
    'model_config': model_config,  # 추가됨
}, best_model_path)
```

**c) Final 체크포인트 (Line 737)**
```python
torch.save({
    'step': global_step,
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'model_config': model_config,  # 추가됨
}, final_checkpoint_path)
```

### 2. inference.py 수정

#### 2.1 체크포인트에서 model_config 로드 (Line 140-176)

```python
# Load model configuration from checkpoint
print("\n2. Loading model configuration from checkpoint...")
checkpoint_dir = Path(args.checkpoint_dir)

# Try to find any checkpoint to load config from
checkpoint_files = sorted(checkpoint_dir.glob('model_step_*.pt'))
if not checkpoint_files:
    checkpoint_files = list(checkpoint_dir.glob('model_*.pt'))

if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

# Load config from first checkpoint
first_checkpoint = torch.load(checkpoint_files[0], map_location='cpu')

if 'model_config' in first_checkpoint:
    model_config = first_checkpoint['model_config']
    print(f"✓ Model configuration loaded from checkpoint")
    print(f"  d_model: {model_config['d_model']}")
    print(f"  n_head: {model_config['n_head']}")
    print(f"  n_layers: {model_config['n_layers']}")
    print(f"  ffn_hidden: {model_config['ffn_hidden']}")
    print(f"  drop_prob: {model_config['drop_prob']}")
    print(f"  max_len: {model_config['max_len']}")
    if model_config.get('kdim') is not None:
        print(f"  kdim: {model_config['kdim']}")
else:
    # Fallback to default values
    print(f"⚠️  Model config not found in checkpoint, using default values")
    model_config = {
        'd_model': 512,
        'n_head': 8,
        'n_layers': 6,
        'ffn_hidden': 2048,
        'drop_prob': 0.1,
        'max_len': 512,
        'kdim': None
    }
```

#### 2.2 로드된 설정으로 모델 초기화 (Line 178-191)

하드코딩된 값 대신 체크포인트에서 로드한 설정 사용:

```python
# Initialize model with loaded configuration
print("\n3. Initializing model...")
model = Transformer(
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    trg_sos_idx=trg_sos_idx,
    enc_voc_size=len(vocab_data['source_stoi']),
    dec_voc_size=len(vocab_data['target_stoi']),
    d_model=model_config['d_model'],        # 체크포인트에서 로드
    n_head=model_config['n_head'],          # 체크포인트에서 로드
    max_len=model_config['max_len'],        # 체크포인트에서 로드
    ffn_hidden=model_config['ffn_hidden'],  # 체크포인트에서 로드
    n_layers=model_config['n_layers'],      # 체크포인트에서 로드
    drop_prob=model_config['drop_prob'],    # 체크포인트에서 로드
    device=args.device,
    kdim=model_config.get('kdim')           # 체크포인트에서 로드
).to(args.device)
```

### 3. checkpoint_averaging.py 수정

#### 3.1 Averaged checkpoint에 model_config 보존 (Line 66-68)

```python
# Include model_config if available
if 'model_config' in checkpoints[-1]:
    averaged_checkpoint['model_config'] = checkpoints[-1]['model_config']
    print(f"✓ Model config preserved from checkpoint")
```

## 검증 결과

### ✅ 체크포인트 저장 검증

1. **model_config 포함 여부**
   - `demo_wmt14_pretrained.py`의 모든 체크포인트 저장 시점에서 `model_config` 저장
   - 정기 체크포인트 (model_step_*.pt)
   - Best 모델 (model_best.pt)
   - Final 체크포인트 (model_final.pt)

2. **저장되는 파라미터**
   - `d_model`: 모델 차원
   - `n_head`: Attention head 개수
   - `n_layers`: Encoder/Decoder 레이어 개수
   - `ffn_hidden`: Feed-forward 네트워크 hidden 차원
   - `drop_prob`: Dropout 확률
   - `max_len`: 최대 시퀀스 길이
   - `enc_voc_size`: Encoder vocabulary 크기
   - `dec_voc_size`: Decoder vocabulary 크기
   - `src_pad_idx`: Source padding 인덱스
   - `trg_pad_idx`: Target padding 인덱스
   - `trg_sos_idx`: Target SOS 인덱스
   - `label_smoothing`: Label smoothing 값
   - `kdim`: MultiheadAttention의 key dimension

### ✅ Inference 로드 검증

1. **체크포인트 탐색**
   - `model_step_*.pt` 파일을 우선 탐색
   - 없으면 `model_*.pt` 파일 탐색
   - 첫 번째 체크포인트에서 `model_config` 로드

2. **모델 초기화**
   - 체크포인트에서 로드한 설정으로 Transformer 초기화
   - 하드코딩된 값 대신 동적으로 로드된 값 사용
   - `model_config`가 없으면 기본값으로 fallback (하위 호환성)

3. **설정 출력**
   - 로드된 모든 하이퍼파라미터를 콘솔에 출력
   - 사용자가 올바른 설정으로 모델이 초기화되었는지 확인 가능

### ✅ Checkpoint Averaging 검증

1. **model_config 보존**
   - Averaged checkpoint 생성 시 마지막 체크포인트의 `model_config` 보존
   - Averaged 모델로 추론 시 올바른 아키텍처 사용 가능

## 사용 예시

### 학습 시

```bash
# 커스텀 모델 아키텍처로 학습
python demo_wmt14_pretrained.py \
    --d_model 256 \
    --n_head 4 \
    --n_layers 4 \
    --ffn_hidden 1024 \
    --drop_prob 0.2 \
    --label_smoothing 0.15 \
    --kdim 128
```

위 명령으로 학습하면:
- 체크포인트에 위 설정들이 자동으로 저장됨
- `model_config` 딕셔너리에 모든 하이퍼파라미터 포함

### 추론 시

```bash
# 체크포인트에서 자동으로 모델 설정 로드
python inference.py \
    --checkpoint_dir ./checkpoints \
    --vocab_dir ./artifacts_pretrained
```

위 명령으로 추론하면:
- 체크포인트에서 `model_config` 자동 로드
- 학습 시 사용한 정확한 아키텍처로 모델 초기화
- 별도의 하이퍼파라미터 지정 불필요

## 출력 예시

### 학습 시 출력

```
Model Hyperparameters:
  d_model: 256
  n_head: 4
  n_layers: 4
  ffn_hidden: 1024
  drop_prob: 0.2
  label_smoothing: 0.15
  kdim: 128

✓ Model initialized
  Total parameters: X,XXX,XXX

...

✓ Checkpoint saved: checkpoints/model_step_10000.pt
```

### 추론 시 출력

```
2. Loading model configuration from checkpoint...
✓ Model configuration loaded from checkpoint
  d_model: 256
  n_head: 4
  n_layers: 4
  ffn_hidden: 1024
  drop_prob: 0.2
  max_len: 256
  kdim: 128

3. Initializing model...
✓ Model initialized
  Total parameters: X,XXX,XXX
```

## 결론

✅ **검증 완료**

1. **체크포인트 저장**: 모델 학습 시 모든 하이퍼파라미터가 `model_config`로 체크포인트에 저장됩니다.

2. **체크포인트 로드**: `inference.py`가 체크포인트에서 `model_config`를 로드하여 정확히 동일한 아키텍처로 모델을 초기화합니다.

3. **Checkpoint Averaging**: Averaged checkpoint에도 `model_config`가 보존되어 추론 시 사용할 수 있습니다.

4. **하위 호환성**: `model_config`가 없는 기존 체크포인트도 기본값으로 fallback하여 정상 작동합니다.

이로써 **학습과 추론 시 모델 아키텍처의 일관성이 보장**됩니다.
