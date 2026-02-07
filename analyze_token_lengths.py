"""
WMT14 데이터셋 토큰 길이 분석
- 사전 학습된 토크나이저 적용
- 문장별 토큰 개수 통계
- 최적 max_len 추천
"""

import torch
from transformers import AutoTokenizer
from util.data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

class PretrainedBPETokenizer:
    """사전 학습된 GPT-2 BPE 토크나이저"""
    def __init__(self, model_name="gpt2"):
        print(f"Loading tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        print(f"✓ Loaded (vocab: {self.vocab_size:,})")
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text.lower())

def analyze_token_lengths(dataset, tokenizer_en, tokenizer_de, dataset_name="Dataset", max_samples=None):
    """
    데이터셋의 토큰 길이 분석
    
    Args:
        dataset: WMT14 데이터셋
        tokenizer_en: 영어 토크나이저
        tokenizer_de: 독일어 토크나이저
        dataset_name: 데이터셋 이름 (출력용)
        max_samples: 분석할 최대 샘플 수 (None=전체)
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {dataset_name}")
    print(f"{'='*80}")
    
    en_lengths = []
    de_lengths = []
    total_lengths = []
    
    total_samples = len(dataset)
    samples_to_analyze = min(max_samples, total_samples) if max_samples else total_samples
    
    print(f"Total samples: {total_samples:,}")
    print(f"Analyzing: {samples_to_analyze:,} samples")
    print(f"Processing...", end='', flush=True)
    
    start_time = time.time()
    
    for idx, example in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
        # --- 수정된 부분 ---
        # WMT14 데이터셋 구조에 맞게 수정
        if 'translation' in example:
            en_text = example['translation']['en']
            de_text = example['translation']['de']
        else:
            # 혹시 모를 예외 처리 (기존 로직 유지)
            print(example)
            en_text = example.src if hasattr(example, 'src') else str(example)
            de_text = example.trg if hasattr(example, 'trg') else str(example)
            # 진행 상황 표시
        if (idx + 1) % 10000 == 0:
            progress = (idx + 1) / samples_to_analyze * 100
            elapsed = time.time() - start_time
            eta = elapsed / (idx + 1) * (samples_to_analyze - idx - 1)
            print(f"\rProcessing... {idx+1:,}/{samples_to_analyze:,} ({progress:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end='', flush=True)
        
        # 영어 문장 토큰화
        en_tokens = tokenizer_en.tokenize(en_text)
        en_length = len(en_tokens)
        en_lengths.append(en_length)
        
        # 독일어 문장 토큰화
        de_tokens = tokenizer_de.tokenize(de_text)
        de_length = len(de_tokens)
        de_lengths.append(de_length)
        
        # 합계 (source + target)
        total_length = en_length + de_length
        total_lengths.append(total_length)
    
    elapsed = time.time() - start_time
    print(f"\rProcessing... {samples_to_analyze:,}/{samples_to_analyze:,} (100.0%) | "
          f"Done! ({elapsed:.1f}s)                    ")
    
    return en_lengths, de_lengths, total_lengths

def print_statistics(lengths, language="Language"):
    """통계 출력"""
    lengths_array = np.array(lengths)
    
    print(f"\n{language} Statistics:")
    print(f"  Total sentences: {len(lengths):,}")
    print(f"  Mean length: {np.mean(lengths_array):.2f} tokens")
    print(f"  Median length: {np.median(lengths_array):.0f} tokens")
    print(f"  Std dev: {np.std(lengths_array):.2f} tokens")
    print(f"  Min length: {np.min(lengths_array)} tokens")
    print(f"  Max length: {np.max(lengths_array)} tokens")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
    print(f"\n  Percentiles:")
    for p in percentiles:
        value = np.percentile(lengths_array, p)
        coverage = (lengths_array <= value).sum() / len(lengths_array) * 100
        print(f"    {p:5.1f}%: {value:6.0f} tokens (covers {coverage:5.1f}% of data)")

def recommend_max_len(en_lengths, de_lengths, total_lengths):
    """최적 max_len 추천"""
    print(f"\n{'='*80}")
    print(f"max_len Recommendations")
    print(f"{'='*80}")
    
    en_array = np.array(en_lengths)
    de_array = np.array(de_lengths)
    total_array = np.array(total_lengths)
    
    # 여러 max_len 후보에 대한 커버리지 계산
    candidates = [64, 128, 256, 384, 512, 768, 1024]
    
    print(f"\nCoverage Analysis:")
    print(f"{'max_len':>10} | {'EN Coverage':>12} | {'DE Coverage':>12} | {'Total Coverage':>15} | {'Truncated':>10}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}-+-{'-'*10}")
    
    for max_len in candidates:
        en_coverage = (en_array <= max_len).sum() / len(en_array) * 100
        de_coverage = (de_array <= max_len).sum() / len(de_array) * 100
        total_coverage = (total_array <= max_len * 2).sum() / len(total_array) * 100
        truncated = 100 - min(en_coverage, de_coverage)
        
        print(f"{max_len:10} | {en_coverage:11.2f}% | {de_coverage:11.2f}% | {total_coverage:14.2f}% | {truncated:9.2f}%")
    
    # 추천
    print(f"\n{'='*80}")
    print(f"Recommendations:")
    print(f"{'='*80}")
    
    # 99% 커버리지
    en_99 = np.percentile(en_array, 99)
    de_99 = np.percentile(de_array, 99)
    max_99 = max(en_99, de_99)
    
    # 95% 커버리지
    en_95 = np.percentile(en_array, 95)
    de_95 = np.percentile(de_array, 95)
    max_95 = max(en_95, de_95)
    
    # 90% 커버리지
    en_90 = np.percentile(en_array, 90)
    de_90 = np.percentile(de_array, 90)
    max_90 = max(en_90, de_90)
    
    print(f"\n1. 메모리 최적화 (90% coverage):")
    print(f"   max_len = {int(np.ceil(max_90 / 64) * 64)} (nearest 64)")
    print(f"   - Covers ~90% of sentences")
    print(f"   - Truncates ~10% of data")
    print(f"   - 메모리 절약: 높음")
    
    print(f"\n2. 균형 설정 (95% coverage):")
    print(f"   max_len = {int(np.ceil(max_95 / 64) * 64)} (nearest 64)")
    print(f"   - Covers ~95% of sentences")
    print(f"   - Truncates ~5% of data")
    print(f"   - 메모리 절약: 중간")
    print(f"   - **권장 설정** ⭐")
    
    print(f"\n3. 안전 설정 (99% coverage):")
    print(f"   max_len = {int(np.ceil(max_99 / 64) * 64)} (nearest 64)")
    print(f"   - Covers ~99% of sentences")
    print(f"   - Truncates ~1% of data")
    print(f"   - 메모리 절약: 낮음")
    
    print(f"\n4. 현재 설정 (논문):")
    print(f"   max_len = 512")
    current_en = (en_array <= 512).sum() / len(en_array) * 100
    current_de = (de_array <= 512).sum() / len(de_array) * 100
    print(f"   - EN coverage: {current_en:.2f}%")
    print(f"   - DE coverage: {current_de:.2f}%")
    
    # 메모리 절약 계산
    current_mem = 512
    recommended_mem = int(np.ceil(max_95 / 64) * 64)
    memory_saving = (1 - (recommended_mem / current_mem) ** 2) * 100
    
    print(f"\n💡 메모리 절약 예상:")
    print(f"   현재 (512): Attention matrix = 512 × 512 = 262,144")
    print(f"   권장 ({recommended_mem}): Attention matrix = {recommended_mem} × {recommended_mem} = {recommended_mem**2:,}")
    print(f"   절약: ~{memory_saving:.1f}%")

def plot_distribution(en_lengths, de_lengths, total_lengths, save_path='token_length_distribution.png'):
    """토큰 길이 분포 시각화"""
    print(f"\n{'='*80}")
    print(f"Generating distribution plot...")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 영어 분포
    axes[0, 0].hist(en_lengths, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(en_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(en_lengths):.1f}')
    axes[0, 0].axvline(np.percentile(en_lengths, 95), color='orange', linestyle='--', linewidth=2, label=f'95%: {np.percentile(en_lengths, 95):.0f}')
    axes[0, 0].set_xlabel('Token Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('English Token Length Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 독일어 분포
    axes[0, 1].hist(de_lengths, bins=100, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(np.mean(de_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(de_lengths):.1f}')
    axes[0, 1].axvline(np.percentile(de_lengths, 95), color='orange', linestyle='--', linewidth=2, label=f'95%: {np.percentile(de_lengths, 95):.0f}')
    axes[0, 1].set_xlabel('Token Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('German Token Length Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 합계 분포
    axes[1, 0].hist(total_lengths, bins=100, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 0].axvline(np.mean(total_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_lengths):.1f}')
    axes[1, 0].axvline(np.percentile(total_lengths, 95), color='orange', linestyle='--', linewidth=2, label=f'95%: {np.percentile(total_lengths, 95):.0f}')
    axes[1, 0].set_xlabel('Token Length (EN + DE)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Total Token Length Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. CDF (Cumulative Distribution)
    sorted_total = np.sort(total_lengths)
    cdf = np.arange(1, len(sorted_total) + 1) / len(sorted_total) * 100
    axes[1, 1].plot(sorted_total, cdf, linewidth=2)
    axes[1, 1].axhline(95, color='orange', linestyle='--', linewidth=2, label='95% coverage')
    axes[1, 1].axhline(99, color='red', linestyle='--', linewidth=2, label='99% coverage')
    axes[1, 1].axvline(512, color='green', linestyle='--', linewidth=2, label='max_len=512')
    axes[1, 1].set_xlabel('Token Length (EN + DE)')
    axes[1, 1].set_ylabel('Cumulative Percentage (%)')
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, min(1000, max(total_lengths)))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {save_path}")
    
    # 콘솔에서도 볼 수 있도록 간단한 ASCII 히스토그램
    print(f"\nASCII Distribution (Total Length):")
    bins = 20
    hist, bin_edges = np.histogram(total_lengths, bins=bins)
    max_count = max(hist)
    
    for i in range(bins):
        bar_length = int((hist[i] / max_count) * 50)
        bar = '█' * bar_length
        print(f"  {bin_edges[i]:6.0f}-{bin_edges[i+1]:6.0f}: {bar} ({hist[i]:,})")

def main():
    print("="*80)
    print(" "*20 + "WMT14 Token Length Analysis")
    print("="*80)
    
    # ------------------------------------------------------------------------
    # 1. 토크나이저 로드
    # ------------------------------------------------------------------------
    print("\n1. Loading tokenizers...")
    tokenizer_en = PretrainedBPETokenizer(model_name="gpt2")
    tokenizer_de = PretrainedBPETokenizer(model_name="gpt2")
    
    # ------------------------------------------------------------------------
    # 2. 데이터셋 로드
    # ------------------------------------------------------------------------
    print("\n2. Loading WMT14 dataset...")
    loader = DataLoader(
        ext=('.en', '.de'),
        tokenize_en=None,
        tokenize_de=None,
        init_token='<sos>',
        eos_token='<eos>'
    )
    
    train, valid, test = loader.make_dataset(dataset_name='wmt14')
    print(f"✓ Dataset loaded")
    print(f"  Train: {len(train):,}")
    print(f"  Valid: {len(valid):,}")
    print(f"  Test: {len(test):,}")
    
    # ------------------------------------------------------------------------
    # 3. Train set 분석 (샘플링)
    # ------------------------------------------------------------------------
    # 전체 분석은 시간이 오래 걸리므로 샘플링
    sample_size = 100000  # 10만 샘플 분석 (전체의 약 2%)
    
    en_lengths, de_lengths, total_lengths = analyze_token_lengths(
        train, 
        tokenizer_en, 
        tokenizer_de, 
        dataset_name="Training Set (Sampled)",
        max_samples=sample_size
    )
    
    # ------------------------------------------------------------------------
    # 4. 통계 출력
    # ------------------------------------------------------------------------
    print_statistics(en_lengths, "English")
    print_statistics(de_lengths, "German")
    print_statistics(total_lengths, "Total (EN + DE)")
    
    # ------------------------------------------------------------------------
    # 5. max_len 추천
    # ------------------------------------------------------------------------
    recommend_max_len(en_lengths, de_lengths, total_lengths)
    
    # ------------------------------------------------------------------------
    # 6. 분포 시각화
    # ------------------------------------------------------------------------
    try:
        plot_distribution(en_lengths, de_lengths, total_lengths)
    except Exception as e:
        print(f"\n⚠️  Could not generate plot: {e}")
        print(f"   (matplotlib may not be available in headless environment)")
    
    # ------------------------------------------------------------------------
    # 7. 요약
    # ------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"\n📊 Quick Summary:")
    print(f"  Analyzed: {len(en_lengths):,} sentence pairs")
    print(f"  EN mean: {np.mean(en_lengths):.1f} tokens")
    print(f"  DE mean: {np.mean(de_lengths):.1f} tokens")
    print(f"  Total mean: {np.mean(total_lengths):.1f} tokens")
    print(f"  Current max_len (512) covers: {(np.array(en_lengths) <= 512).sum() / len(en_lengths) * 100:.1f}% (EN), "
          f"{(np.array(de_lengths) <= 512).sum() / len(de_lengths) * 100:.1f}% (DE)")
    
    recommended = int(np.ceil(max(np.percentile(en_lengths, 95), np.percentile(de_lengths, 95)) / 64) * 64)
    print(f"\n💡 Recommended max_len: {recommended} (95% coverage)")
    print(f"   Use: python demo_wmt14_step_based.py --max_len {recommended}")

if __name__ == '__main__':
    main()
