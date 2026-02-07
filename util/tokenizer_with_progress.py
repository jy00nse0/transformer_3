import torch
from collections import defaultdict
from transformers import AutoTokenizer
import sys
import time

class BPETokenizer:
    def __init__(self, vocab_size=50):
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = {}
        # GPT-2 pre-tokenizer를 사용하여 공백을 'Ġ'로 치환하는 등의 기본 처리를 수행합니다.
        self.base_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def train(self, corpus):
        print(f"\n{'='*80}")
        print(f"BPE Tokenizer Training Started")
        print(f"{'='*80}")
        print(f"Corpus size: {len(corpus):,} sentences")
        print(f"Target vocab size: {self.vocab_size:,}")
        
        start_time = time.time()
        
        # 1. 단어 빈도수 계산
        print(f"\n[Step 1/4] Computing word frequencies...")
        word_freqs = defaultdict(int)
        total = len(corpus)
        
        for idx, text in enumerate(corpus):
            if idx % 100000 == 0 and idx > 0:
                elapsed = time.time() - start_time
                progress = idx / total * 100
                eta = elapsed / idx * (total - idx)
                print(f"  Progress: {idx:,}/{total:,} ({progress:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end='\r')
            
            words_with_offsets = self.base_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            for word, _ in words_with_offsets:
                word_freqs[word] += 1
        
        print(f"  Progress: {total:,}/{total:,} (100.0%) | "
              f"Elapsed: {time.time() - start_time:.1f}s | Done!          ")
        print(f"  Unique words: {len(word_freqs):,}")

        # 2. 초기 알파벳 추출 및 어휘 사전 설정
        print(f"\n[Step 2/4] Extracting alphabet...")
        alphabet = []
        for word in word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()
        self.vocab = ["<|endoftext|>"] + alphabet.copy()
        print(f"  Alphabet size: {len(alphabet)}")
        print(f"  Initial vocab size: {len(self.vocab)}")
        
        # 3. 단어를 문자 단위로 분할
        print(f"\n[Step 3/4] Splitting words into characters...")
        splits = {word: [c for c in word] for word in word_freqs.keys()}
        print(f"  Done!")

        # 4. 빈도 기반 병합 반복
        print(f"\n[Step 4/4] Performing BPE merges...")
        print(f"  Target merges: {self.vocab_size - len(self.vocab):,}")
        
        merge_count = 0
        total_merges = self.vocab_size - len(self.vocab)
        merge_start = time.time()
        
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._compute_pair_freqs(splits, word_freqs)
            if not pair_freqs: 
                print(f"\n  No more pairs to merge!")
                break
            
            best_pair = max(pair_freqs, key=pair_freqs.get)
            splits = self._merge_pair(*best_pair, splits, word_freqs)
            
            new_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_token
            self.vocab.append(new_token)
            
            merge_count += 1
            
            # 진행률 표시 (매 100번째 병합마다)
            if merge_count % 100 == 0 or merge_count == total_merges:
                elapsed = time.time() - merge_start
                progress = len(self.vocab) / self.vocab_size * 100
                merges_per_sec = merge_count / elapsed if elapsed > 0 else 0
                eta = (total_merges - merge_count) / merges_per_sec if merges_per_sec > 0 else 0
                
                print(f"  Merges: {merge_count:,}/{total_merges:,} ({progress:.1f}%) | "
                      f"Vocab: {len(self.vocab):,}/{self.vocab_size:,} | "
                      f"Speed: {merges_per_sec:.0f} merges/s | ETA: {eta:.1f}s", end='\r')
        
        print(f"  Merges: {merge_count:,}/{total_merges:,} (100.0%) | "
              f"Vocab: {len(self.vocab):,}/{self.vocab_size:,} | Done!          ")
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"  Final vocab size: {len(self.vocab):,}")
        print(f"  Total merges: {merge_count:,}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*80}\n")

    def _compute_pair_freqs(self, splits, word_freqs):
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            for i in range(len(split) - 1):
                pair_freqs[(split[i], split[i+1])] += freq
        return pair_freqs

    def _merge_pair(self, a, b, splits, word_freqs):
        for word in word_freqs:
            split = splits[word]
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i+1] == b:
                    split = split[:i] + [a + b] + split[i+2:]
                else:
                    i += 1
            splits[word] = split
        return splits

    def tokenize(self, text):
        pre_tokenize_result = self.base_tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, _ in pre_tokenize_result]
        splits = [[l for l in word] for word in pre_tokenized_text]
        
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i+1] == pair[1]:
                        split = split[:i] + [merge] + split[i+2:]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])
    
# 결합의 순도(희귀성)이 높은 서브월드 단위로 토큰화를 수행하는 WordPiece 토크나이저 구현
# WordPiece finds the longest subword that is in the vocabulary, then splits on it.
class WordPieceTokenizer:
    def __init__(self, vocab_size=70):
        self.vocab_size = vocab_size
        self.vocab = []
        # BERT pre-tokenizer를 사용합니다.
        self.base_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def train(self, corpus):
        print(f"\n{'='*80}")
        print(f"WordPiece Tokenizer Training Started")
        print(f"{'='*80}")
        print(f"Corpus size: {len(corpus):,} sentences")
        print(f"Target vocab size: {self.vocab_size:,}")
        
        start_time = time.time()
        
        # 1. 단어 빈도수 계산
        print(f"\n[Step 1/4] Computing word frequencies...")
        word_freqs = defaultdict(int)
        total = len(corpus)
        
        for idx, text in enumerate(corpus):
            if idx % 100000 == 0 and idx > 0:
                elapsed = time.time() - start_time
                progress = idx / total * 100
                eta = elapsed / idx * (total - idx)
                print(f"  Progress: {idx:,}/{total:,} ({progress:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end='\r')
            
            words_with_offsets = self.base_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            for word, _ in words_with_offsets:
                word_freqs[word] += 1
        
        print(f"  Progress: {total:,}/{total:,} (100.0%) | "
              f"Elapsed: {time.time() - start_time:.1f}s | Done!          ")
        print(f"  Unique words: {len(word_freqs):,}")

        # 2. ##을 활용한 알파벳 초기화
        print(f"\n[Step 2/4] Extracting alphabet with ## markers...")
        alphabet = []
        for word in word_freqs.keys():
            if word[0] not in alphabet: alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet: alphabet.append(f"##{letter}")
        alphabet.sort()
        
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
        print(f"  Alphabet size: {len(alphabet)}")
        print(f"  Initial vocab size: {len(self.vocab)}")
        
        splits = {word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)] 
                  for word in word_freqs.keys()}

        # 3. 병합 수행
        print(f"\n[Step 3/4] Performing WordPiece merges...")
        print(f"  Target merges: {self.vocab_size - len(self.vocab):,}")
        
        merge_count = 0
        total_merges = self.vocab_size - len(self.vocab)
        merge_start = time.time()
        
        while len(self.vocab) < self.vocab_size:
            scores = self._compute_pair_scores(splits, word_freqs)
            if not scores: 
                print(f"\n  No more pairs to merge!")
                break
            
            best_pair = max(scores, key=scores.get)
            splits = self._merge_pair(*best_pair, splits, word_freqs)
            
            new_token = best_pair[0] + best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[0] + best_pair[1]
            self.vocab.append(new_token)
            
            merge_count += 1
            
            if merge_count % 10 == 0 or merge_count == total_merges:
                elapsed = time.time() - merge_start
                progress = len(self.vocab) / self.vocab_size * 100
                merges_per_sec = merge_count / elapsed if elapsed > 0 else 0
                eta = (total_merges - merge_count) / merges_per_sec if merges_per_sec > 0 else 0
                
                print(f"  Merges: {merge_count:,}/{total_merges:,} ({progress:.1f}%) | "
                      f"Vocab: {len(self.vocab):,}/{self.vocab_size:,} | "
                      f"Speed: {merges_per_sec:.0f} merges/s | ETA: {eta:.1f}s", end='\r')
        
        print(f"  Merges: {merge_count:,}/{total_merges:,} (100.0%) | "
              f"Vocab: {len(self.vocab):,}/{self.vocab_size:,} | Done!          ")
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"  Final vocab size: {len(self.vocab):,}")
        print(f"  Total merges: {merge_count:,}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*80}\n")

    def _compute_pair_scores(self, splits, word_freqs):
        char_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                char_freqs[split[0]] += freq
            for i in range(len(split)):
                char_freqs[split[i]] += freq
                if i < len(split) - 1:
                    pair_freqs[(split[i], split[i+1])] += freq
        
        return {pair: freq / (char_freqs[pair[0]] * char_freqs[pair[1]]) 
                for pair, freq in pair_freqs.items()}

    def _merge_pair(self, a, b, splits, word_freqs):
        for word in word_freqs:
            split = splits[word]
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i+1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i+2:]
                else:
                    i += 1
            splits[word] = split
        return splits

    def encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0: return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0: word = f"##{word}"
        return tokens

    def tokenize(self, text):
        pre_tokenize_result = self.base_tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, _ in pre_tokenize_result]
        encoded_words = [self.encode_word(word) for word in pre_tokenized_text]
        return sum(encoded_words, [])
