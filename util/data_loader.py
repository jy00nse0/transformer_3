"""
@author : Modified for HuggingFace tokenizers with Paper-compliant features
@when : 2026-01-29
@description : DataLoader following "Attention Is All You Need" specifications
"""
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import defaultdict
import numpy as np


class TranslationDataset(Dataset):
    """Custom Dataset for translation tasks"""
    
    def __init__(self, hf_dataset, source_lang, target_lang, source_tokenizer, target_tokenizer,
                 source_vocab, target_vocab, max_len=512):
        """
        Args:
            hf_dataset: HuggingFace dataset split (train/valid/test)
            source_lang: source language key (e.g., 'de')
            target_lang: target language key (e.g., 'en')
            source_tokenizer: tokenizer for source language
            target_tokenizer: tokenizer for target language
            source_vocab: source vocabulary dict {token: idx}
            target_vocab: target vocabulary dict {token: idx}
            max_len: maximum sequence length
        """
        self.dataset = hf_dataset
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len
        
        # Pre-compute lengths for bucketing
        self.lengths = []
        for idx in range(len(hf_dataset)):
            item = hf_dataset[idx]
            src_text = item['translation'][self.source_lang]
            trg_text = item['translation'][self.target_lang]
            src_tokens = len(self.source_tokenizer.tokenize(src_text.lower()))
            trg_tokens = len(self.target_tokenizer.tokenize(trg_text.lower()))
            self.lengths.append(max(src_tokens, trg_tokens))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item['translation'][self.source_lang]
        trg_text = item['translation'][self.target_lang]
        
        # Tokenize
        src_tokens = self.source_tokenizer.tokenize(src_text.lower())
        trg_tokens = self.target_tokenizer.tokenize(trg_text.lower())
        
        # Convert to indices
        src_indices = [self.source_vocab.get(token, self.source_vocab['<unk>']) 
                       for token in src_tokens]
        trg_indices = [self.target_vocab.get(token, self.target_vocab['<unk>']) 
                       for token in trg_tokens]
        
        # Add special tokens
        src_indices = [self.source_vocab['<sos>']] + src_indices + [self.source_vocab['<eos>']]
        trg_indices = [self.target_vocab['<sos>']] + trg_indices + [self.target_vocab['<eos>']]
        
        # Truncate if needed
        src_indices = src_indices[:self.max_len]
        trg_indices = trg_indices[:self.max_len]
        
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)


class TokenBucketSampler(Sampler):
    """
    Sampler that groups samples by length and creates batches with approximately
    equal number of TOKENS (not sentences), following the paper's specification.
    
    Paper: "Sentence pairs were batched together by approximate sequence length.
           Each training batch contained approximately 25000 source tokens and 25000 target tokens."
    """
    
    def __init__(self, lengths, max_tokens=25000, shuffle=True, drop_last=False):
        """
        Args:
            lengths: list of sequence lengths (max of src and trg)
            max_tokens: maximum number of tokens per batch
            shuffle: whether to shuffle batches
            drop_last: whether to drop last incomplete batch
        """
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Sort indices by length
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        
        # Create buckets
        self.batches = self._create_batches()
    
    def _create_batches(self):
        """Create batches based on token count"""
        batches = []
        current_batch = []
        current_tokens = 0
        max_len_in_batch = 0
        
        for idx in self.sorted_indices:
            length = self.lengths[idx]
            
            # Calculate tokens if we add this sample
            # Each sample will be padded to max_len_in_batch
            new_max_len = max(max_len_in_batch, length)
            new_tokens = new_max_len * (len(current_batch) + 1)
            
            if new_tokens > self.max_tokens and len(current_batch) > 0:
                # Start new batch
                batches.append(current_batch)
                current_batch = [idx]
                current_tokens = length
                max_len_in_batch = length
            else:
                # Add to current batch
                current_batch.append(idx)
                current_tokens = new_tokens
                max_len_in_batch = new_max_len
        
        # Add last batch
        if len(current_batch) > 0 and not self.drop_last:
            batches.append(current_batch)
        
        return batches
    
    def __iter__(self):
        if self.shuffle:
            # Shuffle batches but keep samples within batch together
            indices = torch.randperm(len(self.batches)).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)


class Batch:
    """Batch object to match torchtext's Batch interface"""
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg


class DataLoader:
    """
    DataLoader following "Attention Is All You Need" paper specifications
    
    Key features matching paper:
    - Token-based batching (~25k tokens per batch)
    - Length-based bucketing for efficiency
    - Vocabulary size limit (37k for shared EN-DE)
    - Support for shared vocabulary (optional)
    """
    
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        """
        Args:
            ext: tuple of file extensions (e.g., ('.en', '.de'))
            tokenize_en: English tokenizer function
            tokenize_de: German tokenizer function
            init_token: start of sequence token (e.g., '<sos>')
            eos_token: end of sequence token (e.g., '<eos>')
        """
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        
        # Create vocab objects to match torchtext interface
        self.source = type('obj', (object,), {'vocab': None})()
        self.target = type('obj', (object,), {'vocab': None})()
        
        print('Dataset initializing start')
        
    def make_dataset(self, dataset_name='bentrevett/multi30k'):
        """
        Load dataset from HuggingFace
        
        Args:
            dataset_name: dataset name on HuggingFace Hub
                         'bentrevett/multi30k' for Multi30k
                         'wmt14' for WMT 2014 (paper's dataset)
        
        Returns:
            train_data, valid_data, test_data
        """
        # Load dataset
        if dataset_name == 'wmt14':
            # WMT 2014 dataset (paper's dataset)
            dataset = load_dataset('wmt14', 'de-en')
            # Note: WMT14 uses 'translation' with 'de' and 'en' keys
        else:
            # Multi30k (smaller, faster for testing)
            dataset = load_dataset(dataset_name)
        
        train_data = dataset['train']
        valid_data = dataset['validation']
        test_data = dataset['test']
        
        # Determine source and target languages based on ext
        if self.ext == ('.de', '.en'):
            self.source_lang = 'de'
            self.target_lang = 'en'
            self.source_tokenizer = self.tokenize_de
            self.target_tokenizer = self.tokenize_en
        elif self.ext == ('.en', '.de'):
            self.source_lang = 'en'
            self.target_lang = 'de'
            self.source_tokenizer = self.tokenize_en
            self.target_tokenizer = self.tokenize_de
        else:
            raise ValueError(f"Unsupported extension pair: {self.ext}")
        
        print(f'Loaded {len(train_data)} training samples')
        print(f'Loaded {len(valid_data)} validation samples')
        print(f'Loaded {len(test_data)} test samples')
        
        return train_data, valid_data, test_data
    
    def build_vocab(self, train_data, min_freq=2, max_vocab_size=None, shared_vocab=False):
        """
        Build vocabulary from training data
        
        Args:
            train_data: training dataset
            min_freq: minimum frequency for a token to be included
            max_vocab_size: maximum vocabulary size (e.g., 37000 for paper's EN-DE)
                          None for unlimited
            shared_vocab: if True, build shared source-target vocabulary (paper's approach)
        
        Paper specifications:
        - EN-DE: shared vocabulary of ~37,000 tokens
        - EN-FR: 32,000 word-piece tokens
        """
        if shared_vocab:
            # Build shared vocabulary (paper's approach for EN-DE)
            combined_counter = defaultdict(int)
            
            for item in train_data:
                src_text = item['translation'][self.source_lang].lower()
                trg_text = item['translation'][self.target_lang].lower()
                
                src_tokens = self.source_tokenizer.tokenize(src_text)
                trg_tokens = self.target_tokenizer.tokenize(trg_text)
                
                for token in src_tokens + trg_tokens:
                    combined_counter[token] += 1
            
            # Build shared vocabulary
            special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
            vocab = {token: idx for idx, token in enumerate(special_tokens)}
            
            # Sort by frequency and add tokens
            sorted_tokens = sorted(combined_counter.items(), key=lambda x: x[1], reverse=True)
            
            for token, freq in sorted_tokens:
                if freq >= min_freq and token not in vocab:
                    if max_vocab_size is None or len(vocab) < max_vocab_size:
                        vocab[token] = len(vocab)
                    else:
                        break
            
            # Use same vocab for both source and target
            itos = {idx: token for token, idx in vocab.items()}
            
            self.source.vocab = type('obj', (object,), {
                'stoi': vocab,
                'itos': itos,
                '__len__': lambda self: len(vocab)
            })()
            
            self.target.vocab = self.source.vocab  # Share vocabulary
            
            print(f'Built shared vocabulary size: {len(vocab)}')
            
        else:
            # Build separate vocabularies
            source_counter = defaultdict(int)
            target_counter = defaultdict(int)
            
            for item in train_data:
                src_text = item['translation'][self.source_lang].lower()
                trg_text = item['translation'][self.target_lang].lower()
                
                src_tokens = self.source_tokenizer.tokenize(src_text)
                trg_tokens = self.target_tokenizer.tokenize(trg_text)
                
                for token in src_tokens:
                    source_counter[token] += 1
                for token in trg_tokens:
                    target_counter[token] += 1
            
            # Build vocabularies with special tokens
            special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
            
            # Source vocabulary
            source_vocab = {token: idx for idx, token in enumerate(special_tokens)}
            sorted_src = sorted(source_counter.items(), key=lambda x: x[1], reverse=True)
            
            for token, freq in sorted_src:
                if freq >= min_freq and token not in source_vocab:
                    if max_vocab_size is None or len(source_vocab) < max_vocab_size:
                        source_vocab[token] = len(source_vocab)
                    else:
                        break
            
            # Target vocabulary
            target_vocab = {token: idx for idx, token in enumerate(special_tokens)}
            sorted_trg = sorted(target_counter.items(), key=lambda x: x[1], reverse=True)
            
            for token, freq in sorted_trg:
                if freq >= min_freq and token not in target_vocab:
                    if max_vocab_size is None or len(target_vocab) < max_vocab_size:
                        target_vocab[token] = len(target_vocab)
                    else:
                        break
            
            # Create reverse mappings
            source_itos = {idx: token for token, idx in source_vocab.items()}
            target_itos = {idx: token for token, idx in target_vocab.items()}
            
            # Create vocab objects
            self.source.vocab = type('obj', (object,), {
                'stoi': source_vocab,
                'itos': source_itos,
                '__len__': lambda self: len(source_vocab)
            })()
            
            self.target.vocab = type('obj', (object,), {
                'stoi': target_vocab,
                'itos': target_itos,
                '__len__': lambda self: len(target_vocab)
            })()
            
            print(f'Source vocabulary size: {len(source_vocab)}')
            print(f'Target vocabulary size: {len(target_vocab)}')
    
    def make_iter(self, train, validate, test, batch_size=None, max_tokens=25000, 
                  device='cpu', num_workers=0, max_len=256):
        """
        Create data iterators with paper-compliant token-based batching
        
        Args:
            train, validate, test: dataset splits
            batch_size: DEPRECATED - use max_tokens instead for paper compliance
            max_tokens: maximum number of tokens per batch (paper: ~25,000)
            device: torch device
            num_workers: number of worker processes for data loading
            max_len: maximum sequence length (must match model's max_len)
        
        Returns:
            train_iterator, valid_iterator, test_iterator
        
        Paper specification:
        "Sentence pairs were batched together by approximate sequence length.
         Each training batch contained approximately 25000 source tokens and 25000 target tokens."
        """
        # Create custom datasets
        train_dataset = TranslationDataset(
            train, self.source_lang, self.target_lang,
            self.source_tokenizer, self.target_tokenizer,
            self.source.vocab.stoi, self.target.vocab.stoi,
            max_len=max_len
        )
        
        valid_dataset = TranslationDataset(
            validate, self.source_lang, self.target_lang,
            self.source_tokenizer, self.target_tokenizer,
            self.source.vocab.stoi, self.target.vocab.stoi,
            max_len=max_len
        )
        
        test_dataset = TranslationDataset(
            test, self.source_lang, self.target_lang,
            self.source_tokenizer, self.target_tokenizer,
            self.source.vocab.stoi, self.target.vocab.stoi,
            max_len=max_len
        )
        
        # Collate function for padding
        def collate_fn(batch):
            src_batch, trg_batch = zip(*batch)
            
            # Pad sequences
            src_padded = pad_sequence(src_batch, batch_first=True, 
                                     padding_value=self.source.vocab.stoi['<pad>'])
            trg_padded = pad_sequence(trg_batch, batch_first=True,
                                     padding_value=self.target.vocab.stoi['<pad>'])
            
            # Move to device
            #src_padded = src_padded.to(device)
            #trg_padded = trg_padded.to(device)
            
            return Batch(src_padded, trg_padded)
        
        # Create token-based bucket samplers (paper-compliant)
        train_sampler = TokenBucketSampler(
            train_dataset.lengths,
            max_tokens=max_tokens,
            shuffle=True,
            drop_last=True
        )
        
        valid_sampler = TokenBucketSampler(
            valid_dataset.lengths,
            max_tokens=max_tokens,
            shuffle=False,
            drop_last=False
        )
        
        test_sampler = TokenBucketSampler(
            test_dataset.lengths,
            max_tokens=max_tokens,
            shuffle=False,
            drop_last=False
        )
        
        # Create DataLoaders with batch_sampler
        train_iterator = TorchDataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(device != 'cpu')
        )
        
        valid_iterator = TorchDataLoader(
            valid_dataset,
            batch_sampler=valid_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(device != 'cpu')
        )
        
        test_iterator = TorchDataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(device != 'cpu')
        )
        
        print('Dataset initializing done')
        print(f'Train batches: {len(train_iterator)}')
        print(f'Valid batches: {len(valid_iterator)}')
        print(f'Test batches: {len(test_iterator)}')
        
        return train_iterator, valid_iterator, test_iterator
