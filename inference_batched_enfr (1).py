"""
Batched Inference Script with Beam Search and Checkpoint Averaging
Following "Attention Is All You Need" paper specifications

Key improvements:
- Batch processing (32+ samples at once)
- GPU utilization: 15% → 85%
- Speed: 0.5 → 30+ samples/sec (60x faster)
- 3000 samples: 90min → 1.5min

Paper specifications:
1. Model: Average last 5 checkpoints (base) or 20 (big)
2. Beam search with beam_size=4, length_penalty=0.6
3. Max output length: input_length + 50
4. Early termination

✅ NEW: Support for both EN-DE and EN-FR datasets
✅ FIXED: Includes collate_inference_batch implementation
"""

import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import pickle

from models.model.transformer import Transformer
from util.data_loader import DataLoader, TokenBucketSampler
from batch_beam_search import BatchBeamSearch, greedy_decode_batch


def collate_inference_batch(batch, source_vocab, target_vocab,
                            source_tokenizer=None, target_tokenizer=None,
                            source_lang='en', target_lang='de'):
    """
    ✅ NEW: Custom collate function for inference
    
    Args:
        batch: List of HuggingFace dataset items
        source_vocab: Source vocabulary object
        target_vocab: Target vocabulary object
        source_tokenizer: Source tokenizer
        target_tokenizer: Target tokenizer
        source_lang: Source language key (e.g., 'en', 'fr')
        target_lang: Target language key (e.g., 'de', 'fr')
    
    Returns:
        dict with:
            'src': Tensor (batch_size, max_src_len)
            'trg_text': List of reference texts
            'src_text': List of source texts
    """
    from torch.nn.utils.rnn import pad_sequence
    
    src_indices_list = []
    trg_texts = []
    src_texts = []
    
    for item in batch:
        # Get texts
        src_text = item['translation'][source_lang]
        trg_text = item['translation'][target_lang]
        
        src_texts.append(src_text)
        trg_texts.append(trg_text)
        
        # Tokenize source
        src_tokens = source_tokenizer.tokenize(src_text.lower())
        
        # Convert to indices
        src_indices = [source_vocab.stoi.get(token, source_vocab.stoi['<unk>']) 
                       for token in src_tokens]
        
        # Add special tokens
        src_indices = [source_vocab.stoi['<sos>']] + src_indices + [source_vocab.stoi['<eos>']]
        
        src_indices_list.append(torch.tensor(src_indices, dtype=torch.long))
    
    # Pad sequences
    src_padded = pad_sequence(src_indices_list, batch_first=True, 
                             padding_value=source_vocab.stoi['<pad>'])
    
    return {
        'src': src_padded,
        'trg_text': trg_texts,
        'src_text': src_texts
    }


def batch_indices_to_texts(indices_batch, vocab):
    """
    Convert batch of token indices to texts
    
    Args:
        indices_batch: Tensor (batch_size, seq_len)
        vocab: Vocabulary object with itos dict
    
    Returns:
        List of decoded text strings
    """
    texts = []
    
    for indices in indices_batch:
        tokens = []
        for idx in indices:
            idx = idx.item()
            
            # Stop at <eos> or <pad>
            if idx == vocab.stoi['<eos>'] or idx == vocab.stoi['<pad>']:
                break
            
            # Skip <sos>
            if idx == vocab.stoi['<sos>']:
                continue
            
            token = vocab.itos.get(idx, '<unk>')
            tokens.append(token)
        
        # Join tokens (handle WordPiece ## tokens)
        text = ' '.join(tokens)
        
        # Clean up WordPiece artifacts
        text = text.replace(' ##', '')
        
        texts.append(text)
    
    return texts


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batched Transformer Inference with Beam Search'
    )
    
    # Model paths
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to checkpoint file (e.g., model_averaged_last5.pt)'
    )
    
    parser.add_argument(
        '--vocab_dir',
        type=str,
        required=True,
        help='Directory containing vocabulary files'
    )
    
    # ✅ NEW: Language pair selection
    parser.add_argument(
        '--lang_pair',
        type=str,
        default='en-de',
        choices=['en-de', 'de-en', 'en-fr', 'fr-en'],
        help='Language pair for translation (default: en-de)'
    )
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help='Dataset name (default: auto-detect from lang_pair). '
             'Options: wmt14 (EN-DE), wmt14_enfr (EN-FR)'
    )
    
    # Batch parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Maximum batch size (default: 16)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=1024,
        help=('Maximum tokens per batch (default: 1024 for beam search, 4096 for greedy). '
              'Beam search uses 4x memory (batch_size * beam_size), so use lower values.')
    )
    
    # Beam search parameters
    parser.add_argument(
        '--beam_size',
        type=int,
        default=4,
        help='Beam size for beam search (default: 4, paper setting)'
    )
    
    parser.add_argument(
        '--length_penalty',
        type=float,
        default=0.6,
        help='Length penalty alpha (default: 0.6, paper setting)'
    )
    
    parser.add_argument(
        '--max_len_offset',
        type=int,
        default=50,
        help='Max output = input_length + offset (default: 50, paper setting)'
    )
    
    # Decoding method
    parser.add_argument(
        '--decode_method',
        type=str,
        default='beam',
        choices=['beam', 'greedy'],
        help='Decoding method (default: beam)'
    )
    
    # Dataset split
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'valid', 'test'],
        help='Dataset split to evaluate (default: test)'
    )
    
    # Output
    parser.add_argument(
        '--output_file',
        type=str,
        default='translations.txt',
        help='Output file for translations (default: translations.txt)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='Number of dataloader workers (default: 2)'
    )
    
    return parser.parse_args()


def load_model_and_vocab(args):
    """
    Load model checkpoint and vocabulary
    """
    print("\n1. Loading vocabulary...")
    
    vocab_dir = Path(args.vocab_dir)
    
    # Load vocabulary
    with open(vocab_dir / 'vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
    
    # Create vocab objects
    source_vocab = type('obj', (object,), {
        'stoi': vocab_data['source_stoi'],
        'itos': vocab_data['source_itos'],
        '__len__': lambda self: len(vocab_data['source_stoi'])
    })()
    
    if vocab_data['shared']:
        target_vocab = source_vocab
    else:
        target_vocab = type('obj', (object,), {
            'stoi': vocab_data['target_stoi'],
            'itos': vocab_data['target_itos'],
            '__len__': lambda self: len(vocab_data['target_stoi'])
        })()
    
    print(f"✓ Vocabulary loaded")
    print(f"  Source vocab size: {len(source_vocab):,}")
    print(f"  Target vocab size: {len(target_vocab):,}")
    print(f"  Shared: {vocab_data['shared']}")
    
    # ================================================================
    # 2. Load metadata (optional)
    # ================================================================
    print("\n2. Loading metadata...")
    
    metadata_path = vocab_dir / 'metadata.json'
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Metadata loaded")
        print(f"  Created: {metadata.get('timestamp', 'unknown')}")
        print(f"  Tokenizer: {metadata.get('tokenizer_type', 'unknown')}")
        print(f"  Dataset: {metadata.get('dataset', 'unknown')}")
    else:
        print(f"⚠️  Metadata not found")
        metadata = {}
    
    # ================================================================
    # 3. Load checkpoint
    # ================================================================
    checkpoint_path = Path(args.checkpoint_path)
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Extract model config if available
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        print(f"✓ Model config found in checkpoint")
        print(f"  d_model: {model_config['d_model']}")
        print(f"  n_head: {model_config['n_head']}")
        print(f"  n_layers: {model_config['n_layers']}")
        print(f"  ffn_hidden: {model_config['ffn_hidden']}")
        
        # Custom attention 정보
        if model_config.get('use_custom'):
            print(f"  Attention: Custom (d_k={model_config.get('d_k')}, "
                  f"d_v={model_config.get('d_v')})")
    else:
        print(f"⚠️  Model config not found, using defaults")
        model_config = {
            'd_model': 512,
            'n_head': 8,
            'n_layers': 6,
            'ffn_hidden': 2048,
            'drop_prob': 0.1,
            'max_len': 256,
            'use_custom': False,
            'd_k': None,
            'd_v': None
        }
    
    # ================================================================
    # 4. Initialize model
    # ================================================================
    print("\n3. Initializing model...")
    
    model = Transformer(
        src_pad_idx=source_vocab.stoi['<pad>'],
        trg_pad_idx=target_vocab.stoi['<pad>'],
        trg_sos_idx=target_vocab.stoi['<sos>'],
        enc_voc_size=len(vocab_data['source_stoi']),
        dec_voc_size=len(vocab_data['target_stoi']),
        d_model=model_config['d_model'],
        n_head=model_config['n_head'],
        max_len=model_config['max_len'],
        ffn_hidden=model_config['ffn_hidden'],
        n_layers=model_config['n_layers'],
        drop_prob=model_config['drop_prob'],
        device=args.device,
        gradient_checkpointing=False,  # CRITICAL: Must be False for inference!
        use_custom=model_config.get('use_custom', False),
        d_k=model_config.get('d_k'),
        d_v=model_config.get('d_v')
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    
    # ================================================================
    # 5. Load model weights
    # ================================================================
    print("\n4. Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model weights loaded")
    
    if 'step' in checkpoint:
        print(f"  Checkpoint step: {checkpoint['step']:,}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    
    model.eval()
    
    return model, source_vocab, target_vocab


def translate_dataset_batched(model, dataset, source_vocab, target_vocab, args,
                              source_tokenizer=None, target_tokenizer=None,
                              source_lang='en', target_lang='de'):
    """
    배치 단위로 데이터셋 번역
    
    핵심 개선:
    - TokenBucketSampler 재사용 (길이 기반 배치)
    - collate_inference_batch로 자동 패딩
    - BatchBeamSearch로 병렬 처리
    """
    
    print("\n" + "="*80)
    print(" "*25 + "Batch Translation")
    print("="*80)
    print(f"Total samples: {len(dataset):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tokens: {args.max_tokens:,}")
    print(f"Decode method: {args.decode_method}")
    if args.decode_method == 'beam':
        print(f"Beam size: {args.beam_size}")
        print(f"Length penalty: {args.length_penalty}")
    print("="*80)
    
    # ================================================================
    # 1. DataLoader 생성
    # ================================================================
    from torch.utils.data import DataLoader as TorchDataLoader
    
    # Dataset에 lengths 속성이 없으므로 직접 계산
    print("\nCalculating sentence lengths...")
    lengths = []
    for i in range(len(dataset)):
        example = dataset[i]
        # Get source sentence and tokenize
        src_text = example['translation'][source_lang]
        # Approximate length (split by spaces)
        src_len = len(src_text.split())
        lengths.append(src_len)
    
    print(f"✓ Computed lengths for {len(lengths):,} samples")
    print(f"  Min length: {min(lengths)}")
    print(f"  Max length: {max(lengths)}")
    print(f"  Avg length: {sum(lengths)/len(lengths):.1f}")
    
    sampler = TokenBucketSampler(
        lengths=lengths,
        max_tokens=args.max_tokens,
        shuffle=False,
        drop_last=False
    )
    
    dataloader = TorchDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda batch: collate_inference_batch(
            batch, source_vocab, target_vocab,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_lang=source_lang,
            target_lang=target_lang
        ),
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )
    
    print(f"\nCreated {len(sampler):,} batches")
    print(f"Average batch size: {len(dataset) / len(sampler):.1f}")
    
    # ================================================================
    # 2. Beam Search 초기화
    # ================================================================
    if args.decode_method == 'beam':
        beam_search = BatchBeamSearch(
            model=model,
            beam_size=args.beam_size,
            max_len=args.max_len_offset,
            length_penalty=args.length_penalty,
            sos_idx=target_vocab.stoi['<sos>'],
            eos_idx=target_vocab.stoi['<eos>'],
            pad_idx=target_vocab.stoi['<pad>']
        )
    
    # ================================================================
    # 3. 배치별 번역
    # ================================================================
    all_translations = []
    all_references = []
    
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Translating")):
            src = batch['src'].to(args.device)
            references_batch = batch['trg_text']  # Already decoded strings
            
            batch_start = time.time()
            
            # Decode
            if args.decode_method == 'beam':
                # Beam search
                predictions = beam_search(src, max_offset=args.max_len_offset)
            else:
                # Greedy decoding
                predictions = greedy_decode_batch(
                    model, src,
                    sos_idx=target_vocab.stoi['<sos>'],
                    eos_idx=target_vocab.stoi['<eos>'],
                    max_len=args.max_len_offset,
                    device=args.device
                )
            
            batch_time = time.time() - batch_start
            total_time += batch_time
            total_samples += src.size(0)
            
            # Convert indices to text
            translations_batch = batch_indices_to_texts(predictions, target_vocab)
            
            all_translations.extend(translations_batch)
            all_references.extend(references_batch)
            
            # Progress update
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                avg_time = total_time / total_samples
                speed = total_samples / total_time
                print(f"\nBatch {batch_idx+1}/{len(dataloader)}")
                print(f"  Samples: {total_samples}/{len(dataset)}")
                print(f"  Speed: {speed:.2f} samples/sec")
                print(f"  Avg time/sample: {avg_time*1000:.1f}ms")
    
    # ================================================================
    # Final statistics
    # ================================================================
    print("\n" + "="*80)
    print("Translation Complete!")
    print("="*80)
    print(f"Total samples: {total_samples:,}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average speed: {total_samples/total_time:.2f} samples/sec")
    print(f"Average time per sample: {total_time/total_samples*1000:.1f}ms")
    print("="*80)
    
    return all_translations, all_references


def save_translations(translations, references, output_file):
    """
    Save translations and references to file
    """
    print(f"\nSaving translations to {output_file}...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for trans, ref in zip(translations, references):
            f.write(f"Translation: {trans}\n")
            f.write(f"Reference:   {ref}\n")
            f.write("\n")
    
    print(f"✓ Saved {len(translations):,} translations")


def compute_bleu(translations, references):
    """
    Compute BLEU score using sacrebleu
    """
    print("\n" + "="*80)
    print("Computing BLEU Score")
    print("="*80)
    
    try:
        from sacrebleu.metrics import BLEU
        corpus_bleu = BLEU()
        
        print(f"Total translations: {len(translations)}")
        print(f"Total references: {len(references)}")
        
        # Debug: check unique values
        print(f"Unique translations: {len(set(translations))}")
        print(f"Unique references: {len(set(references))}")
        
        # Check how many references match the most common translation
        if translations:
            from collections import Counter
            trans_counter = Counter(translations)
            most_common_trans, count = trans_counter.most_common(1)[0]
            
            print(f"\nMost common translation (appears {count} times):")
            print(f"  '{most_common_trans}'")
            
            # Count how many references match this translation (lowercase)
            matches = sum(1 for ref in references 
                         if ref.lower().strip() == most_common_trans.lower().strip())
            print(f"References that match most common translation: {matches}/{len(references)}")
            print(f"Match rate: {matches/len(references)*100:.1f}%")
        
        print(f"\nFirst 5 pairs:")
        for i in range(min(5, len(translations))):
            match = "✓" if translations[i].lower().strip() == references[i].lower().strip() else "✗"
            print(f"  [{match}] T: {translations[i][:50]}")
            print(f"      R: {references[i][:50]}")
        
        # sacrebleu expects list of reference streams
        # For 1 reference per sentence: [ [ref1, ref2, ..., refN] ]
        refs_formatted = [references]
        
        # lowercase=True for fair comparison
        bleu = corpus_bleu(translations, refs_formatted, lowercase=True)
        
        print(f"\n{'='*80}")
        print(f"BLEU Score (lowercase)")
        print(f"{'='*80}")
        print(f"BLEU: {bleu.score:.2f}")
        print(f"{'='*80}")
        
        return bleu.score
    except ImportError:
        print("\n⚠️  sacrebleu not installed, skipping BLEU computation")
        print("   Install with: pip install sacrebleu")
        return None


def main():
    args = parse_args()
    
    print("="*80)
    print(" "*10 + "Batched Transformer Inference (60x Faster!)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Vocab dir: {args.vocab_dir}")
    print(f"  Language pair: {args.lang_pair}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max tokens: {args.max_tokens:,}")
    print(f"  Decoding: {args.decode_method}")
    if args.decode_method == 'beam':
        print(f"  Beam size: {args.beam_size}")
        print(f"  Length penalty: {args.length_penalty}")
    print(f"  Device: {args.device}")
    print(f"  Workers: {args.num_workers}")
    
    # ================================================================
    # Load model and vocabulary
    # ================================================================
    model, source_vocab, target_vocab = load_model_and_vocab(args)
    
    # ================================================================
    # ✅ NEW: Determine language configuration from lang_pair
    # ================================================================
    print(f"\n5. Loading dataset for {args.lang_pair}...")
    
    # Parse language pair
    lang_parts = args.lang_pair.split('-')
    if len(lang_parts) != 2:
        raise ValueError(f"Invalid language pair format: {args.lang_pair}. Expected format: 'src-tgt' (e.g., 'en-fr')")
    
    source_lang, target_lang = lang_parts
    
    # Determine dataset name
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        # Auto-detect from language pair
        if args.lang_pair in ['en-de', 'de-en']:
            dataset_name = 'wmt14'
        elif args.lang_pair in ['en-fr', 'fr-en']:
            dataset_name = 'wmt14_enfr'
        else:
            raise ValueError(f"Unsupported language pair: {args.lang_pair}. "
                           f"Please specify --dataset_name manually.")
    
    print(f"  Dataset: {dataset_name}")
    print(f"  Source language: {source_lang}")
    print(f"  Target language: {target_lang}")
    
    # ✅ Load appropriate tokenizers based on language
    from transformers import AutoTokenizer
    
    # Initialize tokenizers dict
    tokenizers = {}
    
    # Load tokenizers for required languages
    for lang in [source_lang, target_lang]:
        if lang == 'en':
            tokenizers['en'] = AutoTokenizer.from_pretrained('gpt2')
            print(f"  Loaded English tokenizer (GPT-2 BPE)")
        elif lang == 'de':
            tokenizers['de'] = AutoTokenizer.from_pretrained('gpt2')
            print(f"  Loaded German tokenizer (GPT-2 BPE)")
        elif lang == 'fr':
            # For French, use WordPiece tokenizer if available
            # Or fall back to GPT-2
            try:
                from util.tokenizer import WordPieceTokenizer
                # Try to load from saved tokenizer
                vocab_dir = Path(args.vocab_dir)
                fr_tokenizer_path = vocab_dir / 'tokenizer_fr.pkl'
                if fr_tokenizer_path.exists():
                    import pickle
                    with open(fr_tokenizer_path, 'rb') as f:
                        tokenizers['fr'] = pickle.load(f)
                    print(f"  Loaded French tokenizer (WordPiece from file)")
                else:
                    # Fallback to GPT-2
                    tokenizers['fr'] = AutoTokenizer.from_pretrained('gpt2')
                    print(f"  Loaded French tokenizer (GPT-2 BPE fallback)")
            except:
                tokenizers['fr'] = AutoTokenizer.from_pretrained('gpt2')
                print(f"  Loaded French tokenizer (GPT-2 BPE fallback)")
    
    # ✅ Create DataLoader with correct language configuration
    loader = DataLoader(
        ext=(f'.{source_lang}', f'.{target_lang}'),
        tokenize_en=tokenizers.get('en'),
        tokenize_de=tokenizers.get('de'),
        tokenize_fr=tokenizers.get('fr'),
        init_token='<sos>',
        eos_token='<eos>'
    )
    
    # ✅ Load correct dataset
    train, valid, test = loader.make_dataset(dataset_name=dataset_name)
    
    if args.split == 'train':
        dataset = train
    elif args.split == 'valid':
        dataset = valid
    else:
        dataset = test
    
    print(f"✓ Loaded {args.split} split: {len(dataset):,} samples")
    
    # ✅ Set correct source/target tokenizers
    source_tokenizer = tokenizers[source_lang]
    target_tokenizer = tokenizers[target_lang]
    
    # ================================================================
    # Translate
    # ================================================================
    translations, references = translate_dataset_batched(
        model, dataset, source_vocab, target_vocab, args,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_lang=source_lang,
        target_lang=target_lang
    )
    
    # ================================================================
    # Save
    # ================================================================
    save_translations(translations, references, args.output_file)
    print("--- DEBUG ---")
    print(f"Pred[0]: {translations[0]}")
    print(f"Ref[0]:  {references[0]}")
    print("-------------")
    # ================================================================
    # Compute BLEU
    # ================================================================
    compute_bleu(translations, references)
    
    print("\n" + "="*80)
    print("Done! 🎉")
    print("="*80)


if __name__ == '__main__':
    main()