"""
Inference Script with Beam Search and Checkpoint Averaging
Following "Attention Is All You Need" paper specifications

Paper specifications:
1. Model: Average last 5 checkpoints (base) or 20 (big)
2. Beam search with beam_size=4, length_penalty=0.6
3. Max output length: input_length + 50
4. Early termination
"""

import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import time

from models.model.transformer import Transformer
from util.data_loader import DataLoader
from beam_search import BeamSearch, greedy_decode

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Inference with Beam Search')
    
    # Model paths
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to checkpoint file (e.g., model_averaged_last5.pt or model_final.pt)'
    )
    
    parser.add_argument(
        '--vocab_dir',
        type=str,
        required=True,
        help='Directory containing vocabulary files'
    )
    
    # Beam search parameters (paper settings)
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
    
    # Data
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'valid', 'test'],
        help='Dataset split to translate (default: test)'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to translate (default: all)'
    )
    
    # Output
    parser.add_argument(
        '--output_file',
        type=str,
        default='translations.txt',
        help='Output file for translations'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    return parser.parse_args()

def load_model_and_vocab(args):
    """Load model from single checkpoint file and vocabulary"""
    
    print("="*80)
    print(" "*25 + "Model Loading")
    print("="*80)
    
    # Load vocabulary
    print("\n1. Loading vocabulary...")
    import pickle
    
    vocab_path = Path(args.vocab_dir) / 'vocab.pkl'
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    print(f"✓ Vocabulary loaded")
    print(f"  Source vocab size: {len(vocab_data['source_stoi']):,}")
    print(f"  Target vocab size: {len(vocab_data['target_stoi']):,}")
    
    # Get special token indices
    src_pad_idx = vocab_data['source_stoi']['<pad>']
    trg_pad_idx = vocab_data['target_stoi']['<pad>']
    trg_sos_idx = vocab_data['target_stoi']['<sos>']
    trg_eos_idx = vocab_data['target_stoi']['<eos>']
    
    # Load checkpoint
    print(f"\n2. Loading checkpoint: {args.checkpoint_path}")
    checkpoint_path = Path(args.checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
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
    
    # Initialize model
    print("\n3. Initializing model...")
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        enc_voc_size=len(vocab_data['source_stoi']),
        dec_voc_size=len(vocab_data['target_stoi']),
        d_model=model_config['d_model'],
        n_head=model_config['n_head'],
        max_len=model_config['max_len'],
        ffn_hidden=model_config['ffn_hidden'],
        n_layers=model_config['n_layers'],
        drop_prob=model_config['drop_prob'],
        device=args.device,
        kdim=model_config.get('kdim')
    ).to(args.device)
    
    print(f"✓ Model initialized")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Load model state
    print("\n4. Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model weights loaded")
    
    model.eval()
    
    return model, vocab_data, trg_sos_idx, trg_eos_idx

def translate_dataset(model, dataset, beam_search, vocab_data, args):
    """Translate entire dataset"""
    
    print("\n" + "="*80)
    print(" "*25 + "Translation")
    print("="*80)
    print(f"Method: {args.decode_method}")
    if args.decode_method == 'beam':
        print(f"Beam size: {args.beam_size}")
        print(f"Length penalty: {args.length_penalty}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print("="*80)
    
    translations = []
    references = []
    
    # Limit samples
    total_samples = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    
    # Get stoi/itos mappings
    src_stoi = vocab_data['source_stoi']
    trg_itos = vocab_data['target_itos']
    
    start_time = time.time()
    
    for idx in tqdm(range(total_samples), desc="Translating"):
        example = dataset[idx]
        
        # Get source sentence
        src_text = example.src if hasattr(example, 'src') else str(example)
        trg_text = example.trg if hasattr(example, 'trg') else str(example)
        
        # Tokenize source (you'll need to use your tokenizer here)
        # For now, assume it's already tokenized in the dataset
        # In practice, you'd use: src_tokens = tokenizer.tokenize(src_text)
        
        # Convert to indices (simplified - you'll need proper tokenization)
        # This is a placeholder - implement proper tokenization
        src_indices = [src_stoi.get(token, src_stoi['<unk>']) 
                      for token in src_text.split()]
        
        # Add to batch (for now, process one at a time)
        src_tensor = torch.tensor([src_indices], dtype=torch.long, device=args.device)
        
        # Translate
        with torch.no_grad():
            if args.decode_method == 'beam':
                translated = beam_search.translate(src_tensor)
            else:
                translated = greedy_decode(
                    model, src_tensor,
                    sos_idx=beam_search.sos_idx,
                    eos_idx=beam_search.eos_idx
                )
        
        # Convert back to text
        translated_indices = translated[0].cpu().tolist()
        translated_tokens = [trg_itos[idx] for idx in translated_indices 
                           if idx not in [beam_search.sos_idx, beam_search.eos_idx, beam_search.pad_idx]]
        translated_text = ' '.join(translated_tokens)
        
        translations.append(translated_text)
        references.append(trg_text)
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Translation complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Time: {elapsed:.1f}s ({total_samples/elapsed:.2f} samples/s)")
    
    return translations, references

def save_translations(translations, references, output_file):
    """Save translations to file"""
    
    print(f"\nSaving translations to: {output_file}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (trans, ref) in enumerate(zip(translations, references)):
            f.write(f"# Example {i+1}\n")
            f.write(f"Translation: {trans}\n")
            f.write(f"Reference:   {ref}\n")
            f.write("\n")
    
    print(f"✓ Saved {len(translations)} translations")

def compute_bleu(translations, references):
    """Compute BLEU score (simplified)"""
    try:
        from sacrebleu import corpus_bleu
        
        # sacrebleu expects list of references for each translation
        refs_formatted = [[ref] for ref in references]
        
        bleu = corpus_bleu(translations, refs_formatted)
        
        print(f"\n{'='*80}")
        print(f"BLEU Score")
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
    print(" "*15 + "Transformer Inference (Paper Implementation)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Vocab dir: {args.vocab_dir}")
    print(f"  Decoding: {args.decode_method}")
    if args.decode_method == 'beam':
        print(f"  Beam size: {args.beam_size}")
        print(f"  Length penalty: {args.length_penalty}")
    print(f"  Device: {args.device}")
    
    # Load model and vocabulary
    model, vocab_data, trg_sos_idx, trg_eos_idx = load_model_and_vocab(args)
    
    # Initialize beam search
    if args.decode_method == 'beam':
        beam_search = BeamSearch(
            model=model,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            sos_idx=trg_sos_idx,
            eos_idx=trg_eos_idx,
            pad_idx=vocab_data['target_stoi']['<pad>']
        )
    else:
        beam_search = None
    
    # Load dataset
    print("\n5. Loading dataset...")
    loader = DataLoader(
        ext=('.en', '.de'),
        tokenize_en=None,
        tokenize_de=None,
        init_token='<sos>',
        eos_token='<eos>'
    )
    
    train, valid, test = loader.make_dataset(dataset_name='wmt14')
    
    if args.split == 'train':
        dataset = train
    elif args.split == 'valid':
        dataset = valid
    else:
        dataset = test
    
    print(f"✓ Loaded {args.split} split: {len(dataset):,} samples")
    
    # Translate
    translations, references = translate_dataset(
        model, dataset, beam_search, vocab_data, args
    )
    
    # Save
    save_translations(translations, references, args.output_file)
    
    # Compute BLEU
    compute_bleu(translations, references)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)

if __name__ == '__main__':
    main()
