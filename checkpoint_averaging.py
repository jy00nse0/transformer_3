"""
Checkpoint Averaging Utility
Following "Attention Is All You Need" paper specifications

Paper:
- Base model: Average last 5 checkpoints
- Big model: Average last 20 checkpoints
"""

import torch
from pathlib import Path
import re
from collections import OrderedDict
from checkpoint_utils import validate_checkpoint, get_valid_checkpoints

def average_checkpoints(checkpoint_paths, output_path=None):
    """
    Average multiple checkpoints
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        output_path: Where to save averaged checkpoint (optional)
    
    Returns:
        averaged_state_dict: Averaged model state dict
    """
    print(f"Averaging {len(checkpoint_paths)} checkpoints...")
    
    # Validate all checkpoints first
    valid_paths = []
    for path in checkpoint_paths:
        if validate_checkpoint(path, verbose=False):
            valid_paths.append(path)
        else:
            print(f"⚠️  Skipping corrupted checkpoint: {path}")
    
    if len(valid_paths) == 0:
        raise ValueError("No valid checkpoints found! All checkpoints are corrupted.")
    
    if len(valid_paths) < len(checkpoint_paths):
        print(f"⚠️  Using {len(valid_paths)} valid checkpoints out of {len(checkpoint_paths)} total")
    
    # Load all valid checkpoints
    checkpoints = []
    for path in valid_paths:
        print(f"  Loading: {path}")
        ckpt = torch.load(path, map_location='cpu')
        checkpoints.append(ckpt)
    
    # Get model state dict from first checkpoint
    averaged_state_dict = OrderedDict()
    
    # Get all parameter names
    param_names = checkpoints[0]['model_state_dict'].keys()
    
    # Average each parameter
    print(f"Averaging parameters...")
    for name in param_names:
        # Stack all values for this parameter
        param_values = [ckpt['model_state_dict'][name].float() for ckpt in checkpoints]
        
        # Average
        averaged_value = torch.stack(param_values).mean(dim=0)
        
        # Store
        averaged_state_dict[name] = averaged_value
    
    # Create averaged checkpoint
    averaged_checkpoint = {
        'model_state_dict': averaged_state_dict,
        'averaged_from': [str(p) for p in checkpoint_paths],
        'num_checkpoints': len(checkpoint_paths),
    }
    
    # Include other info from last checkpoint
    if 'step' in checkpoints[-1]:
        averaged_checkpoint['step'] = checkpoints[-1]['step']
    if 'epoch' in checkpoints[-1]:
        averaged_checkpoint['epoch'] = checkpoints[-1]['epoch']
    
    # Include model_config if available
    if 'model_config' in checkpoints[-1]:
        averaged_checkpoint['model_config'] = checkpoints[-1]['model_config']
        print(f"✓ Model config preserved from checkpoint")
    
    # Save if output path provided
    if output_path:
        print(f"Saving averaged checkpoint to: {output_path}")
        torch.save(averaged_checkpoint, output_path)
    
    print(f"✓ Checkpoint averaging complete!")
    
    return averaged_state_dict

def get_last_n_checkpoints(checkpoint_dir, n=5, pattern='model_step_*.pt'):
    """
    Get last N checkpoints from directory, excluding corrupted ones
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        n: Number of checkpoints to get (default: 5 for base model)
        pattern: Checkpoint filename pattern
    
    Returns:
        checkpoint_paths: List of last N checkpoint paths (sorted by step)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Get all valid checkpoints (filters out corrupted ones)
    valid_checkpoint_files = get_valid_checkpoints(checkpoint_dir, pattern, verbose=True)
    
    if len(valid_checkpoint_files) == 0:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir} with pattern {pattern}")
    
    # Extract step numbers and sort
    checkpoint_info = []
    for ckpt_file in valid_checkpoint_files:
        # Extract step number from filename
        match = re.search(r'step_(\d+)', ckpt_file.name)
        if match:
            step = int(match.group(1))
            checkpoint_info.append((step, ckpt_file))
    
    if len(checkpoint_info) == 0:
        raise ValueError(f"Could not extract step numbers from checkpoint filenames")
    
    # Sort by step number
    checkpoint_info.sort(key=lambda x: x[0])
    
    # Get last N
    last_n = checkpoint_info[-n:]
    checkpoint_paths = [ckpt_file for _, ckpt_file in last_n]
    
    print(f"\nSelected last {len(checkpoint_paths)} valid checkpoints:")
    for step, ckpt_file in last_n:
        print(f"  Step {step:6d}: {ckpt_file.name}")
    
    return checkpoint_paths

def average_last_n_checkpoints(checkpoint_dir, n=5, output_path=None, pattern='model_step_*.pt'):
    """
    Average last N checkpoints (paper method)
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        n: Number of checkpoints to average (default: 5 for base model)
        output_path: Where to save averaged model
        pattern: Checkpoint filename pattern
    
    Returns:
        averaged_state_dict: Averaged model state dict
    """
    print(f"\n{'='*80}")
    print(f"Checkpoint Averaging (Paper Implementation)")
    print(f"{'='*80}")
    print(f"Directory: {checkpoint_dir}")
    print(f"Number of checkpoints to average: {n}")
    
    # Get last N checkpoints
    checkpoint_paths = get_last_n_checkpoints(checkpoint_dir, n, pattern)
    
    # Average them
    print()
    averaged_state_dict = average_checkpoints(checkpoint_paths, output_path)
    
    print(f"{'='*80}\n")
    
    return averaged_state_dict

def load_averaged_model(model, checkpoint_dir, n=5, pattern='model_step_*.pt'):
    """
    Load averaged model into a Transformer instance
    
    Args:
        model: Transformer model instance
        checkpoint_dir: Directory containing checkpoints
        n: Number of checkpoints to average
        pattern: Checkpoint filename pattern
    
    Returns:
        model: Model with averaged weights loaded
    """
    averaged_state_dict = average_last_n_checkpoints(
        checkpoint_dir, n=n, output_path=None, pattern=pattern
    )
    
    print(f"Loading averaged weights into model...")
    model.load_state_dict(averaged_state_dict)
    print(f"✓ Model loaded with averaged weights")
    
    return model

# ============================================================================
# Example usage
# ============================================================================

def example_usage():
    """Example of how to use checkpoint averaging"""
    
    # Method 1: Average and save
    averaged_state_dict = average_last_n_checkpoints(
        checkpoint_dir='checkpoints',
        n=5,  # Base model: 5, Big model: 20
        output_path='checkpoints/model_averaged.pt'
    )
    
    # Method 2: Load into model directly
    from models.model.transformer import Transformer
    
    model = Transformer(
        src_pad_idx=1,
        trg_pad_idx=1,
        trg_sos_idx=2,
        enc_voc_size=37000,
        dec_voc_size=37000,
        d_model=512,
        n_head=8,
        max_len=512,
        ffn_hidden=2048,
        n_layers=6,
        drop_prob=0.1,
        device='cuda'
    )
    
    model = load_averaged_model(model, checkpoint_dir='checkpoints', n=5)
    
    # Now use model for inference
    model.eval()
    # ...

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python checkpoint_averaging.py <checkpoint_dir> [n_checkpoints] [output_path]")
        print("\nExample:")
        print("  python checkpoint_averaging.py checkpoints 5 checkpoints/model_averaged.pt")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    if output_path is None:
        output_path = f"{checkpoint_dir}/model_averaged_last{n}.pt"
    
    average_last_n_checkpoints(checkpoint_dir, n, output_path)
    
    print(f"\n✓ Done! Averaged checkpoint saved to: {output_path}")
