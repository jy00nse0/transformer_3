"""
Robust Checkpoint Utilities
Provides safe checkpoint saving and validation
"""

import torch
import os
import tempfile
import shutil
from pathlib import Path


def safe_torch_save(obj, path, atomic=True):
    """
    Safely save a PyTorch object to disk with atomic write
    
    Args:
        obj: Object to save (usually a dict containing model state)
        path: Target file path
        atomic: If True, use atomic write (write to temp file then rename)
                This prevents corruption if the process is killed during save
    
    Returns:
        bool: True if save was successful, False otherwise
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if atomic:
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                dir=path.parent,
                prefix=f'.{path.name}.tmp'
            ) as tmp_file:
                tmp_path = tmp_file.name
                torch.save(obj, tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())  # Ensure written to disk
            
            # Atomic rename (replaces existing file if present)
            shutil.move(tmp_path, path)
        else:
            # Direct save (faster but not atomic)
            torch.save(obj, path)
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error saving checkpoint to {path}: {e}")
        # Clean up temp file if it exists
        if atomic and 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        return False


def validate_checkpoint(checkpoint_path, verbose=True):
    """
    Validate that a checkpoint file is readable
    
    Args:
        checkpoint_path: Path to checkpoint file
        verbose: If True, print validation messages
    
    Returns:
        bool: True if checkpoint is valid, False otherwise
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        if verbose:
            print(f"⚠️  Checkpoint does not exist: {checkpoint_path}")
        return False
    
    # Check file size (should not be 0)
    file_size = checkpoint_path.stat().st_size
    if file_size == 0:
        if verbose:
            print(f"⚠️  Checkpoint is empty (0 bytes): {checkpoint_path}")
        return False
    
    # Try to load the checkpoint
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Basic validation - should have model_state_dict
        if not isinstance(ckpt, dict):
            if verbose:
                print(f"⚠️  Checkpoint is not a dictionary: {checkpoint_path}")
            return False
        
        if 'model_state_dict' not in ckpt:
            if verbose:
                print(f"⚠️  Checkpoint missing 'model_state_dict': {checkpoint_path}")
            return False
        
        if verbose:
            print(f"✓ Valid checkpoint: {checkpoint_path.name} ({file_size / 1024**2:.1f} MB)")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Checkpoint corrupted: {checkpoint_path.name}")
            print(f"   Error: {e}")
        return False


def get_valid_checkpoints(checkpoint_dir, pattern='model_step_*.pt', verbose=True):
    """
    Get all valid checkpoints from a directory, filtering out corrupted ones
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint files
        verbose: If True, print validation messages
    
    Returns:
        list: List of valid checkpoint paths
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        if verbose:
            print(f"⚠️  Checkpoint directory does not exist: {checkpoint_dir}")
        return []
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob(pattern))
    
    if verbose:
        print(f"\nValidating checkpoints in {checkpoint_dir}...")
        print(f"Found {len(checkpoint_files)} checkpoint files\n")
    
    # Validate each checkpoint
    valid_checkpoints = []
    invalid_checkpoints = []
    
    for ckpt_file in checkpoint_files:
        if validate_checkpoint(ckpt_file, verbose=verbose):
            valid_checkpoints.append(ckpt_file)
        else:
            invalid_checkpoints.append(ckpt_file)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Validation Results:")
        print(f"  Valid checkpoints: {len(valid_checkpoints)}")
        print(f"  Invalid checkpoints: {len(invalid_checkpoints)}")
        
        if invalid_checkpoints:
            print(f"\n⚠️  Invalid/Corrupted checkpoints:")
            for ckpt in invalid_checkpoints:
                print(f"     - {ckpt.name}")
        
        print(f"{'='*80}\n")
    
    return valid_checkpoints


def remove_corrupted_checkpoints(checkpoint_dir, pattern='model_step_*.pt', dry_run=True):
    """
    Remove corrupted checkpoints from a directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint files
        dry_run: If True, only print what would be deleted without actually deleting
    
    Returns:
        list: List of removed checkpoint paths
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob(pattern))
    
    removed = []
    
    for ckpt_file in checkpoint_files:
        if not validate_checkpoint(ckpt_file, verbose=False):
            if dry_run:
                print(f"Would remove: {ckpt_file}")
            else:
                print(f"Removing corrupted checkpoint: {ckpt_file}")
                ckpt_file.unlink()
            removed.append(ckpt_file)
    
    return removed
