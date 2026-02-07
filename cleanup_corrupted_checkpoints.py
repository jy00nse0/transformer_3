#!/usr/bin/env python3
"""
Clean up corrupted checkpoints from a directory

This script will:
1. Scan the specified checkpoint directory for corrupted checkpoint files
2. Show you which ones are corrupted
3. Optionally remove them (with confirmation)

Usage:
    python cleanup_corrupted_checkpoints.py <checkpoint_dir> [--remove]
    
Examples:
    # Check for corrupted checkpoints (dry run)
    python cleanup_corrupted_checkpoints.py checkpoints/table3_a_h32
    
    # Check and remove corrupted checkpoints
    python cleanup_corrupted_checkpoints.py checkpoints/table3_a_h32 --remove
"""

import argparse
from pathlib import Path
from checkpoint_utils import validate_checkpoint, remove_corrupted_checkpoints


def main():
    parser = argparse.ArgumentParser(
        description='Clean up corrupted checkpoint files'
    )
    parser.add_argument(
        'checkpoint_dir',
        type=str,
        help='Directory containing checkpoints'
    )
    parser.add_argument(
        '--remove',
        action='store_true',
        help='Actually remove corrupted checkpoints (default: dry run)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='model_step_*.pt',
        help='Checkpoint filename pattern (default: model_step_*.pt)'
    )
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"❌ Error: Directory does not exist: {checkpoint_dir}")
        return 1
    
    print("="*80)
    print("Checkpoint Cleanup Utility")
    print("="*80)
    print(f"Directory: {checkpoint_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Mode: {'REMOVE' if args.remove else 'DRY RUN'}")
    print("="*80)
    print()
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob(args.pattern))
    
    if len(checkpoint_files) == 0:
        print(f"No checkpoint files found matching pattern: {args.pattern}")
        return 0
    
    print(f"Found {len(checkpoint_files)} checkpoint files\n")
    
    # Validate each checkpoint
    valid_count = 0
    corrupted_files = []
    
    for ckpt_file in sorted(checkpoint_files):
        if validate_checkpoint(ckpt_file, verbose=False):
            valid_count += 1
            print(f"✓ Valid:     {ckpt_file.name}")
        else:
            corrupted_files.append(ckpt_file)
            print(f"❌ Corrupted: {ckpt_file.name}")
    
    print()
    print("="*80)
    print(f"Summary:")
    print(f"  Valid checkpoints: {valid_count}")
    print(f"  Corrupted checkpoints: {len(corrupted_files)}")
    print("="*80)
    
    if len(corrupted_files) == 0:
        print("\n✓ All checkpoints are valid! No cleanup needed.")
        return 0
    
    print(f"\nCorrupted checkpoint files:")
    for ckpt in corrupted_files:
        file_size_mb = ckpt.stat().st_size / (1024**2)
        print(f"  - {ckpt.name} ({file_size_mb:.1f} MB)")
    
    if not args.remove:
        print("\n⚠️  DRY RUN MODE: No files were removed.")
        print("   To actually remove corrupted checkpoints, run with --remove flag:")
        print(f"   python cleanup_corrupted_checkpoints.py {args.checkpoint_dir} --remove")
        return 0
    
    # Confirm removal
    print("\n⚠️  WARNING: This will permanently delete corrupted checkpoint files!")
    response = input("Are you sure you want to continue? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Cancelled. No files were removed.")
        return 0
    
    # Remove corrupted files
    print("\nRemoving corrupted checkpoints...")
    for ckpt in corrupted_files:
        try:
            ckpt.unlink()
            print(f"  ✓ Removed: {ckpt.name}")
        except Exception as e:
            print(f"  ❌ Failed to remove {ckpt.name}: {e}")
    
    print("\n✓ Cleanup complete!")
    return 0


if __name__ == '__main__':
    exit(main())
