"""
Test script to verify label_smoothing fix
Ensures that model can train properly with both label_smoothing=0.0 and 0.1
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from models.model.transformer import Transformer
from training_utils import NaNSafeTrainer


def create_dummy_batch(batch_size=4, src_len=10, trg_len=12, 
                        src_vocab=1000, trg_vocab=1000,
                        src_pad_idx=0, trg_pad_idx=0):
    """Create dummy data for testing"""
    # Create random source and target sequences
    src = torch.randint(1, src_vocab, (batch_size, src_len))
    trg = torch.randint(1, trg_vocab, (batch_size, trg_len))
    
    # Set some padding
    src[:, -2:] = src_pad_idx
    trg[:, -2:] = trg_pad_idx
    
    return src, trg


def test_label_smoothing(label_smoothing_value, num_steps=10):
    """
    Test training with specific label_smoothing value
    
    Returns:
        success: bool - whether training completed without NaN
        avg_loss: float - average loss over steps
        output_stats: dict - statistics about model outputs
    """
    print(f"\n{'='*80}")
    print(f"Testing label_smoothing={label_smoothing_value}")
    print(f"{'='*80}")
    
    # Model parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 512
    n_head = 8
    n_layers = 2  # Smaller for faster testing
    ffn_hidden = 2048
    drop_prob = 0.1
    max_len = 100
    src_vocab_size = 1000
    trg_vocab_size = 1000
    src_pad_idx = 0
    trg_pad_idx = 0
    trg_sos_idx = 1
    
    # Create model
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        enc_voc_size=src_vocab_size,
        dec_voc_size=trg_vocab_size,
        d_model=d_model,
        n_head=n_head,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    
    def lr_schedule(step):
        return 1.0  # Constant for testing
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    
    # Loss function with specified label_smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=trg_pad_idx,
        label_smoothing=label_smoothing_value
    )
    
    # Create NaN-safe trainer
    trainer = NaNSafeTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=1.0
    )
    
    # Training loop
    losses = []
    output_maxs = []
    output_mins = []
    
    model.train()
    
    for step in range(num_steps):
        # Create dummy batch
        src, trg = create_dummy_batch(
            batch_size=4,
            src_vocab=src_vocab_size,
            trg_vocab=trg_vocab_size,
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx
        )
        src = src.to(device)
        trg = trg.to(device)
        
        # Training step
        success, loss_value, stats = trainer.train_step(src, trg)
        
        if not success:
            print(f"  ❌ Step {step+1}: Training failed - {stats['status']}")
            return False, None, None
        
        losses.append(loss_value)
        if 'output_max' in stats:
            output_maxs.append(stats['output_max'])
            output_mins.append(stats['output_min'])
        
        # Print progress every 3 steps
        if (step + 1) % 3 == 0:
            avg_loss = sum(losses[-3:]) / 3
            print(f"  Step {step+1}/{num_steps}: loss={loss_value:.4f}, "
                  f"avg_loss={avg_loss:.4f}, "
                  f"output_range=[{stats.get('output_min', 0):.2f}, {stats.get('output_max', 0):.2f}]")
    
    # Compute statistics
    avg_loss = sum(losses) / len(losses)
    avg_output_max = sum(output_maxs) / len(output_maxs) if output_maxs else 0
    avg_output_min = sum(output_mins) / len(output_mins) if output_mins else 0
    
    # Get NaN statistics
    nan_stats = trainer.get_nan_statistics()
    
    print(f"\n  ✓ Training completed successfully!")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Average output range: [{avg_output_min:.2f}, {avg_output_max:.2f}]")
    print(f"  NaN occurrences: {nan_stats['nan_count']}/{nan_stats['total_batches']}")
    
    output_stats = {
        'avg_output_max': avg_output_max,
        'avg_output_min': avg_output_min,
        'output_range': avg_output_max - avg_output_min
    }
    
    return True, avg_loss, output_stats


def main():
    """Run tests for both label_smoothing values"""
    
    print("\n" + "="*80)
    print("Label Smoothing Fix Validation Test")
    print("="*80)
    print("\nThis test verifies that the model can train properly with both")
    print("label_smoothing=0.0 and label_smoothing=0.1 after removing logit clamping.")
    
    # Test with label_smoothing=0.0
    success_0, loss_0, stats_0 = test_label_smoothing(0.0, num_steps=10)
    
    # Test with label_smoothing=0.1
    success_1, loss_1, stats_1 = test_label_smoothing(0.1, num_steps=10)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    
    if success_0:
        print(f"✓ label_smoothing=0.0: PASSED")
        print(f"  - Average loss: {loss_0:.4f}")
        print(f"  - Output range: [{stats_0['avg_output_min']:.2f}, {stats_0['avg_output_max']:.2f}]")
    else:
        print(f"✗ label_smoothing=0.0: FAILED")
    
    if success_1:
        print(f"✓ label_smoothing=0.1: PASSED")
        print(f"  - Average loss: {loss_1:.4f}")
        print(f"  - Output range: [{stats_1['avg_output_min']:.2f}, {stats_1['avg_output_max']:.2f}]")
    else:
        print(f"✗ label_smoothing=0.1: FAILED")
    
    print(f"{'='*80}")
    
    # Final verdict
    if success_0 and success_1:
        print("\n🎉 All tests PASSED! The fix works correctly.")
        return 0
    else:
        print("\n❌ Some tests FAILED. Please investigate.")
        return 1


if __name__ == '__main__':
    exit(main())
