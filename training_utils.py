"""
NaN-Safe Training Utilities
Handles numerical instability in extreme configurations (e.g., d_k=16)
"""

import torch
import torch.nn as nn


class NaNSafeTrainer:
    """
    Handles NaN detection and safe training for Transformer
    """
    
    def __init__(self, model, criterion, optimizer, scheduler, max_grad_norm=1.0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        
        # Statistics
        self.total_batches = 0
        self.nan_count = 0
        self.nan_in_output = 0
        self.nan_in_loss = 0
        self.nan_in_gradient = 0
        
    def compute_loss_safe(self, src, trg):
        """
        Compute loss with NaN detection
        
        Returns:
            loss: valid loss tensor or None
            stats: dict with diagnostic information
        """
        # Forward pass
        output = self.model(src, trg[:, :-1])
        
        # ================================================================
        # Check 1: NaN/Inf in model output
        # ================================================================
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        
        if has_nan or has_inf:
            nan_ratio = torch.isnan(output).sum().item() / output.numel()
            inf_ratio = torch.isinf(output).sum().item() / output.numel()
            
            # Get statistics from valid values
            valid_mask = ~(torch.isnan(output) | torch.isinf(output))
            if valid_mask.any():
                valid_output = output[valid_mask]
                output_max = valid_output.max().item()
                output_min = valid_output.min().item()
                output_mean = valid_output.mean().item()
            else:
                output_max = output_min = output_mean = float('nan')
            
            self.nan_in_output += 1
            
            return None, {
                'status': 'output_invalid',
                'nan_ratio': nan_ratio,
                'inf_ratio': inf_ratio,
                'output_max': output_max,
                'output_min': output_min,
                'output_mean': output_mean,
                'has_nan': has_nan.item(),
                'has_inf': has_inf.item()
            }
        
        # ================================================================
        # Check 2: Clamp output to prevent extreme values
        # ================================================================
        # Before softmax (in loss), clamp logits to reasonable range
        output = torch.clamp(output, min=-50, max=50)
        
        # ================================================================
        # Check 3: Compute loss
        # ================================================================
        loss = self.criterion(
            output.contiguous().view(-1, output.shape[-1]),
            trg[:, 1:].contiguous().view(-1)
        )
        
        # ================================================================
        # Check 4: NaN/Inf in loss
        # ================================================================
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_in_loss += 1
            
            return None, {
                'status': 'loss_invalid',
                'loss_value': loss.item() if not torch.isnan(loss) else float('nan')
            }
        
        # ================================================================
        # Check 5: Extreme loss values
        # ================================================================
        if loss.item() > 100 or loss.item() < -100:
            return None, {
                'status': 'loss_extreme',
                'loss_value': loss.item()
            }
        
        return loss, {
            'status': 'ok',
            'loss_value': loss.item(),
            'output_max': output.max().item(),
            'output_min': output.min().item()
        }
    
    def backward_safe(self, loss):
        """
        Safe backward pass with gradient clipping and NaN detection
        
        Returns:
            success: bool
            stats: dict with gradient statistics
        """
        self.optimizer.zero_grad()
        
        # Backward
        loss.backward()
        
        # ================================================================
        # Check 1: NaN/Inf in gradients
        # ================================================================
        nan_params = []
        inf_params = []
        grad_norms = {}
        total_grad_norm_sq = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Check for NaN
                if torch.isnan(param.grad).any():
                    nan_params.append(name)
                
                # Check for Inf
                if torch.isinf(param.grad).any():
                    inf_params.append(name)
                
                # Compute norm
                if not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    param_norm = param.grad.data.norm(2).item()
                    grad_norms[name] = param_norm
                    total_grad_norm_sq += param_norm ** 2
        
        if nan_params or inf_params:
            # NaN/Inf detected - skip this update
            self.optimizer.zero_grad()
            self.nan_in_gradient += 1
            
            return False, {
                'status': 'gradient_invalid',
                'nan_params': nan_params,
                'inf_params': inf_params,
                'total_grad_norm': float('nan')
            }
        
        total_grad_norm = total_grad_norm_sq ** 0.5
        
        # ================================================================
        # Check 2: Gradient clipping
        # ================================================================
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.max_grad_norm
        )
        
        # ================================================================
        # Check 3: Optimizer step
        # ================================================================
        self.optimizer.step()
        self.scheduler.step()
        
        # Find largest gradients (for debugging)
        top_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return True, {
            'status': 'ok',
            'grad_norm': total_grad_norm,
            'max_grad_norm': self.max_grad_norm,
            'top_gradients': top_grads
        }
    
    def train_step(self, src, trg):
        """
        Complete training step with NaN safety
        
        Returns:
            success: bool
            loss_value: float or None
            stats: dict
        """
        self.total_batches += 1
        
        # Compute loss
        loss, loss_stats = self.compute_loss_safe(src, trg)
        
        if loss is None:
            # NaN detected in forward pass
            self.nan_count += 1
            return False, None, loss_stats
        
        # Backward pass
        success, grad_stats = self.backward_safe(loss)
        
        if not success:
            # NaN detected in backward pass
            self.nan_count += 1
            return False, loss.item(), {**loss_stats, **grad_stats}
        
        # Success
        return True, loss.item(), {**loss_stats, **grad_stats}
    
    def get_nan_statistics(self):
        """Get NaN occurrence statistics"""
        if self.total_batches == 0:
            return {
                'total_batches': 0,
                'nan_count': 0,
                'nan_ratio': 0.0,
                'nan_in_output': 0,
                'nan_in_loss': 0,
                'nan_in_gradient': 0
            }
        
        return {
            'total_batches': self.total_batches,
            'nan_count': self.nan_count,
            'nan_ratio': self.nan_count / self.total_batches,
            'nan_in_output': self.nan_in_output,
            'nan_in_loss': self.nan_in_loss,
            'nan_in_gradient': self.nan_in_gradient,
            'nan_in_output_pct': self.nan_in_output / self.total_batches * 100,
            'nan_in_loss_pct': self.nan_in_loss / self.total_batches * 100,
            'nan_in_gradient_pct': self.nan_in_gradient / self.total_batches * 100
        }
    
    def print_nan_report(self):
        """Print detailed NaN statistics"""
        stats = self.get_nan_statistics()
        
        print("\n" + "="*80)
        print("NaN Detection Report")
        print("="*80)
        print(f"Total batches processed: {stats['total_batches']:,}")
        print(f"NaN occurrences: {stats['nan_count']:,} ({stats['nan_ratio']*100:.2f}%)")
        print(f"\nBreakdown:")
        print(f"  NaN in model output:  {stats['nan_in_output']:,} ({stats['nan_in_output_pct']:.2f}%)")
        print(f"  NaN in loss:          {stats['nan_in_loss']:,} ({stats['nan_in_loss_pct']:.2f}%)")
        print(f"  NaN in gradients:     {stats['nan_in_gradient']:,} ({stats['nan_in_gradient_pct']:.2f}%)")
        print("="*80)
        
        # Warning if NaN ratio is high
        if stats['nan_ratio'] > 0.1:
            print("⚠️  WARNING: NaN ratio exceeds 10%!")
            print("   Consider:")
            print("   - Reducing learning rate")
            print("   - Increasing gradient clipping threshold")
            print("   - Using mixed precision training")
            print("="*80)


def print_training_log(step, max_steps, loss, avg_loss, lr, grad_norm, 
                       steps_per_sec, eta_hours, nan_stats):
    """
    Print enhanced training log with NaN statistics
    """
    nan_ratio = nan_stats['nan_ratio'] * 100
    
    # Base log
    log = (f"Step {step:6,}/{max_steps:,} ({step/max_steps*100:5.1f}%) | "
           f"Loss: {loss:.4f} (avg: {avg_loss:.4f}) | "
           f"LR: {lr:.6f} | "
           f"GradNorm: {grad_norm:.2f}")
    
    # Add NaN info if any NaN occurred
    if nan_stats['nan_count'] > 0:
        log += f" | NaN: {nan_ratio:.2f}% ({nan_stats['nan_count']}/{nan_stats['total_batches']})"
    
    # Add speed info
    log += f" | Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_hours:.1f}h"
    
    print(log)


def print_nan_warning(step, loss_stats, nan_stats):
    """
    Print detailed NaN warning
    """
    status = loss_stats['status']
    nan_ratio = nan_stats['nan_ratio'] * 100
    
    print(f"\n⚠️  Step {step}: NaN detected - {status}")
    
    if status == 'output_invalid':
        print(f"   Model output NaN: {loss_stats['nan_ratio']*100:.2f}%")
        print(f"   Model output Inf: {loss_stats['inf_ratio']*100:.2f}%")
        if not all(v == float('nan') for v in [loss_stats['output_max'], 
                                                 loss_stats['output_min'], 
                                                 loss_stats['output_mean']]):
            print(f"   Valid output range: [{loss_stats['output_min']:.2f}, {loss_stats['output_max']:.2f}]")
            print(f"   Valid output mean: {loss_stats['output_mean']:.2f}")
    
    elif status == 'loss_invalid':
        print(f"   Loss value: {loss_stats['loss_value']}")
    
    elif status == 'loss_extreme':
        print(f"   Extreme loss value: {loss_stats['loss_value']:.2f}")
    
    elif status == 'gradient_invalid':
        if loss_stats.get('nan_params'):
            print(f"   NaN in parameters: {len(loss_stats['nan_params'])} params")
            print(f"   Examples: {loss_stats['nan_params'][:3]}")
        if loss_stats.get('inf_params'):
            print(f"   Inf in parameters: {len(loss_stats['inf_params'])} params")
    
    print(f"   Total NaN occurrence: {nan_stats['nan_count']}/{nan_stats['total_batches']} ({nan_ratio:.2f}%)")
    print()
