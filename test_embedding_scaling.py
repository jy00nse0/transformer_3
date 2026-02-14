"""
Test to verify that embedding scaling by sqrt(d_model) is correctly applied.
According to the Transformer paper, embeddings should be multiplied by sqrt(d_model)
before adding positional encodings.
"""
import torch
import math
import sys
import os

# Add the parent directory to the path to import the models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.embedding.embedding import Embedding


def test_embedding_scaling():
    """
    Test that embeddings are scaled by sqrt(d_model)
    """
    print("="*80)
    print("Testing Embedding Scaling by sqrt(d_model)")
    print("="*80)
    
    # Test parameters
    vocab_size = 1000
    d_model = 512
    max_len = 100
    drop_prob = 0.0  # No dropout for testing
    device = 'cpu'
    
    print(f"\nTest Configuration:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  sqrt(d_model): {math.sqrt(d_model):.6f}")
    print(f"  max_len: {max_len}")
    
    # Create embedding layer
    emb = Embedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len, 
                    drop_prob=drop_prob, device=device)
    
    # Create sample input
    sample_input = torch.tensor([[1, 2, 3, 4, 5]])
    
    print(f"\nSample input shape: {sample_input.shape}")
    
    # Get the embedding output
    with torch.no_grad():
        output = emb(sample_input)
        
        # Get token embedding and positional encoding separately
        tok_emb_unscaled = emb.tok_emb(sample_input)
        pos_emb = emb.pos_emb(sample_input)
        
        # Calculate what the scaled embedding should be
        tok_emb_scaled = tok_emb_unscaled * math.sqrt(d_model)
        expected_output = tok_emb_scaled + pos_emb
    
    print(f"\nEmbedding values (first token, first 5 dimensions):")
    print(f"  Unscaled token embedding: {tok_emb_unscaled[0, 0, :5]}")
    print(f"  Scaled token embedding:   {tok_emb_scaled[0, 0, :5]}")
    print(f"  Positional encoding:      {pos_emb[0, :5]}")
    print(f"  Expected output (scaled + pos): {expected_output[0, 0, :5]}")
    print(f"  Actual output:            {output[0, 0, :5]}")
    
    # Verify that the output matches the expected scaled output
    is_correct = torch.allclose(output, expected_output, rtol=1e-5, atol=1e-7)
    
    print(f"\n{'='*80}")
    if is_correct:
        print("✓ TEST PASSED: Embedding is correctly scaled by sqrt(d_model)")
        print(f"  The implementation multiplies token embeddings by sqrt({d_model}) = {math.sqrt(d_model):.6f}")
    else:
        print("✗ TEST FAILED: Embedding scaling is not correctly applied")
        print(f"  Expected scaling by sqrt({d_model}) = {math.sqrt(d_model):.6f}")
        
        # Calculate the actual scaling factor
        diff = output - pos_emb
        actual_scale = (diff / tok_emb_unscaled).mean().item()
        print(f"  Actual scaling factor: {actual_scale:.6f}")
    
    print("="*80)
    
    return is_correct


def test_different_d_models():
    """
    Test embedding scaling with different d_model values
    """
    print("\n" + "="*80)
    print("Testing Embedding Scaling with Different d_model Values")
    print("="*80)
    
    d_models = [128, 256, 512, 1024]
    all_passed = True
    
    for d_model in d_models:
        print(f"\nTesting d_model = {d_model} (sqrt = {math.sqrt(d_model):.6f}):")
        
        emb = Embedding(vocab_size=1000, d_model=d_model, max_len=100, 
                       drop_prob=0.0, device='cpu')
        
        sample_input = torch.tensor([[1, 2, 3]])
        
        with torch.no_grad():
            output = emb(sample_input)
            tok_emb = emb.tok_emb(sample_input)
            pos_emb = emb.pos_emb(sample_input)
            expected = tok_emb * math.sqrt(d_model) + pos_emb
        
        is_correct = torch.allclose(output, expected, rtol=1e-5, atol=1e-7)
        
        if is_correct:
            print(f"  ✓ PASSED: Correctly scaled by {math.sqrt(d_model):.6f}")
        else:
            print(f"  ✗ FAILED: Scaling incorrect")
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    print("\nTransformer Embedding Scaling Test Suite\n")
    
    # Run main test
    test1_passed = test_embedding_scaling()
    
    # Run additional tests with different d_model values
    test2_passed = test_different_d_models()
    
    # Final result
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    if test1_passed and test2_passed:
        print("✓ All tests PASSED")
        print("\nThe Transformer implementation correctly applies sqrt(d_model) scaling")
        print("to embeddings as specified in the paper.")
        sys.exit(0)
    else:
        print("✗ Some tests FAILED")
        sys.exit(1)
