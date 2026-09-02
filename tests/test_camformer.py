#!/usr/bin/env python3
"""
CAMformer Simulator Test Script

Tests the complete CAMformer pipeline with various configurations.
"""

import sys
import os
import numpy as np

# Rely on installed package
from camformer.pipeline import (
    CAMformerPipeline, CAMformerConfig,
    CAMformerAttentionHead, AttentionHeadConfig,
    AssociationStage, AssociationConfig,
    SelectionStage, SelectionConfig,
    ContextualizationStage, ContextualizationConfig,
)


def test_association_stage():
    """Test the Association stage (BA-CAM search)"""
    print("\n" + "="*60)
    print("Testing Association Stage")
    print("="*60)

    config = AssociationConfig(
        max_seq_length=64,
        head_dim=32,
    )

    stage = AssociationStage("test_assoc", config=config)
    stage.setup()

    # Create test data
    N, d = 64, 32
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)

    # Load keys into BA-CAM
    key_latency = stage.load_keys(K)
    print(f"Key loading latency: {key_latency} cycles")

    # Process all queries
    hamming, similarity, latency = stage.process_all_queries(Q)

    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Query processing latency: {latency} cycles")
    print(f"Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")

    # Get stage stats
    stats = stage.get_stage_stats()
    print(f"Stage area: {stats['area_mm2']:.4f} mm²")

    assert hamming.shape == (N, N)
    assert similarity.shape == (N, N)
    assert key_latency > 0 and latency > 0
    assert np.all((similarity >= 0.0) & (similarity <= 1.0))
    assert stats['area_mm2'] > 0

    print("Association stage test PASSED")


def test_selection_stage():
    """Test the Selection stage (Top-K + SoftMax)"""
    print("\n" + "="*60)
    print("Testing Selection Stage")
    print("="*60)

    config = SelectionConfig(
        k_value=8,
        max_seq_length=64,
    )

    stage = SelectionStage("test_select", config=config)
    stage.setup()

    # Create test similarity matrix
    N = 64
    similarity = np.random.rand(N, N).astype(np.float32)

    # Process all rows
    values, indices, weights, latency = stage.process_all_rows(similarity)

    print(f"Top-k values shape: {values.shape}")
    print(f"Top-k indices shape: {indices.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Selection latency: {latency} cycles")

    # Verify softmax sums to 1
    weight_sums = weights.sum(axis=1)
    print(f"Attention weight sums (should be ~1): [{weight_sums.min():.3f}, {weight_sums.max():.3f}]")

    # Verify indices are valid
    assert values.shape == (N, 8)
    assert indices.shape == (N, 8)
    assert weights.shape == (N, 8)
    assert np.allclose(weight_sums, 1.0, atol=1e-6)
    assert indices.min() >= 0 and indices.max() < N, "Invalid indices"

    print("Selection stage test PASSED")


def test_contextualization_stage():
    """Test the Contextualization stage (Sparse MatMul)"""
    print("\n" + "="*60)
    print("Testing Contextualization Stage")
    print("="*60)

    config = ContextualizationConfig(
        head_dim=32,
        k_value=8,
        max_seq_length=64,
    )

    stage = ContextualizationStage("test_context", config=config)
    stage.setup()

    # Create test data
    N, k, d = 64, 8, 32
    V = np.random.randn(N, d).astype(np.float32)
    attention_weights = np.random.rand(N, k).astype(np.float32)
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)  # Normalize
    indices = np.random.randint(0, N, size=(N, k)).astype(np.int32)

    # Load values
    load_latency = stage.load_values(V)
    print(f"Value loading latency: {load_latency} cycles")

    # Compute outputs
    outputs, latency = stage.compute_all_outputs(attention_weights, indices)

    print(f"Output shape: {outputs.shape}")
    print(f"Contextualization latency: {latency} cycles")

    # Verify output is reasonable
    assert outputs.shape == (N, d), f"Wrong output shape: {outputs.shape}"
    assert not np.isnan(outputs).any(), "NaN in outputs"

    print("Contextualization stage test PASSED")


def test_full_pipeline():
    """Test the complete CAMformer pipeline"""
    print("\n" + "="*60)
    print("Testing Full CAMformer Pipeline")
    print("="*60)

    config = CAMformerConfig(
        seq_length=64,
        head_dim=32,
        num_heads=4,
        k_value=8,
    )

    pipeline = CAMformerPipeline("test_pipeline", config=config)
    pipeline.setup()

    # Create test data
    N, h, d = 64, 4, 32
    Q = np.random.randn(h, N, d).astype(np.float32)
    K = np.random.randn(h, N, d).astype(np.float32)
    V = np.random.randn(h, N, d).astype(np.float32)

    # Forward pass
    output, stats = pipeline.forward(Q, K, V)

    print(f"Output shape: {output.shape}")
    print(f"Total latency: {stats['latency']['total']} cycles")
    print(f"Latency breakdown:")
    for key, val in stats['latency'].items():
        if key != 'total':
            print(f"  {key}: {val} cycles")

    # Memory traffic
    print(f"\nMemory traffic:")
    for key, val in stats['memory_traffic'].items():
        print(f"  {key}: {val:,} bytes")

    # Compare with dense attention
    comparison = pipeline.compare_with_dense(N, d)
    print(f"\nCAMformer vs Dense comparison:")
    print(f"  Compute speedup: {comparison['speedup']['compute']:.2f}x")
    print(f"  Memory reduction: {comparison['speedup']['memory']:.2f}x")

    # Get pipeline stats
    pipe_stats = pipeline.get_pipeline_stats()
    print(f"\nPipeline area: {pipe_stats['area']['total_mm2']:.4f} mm²")

    assert output.shape == (h, N, d)
    assert np.isfinite(output).all()
    assert stats['latency']['total'] > 0
    assert comparison['speedup']['compute'] > 1.0
    assert pipe_stats['area']['total_mm2'] > 0

    print("Full pipeline test PASSED")


def test_attention_head():
    """Test the high-level attention head"""
    print("\n" + "="*60)
    print("Testing CAMformer Attention Head")
    print("="*60)

    config = AttentionHeadConfig(
        hidden_dim=256,
        head_dim=32,
        num_heads=4,
        k_value=8,
        max_seq_length=64,
        include_projections=False,
    )

    head = CAMformerAttentionHead("test_head", config=config)
    head.setup()

    # Create test data
    N, h, d = 64, 4, 32
    Q = np.random.randn(h, N, d).astype(np.float32)
    K = np.random.randn(h, N, d).astype(np.float32)
    V = np.random.randn(h, N, d).astype(np.float32)

    # Forward pass
    output, stats = head.forward_attention_only(Q, K, V)

    print(f"Output shape: {output.shape}")
    print(f"Total latency: {stats['latency']['total']} cycles")

    # Get head stats
    head_stats = head.get_head_stats()
    print(f"Head area: {head_stats['area_mm2']:.4f} mm²")

    assert output.shape == (h, N, d)
    assert np.isfinite(output).all()
    assert stats['latency']['total'] > 0
    assert head_stats['area_mm2'] > 0

    print("Attention head test PASSED")


def test_scaling():
    """Test CAMformer with different sequence lengths"""
    print("\n" + "="*60)
    print("Testing CAMformer Scaling")
    print("="*60)

    seq_lengths = [64, 128, 256, 512]
    k_value = 32
    head_dim = 64

    print(f"{'Seq Len':<10} {'Latency':<15} {'Dense Compute':<15} {'CAM Compute':<15} {'Speedup':<10}")
    print("-" * 65)

    latencies = []

    for N in seq_lengths:
        config = CAMformerConfig(
            seq_length=N,
            head_dim=head_dim,
            num_heads=1,
            k_value=min(k_value, N),
        )

        pipeline = CAMformerPipeline(f"scale_{N}", config=config)
        pipeline.setup()

        # Single-head test data
        Q = np.random.randn(N, head_dim).astype(np.float32)
        K = np.random.randn(N, head_dim).astype(np.float32)
        V = np.random.randn(N, head_dim).astype(np.float32)

        output, stats = pipeline.forward(Q, K, V)

        comparison = pipeline.compare_with_dense(N, head_dim)

        assert output.shape == (N, head_dim)
        assert np.isfinite(output).all()
        assert comparison['speedup']['compute'] > 1.0
        latencies.append(stats['latency']['total'])

        print(f"{N:<10} {stats['latency']['total']:<15} "
              f"{comparison['dense']['compute']:<15} "
              f"{comparison['camformer']['compute']:<15} "
              f"{comparison['speedup']['compute']:.2f}x")

    assert latencies == sorted(latencies)
    assert len(set(latencies)) == len(latencies)
    print("\nScaling test PASSED")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("CAMformer Simulator Test Suite")
    print("="*60)

    tests = [
        ("Association Stage", test_association_stage),
        ("Selection Stage", test_selection_stage),
        ("Contextualization Stage", test_contextualization_stage),
        ("Full Pipeline", test_full_pipeline),
        ("Attention Head", test_attention_head),
        ("Scaling", test_scaling),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, True))
        except Exception as e:
            print(f"\n{name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
