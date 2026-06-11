#!/usr/bin/env python3
"""
CAMformer SST Event-Driven Simulation Test

Tests the event-driven simulation framework.
"""

import sys
import os
import numpy as np

# Rely on installed package
from camformer.sst import (
    CAMformerSST, CAMformerSSTConfig,
    SimulationRunner,
    AssociationWrapper,
    SelectionWrapper,
    ContextualizationWrapper,
)
from camformer.core.simulation import Simulation
from camformer.pipeline.association import AssociationConfig
from camformer.pipeline.selection import SelectionConfig
from camformer.pipeline.contextualization import ContextualizationConfig


def test_association_wrapper():
    """Test Association stage event wrapper"""
    print("\n" + "="*60)
    print("Testing Association Wrapper (Event-Driven)")
    print("="*60)

    config = AssociationConfig(max_seq_length=64, head_dim=32)
    wrapper = AssociationWrapper("test_assoc", config=config)
    wrapper.setup()

    # Test data
    N, d = 64, 32
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)

    # Load keys
    wrapper.load_keys(K)

    # Process queries with callback
    results = [None]

    def on_complete(event):
        results[0] = event.get_data('similarity_matrix')
        print(f"  Callback received: similarity shape = {results[0].shape}")

    wrapper.process_queries(Q, callback=on_complete)

    # Verify results
    similarity = wrapper.get_results()
    assert similarity is not None, "No results"
    assert similarity.shape == (N, N), f"Wrong shape: {similarity.shape}"

    print(f"  Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
    print(f"  Events received: {wrapper.get_statistic('events_received').get_value()}")
    print(f"  Total latency: {wrapper.get_statistic('total_latency').get_value()} cycles")

    print("Association wrapper test PASSED")
    return True


def test_selection_wrapper():
    """Test Selection stage event wrapper"""
    print("\n" + "="*60)
    print("Testing Selection Wrapper (Event-Driven)")
    print("="*60)

    config = SelectionConfig(k_value=8, max_seq_length=64)
    wrapper = SelectionWrapper("test_select", config=config)
    wrapper.setup()

    # Test data
    N, k = 64, 8
    similarity = np.random.rand(N, N).astype(np.float32)

    # Process with callback
    results = [None]

    def on_complete(event):
        results[0] = event.get_data('attention_weights')
        print(f"  Callback received: attention shape = {results[0].shape}")

    wrapper.process_similarity(similarity, callback=on_complete)

    # Verify results
    result = wrapper.get_results()
    assert result['attention_weights'] is not None
    assert result['attention_weights'].shape == (N, k)

    # Check softmax sums to 1
    weight_sums = result['attention_weights'].sum(axis=1)
    print(f"  Attention weight sums: [{weight_sums.min():.3f}, {weight_sums.max():.3f}]")
    print(f"  Rows processed: {wrapper.get_statistic('rows_processed').get_value()}")

    print("Selection wrapper test PASSED")
    return True


def test_contextualization_wrapper():
    """Test Contextualization stage event wrapper"""
    print("\n" + "="*60)
    print("Testing Contextualization Wrapper (Event-Driven)")
    print("="*60)

    config = ContextualizationConfig(head_dim=32, k_value=8, max_seq_length=64)
    wrapper = ContextualizationWrapper("test_context", config=config)
    wrapper.setup()

    # Test data
    N, k, d = 64, 8, 32
    V = np.random.randn(N, d).astype(np.float32)
    attention = np.random.rand(N, k).astype(np.float32)
    attention = attention / attention.sum(axis=1, keepdims=True)
    indices = np.random.randint(0, N, size=(N, k)).astype(np.int32)

    # Load values
    wrapper.load_values(V)

    # Process with callback
    results = [None]

    def on_complete(event):
        results[0] = event.get_data('outputs')
        print(f"  Callback received: output shape = {results[0].shape}")

    wrapper.process_attention(attention, indices, callback=on_complete)

    # Verify results
    output = wrapper.get_results()
    assert output is not None
    assert output.shape == (N, d), f"Wrong shape: {output.shape}"

    print(f"  Outputs computed: {wrapper.get_statistic('outputs_computed').get_value()}")
    print(f"  Total latency: {wrapper.get_statistic('total_latency').get_value()} cycles")

    print("Contextualization wrapper test PASSED")
    return True


def test_camformer_sst_pipeline():
    """Test complete CAMformer SST pipeline"""
    print("\n" + "="*60)
    print("Testing CAMformer SST Pipeline (Event-Driven)")
    print("="*60)

    config = CAMformerSSTConfig(
        seq_length=64,
        head_dim=32,
        num_heads=1,
        k_value=8,
    )

    pipeline = CAMformerSST("test_pipeline", config=config)
    pipeline.setup()

    # Test data
    N, d = 64, 32
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    # Run forward pass
    output, stats = pipeline.forward_sync(Q, K, V)

    print(f"  Output shape: {output.shape}")
    print(f"  Total latency: {pipeline.get_total_latency_cycles()} cycles")
    print(f"  Total latency: {pipeline.get_total_latency_ns()/1000:.2f} us")
    print(f"  Area: {pipeline.get_area_mm2():.4f} mm²")

    print("\n  Stage breakdown:")
    for stage, stage_stats in stats.items():
        print(f"    {stage}: {stage_stats}")

    assert output.shape == (N, d)

    print("CAMformer SST pipeline test PASSED")
    return True


def test_simulation_runner():
    """Test simulation runner"""
    print("\n" + "="*60)
    print("Testing Simulation Runner")
    print("="*60)

    config = CAMformerSSTConfig(
        seq_length=64,
        head_dim=32,
        num_heads=1,
        k_value=8,
    )

    runner = SimulationRunner(config=config)
    runner.setup()

    # Single head test
    N, d = 64, 32
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    result = runner.run_single_head(Q, K, V)
    runner.print_result(result)

    # Comparison with dense
    comparison = runner.compare_with_dense(N, d)
    print(f"\n  Dense vs CAMformer:")
    print(f"    Compute speedup: {comparison['speedup']['compute']:.2f}x")
    print(f"    Memory reduction: {comparison['speedup']['memory']:.2f}x")

    print("Simulation runner test PASSED")
    return True


def test_sequence_sweep():
    """Test sequence length sweep"""
    print("\n" + "="*60)
    print("Testing Sequence Length Sweep")
    print("="*60)

    config = CAMformerSSTConfig(
        seq_length=64,
        head_dim=64,
        k_value=32,
    )

    runner = SimulationRunner(config=config)
    runner.setup()

    # Sweep
    seq_lengths = [64, 128, 256, 512]
    results = runner.run_sweep(seq_lengths, head_dim=64, k_value=32)

    runner.print_sweep_summary(results)

    # Verify scaling
    print("\n  Scaling analysis:")
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        N_ratio = curr['config']['seq_length'] / prev['config']['seq_length']
        cycle_ratio = curr['latency_cycles'] / prev['latency_cycles']
        print(f"    N: {prev['config']['seq_length']} -> {curr['config']['seq_length']}: "
              f"cycles {cycle_ratio:.2f}x (ideal O(N): {N_ratio:.1f}x)")

    print("Sequence sweep test PASSED")
    return True


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("CAMformer SST Event-Driven Simulation Tests")
    print("="*60)

    tests = [
        ("Association Wrapper", test_association_wrapper),
        ("Selection Wrapper", test_selection_wrapper),
        ("Contextualization Wrapper", test_contextualization_wrapper),
        ("CAMformer SST Pipeline", test_camformer_sst_pipeline),
        ("Simulation Runner", test_simulation_runner),
        ("Sequence Sweep", test_sequence_sweep),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
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
