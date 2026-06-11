#!/usr/bin/env python3
"""
Compare CAMformerV1 SST Simulator with Paper Results

This script validates the simulator against the paper's expected metrics
(arXiv:2511.19740v1) and Ben's reference implementation.

Usage:
    python compare_with_paper.py
"""

import sys
import os
import numpy as np

# Rely on installed package (pip install -e .)

from camformer.core.paper_hardware import (
    PaperHardwareModel, PaperConfig, PipelineMode, validate_against_paper
)
from camformer.sst.camformer_sst import CAMformerSST, CAMformerSSTConfig


def run_paper_hardware_model():
    """Run the paper hardware model (Ben's implementation)"""
    print("\n" + "="*70)
    print("PAPER HARDWARE MODEL (Ben's Reference Implementation)")
    print("="*70)

    model = PaperHardwareModel()

    # Run all three pipeline modes
    for mode in PipelineMode:
        print(f"\n--- {mode.value.upper()} ---")
        metrics = model.run_attention(mode)
        print(f"  Total Cycles:    {metrics['cycles']['total']:>6}")
        print(f"  Association:     {metrics['cycles']['association']:>6} cycles")
        print(f"  Normalization:   {metrics['cycles']['normalization']:>6} cycles")
        print(f"  Contextualization:{metrics['cycles']['contextualization']:>6} cycles")
        print(f"  Energy (total):  {metrics['energy_nj']['total']:.2f} nJ")
        print(f"  Power (total):   {metrics['power_mw']['total']:.2f} mW")

    # Run realistic mode for full summary
    model.run_attention(PipelineMode.REALISTIC)
    model.print_summary()

    return model.get_metrics()


def run_sst_simulator():
    """Run the SST-based simulator with native Power × Cycles model"""
    print("\n" + "="*70)
    print("SST EVENT-DRIVEN SIMULATOR (Native Metrics)")
    print("="*70)

    # Create configuration matching paper
    config = CAMformerSSTConfig(
        seq_length=1024,
        head_dim=64,
        num_heads=16,
        k_value=32,
        use_paper_model=False,  # Use native SST metrics
    )

    # Create simulator
    camformer = CAMformerSST(config=config)
    camformer.setup()

    # Generate random test data
    np.random.seed(42)
    N = config.seq_length
    d = config.head_dim
    k = config.k_value

    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    # Binarize Q and K (as paper does)
    Q = np.sign(Q).astype(np.float32)
    K = np.sign(K).astype(np.float32)

    # Run single head attention
    outputs, stats = camformer.forward_sync(Q, K, V)

    print(f"\nSST Simulator Results:")
    print(f"  Output shape: {outputs.shape}")

    # Get pipeline statistics (native SST metrics)
    pipeline_stats = camformer.get_pipeline_stats()

    print(f"\n  SST Native Metrics (pipeline_mode={config.pipeline_mode.value}):")
    print(f"  Total Cycles: {pipeline_stats['timing']['total_cycles']}")
    print(f"    Association:      {pipeline_stats['timing']['association_cycles']} cycles")
    print(f"    Normalization:    {pipeline_stats['timing']['normalization_cycles']} cycles")
    print(f"    Contextualization:{pipeline_stats['timing']['contextualization_cycles']} cycles")
    print(f"  Total Energy: {pipeline_stats['energy']['total_nj']:.2f} nJ")
    print(f"    On-chip:          {pipeline_stats['energy']['onchip_nj']:.2f} nJ")
    print(f"    DRAM:             {pipeline_stats['energy']['dram_nj']:.2f} nJ")
    print(f"  Throughput: {pipeline_stats['throughput']['single_core_per_ms']:.2f} att/ms (single-core)")
    print(f"  Area: {pipeline_stats['area_mm2']:.6f} mm²")

    return pipeline_stats


def compare_results():
    """Compare Paper Target vs Ben's Code vs SST Simulator"""
    print("\n" + "="*70)
    print("COMPARISON: Paper Target vs Ben's Code vs Py SST")
    print("="*70)

    # Paper expected values
    expected = {
        "cycles": 721,
        "energy_nj": 54.92,
        "throughput_per_ms": 191.13,
        "area_mm2": 0.258414,
    }

    # Ben's code (paper_hardware.py model)
    paper_model = PaperHardwareModel()
    ben_metrics = paper_model.run_attention(PipelineMode.REALISTIC)

    # SST native metrics
    config = CAMformerSSTConfig(
        seq_length=1024, head_dim=64, num_heads=16, k_value=32,
        use_paper_model=False,
    )
    camformer = CAMformerSST(config=config)
    camformer.setup()
    np.random.seed(42)
    Q = np.sign(np.random.randn(1024, 64)).astype(np.float32)
    K = np.sign(np.random.randn(1024, 64)).astype(np.float32)
    V = np.random.randn(1024, 64).astype(np.float32)
    camformer.forward_sync(Q, K, V)
    sst_metrics = camformer.get_pipeline_stats()

    print("\n{:<25} {:>12} {:>12} {:>12} {:>8}".format(
        "Metric", "Paper", "Ben's Code", "Py SST", "Match?"
    ))
    print("-" * 75)

    results = [
        ("Total Cycles",
         expected["cycles"],
         ben_metrics["cycles"]["total"],
         sst_metrics["timing"]["total_cycles"]),
        ("Energy (nJ)",
         expected["energy_nj"],
         ben_metrics["energy_nj"]["total"],
         sst_metrics["energy"]["total_nj"]),
        ("Throughput (att/ms)",
         expected["throughput_per_ms"],
         ben_metrics["throughput"]["single_core_per_ms"],
         sst_metrics["throughput"]["single_core_per_ms"]),
        ("Area (mm²)",
         expected["area_mm2"],
         ben_metrics["area_mm2"],
         sst_metrics["area_mm2"]),
    ]

    all_match = True
    for name, paper, ben, sst in results:
        ben_diff = abs(ben - paper) / paper if paper != 0 else 0
        sst_diff = abs(sst - paper) / paper if paper != 0 else 0
        match = "✓" if ben_diff < 0.05 and sst_diff < 0.05 else "✗"
        if match == "✗":
            all_match = False
        print(f"{name:<25} {paper:>12.2f} {ben:>12.2f} {sst:>12.2f} {match:>8}")

    print("\n" + "-" * 75)
    print("Stage Cycle Breakdown:")
    print("-" * 75)
    print(f"{'Stage':<25} {'Bens Code':>12} {'Py SST':>12}")
    print(f"{'Association':<25} {ben_metrics['cycles']['association']:>12} {sst_metrics['timing']['association_cycles']:>12}")
    print(f"{'Normalization':<25} {ben_metrics['cycles']['normalization']:>12} {sst_metrics['timing']['normalization_cycles']:>12}")
    print(f"{'Contextualization':<25} {ben_metrics['cycles']['contextualization']:>12} {sst_metrics['timing']['contextualization_cycles']:>12}")

    return all_match


def main():
    """Main comparison function"""
    print("\n" + "#"*70)
    print("# CAMformerV1 Validation: Paper = Ben's Code = Py SST")
    print("# Paper: CAMformer (arXiv:2511.19740v1)")
    print("#"*70)

    # Run paper hardware model (Ben's code)
    paper_metrics = run_paper_hardware_model()

    # Run SST simulator with native metrics
    try:
        sst_stats = run_sst_simulator()
    except Exception as e:
        print(f"\nSST Simulator error: {e}")
        import traceback
        traceback.print_exc()
        sst_stats = None

    # Three-way comparison
    all_match = compare_results()

    # Final validation
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    if all_match:
        print("\n✓ All three match within 5% tolerance!")
        print("  Paper Target = Ben's Code = Py SST")
    else:
        print("\n✗ Some metrics don't match.")

    # Show paper configuration
    config = PaperConfig()
    print(f"\nPaper Configuration:")
    print(f"  Sequence Length: {config.seq_length}")
    print(f"  Head Dimension:  {config.head_dim}")
    print(f"  Number of Heads: {config.num_heads}")
    print(f"  Top-K:           {config.k_value}")
    print(f"  Tiles:           {config.num_tiles}")
    print(f"  MAC Parallelism: {config.c_par}")
    print(f"  Pipeline Mode:   {config.pipeline_mode.value}")


if __name__ == "__main__":
    main()
