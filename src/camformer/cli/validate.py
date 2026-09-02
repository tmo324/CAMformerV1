"""Validate CAMformer hardware metrics against the published paper targets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from camformer.core.paper_hardware import (
    PaperHardwareModel,
    PipelineMode,
    published_energy_efficiency,
)
from camformer.sst.camformer_sst import CAMformerSST, CAMformerSSTConfig


PUBLISHED_TARGETS = {
    "cycles": 721.0,
    "energy_nj": 54.92,
    "throughput_per_ms": 191.13,
    "area_mm2": 0.258414,
    "power_w": 0.17,
    "efficiency_per_mj": 9045.0,
}


def _analytical_metrics() -> dict[str, float]:
    metrics = PaperHardwareModel().run_attention(PipelineMode.REALISTIC)
    return {
        "cycles": float(metrics["cycles"]["total"]),
        "energy_nj": metrics["energy_nj"]["total"],
        "throughput_per_ms": metrics["throughput"]["single_core_per_ms"],
        "area_mm2": metrics["area_mm2"],
        "power_w": metrics["power_mw"]["total"] / 1000.0,
        "efficiency_per_mj": metrics["efficiency"]["queries_per_mj"],
    }


def _event_model_metrics() -> dict[str, float]:
    config = CAMformerSSTConfig(
        seq_length=1024,
        head_dim=64,
        num_heads=16,
        k_value=32,
        use_paper_model=False,
    )
    simulator = CAMformerSST(config=config)
    simulator.setup()

    rng = np.random.default_rng(42)
    q = np.sign(rng.standard_normal((1024, 64))).astype(np.float32)
    k = np.sign(rng.standard_normal((1024, 64))).astype(np.float32)
    v = rng.standard_normal((1024, 64)).astype(np.float32)
    simulator.forward_sync(q, k, v)
    metrics = simulator.get_pipeline_stats()

    energy_nj = metrics["energy"]["total_nj"]
    return {
        "cycles": float(metrics["timing"]["total_cycles"]),
        "energy_nj": energy_nj,
        "throughput_per_ms": metrics["throughput"]["single_core_per_ms"],
        "area_mm2": metrics["area_mm2"],
        "power_w": metrics["power"]["total_mw"] / 1000.0,
        "efficiency_per_mj": published_energy_efficiency(energy_nj),
    }


def metrics_match(
    actual: dict[str, float],
    expected: dict[str, float] = PUBLISHED_TARGETS,
    tolerance: float = 0.05,
) -> bool:
    """Return whether every metric is within the relative tolerance."""
    return all(
        abs(actual[name] - target) / target <= tolerance
        for name, target in expected.items()
    )


def _print_comparison(
    analytical: dict[str, float],
    event_model: dict[str, float],
    tolerance: float,
) -> None:
    labels = {
        "cycles": "Total cycles",
        "energy_nj": "Energy (nJ)",
        "throughput_per_ms": "Throughput (att/ms)",
        "area_mm2": "Area (mm^2)",
        "power_w": "Power (W)",
        "efficiency_per_mj": "Efficiency (qry/mJ)",
    }
    print(f"{'Metric':<24} {'Paper':>12} {'Analytical':>12} {'Event model':>12}")
    print("-" * 64)
    for name, target in PUBLISHED_TARGETS.items():
        print(
            f"{labels[name]:<24} {target:>12.4f} "
            f"{analytical[name]:>12.4f} {event_model[name]:>12.4f}"
        )
    print(f"\nTolerance: {tolerance:.1%}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the published-metric validation and return a shell exit code."""
    parser = argparse.ArgumentParser(
        description="Validate CAMformer metrics against the published targets."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Maximum relative deviation for every metric (default: 0.05).",
    )
    args = parser.parse_args(argv)
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")

    analytical = _analytical_metrics()
    event_model = _event_model_metrics()
    _print_comparison(analytical, event_model, args.tolerance)

    passed = metrics_match(analytical, tolerance=args.tolerance)
    passed = passed and metrics_match(event_model, tolerance=args.tolerance)
    print("\nValidation PASSED" if passed else "\nValidation FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
