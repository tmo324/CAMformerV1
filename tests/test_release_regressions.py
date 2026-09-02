"""Regression tests for the public release surface."""

import numpy as np

from camformer.core.paper_hardware import PaperHardwareModel, PipelineMode
from camformer.cli.sweep import run_tile_sweep, run_topk_sweep
from camformer.sst import CAMformerSST, CAMformerSSTConfig


def test_forward_sync_processes_each_row_once(capsys):
    """The synchronous path must not also traverse the immediate event links."""
    config = CAMformerSSTConfig(
        seq_length=8,
        head_dim=4,
        num_heads=1,
        k_value=2,
        use_paper_model=False,
    )
    simulator = CAMformerSST("release_regression", config=config)
    simulator.setup()

    rng = np.random.default_rng(7)
    q = rng.standard_normal((8, 4), dtype=np.float32)
    k = rng.standard_normal((8, 4), dtype=np.float32)
    v = rng.standard_normal((8, 4), dtype=np.float32)

    output, stats = simulator.forward_sync(q, k, v)
    captured = capsys.readouterr()

    assert output.shape == (8, 4)
    assert stats["selection"]["rows"] == 8
    assert stats["contextualization"]["outputs"] == 8
    assert "Values not loaded" not in captured.err


def test_paper_metrics_match_published_targets():
    """The release must keep the six headline paper metrics stable."""
    metrics = PaperHardwareModel().run_attention(PipelineMode.REALISTIC)

    assert metrics["cycles"]["total"] == 721
    assert np.isclose(metrics["energy_nj"]["total"], 54.92, rtol=0.05)
    assert np.isclose(
        metrics["throughput"]["single_core_per_ms"], 191.13, rtol=0.05
    )
    assert np.isclose(metrics["area_mm2"], 0.258414, rtol=0.05)
    assert np.isclose(metrics["power_mw"]["total"] / 1000.0, 0.17, rtol=0.05)
    assert np.isclose(
        metrics["efficiency"]["queries_per_mj"], 9045.0, rtol=0.05
    )


def test_packaged_cli_modules_are_importable():
    """Console entry points must target modules included in the wheel."""
    from camformer.cli.sweep import main as sweep_main
    from camformer.cli.validate import main as validate_main

    assert callable(sweep_main)
    assert callable(validate_main)


def test_sensitivity_rows_match_published_table():
    """The default sweeps must reproduce the camera-ready Table IV rows."""
    topk_rows = run_topk_sweep()
    tile_rows = run_tile_sweep()

    assert [row["param"] for row in topk_rows] == [8, 16, 24, 32]
    assert [round(row["efficiency"]) for row in topk_rows] == [
        12130,
        10920,
        9930,
        9104,
    ]
    assert all(
        np.isclose(row["area"], 0.258414, rtol=1e-5) for row in topk_rows
    )
    assert [round(row["recall"], 1) for row in topk_rows] == [
        99.9,
        99.4,
        98.5,
        97.4,
    ]

    assert [row["param"] for row in tile_rows] == [4, 8, 16, 32]
    assert [round(row["throughput"]) for row in tile_rows] == [120, 160, 191, 212]
    assert [round(row["efficiency"]) for row in tile_rows] == [
        4198,
        6625,
        9104,
        10600,
    ]
