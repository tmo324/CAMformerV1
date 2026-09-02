"""Regression tests for deterministic paper-figure scripts."""

import struct

import numpy as np

from experiments.figures.fig06_bimm_energy import (
    compute_bimm_energy,
    generate_figure as generate_fig06,
)
from experiments.figures.fig08_area_energy_breakdown import (
    compute_breakdowns,
    generate_figure as generate_fig08,
)
from experiments.figures.fig10_pareto_front import (
    compute_pareto_points,
    generate_figure as generate_fig10,
)


def _png_dimensions(path):
    with path.open("rb") as image_file:
        assert image_file.read(8) == b"\x89PNG\r\n\x1a\n"
        image_file.read(8)
        return struct.unpack(">II", image_file.read(8))


def test_fig06_data_matches_the_paper_plot():
    data = compute_bimm_energy()

    assert np.array_equal(data["matrix_dim"], 2 ** np.arange(9))
    assert np.isclose(data["energy_per_op_pj"][0], 0.123046875)
    assert np.isclose(data["search_only_pj"], 0.024609375)


def test_fig08_breakdowns_cover_all_onchip_modules():
    data = compute_breakdowns()

    assert np.isclose(sum(data["outer_energy_nj"]), data["onchip_energy_nj"])
    assert np.isclose(sum(data["outer_area_um2"]), data["onchip_area_um2"])
    assert data["outer_labels"] == [
        "Association",
        "Normalization",
        "Contextualization",
    ]


def test_fig10_uses_the_validated_camformer_operating_point():
    data = compute_pareto_points()

    assert np.isclose(data["CAMformer"]["performance_per_area"], 3126.487, rtol=1e-3)
    assert np.isclose(data["CAMformer"]["performance_per_watt"], 7372.505, rtol=1e-3)


def test_all_paper_figures_render_headlessly(tmp_path):
    outputs = [
        generate_fig06(tmp_path / "fig06_bimm_energy.png"),
        generate_fig08(tmp_path / "fig08_area_energy_breakdown.png"),
        generate_fig10(tmp_path / "fig10_pareto_front.png"),
    ]

    for output in outputs:
        width, height = _png_dimensions(output)
        assert output.stat().st_size > 10_000
        assert width > 1_000
        assert height > 500
