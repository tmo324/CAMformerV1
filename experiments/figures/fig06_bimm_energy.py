"""Regenerate CAMformer paper Figure 6, BIMM energy per operation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.figures.common import (
    DEFAULT_OUTPUT_DIR,
    MODULES_CSV,
    apply_paper_style,
    load_raw_modules,
    prepare_output,
)
from matplotlib import pyplot as plt


CAMFORMER_BLUE = "#6C8EBF"
PROGRAM_ORANGE = "#D79B00"
SEARCH_GREEN = "#82B366"


def compute_bimm_energy(modules_path: Path = MODULES_CSV) -> dict[str, np.ndarray | float]:
    """Return the values plotted in Figure 6 from the raw BA-CAM result."""
    module = load_raw_modules(modules_path)["BA-CAM 64X64"]
    power_mw = module.power_mw * module.number / 4.0

    matrix_dim = 2 ** np.arange(9)
    rows_per_tile = 16
    dot_product_width = 64
    operations = matrix_dim * rows_per_tile * dot_product_width
    writes = rows_per_tile // 4
    searches = matrix_dim * (rows_per_tile // 16)
    energy_pj = power_mw * (searches + writes)

    return {
        "matrix_dim": matrix_dim,
        "energy_per_op_pj": energy_pj / operations,
        "program_and_search_pj": float(np.max(energy_pj / operations)),
        "search_only_pj": power_mw / (rows_per_tile * dot_product_width),
    }


def generate_figure(
    output_path: Path | str = DEFAULT_OUTPUT_DIR / "fig06_bimm_energy.png",
    modules_path: Path = MODULES_CSV,
) -> Path:
    """Render Figure 6 to a PNG without opening an interactive window."""
    apply_paper_style()
    data = compute_bimm_energy(modules_path)
    x = np.arange(9)

    fig, ax = plt.subplots(figsize=(4, 2 * 4 / 5))
    ax.grid(linewidth=0.4, color="gray", linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    if len(ax.get_ygridlines()) > 3:
        ax.get_ygridlines()[3].set_visible(False)
    ax.set_facecolor("#f7fcff")

    ax.bar(
        x,
        data["energy_per_op_pj"],
        edgecolor="black",
        color=CAMFORMER_BLUE,
        width=0.8,
        linewidth=0.8,
    )
    ax.annotate(
        "BIMV",
        xy=(0.1, 0.075),
        rotation=90,
        fontsize=18,
        color="black",
        ha="center",
        va="center",
    )
    ax.axhline(
        y=data["program_and_search_pj"],
        color=PROGRAM_ORANGE,
        linestyle="--",
        label="Program + Search",
        linewidth=2,
    )
    ax.axhline(
        y=data["search_only_pj"],
        color=SEARCH_GREEN,
        linestyle="--",
        label="Search",
        linewidth=2,
    )
    ax.set_xlabel("M (Matrix Dim.)", fontsize=12)
    ax.set_ylabel("Energy per op (pJ)", fontsize=12)
    ax.set_xticks(x, labels=data["matrix_dim"], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 0.13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="center right",
        fontsize=12,
        bbox_to_anchor=(1, 0.6),
        frameon=True,
    )

    output = prepare_output(output_path)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "fig06_bimm_energy.png",
    )
    args = parser.parse_args(argv)
    print(f"Generated {generate_figure(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
