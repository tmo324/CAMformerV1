"""Regenerate CAMformer paper Figure 10, the PPA Pareto comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from camformer.core.paper_hardware import PaperHardwareModel, PipelineMode
from experiments.figures.common import DEFAULT_OUTPUT_DIR, apply_paper_style, prepare_output
from matplotlib import pyplot as plt


COLORS = {
    "CAMformer": "#6C8EBF",
    "Spatten": "#B85450",
    "Cerebras WSE2": "#D79B00",
    "Groq TSP": "#9673A6",
    "TPUv4": "#82B366",
}

# Reference values preserve the camera-ready Figure 10 plotting convention.
# Their publication sources and scaling assumptions are listed in TRACEABILITY.md.
CEREBRAS = {
    "performance_gops": 7.5e6,
    "area_mm2": 215 * 215,
    "power_w": (30e-3) * 10156 * 84,
}
GROQ = {"performance_gops": 205e3, "area_mm2": 725, "power_w": 300}
TPUV4 = {"performance_gops": 275e3, "area_mm2": 600, "power_w": 192}
SPATTEN_EFFECTIVE_GOPS = 360
SPATTEN_PERFORMANCE_PER_WATT = 382
AREA_SCALE_45_TO_22 = 0.23
ENERGY_SCALE_45_TO_22 = 0.23 / 0.48


def compute_pareto_points() -> dict[str, dict[str, float]]:
    """Calculate the camera-ready Figure 10 operating points."""
    model = PaperHardwareModel()
    metrics = model.run_attention(PipelineMode.REALISTIC)

    effective_ops = (2 * 64 * 1024) + (2 * 1024) + (2 * 64 * 1024)
    effective_gops = effective_ops / metrics["cycles"]["max_stage"]
    area_mm2 = metrics["area_mm2"]
    onchip_power_w = metrics["power_mw"]["onchip"] / 1000.0

    points = {
        "Cerebras WSE2": {
            "performance_per_area": CEREBRAS["performance_gops"]
            / CEREBRAS["area_mm2"],
            "performance_per_watt": CEREBRAS["performance_gops"]
            / CEREBRAS["power_w"],
        },
        "Groq TSP": {
            "performance_per_area": GROQ["performance_gops"] / GROQ["area_mm2"],
            "performance_per_watt": GROQ["performance_gops"] / GROQ["power_w"],
        },
        "TPUv4": {
            "performance_per_area": TPUV4["performance_gops"] / TPUV4["area_mm2"],
            "performance_per_watt": TPUV4["performance_gops"] / TPUV4["power_w"],
        },
        "CAMformer": {
            "performance_per_area": effective_gops / area_mm2,
            "performance_per_watt": effective_gops / onchip_power_w,
        },
        "Spatten": {
            "performance_per_area": SPATTEN_EFFECTIVE_GOPS / area_mm2,
            "performance_per_watt": SPATTEN_PERFORMANCE_PER_WATT,
        },
        "CAMformer Proj.": {
            "performance_per_area": effective_gops / (area_mm2 * AREA_SCALE_45_TO_22),
            "performance_per_watt": effective_gops
            / (onchip_power_w * ENERGY_SCALE_45_TO_22),
        },
        "Spatten Proj.": {
            "performance_per_area": SPATTEN_EFFECTIVE_GOPS
            / (area_mm2 * AREA_SCALE_45_TO_22),
            "performance_per_watt": SPATTEN_EFFECTIVE_GOPS
            / (onchip_power_w * ENERGY_SCALE_45_TO_22),
        },
    }
    return points


def generate_figure(
    output_path: Path | str = DEFAULT_OUTPUT_DIR / "fig10_pareto_front.png",
) -> Path:
    """Render Figure 10 to a PNG without opening an interactive window."""
    apply_paper_style()
    points = compute_pareto_points()
    marker_size = 160

    fig, ax = plt.subplots(figsize=(7, 4))
    for label in ("Cerebras WSE2", "Groq TSP", "TPUv4", "CAMformer", "Spatten"):
        point = points[label]
        ax.scatter(
            point["performance_per_area"],
            point["performance_per_watt"],
            s=marker_size,
            label=label,
            c=COLORS[label],
            edgecolors="black",
            zorder=2 if label in {"TPUv4", "CAMformer", "Spatten"} else None,
        )

    for label, base_label in (
        ("CAMformer Proj.", "CAMformer"),
        ("Spatten Proj.", "Spatten"),
    ):
        point = points[label]
        base = points[base_label]
        ax.scatter(
            point["performance_per_area"],
            point["performance_per_watt"],
            s=marker_size,
            label=label,
            marker="*",
            c=COLORS[base_label],
            edgecolors="black",
            zorder=2,
        )
        ax.annotate(
            "",
            xy=(point["performance_per_area"], point["performance_per_watt"]),
            xytext=(base["performance_per_area"], base["performance_per_watt"]),
            arrowprops={"lw": 1, "shrink": 0.1, "color": "gray"},
        )

    theta = np.linspace(0, 2 * np.pi, 100)
    industry = points["TPUv4"]
    industry_radius = np.hypot(
        industry["performance_per_area"], industry["performance_per_watt"]
    )
    ax.plot(
        industry_radius * np.cos(theta),
        industry_radius * np.sin(theta),
        label="Pareto Front Industry",
        color="lightgreen",
        linestyle="--",
        linewidth=2,
        zorder=1,
    )

    research = points["CAMformer Proj."]
    research_radius = np.hypot(
        research["performance_per_area"], research["performance_per_watt"]
    )
    ax.plot(
        research_radius * np.cos(theta),
        research_radius * np.sin(theta),
        label="Pareto Front Research",
        color="lightblue",
        linestyle="--",
        linewidth=2,
        zorder=1,
    )

    ax.set_facecolor("#f7fcff")
    ax.set_xlabel("Performance per Area (Eff. Gops/mm^2)", fontsize=14)
    ax.set_ylabel("Performance per Watt (Eff. Gops/W)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e2, 3e4)
    ax.set_ylim(1e2, 3e4)
    ax.legend(ncol=3, fontsize=10, loc="center", bbox_to_anchor=(0.5, 1.12))
    ax.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(ax.get_ygridlines()) > 4:
        ax.get_ygridlines()[4].set_visible(False)

    output = prepare_output(output_path)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "fig10_pareto_front.png",
    )
    args = parser.parse_args(argv)
    print(f"Generated {generate_figure(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
