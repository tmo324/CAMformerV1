"""Regenerate CAMformer paper Figure 8, area and energy breakdowns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from camformer.core.paper_hardware import PaperHardwareModel, PipelineMode
from experiments.figures.common import DEFAULT_OUTPUT_DIR, apply_paper_style, prepare_output
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


MODULE_GROUPINGS = {
    "Association": [
        "BA-CAM 16X64",
        "Query Buffer",
        "Key SRAM",
        "6b ADCs",
        "Fixed Scalers",
        "Top2",
        "Ptop Reg",
    ],
    "Normalization": [
        "Top32",
        "SoftMax LUT",
        "SoftMax Div.",
        "SoftMax Acc.",
        "Output Buff",
    ],
    "Contextualization": ["BF16 MACs", "Value SRAM"],
}

COLOR_CONFIG = {
    "Association": {
        "start": np.array([213, 232, 212]) / 255.0,
        "end": np.array([130, 179, 102]) / 255.0,
    },
    "Normalization": {
        "start": np.array([255, 230, 204]) / 255.0,
        "end": np.array([215, 155, 0]) / 255.0,
    },
    "Contextualization": {
        "start": np.array([248, 206, 204]) / 255.0,
        "end": np.array([184, 84, 80]) / 255.0,
    },
}


def _gradient_color(group: str, index: int, total: int) -> np.ndarray:
    fraction = 0.25 + (index / (total - 1)) * 0.75 if total > 1 else 1.0
    config = COLOR_CONFIG[group]
    return config["start"] * (1 - fraction) + config["end"] * fraction


def compute_breakdowns() -> dict[str, object]:
    """Calculate the Figure 8 values using the validated paper model."""
    model = PaperHardwareModel()
    metrics = model.run_attention(PipelineMode.REALISTIC)

    outer_energy_nj = []
    outer_area_um2 = []
    inner_energy_nj = []
    inner_area_um2 = []
    inner_labels = []
    inner_groups = []

    for group, names in MODULE_GROUPINGS.items():
        outer_energy_nj.append(sum(metrics["energy_breakdown"][name] for name in names))
        outer_area_um2.append(sum(model.modules[name].total_area_um2 for name in names))
        for name in names:
            inner_energy_nj.append(metrics["energy_breakdown"][name])
            inner_area_um2.append(model.modules[name].total_area_um2)
            inner_labels.append(name)
            inner_groups.append(group)

    return {
        "outer_labels": list(MODULE_GROUPINGS),
        "outer_energy_nj": outer_energy_nj,
        "outer_area_um2": outer_area_um2,
        "inner_energy_nj": inner_energy_nj,
        "inner_area_um2": inner_area_um2,
        "inner_labels": inner_labels,
        "inner_groups": inner_groups,
        "onchip_energy_nj": metrics["energy_nj"]["onchip"],
        "onchip_area_um2": metrics["area_mm2"] * 1e6,
    }


def generate_figure(
    output_path: Path | str = DEFAULT_OUTPUT_DIR / "fig08_area_energy_breakdown.png",
) -> Path:
    """Render Figure 8 to a PNG without opening an interactive window."""
    apply_paper_style()
    data = compute_breakdowns()
    outer_labels = data["outer_labels"]

    outer_colors = [COLOR_CONFIG[group]["start"] for group in outer_labels]
    inner_colors = []
    for group, names in MODULE_GROUPINGS.items():
        inner_colors.extend(
            _gradient_color(group, index, len(names)) for index in range(len(names))
        )

    fig, (energy_ax, area_ax) = plt.subplots(
        1,
        2,
        figsize=(8, 4),
        subplot_kw={"aspect": "equal"},
    )
    fig.subplots_adjust(wspace=0, top=1, bottom=0, left=0, right=1)

    ring_width = 0.5

    def percentage_label(percent: float) -> str:
        return f"{percent:.0f}%" if percent > 2 else ""

    energy_ax.pie(
        data["outer_energy_nj"],
        radius=1 - ring_width,
        colors=outer_colors,
        textprops={"size": 16},
        wedgeprops={"width": ring_width, "edgecolor": "w"},
        autopct=percentage_label,
        pctdistance=0.5,
        startangle=0,
    )
    energy_ax.pie(
        data["inner_energy_nj"],
        radius=1,
        colors=inner_colors,
        textprops={"size": 16},
        wedgeprops={"width": ring_width, "edgecolor": "w"},
        autopct=percentage_label,
        pctdistance=1.1,
        startangle=0,
    )
    energy_ax.set_title("Energy", fontsize=20)

    area_ax.pie(
        data["outer_area_um2"],
        radius=1 - ring_width,
        colors=outer_colors,
        textprops={"size": 16},
        wedgeprops={"width": ring_width, "edgecolor": "w"},
        autopct=percentage_label,
        pctdistance=0.5,
    )
    area_ax.pie(
        data["inner_area_um2"],
        radius=1,
        colors=inner_colors,
        textprops={"size": 16},
        wedgeprops={"width": ring_width, "edgecolor": "w"},
        autopct=percentage_label,
        pctdistance=1.1,
    )
    area_ax.set_title("Area", fontsize=20)

    legend_patches = [
        Patch(facecolor=outer_colors[index], label=label)
        for index, label in enumerate(outer_labels)
    ]
    legend_patches.extend(
        Patch(facecolor=inner_colors[index], label=label)
        for index, label in enumerate(data["inner_labels"])
    )
    area_ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        fontsize=14,
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
        default=DEFAULT_OUTPUT_DIR / "fig08_area_energy_breakdown.png",
    )
    args = parser.parse_args(argv)
    print(f"Generated {generate_figure(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
