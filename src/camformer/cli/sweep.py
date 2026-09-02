"""Run the CAMformer top-k and tile-size sensitivity sweeps."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from camformer.core.paper_hardware import (
    HardwareModule,
    PAPER_MODULES,
    PaperConfig,
    PaperHardwareModel,
    PipelineMode,
    published_energy_efficiency,
)


def build_tile_modules(cam_rows: int) -> dict[str, HardwareModule]:
    """Scale the tile-dependent modules for a non-default CAM row count."""
    scale = cam_rows / 16.0
    modules: dict[str, HardwareModule] = {}
    for name, module in PAPER_MODULES.items():
        if name == "BA-CAM 16X64":
            modules[name] = HardwareModule(
                name=module.name,
                power_mw=module.power_mw * scale,
                area_um2=module.area_um2 * scale,
                delay_cycles=module.delay_cycles,
                count=module.count,
                power2_mw=module.power2_mw * scale if module.power2_mw else None,
                delay2_cycles=module.delay2_cycles,
            )
        elif name in {"6b ADCs", "Fixed Scalers"}:
            modules[name] = HardwareModule(
                name=module.name,
                power_mw=module.power_mw,
                area_um2=module.area_um2,
                delay_cycles=module.delay_cycles,
                count=cam_rows,
            )
        else:
            modules[name] = module
    return modules


class TileSweepModel(PaperHardwareModel):
    """Generalize the intermediate pruning schedule to any tile count."""

    def run_association(self, pipelined: bool = True) -> int:
        if not pipelined:
            return super().run_association(pipelined=False)

        starting_cycles = self.total_cycles
        config = self.config
        prune_start = config.num_tiles // 2
        prune_step = max(1, config.num_tiles // 4)

        self._run_module("Query Buffer", 1)
        for tile in range(config.num_tiles):
            for _ in range(config.cam_rows // config.rows_per_program):
                self._run_module("Key SRAM", 1)
                self._run_module("BA-CAM 16X64", 1, use_secondary=True)
                self._add_cycles(
                    "BA-CAM 16X64", self.modules["BA-CAM 16X64"].delay2_cycles
                )

            self._run_module("Query Buffer", 1)
            self._run_module("BA-CAM 16X64", 1)
            self._add_cycles(
                "BA-CAM 16X64", self.modules["BA-CAM 16X64"].delay_cycles
            )
            for module_name in ("6b ADCs", "Fixed Scalers", "Top2", "Ptop Reg"):
                self._run_module(module_name, 1)

            if tile >= prune_start and tile % prune_step == 0:
                self._run_module("Top32", 1)
                self._run_module("Ptop Reg", 1)

            self._run_module("DRAM", 1)
            self._run_module("Value SRAM", 1)

        for module_name in ("6b ADCs", "Fixed Scalers", "Top2", "Ptop Reg"):
            self._add_cycles(module_name, self.modules[module_name].delay_cycles)

        self.association_cycles = self.total_cycles - starting_cycles
        return self.association_cycles


def expected_recall(n: int, tile_size: int, local_k: int, final_k: int) -> float:
    """Return hypergeometric expected recall as a percentage."""
    num_tiles = n // tile_size

    def hypergeometric_pmf(j: int) -> float:
        if j < 0 or j > min(tile_size, final_k) or tile_size - j > n - final_k:
            return 0.0
        return (
            math.comb(final_k, j)
            * math.comb(n - final_k, tile_size - j)
            / math.comb(n, tile_size)
        )

    captured = sum(
        min(j, local_k) * hypergeometric_pmf(j)
        for j in range(min(tile_size, final_k) + 1)
    )
    return min(num_tiles * captured / final_k * 100.0, 100.0)


def run_topk_sweep(k_values: Sequence[int] = (8, 16, 24, 32)) -> list[dict]:
    """Sweep the camera-ready Table IV values with fixed Top-32 hardware."""
    rows = []
    for k_value in k_values:
        config = PaperConfig(k_value=k_value)
        model = PaperHardwareModel(config=config)
        metrics = model.run_attention(PipelineMode.REALISTIC)
        rows.append(
            {
                "param": k_value,
                "throughput": metrics["throughput"]["single_core_per_ms"],
                "efficiency": published_energy_efficiency(
                    model.get_total_energy_nj()
                ),
                # Runtime k selection reuses the fixed Top-32 datapath, so the
                # physical area remains constant across the published sweep.
                "area": metrics["area_mm2"],
                "recall": expected_recall(
                    config.seq_length, config.cam_rows, 2, k_value
                ),
            }
        )
    return rows


def run_tile_sweep(tile_values: Sequence[int] = (4, 8, 16, 32)) -> list[dict]:
    """Sweep CAM tile size while holding the final top-k at 32."""
    rows = []
    for tile_size in tile_values:
        config = PaperConfig(cam_rows=tile_size, num_tiles=1024 // tile_size)
        model = TileSweepModel(
            config=config,
            modules=build_tile_modules(tile_size),
        )
        metrics = model.run_attention(PipelineMode.REALISTIC)
        rows.append(
            {
                "param": tile_size,
                "throughput": metrics["throughput"]["single_core_per_ms"],
                "efficiency": published_energy_efficiency(
                    model.get_total_energy_nj()
                ),
                "area": metrics["area_mm2"],
                "recall": expected_recall(
                    config.seq_length, tile_size, 2, config.k_value
                ),
            }
        )
    return rows


def print_table(title: str, parameter: str, rows: list[dict], default: int) -> None:
    """Print one sensitivity table."""
    print(f"\n{title}")
    print("-" * 72)
    print(
        f"  {parameter:<10} {'Throughput':>12} {'Energy Eff.':>14} "
        f"{'Area':>10} {'E[Recall]':>10}"
    )
    print(f"  {'':<10} {'(qry/ms)':>12} {'(qry/mJ)':>14} {'(mm^2)':>10} {'(%)':>10}")
    for row in rows:
        marker = "*" if row["param"] == default else " "
        print(
            f"  {row['param']:<4}{marker:<6} {row['throughput']:>12.1f} "
            f"{row['efficiency']:>14,.0f} {row['area']:>10.3f} "
            f"{row['recall']:>9.1f}"
        )


def generate_svg(
    topk_rows: list[dict], tile_rows: list[dict], output_dir: Path
) -> Path:
    """Generate the two-panel sensitivity figure with a headless backend."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "camformer"
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    figure.patch.set_facecolor("#f7fcff")
    colors = {"throughput": "#6C8EBF", "efficiency": "#D4A76A", "recall": "#82B366"}

    def plot_panel(axis, rows: list[dict], xlabel: str, default: int) -> None:
        x_values = range(len(rows))
        parameters = [row["param"] for row in rows]
        throughputs = [row["throughput"] for row in rows]
        efficiencies = [row["efficiency"] for row in rows]
        recalls = [row["recall"] for row in rows]

        axis.set_facecolor("#f7fcff")
        axis.set_xticks(x_values)
        axis.set_xticklabels(
            [f"{value}*" if value == default else str(value) for value in parameters]
        )
        axis.set_xlabel(xlabel)
        axis.bar(x_values, throughputs, width=0.35, color=colors["throughput"], alpha=0.8)
        axis.set_ylabel("Throughput (qry/ms)", color=colors["throughput"])
        axis.tick_params(axis="y", labelcolor=colors["throughput"])

        twin = axis.twinx()
        twin.plot(x_values, efficiencies, "o-", color=colors["efficiency"], linewidth=2)
        twin.set_ylabel("Energy Eff. (qry/mJ)", color=colors["efficiency"])
        twin.tick_params(axis="y", labelcolor=colors["efficiency"])
        for index, recall in enumerate(recalls):
            axis.annotate(
                f"{recall:.1f}%",
                (index, throughputs[index]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=colors["recall"],
            )

    plot_panel(axes[0], topk_rows, "Top-k Sparsity", 32)
    axes[0].set_title("(a) Top-k Sweep (Tile = 16)")
    plot_panel(axes[1], tile_rows, "Tile Size", 16)
    axes[1].set_title("(b) Tile Sweep (Top-k = 32)")
    figure.tight_layout()

    output_path = output_dir / "sensitivity_study.svg"
    figure.savefig(output_path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)

    # Matplotlib emits trailing spaces in multiline SVG path data. Normalize
    # them so regenerated artifacts pass Git's whitespace checks.
    svg_text = output_path.read_text(encoding="utf-8")
    output_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run both sweeps and optionally render their summary figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="Skip SVG generation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Figure output directory (default: results/figures).",
    )
    args = parser.parse_args(argv)

    topk_rows = run_topk_sweep()
    tile_rows = run_tile_sweep()
    print_table("(a) Top-k Sparsity Sweep", "Top-k", topk_rows, default=32)
    print_table("(b) Tile Size Sweep", "Tile", tile_rows, default=16)
    if not args.no_plot:
        print(f"\nSaved: {generate_svg(topk_rows, tile_rows, args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
