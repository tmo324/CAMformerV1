#!/usr/bin/env python3
"""
CAMformer Sensitivity Study
============================
Sweeps top-k sparsity and tile size, reporting Throughput, Energy Efficiency,
Area, and E[Recall] for the sensitivity analysis table (arXiv:2511.19740v1).

Usage:
    python sensitivity_study.py [--no-plot] [--output-dir PATH]
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.paper_hardware import (
    PaperHardwareModel, PaperConfig, PipelineMode, PAPER_MODULES, HardwareModule
)


# =============================================================================
# Helpers
# =============================================================================

def build_tile_modules(cam_rows: int) -> dict:
    """
    Return a copy of PAPER_MODULES with BA-CAM, ADCs, and Fixed Scalers
    scaled for a non-default tile size.

    BA-CAM power/area scale linearly with cam_rows/16.
    ADC and Fixed Scaler count = cam_rows.
    """
    scale = cam_rows / 16.0
    modules = {}
    for name, mod in PAPER_MODULES.items():
        if name == "BA-CAM 16X64":
            modules[name] = HardwareModule(
                name=mod.name,
                power_mw=mod.power_mw * scale,
                area_um2=mod.area_um2 * scale,
                delay_cycles=mod.delay_cycles,
                count=mod.count,
                power2_mw=mod.power2_mw * scale if mod.power2_mw else None,
                delay2_cycles=mod.delay2_cycles,
            )
        elif name == "6b ADCs":
            modules[name] = HardwareModule(
                name=mod.name,
                power_mw=mod.power_mw,
                area_um2=mod.area_um2,
                delay_cycles=mod.delay_cycles,
                count=cam_rows,
            )
        elif name == "Fixed Scalers":
            modules[name] = HardwareModule(
                name=mod.name,
                power_mw=mod.power_mw,
                area_um2=mod.area_um2,
                delay_cycles=mod.delay_cycles,
                count=cam_rows,
            )
        else:
            modules[name] = mod
    return modules


class TileSweepModel(PaperHardwareModel):
    """
    Subclass that generalises the intermediate Top-32 pruning rule
    so it works for any tile count (not just num_tiles=64).
    """

    def run_association(self, pipelined: bool = True) -> int:
        if not pipelined:
            return super().run_association(pipelined=False)

        starting_cycles = self.total_cycles
        cfg = self.config

        # Generalised pruning schedule: 2 intermediate prunings
        prune_start = cfg.num_tiles // 2
        prune_step = max(1, cfg.num_tiles // 4)

        self._run_module("Query Buffer", 1)

        for tile in range(cfg.num_tiles):
            for _ in range(cfg.cam_rows // cfg.rows_per_program):
                self._run_module("Key SRAM", 1)
                self._run_module("BA-CAM 16X64", 1, use_secondary=True)
                self._add_cycles("BA-CAM 16X64",
                                 self.modules["BA-CAM 16X64"].delay2_cycles)

            self._run_module("Query Buffer", 1)
            self._run_module("BA-CAM 16X64", 1)
            self._add_cycles("BA-CAM 16X64",
                             self.modules["BA-CAM 16X64"].delay_cycles)

            self._run_module("6b ADCs", 1)
            self._run_module("Fixed Scalers", 1)
            self._run_module("Top2", 1)
            self._run_module("Ptop Reg", 1)

            if tile >= prune_start and tile % prune_step == 0:
                self._run_module("Top32", 1)
                self._run_module("Ptop Reg", 1)

            self._run_module("DRAM", 1)
            self._run_module("Value SRAM", 1)

        # Pipeline drain
        self._add_cycles("6b ADCs", self.modules["6b ADCs"].delay_cycles)
        self._add_cycles("Fixed Scalers", self.modules["Fixed Scalers"].delay_cycles)
        self._add_cycles("Top2", self.modules["Top2"].delay_cycles)
        self._add_cycles("Ptop Reg", self.modules["Ptop Reg"].delay_cycles)

        self.association_cycles = self.total_cycles - starting_cycles
        return self.association_cycles


def expected_recall(n: int, T: int, k1: int, k2: int) -> float:
    """
    Hypergeometric E[Recall] (%).

    n  = sequence length (1024)
    T  = tile size (cam_rows)
    k1 = per-tile top-k hardware (2 for Top2)
    k2 = final top-k
    """
    num_tiles = n // T

    # P(X=j) where X ~ Hypergeometric(N=n, K=k2, n=T)
    def hyper_pmf(j):
        if j < 0 or j > min(T, k2) or j > T or (T - j) > (n - k2):
            return 0.0
        return (math.comb(k2, j) * math.comb(n - k2, T - j)) / math.comb(n, T)

    # E[captured per tile] = sum_j min(j, k1) * P(X=j)
    e_captured = 0.0
    for j in range(min(T, k2) + 1):
        e_captured += min(j, k1) * hyper_pmf(j)

    # Total expected captured = num_tiles * E[captured_per_tile]
    # Recall = E[total_captured] / k2
    recall = num_tiles * e_captured / k2 * 100.0
    return min(recall, 100.0)


# =============================================================================
# Sweep runners
# =============================================================================

def score_buffer_area_um2(k: int) -> float:
    """
    Score buffer SRAM area that scales with top-k.

    Storage = k × 512 bytes (scores + indices across tiles).
    Uses 45nm 6T SRAM: 0.100 µm²/cell + 30% peripheral overhead.
    """
    bits = k * 512 * 8  # k * 0.5 KB in bits
    return bits * 0.100 * 1.3  # 45nm SRAM cell + overhead


def run_topk_sweep(k_values=(8, 16, 32, 64)):
    """Sweep top-k with default tile size (16)."""
    rows = []
    for k in k_values:
        cfg = PaperConfig(k_value=k)
        model = PaperHardwareModel(config=cfg)
        metrics = model.run_attention(PipelineMode.REALISTIC)

        throughput = metrics["throughput"]["single_core_per_ms"]
        efficiency = 1.0 / (2 * model.get_total_energy_nj() / 1e6)
        # Adjust area for score buffer SRAM scaling with k
        # Base area (0.258 mm²) already includes buffer for k=32
        area = metrics["area_mm2"] + (score_buffer_area_um2(k) - score_buffer_area_um2(32)) / 1e6
        recall = expected_recall(cfg.seq_length, cfg.cam_rows, 2, k)

        rows.append({
            "param": k,
            "throughput": throughput,
            "efficiency": efficiency,
            "area": area,
            "recall": recall,
        })
    return rows


def run_tile_sweep(tile_values=(4, 8, 16, 32)):
    """Sweep tile size with default top-k (32)."""
    rows = []
    for tile in tile_values:
        num_tiles = 1024 // tile
        cfg = PaperConfig(cam_rows=tile, num_tiles=num_tiles)
        modules = build_tile_modules(tile)

        # Use TileSweepModel for generalised pruning
        model = TileSweepModel(config=cfg, modules=modules)
        metrics = model.run_attention(PipelineMode.REALISTIC)

        throughput = metrics["throughput"]["single_core_per_ms"]
        efficiency = 1.0 / (2 * model.get_total_energy_nj() / 1e6)
        area = metrics["area_mm2"]
        recall = expected_recall(cfg.seq_length, tile, 2, cfg.k_value)

        rows.append({
            "param": tile,
            "throughput": throughput,
            "efficiency": efficiency,
            "area": area,
            "recall": recall,
        })
    return rows


# =============================================================================
# Table printer
# =============================================================================

def print_table(title, param_label, rows, default_val):
    """Print a formatted ASCII table."""
    print(f"\n{title}")
    print(f"{'─' * 72}")
    hdr = f"  {param_label:<10} {'Throughput':>12} {'Energy Eff.':>14} {'Area':>10} {'E[Recall]':>10}"
    print(hdr)
    units = f"  {'':<10} {'(qry/ms)':>12} {'(qry/mJ)':>14} {'(mm²)':>10} {'(%)':>10}"
    print(units)
    print(f"  {'─' * 60}")
    for r in rows:
        star = "*" if r["param"] == default_val else " "
        print(f"  {r['param']:<4}{star:<6} {r['throughput']:>12.1f} {r['efficiency']:>14,.0f} "
              f"{r['area']:>10.3f} {r['recall']:>9.1f}")
    print()


# =============================================================================
# SVG figure
# =============================================================================

def generate_svg(topk_rows, tile_rows, output_dir):
    """Generate sensitivity_study.svg with two subplots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping SVG generation.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#f7fcff")

    for ax in axes:
        ax.set_facecolor("#f7fcff")
        for spine in ax.spines.values():
            spine.set_color("#cccccc")

    # Colours
    c_thr = "#6C8EBF"
    c_eff = "#D4A76A"
    c_rec = "#82B366"

    def _plot_panel(ax, rows, xlabel, default_val):
        params = [r["param"] for r in rows]
        throughputs = [r["throughput"] for r in rows]
        efficiencies = [r["efficiency"] for r in rows]
        recalls = [r["recall"] for r in rows]

        x = range(len(params))
        ax.set_xticks(x)
        labels = [f"{p}*" if p == default_val else str(p) for p in params]
        ax.set_xticklabels(labels, fontfamily="serif")
        ax.set_xlabel(xlabel, fontfamily="serif")

        # Throughput bars
        ax.bar(x, throughputs, width=0.35, color=c_thr, alpha=0.8, label="Throughput (qry/ms)")
        ax.set_ylabel("Throughput (qry/ms)", fontfamily="serif", color=c_thr)
        ax.tick_params(axis="y", labelcolor=c_thr)

        # Energy efficiency on twin axis
        ax2 = ax.twinx()
        ax2.plot(x, efficiencies, "o-", color=c_eff, linewidth=2, label="Energy Eff. (qry/mJ)")
        ax2.set_ylabel("Energy Eff. (qry/mJ)", fontfamily="serif", color=c_eff)
        ax2.tick_params(axis="y", labelcolor=c_eff)

        # Annotate recall
        for i, rec in enumerate(recalls):
            ax.annotate(f"{rec:.1f}%", (i, throughputs[i]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, fontfamily="serif", color=c_rec)

    _plot_panel(axes[0], topk_rows, "Top-k Sparsity", 32)
    axes[0].set_title("(a) Top-k Sweep (Tile = 16)", fontfamily="serif", fontsize=11)

    _plot_panel(axes[1], tile_rows, "Tile Size", 16)
    axes[1].set_title("(b) Tile Sweep (Top-k = 32)", fontfamily="serif", fontsize=11)

    fig.tight_layout()
    path = os.path.join(output_dir, "sensitivity_study.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CAMformer sensitivity study")
    parser.add_argument("--no-plot", action="store_true", help="Skip SVG generation")
    parser.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory for SVG output")
    args = parser.parse_args()

    print("=" * 72)
    print("CAMformer Sensitivity Study (arXiv:2511.19740v1)")
    print("=" * 72)

    topk_rows = run_topk_sweep()
    tile_rows = run_tile_sweep()

    print_table("(a) Top-k Sparsity Sweep  (Tile = 16 fixed)",
                "Top-k", topk_rows, default_val=32)
    print_table("(b) Tile Size Sweep  (Top-k = 32 fixed)",
                "Tile", tile_rows, default_val=16)

    if not args.no_plot:
        generate_svg(topk_rows, tile_rows, args.output_dir)


if __name__ == "__main__":
    main()
