# Reproducing Paper Results

This document provides step-by-step instructions to reproduce the key hardware validation and sensitivity studies presented in the CAMformer paper (arXiv:2511.19740v1).

## 0. Prerequisites

Install the package first so the scripts can import `camformer`:

```bash
make install        # or: pip install -e .
```

## 1. Hardware Model Validation

To validate the cycles, energy, area, and throughput metrics against the reference hardware model:

```bash
python experiments/compare_with_paper.py
```

**Expected Output:**
The script will output the stage-by-stage cycle breakdown and component-wise energy breakdowns. You should see a `<5%` tolerance match with the published targets:
- Total Cycles: ~721
- Energy: ~54.9 nJ
- Area: ~0.26 mm²
- Throughput: ~191 att/ms

## 2. Sensitivity Study

To reproduce the Top-$k$ sparsity and tile size sweep tables:

```bash
python experiments/sensitivity_study.py
```

**Expected Output:**
The script will:
1. Print the analytical tables to the console showing Throughput, Energy Efficiency, Area, and Expected Recall for different Top-$k$ and Tile values.
2. Generate `sensitivity_study.svg` in the current directory.

## 3. Using Make

If you have installed the project via `make install`, you can generate the
architectural tables (Tables III/IV/VI, headline metrics) with a single command:

```bash
make paper
```

## 4. Paper Figures (Fig 6/8/10)

The figure plotters live in `experiments/notebooks/` and need extra dependencies
(pandas, openpyxl, a Jupyter kernel). Install them and regenerate the figures with:

```bash
make install-figures   # pandas, openpyxl, nbconvert, ipykernel
make figures           # executes the plotter notebooks headless
```

This regenerates, into `experiments/notebooks/`:
- `bimm_energy.png` — Fig 6 (per-operation energy vs. batch size)
- `area_energy_breakdown.png` — Fig 8 (area + energy breakdown)
- `pareto_front.png` — Fig 10 (Pareto vs. industry baselines)

**Scope note.** Fig 10's GPU baselines (A100 / L40 / TitanXP) are *committed
profiling measurements* under `benchmarks/`, not regenerated from hardware.
Two paper results are **not** reproducible from this repository and live in
separate pipelines: the algorithmic accuracy of Table V (ImageNet/ViT + language)
and the analog circuit characterization of Table II / Fig 3 (HSPICE).
