# Reproducing Paper Results

This document provides step-by-step instructions to reproduce the key hardware validation and sensitivity studies presented in the CAMformer paper (arXiv:2511.19740v1).

## 1. Hardware Model Validation

To validate the cycles, energy, area, and throughput metrics against the reference hardware model:

```bash
python compare_with_paper.py
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
python sensitivity_study.py
```

**Expected Output:**
The script will:
1. Print the analytical tables to the console showing Throughput, Energy Efficiency, Area, and Expected Recall for different Top-$k$ and Tile values.
2. Generate `sensitivity_study.svg` in the current directory.

## 3. Using Make

If you have installed the project via `make install`, you can generate all paper results with a single command:

```bash
make paper
```
