# CAMformerV1 Simulator

This repository contains the event-driven architectural simulator for **CAMformer**, an energy-efficient sparse attention accelerator utilizing Content-Addressable Memory (CAM).

## Overview
CAMformer accelerates the multi-head attention mechanism by performing highly parallel associative searches in hardware, bypassing the need to fetch and compute full $Q \times K^T$ similarity matrices. This repository provides:
- A cycle-accurate event-driven simulation pipeline (SST).
- Analytical hardware models for area, power, and latency.
- Scripts to reproduce the key hardware validation and sensitivity studies presented in our work.

## Installation

We recommend using a virtual environment (e.g. `conda` or `venv`):
```bash
# Clone the repository
git clone https://github.com/tmo324/CAMformerV1.git
cd CAMformerV1

# Install requirements
pip install -r requirements.txt
```

## Running the Simulator

### 1. Validate Paper Results
To run the full simulation and validate the metrics (Cycles, Energy, Area, Throughput) against the reference hardware model:
```bash
python compare_with_paper.py
```
This script will output the stage-by-stage cycle breakdown and component-wise energy breakdowns, confirming a <5% tolerance match with the published targets.

### 2. Sensitivity Study
To reproduce the Top-$k$ sparsity and tile size sweep table:
```bash
python sensitivity_study.py
```
*Note: This script will generate `sensitivity_study.svg` and print the analytical tables to the console.*

### 3. Run Tests
The repository includes standard test suites for both the analytical models and the SST event-driven wrappers:
```bash
python test_camformer.py
python test_sst.py
```

## Citation

If you find this code useful in your research, please cite our paper:
```bibtex
@article{molomochir2025camformer,
  title={CAMformer: Energy-Efficient Sparse Attention Accelerator Using Content-Addressable Memory},
  author={Molom-Ochir, Tergel and others},
  journal={arXiv preprint arXiv:2511.19740},
  year={2025}
}
```

## License
MIT License
