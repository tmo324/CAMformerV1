<h1 align="center">CAMformerV1</h1>

<p align="center">
  <strong>Energy-Efficient Sparse Attention Accelerator Using Content-Addressable Memory</strong>
</p>

<p align="center">
  <a href="#overview">Architecture</a> ·
  <a href="#installation">Quick Start</a> ·
  <a href="REPRODUCIBILITY.md">Reproduce Results</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <a href="https://github.com/tmo324/CAMformerV1/actions/workflows/ci.yml">
    <img src="https://github.com/tmo324/CAMformerV1/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/tmo324/CAMformerV1/actions/workflows/secret-scan.yml">
    <img src="https://github.com/tmo324/CAMformerV1/actions/workflows/secret-scan.yml/badge.svg" alt="Secret scan">
  </a>
  <img src="https://img.shields.io/badge/python-3.8--3.12-3776AB?logo=python&logoColor=white" alt="Python Supported">
  <a href="CITATION.cff">
    <img src="https://img.shields.io/badge/citation-CFF-4B8BBE" alt="Citation CFF">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License">
  </a>
</p>

---

This repository contains the event-driven architectural simulator for **CAMformer**.

## Overview
CAMformer accelerates the multi-head attention mechanism by performing highly parallel associative searches in hardware, bypassing the need to fetch and compute full $Q \times K^T$ similarity matrices. This repository provides:
- A cycle-accurate event-driven simulation pipeline (SST).
- Analytical hardware models for area, power, and latency.
- Scripts to reproduce the key hardware validation and sensitivity studies presented in our work.

## Installation

We provide a modern Python package structure (`pyproject.toml`) for standard installation. We recommend using a virtual environment (e.g., `conda` or `venv`):

```bash
# Clone the repository
git clone https://github.com/tmo324/CAMformerV1.git
cd CAMformerV1

# Install the package and dependencies
make install

# To install with testing dependencies:
make install-test
```

## Running the Simulator

### Tests
We include standard test suites for both the analytical models and the SST event-driven wrappers. To run all tests:
```bash
make test
```

### Reproducing Paper Results
For detailed, step-by-step instructions on validating the paper's target metrics and generating the sensitivity study figures, please refer to the dedicated [REPRODUCIBILITY.md](REPRODUCIBILITY.md) guide. You can generate all paper results automatically with:
```bash
make paper
```

## Governance & Community
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Third-Party Notices](THIRD_PARTY_NOTICES.md)

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
