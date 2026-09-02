<h1 align="center">CAMformerV1</h1>

<p align="center">
  <strong>Binary Associative Memory Is All You Need</strong>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#reproduce-the-paper-results">Reproduce Results</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <a href="https://github.com/tmo324/CAMformerV1/actions/workflows/ci.yml">
    <img src="https://github.com/tmo324/CAMformerV1/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/tmo324/CAMformerV1/actions/workflows/secret-scan.yml">
    <img src="https://github.com/tmo324/CAMformerV1/actions/workflows/secret-scan.yml/badge.svg" alt="Secret scan">
  </a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-3776AB?logo=python&logoColor=white" alt="Python 3.10 through 3.12">
  <a href="CITATION.cff">
    <img src="https://img.shields.io/badge/citation-CFF-4B8BBE" alt="Citation CFF">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License">
  </a>
</p>

---

This repository contains the open research artifacts for **CAMformer**, including its analytical hardware model, Python discrete-event simulator, paper-figure scripts, source data, and reference SystemVerilog blocks.

> **Release status:** Version 1.0.0 is a reproducibility-oriented research release. It reproduces the published CAMformer operating point within a 5% tolerance and regenerates Figures 6, 8, and 10 without Jupyter. The RTL is reference RTL, not a complete tape-out-ready implementation.

## Paper

This repository accompanies [*CAMformer: Binary Associative Memory Is All You Need*](https://doi.org/10.1109/TCSI.2026.3692014), published in *IEEE Transactions on Circuits and Systems I: Regular Papers* in 2026.

For the default BERT-Large attention workload, the paper reports:

| Metric | Published value | Release-model value |
|---|---:|---:|
| Throughput | 191 qry/ms | 191.13 qry/ms |
| Energy efficiency | 9,045 qry/mJ | 9,103.80 qry/mJ |
| Area | 0.258 mm² | 0.258414 mm² |
| Power | 0.17 W | 0.168 W |

The small energy-efficiency difference is 0.65% and falls within the paper's stated model tolerance.

## Architecture

CAMformer replaces dense query-key arithmetic with content-addressable search. A binary query is compared against stored binary keys in a voltage-domain Binary Attention CAM (BA-CAM), then a hierarchical top-k path retains the most relevant scores before BF16 contextualization.

The modeled datapath has three stages:

1. **Association:** 16×64 BA-CAM tiles compute Hamming similarity and local Top-2 selection.
2. **Normalization:** a Top-32 selector and sparse softmax normalize the retained scores.
3. **Contextualization:** eight BF16 MAC units combine the selected values.

The simulator is implemented in Python with SST-style components, links, events, clocks, and statistics. It does not require Sandia SST.

## Quick Start

```bash
git clone https://github.com/tmo324/CAMformerV1.git
cd CAMformerV1
python3 -m venv .venv
source .venv/bin/activate
make install-test
make check
```

Python 3.10, 3.11, and 3.12 are tested in CI.

## Reproduce the Paper Results

Run the complete Python reproduction path:

```bash
make paper
```

This validates the analytical and event-driven models, prints the published sensitivity sweep, and regenerates the paper plots under [`results/figures/`](results/figures/).

| Paper artifact | Reproduction entry point | Output |
|---|---|---|
| Figure 6, BA-CAM energy | `python -m experiments.figures.fig06_bimm_energy` | `fig06_bimm_energy.png` |
| Figure 8, area and energy breakdown | `python -m experiments.figures.fig08_area_energy_breakdown` | `fig08_area_energy_breakdown.png` |
| Figure 10, Pareto comparison | `python -m experiments.figures.fig10_pareto_front` | `fig10_pareto_front.png` |
| Table III CAMformer operating point | `camformer-validate` | terminal comparison |
| Table IV sensitivity analysis | `camformer-sweep` | terminal table and `sensitivity_study.svg` |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete artifact inventory, expected values, clean-wheel verification, and the boundaries of this release. Numerical and provenance details are recorded in [TRACEABILITY.md](TRACEABILITY.md).

## RTL Reference Artifacts

The [`rtl/`](rtl/) directory contains research RTL for key CAMformer blocks. Two modules have behavioral smoke tests, while the remaining listed modules receive syntax or lint checks. Incomplete top-k prototypes are isolated under `rtl/experimental/unsupported/`.

With Icarus Verilog and Verilator installed:

```bash
make rtl
```

See [rtl/README.md](rtl/README.md) before reusing these blocks.

## Repository Layout

```text
CAMformerV1/
├── src/camformer/          Python simulator and analytical model
├── experiments/figures/    Script-based paper figure generation
├── data/                   Canonical inputs and archival workbooks
├── benchmarks/             Archival GPU profiling records
├── results/figures/        Committed regenerated figures
├── rtl/                    Reference RTL, checks, and testbenches
└── tests/                  Model, CLI, figure, and regression tests
```

## Community and Governance

- [Contributing guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Authors and contributors](AUTHORS.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Citation

If this repository supports your work, cite the paper:

```bibtex
@article{molomochir2026camformer,
  author  = {Molom-Ochir, Tergel and Morris, Benjamin F. and Horton, Mark and Wei, Chiyue and Guo, Cong and Taylor, Brady and Liu, Peter and Wang, Shan X. and Fan, Deliang and Li, Hai and Chen, Yiran},
  title   = {CAMformer: Binary Associative Memory Is All You Need},
  journal = {IEEE Transactions on Circuits and Systems I: Regular Papers},
  year    = {2026},
  doi     = {10.1109/TCSI.2026.3692014}
}
```

GitHub can also generate citation formats from [CITATION.cff](CITATION.cff).

## License

The original code, RTL, data, and generated figures in this repository are released under the [MIT License](LICENSE). External tools and cited source publications retain their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
