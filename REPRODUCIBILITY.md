# Reproducing the CAMformer Results

This guide describes what CAMformerV1 v1.0.0 reproduces from the published paper and what remains outside the release.

## Requirements

- Python 3.10, 3.11, or 3.12
- NumPy and Matplotlib, installed by the package
- pytest for the regression suite
- Icarus Verilog and Verilator only for RTL checks

Create an isolated environment and install the test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install-test
```

## Fast Reproduction

```bash
make check
make paper
```

`make check` compiles the Python sources and runs the regression suite. `make paper` runs model validation, regenerates the sensitivity analysis, and renders Figures 6, 8, and 10.

## Published Operating Point

Run either command:

```bash
camformer-validate
python experiments/compare_with_paper.py
```

Both the analytical hardware model and the Python event model are compared against the same published targets:

| Metric | Target | Expected release result |
|---|---:|---:|
| Total model cycles | 721 | 721 |
| Total modeled energy | 54.92 nJ | about 54.92 nJ |
| Single-core MHA throughput | 191.13 att/ms | about 191.13 att/ms |
| On-chip area | 0.258414 mm² | about 0.258414 mm² |
| Total average power | 0.17 W | about 0.168 W |
| Energy efficiency | 9,045 qry/mJ | about 9,104 qry/mJ |

The command exits unsuccessfully if either model differs from any target by more than 5%.

## Sensitivity Analysis

```bash
camformer-sweep
python experiments/sensitivity_study.py
```

The default sweep matches the camera-ready Table IV parameter sets:

- Top-k: 8, 16, 24, and 32, with fixed Top-32 hardware area
- Tile size: 4, 8, 16, and 32, with final top-k fixed at 32

It prints both tables and writes [`results/figures/sensitivity_study.svg`](results/figures/sensitivity_study.svg). The asterisk marks the default configuration.

## Paper Figures

Generate all code-backed paper plots:

```bash
make figures
```

Or run them individually:

```bash
python -m experiments.figures.fig06_bimm_energy
python -m experiments.figures.fig08_area_energy_breakdown
python -m experiments.figures.fig10_pareto_front
```

The scripts use a headless Matplotlib backend and require no notebook, browser, or display. Matplotlib is pinned to the version used for the release. The numeric plot inputs are regression-tested. PNG dimensions can vary by a few pixels across operating systems because installed font metrics and bounding-box calculations differ, so cross-platform outputs are visually and numerically reproducible rather than guaranteed byte-identical.

## Complete Paper Artifact Inventory

| Published item | Status in this release | Evidence or boundary |
|---|---|---|
| Figure 1, attention as associative memory | Source graphic | Conceptual author artwork, no plotted data |
| Table I, related approaches | Not generated | Manually curated literature comparison |
| Table II, attention-score circuits | Not generated | Literature values plus BA-CAM circuit characterization |
| Figure 2, BA-CAM array | Source graphic | Circuit and architecture artwork |
| Figure 3, BA-CAM characterization | Not generated | Requires the original HSPICE and foundry-PDK flow |
| Figure 4, system architecture | Source graphic | Architecture artwork |
| Figure 5, binary VMM and tiling | Source graphic | Architecture artwork |
| Figure 6, per-operation BA-CAM energy | Reproduced | `fig06_bimm_energy.py` and `hardware_modules.csv` |
| Figure 7, pipelining strategies | Source graphic | Architecture artwork |
| Figure 8, area and energy breakdown | Reproduced | `fig08_area_energy_breakdown.py` and validated hardware model |
| Table III, accelerator comparison | Partially reproduced | CAMformer rows are validated; external baseline rows are cited literature or profiling values |
| Figure 9, stage throughput DSE | Partially reproduced | Default balanced point is validated; the author-composed multi-configuration graphic is not regenerated |
| Table IV, sensitivity analysis | Reproduced | `camformer-sweep` and regression tests |
| Figure 10, Pareto analysis | Reproduced | `fig10_pareto_front.py`, validated CAMformer point, and cited reference specifications |
| Table V, algorithmic accuracy | Not generated | Requires the separate HAD training and evaluation pipeline and datasets |
| Table VI, latency breakdown | Partially reproduced | Aggregate timing and throughput are validated; the published per-stage nanosecond presentation is not regenerated |

“Source graphic” means the item is intentionally drawn or composed rather than produced from a numeric plotting program. Paper source files and proprietary foundry inputs are not included in this software repository.

## Data Inputs

The canonical machine-readable hardware input is [`data/inputs/hardware_modules.csv`](data/inputs/hardware_modules.csv). Archival workbooks under `data/reference/` preserve source calculations and comparisons but are not read by the default reproduction commands. See [data/README.md](data/README.md) and [TRACEABILITY.md](TRACEABILITY.md).

The files under `benchmarks/` preserve GPU profiling summaries and raw text logs. They are supplementary records and are not consumed by `make paper`.

## RTL Checks

After installing Icarus Verilog and Verilator:

```bash
make rtl
```

This performs portable syntax checks, two behavioral smoke tests, and Verilator lint. The exact scope and known limitations are listed in [rtl/README.md](rtl/README.md).

Run every local release gate with:

```bash
make release
```

## Installed-Wheel Check

To verify that the console commands do not depend on an editable checkout:

```bash
python -m pip install build
python -m build --wheel
python -m pip install --force-reinstall dist/camformer-1.0.0-py3-none-any.whl
cd /tmp
camformer-validate
camformer-sweep --no-plot
```

The CI package job performs the same check in a clean runner.
