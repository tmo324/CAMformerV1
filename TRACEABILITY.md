# Traceability Ledger

This ledger guarantees that every number in the CAMformer simulator has a known
provenance, preventing magic numbers and silent desyncs.

## Technology provenance

Digital blocks are synthesized in Verilog RTL with **Synopsys Design Compiler at
TSMC 65 nm**. All reported PPA is then **normalized to 45 nm** using the scaling
methodology cited as **[36]** in the paper, which introduces ~±15% uncertainty.
A few SRAM/ADC macros are characterized near 40 nm and used unscaled (within that
uncertainty band). This matches the paper's statement that area/power estimates
use "45 nm-scaled Synopsys Design Compiler synthesis (digital) and HSPICE
characterization (analog)," with <5% deviation from published, 45 nm-normalized
numbers.

The constants below live in `src/camformer/core/paper_hardware.py` (`PAPER_MODULES`).

## PPA Sources (normalized to 45 nm)

| Component | Metric | Value | Source | Status | Notes |
|---|---|---|---|---|---|
| BA-CAM (16x64) | Area | 0.0003328 mm² | DC synth @ TSMC 65 nm → 45 nm (`rtl/a12_sparseMM.sv`) | Calibrated | Scaled linearly by rows/16 in code |
| BA-CAM (16x64) | Power | 0.1706 mW | DC synth @ TSMC 65 nm → 45 nm | Calibrated | At 1 GHz reference frequency |
| 6b ADC | Power | 0.0520 mW | NeuroSim ADC model (~40 nm, used unscaled) | Provisional | Citation to be finalized against paper ref list |
| BF16 MAC | Area | 0.0003310 mm² | DC synth @ TSMC 65 nm → 45 nm (`rtl/a13_fp32accum.sv`) | Calibrated | |

*Calibration check:* the values above reproduce the paper's headline metrics
(721 cycles, 54.9 nJ, 0.258 mm², 191 att/ms) to <5% via
`python experiments/compare_with_paper.py`.
