# Traceability Ledger

This ledger guarantees that every number in the CAMformer simulator has a known provenance, preventing magic numbers and silent desyncs.

## PPA Sources (28nm)

| Component | Metric | Value | Source | Status | Notes |
|---|---|---|---|---|---|
| BA-CAM (16x64) | Area | 0.0003328 mm² | RTL Synthesis (`rtl/a12_sparseMM.sv`) | Calibrated | Scaled linearly by rows/16 in code |
| BA-CAM (16x64) | Power | 0.1706 mW | RTL Synthesis | Calibrated | At reference frequency |
| 6b ADC | Power | 0.0520 mW | Reference [X] / NeuroSim | Provisional | Needs citation update |
| BF16 MAC | Area | 0.0003310 mm² | RTL Synthesis (`rtl/a13_fp32accum.sv`) | Calibrated | |

*Note: This ledger is currently being backfilled from the hardcoded values in `src/camformer/core/paper_hardware.py`.*
