# Data

This directory separates the canonical machine-readable input from archival research workbooks.

## `inputs/`

`hardware_modules.csv` is the canonical unscaled module table used by the Figure 6 plotter. Each row records delay, power, area, source class, technology node, instance count, and notes.

The validated 45 nm model values are intentionally defined in `src/camformer/core/paper_hardware.py`, where every scaling step is visible and regression-tested.

## `reference/`

- `hardware/`: archived CAM, multiplier, and synthesis calculation workbooks
- `models/`: archived model-dimension tables
- `comparisons/`: archived attention-hardware and SpAtten comparison workbooks

These workbooks preserve research provenance. They are not parsed by the default `make paper` workflow, so editing one does not silently change a released result.

See [../TRACEABILITY.md](../TRACEABILITY.md) for source citations and transformations.
