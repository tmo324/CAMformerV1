# Traceability Ledger

This ledger records the provenance and transformation of values used by the release hardware model and paper plotters. It is a source map, not a claim that third-party circuits were reimplemented or recharacterized here.

## Canonical Inputs

- [`data/inputs/hardware_modules.csv`](data/inputs/hardware_modules.csv) stores the unscaled module table used by Figure 6 and preserved from the original research workflow.
- [`src/camformer/core/paper_hardware.py`](src/camformer/core/paper_hardware.py) contains the normalized 45 nm parameters used by the analytical and event-driven validation paths.
- [`data/reference/`](data/reference/) contains archival calculation workbooks. These are supporting records, not executable inputs to `make paper`.

## Technology Normalization

The paper characterizes or synthesizes modules at 40 nm, 45 nm, and 65 nm, then reports the accelerator at a common 45 nm basis. The release preserves the original normalization factors:

- 65 nm to 45 nm power or energy: multiply by 0.52
- 65 nm to 45 nm area: multiply by 0.66
- 40 nm reference macros: used without an additional scale factor

The scaling reference is Stillmaker and Baas, “Scaling equations for the accurate prediction of CMOS device performance from 180 nm to 7 nm,” *Integration*, 2017, [doi:10.1016/j.vlsi.2017.02.002](https://doi.org/10.1016/j.vlsi.2017.02.002). The paper treats technology extrapolation as approximate and reports about ±15% uncertainty.

At the model's 1 GHz reference clock, `mW × cycles` equals pJ. Module energy is therefore calculated as:

```text
energy_nJ = power_mW × instance_count × active_cycles / 1000
```

## Hardware Module Sources

| Module | Raw source class | Release transformation | Release value |
|---|---|---|---|
| Query Buffer | 65 nm RTL synthesis | power ×0.52, area ×0.66 | 0.80348 mW, 2,475.0 µm² |
| Key SRAM | 40 nm CACTI estimate | no node scaling | 28.4 mW, 27,728.6 µm² |
| BA-CAM 16×64 | 65 nm HSPICE result for 64×64 | divide by four, then power ×0.52 and area ×0.66 | 13.14783 mW, 2,230.272 µm² |
| 6-bit ADCs | published 40 nm ADC | 16 channels, no node scaling | 0.95 mW and 4,000 µm² per channel |
| Fixed Scalers | 65 nm RTL synthesis | power ×0.52, area ×0.66 | 0.02452 mW and 83.6352 µm² per scaler |
| Top-2 | 65 nm RTL synthesis | power ×0.52, area ×0.66 | 6.08367 mW, 8,801.3377 µm² |
| Potential-top register | 65 nm RTL synthesis | power ×0.52, area ×0.66 | 1.52870 mW, 6,042.3 µm² |
| Top-32 | 65 nm RTL synthesis | power ×0.52, area ×0.66 | 52.07715 mW, 67,016.9293 µm² |
| Softmax LUT | 40 nm CACTI estimate | no node scaling | 1.418 mW, 1,612.3735 µm² |
| Softmax divider | published 45 nm BF16 divider | no node scaling | 0.07109 mW, 441.529 µm² |
| Softmax accumulator | published 65 nm BF16 MAC, used conservatively | power ×0.52, area ×0.66 | 1.11704 mW, 2,879.58 µm² |
| Output Buffer | 65 nm RTL synthesis | power ×0.52, area ×0.66 | 1.55478 mW, 5,990.82 µm² |
| Value SRAM | 40 nm CACTI estimate | no node scaling | 34.7 mW, 44,820.0 µm² |
| Eight BF16 MACs | published 65 nm BF16 MAC | power ×0.52, area ×0.66, eight instances | 1.11704 mW and 2,879.58 µm² per MAC |
| DRAM | published memory-energy estimate | off-chip area excluded; energy represented as power over 24 cycles | 12.42667 mW modeled power |

The source classes above mirror the `Source` column in the canonical CSV and the camera-ready methodology. Exact synthesis, SPICE, and workbook records are preserved under `data/reference/` where available. The proprietary TSMC library and HSPICE deck are not included. The release also does not include a standalone CACTI configuration that can reconstruct the SRAM rows.

## Published Circuit References

- 6-bit SAR ADC: Chen et al., “A 0.95-mW 6-b 700-MS/s single-channel loop-unrolled SAR ADC in 40-nm CMOS,” *IEEE TCAS-II*, 2017, [doi:10.1109/TCSII.2016.2559513](https://doi.org/10.1109/TCSII.2016.2559513).
- BF16 MAC: Tiwari, Trivedi, and Guha, “Design of a Low Power Bfloat16 Pipelined MAC Unit for Deep Neural Network Applications,” *IEEE TENSYMP*, 2021, [doi:10.1109/TENSYMP52854.2021.9550912](https://doi.org/10.1109/TENSYMP52854.2021.9550912).
- BF16 divider: Nagakalyan, Sujatha, and Noor Mahammad, “Energy Efficient Approximate Bfloat-16 Floating Point Division Hardware for Image Processing,” *IEEE ISENSE*, 2024, [doi:10.1109/ISENSE63713.2024.10872306](https://doi.org/10.1109/ISENSE63713.2024.10872306).
- DRAM energy reference: Kawata et al., “Modeling Energy Consumption of Memory Systems,” *IEEE CANDAR*, 2015, [doi:10.1109/CANDAR.2015.31](https://doi.org/10.1109/CANDAR.2015.31).

## Figure 10 Reference Points

Figure 10 uses published performance, power, and die-size specifications for its industry and research points. The numeric constants and transformations are named in `experiments/figures/fig10_pareto_front.py`.

- Cerebras WSE-2: Lie, “Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning,” *IEEE Micro*, 2023, [doi:10.1109/MM.2023.3256384](https://doi.org/10.1109/MM.2023.3256384).
- Google TPU v4: Jouppi et al., “TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings,” *ISCA*, 2023, [doi:10.1145/3579371.3589350](https://doi.org/10.1145/3579371.3589350).
- SpAtten: Wang, Zhang, and Han, “SpAtten: Efficient sparse attention architecture with cascade token and head pruning,” *HPCA*, 2021, [doi:10.1109/HPCA51647.2021.00018](https://doi.org/10.1109/HPCA51647.2021.00018).
- Groq TSP: the original plot notes attribute the specifications to *Microprocessor Report*. The paper cites Abts et al., “Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads,” *ISCA*, 2020, [doi:10.1109/ISCA45697.2020.00023](https://doi.org/10.1109/ISCA45697.2020.00023). The release preserves the plot values and does not claim an independent Groq measurement.

For the projected 45 nm to 22 nm points, area is multiplied by 0.23 and energy by `0.23 / 0.48`, preserving the original paper plot convention.

The SpAtten points also preserve the original plotting notebook exactly. The current point uses 360 effective GOPS divided by the modeled CAMformer area for its horizontal coordinate and 382 GOPS/W for its vertical coordinate. The projected point uses `360 / (CAMformer area × 0.23)` and `360 / (CAMformer power × (0.23 / 0.48))`. These coordinates reproduce the camera-ready figure; they are not an independent reconstruction of SpAtten area and power.

## Energy-Efficiency Convention

The original sensitivity workflow reports query energy efficiency using twice the modeled per-pass energy:

```text
queries_per_mJ = 1,000,000 / (2 × total_energy_nJ)
```

The factor is explicit as `PUBLISHED_ENERGY_ACTIVITY_FACTOR` rather than hidden in a derived constant. With 54.9221 nJ modeled energy, the release produces 9,103.8 qry/mJ, 0.65% above the paper's rounded 9,045 qry/mJ value.

## Verification Links

- `camformer-validate` checks analytical and event-model results against six published targets at 5% relative tolerance.
- `tests/test_release_regressions.py` locks the headline metrics and the camera-ready Table IV parameter rows.
- `tests/test_figures.py` locks numerical plot inputs and confirms that all three plotters render headlessly.
- `make rtl` checks the explicitly documented RTL subset. See [rtl/README.md](rtl/README.md).
