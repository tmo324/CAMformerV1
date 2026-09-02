# RTL reference artifacts

This directory contains SystemVerilog blocks used during CAMformer hardware
exploration. They support the paper's synthesis provenance, but they are not a
complete, integrated, tape-out-ready CAMformer implementation.

## Verification status

| Scope | Status | Check |
|---|---|---|
| `a1_q_buffer.sv` | Behaviorally smoke-tested | Icarus testbench |
| `a4_9bit_adc.sv` | Behaviorally smoke-tested | Icarus testbench |
| Files in `filelists/lint.f` | Syntax and lint checked | Icarus where supported, Verilator for all files |
| `a12_sparseMM.sv` | Verilator lint only | Icarus 13 cannot elaborate its parameterized packed-array indexing |
| `experimental/unsupported/a8_top3.sv` | Incomplete prototype | Excluded from supported filelists |
| `experimental/unsupported/a10_top30.sv` | Incomplete prototype | Excluded from supported filelists |

The two top-k prototypes depend on an absent `fifo` module and do not implement
their partition bookkeeping completely. They are preserved for research
traceability, not presented as working RTL.

The `a12_sparseMM.sv` name is historical. Its arithmetic is bit-vector
arithmetic and has not been validated as an IEEE BF16 datapath.

The legacy reference blocks also emit nonfatal lint findings for width
conversion, incomplete cases, blocking assignments in sequential processes,
and one constant comparison. The check reports these findings rather than
silencing them. Treat them as review items before synthesis or reuse.

## Run checks

Install Icarus Verilog and Verilator, then run:

```bash
make rtl
```

The command compiles the portable source set, executes two smoke testbenches,
and lints the supported filelist. A successful run ends with
`PASS RTL syntax, smoke tests, and lint`.
