# Contributing to CAMformerV1

Contributions to the simulator, reproducibility scripts, documentation, tests, and reference RTL are welcome.

## Development Setup

```bash
git clone https://github.com/tmo324/CAMformerV1.git
cd CAMformerV1
python3 -m venv .venv
source .venv/bin/activate
make install-test
```

## Before Opening a Pull Request

Run the Python checks and paper reproduction path:

```bash
make check
make paper
```

If the change affects RTL and Icarus Verilog plus Verilator are installed, also run:

```bash
make rtl
```

Keep pull requests focused. Add a regression test for behavior changes, update provenance when a hardware or comparison constant changes, and regenerate committed figures when their inputs or plotters change.

## Reproducibility Expectations

- Do not add a hardware constant without its source, node, units, and transformation.
- Do not present unsupported or lint-only RTL as behaviorally verified.
- Keep figure generation headless and script-based.
- Preserve the published default configuration unless a change is clearly marked as a new experiment.
- Describe any expected numeric or rendering difference in the pull request.

By contributing, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md) and license your contribution under the [MIT License](LICENSE).
