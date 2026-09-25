# CAMformer Studio

Interactive companion to CAMformer: Binary Associative Memory Is All You Need.

## Run

Serve this directory, then visit http://localhost:8788:

```sh
python3 -m http.server 8788 --bind 127.0.0.1
```

The browser app needs no installation or build step. Plotly 3.1.0 is vendored locally; there are no runtime CDN, API, or analytics calls. Static hosting can publish this directory unchanged.

## Views

- Attention lab: editable 8-bit query, 16 synthetic keys, local top-2/global top-k, exact top-k and dense binary attention, temperature control, output vector and weights.
- Inside BA-CAM: XNOR contributions and normalized ideal response. This is a teaching model, not SPICE or an absolute-voltage prediction.
- Pipeline: five steps through the three hardware stages, alongside released stage timing.
- Design explorer: released Table IV sweeps, configuration pins and default-relative metrics.
- Paper results: interactive Figures 6, 8 and 10, CSV/SVG exports and original author figures.

Controls and query bits are encoded in the URL fragment. Pins remain local to the current page session. Query, temperature and toy-k controls affect the teaching example only, never the hardware-result snapshot.

## Scientific scope

Data are exported from CAMformerV1 commit `33279940d3c101c3f9fefd572e542191bbb2bd7f`. The source snapshot records its generation timestamp and full revision. Hardware values are simulation/model outputs, not fabricated-chip measurements. The baseline is the release's 1 GHz, 45 nm normalized model with 1,024 keys, 64 dimensions and 16 heads.

The top-k sweep holds tile rows at 16 and reuses fixed Top-32 hardware. The tile sweep fixes top-k at 32. The UI intentionally does not extrapolate to unvalidated combinations. Expected recall uses the repository's hypergeometric placement model and is not trained-model accuracy.

The headline displays published rounded values. The explorer displays computed release values (9,103.8 qry/mJ versus the published 9,045, within the release tolerance). The release efficiency uses `1e6 / (2 * total_energy_nJ)`; its activity factor of two is retained.

Figure 6 uses the raw circuit basis from its figure script. Figure 8 excludes DRAM from its on-chip breakdown. Figure 10 preserves the source coordinates and labels projections. Its SpAtten area coordinate uses CAMformer area; projected SpAtten also uses CAMformer power. The interactive chart omits the original circular “Pareto” guides because they are not computed non-dominated fronts. See the upstream TRACEABILITY.md for full provenance.

The toy computation uses match counts as logits divided by an illustrative temperature, synthetic three-dimensional values, and JavaScript floating-point arithmetic. Ties use stable row order. It does not reproduce HAD training, learned scaling, BF16 rounding, ADC or PVT error.

## Regenerate data

Clone CAMformerV1 and check out the recorded revision. Using Python 3.12, create an isolated environment and install `p03_tools/requirements.lock`. From the upstream repository root:

```sh
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/camformer-mpl python /absolute/path/to/camformer-studio/p03_tools/export_data.py
```

The script imports and calls the upstream figure/sweep functions and validates the six baseline metrics before writing `p00_data/results.json`. Recheck scientific scope if upgrading the source revision.

## Verify

```sh
node p04_tests/attention.test.mjs
node --check p01_ui/app.mjs
```

Upstream numerical regression and figure tests are run separately in the source checkout. See `VERIFICATION.md` for evidence and manual browser coverage.

## License

MIT, matching the upstream release. The author figures and model data originate in CAMformerV1. Plotly's MIT license is in `p02_vendor/LICENSE`. Cite the paper using the upstream CITATION.cff or publisher metadata.
