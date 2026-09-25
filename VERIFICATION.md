# Verification

Verified on 2026-09-25.

- All six baseline checks passed against published targets (5% release tolerance).
- Eight upstream numerical regression and figure tests passed.
- Exhaustive toy-model checks passed for 256 queries × 8 k values × 3 selection modes × 3 temperatures (18,432 cases): weight normalization, selected-count and candidate membership, exact selection, convex output bounds, and bit-flip output propagation.
- Browser checks covered all five views, query-bit propagation, dense mode's zero distance from itself, tile/top-k switching, comparison pins, circuit response, pipeline Next, Figure 8 area selection, Figure 10 projections, and hash restoration after reload.
- Desktop (1280 px) and mobile (390 px) inspected. A hidden-chart resize issue was fixed by sizing each Plotly render to the current container. Verified page width 390 and chart width 308 at the mobile breakpoint.
- Downloaded Figure 6 CSV matches all nine exported source values exactly. Downloaded SVG parses as XML.
- No browser console errors in the tested interactions.

The app is an educational and data-exploration companion, not an independent verification of the underlying research, analog circuit behavior, or trained-model accuracy.

## Connected datapath revision

- Hand-calculated two-row weighted output verified in the pure model and browser: [2,4] and [6,8] at equal weights produce [4,6].
- Editing the first stored value to [4,4] updates products and output to [5,6], preserved after reload.
- BF16 round-to-nearest-even ties checked, along with invalid dimensions, nonbinary inputs, nonfinite values and impossible top-k rejection.
- 256 query cases check normalized weights, constant-value preservation and opposite-value cancellation.
- Browser checks cover custom Q/K/V apply, row products, value-cell edits, query/match step progression, navigation to other views and back, and saved custom matrices. No console errors observed.
