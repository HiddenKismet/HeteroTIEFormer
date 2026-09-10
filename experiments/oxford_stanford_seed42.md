# Oxford / Stanford seed-42 experiment

Protocol: 4-step free-rollout training objective, observed-history checkpoint
selection, 150 epochs maximum, patience 60, `d_model=16`, one CPU thread per
process, seed 42.  The reported test rows use the first held-out cell shown
below; they are a reproducibility check, not a cross-cell aggregate.

| Dataset | Context | Held-out cell | Model | Best epoch | Observed AMAE | Observed ARMSE | Observed AR2 |
|---|---:|---|---:|---:|---:|---:|---:|
| Oxford | 64 sampled points | Cell2 | Omni | 149 | 0.011215 | 0.020325 | -0.3883 |
| Oxford | 64 sampled points | Cell2 | V0.1 adaptive | 55 | 0.011366 | 0.018349 | -0.1341 |
| Stanford Dynamic Cycling | 16 SOH observations | Cell_053 | Omni | 107 | 0.001527 | 0.002058 | 0.9890 |
| Stanford Dynamic Cycling | 16 SOH observations | Cell_053 | V0.1 adaptive | 73 | 0.001388 | 0.001920 | 0.9905 |

Oxford V0.1 is essentially tied with Omni on this holdout (lower ARMSE but
slightly higher AMAE).  Stanford V0.1 improves observed AMAE by about 9.1%
and ARMSE by about 6.7%.  Recursive metrics are recorded in each run's
`metrics.json`; they are secondary because the primary protocol refreshes the
input window with measured capacity.

The Oxford values are in the CSV's capacity units (Ah-like values). Stanford
stores normalized SOH rather than Ah, so its MAE/RMSE values are unitless SOH
errors. With a 16-point Stanford context the four candidates share one
routing region; this is a compact-context compatibility result, not evidence
of multi-region heterogeneity.

Data preparation is intentionally kept outside the repository history: use
`tools/prepare_oxford.py` for the public Oxford CSV and
`tools/prepare_stanford_dynamic.py` for `all_cells_SOH.mat`.  The Stanford
processed file contains only 10--32 SOH observations per cell, so the
16-observation context is required; RUL metrics depend on the selected EOL
threshold and may be undefined when a prediction does not cross it.
