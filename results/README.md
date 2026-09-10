# Results

Each JSON file is a compact output of `train_explore.py` for one held-out-cell
run.  No raw data, model checkpoint or epoch log is included.  Values are in
Ah for capacity errors; RUL AE is in cycles and RUL RE is dimensionless.

The `*_fix.json` files are the post-audit round3 runs (selector probability
gradient fixed, then low-temperature and hard-routing controls).  The matching
`*_gate_controls.json` files compare learned routing with uniform, reversed
and region-shuffled controls.  Empty recursive RUL fields indicate that the
rollout never crossed the EOL threshold and therefore has no valid predicted
EOL cycle.

The `residual_rollout_obj.json` and `residual_rollout_w025.json` /
`residual_rollout_w050.json` files are the rollout-aware objective trial. They
use the same residual model and seed-42 protocol with `λ=1.0`, `0.25`, and
`0.5`, respectively.
