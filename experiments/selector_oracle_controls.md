# Selector/oracle scale controls

Date: 2026-09-10  
Purpose: test whether the four candidate branches contain complementary
information before changing the selector.

The diagnostic evaluates the existing seed-42 checkpoints with observed-history
one-step windows. `force_p2` … `force_p16` select one candidate everywhere;
`oracle` chooses the lowest absolute error separately for each window. These
are interventions on old checkpoints, so they are not a replacement for the
clean target-disjoint retraining protocol.

| checkpoint | learned MAE | best fixed | best fixed MAE | oracle MAE | oracle gain vs fixed | learned/oracle agreement |
|---|---:|---|---:|---:|---:|---:|
| TJU residual probe | 0.000961 | P=2 | 0.000835 | 0.000499 | 40.3% | 3.3% |
| NASA residual probe | 0.010274 | P=2 | 0.008594 | 0.006386 | 25.7% | 10.6% |
| Panasonic R1 aligned adaptive | 0.003422 | P=4 | 0.003294 | 0.002362 | 28.3% | 24.6% |

The oracle gains are large enough to show that scale-complementary candidates
are present. The low learned/oracle agreement shows that the current gate is
not recovering the per-window best scale; near-uniform probabilities should
therefore be treated as a selector failure, not evidence that adaptive scale
allocation is useless.

## Code changes accompanying this diagnostic

- `split_window_indices` now makes multi-step train/validation target ranges
  disjoint by trimming `horizon - 1` training starts.
- `train_minmax` uses only the training-cell prefix ending before the first
  validation target; the held-out test cell and validation tails do not set the
  normalization range.
- `--train-objective one_step` provides pure next-cycle supervision. Recursive
  rollout remains a secondary evaluation diagnostic.
- Gate controls now include `force_p2`, `force_p4`, `force_p8`, and `force_p16`,
  plus a per-window fixed-scale oracle summary.

Next experiment: retrain clean V0.1 with one-step loss, zero selector auxiliary
weights, and the target-disjoint split; only then compare degradation-aware and
residual routing variants.
