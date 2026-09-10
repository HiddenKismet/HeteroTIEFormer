# HeteroTIEFormer

Lightweight research code for degradation-aware, region-adaptive battery
capacity forecasting.  The first tracked version is **V0.1**: a four-scale
candidate bank (`P={2,4,8,16}`) on top of OmniTIEFormer, with a shared
region-wise selector and dense routing back to the original `B×C×64` tensor.
The original TCEM, temporal encoder and prediction head are kept unchanged.

This repository intentionally contains no datasets, checkpoints or training
logs.  It is meant to store model structure, reproducible commands and compact
experiment records without becoming a data dump.

## Contents

- `OmniTIEFormer.py` — torch-only compatible upstream model (the Lightning
  wrapper remains optional).
- `HeteroTIEFormer.py` — V0.1 candidate-bank router and the exploratory
  fine-scale residual probe.
- `train_explore.py` — deterministic torch-only training/evaluation loop with
  target-disjoint validation, train-only normalization and early stopping.
- `diagnose_gate_controls.py` — evaluates learned, uniform, reversed,
  region-shuffled and fixed-scale gates, including a per-window oracle bound.
- `configs/` — small JSON snapshots of run settings.
- `results/` — compact JSON metrics from the tracked seed-42 exploration.

## Upstream references

- [OmniTIEFormer-Battery-RUL-Prediction](https://github.com/keepawakeyi/OmniTIEFormer-Battery-RUL-Prediction)
- [TimeMosaic](https://github.com/BenchCouncil/TimeMosaic)

The HeteroTIEFormer files are an experimental extension and should be checked
against the upstream licenses and citations before redistribution.

## Run

Prepare a `.npy` file containing a dictionary of data frames with `Cycle` and
`Capacity` columns, then run, for example:

```bash
python train_explore.py \
  --variant adaptive \
  --data-file external_data/TJU_Data.npy \
  --test-cell CY25_1 --rated 2.5 \
  --normalization train_minmax \
  --start-cycles 300 450 600 \
  --epochs 50 --patience 10 --rollout-horizon 4 \
  --rollout-loss-weight 1.0 \
  --primary-protocol observed_history \
  --seed 42 --out runs/tju_adaptive
```

For a clean selector-mechanism experiment, use pure next-cycle supervision and
disable auxiliary selector priors:

```bash
python train_explore.py \
  --variant adaptive --data-file external_data/TJU_Data.npy \
  --test-cell CY25_1 --rated 2.5 --normalization train_minmax \
  --start-cycles 300 450 600 --epochs 50 --patience 10 \
  --rollout-horizon 4 --train-objective one_step \
  --selector-budget-weight 0 --selector-entropy-weight 0 \
  --primary-protocol observed_history --seed 42 \
  --out runs/tju_clean_v01
```

With a multi-step horizon, validation targets are separated from training
targets by trimming `horizon - 1` window starts.  The `train_minmax` range is
computed only from each training cell's prefix before its first validation
target; validation tails and the held-out test cell cannot set the scale.
Trajectories need at least `window + 2*horizon` observations for this split.

For a layout-only reproduction of historical round1 V0.1, add
`--regional-restore aligned --train-objective scheduled_sampling` and use a
new output directory. The historical V0.1 comparison uses scheduled sampling;
the current runner's default `free_rollout` objective is a separate training
experiment, independent of the primary evaluation protocol.

The evaluator reports both observed-history and free-running recursive
MAE/RMSE/R², plus EOL-based RUL AE/RE.  The primary protocol defaults to
`observed_history`: every prediction uses a window refreshed with measured
capacity.  The `primary` block in `metrics.json` and best-checkpoint selection
follow that protocol; use `--primary-protocol recursive` only when open-loop
stability is the target.  `--variant omni` is the fixed-patch baseline;
`uniform` is a four-candidate fixed mixture; `residual` is an exploratory
fine-scale-anchor control.

After a run, fixed-scale and oracle diagnostics can be written separately so
the checkpoint record is preserved:

```bash
python diagnose_gate_controls.py --run runs/tju_clean_v01 \
  --data-file external_data/TJU_Data.npy --test-cell CY25_1 \
  --start-cycles 300 450 600 \
  --out diagnostics/tju_clean_v01_oracle.json
```

### Regional restoration ablation (R1; not yet benchmarked)

The first structural repair is opt-in via `--regional-restore aligned`.
It only changes how each Hetero candidate restores its RCA output:
`[B*M,N,D,P] -> [B,M,N,D,P] -> [B,N,P,M,D] -> [B,L,M,D]`.
The middle step is a permutation, not a reshape. This restores each patch's
features to their original cycle and variable positions before the existing
embedding residual is added.

`--regional-restore legacy` remains the default and delegates to the upstream
operation exactly. The original Omni source and baseline are unchanged;
`--variant omni --regional-restore aligned` is rejected. The aligned mode applies
to all four candidates, including P=2, in adaptive, uniform and residual routing.
It adds no parameters and does not change Local restoration, RCA weights or
amplitudes, the selector, TCEM, the loss, or early stopping (50 epochs / patience
10 by default). It is a layout correction, not a new attention mechanism.

The mode is recorded in `config.json`, the checkpoint's `config`, and the new
metrics. The gate diagnostic restores this setting; configurations without it
are treated as legacy. A weights-only state dict cannot distinguish the modes,
so always keep its configuration. Loading old weights with `aligned` changes
their computation and is not a faithful reevaluation of the old checkpoint.

For an isolated comparison, keep every other setting and the training objective
identical, use separate output directories, and vary only this switch. In
particular, the current rollout-aware objective differs from the historical
round1 scheduled-sampling objective; a new aligned run under the current default
must not be presented as a layout-only comparison to round1. No new performance
claim is made by this repair, and historical results are not overwritten.

Run the dataset-free regression suite from this directory:

```bash
python -m unittest discover -s tests -v
```

Tests cover all four patch sizes, multiple variables, non-contiguous inputs,
patch-position weighting, gradients, legacy compatibility, and checkpoint/CLI
round trips. A single tiny synthetic epoch checks saving and reloading; it is
not a battery benchmark.

The rollout-aware objective is

```text
L = L_1step + rollout_loss_weight * L_free-rollout
```

where `L_free-rollout` feeds the model's own previous prediction back into the
next window.  The default weight is `1.0`; feedback is stop-gradient during
optimization so the model is trained on its own-state distribution without
unbounded backpropagation through the entire horizon.

## Tracked exploration (seed 42)

Protocol: train-only min–max normalization, 4-step scheduled-sampling
training/evaluation, maximum 150 epochs, patience 60.  TJU holds out
`CY25_1` and NASA holds out `B0005`.

| Dataset / variant | observed MAE | observed RMSE | observed R² | recursive MAE | recursive RMSE | recursive R² |
|---|---:|---:|---:|---:|---:|---:|
| TJU / Omni | 0.001746 | 0.002431 | 0.999155 | 0.012995 | 0.014790 | 0.976646 |
| TJU / adaptive V0.1 | 0.001581 | 0.002561 | 0.999038 | 0.075640 | 0.090227 | 0.150666 |
| TJU / uniform | 0.001941 | 0.003063 | 0.998625 | 0.176533 | 0.279215 | -7.509981 |
| TJU / residual probe | **0.001158** | **0.001828** | **0.999509** | **0.012151** | 0.015538 | 0.974629 |
| NASA / Omni | 0.008045 | 0.012117 | 0.761597 | 0.022829 | 0.028417 | -0.154926 |
| NASA / adaptive V0.1 | 0.012502 | 0.017041 | 0.526829 | **0.022725** | 0.027876 | -0.219050 |
| NASA / uniform | 0.012342 | 0.015005 | 0.739480 | 0.042211 | 0.044499 | -0.185033 |
| NASA / residual probe | **0.007283** | **0.010820** | **0.828519** | 0.023017 | **0.026027** | **0.498960** |

The first V0.1 checkpoints had near-uniform probabilities (entropy ≈ `log 4`),
and an audit found that selector regularization was accidentally reading a
detached gate tensor.  `train_explore.py` now exposes the non-detached
`latest_gate_probs` for the regularization loss while retaining detached gates
for diagnostics.  A small round3 follow-up tested low-temperature annealing
and hard routing after this fix; its compact records are the `*_fix.json`
files in `results/`.

Round3 did make hard routing non-uniform, but it did not improve the
cross-cell recursive protocol.  The best earlier residual probe remains the
strongest TJU result (`recursive MAE=0.012151`), while the best round3 TJU
hard-adaptive run reached `0.029317`.  On NASA, the best earlier Omni result
was `0.022829`; round3 hard-adaptive reached `0.025932`.  Thus the current
default should remain the round1 residual probe until a selector is trained
with a rollout-aware objective.

The post-fix gate-control checks illustrate the failure mode.  With scale order
`P={2,4,8,16}`, hard adaptive routing selected approximately `[0.0039, 0.9961,
0, 0]` on TJU and `[0.9856, 0.0096, 0, 0.0048]` on NASA; the soft residual
run stayed close to uniform.  These controls are diagnostic evidence, not a
claim that the selector has learned the desired degradation semantics.

## Rollout-aware objective trial

The next controlled experiment changed the target to

```text
L = L_1step + λ L_free-rollout
```

with a fully free four-step rollout and stop-gradient feedback.  The residual
architecture was kept fixed and only `λ` was changed:

| Dataset / λ | observed MAE | recursive MAE | recursive R² |
|---|---:|---:|---:|
| TJU / 1.00 | 0.002625 | 0.109654 | -1.146449 |
| TJU / 0.25 | 0.001318 | 0.063847 | 0.394354 |
| TJU / 0.50 | 0.005694 | 0.129218 | -1.461417 |
| NASA / 1.00 | 0.016174 | 0.036253 | 0.001738 |
| NASA / 0.25 | 0.008868 | 0.024960 | 0.465560 |
| NASA / 0.50 | 0.011646 | 0.026826 | 0.424881 |

None of these runs exceeded the round1 residual reference (`0.012151` TJU,
`0.023017` NASA).  The objective is therefore recorded as a negative result:
four-step free-running supervision alone does not correct the much longer
cross-cell recursive distribution shift and can damage one-step capacity fit.

## R1 aligned Regional-restoration trial

The R1 change fixes only the time/feature permutation in Hetero candidates'
Regional restoration. The historical V0.1 training objective, seed, horizon,
epoch limit and patience are retained. Results below are evaluated from the
best validation-rollout checkpoint; the old V0.1 values are included only as a
reference, not retrained.

| Dataset | old V0.1 observed MAE | R1 observed MAE | old V0.1 recursive MAE | R1 recursive MAE |
|---|---:|---:|---:|---:|
| TJU | 0.001581 | 0.002037 | 0.075640 | **0.045420** |
| NASA | 0.012502 | **0.009504** | **0.022725** | 0.038503 |
| Panasonic | — | 0.003592 | — | 0.299881 |

R1 improves TJU recursive stability and NASA observed-history accuracy, but it
does not improve every metric. The adaptive gate remains near-uniform on all
three datasets, so this result supports a representation-alignment repair, not
yet a claim of learned degradation-dependent granularity. The corresponding
compact settings are `configs/r1_aligned_*.json`, and the full local metrics,
checkpoints and gate controls remain under the `runs/r1_aligned_*` directories.
