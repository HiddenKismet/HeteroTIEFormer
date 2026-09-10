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
  scheduled-sampling recursive rollout and early stopping.
- `diagnose_gate_controls.py` — evaluates a trained adaptive checkpoint with
  learned, uniform, reversed and region-shuffled gates.
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
  --epochs 150 --patience 60 --rollout-horizon 4 \
  --seed 42 --out runs/tju_adaptive
```

The evaluator reports both observed-history and free-running recursive
MAE/RMSE/R², plus EOL-based RUL AE/RE.  `--variant omni` is the fixed-patch
baseline; `uniform` is a four-candidate fixed mixture; `residual` is an
exploratory fine-scale-anchor control.

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

The selector-control diagnostic found near-uniform probabilities in the first
V0.1 checkpoints (entropy ≈ `log 4`), so the residual improvement should not
be attributed to successful adaptive routing yet.  The next controlled step is
to test hard/low-temperature selector training and retain the same protocol.
