# Migration manifest

This repository is the lightweight, runnable HeteroTIEFormer workspace.

Tracked in GitHub:

- model source, Omni reference source, runner, gate diagnostics, tests, and
  launch scripts;
- all experiment configurations, result summaries, paper-reference tables,
  and per-start-point comparisons;
- the small processed datasets used by the local commands;
- the seed-42 clean V0.1 checkpoints and their JSON records under
  `results/clean_matrix/`.

The full historical `runs/` directory and training logs are intentionally not
tracked because they contain many duplicate checkpoints.  The JSON summaries
and selected V0.1 checkpoints are sufficient to continue or reproduce the
reported comparisons without making the repository unnecessarily heavy.

## On a new computer

```bash
git clone https://github.com/HiddenKismet/HeteroTIEFormer.git
cd HeteroTIEFormer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

New clean runs use `epochs=50`, `patience=10`, seed 42, one-step training,
train-only min--max normalization, target-disjoint validation, and observed
history as the primary protocol.  The default launcher runs only our model
variants; set `RUN_OMNI_BASELINE=1` only when an additional local Omni run is
explicitly needed.

