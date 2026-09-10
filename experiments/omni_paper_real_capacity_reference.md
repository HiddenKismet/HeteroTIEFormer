# OmniTIEFormer paper reference (real-capacity filling)

The OmniTIEFormer values used in the paper-style main comparison are taken
from the supplied screenshot of Tables 5--7, which reports the real-capacity
filling protocol.  They are reference values; the locally retrained Omni
checkpoints and Omni gate diagnostics are not used as paper baselines.

| Dataset | AMAE (Ah) | ARMSE (Ah) | AR² | AAE (cycle) | ARE |
|---|---:|---:|---:|---:|---:|
| PANASONIC | 0.0073 | 0.0137 | 0.9909 | 1.00 | 0.0072 |
| CALCE | 0.0055 | 0.0138 | 0.9953 | 0.80 | 0.0020 |
| GOTION | 0.0560 | 0.0745 | 0.9948 | 0.94 | 0.0012 |

## Comparison rule

Our model is compared with these values only after matching the dataset,
starting points, horizon, and capacity-filling/observed-history protocol.
Clean seed-42 runs currently stored under `runs/clean_*_one_step` use the
target-disjoint split, train-only min--max normalization, and one-step
training; they must be labelled with that protocol when reported.  In
particular, the clean TJU and NASA runs are not substituted for the GOTION and
CALCE rows above.

