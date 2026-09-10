"""Convert the public Oxford cycle-capacity CSV to the runner's .npy format.

The repository's training runner consumes a pickled dictionary of pandas
DataFrames with ``Cycle`` and ``Capacity`` columns.  Keeping this conversion
explicit makes the provenance and the 100-cycle sampling interval visible,
without committing a generated binary dataset to the code repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="external_data/Oxford.csv")
    ap.add_argument("--out", default="external_data/Oxford_Data.npy")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    frame = pd.read_csv(csv_path)
    required = {"cycle_number", "cell_number", "capacity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    cells = {}
    for name, group in frame.groupby("cell_number", sort=True):
        group = group.sort_values("cycle_number")
        out = group.rename(columns={"cycle_number": "Cycle", "capacity": "Capacity"})[
            ["Cycle", "Capacity"]
        ].reset_index(drop=True)
        if len(out) <= 64:
            print(f"skip {name}: {len(out)} points (requires >64)")
            continue
        if not np.isfinite(out["Capacity"].to_numpy()).all():
            raise ValueError(f"non-finite capacity in {name}")
        if not np.all(np.diff(out["Cycle"].to_numpy()) > 0):
            raise ValueError(f"non-increasing cycle values in {name}")
        cells[str(name)] = out

    if len(cells) < 2:
        raise ValueError("need at least two cells after filtering")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, cells, allow_pickle=True)
    print(f"wrote {out_path} with {len(cells)} cells")
    for name, out in cells.items():
        print(f"  {name}: n={len(out)} cycle=[{out.Cycle.iloc[0]}, {out.Cycle.iloc[-1]}]")


if __name__ == "__main__":
    main()
