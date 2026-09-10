"""Convert Stanford Dynamic Cycling ``all_cells_SOH.mat`` to runner format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="external_data/Stanford_all_cells_SOH.mat")
    ap.add_argument("--out", default="external_data/Stanford_Data.npy")
    ap.add_argument("--min-points", type=int, default=17)
    args = ap.parse_args()
    source = loadmat(args.mat, squeeze_me=True, struct_as_record=False)
    cells = {}
    for name in sorted(k for k in source if not k.startswith("__")):
        arr = np.asarray(source[name], dtype=float)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError(f"unexpected shape for {name}: {arr.shape}")
        cycles, soh = arr[0], arr[1]
        if len(soh) < args.min_points:
            continue
        if not np.isfinite(cycles).all() or not np.isfinite(soh).all():
            raise ValueError(f"non-finite values in {name}")
        order = np.argsort(cycles)
        cycles, soh = cycles[order], soh[order]
        if len(cycles) <= 16 or not np.all(np.diff(cycles) > 0):
            raise ValueError(f"invalid cycle sequence in {name}")
        cells[name] = pd.DataFrame({"Cycle": cycles, "Capacity": soh})
    if len(cells) < 2:
        raise ValueError("need at least two sufficiently long cells")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, cells, allow_pickle=True)
    print(f"wrote {out_path} with {len(cells)} cells")


if __name__ == "__main__":
    main()
