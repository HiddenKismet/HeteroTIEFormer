"""Download and normalize the public Battery Archive CALCE CX2 study.

The training runner expects a NumPy object containing
``{cell_name: DataFrame(Cycle, Capacity)}``.  Battery Archive exposes the
cycle-level CSVs at a stable URL, so the remote GPU host can prepare the data
without checking a large binary file into Git.

The blockwise screen removes obvious capacity glitches (for example a single
half-capacity measurement) while preserving the original order.  Retained
rows are re-indexed because all experiments use the measured-cycle index as
the temporal coordinate.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


CELL_IDS = [
    "CALCE_CX2-16_prism_LCO_25C_0-100_0.5/0.5C_a",
    "CALCE_CX2-25_prism_LCO_25C_0-100_0.5/0.5C_b",
    "CALCE_CX2-33_prism_LCO_25C_0-100_0.5/0.5C_d",
    "CALCE_CX2-34_prism_LCO_25C_0-100_0.5/0.5C_e",
    "CALCE_CX2-36_prism_LCO_25C_0-100_0.5/0.5C_f",
    "CALCE_CX2-37_prism_LCO_25C_0-100_0.5/0.5C_g",
    "CALCE_CX2-38_prism_LCO_25C_0-100_0.5/0.5C_h",
]


def screen_blockwise(values: np.ndarray, block_size: int = 40) -> np.ndarray:
    """Keep points within two standard deviations of their local block."""

    keep = np.ones(len(values), dtype=bool)
    for start in range(0, len(values), block_size):
        stop = min(len(values), start + block_size)
        block = values[start:stop]
        if len(block) < 5:
            continue
        sigma = float(np.std(block))
        if sigma > 0:
            keep[start:stop] = np.abs(block - float(np.mean(block))) <= 2.0 * sigma
    return keep


def fetch_cell(cell_id: str, timeout: int = 120) -> tuple[pd.DataFrame, dict]:
    file_name = cell_id.replace("/", "-")
    url = f"https://www.batteryarchive.org/data/{file_name}_cycle_data.csv"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(io.BytesIO(response.content))
    if "Cycle_Index" not in frame or "Discharge_Capacity (Ah)" not in frame:
        raise ValueError(f"unexpected columns for {cell_id}: {list(frame.columns)}")

    frame = frame[["Cycle_Index", "Discharge_Capacity (Ah)"]].copy()
    frame.columns = ["Cycle", "Capacity"]
    frame["Cycle"] = pd.to_numeric(frame["Cycle"], errors="coerce")
    frame["Capacity"] = pd.to_numeric(frame["Capacity"], errors="coerce")
    frame = frame.dropna().sort_values("Cycle").drop_duplicates("Cycle")
    q = frame["Capacity"].to_numpy(dtype=float)
    keep = screen_blockwise(q)
    q = q[keep]
    if len(q) <= 64 or not np.isfinite(q).all():
        raise ValueError(f"invalid or too-short trajectory for {cell_id}: {len(q)} rows")
    clean = pd.DataFrame({"Cycle": np.arange(1, len(q) + 1, dtype=float), "Capacity": q})
    meta = {
        "cell_id": cell_id,
        "url": url,
        "raw_rows": int(len(frame)),
        "retained_rows": int(len(clean)),
        "first_capacity_Ah": float(q[0]),
        "last_capacity_Ah": float(q[-1]),
    }
    return clean, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="external_data/BatteryArchive_CALCE_CX2.npy")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.json")

    cells = {}
    records = []
    for cell_id in CELL_IDS:
        print(f"downloading {cell_id}", flush=True)
        cells[cell_id], meta = fetch_cell(cell_id)
        records.append(meta)
        print(f"  retained={meta['retained_rows']} last={meta['last_capacity_Ah']:.6f} Ah", flush=True)

    np.save(out, cells, allow_pickle=True)
    manifest.write_text(json.dumps({"source": "Battery Archive", "cells": records}, indent=2))
    print(f"wrote {out} and {manifest}", flush=True)


if __name__ == "__main__":
    main()
