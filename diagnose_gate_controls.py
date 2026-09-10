"""Evaluate a trained adaptive checkpoint with routing controls.

This isolates the contribution of the selector: the same weights are scored
with learned, uniform, scale-reversed, and region-shuffled gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_explore import (
    WINDOW,
    RelativeCapacityForecaster,
    aggregate,
    evaluate_one,
    load_cells,
    make_windows,
    build_model,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='directory containing model.pt/config.json')
    ap.add_argument('--data-file', required=True)
    ap.add_argument('--test-cell', required=True)
    ap.add_argument('--start-cycles', type=int, nargs='+', required=True)
    ap.add_argument('--modes', nargs='+', default=['learned', 'uniform', 'reverse', 'region_shuffle'])
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = Path(args.run)
    config = json.loads((run / 'config.json').read_text())
    if config['variant'] not in ('adaptive', 'residual'):
        raise ValueError('gate controls require an adaptive or residual run')
    model = RelativeCapacityForecaster(
        build_model(config['variant'], int(config['d_model']),
                    float(config.get('selector_tau', 1.5)),
                    bool(config.get('selector_hard', False)),
                    float(config.get('selector_init_std', 0.0))),
        float(config['delta_scale']),
    )
    checkpoint = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    cells_raw = load_cells(Path(args.data_file))
    cycles_raw, q_raw = cells_raw[args.test_cell]
    norm_min = float(config['normalization_min'])
    norm_range = float(config['normalization_range'])
    q = (q_raw - norm_min) / norm_range
    cycles = cycles_raw
    eol_norm = (float(config['rated']) * float(config['eol_ratio']) - norm_min) / norm_range
    low, high = config['feedback_bounds']

    # A fixed window set gives comparable gate summaries for every control.
    starts = [int(np.searchsorted(cycles, s)) for s in args.start_cycles]
    probe_starts = np.arange(WINDOW, len(q), max(1, len(q) // 64))
    probe_x, _ = make_windows(q, probe_starts, 1)
    result = {}
    for mode in args.modes:
        model.base.set_gate_mode(mode)
        gates = []
        with torch.no_grad():
            for batch in probe_x.split(args.batch_size):
                model(batch)
                gates.append(model.base.latest_gate.cpu())
        gate = torch.cat(gates, dim=0).numpy()
        mean_gate = gate.mean(axis=(0, 1))
        expected_tokens = float(np.dot(mean_gate, [8.0, 4.0, 2.0, 1.0]))
        entropy = float(-(mean_gate * np.log(np.clip(mean_gate, 1e-8, 1.0))).sum())
        rows = []
        for start in starts:
            if start < WINDOW or start >= len(q):
                raise ValueError(f'bad start index {start}')
            rows.append(evaluate_one(model, q, cycles, start, norm_min, norm_range,
                                     eol_norm, low, high, args.batch_size))
        result[mode] = {
            'mean_gate_by_scale': mean_gate.tolist(),
            'expected_patches_per_16_cycles': expected_tokens,
            'gate_entropy': entropy,
            'observed': aggregate(rows, 'observed'),
            'recursive': aggregate(rows, 'recursive'),
            'start_points': rows,
        }

    (run / 'gate_controls.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
