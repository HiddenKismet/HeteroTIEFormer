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

import train_explore
from train_explore import (
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
    ap.add_argument('--modes', nargs='+', default=[
        'learned', 'uniform', 'reverse', 'region_shuffle',
        'force_p2', 'force_p4', 'force_p8', 'force_p16',
    ])
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--out', default=None,
                    help='JSON output path (default: <run>/gate_controls.json)')
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = Path(args.run)
    config = json.loads((run / 'config.json').read_text())
    if config['variant'] not in ('adaptive', 'residual'):
        raise ValueError('gate controls require an adaptive or residual run')
    window = int(config.get('window', 64))
    train_explore.WINDOW = window
    model = RelativeCapacityForecaster(
        build_model(config['variant'], int(config['d_model']),
                    float(config.get('selector_tau', 1.5)),
                    bool(config.get('selector_hard', False)),
                    float(config.get('selector_init_std', 0.0)),
                    regional_restore=config.get('regional_restore', 'legacy'),
                    seq_len=window),
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
    probe_starts = np.arange(window, len(q), max(1, len(q) // 64))
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
            if start < window or start >= len(q):
                raise ValueError(f'bad start index {start}')
            rows.append(evaluate_one(model, q, cycles, start, norm_min, norm_range,
                                     eol_norm, low, high, args.batch_size))
        result[mode] = {
            'regional_restore': config.get('regional_restore', 'legacy'),
            'mean_gate_by_scale': mean_gate.tolist(),
            'expected_patches_per_16_cycles': expected_tokens,
            'gate_entropy': entropy,
            'observed': aggregate(rows, 'observed'),
            'recursive': aggregate(rows, 'recursive'),
            'start_points': rows,
        }

    # Fixed-scale interventions on every observed-history one-step window
    # expose the upper bound of the current candidate bank.  This is more
    # informative than gate entropy alone: if the per-window oracle is much
    # better than the best fixed scale, the candidates are complementary and
    # the selector (rather than the representation bank) is the bottleneck.
    all_starts = np.arange(window, len(q))
    target_a = q[all_starts] * norm_range + norm_min
    fixed_modes = [f'force_p{p}' for p in (2, 4, 8, 16)]
    scale_errors = {}
    scale_gate_argmax = {}
    for mode in fixed_modes + ['learned']:
        model.base.set_gate_mode(mode)
        preds = []
        gates = []
        with torch.no_grad():
            x, _ = make_windows(q, all_starts, 1)
            for batch in x.split(args.batch_size):
                preds.append(model(batch).cpu().squeeze(-1))
                gate = model.base.latest_gate
                if gate is not None:
                    gates.append(gate.cpu())
        pred_a = torch.cat(preds).numpy() * norm_range + norm_min
        scale_errors[mode] = np.abs(pred_a - target_a)
        if gates:
            scale_gate_argmax[mode] = torch.cat(gates).numpy().mean(axis=1).argmax(axis=-1).tolist()

    fixed_matrix = np.stack([scale_errors[m] for m in fixed_modes], axis=1)
    oracle_index = fixed_matrix.argmin(axis=1)
    oracle_error = fixed_matrix[np.arange(len(all_starts)), oracle_index]
    fixed_means = fixed_matrix.mean(axis=0)
    learned_mean = float(scale_errors['learned'].mean())
    oracle = {
        'window_count': int(len(all_starts)),
        'fixed_modes': fixed_modes,
        'fixed_observed_mae_Ah': {
            mode: float(scale_errors[mode].mean()) for mode in fixed_modes
        },
        'best_fixed_mode': fixed_modes[int(fixed_means.argmin())],
        'best_fixed_observed_mae_Ah': float(fixed_means.min()),
        'learned_observed_mae_Ah': learned_mean,
        'oracle_observed_mae_Ah': float(oracle_error.mean()),
        'oracle_gain_vs_best_fixed': float(1.0 - oracle_error.mean() / max(fixed_means.min(), 1e-12)),
        'oracle_gain_vs_learned': float(1.0 - oracle_error.mean() / max(learned_mean, 1e-12)),
        'oracle_scale_fraction': {
            mode: float(np.mean(oracle_index == i)) for i, mode in enumerate(fixed_modes)
        },
    }
    if 'learned' in scale_gate_argmax:
        learned_scale = np.asarray(scale_gate_argmax['learned'])
        oracle['learned_vs_oracle_agreement'] = float(np.mean(learned_scale == oracle_index))
    result['oracle_fixed_scale'] = oracle

    output_path = Path(args.out) if args.out else run / 'gate_controls.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
