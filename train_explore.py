"""Small, reproducible V0.1 exploration runner.

The runner keeps the original Omni network and uses a torch-only loop so that
the diagnostics are independent of pytorch-forecasting/Lightning.  It supports
the original Omni branch, V0.1 adaptive routing, a uniform gate control, and the
fine-scale residual routing probe.

The primary evaluation protocol is ``observed_history``: every prediction is
made from a window containing the measured capacity values.  A fully
free-running rollout is still computed and recorded as a secondary stability
diagnostic, but it is not used to select the primary checkpoint unless the
caller explicitly requests ``--primary-protocol recursive``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from HeteroTIEFormer import HeteroTIEFormer
from OmniTIEFormer import OmniTIEFormer


WINDOW = 64


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cells(path: Path):
    data = np.load(path, allow_pickle=True).item()
    cells = {}
    for name, frame in data.items():
        cycle_col = 'Cycle' if 'Cycle' in frame.columns else 'cycle'
        capacity_col = 'Capacity' if 'Capacity' in frame.columns else 'capacity'
        frame = frame.sort_values(cycle_col)
        cycles = frame[cycle_col].to_numpy(dtype=float)
        q = frame[capacity_col].to_numpy(dtype=float)
        if len(q) <= WINDOW or not np.isfinite(q).all() or not np.all(np.diff(cycles) > 0):
            raise ValueError(f'invalid or too-short trajectory: {name}')
        cells[name] = (cycles, q)
    return cells


def make_windows(q, indices, horizon):
    x = np.stack([q[t - WINDOW:t] for t in indices]).astype('float32')
    y = np.stack([q[t:t + horizon] for t in indices]).astype('float32')
    return torch.from_numpy(x[..., None]), torch.from_numpy(y)


class RelativeCapacityForecaster(torch.nn.Module):
    """Predict a scaled increment while exposing an absolute capacity output."""

    def __init__(self, base, delta_scale=10.0):
        super().__init__()
        self.base = base
        self.delta_scale = float(delta_scale)

    @property
    def latest_gate(self):
        return getattr(self.base, 'latest_gate', None)

    @property
    def latest_gate_probs(self):
        # Keep the non-detached probabilities for selector regularization.
        # ``latest_gate`` is intentionally detached for diagnostics/logging.
        return getattr(self.base, 'latest_gate_probs', None)

    def forward(self, x):
        anchor = x[:, -1, 0].unsqueeze(-1)
        relative = (x - anchor[:, None, :]) * self.delta_scale
        delta = self.base(relative) / self.delta_scale
        return anchor + delta


def rollout(model, context, future=None, teacher_forcing=0.0,
            feedback_low=-0.25, feedback_high=1.25, collect_gates=False):
    horizon = int(future.shape[1]) if future is not None else 1
    history = context
    predictions = []
    gates = []
    for step in range(horizon):
        pred = model(history).squeeze(-1)
        predictions.append(pred)
        if collect_gates:
            gate = getattr(model, 'latest_gate_probs', None)
            if gate is None:
                gate = getattr(model, 'latest_gate', None)
            if gate is not None:
                gates.append(gate)
        if step + 1 == horizon:
            break
        feedback = pred.detach().clamp(feedback_low, feedback_high)
        if future is not None and teacher_forcing > 0:
            mask = torch.rand_like(feedback) < teacher_forcing
            feedback = torch.where(mask, future[:, step], feedback)
        history = torch.cat([history[:, 1:, :], feedback[:, None, None]], dim=1)
    result = torch.stack(predictions, dim=1)
    return (result, gates) if collect_gates else result


def selector_regularization(gates, budget=4.0, entropy_target=1.0):
    if not gates:
        return None
    probs = torch.stack(gates, dim=0).mean(dim=(0, 1, 2))
    counts = probs.new_tensor([8.0, 4.0, 2.0, 1.0])
    expected = (probs * counts).sum()
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum()
    budget_penalty = F.relu(expected - budget).pow(2)
    entropy_penalty = F.relu(entropy_target - entropy).pow(2)
    return budget_penalty, entropy_penalty, expected, entropy


def build_model(variant, d_model, selector_tau=1.5, selector_hard=False,
                selector_init_std=0.0, regional_restore='legacy', seq_len=64):
    if regional_restore not in ('legacy', 'aligned'):
        raise ValueError('regional_restore must be legacy or aligned')
    common = dict(d_model=d_model, n_heads=4, e_layers=1, dropout=0.1)
    if variant == 'omni':
        if regional_restore != 'legacy':
            raise ValueError('aligned Regional restore applies only to Hetero variants; '
                             'the Omni reference remains unchanged')
        return OmniTIEFormer(patch_len=2, seq_len=seq_len, pred_len=1, enc_in=1, **common)
    model = HeteroTIEFormer(seq_len=seq_len, routing=variant, tau=selector_tau,
                            selector_hard=selector_hard,
                            regional_restore=regional_restore, **common)
    if selector_init_std > 0 and model.selector is not None:
        nn = torch.nn
        torch.nn.init.normal_(model.selector[-1].weight, mean=0.0,
                              std=float(selector_init_std))
        torch.nn.init.zeros_(model.selector[-1].bias)
    return model


def crossing(q, cycles, start, threshold):
    """Return the first threshold crossing with linear interpolation.

    The paper defines EOL at the first segment crossing rather than at the
    first sampled cycle.  Keeping this helper shared by observed-history and
    free-running evaluation makes the two protocols differ only in how the
    input window is updated.
    """
    ids = np.flatnonzero(q <= threshold)
    if len(ids) == 0:
        return None
    i = int(ids[0])
    if i == 0:
        return float(cycles[start])
    y0, y1 = float(q[i - 1]), float(q[i])
    c0, c1 = float(cycles[start + i - 1]), float(cycles[start + i])
    if abs(y1 - y0) <= 1e-12:
        return c1
    alpha = (float(threshold) - y0) / (y1 - y0)
    return float(c0 + alpha * (c1 - c0))


def evaluate_one(model, q, cycles, start, norm_min, norm_range, eol_norm,
                 feedback_low, feedback_high, batch_size=256):
    truth = q[start:]
    x, _ = make_windows(q, np.arange(start, len(q)), horizon=1)
    with torch.no_grad():
        observed = torch.cat([
            model(batch).cpu().squeeze(-1).clamp(feedback_low, feedback_high)
            for batch in x.split(batch_size)
        ]).numpy()
        history = q[:start].tolist()
        recursive = []
        for _ in range(len(truth)):
            inp = torch.tensor(np.asarray(history[-WINDOW:]), dtype=torch.float32)[None, :, None]
            pred = model(inp).squeeze(-1).clamp(feedback_low, feedback_high).item()
            history.append(pred)
            recursive.append(pred)
    recursive = np.asarray(recursive)
    true_a = truth * norm_range + norm_min
    obs_a = observed * norm_range + norm_min
    rec_a = recursive * norm_range + norm_min
    actual = crossing(truth, cycles, start, eol_norm)
    obs_eol = crossing(observed, cycles, start, eol_norm)
    rec_eol = crossing(recursive, cycles, start, eol_norm)
    # Equation (2) in the paper defines RUL at the requested starting cycle:
    # RUL = EOL - SP.  Do not use SP-1 here (that introduces a one-cycle bias).
    origin = float(cycles[start])

    def r2(y, p):
        den = np.sum((y - y.mean()) ** 2)
        return None if den <= 1e-12 else float(1 - np.sum((y - p) ** 2) / den)

    def rul_metrics(pred_eol):
        if actual is None or pred_eol is None:
            return None, None
        ae = abs(actual - pred_eol)
        return float(ae), float(ae / (actual - origin))

    obs_ae, obs_re = rul_metrics(obs_eol)
    rec_ae, rec_re = rul_metrics(rec_eol)
    return {
        'requested_start_cycle': int(cycles[start]),
        'origin_cycle': origin,
        'actual_eol_cycle': actual,
        'observed_mae_Ah': float(np.abs(obs_a - true_a).mean()),
        'observed_rmse_Ah': float(np.sqrt(np.mean((obs_a - true_a) ** 2))),
        'observed_r2': r2(true_a, obs_a),
        'observed_pred_eol_cycle': obs_eol,
        'observed_rul_ae': obs_ae,
        'observed_rul_re': obs_re,
        'recursive_mae_Ah': float(np.abs(rec_a - true_a).mean()),
        'recursive_rmse_Ah': float(np.sqrt(np.mean((rec_a - true_a) ** 2))),
        'recursive_r2': r2(true_a, rec_a),
        'pred_eol_cycle': rec_eol,
        'recursive_rul_ae': rec_ae,
        'recursive_rul_re': rec_re,
    }


def aggregate(rows, prefix):
    out = {
        prefix + '_AMAE_Ah': float(np.mean([r[prefix + '_mae_Ah'] for r in rows])),
        prefix + '_ARMSE_Ah': float(np.mean([r[prefix + '_rmse_Ah'] for r in rows])),
    }
    r2s = [r[prefix + '_r2'] for r in rows if r[prefix + '_r2'] is not None]
    if r2s:
        out[prefix + '_AR2'] = float(np.mean(r2s))
    aes = [r[prefix + '_rul_ae'] for r in rows if r[prefix + '_rul_ae'] is not None]
    res = [r[prefix + '_rul_re'] for r in rows if r[prefix + '_rul_re'] is not None]
    if aes:
        out[prefix + '_AAE_cycles'] = float(np.mean(aes))
        out[prefix + '_ARE'] = float(np.mean(res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['omni', 'adaptive', 'uniform', 'residual'], default='adaptive')
    ap.add_argument('--regional-restore', choices=['legacy', 'aligned'], default='legacy',
                    help='Regional patch restoration; aligned fixes the time/feature '
                         'layout in Hetero candidates only (legacy preserves old runs)')
    ap.add_argument('--data-file', required=True)
    ap.add_argument('--window', type=int, default=64,
                    help='number of observations in each context window (default: 64)')
    ap.add_argument('--test-cell', required=True)
    ap.add_argument('--rated', type=float, required=True)
    ap.add_argument('--eol-ratio', type=float, default=0.7)
    ap.add_argument('--normalization', choices=['rated', 'train_minmax'], default='train_minmax')
    ap.add_argument('--start-cycles', type=int, nargs='+', required=True)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--d-model', type=int, default=16)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--torch-threads', type=int, default=1,
                    help='intra-op CPU threads for this process (default: 1)')
    ap.add_argument('--torch-interop-threads', type=int, default=1,
                    help='inter-op CPU threads for this process (default: 1)')
    ap.add_argument('--rollout-horizon', type=int, default=4)
    ap.add_argument('--train-objective', choices=['free_rollout', 'scheduled_sampling'],
                    default='free_rollout',
                    help='training objective; scheduled_sampling reproduces round1 V0.1')
    ap.add_argument('--rollout-loss-weight', type=float, default=1.0,
                    help='weight of the fully free-running rollout MAE in training')
    ap.add_argument('--delta-scale', type=float, default=10.0)
    ap.add_argument('--selector-tau', type=float, default=1.5)
    ap.add_argument('--selector-tau-final', type=float, default=None)
    ap.add_argument('--selector-hard', action='store_true')
    ap.add_argument('--selector-init-std', type=float, default=0.0)
    ap.add_argument('--selector-budget-weight', type=float, default=5e-4)
    ap.add_argument('--selector-entropy-weight', type=float, default=5e-3)
    ap.add_argument('--primary-protocol', choices=['observed_history', 'recursive'],
                    default='observed_history',
                    help='protocol used for checkpoint selection and the primary '
                         'summary; observed_history refreshes every window with '
                         'measured capacity')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.variant == 'omni' and args.regional_restore != 'legacy':
        ap.error('--regional-restore=aligned applies only to Hetero variants')
    if args.rollout_loss_weight < 0:
        raise ValueError('--rollout-loss-weight must be non-negative')
    if args.window < 2:
        raise ValueError('--window must be at least 2')
    if args.train_objective == 'scheduled_sampling' and args.rollout_loss_weight != 1.0:
        raise ValueError('--rollout-loss-weight must remain 1.0 with scheduled_sampling')

    if args.torch_threads < 1 or args.torch_interop_threads < 1:
        raise ValueError('torch thread counts must be positive')
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    global WINDOW
    WINDOW = int(args.window)
    seed_everything(args.seed)
    root = Path(__file__).resolve().parent
    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = root / data_path
    cells_raw = load_cells(data_path)
    if args.test_cell not in cells_raw:
        raise KeyError(args.test_cell)
    train_raw = np.concatenate([q for name, (_, q) in cells_raw.items() if name != args.test_cell])
    if args.normalization == 'train_minmax':
        norm_min, norm_max = float(train_raw.min()), float(train_raw.max())
    else:
        norm_min, norm_max = 0.0, float(args.rated)
    norm_range = norm_max - norm_min
    cells = {name: (c, (q - norm_min) / norm_range) for name, (c, q) in cells_raw.items()}
    train_cells = {k: v for k, v in cells.items() if k != args.test_cell}
    horizon = args.rollout_horizon
    train_pairs, val_pairs = [], []
    for _, (_, q) in train_cells.items():
        # Split the windows that actually exist after applying the context
        # length and forecast horizon.  Splitting the raw trajectory length
        # first leaves no training samples for short, sparsely sampled public
        # datasets such as Oxford (about 80 points per cell with WINDOW=64).
        available = np.arange(WINDOW, len(q) - horizon + 1)
        if len(available) < 2:
            raise ValueError(
                f'trajectory has too few usable windows: {len(q)} points, '
                f'horizon={horizon}, window={WINDOW}'
            )
        cut = int(np.ceil(0.8 * len(available)))
        cut = min(max(cut, 1), len(available) - 1)
        train_idx = available[:cut]
        val_idx = available[cut:]
        train_pairs.append(make_windows(q, train_idx, horizon))
        val_pairs.append(make_windows(q, val_idx, horizon))
    train_ds = TensorDataset(torch.cat([x for x, _ in train_pairs]), torch.cat([y for _, y in train_pairs]))
    val_ds = TensorDataset(torch.cat([x for x, _ in val_pairs]), torch.cat([y for _, y in val_pairs]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RelativeCapacityForecaster(
        build_model(args.variant, args.d_model, args.selector_tau,
                    args.selector_hard, args.selector_init_std,
                    regional_restore=args.regional_restore, seq_len=WINDOW),
        args.delta_scale,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    feedback_low, feedback_high = ((0.0, 1.0) if args.normalization == 'rated' else (-0.25, 1.25))
    best, stale = float('inf'), 0
    out = root / args.out
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), normalization_min=norm_min, normalization_max=norm_max,
                  normalization_range=norm_range, eol_threshold_Ah=args.rated * args.eol_ratio,
                  feedback_bounds=[feedback_low, feedback_high], train_samples=len(train_ds),
                  val_samples=len(val_ds), parameters=sum(p.numel() for p in model.parameters()))
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    with (out / 'epochs.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'epoch', 'train_objective', 'train_1step_mae', 'train_rollout_mae',
            'val_rollout_mae', 'val_1step_mae', 'monitor_mae'
        ])
        for epoch in range(args.epochs):
            tf = 1.0 + (0.2 - 1.0) * epoch / max(1, args.epochs - 1)
            if args.variant in ('adaptive', 'residual'):
                tau_final = args.selector_tau if args.selector_tau_final is None else args.selector_tau_final
                tau = args.selector_tau + (tau_final - args.selector_tau) * epoch / max(1, args.epochs - 1)
                model.base.set_selector(tau=tau)
            model.train()
            train_total = 0.0
            train_one_total = 0.0
            train_roll_total = 0.0
            for x, future in train_loader:
                opt.zero_grad()
                if args.train_objective == 'scheduled_sampling':
                    if args.variant in ('adaptive', 'residual'):
                        pred, gates = rollout(model, x, future, tf,
                                              feedback_low, feedback_high, True)
                    else:
                        pred = rollout(model, x, future, tf,
                                       feedback_low, feedback_high)
                        gates = []
                    # This is the historical round1 objective. The first
                    # rollout prediction is also the one-step prediction.
                    one_loss = (pred[:, 0] - future[:, 0]).abs().mean()
                    rollout_loss = (pred - future).abs().mean()
                    loss = rollout_loss
                else:
                    one_step = model(x).squeeze(-1)
                    if args.variant in ('adaptive', 'residual'):
                        # The second term is genuinely free-running: every
                        # feedback value comes from the model prediction
                        # (with stop-gradient through the feedback path).
                        pred, gates = rollout(model, x, future, 0.0,
                                              feedback_low, feedback_high, True)
                    else:
                        pred = rollout(model, x, future, 0.0,
                                       feedback_low, feedback_high)
                        gates = []
                    one_loss = (one_step - future[:, 0]).abs().mean()
                    rollout_loss = (pred - future).abs().mean()
                    loss = one_loss + args.rollout_loss_weight * rollout_loss
                reg = selector_regularization(gates)
                if reg is not None:
                    budget_penalty, entropy_penalty, _, _ = reg
                    loss = loss + args.selector_budget_weight * budget_penalty + args.selector_entropy_weight * entropy_penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
                opt.step()
                train_total += loss.item() * len(x)
                train_one_total += one_loss.item() * len(x)
                train_roll_total += rollout_loss.item() * len(x)
            model.eval(); val_total = 0.0; one_total = 0.0
            with torch.no_grad():
                for x, future in val_loader:
                    one_total += (model(x).squeeze(-1) - future[:, 0]).abs().sum().item()
                    val_total += (rollout(model, x, future, 0.0, feedback_low, feedback_high) - future).abs().sum().item()
            val_score = val_total / (len(val_ds) * horizon)
            one_score = one_total / len(val_ds)
            train_objective = train_total / len(train_ds)
            train_one = train_one_total / len(train_ds)
            train_roll = train_roll_total / len(train_ds)
            monitor_score = (one_score if args.primary_protocol == 'observed_history'
                             else val_score)
            writer.writerow([
                epoch + 1, train_objective, train_one, train_roll, val_score, one_score,
                monitor_score
            ]); fh.flush()
            print(f'epoch={epoch + 1} objective={args.train_objective} '
                  f'tf={tf:.3f} rollout_w={args.rollout_loss_weight:.3f} '
                  f'train={train_objective:.6f} train_1step={train_one:.6f} '
                  f'train_roll={train_roll:.6f} '
                  f'val_roll={val_score:.6f} val_1step={one_score:.6f} '
                  f'monitor[{args.primary_protocol}]={monitor_score:.6f}', flush=True)
            if monitor_score < best:
                best, stale = monitor_score, 0
                torch.save({'model': model.state_dict(), 'epoch': epoch + 1,
                            # Keep val_mae for backward compatibility; it is
                            # the recursive validation score used historically.
                            'val_mae': val_score,
                            'monitor_mae': best,
                            'monitor_protocol': args.primary_protocol,
                            'config': config}, out / 'model.pt')
            else:
                stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model']); model.eval()
    test_cycles, test_q = cells[args.test_cell]
    eol_norm = (args.rated * args.eol_ratio - norm_min) / norm_range
    rows = []
    for requested in args.start_cycles:
        start = int(np.searchsorted(test_cycles, requested))
        if start < WINDOW or start >= len(test_q):
            raise ValueError(f'bad start: {requested}')
        row = evaluate_one(model, test_q, test_cycles, start, norm_min, norm_range,
                           eol_norm, feedback_low, feedback_high, args.batch_size)
        rows.append(row)
    summary = {
        'variant': args.variant,
        'regional_restore': args.regional_restore,
        'primary_protocol': args.primary_protocol,
        'checkpoint_epoch': int(checkpoint['epoch']),
        'best_val_mae': float(checkpoint['val_mae']),
        'best_monitor_mae': float(checkpoint.get('monitor_mae', checkpoint['val_mae'])),
        'start_points': rows,
        'observed': aggregate(rows, 'observed'),
        'recursive': aggregate(rows, 'recursive'),
    }
    primary_key = 'observed' if args.primary_protocol == 'observed_history' else 'recursive'
    summary['primary'] = summary[primary_key]
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
