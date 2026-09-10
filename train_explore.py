"""Small, reproducible V0.1 exploration runner.

The runner keeps the original Omni network and uses a torch-only loop so that
the diagnostics are independent of pytorch-forecasting/Lightning.  It supports
the original Omni branch, V0.1 adaptive routing, a uniform gate control, and the
fine-scale residual routing probe.
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
                selector_init_std=0.0):
    common = dict(d_model=d_model, n_heads=4, e_layers=1, dropout=0.1)
    if variant == 'omni':
        return OmniTIEFormer(patch_len=2, seq_len=64, pred_len=1, enc_in=1, **common)
    model = HeteroTIEFormer(routing=variant, tau=selector_tau,
                            selector_hard=selector_hard, **common)
    if selector_init_std > 0 and model.selector is not None:
        nn = torch.nn
        torch.nn.init.normal_(model.selector[-1].weight, mean=0.0,
                              std=float(selector_init_std))
        torch.nn.init.zeros_(model.selector[-1].bias)
    return model


def crossing(q, cycles, start, threshold):
    ids = np.flatnonzero(q <= threshold)
    return None if len(ids) == 0 else float(cycles[start + ids[0]])


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
    origin = float(cycles[start - 1])

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
    ap.add_argument('--data-file', required=True)
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
    ap.add_argument('--rollout-horizon', type=int, default=4)
    ap.add_argument('--delta-scale', type=float, default=10.0)
    ap.add_argument('--selector-tau', type=float, default=1.5)
    ap.add_argument('--selector-tau-final', type=float, default=None)
    ap.add_argument('--selector-hard', action='store_true')
    ap.add_argument('--selector-init-std', type=float, default=0.0)
    ap.add_argument('--selector-budget-weight', type=float, default=5e-4)
    ap.add_argument('--selector-entropy-weight', type=float, default=5e-3)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
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
        cut = int(len(q) * 0.8)
        train_idx = np.arange(WINDOW, cut - horizon + 1)
        val_idx = np.arange(max(cut, WINDOW), len(q) - horizon + 1)
        train_pairs.append(make_windows(q, train_idx, horizon))
        val_pairs.append(make_windows(q, val_idx, horizon))
    train_ds = TensorDataset(torch.cat([x for x, _ in train_pairs]), torch.cat([y for _, y in train_pairs]))
    val_ds = TensorDataset(torch.cat([x for x, _ in val_pairs]), torch.cat([y for _, y in val_pairs]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RelativeCapacityForecaster(
        build_model(args.variant, args.d_model, args.selector_tau,
                    args.selector_hard, args.selector_init_std),
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
        writer.writerow(['epoch', 'train_mae', 'val_rollout_mae', 'val_1step_mae'])
        for epoch in range(args.epochs):
            tf = 1.0 + (0.2 - 1.0) * epoch / max(1, args.epochs - 1)
            if args.variant in ('adaptive', 'residual'):
                tau_final = args.selector_tau if args.selector_tau_final is None else args.selector_tau_final
                tau = args.selector_tau + (tau_final - args.selector_tau) * epoch / max(1, args.epochs - 1)
                model.base.set_selector(tau=tau)
            model.train(); train_total = 0.0
            for x, future in train_loader:
                opt.zero_grad()
                if args.variant in ('adaptive', 'residual'):
                    pred, gates = rollout(model, x, future, tf, feedback_low, feedback_high, True)
                else:
                    pred = rollout(model, x, future, tf, feedback_low, feedback_high)
                    gates = []
                loss = (pred - future).abs().mean()
                reg = selector_regularization(gates)
                if reg is not None:
                    budget_penalty, entropy_penalty, _, _ = reg
                    loss = loss + args.selector_budget_weight * budget_penalty + args.selector_entropy_weight * entropy_penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
                opt.step()
                train_total += loss.item() * len(x)
            model.eval(); val_total = 0.0; one_total = 0.0
            with torch.no_grad():
                for x, future in val_loader:
                    one_total += (model(x).squeeze(-1) - future[:, 0]).abs().sum().item()
                    val_total += (rollout(model, x, future, 0.0, feedback_low, feedback_high) - future).abs().sum().item()
            val_score = val_total / (len(val_ds) * horizon)
            one_score = one_total / len(val_ds)
            writer.writerow([epoch + 1, train_total / len(train_ds), val_score, one_score]); fh.flush()
            print(f'epoch={epoch + 1} tf={tf:.3f} train={train_total / len(train_ds):.6f} '
                  f'val_roll={val_score:.6f} val_1step={one_score:.6f}', flush=True)
            if val_score < best:
                best, stale = val_score, 0
                torch.save({'model': model.state_dict(), 'epoch': epoch + 1,
                            'val_mae': best, 'config': config}, out / 'model.pt')
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
        'checkpoint_epoch': int(checkpoint['epoch']),
        'best_val_mae': float(checkpoint['val_mae']),
        'start_points': rows,
        'observed': aggregate(rows, 'observed'),
        'recursive': aggregate(rows, 'recursive'),
    }
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
