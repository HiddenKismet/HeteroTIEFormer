"""Regional alignment regression tests; no external dataset is required."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

import diagnose_gate_controls
import train_explore
from HeteroTIEFormer import CandidateBranch, HeteroTIEFormer
from OmniTIEFormer import OmniTIEFormer
from train_explore import RelativeCapacityForecaster, build_model, split_window_indices


SCALES = (2, 4, 8, 16)


class PatchIndexWeight(nn.Module):
    def forward(self, x):
        weights = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)
        return x * weights[None, :, None, None]


class RegionalRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identity_round_trip_all_scales_and_multiple_variables(self):
        for p in SCALES:
            for b, m, d in ((2, 1, 16), (3, 2, 5), (1, 3, 7)):
                with self.subTest(p=p, batch=b, variables=m, features=d):
                    # M > 1 also exercises a non-contiguous input.
                    x = torch.arange(b * m * 64 * d, dtype=torch.float32)
                    x = x.reshape(b, m, 64, d).transpose(1, 2)
                    branch = CandidateBranch(p, 64, d, m, regional_restore='aligned')
                    branch.rca = nn.Identity()
                    torch.testing.assert_close(
                        branch.extract_regional_features(x), 2 * x, rtol=0, atol=0
                    )

    def test_target_disjoint_split_for_multistep_horizon(self):
        train, val, validation_start = split_window_indices(96, 64, 4)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        train_targets = {i for t in train for i in range(int(t), int(t) + 4)}
        val_targets = {i for t in val for i in range(int(t), int(t) + 4)}
        self.assertTrue(train_targets.isdisjoint(val_targets))
        self.assertEqual(validation_start, int(val[0]))
        self.assertEqual(max(train_targets), validation_start - 1)

    def test_target_disjoint_split_rejects_short_sparse_trajectory(self):
        with self.assertRaises(ValueError):
            split_window_indices(22, 16, 4)

    def test_patch_weights_return_to_their_own_time_positions(self):
        for p in SCALES:
            with self.subTest(p=p):
                x = torch.arange(2 * 64 * 3 * 5, dtype=torch.float32).reshape(2, 64, 3, 5)
                branch = CandidateBranch(p, 64, 5, 3, regional_restore='aligned')
                branch.rca = PatchIndexWeight()
                weights = (torch.arange(64) // p + 1)[None, :, None, None]
                torch.testing.assert_close(
                    branch.extract_regional_features(x), x * weights + x, rtol=0, atol=0
                )

    def test_identity_input_gradient(self):
        for p in SCALES:
            with self.subTest(p=p):
                x = torch.randn(2, 64, 2, 5, requires_grad=True)
                branch = CandidateBranch(p, 64, 5, 2, regional_restore='aligned')
                branch.rca = nn.Identity()
                branch.extract_regional_features(x).sum().backward()
                torch.testing.assert_close(x.grad, torch.full_like(x, 2), rtol=0, atol=0)

    def test_real_rca_receives_finite_gradients(self):
        for p in SCALES:
            with self.subTest(p=p):
                torch.manual_seed(42)
                x = torch.randn(2, 64, 2, 5, requires_grad=True)
                branch = CandidateBranch(p, 64, 5, 2, regional_restore='aligned')
                branch.extract_regional_features(x).square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for param in branch.rca.parameters():
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in branch.rca.parameters()), 0)

    def test_legacy_matches_upstream_exactly(self):
        for p in SCALES:
            with self.subTest(p=p):
                branch = CandidateBranch(p, 64, 16)
                x = torch.randn(2, 64, 1, 16)
                torch.testing.assert_close(
                    branch.extract_regional_features(x),
                    OmniTIEFormer.extract_regional_features(branch, x), rtol=0, atol=0
                )

    def test_fix_does_not_change_local_branch_or_rca_parameters(self):
        for p in SCALES:
            with self.subTest(p=p):
                legacy = CandidateBranch(p, 64, 16)
                aligned = CandidateBranch(p, 64, 16, regional_restore='aligned')
                aligned.load_state_dict(legacy.state_dict(), strict=True)
                x = torch.randn(2, 64, 1, 16)
                torch.testing.assert_close(
                    legacy.extract_local_features(x), aligned.extract_local_features(x), rtol=0, atol=0
                )
                self.assertEqual(list(legacy.state_dict()), list(aligned.state_dict()))

    def test_all_routing_variants_support_aligned_forward_backward(self):
        for variant in ('adaptive', 'uniform', 'residual'):
            with self.subTest(variant=variant):
                torch.manual_seed(42)
                model = RelativeCapacityForecaster(build_model(variant, 16, regional_restore='aligned'))
                x = torch.randn(2, 64, 1) * 0.01 + 0.8
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                prediction = model(x)
                self.assertEqual(tuple(prediction.shape), (2, 1))
                loss = (prediction - 0.7).square().mean()
                loss.backward()
                if variant != 'uniform':
                    grad = model.base.selector[-1].weight.grad
                    self.assertIsNotNone(grad)
                    self.assertGreater(float(grad.abs().sum()), 0)
                optimizer.step()
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                self.assertTrue(all(b.regional_restore == 'aligned' for b in model.base.bank))

    def test_parameter_layout_and_initialization_are_unchanged(self):
        torch.manual_seed(42)
        legacy = build_model('adaptive', 16)
        torch.manual_seed(42)
        aligned = build_model('adaptive', 16, regional_restore='aligned')
        self.assertEqual(list(legacy.state_dict()), list(aligned.state_dict()))
        for key, value in legacy.state_dict().items():
            torch.testing.assert_close(value, aligned.state_dict()[key], rtol=0, atol=0)

    def test_checkpoint_round_trip_and_legacy_default(self):
        for restore in ('legacy', 'aligned'):
            with self.subTest(restore=restore):
                config = {'variant': 'adaptive', 'd_model': 16}
                if restore == 'aligned':
                    config['regional_restore'] = restore
                model = build_model('adaptive', 16, regional_restore=restore).eval()
                buffer = io.BytesIO()
                torch.save({'model': model.state_dict(), 'config': config}, buffer)
                buffer.seek(0)
                checkpoint = torch.load(buffer, map_location='cpu', weights_only=True)
                cfg = checkpoint['config']
                restored = build_model(cfg['variant'], cfg['d_model'],
                                       regional_restore=cfg.get('regional_restore', 'legacy')).eval()
                restored.load_state_dict(checkpoint['model'], strict=True)
                x = torch.randn(2, 64, 1)
                with torch.no_grad():
                    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_invalid_modes_and_omni_alignment_are_rejected(self):
        with self.assertRaises(ValueError):
            CandidateBranch(2, 64, 16, regional_restore='invalid')
        with self.assertRaises(ValueError):
            HeteroTIEFormer(regional_restore='invalid')
        with self.assertRaises(ValueError):
            build_model('adaptive', 16, regional_restore='invalid')
        with self.assertRaises(ValueError):
            build_model('omni', 16, regional_restore='aligned')
        self.assertIsInstance(build_model('omni', 16), OmniTIEFormer)

        model = build_model('adaptive', 16)
        x = torch.randn(2, 64, 1)
        for scale, index in ((2, 0), (4, 1), (8, 2), (16, 3)):
            model.set_gate_mode(f'force_p{scale}')
            with torch.no_grad():
                model(x)
            expected = torch.zeros_like(model.latest_gate)
            expected[..., index] = 1
            torch.testing.assert_close(model.latest_gate, expected, rtol=0, atol=0)

    def test_training_and_diagnostic_cli_preserve_restore_mode(self):
        # One tiny synthetic epoch checks integration, not experimental accuracy.
        cycles = np.arange(1, 97, dtype=float)
        cells = {name: (cycles, 2.0 - 0.003 * cycles + offset)
                 for name, offset in (('train_a', 0.0), ('train_b', 0.02), ('test', -0.01))}
        with tempfile.TemporaryDirectory(prefix='heterotie-restore-test-') as tmp:
            run = Path(tmp) / 'smoke'
            argv = ['train_explore.py', '--variant', 'adaptive', '--regional-restore', 'aligned',
                    '--data-file', 'unused.npy', '--test-cell', 'test', '--rated', '2.0',
                    '--start-cycles', '80', '--epochs', '1', '--train-objective', 'one_step',
                    '--out', str(run)]
            with mock.patch('sys.argv', argv), mock.patch.object(train_explore, 'load_cells', return_value=cells), \
                    mock.patch('torch.set_num_interop_threads'), contextlib.redirect_stdout(io.StringIO()):
                train_explore.main()
            config = json.loads((run / 'config.json').read_text())
            metrics = json.loads((run / 'metrics.json').read_text())
            checkpoint = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
            self.assertEqual(config['patience'], 60)
            self.assertEqual(config['seed'], 42)
            self.assertEqual(config['regional_restore'], 'aligned')
            self.assertEqual(config['train_objective'], 'one_step')
            self.assertEqual(config['split_policy'], 'target_disjoint')
            self.assertEqual(config['normalization_source'], 'train_cells_train_target_prefix')
            self.assertEqual(checkpoint['config']['regional_restore'], 'aligned')
            self.assertEqual(metrics['regional_restore'], 'aligned')
            self.assertEqual(checkpoint['epoch'], metrics['checkpoint_epoch'])
            argv = ['diagnose_gate_controls.py', '--run', str(run), '--data-file', 'unused.npy',
                    '--test-cell', 'test', '--start-cycles', '80', '--modes', 'learned']
            with mock.patch('sys.argv', argv), mock.patch.object(diagnose_gate_controls, 'load_cells', return_value=cells), \
                    mock.patch('torch.set_num_interop_threads'), contextlib.redirect_stdout(io.StringIO()):
                diagnose_gate_controls.main()
            result = json.loads((run / 'gate_controls.json').read_text())['learned']
            self.assertEqual(result['regional_restore'], 'aligned')
            self.assertEqual(result['observed'], metrics['observed'])
            self.assertEqual(result['recursive'], metrics['recursive'])


if __name__ == '__main__':
    unittest.main()
