"""Dense adaptive-scale wrapper used for the V0.1 exploration.

The four candidate branches keep OmniTIEFormer's Local/Regional operators and
are evaluated in parallel.  A shared selector chooses a soft mixture for each
16-cycle region, so every routed branch remains a dense ``[B, L, M, D]``
tensor.  ``residual`` routing is an exploratory variant that keeps the fine
candidate as an anchor and learns only a gated correction from coarser scales.
"""

import torch
from torch import nn
from torch.nn import functional as F

from OmniTIEFormer import (
    OmniTIEFormer,
    LocalHybridFusionAttention,
    RegionalChannelAttention,
)


class CandidateBranch(nn.Module):
    """One fixed-patch Local/Regional candidate with Omni-compatible shapes."""

    extract_local_features = OmniTIEFormer.extract_local_features
    extract_regional_features = OmniTIEFormer.extract_regional_features

    def __init__(self, patch_size, seq_len, d_model, enc_in=1):
        super().__init__()
        if seq_len % patch_size:
            raise ValueError('seq_len must be divisible by patch_size')
        self.patch_size = patch_size
        self.patch_stride = patch_size
        self.patch_nums = seq_len // patch_size
        self.local_anchor_embeddings = nn.Parameter(
            torch.rand(self.patch_nums, 1, 1, enc_in, 16)
        )
        self.local_embedding_generators = nn.ModuleList(
            [nn.Sequential(nn.Linear(16, d_model)) for _ in range(self.patch_nums)]
        )
        self.lhfa = LocalHybridFusionAttention(
            channel=patch_size + 1,
            reduction=max(1, (patch_size + 1) // 16),
            kernel_size=min(3, patch_size + 1),
            HW=enc_in * d_model,
        )
        self.local_feature_projection = nn.Linear(self.patch_nums, seq_len)
        self.rca = RegionalChannelAttention(self.patch_nums)

    def forward(self, embedded_x):
        return self.extract_local_features(embedded_x), self.extract_regional_features(embedded_x)


class HeteroTIEFormer(OmniTIEFormer):
    """Omni with a shared region-wise scale selector.

    ``routing='adaptive'`` is the V0.1 model.  ``routing='uniform'`` is a
    diagnostic fixed mixture.  ``routing='residual'`` retains the P=2 branch
    and adds a bounded correction from the coarser candidate branches.
    """

    def __init__(
        self,
        seq_len=64,
        d_model=16,
        routing='adaptive',
        tau=1.5,
        selector_hard=False,
        residual_gain_init=0.25,
        **kwargs,
    ):
        if seq_len % 16 or seq_len < 32:
            raise ValueError('seq_len must be divisible by 16 and >=32')
        if routing not in ('adaptive', 'uniform', 'residual'):
            raise ValueError(f'unknown routing mode: {routing}')
        if tau <= 0:
            raise ValueError('selector temperature must be positive')
        super().__init__(
            patch_len=2,
            seq_len=seq_len,
            pred_len=1,
            enc_in=1,
            d_model=d_model,
            **kwargs,
        )

        # The parent fixed-patch modules are replaced by the candidate bank.
        for name in (
            'local_anchor_embeddings',
            'local_embedding_generators',
            'lhfa',
            'local_feature_projection',
            'rca',
        ):
            delattr(self, name)

        self.scales = (2, 4, 8, 16)
        self.seq_len = seq_len
        self.routing = routing
        self.tau = float(tau)
        self.selector_hard = bool(selector_hard)
        self.bank = nn.ModuleList(
            [CandidateBranch(p, seq_len, d_model) for p in self.scales]
        )
        self.selector = (
            nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))
            if routing in ('adaptive', 'residual') else None
        )
        if self.selector is not None:
            # Neutral initialization prevents seed-dependent P=2 collapse.
            nn.init.zeros_(self.selector[-1].weight)
            nn.init.zeros_(self.selector[-1].bias)

        gain = max(1e-4, min(0.999, float(residual_gain_init)))
        self.residual_gain_logit = nn.Parameter(
            torch.tensor(gain / (1.0 - gain)).log()
        )
        # Evaluation-only controls used to test whether improvements come from
        # learned region routing rather than from the extra candidate bank.
        self.gate_mode = 'learned'
        self.latest_gate = None
        self.latest_gate_probs = None

    def set_selector(self, tau=None, hard=None):
        if tau is not None:
            if tau <= 0:
                raise ValueError('selector temperature must be positive')
            self.tau = float(tau)
        if hard is not None:
            self.selector_hard = bool(hard)

    def set_gate_mode(self, mode='learned'):
        """Select the gate used at evaluation time.

        ``learned`` is the normal V0.1 behavior.  ``uniform`` removes
        region-wise routing, ``reverse`` mirrors scale probabilities, and
        ``region_shuffle`` moves each region's learned decision to a different
        region.  These controls do not change trainable parameters.
        """
        allowed = {'learned', 'uniform', 'reverse', 'region_shuffle'}
        if mode not in allowed:
            raise ValueError(f'unknown gate mode: {mode}')
        self.gate_mode = mode

    @staticmethod
    def route(candidates, gate):
        # candidates: [B, K, L, M, D], gate: [B, R, K]
        b, _, length, _, _ = candidates.shape
        if length % gate.shape[1]:
            raise ValueError('candidate length must be divisible by region count')
        region_len = length // gate.shape[1]
        weights = gate.repeat_interleave(region_len, dim=1)
        weights = weights.transpose(1, 2)[..., None, None]
        return (candidates * weights).sum(dim=1)

    def _gate(self, x):
        regions = x[:, :, 0].reshape(x.shape[0], -1, 16)
        if self.gate_mode == 'uniform' or self.selector is None:
            gate = x.new_full((*regions.shape[:2], 4), 0.25)
        else:
            logits = self.selector(regions)
            if self.selector_hard:
                gate = (
                    F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
                    if self.training
                    else F.one_hot(logits.argmax(-1), 4).to(x.dtype)
                )
            else:
                gate = F.softmax(logits / self.tau, dim=-1)
            if self.gate_mode == 'reverse':
                gate = gate.flip(-1)
            elif self.gate_mode == 'region_shuffle':
                gate = gate.roll(shifts=1, dims=1)
        self.latest_gate_probs = gate
        self.latest_gate = gate.detach()
        return gate

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, 1):
            raise ValueError(f'Expected [B, {self.seq_len}, 1], got {tuple(x.shape)}')
        gate = self._gate(x)
        embedded = self.input_embedding(x)
        pairs = [branch(embedded) for branch in self.bank]
        local_candidates = torch.stack([pair[0] for pair in pairs], dim=1)
        regional_candidates = torch.stack([pair[1] for pair in pairs], dim=1)

        if self.routing == 'residual':
            gain = torch.sigmoid(self.residual_gain_logit)
            local_base = local_candidates[:, 0]
            regional_base = regional_candidates[:, 0]
            local_delta = local_candidates[:, 1:] - local_base[:, None]
            regional_delta = regional_candidates[:, 1:] - regional_base[:, None]
            coarse_gate = gate[:, :, 1:].repeat_interleave(16, dim=1)
            coarse_gate = coarse_gate.transpose(1, 2)[..., None, None]
            local = local_base + gain * (local_delta * coarse_gate).sum(dim=1)
            regional = regional_base + gain * (regional_delta * coarse_gate).sum(dim=1)
        else:
            local = self.route(local_candidates, gate)
            regional = self.route(regional_candidates, gate)
        return self.fuse_and_predict(local, regional, self.extract_global_features(embedded))
