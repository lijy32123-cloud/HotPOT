"""Simulated combustion: data access and the three models compared in the appendix.

Every model reads the past 10 frames of the same 15 fields and nothing else: no fuel blend, equivalence ratio or
trajectory label, no future frame, no hidden solver state.

  moepot               MoE-POT-T (official code) with new 15-field input and output layers
  hotpot_direct        shared pre-trained history encoder + a new private predictor of the next state
  hotpot_increment     same network as hotpot_direct; the output is the increment, the update adds it to the last frame

Environment:
  COMBUSTION_CACHE     the .npy written by prepare_data.py (its .json lies next to it)
  HOTPOT_T_CHECKPOINT  HotPOT-T_pretrain_E1000.pt
  MOEPOT_CODE          root of the official MoE-POT repository (contains models/moepot.py)
  MOEPOT_T_CHECKPOINT  MoE-POT-T pre-trained on the same six datasets ({'model': state_dict})
"""
import hashlib
import importlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get('HOTPOT_CODE', str(BASE.parents[1])))  # the repository root (package `hotpot`)
ARMS = ('moepot', 'hotpot_direct', 'hotpot_increment')
FIELDS = ('p_abs', 'heat_release', 'CH4', 'CO', 'CO2', 'H2O', 'NH2', 'NH3', 'OH', 'p_gauge', 'T', 'u', 'v', 'w', 'speed')
SEED = 20261108


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    p = Path(path)
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(obj, indent=2, allow_nan=False))
    q.replace(p)


def state_hash(model):
    h = hashlib.sha256()
    for name, v in model.state_dict().items():
        h.update(name.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class HotPOTDirect(nn.Module):
    def __init__(self):
        super().__init__()
        from hotpot._internal.foundation.config import tiny_17m_config
        from hotpot._internal.general_fields.model import GeneralSemanticFieldChart, TypedSavedStepHead
        from hotpot._internal.encoder.analyzer import SharedScalarFieldAnalyzer
        from hotpot._internal.encoder.private_mixer import PrivateTypedFieldMixer
        from hotpot._internal.encoder.action_decoder import ActionContextDecoder
        from hotpot._internal.history_features.nodiff_chart import install_no_difference_chart
        from hotpot._internal.data_protocol.manifest import CONDITION_FIELDS
        config = replace(tiny_17m_config(), max_fields=15)
        self.chart = GeneralSemanticFieldChart(config, dynamic_channels=15, boundary='closed')
        self.shared_analyzer = SharedScalarFieldAnalyzer(config)
        self.private_mixer = PrivateTypedFieldMixer(config, dynamic_channels=15)
        self.decoder = ActionContextDecoder(config)
        self.head = TypedSavedStepHead(config, channels=15, boundary='closed')
        self.nconditions = len(CONDITION_FIELDS)
        install_no_difference_chart(self)

    def forward(self, history):
        values = history.new_zeros(len(history), self.nconditions)
        mask = torch.zeros_like(values)
        features, field_mask, mean, log_scale, lags = self.chart(history)
        memory = self.shared_analyzer(features, field_mask, lags)
        private = self.private_mixer(memory, field_mask, mean, log_scale, values, mask)
        current = history[..., -1, :].permute(0, 3, 1, 2).contiguous()
        return self.head(self.decoder(private), current, values, mask).permute(0, 2, 3, 1)


class MoEAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.core = model

    def forward(self, h):
        return self.core(h)[0][..., 0, :]


@contextmanager
def _official_moepot():
    """The official MoEPOTNet class, imported from MOEPOT_CODE without leaving `models.*` in sys.modules."""
    root = Path(os.environ['MOEPOT_CODE']).resolve(strict=True)
    before_path = list(sys.path)
    before = {n: m for n, m in sys.modules.items() if n == 'models' or n.startswith('models.')}
    for n in before:
        sys.modules.pop(n)
    try:
        sys.path.insert(0, str(root))
        yield importlib.import_module('models.moepot').MoEPOTNet
    finally:
        for n in list(sys.modules):
            if n == 'models' or n.startswith('models.'):
                sys.modules.pop(n, None)
        sys.modules.update(before)
        sys.path[:] = before_path


def _build_direct(seed):
    torch.manual_seed(seed)
    source = Path(os.environ['HOTPOT_T_CHECKPOINT'])
    model = HotPOTDirect()
    snapshot = torch.load(source, map_location='cpu', weights_only=True)
    state = next((snapshot[k] for k in ['model', 'model_state', 'state_dict', 'full_model_state'] if k in snapshot), snapshot)
    prefix = 'model.shared_analyzer.'
    encoder = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    assert encoder, 'the checkpoint holds no shared history encoder'
    model.shared_analyzer.load_state_dict(encoder, strict=True)
    copied = sum(v.numel() for v in encoder.values())
    return model, source, copied, 'pre-trained shared history encoder; new 15-field mixer, decoder and head'


def _build_moepot(seed):
    torch.manual_seed(seed)
    source = Path(os.environ['MOEPOT_T_CHECKPOINT'])
    with _official_moepot() as cls:
        core = cls(img_size=128, patch_size=8, mixing_type='afno', in_channels=4, out_channels=4, in_timesteps=10, out_timesteps=1,
                   n_blocks=4, embed_dim=512, out_layer_dim=32, depth=4, modes=32, mlp_ratio=1, n_cls=6, normalize=False, act='gelu',
                   time_agg='exp_mlp', is_finetune=True)
    snapshot = torch.load(source, map_location='cpu', weights_only=False)
    core.load_state_dict(snapshot['model'], strict=True)
    # Only the input and output channel interfaces required by 15 fields are re-initialised.
    old = core.patch_embed.proj[0]
    core.patch_embed.proj[0] = nn.Conv2d(18, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding)
    old_out = core.out_layer[-1]
    core.out_layer[-1] = nn.Conv2d(old_out.in_channels, 15, kernel_size=old_out.kernel_size, stride=old_out.stride, padding=old_out.padding)
    core.in_channels = 15
    core.out_channels = 15
    core.requires_grad_(True)
    core.cls_head.requires_grad_(False)  # the dataset classifier is not part of this target's loss
    copied = sum(v.numel() for k, v in snapshot['model'].items() if not k.startswith(('patch_embed.proj.0.', 'out_layer.4.')))
    assert [float(b.MoE.gating.temperature) for b in core.blocks] == [.5] * 4
    return MoEAdapter(core), source, copied, 'pre-trained temporal/spatial/MoE trunk; new 15-field input and output layers; router trainable'


def build(arm, normalization=None, seed=SEED):
    """Model initialised from its six-dataset pre-training checkpoint, and a ledger of what was reused.

    `normalization` is accepted for a uniform call signature; the three models do not need it to be built."""
    assert arm in ARMS
    if arm == 'moepot':
        model, source, copied, adaptation = _build_moepot(seed)
    else:
        model, source, copied, adaptation = _build_direct(seed)
        if arm == 'hotpot_increment':
            adaptation += '; the head output is the increment of the standardised state'
    model.combustion_arm = arm
    ledger = {'arm': arm, 'source': source.name, 'copied_state_elements': copied,
              'stored_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'initialization_sha256': state_hash(model), 'adaptation': adaptation}
    return model, ledger


def _network(model, h):
    # Data and scoring at the native 64x64; both pre-trained architectures run at 128x128.
    b, hh, ww, t, c = h.shape
    packed = h.permute(0, 3, 4, 1, 2).reshape(b, t * c, hh, ww)
    up = F.interpolate(packed, size=(128, 128), mode='bilinear', align_corners=False)
    up = up.reshape(b, t, c, 128, 128).permute(0, 3, 4, 1, 2).contiguous()
    result = model(up).permute(0, 3, 1, 2)
    return F.interpolate(result, size=(hh, ww), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()


def predict(model, h):
    """h: (batch, 64, 64, 10, 15) standardised history -> (batch, 64, 64, 15) standardised next frame."""
    out = _network(model, h)
    return h[..., -1, :] + out if model.combustion_arm == 'hotpot_increment' else out


class CombustionData:
    def __init__(self):
        cache = Path(os.environ['COMBUSTION_CACHE'])
        self.array = np.load(cache, mmap_mode='r')
        meta = json.loads(cache.with_suffix('.json').read_text())
        assert self.array.shape == (30, 2001, 16, 64, 64) and meta['fields'] == ['observed', 'numerical']
        self.ids = meta['sim_ids']
        self.split = json.loads((BASE / 'SPLIT_MANIFEST.json').read_text())
        self.train = [self.ids.index(n) for n in self.split['train']]
        self.val = [self.ids.index(n) for n in self.split['validation']]
        self.test = [self.ids.index(n) for n in self.split['test']]
        assert len(self.train) == 18 and len(self.val) == 3 and len(self.test) == 3
        self.mu = None
        self.std = None

    def raw(self, row, start, count):
        # Channel 0 of the cache is the surrogate-derived "observed" field: excluded on purpose.
        x = np.array(self.array[row, start:start + count, 1:16], copy=True)
        assert np.isfinite(x).all()
        return torch.from_numpy(x).permute(2, 3, 0, 1).contiguous()

    def fit_normalization(self):
        total = np.zeros(15, np.float64)
        square = np.zeros(15, np.float64)
        n = 0
        for row in self.train:
            for start in range(0, 2001, 32):
                a = np.asarray(self.array[row, start, 1:16], dtype=np.float64)
                assert np.isfinite(a).all()
                total += a.sum(axis=(1, 2))
                square += (a * a).sum(axis=(1, 2))
                n += 64 * 64
        mu = total / n
        rms = np.sqrt(square / n)
        std = np.sqrt(np.maximum(square / n - mu * mu, 0))
        std = np.maximum(std, 1e-6 * np.maximum(rms, 1e-6))
        return {'mean': mu.tolist(), 'std': std.tolist(), 'rms': rms.tolist(), 'fit_rows': self.train,
                'fit_sim_ids': [self.ids[i] for i in self.train], 'frames': list(range(0, 2001, 32)), 'observed_channel_used': False}

    def set_normalization(self, norm):
        self.mu = torch.tensor(norm['mean'])
        self.std = torch.tensor(norm['std'])

    def normalize(self, x):
        return (x - self.mu.to(x.device)) / self.std.to(x.device)

    def unnormalize(self, x):
        return x * self.std.to(x.device) + self.mu.to(x.device)

    def batch(self, step, batch=8):
        # Counter-based sampling: every model sees the same windows at the same update.
        rng = np.random.default_rng(np.random.SeedSequence([SEED, step]))
        rows = rng.choice(self.train, batch)
        starts = rng.integers(0, 1991, size=batch)
        x = torch.stack([self.raw(int(r), int(t), 11) for r, t in zip(rows, starts)])
        return self.normalize(x[..., :10, :]), self.normalize(x[..., 10, :]), rows.tolist(), starts.tolist()


@torch.inference_mode()
def rollout_error(model, data, rows, starts, horizon, device='cuda'):
    """Closed-loop rollout scored two ways: per-field relative L2 in physical units and normalised MSE."""
    model.eval()
    records = []
    for row in rows:
        for start in starts:
            raw = data.raw(row, start, 10 + horizon).to(device)
            z = data.normalize(raw)
            h = z[None, ..., :10, :]
            predictions = []
            for _ in range(horizon):
                pred = predict(model, h)
                if not torch.isfinite(pred).all():
                    raise FloatingPointError('nonfinite rollout')
                predictions.append(pred[0])
                h = torch.cat([h[..., 1:, :], pred.unsqueeze(-2)], -2)
            p = torch.stack(predictions, 2)
            truth = raw[..., 10:, :]
            delta = data.unnormalize(p) - truth
            den = truth.square().sum((0, 1, 2)).sqrt()
            floor = data.std.to(device) * 1e-6 * (64 * 64 * horizon) ** .5
            error = (delta.square().sum((0, 1, 2)).sqrt() / torch.maximum(den, floor)).cpu().tolist()
            nmse = (p - z[..., 10:, :]).square().mean((0, 1, 2))
            records.append({'sim_id': data.ids[row], 'start': start, 'per_channel_l2re': error, 'mean_l2re': float(np.mean(error)),
                            'normalized_mse': float(nmse.mean()), 'per_channel_normalized_mse': nmse.cpu().tolist()})
    return {'horizon': horizon, 'records': records, 'mean_l2re': float(np.mean([r['mean_l2re'] for r in records])),
            'normalized_mse': float(np.mean([r['normalized_mse'] for r in records]))}
