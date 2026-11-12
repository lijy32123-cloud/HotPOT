from __future__ import annotations


import hashlib


import json


import torch


from torch import Tensor, nn


from hotpot._internal.data_protocol.manifest import CONDITION_FIELDS


from hotpot._internal.foundation.config import tiny_17m_config


from hotpot._internal.encoder.action_decoder import ActionContextDecoder


from hotpot._internal.encoder.analyzer import SharedScalarFieldAnalyzer


from hotpot._internal.encoder.private_mixer import PrivateTypedFieldMixer


from hotpot._internal.general_fields.model import GeneralSemanticFieldChart, TypedSavedStepHead


ARMS = ("hetpot_transfer", "hetpot_scratch", "moepot_transfer")


TARGET_SHARED_PREFIX = "shared_analyzer."


def _check_history(history: Tensor) -> None:
    if history.ndim != 5 or tuple(history.shape[1:]) != (128, 128, 10, 1):
        raise ValueError("Wave history must be [B,128,128,10,1]")
    if history.shape[0] < 1 or not history.is_floating_point():
        raise ValueError("Wave history requires a nonempty floating-point batch")


class WaveResidualModel(nn.Module):
    """Shared H10 analysis, one random private branch, and u_next = u + delta."""

    def __init__(self) -> None:
        super().__init__()
        config = tiny_17m_config()
        self.config = config
        # Keep this exact constructor order for both HetPOT arms.
        self.chart = GeneralSemanticFieldChart(config, dynamic_channels=1, boundary="closed")
        self.shared_analyzer = SharedScalarFieldAnalyzer(config)
        self.private_mixer = PrivateTypedFieldMixer(config, dynamic_channels=1)
        self.decoder = ActionContextDecoder(config)
        self.head = TypedSavedStepHead(config, channels=1, boundary="closed")

    def forward(self, history: Tensor) -> Tensor:
        _check_history(history)
        # Missing physical information: zero values AND zero availability masks.
        values = history.new_zeros(history.shape[0], len(CONDITION_FIELDS))
        mask = torch.zeros_like(values)
        features, field_mask, mean, log_scale, lags = self.chart(history)
        memory = self.shared_analyzer(features, field_mask, lags)
        private = self.private_mixer(memory, field_mask, mean, log_scale, values, mask)
        current = history[..., -1, :].permute(0, 3, 1, 2).contiguous()
        delta = self.head(self.decoder(private), current, values, mask)
        # A saved-step displacement increment, not velocity or a PDE derivative.
        return (current + delta).permute(0, 2, 3, 1).contiguous()


def _state_digest(model: nn.Module, *, skip_shared: bool = False) -> str:
    """Hash selected model state, including buffers, with exact dtype and bytes.

    Format v1: sorted state keys; length-prefixed compact JSON [name,dtype,shape]
    followed by length-prefixed contiguous CPU tensor bytes (8-byte big endian
    lengths). No floating-point cast or parameter-only filter is performed.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if skip_shared and name.startswith(TARGET_SHARED_PREFIX):
            continue
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            [name, str(value.dtype), list(value.shape)], separators=(",", ":")
        ).encode("utf-8")
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (header, raw):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def predict(model: nn.Module, arm: str, history: Tensor) -> Tensor:
    """Differentiable one-step prediction; returns [B,128,128,1]."""
    if arm not in ARMS:
        raise ValueError(f"unknown Wave arm: {arm!r}")
    _check_history(history)
    if arm == "moepot_transfer":
        padded = history.new_ones(*history.shape[:-1], 4)
        padded[..., :1] = history
        prediction = model(padded)[0][..., 0, :1]
    else:
        prediction = model(history)
    if tuple(prediction.shape) != (history.shape[0], 128, 128, 1):
        raise RuntimeError(f"Wave prediction ABI mismatch: {tuple(prediction.shape)}")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Wave prediction is non-finite")
    return prediction
