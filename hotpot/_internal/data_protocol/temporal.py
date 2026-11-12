from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .factorized_core import (
    FactorizedCoreConfig,
    FactorizedSharedCore,
    HistoryContext,
    StageFeatures,
)

from .manifest import CONDITION_FIELDS, HISTORY
from .native_charts import fno_native64_exact_left_inverse


@dataclass(frozen=True)
class ChartDefinition:
    key: str
    channels: int
    boundary_padding: str


CHARTS: Mapping[str, ChartDefinition] = {
    "fno_vorticity": ChartDefinition("fno_vorticity", 1, "circular"),
    "cns4": ChartDefinition("cns4", 4, "circular"),
    "swe_height": ChartDefinition("swe_height", 1, "replicate"),
    "reaction_diffusion": ChartDefinition("reaction_diffusion", 2, "replicate"),
    "cfdb_cavity": ChartDefinition("cfdb_cavity", 3, "replicate"),
    "cfdb_cylinder": ChartDefinition("cfdb_cylinder", 3, "replicate"),
    "cfdb_tube": ChartDefinition("cfdb_tube", 3, "replicate"),
}


@dataclass(frozen=True)
class SharedCarrierConfig:
    model_size: int = 128
    latent_size: int = 16
    width: int = 512
    num_blocks: int = 4
    ffn_hidden: int = 1920
    # Only the first analysis block is shared before physical-program
    # branching.  Four parameter-matched spectral basis blocks are then
    # interleaved with typed operator adapters rather than executed as one
    # deep task-agnostic trunk.
    history_depth: int = 1
    current_depth: int = 1
    operator_basis_depth: int = 4
    history_channels: int = 32
    state_channels: int = 16
    frame_width: int = 16
    condition_hidden: int = 128
    detail_width: int = 32

    def core(self) -> FactorizedCoreConfig:
        return FactorizedCoreConfig(
            history_channels=self.history_channels,
            state_channels=self.state_channels,
            native_size=self.model_size,
            latent_size=self.latent_size,
            width=self.width,
            num_blocks=self.num_blocks,
            ffn_hidden=self.ffn_hidden,
            history_depth=self.history_depth,
            current_depth=self.current_depth,
            operator_basis_depth=self.operator_basis_depth,
            mechanism_depth=3,
            mechanism_hidden=512 if self.width >= 512 else self.width,
            adapter_depth=1,
            adapter_hidden=192 if self.width >= 192 else self.width,
            detail_width=self.detail_width,
            boundary_padding="replicate",
        )


def canonical_history(value: Tensor, channels: int) -> Tensor:
    """Return real observations as `[B,H10,C,H,W]` without truncation."""

    if value.ndim != 5:
        raise ValueError("history must be rank five")
    candidates = []
    if value.shape[1] == HISTORY and value.shape[2] == channels:
        candidates.append(value)
    if value.shape[1] == HISTORY and value.shape[-1] == channels:
        candidates.append(value.permute(0, 1, 4, 2, 3))
    if value.shape[-2] == HISTORY and value.shape[-1] == channels:
        candidates.append(value.permute(0, 3, 4, 1, 2))
    if value.shape[2] == HISTORY and value.shape[1] == channels:
        candidates.append(value.permute(0, 2, 1, 3, 4))
    if len(candidates) != 1:
        raise ValueError(
            f"cannot uniquely canonicalize {tuple(value.shape)} as H10/C{channels}"
        )
    result = candidates[0].contiguous()
    if tuple(result.shape[1:3]) != (HISTORY, channels):
        raise AssertionError("history canonicalization failed")
    if result.dtype != torch.float32:
        result = result.float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("history contains non-finite values")
    return result


def _coordinates(reference: Tensor) -> Tensor:
    batch, _channels, height, width = reference.shape
    y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


class FullHistoryChartAdapter(nn.Module):
    """Chart-private frontend whose output has a common, task-free shape.

    Every frame is projected before a depthwise temporal convolution and the
    ten resulting features are concatenated.  Consequently no prefix frame can
    disappear through packed-channel clipping.
    """

    def __init__(self, definition: ChartDefinition, config: SharedCarrierConfig) -> None:
        super().__init__()
        self.definition = definition
        self.config = config
        padding = definition.boundary_padding
        self.frame_projection = nn.Conv2d(definition.channels, config.frame_width, 1)
        self.temporal_position = nn.Parameter(
            0.02 * torch.randn(1, config.frame_width, HISTORY, 1, 1)
        )
        self.temporal_mixer = nn.Conv3d(
            config.frame_width,
            config.frame_width,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=config.frame_width,
            bias=False,
        )
        self.history_projection = nn.Sequential(
            nn.Conv2d(
                HISTORY * config.frame_width + 2,
                config.history_channels,
                3,
                padding=1,
                padding_mode=padding,
            ),
            nn.GELU(),
            nn.Conv2d(config.history_channels, config.history_channels, 1),
        )
        self.state_projection = nn.Sequential(
            nn.Conv2d(
                definition.channels + 2,
                config.state_channels,
                3,
                padding=1,
                padding_mode=padding,
            ),
            nn.GELU(),
            nn.Conv2d(config.state_channels, config.state_channels, 1),
        )

    def _resize(self, value: Tensor) -> Tensor:
        if value.shape[-2:] == (self.config.model_size, self.config.model_size):
            return value
        return F.interpolate(
            value,
            size=(self.config.model_size, self.config.model_size),
            mode="bilinear",
            align_corners=False,
        )

    def history(self, value: Tensor) -> Tensor:
        history = canonical_history(value, self.definition.channels)
        if self.definition.key == "fno_vorticity":
            # Preserve the already-qualified native64 chart before the common
            # carrier lift.  This is exact for unperturbed official inputs.
            history = fno_native64_exact_left_inverse(history)
        batch, frames, channels, height, width = history.shape
        flat = history.reshape(batch * frames, channels, height, width)
        flat = self._resize(flat)
        frame_features = self.frame_projection(flat).reshape(
            batch,
            frames,
            self.config.frame_width,
            self.config.model_size,
            self.config.model_size,
        ).permute(0, 2, 1, 3, 4)
        mixed = self.temporal_mixer(frame_features + self.temporal_position)
        packed = mixed.reshape(
            batch,
            self.config.frame_width * HISTORY,
            self.config.model_size,
            self.config.model_size,
        )
        coordinates = _coordinates(packed)
        return self.history_projection(torch.cat((packed, coordinates), dim=1))

    def state(self, current: Tensor) -> Tensor:
        if current.ndim != 4 or current.shape[1] != self.definition.channels:
            raise ValueError(
                f"{self.definition.key} current state must be [B,{self.definition.channels},H,W]"
            )
        current = self._resize(current.float())
        return self.state_projection(torch.cat((current, _coordinates(current)), dim=1))


class ContinuousConditioner(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        inputs = 2 * len(CONDITION_FIELDS)
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * width),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def features(value: Tensor, mask: Tensor) -> Tensor:
        if value.ndim != 2 or value.shape[1] != len(CONDITION_FIELDS):
            raise ValueError("condition values must be [B,8]")
        if mask.shape != value.shape:
            raise ValueError("condition mask and values must have equal shape")
        if not bool(torch.isfinite(value).all()) or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("invalid condition values or mask")
        transformed = torch.sign(value) * torch.log1p(value.abs())
        return torch.cat((transformed * mask, mask), dim=1)

    def forward(self, latent: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        affine = self.net(self.features(value.to(latent.dtype), mask.to(latent.dtype)))
        gain, shift = affine.chunk(2, dim=1)
        gain = 0.25 * torch.tanh(gain)[:, :, None, None]
        shift = 0.25 * torch.tanh(shift)[:, :, None, None]
        return latent * (1.0 + gain) + shift


class UniversalSharedCarrier(nn.Module):
    """Shallow common chart followed by an interleavable operator basis."""

    def __init__(
        self,
        config: SharedCarrierConfig = SharedCarrierConfig(),
        charts: Mapping[str, ChartDefinition] = CHARTS,
    ) -> None:
        super().__init__()
        self.config = config
        self.core_config = config.core()
        self.chart_adapters = nn.ModuleDict(
            {key: FullHistoryChartAdapter(value, config) for key, value in charts.items()}
        )
        self.shared_core = FactorizedSharedCore(self.core_config)
        self.conditioner = ContinuousConditioner(config.width, config.condition_hidden)
        for name, parameter in self.named_parameters():
            owner = "shared::shallow_representation_basis"
            if name.startswith("chart_adapters."):
                owner = "chart::" + name.split(".")[1]
            parameter._base6_owner = owner  # type: ignore[attr-defined]

    def encode_history(
        self,
        chart_key: str,
        history: Tensor,
        condition: Tensor,
        condition_mask: Tensor,
    ) -> HistoryContext:
        if chart_key not in self.chart_adapters:
            raise ValueError(f"unknown chart {chart_key!r}")
        packed = self.chart_adapters[chart_key].history(history)
        context = self.shared_core.encode_context(packed)
        conditioned = self.conditioner(context.latent, condition, condition_mask)
        return HistoryContext(conditioned, context.state_scale, context.native_size)

    def encode_stage(
        self,
        chart_key: str,
        current: Tensor,
        context: HistoryContext,
    ) -> StageFeatures:
        if chart_key not in self.chart_adapters:
            raise ValueError(f"unknown chart {chart_key!r}")
        packed = self.chart_adapters[chart_key].state(current)
        return self.shared_core.encode_stage(packed, context)

    def expert_config(self, *, boundary_padding: str) -> FactorizedCoreConfig:
        return replace(self.core_config, boundary_padding=boundary_padding)


__all__ = [
    "CHARTS",
    "ChartDefinition",
    "ContinuousConditioner",
    "FullHistoryChartAdapter",
    "SharedCarrierConfig",
    "UniversalSharedCarrier",
    "canonical_history",
]
