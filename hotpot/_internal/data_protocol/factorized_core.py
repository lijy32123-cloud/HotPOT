from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .production_primitives import DepthwiseResidualBlock, SpectralResidualBlock


@dataclass(frozen=True)
class FactorizedCoreConfig:
    history_channels: int
    state_channels: int
    native_size: int = 64
    latent_size: int = 16
    width: int = 512
    num_blocks: int = 4
    ffn_hidden: int = 1920
    history_depth: int = 4
    current_depth: int = 2
    operator_basis_depth: int = 0
    mechanism_depth: int = 3
    mechanism_hidden: int = 512
    adapter_depth: int = 1
    adapter_hidden: int = 192
    detail_width: int = 32
    boundary_padding: str = "circular"

    def validate(self) -> None:
        values = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("boundary_padding", "operator_basis_depth")
        }
        bad = {name: value for name, value in values.items() if value < 1}
        if bad:
            raise ValueError(f"core entries must be positive: {bad}")
        if self.operator_basis_depth < 0:
            raise ValueError("operator_basis_depth must be nonnegative")
        if self.native_size % self.latent_size:
            raise ValueError("native_size must be divisible by latent_size")
        if self.width % self.num_blocks:
            raise ValueError("width must be divisible by num_blocks")
        if self.boundary_padding not in ("circular", "replicate"):
            raise ValueError("unsupported boundary padding")

    @property
    def patch_size(self) -> int:
        return self.native_size // self.latent_size


@dataclass(frozen=True)
class HistoryContext:
    latent: Tensor
    state_scale: Tensor
    native_size: int


@dataclass(frozen=True)
class StageFeatures:
    latent: Tensor
    carrier: Tensor


class _SpectralAnalysisTower(nn.Module):
    def __init__(
        self,
        in_channels: int,
        config: FactorizedCoreConfig,
        *,
        depth: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(
            in_channels,
            config.width,
            config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            SpectralResidualBlock(config.width, config.num_blocks, config.ffn_hidden)
            for _ in range(depth)
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, config.width)

    def forward(self, value: Tensor) -> Tensor:
        value = self.input_projection(value)
        for block in self.blocks:
            value = block(value)
        return self.final_norm(value)


class StageInvariantHistoryEncoder(nn.Module):
    def __init__(self, config: FactorizedCoreConfig) -> None:
        super().__init__()
        self.config = config
        self.analysis = _SpectralAnalysisTower(
            config.history_channels, config, depth=config.history_depth
        )

    def forward(self, history: Tensor) -> HistoryContext:
        expected = (
            history.shape[0],
            self.config.history_channels,
            self.config.native_size,
            self.config.native_size,
        )
        if tuple(history.shape) != expected:
            raise ValueError(f"packed history must be {expected}, got {tuple(history.shape)}")
        if history.dtype != torch.float32:
            raise TypeError("shared production core is FP32-only")
        scale = history.square().mean((1, 2, 3)).sqrt().clamp_min(1.0e-6)
        latent = self.analysis(history / scale[:, None, None, None])
        return HistoryContext(latent, scale, self.config.native_size)


class StatelessCurrentStateEncoder(nn.Module):
    def __init__(self, config: FactorizedCoreConfig) -> None:
        super().__init__()
        self.config = config
        self.analysis = _SpectralAnalysisTower(
            config.state_channels, config, depth=config.current_depth
        )

    def forward(self, current: Tensor, context: HistoryContext) -> StageFeatures:
        expected = (
            current.shape[0],
            self.config.state_channels,
            self.config.native_size,
            self.config.native_size,
        )
        if tuple(current.shape) != expected:
            raise ValueError(f"current state must be {expected}, got {tuple(current.shape)}")
        if current.dtype != torch.float32:
            raise TypeError("shared production core is FP32-only")
        if current.shape[0] != context.latent.shape[0]:
            raise ValueError("context/current batch mismatch")
        normalized = current / context.state_scale[:, None, None, None]
        # Boundary-sensitive local processing belongs to the typed operator,
        # not to the globally shared stem.  Preserve the normalized packed
        # state here and let each operator apply its declared padding rule.
        return StageFeatures(self.analysis(normalized), normalized)


class FactorizedSharedCore(nn.Module):
    def __init__(self, config: FactorizedCoreConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.history_context_encoder = StageInvariantHistoryEncoder(config)
        self.current_state_encoder = StatelessCurrentStateEncoder(config)
        # These blocks are a shared numerical representation basis, not a
        # monolithic task-agnostic evolution trunk.  Typed adapters are
        # interleaved between them by ProgramTypedOperatorExpert.
        self.operator_basis = nn.ModuleList(
            SpectralResidualBlock(config.width, config.num_blocks, config.ffn_hidden)
            for _ in range(config.operator_basis_depth)
        )

    def encode_context(self, history: Tensor) -> HistoryContext:
        return self.history_context_encoder(history)

    def encode_stage(self, current: Tensor, context: HistoryContext) -> StageFeatures:
        return self.current_state_encoder(current, context)


class TypedOperatorExpert(nn.Module):
    def __init__(
        self,
        config: FactorizedCoreConfig,
        *,
        signature: str,
        out_channels: int,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if not signature or out_channels < 1:
            raise ValueError("signature and positive out_channels are required")
        self.signature = signature
        self.out_channels = int(out_channels)
        self.mechanism_dictionary = nn.ModuleList(
            DepthwiseResidualBlock(
                config.width,
                config.mechanism_hidden,
                padding_mode=config.boundary_padding,
            )
            for _ in range(config.mechanism_depth)
        )
        self.operator_adapter = nn.ModuleList(
            DepthwiseResidualBlock(
                config.width,
                config.adapter_hidden,
                padding_mode=config.boundary_padding,
            )
            for _ in range(config.adapter_depth)
        )
        self.coarse_head = nn.Conv2d(config.width, self.out_channels, 1)
        self.detail_carrier = nn.Sequential(
            nn.Conv2d(
                config.state_channels,
                config.detail_width,
                3,
                padding=1,
                padding_mode=config.boundary_padding,
            ),
            nn.GELU(),
            nn.Conv2d(
                config.detail_width,
                config.detail_width,
                3,
                padding=1,
                padding_mode=config.boundary_padding,
            ),
        )
        self.detail_fusion = nn.Sequential(
            nn.Conv2d(config.width + config.detail_width, config.detail_width, 1),
            nn.GELU(),
            nn.Conv2d(
                config.detail_width,
                config.detail_width,
                3,
                padding=1,
                padding_mode=config.boundary_padding,
            ),
            nn.GELU(),
        )
        self.detail_head = nn.Conv2d(config.detail_width, self.out_channels, 1)
        if zero_init:
            nn.init.zeros_(self.coarse_head.weight)
            nn.init.zeros_(self.coarse_head.bias)
            nn.init.zeros_(self.detail_head.weight)
            nn.init.zeros_(self.detail_head.bias)

    def forward(self, context: HistoryContext, stage: StageFeatures) -> Tensor:
        value = context.latent + stage.latent
        for block in self.mechanism_dictionary:
            value = block(value)
        for block in self.operator_adapter:
            value = block(value)
        upsampled = F.interpolate(
            value,
            size=(context.native_size, context.native_size),
            mode="bilinear",
            align_corners=False,
        )
        local_detail = self.detail_carrier(stage.carrier)
        detail = self.detail_fusion(torch.cat((upsampled, local_detail), dim=1))
        return self.coarse_head(upsampled) + self.detail_head(detail)


__all__ = [
    "FactorizedCoreConfig",
    "FactorizedSharedCore",
    "HistoryContext",
    "StageFeatures",
    "TypedOperatorExpert",
]
