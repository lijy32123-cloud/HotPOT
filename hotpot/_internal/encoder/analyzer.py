from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from hotpot._vendor.afno import AFNO2D

from .config import FoundationConfig


def _official_afno(width: int, *, num_blocks: int, modes: int) -> nn.Module:
    return AFNO2D(
        width=width,
        num_blocks=num_blocks,
        channel_first=True,
        modes=modes,
        hidden_size_factor=1,
        act="gelu",
    )


class RelativeTemporalAggregator(nn.Module):
    def __init__(self, steps: int, width: int) -> None:
        super().__init__()
        self.steps = int(steps)
        self.width = int(width)
        scale = 1.0 / (self.steps * math.sqrt(self.width))
        self.weight = nn.Parameter(scale * torch.randn(self.steps, self.width, self.width))
        frequencies = 2.0 ** torch.linspace(-6.0, 6.0, self.width)
        self.log_frequency = nn.Parameter(frequencies.log())

    def forward(self, frames: Tensor, relative_lags: Tensor) -> Tensor:
        if frames.ndim != 5:
            raise ValueError("temporal tokens must be [B,T,C,H,W]")
        batch, steps, width, _height, _grid_width = frames.shape
        if (steps, width) != (self.steps, self.width):
            raise ValueError("temporal token ABI mismatch")
        if relative_lags.shape != (batch, steps):
            raise ValueError("relative lags must be [B,T]")
        frequency = self.log_frequency.exp().to(frames.dtype)
        phase = torch.cos(relative_lags.to(frames.dtype)[..., None] * frequency)
        return torch.einsum(
            "tij,btihw->bjhw", self.weight, frames * phase[..., None, None]
        )


class AnalysisBlock(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        num_blocks: int,
        modes: int,
        ffn_hidden: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.spectral = _official_afno(width, num_blocks=num_blocks, modes=modes)
        self.local = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.norm2 = nn.GroupNorm(8, width)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, ffn_hidden, 1),
            nn.GELU(),
            nn.Conv2d(ffn_hidden, width, 1),
        )

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        normalized = self.norm1(value)
        mixed = residual + self.spectral(normalized) + self.local(normalized)
        return mixed + self.ffn(self.norm2(mixed))


class SharedScalarFieldAnalyzer(nn.Module):
    """Shared weights; no mixing between fields with different meanings."""

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.patch_tokenizer = nn.Sequential(
            nn.Conv2d(
                config.semantic_channels,
                2 * config.shared_width,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            ),
            nn.GELU(),
            nn.Conv2d(2 * config.shared_width, config.shared_width, 1),
        )
        self.positional = nn.Parameter(
            torch.zeros(1, config.shared_width, config.latent_size, config.latent_size)
        )
        nn.init.trunc_normal_(self.positional, std=0.02)
        self.temporal = RelativeTemporalAggregator(config.history_steps, config.shared_width)
        self.blocks = nn.ModuleList(
            AnalysisBlock(
                config.shared_width,
                num_blocks=config.afno_blocks,
                modes=config.afno_modes,
                ffn_hidden=config.shared_ffn_hidden,
            )
            for _ in range(config.shared_depth)
        )
        self.final_norm = nn.GroupNorm(8, config.shared_width)
        self.call_count = 0

    def reset_audit(self) -> None:
        self.call_count = 0

    def forward(
        self,
        features: Tensor,
        field_mask: Tensor,
        relative_lags: Tensor,
    ) -> Tensor:
        self.call_count += 1
        if features.ndim != 6:
            raise ValueError("semantic features must be [B,F,T,K,H,W]")
        batch, fields, steps, channels, height, width = features.shape
        expected = (
            self.config.max_fields,
            self.config.history_steps,
            self.config.semantic_channels,
            self.config.resolution,
            self.config.resolution,
        )
        if (fields, steps, channels, height, width) != expected:
            raise ValueError(f"semantic feature ABI requires [B,{expected}]")
        if field_mask.shape != (batch, fields):
            raise ValueError("field mask must be [B,F]")
        if relative_lags.shape != (batch, steps):
            raise ValueError("relative lags must be [B,T]")
        packed = features.reshape(
            batch * fields * steps, channels, height, width
        )
        tokens = self.patch_tokenizer(packed) + self.positional
        tokens = tokens.reshape(
            batch * fields,
            steps,
            self.config.shared_width,
            self.config.latent_size,
            self.config.latent_size,
        )
        lags = relative_lags[:, None].expand(-1, fields, -1).reshape(batch * fields, steps)
        value = self.temporal(tokens, lags)
        for block in self.blocks:
            value = block(value)
        value = self.final_norm(value).reshape(
            batch,
            fields,
            self.config.shared_width,
            self.config.latent_size,
            self.config.latent_size,
        )
        return value * field_mask[:, :, None, None, None]


__all__ = [
    "AnalysisBlock",
    "RelativeTemporalAggregator",
    "SharedScalarFieldAnalyzer",
]
