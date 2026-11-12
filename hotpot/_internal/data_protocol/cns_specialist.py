from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .factorized_core import FactorizedCoreConfig
from .periodic_spectral_analysis import (
    PeriodicSpectralHistoryEncoder,
    periodic_spectral_resize,
)
from .production_primitives import DepthwiseResidualBlock


@dataclass(frozen=True)
class PeriodicMarkovConfig:
    native_size: int = 128
    latent_size: int = 16
    width: int = 512
    num_blocks: int = 4
    ffn_hidden: int = 1920
    context_channels: int = 32
    history_depth: int = 4
    dictionary_depth: int = 7
    dictionary_hidden: int = 1024
    local_width: int = 64
    local_hidden: int = 128
    local_depth: int = 3
    fusion_hidden: int = 128
    gamma: float = 5.0 / 3.0
    frame_dt: float = 5.0e-2
    density_floor: float = 1.0e-7
    pressure_floor: float = 1.0e-7
    chart_floor: float = 1.0e-6

    def core(self) -> FactorizedCoreConfig:
        return FactorizedCoreConfig(
            history_channels=15,
            state_channels=4,
            native_size=self.native_size,
            latent_size=self.latent_size,
            width=self.width,
            num_blocks=self.num_blocks,
            ffn_hidden=self.ffn_hidden,
            history_depth=self.history_depth,
            current_depth=1,
            mechanism_depth=self.dictionary_depth,
            mechanism_hidden=self.dictionary_hidden,
            adapter_depth=1,
            adapter_hidden=192,
            detail_width=32,
            boundary_padding="circular",
        )


def primitive_to_conservative(primitive: Tensor, gamma: float) -> Tensor:
    if primitive.ndim != 4 or primitive.shape[-1] != 4:
        raise ValueError("primitive must be [B,H,W,4]")
    vx, vy, rho, pressure = primitive.unbind(dim=-1)
    energy = pressure / (gamma - 1.0) + 0.5 * rho * (
        vx.square() + vy.square()
    )
    return torch.stack((rho, rho * vx, rho * vy, energy), dim=1)


def conservative_to_primitive(
    value: Tensor, gamma: float, density_floor: float
) -> Tensor:
    if value.ndim != 4 or value.shape[1] != 4:
        raise ValueError("conservative must be [B,4,H,W]")
    rho, mx, my, energy = value.unbind(dim=1)
    safe_rho = rho.clamp_min(density_floor)
    vx, vy = mx / safe_rho, my / safe_rho
    pressure = (gamma - 1.0) * (
        energy - 0.5 * (mx.square() + my.square()) / safe_rho
    )
    return torch.stack((vx, vy, rho, pressure), dim=-1)


def _admissible(value: Tensor, config: PeriodicMarkovConfig) -> Tensor:
    primitive = conservative_to_primitive(
        value, config.gamma, config.density_floor
    )
    return (
        torch.isfinite(value).all(dim=(1, 2, 3))
        & (primitive[..., 2] > config.density_floor).all(dim=(1, 2))
        & (primitive[..., 3] > config.pressure_floor).all(dim=(1, 2))
    )


def positivity_segment(
    base: Tensor, candidate: Tensor, config: PeriodicMarkovConfig
) -> tuple[Tensor, Tensor]:
    full = _admissible(candidate, config)
    if bool(full.all().item()):
        return candidate, torch.ones(
            base.shape[0], device=base.device, dtype=base.dtype
        )
    delta = candidate - base
    lower = torch.zeros(base.shape[0], device=base.device, dtype=base.dtype)
    upper = torch.ones_like(lower)
    for _ in range(24):
        middle = 0.5 * (lower + upper)
        good = _admissible(
            base + middle[:, None, None, None] * delta, config
        )
        lower = torch.where(good, middle, lower)
        upper = torch.where(good, upper, middle)
    theta = torch.where(full, torch.ones_like(lower), lower)
    return base + theta[:, None, None, None] * delta, theta


def _current_chart(
    current: Tensor, config: PeriodicMarkovConfig
) -> tuple[Tensor, Tensor]:
    value = current.permute(0, 3, 1, 2)
    location = value.mean(dim=(-2, -1), keepdim=True)
    scale = value.std(dim=(-2, -1), keepdim=True).clamp_min(
        config.chart_floor
    )
    normalized = (value - location) / scale
    height, width = current.shape[1:3]
    stats = torch.cat((location / scale, torch.log(scale)), dim=1).expand(
        -1, -1, height, width
    )
    constants = value.new_tensor(
        (
            config.gamma,
            math.log2(1.0 / height),
            config.frame_dt * height,
        )
    )[None, :, None, None].expand(value.shape[0], -1, height, width)
    chart = torch.cat((normalized, stats, constants), dim=1)
    conservative = primitive_to_conservative(current, config.gamma)
    centered = conservative - conservative.mean(
        dim=(-2, -1), keepdim=True
    )
    state_scale = centered.square().mean(
        dim=(-2, -1), keepdim=True
    ).sqrt().clamp_min(config.chart_floor)
    return chart, state_scale


class PeriodicMarkovEncoder(nn.Module):
    def __init__(self, config: PeriodicMarkovConfig, seed: int) -> None:
        super().__init__()
        core = config.core()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.analysis = PeriodicSpectralHistoryEncoder(core)
            self.dictionary = nn.ModuleList(
                DepthwiseResidualBlock(
                    core.width,
                    config.dictionary_hidden,
                    padding_mode="circular",
                )
                for _ in range(config.dictionary_depth)
            )
            self.head = nn.Conv2d(
                core.width, config.context_channels, 1
            )
            for block in self.dictionary:
                nn.init.constant_(block.scale, 1.0e-3)

    def forward(self, chart: Tensor) -> Tensor:
        encoded = self.analysis(chart).latent
        for block in self.dictionary:
            encoded = block(encoded)
        return periodic_spectral_resize(
            self.head(encoded), (chart.shape[-2], chart.shape[-1])
        )


class PeriodicMarkovHead(nn.Module):
    def __init__(
        self, context_channels: int, config: PeriodicMarkovConfig, seed: int
    ) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 41)
            self.local_lift = nn.Sequential(
                nn.Conv2d(
                    15,
                    config.local_width,
                    3,
                    padding=1,
                    padding_mode="circular",
                ),
                nn.GELU(),
            )
            self.local_blocks = nn.ModuleList(
                DepthwiseResidualBlock(
                    config.local_width,
                    config.local_hidden,
                    padding_mode="circular",
                )
                for _ in range(config.local_depth)
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(
                    context_channels + config.local_width,
                    config.fusion_hidden,
                    1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    config.fusion_hidden, config.fusion_hidden, 1
                ),
                nn.GELU(),
                nn.Conv2d(config.fusion_hidden, 4, 1),
            )
            nn.init.normal_(self.fusion[-1].weight, std=1.0e-4)
            nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, context: Tensor, chart: Tensor) -> Tensor:
        local = self.local_lift(chart)
        for block in self.local_blocks:
            local = block(local)
        return self.fusion(torch.cat((context, local), dim=1))


class CNSPeriodicAliasSafeMarkovAction(nn.Module):
    def __init__(
        self,
        config: PeriodicMarkovConfig = PeriodicMarkovConfig(),
        seed: int = 20260811,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "alias_safe_representation_version",
            torch.tensor(1, dtype=torch.int64),
            persistent=True,
        )
        self.context_encoder = PeriodicMarkovEncoder(config, seed)
        self.action_head = PeriodicMarkovHead(
            config.context_channels, config, seed
        )
        for parameter in self.parameters():
            parameter._base6_owner = "program::cns_compressible"  # type: ignore[attr-defined]

    def forward(self, history_native: Tensor, *, return_audit: bool = False):
        if (
            history_native.ndim != 5
            or history_native.shape[1] != 10
            or history_native.shape[-1] != 4
        ):
            raise ValueError("history_native must be [B,10,H,W,4]")
        current_public = history_native[:, -1]
        chart, state_scale = _current_chart(current_public, self.config)
        context = self.context_encoder(chart)
        normalized_action = self.action_head(context, chart)
        normalized_action = normalized_action - normalized_action.mean(
            dim=(-2, -1), keepdim=True
        )
        current = primitive_to_conservative(
            current_public, self.config.gamma
        )
        candidate = current + state_scale * normalized_action
        candidate = candidate + current.mean(
            dim=(-2, -1), keepdim=True
        ) - candidate.mean(dim=(-2, -1), keepdim=True)
        next_state, theta = positivity_segment(
            current.double(), candidate.double(), self.config
        )
        prediction = conservative_to_primitive(
            next_state, self.config.gamma, self.config.density_floor
        ).to(history_native.dtype)
        if not return_audit:
            return prediction
        drift = (next_state - current.double()).mean(
            dim=(-2, -1)
        ).abs().amax()
        return prediction, {
            "program": "cns_compressible",
            "learned_quantity": "periodic_alias_safe_conservative_saved_step_action",
            "network_stage_evaluations": 1,
            "typed_head_evaluations": 1,
            "native_size": self.config.native_size,
            "all_finite": bool(torch.isfinite(next_state).all().detach()),
            "theta_min": float(theta.amin().detach().cpu()),
            "conservation_mean_drift_max": float(drift.detach().cpu()),
            "history_prefix_invariant": True,
        }


__all__ = ["CNSPeriodicAliasSafeMarkovAction", "PeriodicMarkovConfig"]
