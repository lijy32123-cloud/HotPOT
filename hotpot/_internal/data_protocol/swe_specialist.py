from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .factorized_core import (
    FactorizedCoreConfig,
    FactorizedSharedCore,
    HistoryContext,
    TypedOperatorExpert,
)


@dataclass(frozen=True)
class PhysicsContext:
    gravity: float
    dt: float
    dx: float
    dy: float

    def validate(self) -> None:
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.gravity, self.dt, self.dx, self.dy)
        ):
            raise ValueError("gravity, dt, dx and dy must be finite and positive")


@dataclass(frozen=True)
class PotentialFlowConfig:
    history_frames: int = 10
    native_size: int = 128
    minimum_height: float = 1.0e-6
    conditioner_hidden: int = 128
    detail_width: int = 128
    detail_depth: int = 13

    def core_config(self) -> FactorizedCoreConfig:
        return FactorizedCoreConfig(
            history_channels=self.history_frames + 3,
            state_channels=2 + 3,
            native_size=self.native_size,
            latent_size=16,
            width=512,
            num_blocks=4,
            ffn_hidden=1920,
            history_depth=4,
            current_depth=2,
            mechanism_depth=1,
            mechanism_hidden=512,
            adapter_depth=1,
            adapter_hidden=192,
            detail_width=32,
            boundary_padding="replicate",
        )


def _coordinates(reference: Tensor) -> Tensor:
    batch, _channels, height, width = reference.shape
    y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    distance = torch.minimum(1.0 - xx.abs(), 1.0 - yy.abs())
    return torch.stack((xx, yy, distance), dim=0).unsqueeze(0).expand(
        batch, -1, -1, -1
    )


class FullResolutionBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8 if width % 8 == 0 else 1, width)
        self.conv = nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate")

    def forward(self, value: Tensor) -> Tensor:
        return value + 0.1 * torch.nn.functional.gelu(self.conv(self.norm(value)))


class FullResolutionScalarHead(nn.Module):
    def __init__(self, carrier_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(
            carrier_channels + 1, width, 3, padding=1, padding_mode="replicate"
        )
        self.blocks = nn.ModuleList(FullResolutionBlock(width) for _ in range(depth))
        self.final_norm = nn.GroupNorm(8 if width % 8 == 0 else 1, width)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1.0e-5)
        nn.init.zeros_(self.head.bias)

    def forward(self, carrier: Tensor, coarse: Tensor) -> Tensor:
        value = torch.nn.functional.gelu(
            self.stem(torch.cat((carrier, coarse), dim=1))
        )
        for block in self.blocks:
            value = block(value)
        return self.head(self.final_norm(value))


class SavedIntervalScalarCore(nn.Module):
    def __init__(self, signature: str, config: PotentialFlowConfig) -> None:
        super().__init__()
        self.config = config
        core = config.core_config()
        self.shared = FactorizedSharedCore(core)
        self.physics_conditioner = nn.Sequential(
            nn.Linear(5, config.conditioner_hidden),
            nn.GELU(),
            nn.Linear(config.conditioner_hidden, 2 * core.width),
        )
        self.typed_scalar = TypedOperatorExpert(
            core, signature=signature, out_channels=1, zero_init=False
        )
        nn.init.normal_(self.typed_scalar.coarse_head.weight, mean=0.0, std=1.0e-5)
        nn.init.zeros_(self.typed_scalar.coarse_head.bias)
        nn.init.normal_(self.typed_scalar.detail_head.weight, mean=0.0, std=1.0e-5)
        nn.init.zeros_(self.typed_scalar.detail_head.bias)
        nn.init.zeros_(self.physics_conditioner[-1].weight)
        nn.init.zeros_(self.physics_conditioner[-1].bias)
        self.full_resolution_scalar = FullResolutionScalarHead(
            core.detail_width, config.detail_width, config.detail_depth
        )
        self.last_audit: dict[str, int | str] | None = None

    def _physics_vector(self, history: Tensor, physics: PhysicsContext) -> Tensor:
        physics.validate()
        mean_height = history[:, -1:].mean((1, 2, 3)).clamp_min(1.0e-8)
        constants = history.new_tensor(
            [physics.gravity, physics.dt, physics.dx, physics.dy]
        ).log().unsqueeze(0).expand(history.shape[0], -1)
        return torch.cat((constants, mean_height.log().unsqueeze(1)), dim=1)

    def raw_scalar(self, history: Tensor, physics: PhysicsContext) -> Tensor:
        expected = (
            history.shape[0],
            self.config.history_frames,
            self.config.native_size,
            self.config.native_size,
        )
        if history.dtype != torch.float32 or tuple(history.shape) != expected:
            raise ValueError(f"history must be FP32 {expected}")
        if not bool(torch.isfinite(history).all()):
            raise ValueError("history contains non-finite values")
        current = history[:, -1:].contiguous()
        previous = history[:, -2:-1].contiguous()
        centered_history = history - history.mean((-2, -1), keepdim=True)
        coordinates = _coordinates(current)
        context = self.shared.encode_context(
            torch.cat((centered_history, coordinates), dim=1)
        )
        stage = self.shared.encode_stage(
            torch.cat(
                (
                    current - current.mean((-2, -1), keepdim=True),
                    current - previous,
                    coordinates,
                ),
                dim=1,
            ),
            context,
        )
        film = self.physics_conditioner(self._physics_vector(history, physics))
        gain, shift = film.chunk(2, dim=1)
        combined = context.latent + stage.latent
        conditioned = combined * (
            1.0 + 0.1 * torch.tanh(gain)[:, :, None, None]
        )
        conditioned = conditioned + 0.1 * torch.tanh(shift)[:, :, None, None]
        conditioned_context = HistoryContext(
            latent=conditioned - stage.latent,
            state_scale=context.state_scale,
            native_size=context.native_size,
        )
        coarse = self.typed_scalar(conditioned_context, stage)
        # The full-resolution head is dimensioned for the SWE-private
        # detail representation, not the raw packed stage state.
        detail_carrier = self.typed_scalar.detail_carrier(stage.carrier)
        return coarse + self.full_resolution_scalar(detail_carrier, coarse)


def observable_step_scale(history: Tensor) -> Tensor:
    centered = history - history.mean((-2, -1), keepdim=True)
    differences = history[:, 1:] - history[:, :-1]
    height_scale = centered.square().mean((1, 2, 3), keepdim=True).sqrt()
    return differences.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(
        height_scale * 1.0e-4
    )


def global_positive_line_search(
    current: Tensor, proposal: Tensor, floor: float
) -> tuple[Tensor, Tensor]:
    increment = proposal - current
    negative = increment < 0
    safe = torch.where(negative, increment, -torch.ones_like(increment))
    ratios = torch.where(
        negative,
        (current - floor) / (-safe).clamp_min(1.0e-12),
        torch.ones_like(increment),
    )
    alpha = ratios.amin((-2, -1), keepdim=True).clamp(0.0, 1.0)
    return current + alpha * increment, alpha


class SWEDirectSavedStepControl(SavedIntervalScalarCore):
    """One direct H10-conditioned saved-frame action and positivity program."""

    def __init__(self, config: PotentialFlowConfig | None = None) -> None:
        config = config or PotentialFlowConfig()
        super().__init__("control.direct_saved_step_height_increment", config)

    def forward_with_field(
        self, history: Tensor, physics: PhysicsContext
    ) -> tuple[Tensor, Tensor, Tensor]:
        field = self.raw_scalar(history, physics)[:, 0]
        field = field - field.mean((-2, -1), keepdim=True)
        current = history[:, -1]
        proposal = current + observable_step_scale(history)[:, 0] * field
        prediction, alpha = global_positive_line_search(
            current, proposal, self.config.minimum_height
        )
        self.last_audit = {
            "network_calls_per_saved_step": 1,
            "learned_quantity": "direct_saved_step_height_increment_control",
            "second_order_recurrence": "absent",
        }
        return prediction, field, alpha

    def forward(self, history: Tensor, physics: PhysicsContext) -> Tensor:
        return self.forward_with_field(history, physics)[0]


__all__ = [
    "PhysicsContext",
    "PotentialFlowConfig",
    "SWEDirectSavedStepControl",
    "global_positive_line_search",
    "observable_step_scale",
]
