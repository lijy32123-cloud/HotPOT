from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hotpot._internal.data_protocol.manifest import CONDITION_FIELDS
from hotpot._internal.foundation.config import tiny_17m_config
from hotpot._internal.encoder.action_decoder import ActionContextDecoder
from hotpot._internal.encoder.analyzer import SharedScalarFieldAnalyzer
from hotpot._internal.encoder.private_mixer import PrivateTypedFieldMixer


SEMANTIC_CHANNELS = 11


class GeneralSemanticFieldChart(nn.Module):
    """Apply the frozen V1.3 scalar-field chart to 3- or 5-field histories."""

    reads_full_history = True

    def __init__(self, config, *, dynamic_channels: int, boundary: str) -> None:
        super().__init__()
        if dynamic_channels > config.max_fields:
            raise ValueError("dynamic channels exceed scalar-field ABI")
        if boundary not in {"periodic", "closed"}:
            raise ValueError(boundary)
        self.config = config
        self.dynamic_channels = int(dynamic_channels)
        self.boundary = boundary

    def _differences(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        flat = value.flatten(0, 2)[:, None]
        if self.boundary == "periodic":
            left, right = torch.roll(flat, 1, -1), torch.roll(flat, -1, -1)
            down, up = torch.roll(flat, 1, -2), torch.roll(flat, -1, -2)
        else:
            padded = F.pad(flat, (1, 1, 1, 1), mode="replicate")
            left, right = padded[..., 1:-1, :-2], padded[..., 1:-1, 2:]
            down, up = padded[..., :-2, 1:-1], padded[..., 2:, 1:-1]
        shape = value.shape
        return (
            (0.5 * (right - left))[:, 0].reshape(shape),
            (0.5 * (up - down))[:, 0].reshape(shape),
            (left + right + down + up - 4.0 * flat)[:, 0].reshape(shape),
        )

    def forward(self, history: Tensor):
        expected = (128, 128, self.config.history_steps, self.dynamic_channels)
        if history.ndim != 5 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"history must be [B,{expected}]")
        batch = history.shape[0]
        dynamic = history.permute(0, 4, 3, 1, 2).contiguous()
        mean = dynamic.mean((2, 3, 4), keepdim=True)
        scale = dynamic.std((2, 3, 4), keepdim=True).clamp_min(1.0e-6)
        normalized_active = (dynamic - mean) / scale
        normalized = history.new_zeros(
            batch, self.config.max_fields, self.config.history_steps, 128, 128
        )
        normalized[:, : self.dynamic_channels] = normalized_active
        field_mask = history.new_zeros(batch, self.config.max_fields)
        field_mask[:, : self.dynamic_channels] = 1.0
        first = torch.zeros_like(normalized)
        first[:, :, 1:] = normalized[:, :, 1:] - normalized[:, :, :-1]
        first[:, :, 0] = first[:, :, 1]
        second = torch.zeros_like(normalized)
        second[:, :, 1:] = first[:, :, 1:] - first[:, :, :-1]
        second[:, :, 0] = second[:, :, 1]
        grad_x, grad_y, laplacian = self._differences(normalized)
        axis = torch.linspace(-1.0, 1.0, 128, device=history.device, dtype=history.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        distance = torch.ones_like(xx) if self.boundary == "periodic" else torch.minimum(
            1.0 - xx.abs(), 1.0 - yy.abs()
        ).clamp_min(0.0)
        shape = (1, 1, 1, 128, 128)
        expanded = normalized.shape
        xx = xx.reshape(shape).expand(expanded)
        yy = yy.reshape(shape).expand(expanded)
        distance = distance.reshape(shape).expand(expanded)
        geometry = torch.ones_like(normalized)
        validity = field_mask[:, :, None, None, None].expand_as(normalized)
        features = torch.stack(
            (
                normalized,
                first,
                second,
                grad_x,
                grad_y,
                laplacian,
                xx,
                yy,
                distance,
                geometry,
                validity,
            ),
            dim=3,
        ) * validity[:, :, :, None]
        field_mean = history.new_zeros(batch, self.config.max_fields)
        field_log_scale = history.new_zeros(batch, self.config.max_fields)
        field_mean[:, : self.dynamic_channels] = mean[:, :, 0, 0, 0]
        field_log_scale[:, : self.dynamic_channels] = scale[:, :, 0, 0, 0].log()
        relative_lags = torch.linspace(
            -1.0, 0.0, self.config.history_steps, device=history.device, dtype=history.dtype
        )[None].expand(batch, -1)
        return features, field_mask, field_mean, field_log_scale, relative_lags


def _coordinates(reference: Tensor) -> Tensor:
    batch, _channels, height, width = reference.shape
    y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), 0)[None].expand(batch, -1, -1, -1)


class TypedSavedStepHead(nn.Module):
    reads_full_history = False

    def __init__(self, config, channels: int, *, boundary: str) -> None:
        super().__init__()
        padding_mode = "circular" if boundary == "periodic" else "replicate"
        self.condition = nn.Sequential(
            nn.Linear(2 * len(CONDITION_FIELDS), config.condition_feature_hidden),
            nn.GELU(),
            nn.Linear(config.condition_feature_hidden, config.condition_feature_channels),
        )
        inputs = config.decoded_channels + channels + config.condition_feature_channels + 2
        self.net = nn.Sequential(
            nn.Conv2d(inputs, config.typed_head_hidden, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
            nn.Conv2d(config.typed_head_hidden, config.typed_head_hidden, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
            nn.Conv2d(config.typed_head_hidden, channels, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=1.0e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, decoded: Tensor, current: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        payload = torch.cat((values * mask, mask), 1).to(decoded.dtype)
        condition = self.condition(payload)[:, :, None, None].expand(
            -1, -1, current.shape[-2], current.shape[-1]
        )
        return self.net(torch.cat((decoded, current, condition, _coordinates(current)), 1))


def _closed_box_projection(velocity: Tensor) -> Tensor:
    u, v = velocity[:, 0], velocity[:, 1]
    pu = F.pad(u, (1, 1, 0, 0), mode="replicate")
    pv = F.pad(v, (0, 0, 1, 1), mode="replicate")
    divergence = 0.5 * (pu[:, :, 2:] - pu[:, :, :-2]) + 0.5 * (
        pv[:, 2:, :] - pv[:, :-2, :]
    )
    divergence = divergence - divergence.mean((1, 2), keepdim=True)
    height, width = divergence.shape[-2:]
    extended = torch.cat((divergence, divergence.flip(-1)), -1)
    extended = torch.cat((extended, extended.flip(-2)), -2)
    fy = torch.fft.fftfreq(2 * height, device=velocity.device, dtype=velocity.dtype)
    fx = torch.fft.fftfreq(2 * width, device=velocity.device, dtype=velocity.dtype)
    eigen = -4.0 * (
        torch.sin(torch.pi * fy)[:, None].square() + torch.sin(torch.pi * fx)[None].square()
    )
    spectrum = torch.fft.fft2(extended)
    safe = torch.where(eigen == 0, torch.ones_like(eigen), eigen)
    potential_spectrum = spectrum / safe[None]
    potential_spectrum[:, 0, 0] = 0
    potential = torch.fft.ifft2(potential_spectrum).real[:, :height, :width]
    px = F.pad(potential, (1, 1, 0, 0), mode="replicate")
    py = F.pad(potential, (0, 0, 1, 1), mode="replicate")
    projected = torch.stack(
        (u - 0.5 * (px[:, :, 2:] - px[:, :, :-2]), v - 0.5 * (py[:, 2:, :] - py[:, :-2, :])),
        1,
    )
    projected = projected.clone()
    projected[:, :, 0, :] = 0
    projected[:, :, -1, :] = 0
    projected[:, :, :, 0] = 0
    projected[:, :, :, -1] = 0
    return projected


class TypedSavedStepProgram(nn.Module):
    """Learn an identifiable finite saved-step action, then enforce admissibility."""

    reads_full_history = False

    def __init__(self, config, channels: int, program: str) -> None:
        super().__init__()
        self.channels = int(channels)
        self.program = program
        boundary = "closed" if program == "smoke_velocity_saved_step" else "periodic"
        self.head = TypedSavedStepHead(config, channels, boundary=boundary)

    def forward(self, history: Tensor, decoded: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        active = history.permute(0, 4, 3, 1, 2).contiguous()
        # ``active`` is [B,C,T,H,W].  The saved-step state is the final time
        # slice for every physical channel, not the full history of the last
        # channel.
        current = active[:, :, -1]
        raw = self.head(decoded, current, values, mask)
        difference = active[:, :, 1:] - active[:, :, :-1]
        step_scale = difference.square().mean((2, 3, 4)).sqrt()
        state_scale = active.square().mean((2, 3, 4)).sqrt()
        step_scale = step_scale.clamp_min(1.0e-4 * state_scale.clamp_min(1.0e-6))
        proposal = current + step_scale[:, :, None, None] * torch.tanh(raw)
        if self.program == "smoke_velocity_saved_step":
            proposal = torch.cat((proposal[:, :1].clamp_min(0.0), _closed_box_projection(proposal[:, 1:3])), 1)
        elif self.program == "shallow_water_saved_step":
            # The converted source exposes [u,v,div,vor,pres].  The precise
            # spherical metric metadata is absent, so we impose only the
            # source-safe closed-system mean constraint on the final scalar.
            scalar = proposal[:, 4:5]
            scalar = scalar - scalar.mean((2, 3), keepdim=True) + current[:, 4:5].mean((2, 3), keepdim=True)
            proposal = torch.cat((proposal[:, :4], scalar), 1)
        else:
            raise ValueError(self.program)
        return proposal.permute(0, 2, 3, 1).contiguous()


class PDEArenaV13TransferModel(nn.Module):
    """Exact V1.3 shared weights plus one randomly initialized typed task branch."""

    def __init__(self, channels: int, program: str) -> None:
        super().__init__()
        config = tiny_17m_config()
        if channels > config.max_fields:
            config = replace(config, max_fields=channels)
        boundary = "closed" if program == "smoke_velocity_saved_step" else "periodic"
        self.config = config
        self.chart = GeneralSemanticFieldChart(config, dynamic_channels=channels, boundary=boundary)
        self.shared_analyzer = SharedScalarFieldAnalyzer(config)
        self.private_mixer = PrivateTypedFieldMixer(config, dynamic_channels=channels)
        self.decoder = ActionContextDecoder(config)
        self.program = TypedSavedStepProgram(config, channels, program)

    def forward(self, history: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        features, field_mask, mean, log_scale, lags = self.chart(history)
        memory = self.shared_analyzer(features, field_mask, lags)
        private = self.private_mixer(memory, field_mask, mean, log_scale, values, mask)
        prediction = self.program(history, self.decoder(private), values, mask)
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("PDEArena prediction is non-finite")
        return prediction


__all__ = ["PDEArenaV13TransferModel", "TypedSavedStepProgram"]
