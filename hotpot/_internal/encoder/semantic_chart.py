from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hotpot._internal.data_protocol.native_charts import (
    fno_native64_exact_left_inverse,
)

from .config import FoundationConfig


SEMANTIC_FEATURES = (
    "normalized_value",
    "saved_step_first_difference",
    "saved_step_second_difference",
    "grid_central_difference_x",
    "grid_central_difference_y",
    "grid_laplacian",
    "coordinate_x",
    "coordinate_y",
    "boundary_distance",
    "geometry_support",
    "field_validity",
)


@dataclass(frozen=True)
class SemanticHistory:
    features: Tensor
    field_mask: Tensor
    field_mean: Tensor
    field_log_scale: Tensor
    relative_lags: Tensor


class SemanticFieldChart(nn.Module):
    """Convert one task family into fixed, named scalar-field features."""

    def __init__(
        self,
        config: FoundationConfig,
        *,
        dynamic_channels: int,
        geometry_channel: int | None,
        boundary: str,
        native_size: int,
    ) -> None:
        super().__init__()
        config.validate()
        if not 1 <= dynamic_channels <= config.max_fields:
            raise ValueError("dynamic_channels outside scalar-field ABI")
        if geometry_channel is not None and not 0 <= geometry_channel < 4:
            raise ValueError("geometry channel must be a public channel index")
        if boundary not in {"periodic", "closed"}:
            raise ValueError("boundary must be periodic or closed")
        if native_size not in {64, config.resolution}:
            raise ValueError("the scalar-field chart supports native64 or public resolution")
        if config.semantic_channels != len(SEMANTIC_FEATURES):
            raise ValueError("semantic channel count drifted from named feature ABI")
        self.config = config
        self.dynamic_channels = int(dynamic_channels)
        self.geometry_channel = geometry_channel
        self.boundary = boundary
        self.native_size = int(native_size)

    def _to_native(self, value: Tensor) -> Tensor:
        if self.native_size == self.config.resolution:
            return value
        batch, fields, steps, height, width = value.shape
        if (height, width) != (self.config.resolution, self.config.resolution):
            raise ValueError("public chart resolution drifted")
        packed = value.reshape(batch * fields, steps, height, width)
        native = fno_native64_exact_left_inverse(packed)
        return native.reshape(batch, fields, steps, self.native_size, self.native_size)

    def _spatial_differences(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # value: [B,F,T,H,W].  Differences are dimensionless grid operators;
        # physical dx/dy remain typed metadata for the private action/program.
        flat = value.flatten(0, 2)[:, None]
        if self.boundary == "periodic":
            left = torch.roll(flat, 1, dims=-1)
            right = torch.roll(flat, -1, dims=-1)
            down = torch.roll(flat, 1, dims=-2)
            up = torch.roll(flat, -1, dims=-2)
        else:
            padded = F.pad(flat, (1, 1, 1, 1), mode="replicate")
            left = padded[..., 1:-1, :-2]
            right = padded[..., 1:-1, 2:]
            down = padded[..., :-2, 1:-1]
            up = padded[..., 2:, 1:-1]
        grad_x = 0.5 * (right - left)
        grad_y = 0.5 * (up - down)
        laplacian = left + right + down + up - 4.0 * flat
        shape = value.shape
        return (
            grad_x[:, 0].reshape(shape),
            grad_y[:, 0].reshape(shape),
            laplacian[:, 0].reshape(shape),
        )

    def _coordinates(self, reference: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, fields, steps, height, width = reference.shape
        y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        if self.boundary == "periodic":
            distance = torch.ones_like(xx)
        else:
            distance = torch.minimum(1.0 - xx.abs(), 1.0 - yy.abs()).clamp_min(0.0)
        shape = (1, 1, 1, height, width)
        expanded = (batch, fields, steps, height, width)
        return tuple(item.reshape(shape).expand(expanded) for item in (xx, yy, distance))

    def forward(self, history: Tensor) -> SemanticHistory:
        expected = (
            self.config.resolution,
            self.config.resolution,
            self.config.history_steps,
            4,
        )
        if history.ndim != 5 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"public history must be [B,{expected}]")
        batch = history.shape[0]
        dynamic = history[..., : self.dynamic_channels].permute(0, 4, 3, 1, 2)
        dynamic = self._to_native(dynamic)
        mean = dynamic.mean(dim=(2, 3, 4), keepdim=True)
        scale = dynamic.std(dim=(2, 3, 4), keepdim=True).clamp_min(1.0e-6)
        normalized_active = (dynamic - mean) / scale

        normalized = history.new_zeros(
            batch,
            self.config.max_fields,
            self.config.history_steps,
            self.native_size,
            self.native_size,
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
        grad_x, grad_y, laplacian = self._spatial_differences(normalized)
        xx, yy, boundary_distance = self._coordinates(normalized)

        if self.geometry_channel is None:
            geometry = torch.ones_like(normalized)
        else:
            public_support = history[..., self.geometry_channel].permute(0, 3, 1, 2)
            support = self._to_native(public_support[:, None])[:, 0].mean(dim=1).clamp(0.0, 1.0)
            geometry = support[:, None, None].expand_as(normalized)
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
                boundary_distance,
                geometry,
                validity,
            ),
            dim=3,
        )
        features = features * validity[:, :, :, None]
        if self.native_size != self.config.resolution:
            packed = features.reshape(
                batch * self.config.max_fields * self.config.history_steps,
                self.config.semantic_channels,
                self.native_size,
                self.native_size,
            )
            features = F.interpolate(
                packed,
                size=(self.config.resolution, self.config.resolution),
                mode="bilinear",
                align_corners=False,
            ).reshape(
                batch,
                self.config.max_fields,
                self.config.history_steps,
                self.config.semantic_channels,
                self.config.resolution,
                self.config.resolution,
            )
        field_mean = history.new_zeros(batch, self.config.max_fields)
        field_log_scale = history.new_zeros(batch, self.config.max_fields)
        field_mean[:, : self.dynamic_channels] = mean[:, :, 0, 0, 0]
        field_log_scale[:, : self.dynamic_channels] = scale[:, :, 0, 0, 0].log()
        relative_lags = torch.linspace(
            -1.0,
            0.0,
            self.config.history_steps,
            device=history.device,
            dtype=history.dtype,
        )[None].expand(batch, -1)
        return SemanticHistory(
            features=features,
            field_mask=field_mask,
            field_mean=field_mean,
            field_log_scale=field_log_scale,
            relative_lags=relative_lags,
        )


__all__ = ["SEMANTIC_FEATURES", "SemanticFieldChart", "SemanticHistory"]
