from __future__ import annotations

import torch
from torch import Tensor, nn

from hotpot._internal.data_protocol.cns_specialist import (
    PeriodicMarkovConfig,
    _current_chart,
)
from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
    ProgramId,
)
from hotpot._internal.data_protocol.native_charts import (
    fno_native64_exact_left_inverse,
)
from hotpot._internal.operators.experts import (
    PROGRAM_CODE,
)


def _coordinates(reference: Tensor) -> Tensor:
    batch, _channels, height, width = reference.shape
    y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0)[None].expand(batch, -1, -1, -1)


class ConditionFeature(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.channels = int(channels)
        self.net = nn.Sequential(
            nn.Linear(2 * len(CONDITION_FIELDS), 64),
            nn.GELU(),
            nn.Linear(64, self.channels),
        )

    def forward(
        self,
        conditions: Tensor,
        mask: Tensor,
        *,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> Tensor:
        payload = torch.cat((conditions * mask, mask), dim=1).to(dtype)
        return self.net(payload)[:, :, None, None].expand(-1, -1, height, width)


class _SpatialHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, padding_mode: str) -> None:
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class FNOCurrentStateHead(nn.Module):
    reads_full_history = False

    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.head = _SpatialHead(32 + 1 + 16 + 2, 1, padding_mode="circular")
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(self, decoded: Tensor, current: Tensor, conditions: Tensor, mask: Tensor) -> Tensor:
        self.call_count += 1
        context = fno_native64_exact_left_inverse(decoded)
        condition = self.condition(conditions, mask, height=64, width=64, dtype=decoded.dtype)
        return self.head(torch.cat((context, current, condition, _coordinates(current)), dim=1))


class DRCurrentStateHead(nn.Module):
    reads_full_history = False

    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.head = _SpatialHead(32 + 2 + 16 + 2, 4, padding_mode="replicate")
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(
        self,
        decoded: Tensor,
        current_bhwc: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.call_count += 1
        current = current_bhwc.permute(0, 3, 1, 2).contiguous()
        condition = self.condition(conditions, mask, height=128, width=128, dtype=decoded.dtype)
        raw = self.head(torch.cat((decoded, current, condition, _coordinates(current)), dim=1))
        return raw[:, :2].permute(0, 2, 3, 1), raw[:, 2:].permute(0, 2, 3, 1)


class DRKnownSourceResidualHead(nn.Module):
    """Endpoint closure after the registered reaction--diffusion step.

    The head has the same latent/current-state ABI as the legacy DR head, but
    it emits only the two-channel finite-step discrepancy left after executing
    the known source law and diffusion.  It never emits another diffusion or
    reaction term.
    """

    reads_full_history = False

    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.head = _SpatialHead(32 + 2 + 16 + 2, 2, padding_mode="replicate")
        # The deterministic DR step is already accurate.  Zero initialization
        # makes checkpoint migration start exactly from that known step.
        nn.init.zeros_(self.head.net[-1].weight)
        nn.init.zeros_(self.head.net[-1].bias)
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(
        self,
        decoded: Tensor,
        current_bhwc: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        current = current_bhwc.permute(0, 3, 1, 2).contiguous()
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        return self.head(
            torch.cat((decoded, current, condition, _coordinates(current)), dim=1)
        ).permute(0, 2, 3, 1)


class SWECurrentStateHead(nn.Module):
    """No private history difference: all learned H10 analysis comes from decoded."""

    reads_full_history = False

    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.head = _SpatialHead(32 + 1 + 16 + 2, 1, padding_mode="replicate")
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(
        self,
        decoded: Tensor,
        history_bthw: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        # The typed program owns the history-shaped compatibility port.  The
        # trainable head reads only x_t; previous frames have already crossed
        # the shared scalar-field analyzer.
        current = history_bthw[:, -1:]
        condition = self.condition(conditions, mask, height=128, width=128, dtype=decoded.dtype)
        return self.head(
            torch.cat((decoded, current, condition, _coordinates(current)), dim=1)
        )[:, 0]


class SWECalibratedIncrementHead(nn.Module):
    """Dimensionless increment shape plus one saved-step gain per sample."""

    reads_full_history = False

    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.head = _SpatialHead(32 + 1 + 16 + 2, 2, padding_mode="replicate")
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(
        self,
        decoded: Tensor,
        history_bthw: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.call_count += 1
        current = history_bthw[:, -1:]
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        raw = self.head(
            torch.cat((decoded, current, condition, _coordinates(current)), dim=1)
        )
        direction = raw[:, 0]
        # A spatial average makes the second output a scalar saved-step
        # calibration, not another unconstrained height field.
        log_gain = raw[:, 1].mean(dim=(-2, -1))
        return direction, log_gain


class CNSCurrentStateHead(nn.Module):
    reads_full_history = False

    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.config = PeriodicMarkovConfig()
        self.head = _SpatialHead(32 + 15 + 16, 4, padding_mode="circular")
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(self, decoded: Tensor, current_bhwc: Tensor, conditions: Tensor, mask: Tensor) -> Tensor:
        self.call_count += 1
        chart, _scale = _current_chart(current_bhwc, self.config)
        condition = self.condition(conditions, mask, height=128, width=128, dtype=decoded.dtype)
        return self.head(torch.cat((decoded, chart, condition), dim=1))


class _CFDBFamilyObservationHead(nn.Module):
    """One family-private direct saved-frame velocity readout."""

    reads_full_history = False

    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.head = _SpatialHead(32 + 4 + 16 + 2, 2, padding_mode="replicate")

    def forward(
        self,
        decoded: Tensor,
        current_bhwc: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        current = current_bhwc.permute(0, 3, 1, 2).contiguous()
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        return self.head(
            torch.cat((decoded, current, condition, _coordinates(current)), dim=1)
        ).permute(0, 2, 3, 1)


class CFDBDirectObservationHead(nn.Module):
    """Hard-routed cavity/cylinder/tube next-observation heads."""

    reads_full_history = False

    def __init__(self) -> None:
        super().__init__()
        self.family_heads = nn.ModuleDict(
            {
                "cavity": _CFDBFamilyObservationHead(),
                "cylinder": _CFDBFamilyObservationHead(),
                "tube": _CFDBFamilyObservationHead(),
            }
        )
        self.call_count = 0
        self.last_route_counts = {family: 0 for family in self.family_heads}
        self._codes = {
            "cavity": PROGRAM_CODE[ProgramId.CFDB_CAVITY],
            "cylinder": PROGRAM_CODE[ProgramId.CFDB_CYLINDER],
            "tube": PROGRAM_CODE[ProgramId.CFDB_TUBE],
        }

    def reset_calls(self) -> None:
        self.call_count = 0
        self.last_route_counts = {family: 0 for family in self.family_heads}

    def forward(
        self,
        decoded: Tensor,
        current_bhwc: Tensor,
        program_codes: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        if program_codes.shape != (decoded.shape[0],):
            raise ValueError("CFDB program codes must be [B]")
        prediction = current_bhwc.new_empty(
            current_bhwc.shape[0], current_bhwc.shape[1], current_bhwc.shape[2], 2
        )
        for family, code in self._codes.items():
            index = (program_codes == code).nonzero(as_tuple=False).flatten()
            self.last_route_counts[family] = int(index.numel())
            if index.numel() == 0:
                continue
            value = self.family_heads[family](
                decoded.index_select(0, index),
                current_bhwc.index_select(0, index),
                conditions.index_select(0, index),
                mask.index_select(0, index),
            )
            prediction.index_copy_(0, index, value)
        return prediction


__all__ = [
    "CFDBDirectObservationHead",
    "CNSCurrentStateHead",
    "DRCurrentStateHead",
    "DRKnownSourceResidualHead",
    "FNOCurrentStateHead",
    "SWECalibratedIncrementHead",
    "SWECurrentStateHead",
]
