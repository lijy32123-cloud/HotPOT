from __future__ import annotations

import torch
import torch.nn.functional as F
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

from .experts import PROGRAM_CODE


def _coordinates(reference: Tensor) -> Tensor:
    batch, _channels, height, width = reference.shape
    y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


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
        self, conditions: Tensor, mask: Tensor, *, height: int, width: int, dtype: torch.dtype
    ) -> Tensor:
        payload = torch.cat((conditions * mask, mask), dim=1).to(dtype)
        return self.net(payload)[:, :, None, None].expand(-1, -1, height, width)


class _SpatialHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, padding_mode: str) -> None:
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class FNOTransportHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.head = _SpatialHead(32 + 1 + 16 + 2, 1, padding_mode="circular")
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(
        self,
        decoded: Tensor,
        current: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        context = fno_native64_exact_left_inverse(decoded)
        condition = self.condition(
            conditions, mask, height=64, width=64, dtype=decoded.dtype
        )
        return self.head(torch.cat((context, current, condition, _coordinates(current)), dim=1))


class DRClosureHead(nn.Module):
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
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        raw = self.head(torch.cat((decoded, current, condition, _coordinates(current)), dim=1))
        return raw[:, :2].permute(0, 2, 3, 1), raw[:, 2:].permute(0, 2, 3, 1)


class SWEControlHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.head = _SpatialHead(32 + 2 + 16 + 2, 1, padding_mode="replicate")
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
        current = history_bthw[:, -1:]
        previous_difference = current - history_bthw[:, -2:-1]
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        return self.head(
            torch.cat(
                (decoded, current, previous_difference, condition, _coordinates(current)),
                dim=1,
            )
        )[:, 0]


class CNSConservativeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.config = PeriodicMarkovConfig()
        self.head = _SpatialHead(32 + 15 + 16, 4, padding_mode="circular")
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
        chart, _scale = _current_chart(current_bhwc, self.config)
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        return self.head(torch.cat((decoded, chart, condition), dim=1))


class CFDBObservationFlowHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition = ConditionFeature()
        self.head = _SpatialHead(32 + 4 + 16 + 2 + 3, 2, padding_mode="replicate")
        self.call_count = 0
        self._codes = (
            PROGRAM_CODE[ProgramId.CFDB_CAVITY],
            PROGRAM_CODE[ProgramId.CFDB_CYLINDER],
            PROGRAM_CODE[ProgramId.CFDB_TUBE],
        )

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(
        self,
        decoded: Tensor,
        current_bhwc: Tensor,
        program_codes: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        current = current_bhwc.permute(0, 3, 1, 2).contiguous()
        family = torch.stack(
            tuple((program_codes == code).to(decoded.dtype) for code in self._codes), dim=1
        )[:, :, None, None].expand(-1, -1, 128, 128)
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        return self.head(
            torch.cat((decoded, current, condition, _coordinates(current), family), dim=1)
        ).permute(0, 2, 3, 1)


__all__ = [
    "CFDBObservationFlowHead",
    "CNSConservativeHead",
    "DRClosureHead",
    "FNOTransportHead",
    "SWEControlHead",
]
