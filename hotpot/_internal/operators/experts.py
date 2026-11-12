from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
    ProgramId,
)
from hotpot._internal.data_protocol.production_primitives import (
    DepthwiseResidualBlock,
    PointwiseFFN,
    SpectralResidualBlock,
)


PROGRAM_ORDER = tuple(ProgramId)
PROGRAM_CODE = {program: index for index, program in enumerate(PROGRAM_ORDER)}
FAMILY_BY_PROGRAM = {
    ProgramId.FNO_VORTICITY_NS: "fno",
    ProgramId.CNS_COMPRESSIBLE: "cns",
    ProgramId.SWE_HEIGHT: "swe",
    ProgramId.DR_REACTION_DIFFUSION: "dr",
    ProgramId.CFDB_CAVITY: "cfdb",
    ProgramId.CFDB_CYLINDER: "cfdb",
    ProgramId.CFDB_TUBE: "cfdb",
}
CFDB_PROGRAMS = (
    ProgramId.CFDB_CAVITY,
    ProgramId.CFDB_CYLINDER,
    ProgramId.CFDB_TUBE,
)


def normalize_programs(
    programs: ProgramId | str | Sequence[ProgramId | str],
    *,
    batch: int,
    device: torch.device,
) -> tuple[tuple[ProgramId, ...], Tensor]:
    if isinstance(programs, (ProgramId, str)):
        raw = (programs,) * batch
    else:
        raw = tuple(programs)
    if len(raw) != batch:
        raise ValueError(f"expected {batch} program IDs, received {len(raw)}")
    normalized: list[ProgramId] = []
    for value in raw:
        try:
            normalized.append(value if isinstance(value, ProgramId) else ProgramId(value))
        except ValueError as error:
            raise ValueError(f"unknown physical program {value!r}") from error
    codes = torch.tensor(
        [PROGRAM_CODE[value] for value in normalized],
        dtype=torch.long,
        device=device,
    )
    return tuple(normalized), codes


class ConditionFiLM(nn.Module):
    """Task-owned conditioning; zero initialization preserves an unmodulated ABI."""

    def __init__(self, width: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or max(32, min(128, width // 4))
        self.net = nn.Sequential(
            nn.Linear(2 * len(CONDITION_FIELDS), hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * width),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: Tensor, conditions: Tensor, mask: Tensor) -> Tensor:
        if conditions.shape != mask.shape or conditions.shape[1] != len(CONDITION_FIELDS):
            raise ValueError("conditions/mask must be [B,len(CONDITION_FIELDS)]")
        payload = torch.cat((conditions * mask, mask), dim=1).to(value.dtype)
        gain, shift = self.net(payload).chunk(2, dim=1)
        return value * (1.0 + 0.1 * torch.tanh(gain)[:, :, None, None]) + (
            0.1 * torch.tanh(shift)[:, :, None, None]
        )


class PeriodicSpectralTransportExpert(nn.Module):
    """FNO-family periodic spectral transport representation update."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.condition = ConditionFiLM(width)
        self.block = SpectralResidualBlock(
            width,
            num_blocks=4 if width % 4 == 0 else 1,
            ffn_hidden=5 * width,
        )

    def forward(self, value: Tensor, conditions: Tensor, mask: Tensor, _codes: Tensor) -> Tensor:
        base = self.condition(value, conditions, mask)
        return self.block(base) - base


class ReactionDiffusionExpert(nn.Module):
    """Closed-boundary local reaction/diffusion latent update."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.condition = ConditionFiLM(width)
        hidden = max(64, 11 * width // 4)
        self.blocks = nn.ModuleList(
            DepthwiseResidualBlock(width, hidden, padding_mode="replicate")
            for _ in range(2)
        )
        for block in self.blocks:
            nn.init.constant_(block.scale, 1.0e-3)

    def forward(self, value: Tensor, conditions: Tensor, mask: Tensor, _codes: Tensor) -> Tensor:
        base = self.condition(value, conditions, mask)
        current = base
        for block in self.blocks:
            current = block(current)
        return current - base


class ShallowWaterPotentialExpert(nn.Module):
    """Long-range potential carrier plus closed-boundary local correction."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.condition = ConditionFiLM(width)
        self.potential = SpectralResidualBlock(
            width,
            num_blocks=4 if width % 4 == 0 else 1,
            ffn_hidden=15 * width // 4,
        )
        self.boundary = DepthwiseResidualBlock(
            width, max(64, 5 * width // 4), padding_mode="replicate"
        )
        nn.init.constant_(self.boundary.scale, 1.0e-3)

    def forward(self, value: Tensor, conditions: Tensor, mask: Tensor, _codes: Tensor) -> Tensor:
        base = self.condition(value, conditions, mask)
        return self.boundary(self.potential(base)) - base


class ConservativePeriodicExpert(nn.Module):
    """CNS-family periodic local update with a zero spatial-mean action."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.condition = ConditionFiLM(width)
        hidden = max(64, 11 * width // 4)
        self.blocks = nn.ModuleList(
            DepthwiseResidualBlock(width, hidden, padding_mode="circular")
            for _ in range(2)
        )
        for block in self.blocks:
            nn.init.constant_(block.scale, 1.0e-3)

    def forward(self, value: Tensor, conditions: Tensor, mask: Tensor, _codes: Tensor) -> Tensor:
        base = self.condition(value, conditions, mask)
        current = base
        for block in self.blocks:
            current = block(current)
        update = current - base
        return update - update.mean(dim=(-2, -1), keepdim=True)


class _LowRankFamilyPort(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(width, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, width, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: Tensor) -> Tensor:
        return self.up(torch.nn.functional.gelu(self.down(value)))


class CFDBObservationFlowExpert(nn.Module):
    """CFDB-shared observation flow with one statically selected family port."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.condition = ConditionFiLM(width)
        # 2736 hidden channels at width 512 keeps the selected CFDB path at
        # the same approximately-17M active budget as the other programs.
        self.shared_flow = PointwiseFFN(width, max(64, 171 * width // 32))
        rank = max(8, width // 16)
        self.family_ports = nn.ModuleDict(
            {
                "cavity": _LowRankFamilyPort(width, rank),
                "cylinder": _LowRankFamilyPort(width, rank),
                "tube": _LowRankFamilyPort(width, rank),
            }
        )
        self._family_code = {
            PROGRAM_CODE[ProgramId.CFDB_CAVITY]: "cavity",
            PROGRAM_CODE[ProgramId.CFDB_CYLINDER]: "cylinder",
            PROGRAM_CODE[ProgramId.CFDB_TUBE]: "tube",
        }

    def forward(self, value: Tensor, conditions: Tensor, mask: Tensor, codes: Tensor) -> Tensor:
        base = self.condition(value, conditions, mask)
        update = self.shared_flow(base)
        family_update = torch.zeros_like(update)
        for code, family in self._family_code.items():
            index = torch.nonzero(codes == code, as_tuple=False).flatten()
            if index.numel() == 0:
                continue
            selected = self.family_ports[family](base.index_select(0, index))
            family_update = family_update.index_copy(0, index, selected)
        return update + family_update

    def active_parameter_numel(self, program: ProgramId) -> int:
        family = {
            ProgramId.CFDB_CAVITY: "cavity",
            ProgramId.CFDB_CYLINDER: "cylinder",
            ProgramId.CFDB_TUBE: "tube",
        }[program]
        modules = (self.condition, self.shared_flow, self.family_ports[family])
        return sum(parameter.numel() for module in modules for parameter in module.parameters())


class StaticHeterogeneousExpertLayer(nn.Module):
    """Gather/execute/scatter dispatch with exactly one family expert per sample."""

    def __init__(self, width: int = 512) -> None:
        super().__init__()
        self.width = int(width)
        self.experts = nn.ModuleDict(
            {
                "fno": PeriodicSpectralTransportExpert(width),
                "dr": ReactionDiffusionExpert(width),
                "swe": ShallowWaterPotentialExpert(width),
                "cns": ConservativePeriodicExpert(width),
                "cfdb": CFDBObservationFlowExpert(width),
            }
        )
        self.call_counts = {key: 0 for key in self.experts}

    def reset_call_counts(self) -> None:
        for key in self.call_counts:
            self.call_counts[key] = 0

    def forward(
        self,
        value: Tensor,
        program_codes: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        if value.ndim != 4 or value.shape[1] != self.width:
            raise ValueError(f"expert ABI requires [B,{self.width},H,W]")
        if program_codes.shape != (value.shape[0],):
            raise ValueError("program_codes must be [B]")
        output = torch.empty_like(value)
        for family, expert in self.experts.items():
            legal_codes = [
                PROGRAM_CODE[program]
                for program, owner in FAMILY_BY_PROGRAM.items()
                if owner == family
            ]
            selected = torch.zeros_like(program_codes, dtype=torch.bool)
            for code in legal_codes:
                selected |= program_codes == code
            index = torch.nonzero(selected, as_tuple=False).flatten()
            if index.numel() == 0:
                continue
            self.call_counts[family] += 1
            update = expert(
                value.index_select(0, index),
                conditions.index_select(0, index),
                condition_mask.index_select(0, index),
                program_codes.index_select(0, index),
            )
            if update.shape != value.index_select(0, index).shape:
                raise RuntimeError(f"{family} expert violated the same-shape ABI")
            output = output.index_copy(0, index, update)
        return output


__all__ = [
    "CFDB_PROGRAMS",
    "FAMILY_BY_PROGRAM",
    "PROGRAM_CODE",
    "PROGRAM_ORDER",
    "StaticHeterogeneousExpertLayer",
    "normalize_programs",
]
