from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .factorized_core import (
    HistoryContext,
    StageFeatures,
    TypedOperatorExpert,
)

from .cfdb_sidecar import CertifiedCFDBSidecar
from .manifest import BASE6, PROGRAMS, ProgramId
from .temporal import UniversalSharedCarrier


OUTPUT_CHANNELS: Mapping[ProgramId, int] = {
    ProgramId.FNO_VORTICITY_NS: 1,
    ProgramId.CNS_COMPRESSIBLE: 4,
    ProgramId.SWE_HEIGHT: 1,
    ProgramId.DR_REACTION_DIFFUSION: 4,
    ProgramId.CFDB_CAVITY: 2,
    ProgramId.CFDB_CYLINDER: 2,
    ProgramId.CFDB_TUBE: 2,
}


class BoundaryNeutralTransportCore(nn.Module):
    """One actually shared transport mechanism without a boundary stencil.

    FNO-NS is periodic whereas PDEArena smoke uses a bounded zero-velocity
    box.  Sharing a circular or replicated depthwise convolution would bake
    one task's boundary condition into the other.  This residual 1x1 core
    instead mixes the common latent mechanism coordinates pointwise; each
    program keeps its own boundary-aware private blocks and typed head.
    """

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, width)
        self.in_projection = nn.Conv2d(width, hidden, 1)
        self.out_projection = nn.Conv2d(hidden, width, 1)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, value: Tensor) -> Tensor:
        hidden = F.gelu(self.in_projection(self.norm(value)))
        return value + torch.tanh(self.scale) * self.out_projection(hidden)


class ProgramTypedOperatorExpert(TypedOperatorExpert):
    """Typed expert with early program branching through a shared basis.

    The spectral basis is serialized once by the carrier.  Program-owned
    mechanism blocks are interleaved after successive basis transforms, so
    different PDEs stop sharing an identical deep hidden trajectory after the
    first shallow stem.  The tuple is deliberately non-registering here.
    """

    def __init__(
        self,
        *args,
        output_size: int,
        shared_mechanisms: tuple[nn.Module, ...] = (),
        shared_basis: tuple[nn.Module, ...] = (),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if output_size < 1:
            raise ValueError("output_size must be positive")
        if len(shared_mechanisms) > len(self.mechanism_dictionary):
            raise ValueError("shared mechanism count exceeds configured mechanism depth")
        self.output_size = int(output_size)
        # Replace an equal number of private blocks rather than adding capacity.
        # This preserves the Tiny active-parameter budget and makes reuse real.
        self.mechanism_dictionary = nn.ModuleList(
            list(self.mechanism_dictionary)[len(shared_mechanisms):]
        )
        # Keep a non-registering reference here.  The registry owns and
        # serializes the shared module exactly once in `shared_mechanisms`.
        self.shared_mechanisms = tuple(shared_mechanisms)
        self.shared_basis = tuple(shared_basis)

    def forward(self, context: HistoryContext, stage: StageFeatures) -> Tensor:
        value = context.latent + stage.latent
        for block in self.shared_mechanisms:
            value = block(value)
        typed_blocks = tuple(self.mechanism_dictionary)
        for index, block in enumerate(self.shared_basis):
            value = block(value)
            if index < len(typed_blocks):
                value = typed_blocks[index](value)
        for block in typed_blocks[len(self.shared_basis):]:
            value = block(value)
        for block in self.operator_adapter:
            value = block(value)
        upsampled = F.interpolate(
            value,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        detail_carrier = F.interpolate(
            self.detail_carrier(stage.carrier),
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        detail = self.detail_fusion(torch.cat((upsampled, detail_carrier), dim=1))
        return self.coarse_head(upsampled) + self.detail_head(detail)


class HardProgramRegistry(nn.Module):
    """One typed bank per legal program; there is no learned routing score."""

    def __init__(
        self,
        carrier: UniversalSharedCarrier | None,
        programs: Iterable[ProgramId] | None = None,
    ) -> None:
        super().__init__()
        selected = tuple(ProgramId) if programs is None else tuple(programs)
        # Base6 has no second cross-PDE owner with the same exact operator
        # contract.  Reuse is instead explicit inside the FNO viscosity family
        # and the three CFDB family heads supplied by the specialist bank.
        self.shared_mechanisms = nn.ModuleDict({})
        self.programs = nn.ModuleDict()
        for identifier in selected:
            if carrier is None:
                raise ValueError("generic program registry requires a carrier")
            spec = PROGRAMS[identifier]
            config = carrier.expert_config(boundary_padding=(
                "circular" if spec.boundary == "periodic" else "replicate"
            ))
            expert = ProgramTypedOperatorExpert(
                config,
                signature=f"{identifier.value}:{spec.output_type}",
                out_channels=OUTPUT_CHANNELS[identifier],
                zero_init=True,
                output_size=64 if identifier is ProgramId.FNO_VORTICITY_NS else 128,
                shared_mechanisms=(),
                shared_basis=tuple(carrier.shared_core.operator_basis),
            )
            self.programs[identifier.value] = expert
            for parameter in expert.parameters():
                parameter._base6_owner = f"program::{identifier.value}"  # type: ignore[attr-defined]

    def expert(self, identifier: ProgramId) -> ProgramTypedOperatorExpert:
        if identifier.value not in self.programs:
            raise KeyError(f"program is delegated to a qualified specialist: {identifier.value}")
        return self.programs[identifier.value]

    @staticmethod
    def resolve(
        dataset_index: int,
        row_index: int,
        cfdb_sidecar: CertifiedCFDBSidecar | None,
    ) -> ProgramId:
        if dataset_index < 0 or dataset_index >= len(BASE6):
            raise IndexError(dataset_index)
        entry = BASE6[dataset_index]
        if entry.program is not None:
            return entry.program
        if entry.key != "cfdbench" or cfdb_sidecar is None:
            raise ValueError("CFDB rows require a certified sidecar")
        return cfdb_sidecar.program(row_index)

    @classmethod
    def group_indices(
        cls,
        dataset_indices: Iterable[int],
        row_indices: Iterable[int],
        cfdb_sidecar: CertifiedCFDBSidecar | None,
    ) -> dict[ProgramId, tuple[int, ...]]:
        datasets = tuple(int(value) for value in dataset_indices)
        rows = tuple(int(value) for value in row_indices)
        if len(datasets) != len(rows):
            raise ValueError("dataset and row index lengths differ")
        grouped: dict[ProgramId, list[int]] = defaultdict(list)
        for position, (dataset, row) in enumerate(zip(datasets, rows, strict=True)):
            grouped[cls.resolve(dataset, row, cfdb_sidecar)].append(position)
        return {key: tuple(value) for key, value in grouped.items()}

    def ownership(self) -> dict[str, int]:
        ledger: dict[str, int] = defaultdict(int)
        for parameter in self.parameters():
            owner = str(getattr(parameter, "_base6_owner", "UNOWNED"))
            ledger[owner] += parameter.numel()
        if "UNOWNED" in ledger:
            raise RuntimeError("program bank contains unowned parameters")
        return dict(sorted(ledger.items()))


__all__ = [
    "BoundaryNeutralTransportCore",
    "HardProgramRegistry",
    "OUTPUT_CHANNELS",
]
