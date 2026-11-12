from __future__ import annotations

from torch import Tensor, nn

from hotpot._internal.data_protocol.cfdb_sidecar import CertifiedCFDBSidecar
from hotpot._internal.data_protocol.data import CFDB_CONDITION
from hotpot._internal.data_protocol.manifest import BASE6, CONDITION_FIELDS
from hotpot._internal.data_protocol.registry import HardProgramRegistry

from .accounting import parameter_ledger
from .config import FoundationScaleConfig
from hotpot._internal.encoder.model import (
    LEGACY_INTERFACE_REVISION,
)
from .model import SemanticFieldProgramFoundationS90M


class Base6SemanticFieldFoundationS90M(nn.Module):
    def __init__(
        self,
        workspace=None,
        config: FoundationScaleConfig | None = None,
        *,
        interface_revision: str = LEGACY_INTERFACE_REVISION,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.model = SemanticFieldProgramFoundationS90M(
            config, interface_revision=interface_revision
        )

    def forward(
        self,
        history: Tensor,
        dataset_indices: Tensor,
        row_indices: Tensor,
        *,
        condition_values: Tensor | None = None,
        condition_mask: Tensor | None = None,
        cfdb_sidecar: CertifiedCFDBSidecar | None,
        return_audit: bool = False,
    ):
        batch = history.shape[0]
        if dataset_indices.shape != (batch,) or row_indices.shape != (batch,):
            raise ValueError("dataset_indices and row_indices must be [B]")
        if (condition_values is None) != (condition_mask is None):
            raise ValueError("condition values and mask must be supplied together")
        if condition_values is None:
            condition_values = history.new_zeros(batch, len(CONDITION_FIELDS))
            condition_mask = history.new_zeros(batch, len(CONDITION_FIELDS))
            for position, (dataset, row) in enumerate(
                zip(dataset_indices.detach().cpu().tolist(), row_indices.detach().cpu().tolist(), strict=True)
            ):
                mapping = BASE6[int(dataset)].condition_mapping()
                if int(dataset) == 5:
                    if cfdb_sidecar is None:
                        raise ValueError("CFDB conditions require a certified sidecar")
                    mapping = CFDB_CONDITION[cfdb_sidecar.family(int(row))]
                for name, value in mapping.items():
                    column = CONDITION_FIELDS.index(name)
                    condition_values[position, column] = float(value)
                    condition_mask[position, column] = 1.0
        assert condition_mask is not None
        programs = tuple(
            HardProgramRegistry.resolve(int(dataset), int(row), cfdb_sidecar)
            for dataset, row in zip(
                dataset_indices.detach().cpu().tolist(),
                row_indices.detach().cpu().tolist(),
                strict=True,
            )
        )
        return self.model(
            history,
            programs,
            condition_values,
            condition_mask,
            return_audit=return_audit,
        )

    def parameter_ledger(self) -> dict[str, object]:
        return parameter_ledger(self.model)


__all__ = ["Base6SemanticFieldFoundationS90M"]
