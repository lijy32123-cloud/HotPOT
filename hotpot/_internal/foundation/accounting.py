from __future__ import annotations

from hotpot._internal.encoder.accounting import (
    parameter_ledger as _parameter_ledger,
)

from .config import FoundationScaleConfig
from .model import SemanticFieldProgramFoundationS90M


def parameter_ledger(model: SemanticFieldProgramFoundationS90M) -> dict[str, object]:
    payload = _parameter_ledger(model)
    payload["schema"] = "semantic_field_program_foundation.parameter_ledger.v1.1-s90m"
    return payload


def build_and_count(config: FoundationScaleConfig | None = None) -> dict[str, object]:
    return parameter_ledger(SemanticFieldProgramFoundationS90M(config))


__all__ = ["build_and_count", "parameter_ledger"]
