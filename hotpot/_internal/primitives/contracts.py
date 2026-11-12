from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from hotpot._internal.data_protocol.manifest import ProgramId


class StepKind(str, Enum):
    CONDITION = "condition"
    TRANSFORM = "transform"
    SUBTRACT_BASE = "subtract_base"
    MEAN_FREE = "mean_free"
    SET_FROM_BASE = "set_from_base"
    ADD_FROM_BASE = "add_from_base"


@dataclass(frozen=True)
class PrimitiveContract:
    identifier: str
    step_kind: StepKind
    input_chart: str
    output_chart: str
    boundary: str
    mathematical_role: str
    condition_fields: tuple[str, ...]
    parameter_scope: str
    owner_family: str
    qualified_programs: tuple[ProgramId, ...]
    qualified_environments: tuple[str, ...] = ()
    parameterized: bool = True
    qualification: str = "contract_exact"

    def payload(self) -> dict[str, object]:
        raw = asdict(self)
        raw["step_kind"] = self.step_kind.value
        raw["qualified_programs"] = [value.value for value in self.qualified_programs]
        return raw


@dataclass(frozen=True)
class ProgramStep:
    kind: StepKind
    primitive_id: str | None = None


@dataclass(frozen=True)
class LatentProgramSpec:
    identifier: ProgramId
    family: str
    steps: tuple[ProgramStep, ...]

    def learned_primitive_ids(self) -> tuple[str, ...]:
        return tuple(step.primitive_id for step in self.steps if step.primitive_id)

    def payload(self) -> dict[str, object]:
        return {
            "identifier": self.identifier.value,
            "family": self.family,
            "steps": [
                {"kind": step.kind.value, "primitive_id": step.primitive_id}
                for step in self.steps
            ],
        }


@dataclass(frozen=True)
class PhysicalOperatorContract:
    identifier: str
    input_chart: str
    output_chart: str
    boundary: str
    mathematical_role: str
    qualified_programs: tuple[ProgramId, ...]
    source: str
    parameterized: bool = False

    def payload(self) -> dict[str, object]:
        raw = asdict(self)
        raw["qualified_programs"] = [value.value for value in self.qualified_programs]
        return raw


__all__ = [
    "LatentProgramSpec",
    "PrimitiveContract",
    "PhysicalOperatorContract",
    "ProgramStep",
    "StepKind",
]
