from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Mapping


HISTORY = 10
TARGET_STEPS = 1
MODEL_RESOLUTION = 128
CONDITION_FIELDS = (
    "observation_dt",
    "dx",
    "dy",
    "viscosity",
    "combined_dissipation",
    "mach_regime_tag",
    "buoyancy_y",
    "gravity",
)


class ProgramId(str, Enum):
    FNO_VORTICITY_NS = "fno_vorticity_ns"
    CNS_COMPRESSIBLE = "cns_compressible"
    SWE_HEIGHT = "swe_height"
    DR_REACTION_DIFFUSION = "dr_reaction_diffusion"
    CFDB_CAVITY = "cfdb_cavity"
    CFDB_CYLINDER = "cfdb_cylinder"
    CFDB_TUBE = "cfdb_tube"


@dataclass(frozen=True)
class ProgramSpec:
    identifier: ProgramId
    family: str
    output_type: str
    numerical_program: str
    boundary: str
    globally_shared: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    index: int
    key: str
    program: ProgramId | None
    family: str
    train_path: str
    storage: str
    ntrain: int
    available_train_rows: int
    data_weight: int
    channels: int
    predicted_channels: int
    native_size: int
    total_frames: int
    observation_dt: float
    condition: tuple[tuple[str, float], ...] = ()
    row_program_sidecar: str | None = None

    @property
    def weighted_rows(self) -> int:
        return self.ntrain * self.data_weight

    def condition_mapping(self) -> dict[str, float]:
        return dict(self.condition)


PROGRAMS: Mapping[ProgramId, ProgramSpec] = {
    ProgramId.FNO_VORTICITY_NS: ProgramSpec(
        ProgramId.FNO_VORTICITY_NS,
        "incompressible_vorticity_ns",
        "scalar_transport_rate",
        "exact_CN_diffusion_plus_strang_midpoint",
        "periodic",
    ),
    ProgramId.CNS_COMPRESSIBLE: ProgramSpec(
        ProgramId.CNS_COMPRESSIBLE,
        "compressible_ns",
        "zero_mean_conservative_action",
        "periodic_alias_safe_action_plus_positivity",
        "periodic",
    ),
    ProgramId.SWE_HEIGHT: ProgramSpec(
        ProgramId.SWE_HEIGHT,
        "height_observed_shallow_water",
        "height_observable_effective_acceleration",
        "kick_drift_mean_preserving_positivity",
        "closed_neumann",
    ),
    ProgramId.DR_REACTION_DIFFUSION: ProgramSpec(
        ProgramId.DR_REACTION_DIFFUSION,
        "fitzhugh_nagumo_reaction_diffusion",
        "unit_diffusion_correction_and_local_reaction_rate",
        "exact_neumann_heat_plus_strang_midpoint",
        "closed_neumann",
    ),
    ProgramId.CFDB_CAVITY: ProgramSpec(
        ProgramId.CFDB_CAVITY,
        "cfdb_cavity",
        "history_conditioned_observation_tendency",
        "cavity_saved_cadence_boundary_program",
        "cavity_walls",
    ),
    ProgramId.CFDB_CYLINDER: ProgramSpec(
        ProgramId.CFDB_CYLINDER,
        "cfdb_cylinder",
        "history_conditioned_observation_tendency",
        "cylinder_saved_cadence_boundary_program",
        "inlet_outlet_obstacle",
    ),
    ProgramId.CFDB_TUBE: ProgramSpec(
        ProgramId.CFDB_TUBE,
        "cfdb_tube",
        "history_conditioned_observation_tendency",
        "tube_saved_cadence_boundary_program",
        "inlet_outlet_walls",
    ),
}


def _condition(**values: float) -> tuple[tuple[str, float], ...]:
    unknown = set(values).difference(CONDITION_FIELDS)
    if unknown:
        raise ValueError(f"unknown condition fields: {sorted(unknown)}")
    return tuple((name, float(values[name])) for name in CONDITION_FIELDS if name in values)


BASE6: tuple[DatasetSpec, ...] = (
    DatasetSpec(0, "ns2d_fno_1e-5", ProgramId.FNO_VORTICITY_NS,
                "fno_vorticity", "data/large/ns2d_1e-5_train.hdf5", "packed_hdf5",
                1000, 1000, 1, 1, 1, 64, 20, 1.0,
                _condition(observation_dt=1.0, dx=1.0 / 64.0, dy=1.0 / 64.0, viscosity=1.0e-5)),
    DatasetSpec(1, "ns2d_fno_1e-3", ProgramId.FNO_VORTICITY_NS,
                "fno_vorticity", "data/large/ns2d_1e-3_train.hdf5", "packed_hdf5",
                1000, 4800, 1, 1, 1, 64, 50, 1.0,
                _condition(observation_dt=1.0, dx=1.0 / 64.0, dy=1.0 / 64.0, viscosity=1.0e-3)),
    DatasetSpec(2, "ns2d_pdb_M1e-1_eta1e-2_zeta1e-2", ProgramId.CNS_COMPRESSIBLE,
                "cns4", "data/large/pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/train", "scatter_hdf5",
                9000, 9000, 1, 4, 4, 128, 21, 0.05,
                _condition(observation_dt=0.05, dx=1.0 / 128.0, dy=1.0 / 128.0,
                           combined_dissipation=1.0e-2, mach_regime_tag=1.0e-1)),
    DatasetSpec(3, "swe_pdb", ProgramId.SWE_HEIGHT,
                "swe_height", "data/large/pdebench/swe_pdb/train", "scatter_hdf5",
                900, 900, 1, 1, 1, 128, 101, 0.01,
                _condition(observation_dt=0.01, dx=5.0 / 128.0, dy=5.0 / 128.0, gravity=1.0)),
    DatasetSpec(4, "dr_pdb", ProgramId.DR_REACTION_DIFFUSION,
                "reaction_diffusion", "data/large/pdebench/dr_pdb/train", "scatter_hdf5",
                900, 900, 1, 2, 2, 128, 101, 0.05,
                _condition(observation_dt=0.05, dx=2.0 / 128.0, dy=2.0 / 128.0)),
    DatasetSpec(5, "cfdbench", None,
                "cfdb", "data/large/cfdbench/ns2d_cdb_train.hdf5", "packed_hdf5",
                9000, 9017, 1, 3, 2, 64, 20, 0.0, (),
                "data/metadata/cfdb_train.json"),
)


def validate_manifest(entries: tuple[DatasetSpec, ...] = BASE6) -> None:
    if len(entries) != 6 or tuple(item.index for item in entries) != tuple(range(6)):
        raise ValueError("Base6 must contain exactly six ordered entries")
    if len({item.key for item in entries}) != 6:
        raise ValueError("dataset keys must be unique")
    if [item.ntrain for item in entries] != [1000, 1000, 9000, 900, 900, 9000]:
        raise ValueError("MoE-POT Tiny Base6 ntrain contract drifted")
    if [item.data_weight for item in entries] != [1, 1, 1, 1, 1, 1]:
        raise ValueError("MoE-POT Tiny Base6 unit-weight contract drifted")
    if any(item.ntrain > item.available_train_rows for item in entries):
        raise ValueError("requested rows exceed audited source rows")
    cfdb = entries[-1]
    if cfdb.key != "cfdbench" or cfdb.program is not None or not cfdb.row_program_sidecar:
        raise ValueError("CFDB storage entry must resolve its program from a sidecar")
    represented = {item.program for item in entries if item.program is not None}
    represented.update(
        {ProgramId.CFDB_CAVITY, ProgramId.CFDB_CYLINDER, ProgramId.CFDB_TUBE}
    )
    if represented != set(ProgramId):
        raise ValueError("all seven physical programs must be represented")


def manifest_payload(entries: tuple[DatasetSpec, ...] = BASE6) -> dict[str, object]:
    validate_manifest(entries)
    payload = {
        "schema": "typed_sparse_evolution.base6_manifest.v1",
        "history": HISTORY,
        "target_steps": TARGET_STEPS,
        "model_resolution": MODEL_RESOLUTION,
        "condition_fields": list(CONDITION_FIELDS),
        "datasets": [
            {**asdict(item), "program": item.program.value if item.program else None}
            for item in entries
        ],
        "programs": {
            key.value: {**asdict(value), "identifier": value.identifier.value}
            for key, value in PROGRAMS.items()
        },
        "weighted_epoch_rows": sum(item.weighted_rows for item in entries),
        "official_test_accessed": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def write_manifest(path: str | Path) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(manifest_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


validate_manifest()


__all__ = [
    "BASE6",
    "CONDITION_FIELDS",
    "DatasetSpec",
    "HISTORY",
    "MODEL_RESOLUTION",
    "PROGRAMS",
    "ProgramId",
    "ProgramSpec",
    "TARGET_STEPS",
    "manifest_payload",
    "validate_manifest",
    "write_manifest",
]
