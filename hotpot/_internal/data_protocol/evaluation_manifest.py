from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

from .manifest import BASE6


@dataclass(frozen=True)
class EvaluationSpec:
    dataset_index: int
    key: str
    test_path: str
    storage: str
    samples: int
    native_size: int
    total_frames: int
    channels: int
    scored_channels: int
    rollout_steps: int

    @property
    def report_horizons(self) -> tuple[int, ...]:
        values = (1, min(5, self.rollout_steps), min(10, self.rollout_steps), self.rollout_steps)
        return tuple(dict.fromkeys(values))


EVALUATION: tuple[EvaluationSpec, ...] = (
    EvaluationSpec(0, "ns2d_fno_1e-5", "data/large/ns2d_1e-5_test.hdf5",
                   "packed_hdf5", 200, 64, 20, 1, 1, 10),
    EvaluationSpec(1, "ns2d_fno_1e-3", "data/large/ns2d_1e-3_test.hdf5",
                   "packed_hdf5", 200, 64, 50, 1, 1, 20),
    EvaluationSpec(2, "ns2d_pdb_M1e-1_eta1e-2_zeta1e-2",
                   "data/large/pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/test",
                   "scatter_hdf5", 200, 128, 21, 4, 4, 11),
    EvaluationSpec(3, "swe_pdb", "data/large/pdebench/swe_pdb/test",
                   "scatter_hdf5", 60, 128, 101, 1, 1, 91),
    EvaluationSpec(4, "dr_pdb", "data/large/pdebench/dr_pdb/test",
                   "scatter_hdf5", 60, 128, 101, 2, 2, 91),
    EvaluationSpec(5, "cfdbench", "data/large/cfdbench/ns2d_cdb_test.hdf5",
                   "packed_hdf5", 1000, 64, 20, 3, 2, 10),
)


def validate_evaluation_manifest(
    rows: tuple[EvaluationSpec, ...] = EVALUATION,
) -> None:
    if len(rows) != len(BASE6) or tuple(row.dataset_index for row in rows) != tuple(range(6)):
        raise ValueError("evaluation manifest must match the six ordered training entries")
    for row, train in zip(rows, BASE6, strict=True):
        if row.key != train.key or row.storage != train.storage:
            raise ValueError(f"evaluation/train identity drift for {train.key}")
        if row.native_size != train.native_size or row.channels != train.channels:
            raise ValueError(f"evaluation chart drift for {train.key}")
        if row.scored_channels != train.predicted_channels:
            raise ValueError(f"evaluation channel contract drift for {train.key}")
        if row.total_frames < 10 + row.rollout_steps:
            raise ValueError(f"evaluation horizon drift for {train.key}")
        if "test" not in row.test_path.lower() or "train" in row.test_path.lower():
            raise ValueError(f"evaluation path is not test-only for {train.key}")


def evaluation_protocol_payload() -> dict[str, object]:
    validate_evaluation_manifest()
    payload: dict[str, object] = {
        "schema": "typed_sparse_evolution.base6_fixed_e1000_rollout.v1",
        "checkpoint_rule": "fixed_epoch_1000_only_for_both_arms",
        "checkpoint_selection_from_rollout": False,
        "own_feedback": True,
        "history_frames": 10,
        "metric": "released_original_grid_masked_per_channel_SimpleLpLoss",
        "paired_sample_population": True,
        "datasets": [
            {**asdict(row), "report_horizons": list(row.report_horizons)}
            for row in EVALUATION
        ],
        "cfdb_test_program_binding": "separately_certified_OFFICIAL_TEST_ONLY_sidecar",
        "official_test_accessed": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


EVALUATION_BY_INDEX: Mapping[int, EvaluationSpec] = {
    row.dataset_index: row for row in EVALUATION
}


__all__ = [
    "EVALUATION",
    "EVALUATION_BY_INDEX",
    "EvaluationSpec",
    "evaluation_protocol_payload",
    "validate_evaluation_manifest",
]
