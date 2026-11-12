from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
    ProgramId,
)
from hotpot._internal.primitives.executor import normalize_programs
from hotpot._internal.numerics.toolbox import CanonicalNumericalToolbox

from .action_decoder import ActionContextDecoder
from .analyzer import SharedScalarFieldAnalyzer
from .config import FoundationConfig
from .private_mixer import CFDBTypedFieldMixer, PrivateTypedFieldMixer
from .program_graphs import (
    FAMILY_BY_PROGRAM,
    program_graphs_for_revision,
    registry_payload,
)
from .programs import (
    CFDBMaskedVelocityProgram,
    CFDBNormalizedVelocityProgram,
    CFDBProgram,
    CNSProgram,
    DRDirectObservationProgram,
    DRKnownSourceResidualProgram,
    DRProgram,
    FNOInvariantTransportProgram,
    FNOProgram,
    SWECalibratedSavedStepProgram,
    SWEDirectObservationProgram,
    SWEProgram,
)
from .semantic_chart import SEMANTIC_FEATURES, SemanticFieldChart, SemanticHistory


FAMILY_FIELDS = {"fno": 1, "dr": 2, "swe": 1, "cns": 4, "cfdb": 2}
LEGACY_INTERFACE_REVISION = "legacy_v1"
PILOT_INTERFACE_REVISION = "identifiable_swe_dr_v2"
FINAL_INTERFACE_REVISION = "known_source_dr_v2"
FNO_INVARIANT_INTERFACE_REVISION = "fno_invariant_transport_v3"
SWE_OBSERVATION_INTERFACE_REVISION = "swe_direct_observation_v3"
NATIVE_ACTION_INTERFACE_REVISION = "native_action_experts_v4"
FINITE_TIME_UPDATE_INTERFACE_REVISION = "finite_time_update_v5"
UNIFIED_TYPED_INTERFACE_REVISION = "unified_typed_finite_step_v6"
UNIFIED_TYPED_CLOSED_INTERFACE_REVISION = "unified_typed_closed_v7"
EVIDENCE_SELECTED_INTERFACE_REVISION = "evidence_selected_typed_program_v8"
INTERFACE_REVISIONS = (
    LEGACY_INTERFACE_REVISION,
    PILOT_INTERFACE_REVISION,
    FINAL_INTERFACE_REVISION,
    FNO_INVARIANT_INTERFACE_REVISION,
    SWE_OBSERVATION_INTERFACE_REVISION,
    NATIVE_ACTION_INTERFACE_REVISION,
    FINITE_TIME_UPDATE_INTERFACE_REVISION,
    UNIFIED_TYPED_INTERFACE_REVISION,
    UNIFIED_TYPED_CLOSED_INTERFACE_REVISION,
    EVIDENCE_SELECTED_INTERFACE_REVISION,
)


@dataclass(frozen=True)
class ModelOutput:
    prediction: Tensor
    audit: dict[str, Any]


class SemanticFieldProgramFoundation(nn.Module):
    """One shared field analyzer; no trainable private H10 re-encoder."""

    def __init__(
        self,
        config: FoundationConfig | None = None,
        *,
        interface_revision: str = LEGACY_INTERFACE_REVISION,
    ) -> None:
        super().__init__()
        if interface_revision not in INTERFACE_REVISIONS:
            raise ValueError(f"unknown typed interface revision {interface_revision!r}")
        self.interface_revision = interface_revision
        self.config = FoundationConfig() if config is None else config
        self.config.validate()
        self.charts = nn.ModuleDict(
            {
                "fno": SemanticFieldChart(self.config, dynamic_channels=1, geometry_channel=None, boundary="periodic", native_size=64),
                "dr": SemanticFieldChart(self.config, dynamic_channels=2, geometry_channel=None, boundary="closed", native_size=128),
                "swe": SemanticFieldChart(self.config, dynamic_channels=1, geometry_channel=None, boundary="closed", native_size=128),
                "cns": SemanticFieldChart(self.config, dynamic_channels=4, geometry_channel=None, boundary="periodic", native_size=128),
                "cfdb": SemanticFieldChart(self.config, dynamic_channels=2, geometry_channel=2, boundary="closed", native_size=64),
            }
        )
        if any(tuple(chart.parameters()) for chart in self.charts.values()):
            raise RuntimeError("semantic charts must remain parameter-free")
        self.shared_analyzer = SharedScalarFieldAnalyzer(self.config)
        self.private_mixers = nn.ModuleDict(
            {
                "fno": PrivateTypedFieldMixer(self.config, dynamic_channels=1),
                "dr": PrivateTypedFieldMixer(self.config, dynamic_channels=2),
                "swe": PrivateTypedFieldMixer(self.config, dynamic_channels=1),
                "cns": PrivateTypedFieldMixer(self.config, dynamic_channels=4),
                "cfdb": CFDBTypedFieldMixer(self.config),
            }
        )
        self.action_context_decoders = nn.ModuleDict(
            {family: ActionContextDecoder(self.config) for family in FAMILY_FIELDS}
        )
        self.toolbox = CanonicalNumericalToolbox()
        if interface_revision in (
            FINITE_TIME_UPDATE_INTERFACE_REVISION,
            UNIFIED_TYPED_INTERFACE_REVISION,
            UNIFIED_TYPED_CLOSED_INTERFACE_REVISION,
            EVIDENCE_SELECTED_INTERFACE_REVISION,
        ):
            dr_program = DRDirectObservationProgram(self.toolbox)
        elif interface_revision in (
            LEGACY_INTERFACE_REVISION,
            FNO_INVARIANT_INTERFACE_REVISION,
            SWE_OBSERVATION_INTERFACE_REVISION,
        ):
            dr_program = DRProgram(self.toolbox)
        else:
            dr_program = DRKnownSourceResidualProgram(self.toolbox)
        # The SWE gain interface was a controlled pilot and failed the held-out
        # H91 gate.  The selected final revision therefore keeps the proven
        
        swe_program = (
            SWECalibratedSavedStepProgram(self.toolbox)
            if interface_revision == PILOT_INTERFACE_REVISION
            else (
                SWEDirectObservationProgram(self.toolbox)
                if interface_revision
                in (
                    SWE_OBSERVATION_INTERFACE_REVISION,
                    FINITE_TIME_UPDATE_INTERFACE_REVISION,
                    UNIFIED_TYPED_INTERFACE_REVISION,
                    UNIFIED_TYPED_CLOSED_INTERFACE_REVISION,
                    EVIDENCE_SELECTED_INTERFACE_REVISION,
                )
                else SWEProgram(self.toolbox)
            )
        )
        fno_program = (
            FNOInvariantTransportProgram(self.toolbox)
            if interface_revision == FNO_INVARIANT_INTERFACE_REVISION
            else FNOProgram(self.toolbox)
        )
        self.programs = nn.ModuleDict(
            {
                "fno": fno_program,
                "dr": dr_program,
                "swe": swe_program,
                "cns": CNSProgram(self.toolbox),
                "cfdb": (
                    CFDBNormalizedVelocityProgram(self.toolbox)
                    if interface_revision == UNIFIED_TYPED_CLOSED_INTERFACE_REVISION
                    else (
                        CFDBMaskedVelocityProgram(self.toolbox)
                        if interface_revision == UNIFIED_TYPED_INTERFACE_REVISION
                        else CFDBProgram(self.toolbox)
                    )
                ),
            }
        )

    def reset_audit_state(self) -> None:
        self.shared_analyzer.reset_audit()
        self.toolbox.reset_audit()
        for program in self.programs.values():
            program.head.reset_calls()

    @staticmethod
    def _groups(programs: Sequence[ProgramId]) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = {}
        for position, program in enumerate(programs):
            grouped.setdefault(FAMILY_BY_PROGRAM[program], []).append(position)
        return grouped

    def _semantic_batch(
        self,
        history: Tensor,
        grouped: dict[str, list[int]],
    ) -> tuple[SemanticHistory, dict[str, Tensor]]:
        batch = history.shape[0]
        features = history.new_zeros(
            batch,
            self.config.max_fields,
            self.config.history_steps,
            self.config.semantic_channels,
            self.config.resolution,
            self.config.resolution,
        )
        field_mask = history.new_zeros(batch, self.config.max_fields)
        field_mean = torch.zeros_like(field_mask)
        field_log_scale = torch.zeros_like(field_mask)
        relative_lags = history.new_zeros(batch, self.config.history_steps)
        indices: dict[str, Tensor] = {}
        for family, positions in grouped.items():
            index = torch.tensor(positions, dtype=torch.long, device=history.device)
            indices[family] = index
            value = self.charts[family](history.index_select(0, index))
            features = features.index_copy(0, index, value.features)
            field_mask = field_mask.index_copy(0, index, value.field_mask)
            field_mean = field_mean.index_copy(0, index, value.field_mean)
            field_log_scale = field_log_scale.index_copy(0, index, value.field_log_scale)
            relative_lags = relative_lags.index_copy(0, index, value.relative_lags)
        return (
            SemanticHistory(features, field_mask, field_mean, field_log_scale, relative_lags),
            indices,
        )

    def forward(
        self,
        history: Tensor,
        programs: ProgramId | str | Sequence[ProgramId | str],
        conditions: Tensor,
        condition_mask: Tensor | None = None,
        *,
        return_audit: bool = False,
    ) -> Tensor | ModelOutput:
        expected = (
            self.config.resolution,
            self.config.resolution,
            self.config.history_steps,
            4,
        )
        if history.ndim != 5 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"history must be [B,{expected}]")
        batch = history.shape[0]
        if conditions.shape != (batch, len(CONDITION_FIELDS)):
            raise ValueError("condition tensor ABI mismatch")
        if condition_mask is None:
            condition_mask = torch.ones_like(conditions)
        if condition_mask.shape != conditions.shape:
            raise ValueError("condition mask must match conditions")
        normalized, codes = normalize_programs(programs, batch=batch, device=history.device)
        grouped = self._groups(normalized)
        program_graphs = program_graphs_for_revision(self.interface_revision)
        self.reset_audit_state()

        semantic, indices = self._semantic_batch(history, grouped)
        # Exactly one all-PDE call.  Every learned use of frames t-9...t-1
        # must pass through this analyzer.
        field_memory = self.shared_analyzer(
            semantic.features, semantic.field_mask, semantic.relative_lags
        )
        decoded_by_family: dict[str, Tensor] = {}
        for family, index in indices.items():
            mixer_inputs = (
                field_memory.index_select(0, index),
                semantic.field_mask.index_select(0, index),
                semantic.field_mean.index_select(0, index),
                semantic.field_log_scale.index_select(0, index),
                conditions.index_select(0, index),
                condition_mask.index_select(0, index),
            )
            if family == "cfdb":
                private_latent = self.private_mixers[family](
                    *mixer_inputs, codes.index_select(0, index)
                )
            else:
                private_latent = self.private_mixers[family](*mixer_inputs)
            decoded_by_family[family] = self.action_context_decoders[family](private_latent)

        prediction = history.new_ones(batch, self.config.resolution, self.config.resolution, 1, 4)
        for family, index in indices.items():
            group_history = history.index_select(0, index)
            group_conditions = conditions.index_select(0, index)
            group_mask = condition_mask.index_select(0, index)
            group_output = prediction.index_select(0, index)
            decoded = decoded_by_family[family]
            if family == "fno":
                native = self.programs[family](group_history, decoded, group_conditions, group_mask)
                public = F.interpolate(native, size=(128, 128), mode="bilinear", align_corners=False)[:, 0]
                group_output[..., 0, 0] = public
            elif family == "dr":
                group_output[..., 0, :2] = self.programs[family](group_history, decoded, group_conditions, group_mask)
            elif family == "swe":
                group_output[..., 0, 0] = self.programs[family](group_history, decoded, group_conditions, group_mask)
            elif family == "cns":
                group_output[..., 0, :4] = self.programs[family](group_history, decoded, group_conditions, group_mask)
            elif family == "cfdb":
                group_output[..., 0, :4] = self.programs[family](
                    group_history,
                    decoded,
                    codes.index_select(0, index),
                    group_conditions,
                    group_mask,
                )
            else:
                raise RuntimeError(f"unhandled family {family}")
            prediction = prediction.index_copy(0, index, group_output)

        if self.shared_analyzer.call_count != 1:
            raise RuntimeError("all-PDE shared analyzer must execute exactly once")
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("prediction contains non-finite values")
        if not return_audit:
            return prediction
        return ModelOutput(
            prediction=prediction,
            audit={
                "schema": "semantic_field_program_foundation.forward_audit.v1",
                "interface_revision": self.interface_revision,
                "programs": [value.value for value in normalized],
                "families": sorted(grouped),
                "shared_analyzer_calls": self.shared_analyzer.call_count,
                "semantic_feature_names": list(SEMANTIC_FEATURES),
                "shared_weights_are_fieldwise": True,
                "private_full_history_encoder": False,
                "private_mixer_calls": {family: 1 for family in sorted(grouped)},
                "action_context_decoder_calls": {family: 1 for family in sorted(grouped)},
                "typed_program_nodes": {
                    program.value: [node.identifier for node in program_graphs[program]]
                    for program in sorted(set(normalized), key=lambda item: item.value)
                },
                "closure_calls": {
                    family: int(program.head.call_count)
                    for family, program in self.programs.items()
                },
                "cfdb_family_route_counts": dict(
                    self.programs["cfdb"].head.last_route_counts
                ),
                "toolbox": self.toolbox.audit_payload(),
                "auxiliary_history_objective": False,
                "official_test_accessed": False,
                "scientific_rollout_evidence": False,
            },
        )

    def operator_registry(self) -> dict[str, object]:
        return registry_payload(self.interface_revision)


__all__ = [
    "EVIDENCE_SELECTED_INTERFACE_REVISION",
    "FINAL_INTERFACE_REVISION",
    "FINITE_TIME_UPDATE_INTERFACE_REVISION",
    "FNO_INVARIANT_INTERFACE_REVISION",
    "FAMILY_FIELDS",
    "INTERFACE_REVISIONS",
    "LEGACY_INTERFACE_REVISION",
    "NATIVE_ACTION_INTERFACE_REVISION",
    "ModelOutput",
    "PILOT_INTERFACE_REVISION",
    "SWE_OBSERVATION_INTERFACE_REVISION",
    "UNIFIED_TYPED_INTERFACE_REVISION",
    "UNIFIED_TYPED_CLOSED_INTERFACE_REVISION",
    "SemanticFieldProgramFoundation",
]
