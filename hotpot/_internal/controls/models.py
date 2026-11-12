from __future__ import annotations

import copy
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
    ProgramId,
)
from hotpot._internal.data_protocol.native_charts import (
    fno_native64_exact_left_inverse,
)
from hotpot._internal.primitives.executor import normalize_programs
from hotpot._internal.encoder.model import (
    EVIDENCE_SELECTED_INTERFACE_REVISION,
    FAMILY_FIELDS,
    ModelOutput,
)
from hotpot._internal.encoder.program_graphs import (
    program_graphs_for_revision,
)
from hotpot._internal.encoder.semantic_chart import (
    SEMANTIC_FEATURES,
    SemanticHistory,
)
from hotpot._internal.foundation.accounting import (
    parameter_ledger,
)
from hotpot._internal.foundation.base6 import (
    Base6SemanticFieldFoundationS90M,
)
from hotpot._internal.foundation.config import (
    FoundationScaleConfig,
    tiny_17m_config,
)
from hotpot._internal.foundation.model import (
    SemanticFieldProgramFoundationS90M,
)

from .accounting import no_sharing_parameter_ledger


def _resolved(config: FoundationScaleConfig | None) -> FoundationScaleConfig:
    return tiny_17m_config() if config is None else config


def _validate_and_prepare(
    model: SemanticFieldProgramFoundationS90M,
    history: Tensor,
    programs: ProgramId | str | Sequence[ProgramId | str],
    conditions: Tensor,
    condition_mask: Tensor | None,
) -> tuple[list[ProgramId], Tensor, dict[str, list[int]], SemanticHistory, dict[str, Tensor], Tensor]:
    expected = (
        model.config.resolution,
        model.config.resolution,
        model.config.history_steps,
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
    grouped = model._groups(normalized)
    semantic, indices = model._semantic_batch(history, grouped)
    return normalized, codes, grouped, semantic, indices, condition_mask


def _decode_private_latents(
    model: SemanticFieldProgramFoundationS90M,
    field_memory: Tensor,
    semantic: SemanticHistory,
    indices: dict[str, Tensor],
    codes: Tensor,
    conditions: Tensor,
    condition_mask: Tensor,
) -> dict[str, Tensor]:
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
            private_latent = model.private_mixers[family](
                *mixer_inputs, codes.index_select(0, index)
            )
        else:
            private_latent = model.private_mixers[family](*mixer_inputs)
        decoded_by_family[family] = model.action_context_decoders[family](private_latent)
    return decoded_by_family


def _typed_prediction(
    model: SemanticFieldProgramFoundationS90M,
    history: Tensor,
    indices: dict[str, Tensor],
    codes: Tensor,
    conditions: Tensor,
    condition_mask: Tensor,
    decoded_by_family: dict[str, Tensor],
) -> Tensor:
    prediction = history.new_ones(
        history.shape[0], model.config.resolution, model.config.resolution, 1, 4
    )
    for family, index in indices.items():
        group_history = history.index_select(0, index)
        group_conditions = conditions.index_select(0, index)
        group_mask = condition_mask.index_select(0, index)
        group_output = prediction.index_select(0, index)
        decoded = decoded_by_family[family]
        if family == "fno":
            native = model.programs[family](group_history, decoded, group_conditions, group_mask)
            group_output[..., 0, 0] = F.interpolate(
                native, size=(128, 128), mode="bilinear", align_corners=False
            )[:, 0]
        elif family == "dr":
            group_output[..., 0, :2] = model.programs[family](
                group_history, decoded, group_conditions, group_mask
            )
        elif family == "swe":
            group_output[..., 0, 0] = model.programs[family](
                group_history, decoded, group_conditions, group_mask
            )
        elif family == "cns":
            group_output[..., 0, :4] = model.programs[family](
                group_history, decoded, group_conditions, group_mask
            )
        elif family == "cfdb":
            group_output[..., 0, :4] = model.programs[family](
                group_history,
                decoded,
                codes.index_select(0, index),
                group_conditions,
                group_mask,
            )
        else:
            raise RuntimeError(f"unhandled family {family}")
        prediction = prediction.index_copy(0, index, group_output)
    return prediction


def _audit(
    model: SemanticFieldProgramFoundationS90M,
    normalized: Sequence[ProgramId],
    grouped: dict[str, list[int]],
) -> dict[str, Any]:
    graphs = program_graphs_for_revision(model.interface_revision)
    return {
        "schema": "semantic_field_program_foundation.control_forward_audit.v1.3",
        "interface_revision": model.interface_revision,
        "programs": [value.value for value in normalized],
        "families": sorted(grouped),
        "semantic_feature_names": list(SEMANTIC_FEATURES),
        "shared_weights_are_fieldwise": True,
        "private_full_history_encoder": False,
        "private_mixer_calls": {family: 1 for family in sorted(grouped)},
        "action_context_decoder_calls": {family: 1 for family in sorted(grouped)},
        "typed_program_nodes": {
            program.value: [node.identifier for node in graphs[program]]
            for program in sorted(set(normalized), key=lambda item: item.value)
        },
        "closure_calls": {
            family: int(program.head.call_count) for family, program in model.programs.items()
        },
        "cfdb_family_route_counts": dict(model.programs["cfdb"].head.last_route_counts),
        "toolbox": model.toolbox.audit_payload(),
        "auxiliary_history_objective": False,
        "official_test_accessed": False,
        "scientific_rollout_evidence": False,
    }


class V13NoSharingControl(SemanticFieldProgramFoundationS90M):
    """Five family analyzers; no learned parameter is tied across PDE families."""

    def __init__(self, config: FoundationScaleConfig | None = None) -> None:
        super().__init__(_resolved(config), interface_revision=EVIDENCE_SELECTED_INTERFACE_REVISION)
        template = self.shared_analyzer
        self.family_analyzers = nn.ModuleDict(
            {family: copy.deepcopy(template) for family in FAMILY_FIELDS}
        )
        del self.shared_analyzer

    def reset_audit_state(self) -> None:
        for analyzer in self.family_analyzers.values():
            analyzer.reset_audit()
        self.toolbox.reset_audit()
        for program in self.programs.values():
            program.head.reset_calls()

    def forward(self, history, programs, conditions, condition_mask=None, *, return_audit=False):
        self.reset_audit_state()
        normalized, codes, grouped, semantic, indices, condition_mask = _validate_and_prepare(
            self, history, programs, conditions, condition_mask
        )
        field_memory = history.new_zeros(
            history.shape[0],
            self.config.max_fields,
            self.config.shared_width,
            self.config.latent_size,
            self.config.latent_size,
        )
        for family, index in indices.items():
            memory = self.family_analyzers[family](
                semantic.features.index_select(0, index),
                semantic.field_mask.index_select(0, index),
                semantic.relative_lags.index_select(0, index),
            )
            field_memory = field_memory.index_copy(0, index, memory)
        decoded = _decode_private_latents(
            self, field_memory, semantic, indices, codes, conditions, condition_mask
        )
        prediction = _typed_prediction(
            self, history, indices, codes, conditions, condition_mask, decoded
        )
        expected = {family: int(family in indices) for family in self.family_analyzers}
        actual = {family: analyzer.call_count for family, analyzer in self.family_analyzers.items()}
        if actual != expected:
            raise RuntimeError("family analyzer call contract drifted")
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("prediction contains non-finite values")
        if not return_audit:
            return prediction
        payload = _audit(self, normalized, grouped)
        payload.update(
            {
                "control": "v1.3_no_cross_pde_sharing",
                "cross_pde_parameter_tying": False,
                "family_analyzer_calls": actual,
            }
        )
        return ModelOutput(prediction=prediction, audit=payload)


class V13DirectStateControl(SemanticFieldProgramFoundationS90M):
    """Same learned graph and call clocks, without deterministic programs."""

    expected_calls = {"fno": 8, "dr": 2, "swe": 1, "cns": 1, "cfdb": 1}

    def __init__(self, config: FoundationScaleConfig | None = None) -> None:
        super().__init__(_resolved(config), interface_revision=EVIDENCE_SELECTED_INTERFACE_REVISION)

    def forward(self, history, programs, conditions, condition_mask=None, *, return_audit=False):
        self.reset_audit_state()
        normalized, codes, grouped, semantic, indices, condition_mask = _validate_and_prepare(
            self, history, programs, conditions, condition_mask
        )
        field_memory = self.shared_analyzer(
            semantic.features, semantic.field_mask, semantic.relative_lags
        )
        decoded_by_family = _decode_private_latents(
            self, field_memory, semantic, indices, codes, conditions, condition_mask
        )
        prediction = history.new_ones(
            history.shape[0], self.config.resolution, self.config.resolution, 1, 4
        )
        for family, index in indices.items():
            group_history = history.index_select(0, index)
            group_conditions = conditions.index_select(0, index)
            group_mask = condition_mask.index_select(0, index)
            group_output = prediction.index_select(0, index)
            decoded = decoded_by_family[family]
            head = self.programs[family].head
            head.reset_calls()
            if family == "fno":
                scalar = fno_native64_exact_left_inverse(
                    group_history[..., 0].permute(0, 3, 1, 2).contiguous()
                )
                current = scalar[:, -1:].contiguous()
                increments = [
                    head(decoded, current, group_conditions, group_mask)
                    for _ in range(self.expected_calls[family])
                ]
                native = current + torch.stack(increments).mean(0)
                group_output[..., 0, 0] = F.interpolate(
                    native, size=(128, 128), mode="bilinear", align_corners=False
                )[:, 0]
            elif family == "dr":
                current = group_history[..., -1, :2].contiguous()
                proposals = []
                for _ in range(self.expected_calls[family]):
                    update_a, update_b = head(decoded, current, group_conditions, group_mask)
                    proposals.append(update_a + update_b)
                group_output[..., 0, :2] = current + torch.stack(proposals).mean(0)
            elif family == "swe":
                scalar_history = group_history[..., 0].permute(0, 3, 1, 2).contiguous()
                current = scalar_history[:, -1]
                group_output[..., 0, 0] = current + head(
                    decoded, scalar_history, group_conditions, group_mask
                )
            elif family == "cns":
                current = group_history[..., -1, :4].contiguous()
                residual = head(decoded, current, group_conditions, group_mask).permute(0, 2, 3, 1)
                group_output[..., 0, :4] = current + residual
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
            if head.call_count != self.expected_calls[family]:
                raise RuntimeError(f"direct-state call clock drifted for {family}")
            prediction = prediction.index_copy(0, index, group_output)
        if self.shared_analyzer.call_count != 1:
            raise RuntimeError("shared analyzer must execute exactly once")
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("prediction contains non-finite values")
        if not return_audit:
            return prediction
        payload = _audit(self, normalized, grouped)
        payload.update(
            {
                "control": "v1.3_direct_state",
                "shared_analyzer_calls": self.shared_analyzer.call_count,
                "deterministic_program_execution": {
                    family: family == "cfdb" for family in sorted(grouped)
                },
                "direct_state_transition": {
                    family: family != "cfdb" for family in sorted(grouped)
                },
                "cfdb_unchanged_and_excluded": "cfdb" in grouped,
            }
        )
        return ModelOutput(prediction=prediction, audit=payload)


class Base6V13NoSharingControl(Base6SemanticFieldFoundationS90M):
    def __init__(self, workspace=None, config: FoundationScaleConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.workspace = workspace
        self.model = V13NoSharingControl(_resolved(config))

    def parameter_ledger(self) -> dict[str, object]:
        return no_sharing_parameter_ledger(self.model)


class Base6V13DirectStateControl(Base6SemanticFieldFoundationS90M):
    def __init__(self, workspace=None, config: FoundationScaleConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.workspace = workspace
        self.model = V13DirectStateControl(_resolved(config))

    def parameter_ledger(self) -> dict[str, object]:
        return parameter_ledger(self.model)


__all__ = [
    "Base6V13DirectStateControl",
    "Base6V13NoSharingControl",
    "V13DirectStateControl",
    "V13NoSharingControl",
]
