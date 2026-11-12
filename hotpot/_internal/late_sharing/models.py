from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
    ProgramId,
)
from hotpot._internal.primitives.executor import (
    normalize_programs,
)
from hotpot._internal.encoder.analyzer import (
    AnalysisBlock,
)
from hotpot._internal.encoder.model import (
    EVIDENCE_SELECTED_INTERFACE_REVISION,
    ModelOutput,
)
from hotpot._internal.encoder.program_graphs import (
    ACTION_SCOPE_BY_PROGRAM,
    FAMILY_BY_PROGRAM,
    program_graphs_for_revision,
)
from hotpot._internal.encoder.semantic_chart import (
    SEMANTIC_FEATURES,
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


def _resolved(config: FoundationScaleConfig | None) -> FoundationScaleConfig:
    return tiny_17m_config() if config is None else config


class FixedSemanticHistoryMap(nn.Module):
    """Task-independent H10-to-latent map with no learned parameters."""

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        config.validate()
        packed_channels = config.history_steps * config.semantic_channels
        if packed_channels > config.shared_width:
            raise ValueError(
                "fixed map requires shared_width >= history_steps * semantic_channels"
            )
        self.config = config
        self.packed_channels = int(packed_channels)
        self.call_count = 0

    def reset_audit(self) -> None:
        self.call_count = 0

    def forward(
        self, features: Tensor, field_mask: Tensor, relative_lags: Tensor
    ) -> Tensor:
        self.call_count += 1
        if features.ndim != 6:
            raise ValueError("semantic features must be [B,F,T,K,H,W]")
        batch, fields, steps, channels, height, width = features.shape
        expected = (
            self.config.max_fields,
            self.config.history_steps,
            self.config.semantic_channels,
            self.config.resolution,
            self.config.resolution,
        )
        if (fields, steps, channels, height, width) != expected:
            raise ValueError(f"semantic feature ABI requires [B,{expected}]")
        if field_mask.shape != (batch, fields):
            raise ValueError("field mask must be [B,F]")
        if relative_lags.shape != (batch, steps):
            raise ValueError("relative lags must be [B,T]")
        packed = features.reshape(batch * fields * steps, channels, height, width)
        pooled = F.avg_pool2d(
            packed,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        ).reshape(
            batch,
            fields,
            steps * channels,
            self.config.latent_size,
            self.config.latent_size,
        )
        padding = self.config.shared_width - self.packed_channels
        if padding:
            pooled = torch.cat(
                (
                    pooled,
                    pooled.new_zeros(
                        batch,
                        fields,
                        padding,
                        self.config.latent_size,
                        self.config.latent_size,
                    ),
                ),
                dim=2,
            )
        return pooled * field_mask[:, :, None, None, None]


class SharedLateEvolutionBlock(nn.Module):
    """One all-PDE latent analysis block after private field coupling."""

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        if config.private_width != 512:
            raise ValueError("registered late-sharing control requires private_width=512")
        self.down = nn.Conv2d(config.private_width, 448, 1)
        self.block = AnalysisBlock(
            448,
            num_blocks=config.afno_blocks,
            modes=config.afno_modes,
            ffn_hidden=1273,
        )
        self.up = nn.Conv2d(448, config.private_width, 1)
        self.call_count = 0

    def reset_audit(self) -> None:
        self.call_count = 0

    def forward(self, value: Tensor) -> Tensor:
        self.call_count += 1
        if value.ndim != 4 or value.shape[1] != 512:
            raise ValueError("late shared latent ABI must be [B,512,H,W]")
        return value + self.up(self.block(self.down(value)))


class V13LateSharedEvolutionControl(SemanticFieldProgramFoundationS90M):
    """Move the matched learned shared budget from history to evolution."""

    def __init__(self, config: FoundationScaleConfig | None = None) -> None:
        resolved = _resolved(config)
        super().__init__(
            resolved, interface_revision=EVIDENCE_SELECTED_INTERFACE_REVISION
        )
        self.shared_analyzer = FixedSemanticHistoryMap(resolved)
        self.late_shared = SharedLateEvolutionBlock(resolved)

    def reset_audit_state(self) -> None:
        super().reset_audit_state()
        self.late_shared.reset_audit()

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
        normalized, codes = normalize_programs(
            programs, batch=batch, device=history.device
        )
        grouped = self._groups(normalized)
        program_graphs = program_graphs_for_revision(self.interface_revision)
        self.reset_audit_state()

        semantic, indices = self._semantic_batch(history, grouped)
        field_memory = self.shared_analyzer(
            semantic.features, semantic.field_mask, semantic.relative_lags
        )
        private_batch = history.new_zeros(
            batch,
            self.config.private_width,
            self.config.latent_size,
            self.config.latent_size,
        )
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
            private_batch = private_batch.index_copy(0, index, private_latent)

        shared_late = self.late_shared(private_batch)
        decoded_by_family = {
            family: self.action_context_decoders[family](
                shared_late.index_select(0, index)
            )
            for family, index in indices.items()
        }

        prediction = history.new_ones(
            batch, self.config.resolution, self.config.resolution, 1, 4
        )
        for family, index in indices.items():
            group_history = history.index_select(0, index)
            group_conditions = conditions.index_select(0, index)
            group_mask = condition_mask.index_select(0, index)
            group_output = prediction.index_select(0, index)
            decoded = decoded_by_family[family]
            if family == "fno":
                native = self.programs[family](
                    group_history, decoded, group_conditions, group_mask
                )
                public = F.interpolate(
                    native,
                    size=(128, 128),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                group_output[..., 0, 0] = public
            elif family == "dr":
                group_output[..., 0, :2] = self.programs[family](
                    group_history, decoded, group_conditions, group_mask
                )
            elif family == "swe":
                group_output[..., 0, 0] = self.programs[family](
                    group_history, decoded, group_conditions, group_mask
                )
            elif family == "cns":
                group_output[..., 0, :4] = self.programs[family](
                    group_history, decoded, group_conditions, group_mask
                )
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

        if self.shared_analyzer.call_count != 1 or self.late_shared.call_count != 1:
            raise RuntimeError("both deterministic history and late sharing must run once")
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("prediction contains non-finite values")
        if not return_audit:
            return prediction
        return ModelOutput(
            prediction=prediction,
            audit={
                "schema": "semantic_field_program_foundation.forward_audit.v1.3-late-shared",
                "interface_revision": self.interface_revision,
                "programs": [value.value for value in normalized],
                "families": sorted(grouped),
                "shared_analyzer_calls": self.shared_analyzer.call_count,
                "late_shared_calls": self.late_shared.call_count,
                "semantic_feature_names": list(SEMANTIC_FEATURES),
                "shared_weights_are_fieldwise": False,
                "shared_boundary": "post_cross_field_pre_action",
                "private_full_history_encoder": False,
                "private_mixer_calls": {family: 1 for family in sorted(grouped)},
                "action_context_decoder_calls": {
                    family: 1 for family in sorted(grouped)
                },
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


def parameter_ledger(model: V13LateSharedEvolutionControl) -> dict[str, object]:
    named = dict(model.named_parameters())
    if len({id(value) for value in named.values()}) != len(named):
        raise RuntimeError("parameter identity aliasing needs an explicit ledger rule")
    scopes: dict[str, list[str]] = defaultdict(list)
    for name in named:
        if name.startswith("late_shared."):
            scopes["all_pde_shared_late_evolution"].append(name)
        elif name.startswith("private_mixers."):
            family = name.split(".", 2)[1]
            if family == "cfdb" and ".family_ffns." in name:
                cfdb_family = name.split(".family_ffns.", 1)[1].split(".", 1)[0]
                scopes[f"private::cfdb_{cfdb_family}::field_mixer_ffn"].append(name)
            elif family == "cfdb":
                scopes["private::cfdb_shared::field_mixer"].append(name)
            else:
                scopes[f"private::{family}::field_mixer"].append(name)
        elif name.startswith("action_context_decoders."):
            family = name.split(".", 2)[1]
            scope = "cfdb_shared" if family == "cfdb" else family
            scopes[f"private::{scope}::action_context_decoder"].append(name)
        elif name.startswith("programs."):
            family = name.split(".", 2)[1]
            if family == "cfdb" and ".family_heads." in name:
                cfdb_family = name.split(".family_heads.", 1)[1].split(".", 1)[0]
                scopes[f"private::cfdb_{cfdb_family}::typed_action"].append(name)
            else:
                scopes[f"private::{family}::typed_action"].append(name)
        elif name.startswith("charts."):
            raise RuntimeError("semantic chart unexpectedly acquired parameters")
        elif name.startswith("shared_analyzer."):
            raise RuntimeError("fixed early history map unexpectedly acquired parameters")
        elif name.startswith("toolbox."):
            scopes["parameter_free_toolbox"].append(name)
        else:
            raise RuntimeError(f"unclassified parameter: {name}")
    counts = {
        scope: sum(named[name].numel() for name in names)
        for scope, names in sorted(scopes.items())
    }
    shared = counts.get("all_pde_shared_late_evolution", 0)
    active: dict[str, int] = {}
    for program in ProgramId:
        family = FAMILY_BY_PROGRAM[program]
        if family == "cfdb":
            action_scope = ACTION_SCOPE_BY_PROGRAM[program]
            prefixes = ("private::cfdb_shared::", f"private::{action_scope}::")
        else:
            prefixes = (f"private::{family}::",)
        private = sum(
            value for scope, value in counts.items() if scope.startswith(prefixes)
        )
        active[program.value] = shared + private
    total = sum(parameter.numel() for parameter in model.parameters())
    if total != sum(counts.values()):
        raise RuntimeError("late-sharing ledger does not close")
    return {
        "schema": "semantic_field_program_foundation.parameter_ledger.v1.3-late-shared",
        "control": "late_shared_evolution",
        "stored_total": total,
        "all_pde_shared_inference_parameters": shared,
        "all_pde_shared_late_evolution_parameters": shared,
        "shared_training_only_parameters": 0,
        "scope_counts": counts,
        "active_by_program": active,
        "active_min": min(active.values()),
        "active_max": max(active.values()),
        "inactive_stored_parameters_are_other_typed_programs": True,
        "private_full_history_encoder_parameters": 0,
        "unclassified_parameters": 0,
    }


class Base6V13LateSharedEvolutionControl(Base6SemanticFieldFoundationS90M):
    def __init__(
        self, workspace=None, config: FoundationScaleConfig | None = None
    ) -> None:
        nn.Module.__init__(self)
        self.workspace = workspace
        self.model = V13LateSharedEvolutionControl(_resolved(config))

    def parameter_ledger(self) -> dict[str, object]:
        return parameter_ledger(self.model)


__all__ = [
    "Base6V13LateSharedEvolutionControl",
    "FixedSemanticHistoryMap",
    "SharedLateEvolutionBlock",
    "V13LateSharedEvolutionControl",
    "parameter_ledger",
]
