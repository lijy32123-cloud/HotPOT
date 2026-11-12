from __future__ import annotations

from collections import defaultdict

from hotpot._internal.data_protocol.manifest import ProgramId

from .model import SemanticFieldProgramFoundation
from .program_graphs import ACTION_SCOPE_BY_PROGRAM, FAMILY_BY_PROGRAM


def parameter_ledger(model: SemanticFieldProgramFoundation) -> dict[str, object]:
    named = dict(model.named_parameters())
    if len({id(value) for value in named.values()}) != len(named):
        raise RuntimeError("parameter identity aliasing requires an explicit ledger rule")
    scopes: dict[str, list[str]] = defaultdict(list)
    for name in named:
        if name.startswith("shared_analyzer."):
            scopes["all_pde_shared_scalar_field_analyzer"].append(name)
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
        elif name.startswith("toolbox."):
            scopes["parameter_free_toolbox"].append(name)
        else:
            raise RuntimeError(f"unclassified parameter: {name}")
    counted = {name for values in scopes.values() for name in values}
    if counted != set(named):
        raise RuntimeError("parameter ledger is not closed")
    counts = {
        scope: sum(named[name].numel() for name in names)
        for scope, names in sorted(scopes.items())
    }
    shared = counts.get("all_pde_shared_scalar_field_analyzer", 0)
    active: dict[str, int] = {}
    for program in ProgramId:
        family = FAMILY_BY_PROGRAM[program]
        if family == "cfdb":
            action_scope = ACTION_SCOPE_BY_PROGRAM[program]
            prefixes = ("private::cfdb_shared::", f"private::{action_scope}::")
        else:
            prefixes = (f"private::{family}::",)
        private = sum(
            value
            for scope, value in counts.items()
            if scope.startswith(prefixes)
        )
        active[program.value] = shared + private
    total = sum(parameter.numel() for parameter in model.parameters())
    if total != sum(counts.values()):
        raise RuntimeError("stored parameter total does not close")
    return {
        "schema": "semantic_field_program_foundation.parameter_ledger.v1.1",
        "stored_total": total,
        "all_pde_shared_inference_parameters": shared,
        "shared_training_only_parameters": 0,
        "scope_counts": counts,
        "active_by_program": active,
        "active_min": min(active.values()),
        "active_max": max(active.values()),
        "inactive_stored_parameters_are_other_typed_programs": True,
        "private_full_history_encoder_parameters": 0,
        "unclassified_parameters": 0,
    }


def build_and_count(config=None) -> dict[str, object]:
    return parameter_ledger(SemanticFieldProgramFoundation(config))


__all__ = ["build_and_count", "parameter_ledger"]
