from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from hotpot._internal.data_protocol.manifest import ProgramId
from hotpot._internal.encoder.program_graphs import (
    ACTION_SCOPE_BY_PROGRAM,
    FAMILY_BY_PROGRAM,
)

if TYPE_CHECKING:
    from .models import V13NoSharingControl


def no_sharing_parameter_ledger(model: "V13NoSharingControl") -> dict[str, object]:
    named = dict(model.named_parameters())
    if len({id(value) for value in named.values()}) != len(named):
        raise RuntimeError("control parameter identity aliasing is not allowed")
    scopes: dict[str, list[str]] = defaultdict(list)
    for name in named:
        if name.startswith("family_analyzers."):
            family = name.split(".", 2)[1]
            scope = "cfdb_shared" if family == "cfdb" else family
            scopes[f"private::{scope}::observation_analyzer"].append(name)
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
            raise RuntimeError(f"unclassified control parameter: {name}")
    counted = {name for values in scopes.values() for name in values}
    if counted != set(named):
        raise RuntimeError("control parameter ledger is not closed")
    counts = {
        scope: sum(named[name].numel() for name in names)
        for scope, names in sorted(scopes.items())
    }
    active: dict[str, int] = {}
    for program in ProgramId:
        family = FAMILY_BY_PROGRAM[program]
        if family == "cfdb":
            action_scope = ACTION_SCOPE_BY_PROGRAM[program]
            prefixes = ("private::cfdb_shared::", f"private::{action_scope}::")
        else:
            prefixes = (f"private::{family}::",)
        active[program.value] = sum(
            value for scope, value in counts.items() if scope.startswith(prefixes)
        )
    total = sum(parameter.numel() for parameter in model.parameters())
    if total != sum(counts.values()):
        raise RuntimeError("control stored parameter total does not close")
    return {
        "schema": "semantic_field_program_foundation.parameter_ledger.v1.3-no-sharing",
        "control": "no_cross_pde_sharing",
        "stored_total": total,
        "all_pde_shared_inference_parameters": 0,
        "within_family_observation_analyzer_parameters": {
            family: sum(
                value
                for scope, value in counts.items()
                if scope.startswith(
                    f"private::{('cfdb_shared' if family == 'cfdb' else family)}::observation_analyzer"
                )
            )
            for family in ("fno", "dr", "swe", "cns", "cfdb")
        },
        "shared_training_only_parameters": 0,
        "scope_counts": counts,
        "active_by_program": active,
        "active_min": min(active.values()),
        "active_max": max(active.values()),
        "inactive_stored_parameters_are_other_typed_programs": True,
        "private_full_history_encoder_parameters": 0,
        "unclassified_parameters": 0,
    }


__all__ = ["no_sharing_parameter_ledger"]
