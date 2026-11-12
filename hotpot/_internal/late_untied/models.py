import copy


from hotpot._internal.late_sharing import models as source


from .contract import FAMILIES, KIND, SCALE


def parameter_ledger(model):
    from hotpot._internal.encoder.program_graphs import FAMILY_BY_PROGRAM
    reference = copy.deepcopy(model._original_late_ledger)
    shared = reference["all_pde_shared_late_evolution_parameters"]
    counts = dict(reference["scope_counts"])
    del counts["all_pde_shared_late_evolution"]
    for family, block in model.late_shared.blocks.items():
        count = sum(p.numel() for p in block.parameters())
        if count != shared:
            raise RuntimeError("Late per-family capacity changed")
        counts[f"private::{family}::late_evolution"] = count
    named = list(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in named}) != len(named):
        raise RuntimeError("Late-Untied parameter identity aliasing")
    if len({p.data_ptr() for _, p in named}) != len(named):
        raise RuntimeError("Late-Untied parameter storage aliasing")
    total = sum(p.numel() for _, p in named)
    if total != sum(counts.values()) or total != reference["stored_total"] + 4 * shared:
        raise RuntimeError("Late-Untied parameter ledger does not close")
    if list(model.shared_analyzer.parameters()):
        raise RuntimeError("Fixed history map acquired parameters")
    return {
        **reference,
        "schema": "hotpot.late_untied.parameter_ledger.v1", "control": KIND,
        "stored_total": total, "scope_counts": dict(sorted(counts.items())),
        "original_late_stored_total": reference["stored_total"],
        "additional_stored_parameters": 4 * shared,
        "all_pde_shared_inference_parameters": 0,
        "all_pde_shared_late_evolution_parameters": 0,
        "shared_history_parameters": 0,
        "late_evolution_parameters_per_family": shared,
        "late_evolution_stored_parameters": 5 * shared,
        "late_family_count": 5,
        "late_family_by_program": {p.value: f for p, f in FAMILY_BY_PROGRAM.items()},
        "per_route_capacity_matches_original_late": True,
        "inactive_stored_parameters_are_other_typed_programs": False,
        "inactive_stored_parameters_are_other_families_or_typed_programs": True,
        "optimizer_inactive_grad_none": True,
    }


def _registry(original):
    result = copy.deepcopy(original)
    result.update({
        "control_kind": KIND, "registry_revision": "hotpot.late_untied.execution.v1",
        "shared_trainable_representation": [],
        "prediction_gradient_to_shared_history": False,
        "prediction_gradient_to_shared_late_evolution": False,
        "prediction_gradient_to_family_private_late_evolution": True,
        "private_full_history_encoder": False,
        "shared_boundary": "none_trainable",
        "late_evolution_boundary": "post_cross_field_pre_action",
        "late_families": list(FAMILIES), "fno_viscosities_share_one_family": True,
        "semantic_claim": "fixed history map; family-private coupling, late evolution and typed programs",
    })
    for nodes in result["programs"].values():
        early = [n for n in nodes if n["identifier"] == "shared.scalar_field_history_analyzer"]
        # CFDB records its carrier and family-specific mixer FFN separately.
        # Insert AFTER both; its late copy is shared within CFDB only.
        mixers = [i for i, n in enumerate(nodes) if n["identifier"] in
                  ("private.typed_cross_field_mixer", "cfdb.family_private_pointwise_flow")]
        if len(early) != 1 or len(mixers) != 1:
            raise RuntimeError("Original registry representation boundary changed")
        early[0].update(identifier="fixed.scalar_field_history_map",
                        parameter_scope="parameter_free",
                        mathematical_role="original parameter-free FixedSemanticHistoryMap")
        nodes.insert(mixers[0] + 1, {
            "identifier": "private.post_cross_field_evolution", "kind": "representation",
            "parameter_scope": "family_private",
            "mathematical_role": "one same-initialized late block per family, before action decoding",
        })
    return result


def model_class(repo, bindings, kind=KIND, scale=SCALE, config=None):
    """Return a Base6 wrapper with the exact canonical training keyword ABI."""
    from torch import nn
    from .dispatcher import FamilyLateDispatcher

    class V13LateUntiedControl(source.V13LateSharedEvolutionControl):
        def __init__(self, config=None):
            super().__init__(config)
            self._original_late_ledger = source.parameter_ledger(self)
            self.late_shared = FamilyLateDispatcher(self.late_shared)

        def _semantic_batch(self, history, grouped):
            semantic, indices = super()._semantic_batch(history, grouped)
            self.late_shared.bind(indices, history.shape[0])
            return semantic, indices

        def forward(self, history, programs, conditions, condition_mask=None, *, return_audit=False):
            try:
                output = super().forward(history, programs, conditions, condition_mask,
                                         return_audit=return_audit)
                if return_audit:
                    registry = self.operator_registry()
                    output.audit.update({
                        "schema": "hotpot.late_untied.forward_audit.v1", "control_kind": KIND,
                        "shared_boundary": "none_trainable",
                        "late_evolution_boundary": "post_cross_field_pre_action",
                        "late_shared_calls": 0,
                        "late_dispatcher_calls": self.late_shared.call_count,
                        "family_late_calls": {f: b.call_count for f, b in self.late_shared.blocks.items()},
                        "typed_program_nodes": {p: [n["identifier"] for n in registry["programs"][p]]
                                                for p in output.audit["typed_program_nodes"]},
                    })
                return output
            finally:
                # Neither success nor a failed forward leaves stale family routing.
                self.late_shared.clear_routes()

        def operator_registry(self):
            return _registry(super().operator_registry())

    resolved = config

    class Base6V13LateUntiedControl(source.Base6V13LateSharedEvolutionControl):
        def __init__(self, workspace=None, config=None):
            nn.Module.__init__(self)
            self.workspace = workspace
            self.model = V13LateUntiedControl(resolved if config is None else config)

        def parameter_ledger(self):
            return parameter_ledger(self.model)

    return Base6V13LateUntiedControl
