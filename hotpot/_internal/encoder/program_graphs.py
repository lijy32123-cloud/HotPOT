from __future__ import annotations

from dataclasses import asdict, dataclass

from hotpot._internal.data_protocol.manifest import ProgramId


FAMILY_BY_PROGRAM = {
    ProgramId.FNO_VORTICITY_NS: "fno",
    ProgramId.DR_REACTION_DIFFUSION: "dr",
    ProgramId.SWE_HEIGHT: "swe",
    ProgramId.CNS_COMPRESSIBLE: "cns",
    ProgramId.CFDB_CAVITY: "cfdb",
    ProgramId.CFDB_CYLINDER: "cfdb",
    ProgramId.CFDB_TUBE: "cfdb",
}


ACTION_SCOPE_BY_PROGRAM = {
    ProgramId.FNO_VORTICITY_NS: "fno",
    ProgramId.DR_REACTION_DIFFUSION: "dr",
    ProgramId.SWE_HEIGHT: "swe",
    ProgramId.CNS_COMPRESSIBLE: "cns",
    ProgramId.CFDB_CAVITY: "cfdb_cavity",
    ProgramId.CFDB_CYLINDER: "cfdb_cylinder",
    ProgramId.CFDB_TUBE: "cfdb_tube",
}


@dataclass(frozen=True)
class ProgramNode:
    identifier: str
    kind: str
    parameter_scope: str
    mathematical_role: str


_PREFIX = (
    ProgramNode(
        "chart.explicit_scalar_history_features",
        "representation",
        "parameter_free_private",
        "typed units and boundaries mapped to named scalar-field differences",
    ),
    ProgramNode(
        "shared.scalar_field_history_analyzer",
        "representation",
        "all_pde_shared",
        "identical local/spectral/temporal analysis applied independently per field",
    ),
    ProgramNode(
        "private.typed_cross_field_mixer",
        "representation",
        "family_private",
        "PDE-specific variable coupling without a raw H10 input port",
    ),
)


def _nodes(*values: tuple[str, str, str, str]) -> tuple[ProgramNode, ...]:
    return _PREFIX + tuple(ProgramNode(*value) for value in values)


def _cfdb_nodes(*values: tuple[str, str, str, str]) -> tuple[ProgramNode, ...]:
    return _PREFIX[:2] + (
        ProgramNode(
            "cfdb.shared_velocity_carrier",
            "representation",
            "cfdb_shared",
            "shared cross-velocity projection plus spectral and local analysis",
        ),
        ProgramNode(
            "cfdb.family_private_pointwise_flow",
            "learned_action",
            "cfdb_family_private",
            "hard-routed cavity, cylinder or tube nonlinear saved-frame flow",
        ),
    ) + tuple(ProgramNode(*value) for value in values)


PROGRAM_GRAPHS = {
    ProgramId.FNO_VORTICITY_NS: _nodes(
        ("learned.fno_transport_action", "learned_action", "fno_private", "native64 effective transport action"),
        ("fno.source_cn_diffusion", "numerical_backend", "parameter_free_private", "source-discrete periodic diffusion"),
        ("time.explicit_midpoint", "algorithm", "parameter_free_shared", "second-order action integration"),
        ("time.strang_split", "algorithm", "parameter_free_shared", "symmetric split composition"),
    ),
    ProgramId.DR_REACTION_DIFFUSION: _nodes(
        ("learned.dr_coupled_action", "learned_action", "dr_private", "endpoint-identifiable coupled correction"),
        ("dr.exact_discrete_neumann_heat", "numerical_backend", "parameter_free_private", "exact Neumann heat subflow"),
        ("time.explicit_midpoint", "algorithm", "parameter_free_shared", "second-order action integration"),
        ("time.strang_split", "algorithm", "parameter_free_shared", "symmetric split composition"),
    ),
    ProgramId.SWE_HEIGHT: _nodes(
        ("learned.swe_observation_action", "learned_action", "swe_private", "height-only saved-step action"),
        ("constraint.uniform_mean_zero", "algorithm", "parameter_free_shared", "mass-compatible action projection"),
        ("swe.positive_height_segment", "constraint", "parameter_free_private", "positive height update"),
    ),
    ProgramId.CNS_COMPRESSIBLE: _nodes(
        ("learned.cns_conservative_action", "learned_action", "cns_private", "finite-time conservative action"),
        ("constraint.uniform_mean_zero", "algorithm", "parameter_free_shared", "conservative action projection"),
        ("cns.density_pressure_segment", "constraint", "parameter_free_private", "EOS admissibility"),
    ),
    ProgramId.CFDB_CAVITY: _cfdb_nodes(
        ("learned.cfdb_direct_observation", "learned_action", "cfdb_family_private", "cavity next-velocity observation transition"),
        ("cfdb.static_support_copy", "constraint", "parameter_free_private", "preserve support channels"),
    ),
    ProgramId.CFDB_CYLINDER: _cfdb_nodes(
        ("learned.cfdb_direct_observation", "learned_action", "cfdb_family_private", "cylinder next-velocity observation transition"),
        ("cfdb.static_support_copy", "constraint", "parameter_free_private", "preserve support channels"),
    ),
    ProgramId.CFDB_TUBE: _cfdb_nodes(
        ("learned.cfdb_direct_observation", "learned_action", "cfdb_family_private", "tube next-velocity observation transition"),
        ("cfdb.static_support_copy", "constraint", "parameter_free_private", "preserve support channels"),
    ),
}


KNOWN_SOURCE_DR_PROGRAM_GRAPHS = dict(PROGRAM_GRAPHS)
KNOWN_SOURCE_DR_PROGRAM_GRAPHS[ProgramId.DR_REACTION_DIFFUSION] = _nodes(
    (
        "dr.known_pointwise_source",
        "numerical_backend",
        "parameter_free_private",
        "registered FitzHugh--Nagumo source law",
    ),
    (
        "dr.exact_discrete_neumann_heat",
        "numerical_backend",
        "parameter_free_private",
        "exact Neumann heat subflow",
    ),
    (
        "time.explicit_midpoint",
        "algorithm",
        "parameter_free_shared",
        "second-order known-source integration",
    ),
    (
        "time.strang_split",
        "algorithm",
        "parameter_free_shared",
        "symmetric known reaction--diffusion composition",
    ),
    (
        "learned.dr_endpoint_discrepancy",
        "learned_action",
        "dr_private",
        "bounded residual of the saved-step observation after the known flow",
    ),
)
IDENTIFIABLE_SWE_DR_PROGRAM_GRAPHS = dict(KNOWN_SOURCE_DR_PROGRAM_GRAPHS)
IDENTIFIABLE_SWE_DR_PROGRAM_GRAPHS[ProgramId.SWE_HEIGHT] = _nodes(
    (
        "learned.swe_saved_step_shape_and_gain",
        "learned_action",
        "swe_private",
        "height-only saved-step direction with bounded latent-conditioned calibration",
    ),
    (
        "constraint.uniform_mean_zero",
        "algorithm",
        "parameter_free_shared",
        "mass-compatible action projection",
    ),
    (
        "swe.positive_height_segment",
        "constraint",
        "parameter_free_private",
        "positive height update",
    ),
)

FNO_INVARIANT_PROGRAM_GRAPHS = dict(PROGRAM_GRAPHS)
FNO_INVARIANT_PROGRAM_GRAPHS[ProgramId.FNO_VORTICITY_NS] = _nodes(
    (
        "learned.fno_transport_action",
        "learned_action",
        "fno_private",
        "native64 effective inviscid transport action",
    ),
    (
        "constraint.fno_inviscid_tangent_projection",
        "constraint",
        "parameter_free_private",
        "remove circulation, enstrophy and kinetic-energy change from transport",
    ),
    (
        "fno.source_cn_diffusion",
        "numerical_backend",
        "parameter_free_private",
        "source-discrete periodic diffusion",
    ),
    (
        "time.explicit_midpoint",
        "algorithm",
        "parameter_free_shared",
        "second-order action integration",
    ),
    (
        "time.strang_split",
        "algorithm",
        "parameter_free_shared",
        "symmetric split composition",
    ),
)

SWE_DIRECT_OBSERVATION_PROGRAM_GRAPHS = dict(PROGRAM_GRAPHS)
SWE_DIRECT_OBSERVATION_PROGRAM_GRAPHS[ProgramId.SWE_HEIGHT] = _nodes(
    (
        "learned.swe_direct_saved_step_increment",
        "learned_action",
        "swe_private",
        "finite-time transition of the released height observation",
    ),
    (
        "constraint.uniform_mean_zero",
        "algorithm",
        "parameter_free_shared",
        "mass-compatible increment projection",
    ),
    (
        "swe.positive_height_segment",
        "constraint",
        "parameter_free_private",
        "positive height update",
    ),
)

FINITE_TIME_UPDATE_PROGRAM_GRAPHS = dict(SWE_DIRECT_OBSERVATION_PROGRAM_GRAPHS)
FINITE_TIME_UPDATE_PROGRAM_GRAPHS[ProgramId.DR_REACTION_DIFFUSION] = _nodes(
    (
        "learned.dr_direct_saved_step_increment",
        "learned_action",
        "dr_private",
        "finite-time transition of the released two-field observation",
    ),
    (
        "time.two_stage_endpoint_refinement",
        "algorithm",
        "parameter_free_shared",
        "midpoint refinement in finite-time update coordinates",
    ),
)

EVIDENCE_SELECTED_PROGRAM_GRAPHS = dict(FINITE_TIME_UPDATE_PROGRAM_GRAPHS)
EVIDENCE_SELECTED_PROGRAM_GRAPHS[ProgramId.FNO_VORTICITY_NS] = _nodes(
    (
        "chart.fno_source_aligned_stage_context",
        "representation",
        "parameter_free_private",
        "current-stage Biot--Savart transport rate and known forcing replace generic absolute coordinates",
    ),
    (
        "learned.fno_instantaneous_transport_rate",
        "learned_action",
        "fno_private",
        "native64 instantaneous transport rate conditioned on shared history and the source-aligned stage context",
    ),
    (
        "fno.source_cn_diffusion",
        "numerical_backend",
        "parameter_free_private",
        "source-discrete periodic viscous diffusion",
    ),
    (
        "time.four_substep_explicit_midpoint",
        "algorithm",
        "parameter_free_shared",
        "four source-consistent midpoint substeps over one observation interval",
    ),
    (
        "time.strang_split",
        "algorithm",
        "parameter_free_shared",
        "symmetric diffusion--transport composition within each substep",
    ),
)

UNIFIED_TYPED_PROGRAM_GRAPHS = dict(FINITE_TIME_UPDATE_PROGRAM_GRAPHS)
for _cfdb_program in (
    ProgramId.CFDB_CAVITY,
    ProgramId.CFDB_CYLINDER,
    ProgramId.CFDB_TUBE,
):
    UNIFIED_TYPED_PROGRAM_GRAPHS[_cfdb_program] = _nodes(
        (
            "learned.cfdb_saved_step_velocity_action",
            "learned_action",
            "cfdb_family_private",
            "family-routed dimensionless finite-step velocity action",
        ),
        (
            "constraint.cfdb_public_fluid_mask",
            "constraint",
            "parameter_free_private",
            "recover static support from H10 and retain observed boundary velocity",
        ),
    )

UNIFIED_TYPED_CLOSED_PROGRAM_GRAPHS = dict(UNIFIED_TYPED_PROGRAM_GRAPHS)
for _cfdb_program in (
    ProgramId.CFDB_CAVITY,
    ProgramId.CFDB_CYLINDER,
    ProgramId.CFDB_TUBE,
):
    UNIFIED_TYPED_CLOSED_PROGRAM_GRAPHS[_cfdb_program] = _nodes(
        (
            "learned.cfdb_dimensionless_saved_step_velocity_action",
            "learned_action",
            "cfdb_family_private",
            "family-routed unbounded finite-step velocity action",
        ),
        (
            "chart.cfdb_observed_step_scale",
            "representation",
            "parameter_free_private",
            "invertible map from dimensionless action to released velocity units",
        ),
        (
            "time.cfdb_affine_saved_step",
            "algorithm",
            "parameter_free_shared",
            "finite-step velocity composition with static support-channel copy",
        ),
    )

NATIVE_ACTION_PROGRAM_GRAPHS = dict(KNOWN_SOURCE_DR_PROGRAM_GRAPHS)
NATIVE_ACTION_PROGRAM_GRAPHS[ProgramId.FNO_VORTICITY_NS] = _nodes(
    (
        "chart.fno_stage_velocity_gradient",
        "representation",
        "parameter_free_private",
        "periodic streamfunction, velocity and vorticity gradient recomputed at every midpoint stage",
    ),
    (
        "learned.fno_transport_action",
        "learned_action",
        "fno_private",
        "native64 total transport action conditioned on the stage chart",
    ),
    ("fno.source_cn_diffusion", "numerical_backend", "parameter_free_private", "source-discrete periodic diffusion"),
    ("time.explicit_midpoint", "algorithm", "parameter_free_shared", "second-order action integration"),
    ("time.strang_split", "algorithm", "parameter_free_shared", "symmetric split composition"),
)
NATIVE_ACTION_PROGRAM_GRAPHS[ProgramId.SWE_HEIGHT] = _nodes(
    (
        "chart.swe_recent_phase_differences",
        "representation",
        "parameter_free_private",
        "native-grid first and second saved-step height differences",
    ),
    (
        "learned.swe_observation_action",
        "learned_action",
        "swe_private",
        "phase-aware height-only saved-step action",
    ),
    ("constraint.uniform_mean_zero", "algorithm", "parameter_free_shared", "mass-compatible action projection"),
    ("swe.positive_height_segment", "constraint", "parameter_free_private", "positive height update"),
)


def program_graphs_for_revision(interface_revision: str):
    if interface_revision == "legacy_v1":
        return PROGRAM_GRAPHS
    if interface_revision == "identifiable_swe_dr_v2":
        return IDENTIFIABLE_SWE_DR_PROGRAM_GRAPHS
    if interface_revision == "known_source_dr_v2":
        return KNOWN_SOURCE_DR_PROGRAM_GRAPHS
    if interface_revision == "fno_invariant_transport_v3":
        return FNO_INVARIANT_PROGRAM_GRAPHS
    if interface_revision == "swe_direct_observation_v3":
        return SWE_DIRECT_OBSERVATION_PROGRAM_GRAPHS
    if interface_revision == "native_action_experts_v4":
        return NATIVE_ACTION_PROGRAM_GRAPHS
    if interface_revision == "finite_time_update_v5":
        return FINITE_TIME_UPDATE_PROGRAM_GRAPHS
    if interface_revision == "unified_typed_finite_step_v6":
        return UNIFIED_TYPED_PROGRAM_GRAPHS
    if interface_revision == "unified_typed_closed_v7":
        return UNIFIED_TYPED_CLOSED_PROGRAM_GRAPHS
    if interface_revision == "evidence_selected_typed_program_v8":
        return EVIDENCE_SELECTED_PROGRAM_GRAPHS
    raise ValueError(f"unknown typed interface revision {interface_revision!r}")


def registry_payload(interface_revision: str = "legacy_v1") -> dict[str, object]:
    graphs = program_graphs_for_revision(interface_revision)
    return {
        "schema": "semantic_field_program_foundation.typed_dag.v1",
        "interface_revision": interface_revision,
        "programs": {
            program.value: [asdict(node) for node in nodes]
            for program, nodes in graphs.items()
        },
        "shared_trainable_representation": ["shared.scalar_field_history_analyzer"],
        "shared_training_only_objective": [],
        "private_full_history_encoder": False,
        "prediction_gradient_to_shared_history": True,
        "gradient_firewall": False,
        "gradient_surgery": False,
        "learned_router": False,
        "top_k": None,
        "semantic_claim": "shared computation over scalar fields, not shared physical meaning",
        "qualified_cross_pde_shared_learned_physical_actions": [],
    }


__all__ = [
    "ACTION_SCOPE_BY_PROGRAM",
    "EVIDENCE_SELECTED_PROGRAM_GRAPHS",
    "FAMILY_BY_PROGRAM",
    "FNO_INVARIANT_PROGRAM_GRAPHS",
    "FINITE_TIME_UPDATE_PROGRAM_GRAPHS",
    "IDENTIFIABLE_SWE_DR_PROGRAM_GRAPHS",
    "KNOWN_SOURCE_DR_PROGRAM_GRAPHS",
    "NATIVE_ACTION_PROGRAM_GRAPHS",
    "PROGRAM_GRAPHS",
    "SWE_DIRECT_OBSERVATION_PROGRAM_GRAPHS",
    "UNIFIED_TYPED_PROGRAM_GRAPHS",
    "ProgramNode",
    "program_graphs_for_revision",
    "registry_payload",
]
