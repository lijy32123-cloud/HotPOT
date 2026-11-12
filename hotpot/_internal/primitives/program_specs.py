from __future__ import annotations

from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
    ProgramId,
)

from .contracts import (
    LatentProgramSpec,
    PhysicalOperatorContract,
    PrimitiveContract,
    ProgramStep,
    StepKind,
)


PROGRAM_ORDER = tuple(ProgramId)
PROGRAM_CODE = {program: index for index, program in enumerate(PROGRAM_ORDER)}
FAMILY_BY_PROGRAM = {
    ProgramId.FNO_VORTICITY_NS: "fno",
    ProgramId.CNS_COMPRESSIBLE: "cns",
    ProgramId.SWE_HEIGHT: "swe",
    ProgramId.DR_REACTION_DIFFUSION: "dr",
    ProgramId.CFDB_CAVITY: "cfdb",
    ProgramId.CFDB_CYLINDER: "cfdb",
    ProgramId.CFDB_TUBE: "cfdb",
}
CFDB_PORT = {
    ProgramId.CFDB_CAVITY: "cfdb__port_cavity",
    ProgramId.CFDB_CYLINDER: "cfdb__port_cylinder",
    ProgramId.CFDB_TUBE: "cfdb__port_tube",
}


def _step(kind: StepKind, identifier: str | None = None) -> ProgramStep:
    return ProgramStep(kind, identifier)


PROGRAM_SPECS = {
    ProgramId.FNO_VORTICITY_NS: LatentProgramSpec(
        ProgramId.FNO_VORTICITY_NS,
        "fno",
        (
            _step(StepKind.CONDITION, "fno__condition"),
            _step(StepKind.TRANSFORM, "fno__periodic_transport"),
            _step(StepKind.SUBTRACT_BASE),
        ),
    ),
    ProgramId.DR_REACTION_DIFFUSION: LatentProgramSpec(
        ProgramId.DR_REACTION_DIFFUSION,
        "dr",
        (
            _step(StepKind.CONDITION, "dr__condition"),
            _step(StepKind.TRANSFORM, "dr__neumann_local_0"),
            _step(StepKind.TRANSFORM, "dr__neumann_local_1"),
            _step(StepKind.SUBTRACT_BASE),
        ),
    ),
    ProgramId.SWE_HEIGHT: LatentProgramSpec(
        ProgramId.SWE_HEIGHT,
        "swe",
        (
            _step(StepKind.CONDITION, "swe__condition"),
            _step(StepKind.TRANSFORM, "swe__potential"),
            _step(StepKind.TRANSFORM, "swe__neumann_boundary"),
            _step(StepKind.SUBTRACT_BASE),
        ),
    ),
    ProgramId.CNS_COMPRESSIBLE: LatentProgramSpec(
        ProgramId.CNS_COMPRESSIBLE,
        "cns",
        (
            _step(StepKind.CONDITION, "cns__condition"),
            _step(StepKind.TRANSFORM, "cns__periodic_local_0"),
            _step(StepKind.TRANSFORM, "cns__periodic_local_1"),
            _step(StepKind.SUBTRACT_BASE),
            _step(StepKind.MEAN_FREE),
        ),
    ),
    **{
        program: LatentProgramSpec(
            program,
            "cfdb",
            (
                _step(StepKind.CONDITION, "cfdb__condition"),
                _step(StepKind.SET_FROM_BASE, "cfdb__observation_flow"),
                _step(StepKind.ADD_FROM_BASE, port),
            ),
        )
        for program, port in CFDB_PORT.items()
    },
}


def _contract(
    identifier: str,
    kind: StepKind,
    boundary: str,
    role: str,
    scope: str,
    family: str,
    programs: tuple[ProgramId, ...],
    *,
    environments: tuple[str, ...] = (),
) -> PrimitiveContract:
    return PrimitiveContract(
        identifier=identifier,
        step_kind=kind,
        input_chart="latent_512_at_depth",
        output_chart="latent_512_at_depth",
        boundary=boundary,
        mathematical_role=role,
        condition_fields=tuple(CONDITION_FIELDS) if kind is StepKind.CONDITION else (),
        parameter_scope=scope,
        owner_family=family,
        qualified_programs=programs,
        qualified_environments=environments,
    )


FNO = (ProgramId.FNO_VORTICITY_NS,)
DR = (ProgramId.DR_REACTION_DIFFUSION,)
SWE = (ProgramId.SWE_HEIGHT,)
CNS = (ProgramId.CNS_COMPRESSIBLE,)
CFDB = tuple(CFDB_PORT)

LEARNED_CONTRACTS = {
    row.identifier: row
    for row in (
        _contract("fno__condition", StepKind.CONDITION, "periodic", "typed_condition", "coefficient_family_shared", "fno", FNO, environments=("ns2d_fno_1e-5", "ns2d_fno_1e-3")),
        _contract("fno__periodic_transport", StepKind.TRANSFORM, "periodic", "spectral_transport_representation", "coefficient_family_shared", "fno", FNO, environments=("ns2d_fno_1e-5", "ns2d_fno_1e-3")),
        _contract("dr__condition", StepKind.CONDITION, "neumann", "typed_condition", "program_private", "dr", DR),
        _contract("dr__neumann_local_0", StepKind.TRANSFORM, "neumann", "reaction_diffusion_local_stage_0", "program_private", "dr", DR),
        _contract("dr__neumann_local_1", StepKind.TRANSFORM, "neumann", "reaction_diffusion_local_stage_1", "program_private", "dr", DR),
        _contract("swe__condition", StepKind.CONDITION, "closed_neumann", "typed_condition", "program_private", "swe", SWE),
        _contract("swe__potential", StepKind.TRANSFORM, "closed_neumann", "height_observation_potential", "program_private", "swe", SWE),
        _contract("swe__neumann_boundary", StepKind.TRANSFORM, "closed_neumann", "height_boundary_correction", "program_private", "swe", SWE),
        _contract("cns__condition", StepKind.CONDITION, "periodic", "typed_condition", "program_private", "cns", CNS),
        _contract("cns__periodic_local_0", StepKind.TRANSFORM, "periodic", "compressible_conservative_local_stage_0", "program_private", "cns", CNS),
        _contract("cns__periodic_local_1", StepKind.TRANSFORM, "periodic", "compressible_conservative_local_stage_1", "program_private", "cns", CNS),
        _contract("cfdb__condition", StepKind.CONDITION, "geometry_typed", "observation_condition", "cfdb_family_shared", "cfdb", CFDB),
        _contract("cfdb__observation_flow", StepKind.SET_FROM_BASE, "geometry_neutral", "velocity_observation_flow", "cfdb_family_shared", "cfdb", CFDB),
        _contract("cfdb__port_cavity", StepKind.ADD_FROM_BASE, "cavity_walls", "cavity_geometry_port", "geometry_private", "cfdb", (ProgramId.CFDB_CAVITY,)),
        _contract("cfdb__port_cylinder", StepKind.ADD_FROM_BASE, "inlet_outlet_obstacle", "cylinder_geometry_port", "geometry_private", "cfdb", (ProgramId.CFDB_CYLINDER,)),
        _contract("cfdb__port_tube", StepKind.ADD_FROM_BASE, "inlet_outlet_walls", "tube_geometry_port", "geometry_private", "cfdb", (ProgramId.CFDB_TUBE,)),
    )
}

DETERMINISTIC_CONTRACTS = {
    "subtract_base": PrimitiveContract(
        "subtract_base", StepKind.SUBTRACT_BASE, "latent_512_at_depth",
        "latent_512_at_depth", "neutral", "residual_coordinate_difference", (),
        "parameter_free", "all", tuple(ProgramId), parameterized=False,
    ),
    "mean_free": PrimitiveContract(
        "mean_free", StepKind.MEAN_FREE, "latent_512_at_depth",
        "latent_512_at_depth", "periodic", "zero_spatial_mean_projection", (),
        "parameter_free", "cns", CNS, parameterized=False,
    ),
}

ABSTAINED_CANDIDATES = {
    "cross_pde_advection": {
        "candidate": "E_advect(q,u)=-u dot grad(q)",
        "status": "UNQUALIFIED_NOT_IN_GRAPH",
        "reason": "Base6 released observations do not expose one common identifiable state contract across the proposed source programs",
    }
}


def _physical(
    identifier: str,
    input_chart: str,
    output_chart: str,
    boundary: str,
    role: str,
    programs: tuple[ProgramId, ...],
    source: str,
) -> PhysicalOperatorContract:
    return PhysicalOperatorContract(
        identifier, input_chart, output_chart, boundary, role, programs, source
    )


PHYSICAL_NUMERICAL_CONTRACTS = {
    row.identifier: row
    for row in (
        _physical("fno_native64_exact_left_inverse", "public128_bilinear_scalar", "native64_scalar", "periodic", "exact_chart_restriction", FNO, "native_charts.fno_native64_exact_left_inverse"),
        _physical("fno_source_cn_diffusion", "native64_vorticity", "native64_vorticity", "periodic", "exact_source_discrete_diffusion_subflow", FNO, "composers.FNOProgram._diffuse"),
        _physical("kolmogorov_forcing", "native64_coordinates", "native64_vorticity_rate", "periodic", "deterministic_forcing", FNO, "physics.kolmogorov_forcing"),
        _physical("fno_midpoint_strang", "native64_vorticity_and_closure", "native64_vorticity", "periodic", "two_substep_midpoint_strang_composition", FNO, "composers.FNOProgram.forward"),
        _physical("dr_diffusivity_tensor", "dr_two_species_state", "diagonal_diffusivity", "neumann", "typed_constant_diffusivity", DR, "physics.DRContract.diffusivity_tensor"),
        _physical("exact_discrete_neumann_heat", "dr_two_species_state", "dr_two_species_state", "neumann", "exact_linear_diffusion_subflow", DR, "physics.exact_discrete_neumann_heat"),
        _physical("neumann_laplacian", "dr_diffusion_correction", "dr_diffusion_rate", "neumann", "boundary_typed_laplacian", DR, "physics.neumann_laplacian"),
        _physical("dr_midpoint_strang", "dr_state_and_closure", "dr_two_species_state", "neumann", "heat_midpoint_heat_composition", DR, "composers.DRProgram.forward"),
        _physical("channelwise_spatial_mean_projection", "spatial_action", "mean_free_spatial_action", "boundary_typed", "exact_channelwise_mean_removal", (ProgramId.SWE_HEIGHT, ProgramId.CNS_COMPRESSIBLE), "composers.SWEProgram.forward|CNSProgram.forward"),
        _physical("swe_observable_step_scale", "height_history", "positive_step_scale", "closed_neumann", "saved_cadence_observable_scale", SWE, "swe_specialist.observable_step_scale"),
        _physical("global_positive_line_search", "current_and_proposed_height", "positive_height", "closed_neumann", "global_height_positivity_projection", SWE, "swe_specialist.global_positive_line_search"),
        _physical("cns_current_chart", "primitive_cns_state", "dimensionless_local_chart", "periodic", "deterministic_current_state_chart", CNS, "cns_specialist._current_chart"),
        _physical("primitive_to_conservative", "primitive_cns_state", "conservative_cns_state", "periodic", "exact_variable_transform", CNS, "cns_specialist.primitive_to_conservative"),
        _physical("conservative_mean_correction", "candidate_conservative_state", "mean_preserving_conservative_state", "periodic", "exact_discrete_mean_correction", CNS, "composers.CNSProgram.forward"),
        _physical("cns_positivity_segment", "current_and_candidate_conservative_state", "admissible_conservative_state", "periodic", "density_pressure_positivity_segment", CNS, "cns_specialist.positivity_segment"),
        _physical("conservative_to_primitive", "conservative_cns_state", "primitive_cns_state", "periodic", "exact_variable_transform", CNS, "cns_specialist.conservative_to_primitive"),
        _physical("cfdb_velocity_observation_euler", "current_velocity_and_rate", "next_velocity_observation", "geometry_typed", "saved_cadence_observation_update", CFDB, "composers.CFDBProgram.forward"),
        _physical("cfdb_static_support_copy", "current_public_support_channels", "next_public_support_channels", "geometry_typed", "preserve_nonpredicted_observation_channels", CFDB, "composers.CFDBProgram.forward"),
    )
}


def registry_payload() -> dict[str, object]:
    return {
        "programs": {key.value: value.payload() for key, value in PROGRAM_SPECS.items()},
        "learned_primitives": {
            key: value.payload() for key, value in LEARNED_CONTRACTS.items()
        },
        "deterministic_primitives": {
            key: value.payload() for key, value in DETERMINISTIC_CONTRACTS.items()
        },
        "physical_numerical_operators": {
            key: value.payload()
            for key, value in PHYSICAL_NUMERICAL_CONTRACTS.items()
        },
        "abstained_candidates": ABSTAINED_CANDIDATES,
        "cross_pde_shared_learned_primitives": [],
    }


__all__ = [
    "ABSTAINED_CANDIDATES",
    "CFDB_PORT",
    "DETERMINISTIC_CONTRACTS",
    "FAMILY_BY_PROGRAM",
    "LEARNED_CONTRACTS",
    "PROGRAM_CODE",
    "PROGRAM_ORDER",
    "PROGRAM_SPECS",
    "PHYSICAL_NUMERICAL_CONTRACTS",
    "registry_payload",
]
