from __future__ import annotations

from hotpot._internal.numerics.programs import (
    CFDBProgram as _CFDBProgram,
    CNSProgram as _CNSProgram,
    DRProgram as _DRProgram,
    FNOProgram as _FNOProgram,
    SWEProgram as _SWEProgram,
)
import torch
from torch import Tensor

from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
)
from hotpot._internal.data_protocol.native_charts import (
    fno_native64_exact_left_inverse,
)
from hotpot._internal.data_protocol.physics import (
    exact_discrete_neumann_heat,
    source_reaction_rhs,
)
from hotpot._internal.data_protocol.swe_specialist import (
    global_positive_line_search,
    observable_step_scale,
)

from .heads import (
    CFDBDirectObservationHead,
    CNSCurrentStateHead,
    DRCurrentStateHead,
    DRKnownSourceResidualHead,
    FNOCurrentStateHead,
    SWECalibratedIncrementHead,
    SWECurrentStateHead,
)


_CONDITION = {name: index for index, name in enumerate(CONDITION_FIELDS)}


def _uniform_condition(conditions: Tensor, name: str) -> float:
    values = conditions[:, _CONDITION[name]]
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError(f"missing finite condition {name}")
    reference = values[0]
    if not bool(torch.allclose(values, reference.expand_as(values), rtol=0.0, atol=1e-8)):
        raise ValueError(f"condition {name} must be uniform within one program group")
    return float(reference.detach().cpu())


class FNOProgram(_FNOProgram):
    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = FNOCurrentStateHead()


class _InvariantFNOForward:
    """Project only the learned inviscid transport onto its exact tangent space.

    Periodic two-dimensional incompressible transport preserves circulation,
    enstrophy and kinetic energy.  Diffusion and forcing are already executed
    by separate deterministic branches, so their legitimate changes must not
    leak into this projection.
    """

    projection_floor = 1.0e-12

    def _streamfunction(self, vorticity: Tensor) -> Tensor:
        eigenvalue = self.minus_laplacian_eigenvalue.to(
            device=vorticity.device, dtype=vorticity.dtype
        )
        inverse = torch.where(
            eigenvalue > 0,
            eigenvalue.reciprocal(),
            torch.zeros_like(eigenvalue),
        )
        spectrum = torch.fft.rfft2(vorticity, norm="ortho")
        return torch.fft.irfft2(
            spectrum * inverse[None], s=vorticity.shape[-2:], norm="ortho"
        )

    def _remove_direction(self, action: Tensor, direction: Tensor) -> Tensor:
        numerator = (action * direction).mean(dim=(-2, -1), keepdim=True)
        denominator = direction.square().mean(dim=(-2, -1), keepdim=True)
        coefficient = numerator / denominator.clamp_min(self.projection_floor)
        coefficient = torch.where(
            denominator > self.projection_floor,
            coefficient,
            torch.zeros_like(coefficient),
        )
        return action - coefficient * direction

    def project_transport_action(self, action: Tensor, state: Tensor) -> Tensor:
        if action.shape != state.shape or action.ndim != 4 or action.shape[1] != 1:
            raise ValueError("FNO transport projection expects matching [B,1,H,W]")
        projected = action - action.mean(dim=(-2, -1), keepdim=True)
        centered = state - state.mean(dim=(-2, -1), keepdim=True)
        projected = self._remove_direction(projected, centered)
        streamfunction = self._streamfunction(centered[:, 0])[:, None]
        # Orthogonalize the energy gradient against the enstrophy gradient so
        # the second removal does not reintroduce the first component.
        centered_norm = centered.square().mean(dim=(-2, -1), keepdim=True)
        overlap = (streamfunction * centered).mean(
            dim=(-2, -1), keepdim=True
        ) / centered_norm.clamp_min(self.projection_floor)
        overlap = torch.where(
            centered_norm > self.projection_floor,
            overlap,
            torch.zeros_like(overlap),
        )
        energy_direction = streamfunction - overlap * centered
        projected = self._remove_direction(projected, energy_direction)
        # Mean zero is re-applied to suppress roundoff from the two projections.
        return projected - projected.mean(dim=(-2, -1), keepdim=True)

    def forward(
        self,
        history: Tensor,
        decoded: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        self.head.reset_calls()
        scalar = fno_native64_exact_left_inverse(
            history[..., 0].permute(0, 3, 1, 2).contiguous()
        )
        current = scalar[:, -1:].contiguous()
        viscosity = conditions[:, _CONDITION["viscosity"]]
        dt = _uniform_condition(conditions, "observation_dt")
        step = dt / 2.0
        forcing = self.forcing.to(current.dtype).expand_as(current)

        def half_flow(state: Tensor) -> Tensor:
            return self._diffuse(state, viscosity, 0.5 * step)

        def nonlinear_flow(state: Tensor) -> Tensor:
            def rhs(stage: Tensor) -> Tensor:
                learned = self.head(decoded, stage, conditions, condition_mask)
                learned = self.project_transport_action(learned, stage)
                return self.toolbox.add_terms(
                    learned, forcing, owner="fno_vorticity_ns"
                )

            return self.toolbox.explicit_midpoint(
                state,
                step,
                rhs,
                self.toolbox.affine_update,
                owner="fno_vorticity_ns",
            )

        for _ in range(2):
            current = self.toolbox.strang_split(
                current,
                half_flow,
                nonlinear_flow,
                owner="fno_vorticity_ns",
            )
        if self.head.call_count != 4:
            raise RuntimeError("FNO closure must be queried four times")
        return current


class FNOInvariantTransportProgram(_InvariantFNOForward, FNOProgram):
    """Current V1.1 split with a parameter-free inviscid-action projection."""


class DRProgram(_DRProgram):
    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = DRCurrentStateHead()


class DRKnownSourceResidualProgram(_DRProgram):
    """Known reaction--diffusion flow plus a bounded endpoint discrepancy."""

    correction_fraction = 0.02

    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = DRKnownSourceResidualHead()

    def forward(
        self,
        history: Tensor,
        decoded: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        self.head.reset_calls()
        state = history[..., -1, :2].contiguous()
        # Base6 exposes a single registered DR coefficient point.  These fixed
        # source constants therefore belong to the program contract, while
        # cadence and grid spacing remain explicit condition inputs.
        diffusivity = self.contract.diffusivity_tensor(state)
        reaction_k = self.contract.reaction_k
        dt = _uniform_condition(conditions, "observation_dt")
        dx = _uniform_condition(conditions, "dx")
        dy = _uniform_condition(conditions, "dy")

        def half_flow(value: Tensor) -> Tensor:
            return exact_discrete_neumann_heat(
                value, diffusivity, 0.5 * dt, dx=dx, dy=dy
            )

        def known_reaction_flow(value: Tensor) -> Tensor:
            first = source_reaction_rhs(value, reaction_k)
            midpoint = value + 0.5 * dt * first
            return value + dt * source_reaction_rhs(midpoint, reaction_k)

        known = self.toolbox.strang_split(
            state,
            half_flow,
            known_reaction_flow,
            owner="dr_reaction_diffusion_known_source",
        )
        raw = self.head(decoded, known, conditions, condition_mask)
        state_scale = known.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-3)
        correction = self.correction_fraction * state_scale * torch.tanh(raw)
        prediction = self.toolbox.affine_update(
            known, correction, 1.0, owner="dr_endpoint_closure"
        )
        if self.head.call_count != 1:
            raise RuntimeError("DR endpoint closure must be queried once")
        return prediction


class DRDirectObservationProgram(_DRProgram):
    """Two-stage finite-time update of the observed DR state.

    Endpoint supervision identifies the saved-interval transition of the two
    observed concentrations, but it does not uniquely identify separate
    reaction and diffusion generators.  The two equal-capacity output groups
    of the legacy head are therefore interpreted as endpoint-update proposals
    and refined at a midpoint state.  No known-source solver is inserted in
    this interface.
    """

    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = DRCurrentStateHead()

    def forward(
        self,
        history: Tensor,
        decoded: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        self.head.reset_calls()
        current = history[..., -1, :2].contiguous()
        first_a, first_b = self.head(
            decoded, current, conditions, condition_mask
        )
        first = self.toolbox.add_terms(
            first_a, first_b, owner="dr_finite_time_observation"
        )
        midpoint = self.toolbox.affine_update(
            current,
            first,
            0.5,
            owner="dr_finite_time_observation",
        )
        second_a, second_b = self.head(
            decoded, midpoint, conditions, condition_mask
        )
        second = self.toolbox.add_terms(
            second_a, second_b, owner="dr_finite_time_observation"
        )
        averaged = 0.5 * self.toolbox.add_terms(
            first, second, owner="dr_finite_time_observation"
        )
        prediction = self.toolbox.affine_update(
            current,
            averaged,
            1.0,
            owner="dr_finite_time_observation",
        )
        if self.head.call_count != 2:
            raise RuntimeError("DR finite-time update must be queried twice")
        return prediction


class SWEProgram(_SWEProgram):
    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = SWECurrentStateHead()


class SWECalibratedSavedStepProgram(_SWEProgram):
    """Mass-preserving saved-step action with a learned bounded gain."""

    maximum_log_gain = 0.5

    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = SWECalibratedIncrementHead()
        self.last_gain: Tensor | None = None
        self.last_alpha: Tensor | None = None

    def forward(
        self,
        history: Tensor,
        decoded: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        self.head.reset_calls()
        scalar_history = history[..., 0].permute(0, 3, 1, 2).contiguous()
        direction, log_gain_raw = self.head(
            decoded, scalar_history, conditions, condition_mask
        )
        direction = self.toolbox.uniform_mean_zero(direction, owner="swe_height")
        current = scalar_history[:, -1]
        base_scale = observable_step_scale(scalar_history)[:, 0]
        gain = torch.exp(self.maximum_log_gain * torch.tanh(log_gain_raw))[:, None, None]
        proposal = self.toolbox.affine_update(
            current, direction, base_scale * gain, owner="swe_height"
        )
        prediction, alpha = global_positive_line_search(
            current, proposal, self.minimum_height
        )
        self.last_gain = gain.detach()
        self.last_alpha = alpha.detach()
        if self.head.call_count != 1:
            raise RuntimeError("SWE calibrated closure must be queried once")
        return prediction


class SWEDirectObservationProgram(_SWEProgram):
    """Identifiable finite-time height transition with hard admissibility.

    The released SWE lane observes height only.  The learned object is thus a
    saved-step height increment, not an unobserved momentum/flux generator.
    """

    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = SWECurrentStateHead()
        self.last_alpha: Tensor | None = None

    def forward(
        self,
        history: Tensor,
        decoded: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        self.head.reset_calls()
        scalar_history = history[..., 0].permute(0, 3, 1, 2).contiguous()
        increment = self.head(
            decoded, scalar_history, conditions, condition_mask
        )
        increment = self.toolbox.uniform_mean_zero(
            increment, owner="swe_height"
        )
        current = scalar_history[:, -1]
        proposal = self.toolbox.affine_update(
            current, increment, 1.0, owner="swe_height"
        )
        prediction, alpha = global_positive_line_search(
            current, proposal, self.minimum_height
        )
        self.last_alpha = alpha.detach()
        if self.head.call_count != 1:
            raise RuntimeError("SWE direct observation closure must be queried once")
        return prediction


class CNSProgram(_CNSProgram):
    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = CNSCurrentStateHead()


class CFDBProgram(_CFDBProgram):
    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = CFDBDirectObservationHead()

    def forward(
        self,
        history: Tensor,
        decoded: Tensor,
        program_codes: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        """Predict the released next velocity observation without a raw-rate ABI."""
        self.head.reset_calls()
        current = history[..., -1, :4].contiguous()
        next_velocity = self.head(
            decoded, current, program_codes, conditions, condition_mask
        )
        prediction = current.clone()
        prediction[..., :2] = next_velocity
        if self.head.call_count != 1:
            raise RuntimeError("CFDB direct observation closure must be queried once")
        return prediction


def _cfdb_saved_step_scale(history: Tensor) -> Tensor:
    """Per-velocity saved-step scale from the same public H10 observation."""

    velocity = history[..., :2].permute(0, 3, 4, 1, 2).contiguous()
    differences = velocity[:, 1:] - velocity[:, :-1]
    scale = differences.square().mean((1, 3, 4)).sqrt()
    state = velocity.square().mean((1, 3, 4)).sqrt()
    return scale.clamp_min(1.0e-4 * state.clamp_min(1.0e-6))[:, None, None, :]


class CFDBMaskedVelocityProgram(_CFDBProgram):
    """Family-routed finite-step velocity action with public-mask execution.

    The neural head predicts a dimensionless saved-step velocity action.  The
    deterministic program recovers the time-invariant fluid mask from the same
    (possibly augmented) H10 input, applies the action only in fluid cells, and
    retains observed obstacle/wall/inlet velocities elsewhere.  No pressure or
    hidden flow solver is introduced.
    """

    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = CFDBDirectObservationHead()

    def forward(
        self,
        history: Tensor,
        decoded: Tensor,
        program_codes: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        self.head.reset_calls()
        current = history[..., -1, :4].contiguous()
        raw_action = self.head(
            decoded, current, program_codes, conditions, condition_mask
        )
        step_scale = _cfdb_saved_step_scale(history)
        fluid_mask = (history[..., 2].mean(dim=3) >= 0.5).to(history.dtype)[..., None]
        increment = fluid_mask * step_scale * torch.tanh(raw_action)
        prediction = current.clone()
        prediction[..., :2] = self.toolbox.affine_update(
            current[..., :2], increment, 1.0, owner="cfdb_saved_step_velocity"
        )
        prediction[..., 2] = fluid_mask[..., 0]
        if self.head.call_count != 1:
            raise RuntimeError("CFDB masked velocity closure must be queried once")
        return prediction


class CFDBNormalizedVelocityProgram(_CFDBProgram):
    """Invertible dimensionless finite-step velocity action.

    Endpoint observations identify the saved-step velocity increment, not a
    pointwise PDE generator.  The public H10 velocity differences therefore
    define a positive per-sample, per-component chart scale.  The neural head
    predicts the unbounded dimensionless action in that chart, and this
    deterministic program maps it back to physical velocity units before the
    affine finite-step update.  Unlike the failed V6 interface, this map is
    bijective for every positive scale: it neither clips local increments nor
    freezes velocities solely from a resized support mask.  Static support
    channels remain copied from the released observation.
    """

    def __init__(self, toolbox) -> None:
        super().__init__(toolbox)
        self.head = CFDBDirectObservationHead()

    def forward(
        self,
        history: Tensor,
        decoded: Tensor,
        program_codes: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        self.head.reset_calls()
        current = history[..., -1, :4].contiguous()
        dimensionless_action = self.head(
            decoded, current, program_codes, conditions, condition_mask
        )
        step_scale = _cfdb_saved_step_scale(history)
        physical_increment = step_scale * dimensionless_action
        prediction = current.clone()
        prediction[..., :2] = self.toolbox.affine_update(
            current[..., :2],
            physical_increment,
            1.0,
            owner="cfdb_normalized_saved_step_velocity",
        )
        if self.head.call_count != 1:
            raise RuntimeError("CFDB normalized velocity closure must be queried once")
        return prediction


__all__ = [
    "CFDBMaskedVelocityProgram",
    "CFDBNormalizedVelocityProgram",
    "CFDBProgram",
    "CNSProgram",
    "DRDirectObservationProgram",
    "DRKnownSourceResidualProgram",
    "DRProgram",
    "FNOInvariantTransportProgram",
    "FNOProgram",
    "SWECalibratedSavedStepProgram",
    "SWEDirectObservationProgram",
    "SWEProgram",
    "_InvariantFNOForward",
]
