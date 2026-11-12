from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from hotpot._internal.data_protocol.swe_specialist import (
    global_positive_line_search,
    observable_step_scale,
)
from hotpot._internal.data_protocol.native_charts import (
    fno_native64_exact_left_inverse,
)
from hotpot._internal.encoder.programs import (
    CFDBMaskedVelocityProgram as _CFDBMaskedVelocityProgram,
    DRDirectObservationProgram as _DRDirectObservationProgram,
    DRKnownSourceResidualProgram as _DRKnownSourceResidualProgram,
    _InvariantFNOForward,
)

from hotpot._internal.numerics.programs import (
    CFDBProgram as _CFDBProgram,
    CNSProgram as _CNSProgram,
    DRProgram as _DRProgram,
    FNOProgram as _FNOProgram,
    SWEProgram as _SWEProgram,
    _CONDITION,
    _uniform,
)

from .config import FoundationScaleConfig
from .heads import (
    CFDBDirectObservationHead,
    CNSCurrentStateHead,
    DRCurrentStateHead,
    DRKnownSourceResidualHead,
    FNOAnalyticGateHead,
    FNOCurrentStateHead,
    FNOHighBandClosureHead,
    FNOSourceAlignedStageChartHead,
    FNOSpectralActionHead,
    FNOStageChartHead,
    SWECurrentStateHead,
    SWEPhaseAwareDetailHead,
    SWELocalPhaseGate,
    SWEMultiscalePhaseGate,
    SWEWaveFluxGate,
)


class FNOProgram(_FNOProgram):
    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = FNOCurrentStateHead(config)


class FNOSourceAlignedSubstep4Program(FNOProgram):
    """Source-aligned instantaneous transport with numerical integration."""

    action_abi = "instantaneous_transport_rate_v1"
    interface_revision = "fno_source_aligned_substep4_v3"
    substeps = 4

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox, config)
        self.head = FNOSourceAlignedStageChartHead(config)

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
        observation_dt = _uniform(conditions, "observation_dt")
        step = observation_dt / float(self.substeps)
        forcing = self.forcing.to(current.dtype).expand_as(current)

        def half_flow(state: Tensor) -> Tensor:
            return self._diffuse(state, viscosity, 0.5 * step)

        def nonlinear_flow(state: Tensor) -> Tensor:
            def rhs(stage: Tensor) -> Tensor:
                learned = self.head(decoded, stage, conditions, condition_mask)
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

        for _ in range(self.substeps):
            current = self.toolbox.strang_split(
                current,
                half_flow,
                nonlinear_flow,
                owner="fno_vorticity_ns",
            )
        if self.head.call_count != 2 * self.substeps:
            raise RuntimeError("FNO refined closure call count drifted")
        return current


class FNOInvariantTransportProgram(_InvariantFNOForward, FNOProgram):
    """90M-width head inside the same parameter-free invariant projection."""


class FNOStageChartProgram(_FNOProgram):
    """Unchanged CN/Strang/midpoint program with a stage-global action chart."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = FNOStageChartHead(config)


class FNOAnalyticGateProgram(_FNOProgram):
    """Legacy action with a stage-recomputed analytic transport gate."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = FNOAnalyticGateHead(config)


class FNOSpectralActionProgram(_FNOProgram):
    """Legacy CN/Strang/midpoint solver with a typed spectral action chart."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = FNOSpectralActionHead(config)


class FNOHighBandClosureProgram(_FNOProgram):
    """Legacy numerical program with a diagnosed high-band action closure."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = FNOHighBandClosureHead(config)


class DRProgram(_DRProgram):
    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = DRCurrentStateHead(config)


class DRKnownSourceResidualProgram(_DRKnownSourceResidualProgram):
    """90M-width closure around the selected deterministic DR source step."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = DRKnownSourceResidualHead(config)


class DRDirectObservationProgram(_DRDirectObservationProgram):
    """Width-scaled finite-step DR action with midpoint refinement."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = DRCurrentStateHead(config)


class SWEProgram(_SWEProgram):
    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = SWECurrentStateHead(config)


class SWEDirectObservationProgram(_SWEProgram):
    """90M-width finite-time height transition with mass and positivity."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = SWECurrentStateHead(config)
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
        increment = self.head(decoded, scalar_history, conditions, condition_mask)
        increment = self.toolbox.uniform_mean_zero(increment, owner="swe_height")
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


class SWEPhaseAwareProgram(_SWEProgram):
    """Unchanged saved-step composer with a native-grid phase-aware head."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = SWEPhaseAwareDetailHead(config)


class SWEPhaseCarryProgram(_SWEProgram):
    """Saved-step SWE composer with an explicit observable phase carry.

    The existing learned action remains unchanged.  A single zero-initialized
    coefficient can carry the most recent observed height displacement into
    the next saved step.  Both terms are projected to zero mean before the
    original positivity line search, so the interface keeps the same mass and
    positivity contract and is exactly function preserving at initialization.
    """

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = SWEPhaseAwareDetailHead(config)
        self.phase_carry_logit = nn.Parameter(torch.zeros(()))
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
        field = self.head(decoded, scalar_history, conditions, condition_mask)
        field = self.toolbox.uniform_mean_zero(field, owner="swe_height")
        current = scalar_history[:, -1]
        previous = scalar_history[:, -2]
        learned_increment = field * observable_step_scale(scalar_history)[:, 0]
        observed_phase = self.toolbox.uniform_mean_zero(
            current - previous, owner="swe_height"
        )
        increment = self.toolbox.add_terms(
            learned_increment,
            torch.tanh(self.phase_carry_logit) * observed_phase,
            owner="swe_height",
        )
        increment = self.toolbox.uniform_mean_zero(increment, owner="swe_height")
        proposal = self.toolbox.affine_update(
            current, increment, 1.0, owner="swe_height"
        )
        prediction, alpha = global_positive_line_search(
            current, proposal, self.minimum_height
        )
        self.last_alpha = alpha.detach()
        if self.head.call_count != 1:
            raise RuntimeError("SWE phase-carry closure must be queried once")
        return prediction


class SWELocalPhaseGateProgram(_SWEProgram):
    """Legacy saved-step action composed with local phase/acceleration gates."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = SWECurrentStateHead(config)
        self.phase_gate = SWELocalPhaseGate(config)
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
        field = self.head(decoded, scalar_history, conditions, condition_mask)
        field = self.toolbox.uniform_mean_zero(field, owner="swe_height")
        current = scalar_history[:, -1]
        previous = scalar_history[:, -2]
        before_previous = scalar_history[:, -3]
        phase_velocity = current - previous
        phase_acceleration = current - 2.0 * previous + before_previous
        gates = self.phase_gate(decoded, scalar_history)
        learned_increment = field * observable_step_scale(scalar_history)[:, 0]
        phase_increment = self.toolbox.add_terms(
            gates[:, 0] * phase_velocity,
            gates[:, 1] * phase_acceleration,
            owner="swe_height",
        )
        increment = self.toolbox.add_terms(
            learned_increment, phase_increment, owner="swe_height"
        )
        increment = self.toolbox.uniform_mean_zero(increment, owner="swe_height")
        proposal = self.toolbox.affine_update(
            current, increment, 1.0, owner="swe_height"
        )
        prediction, alpha = global_positive_line_search(
            current, proposal, self.minimum_height
        )
        self.last_alpha = alpha.detach()
        if self.head.call_count != 1:
            raise RuntimeError("SWE local phase-gate closure must be queried once")
        return prediction


class SWEMultiscalePhaseGateProgram(_SWEProgram):
    """Mass/positivity-safe SWE composer with multiscale observable phase."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = SWECurrentStateHead(config)
        self.phase_gate = SWEMultiscalePhaseGate(config)
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
        field = self.head(decoded, scalar_history, conditions, condition_mask)
        field = self.toolbox.uniform_mean_zero(field, owner="swe_height")
        current = scalar_history[:, -1]
        previous = scalar_history[:, -2]
        before_previous = scalar_history[:, -3]
        gates = self.phase_gate(decoded, scalar_history)
        learned_increment = field * observable_step_scale(scalar_history)[:, 0]
        phase_increment = self.toolbox.add_terms(
            gates[:, 0] * (current - previous),
            gates[:, 1] * (current - 2.0 * previous + before_previous),
            owner="swe_height",
        )
        increment = self.toolbox.add_terms(
            learned_increment, phase_increment, owner="swe_height"
        )
        increment = self.toolbox.uniform_mean_zero(increment, owner="swe_height")
        proposal = self.toolbox.affine_update(
            current, increment, 1.0, owner="swe_height"
        )
        prediction, alpha = global_positive_line_search(
            current, proposal, self.minimum_height
        )
        self.last_alpha = alpha.detach()
        if self.head.call_count != 1:
            raise RuntimeError("SWE multiscale phase closure must be queried once")
        return prediction


class SWEWaveFluxGateProgram(_SWEProgram):
    """Mass/positivity-safe composition of observable wave mechanisms."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = SWECurrentStateHead(config)
        self.phase_gate = SWEWaveFluxGate(config)
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
        field = self.head(decoded, scalar_history, conditions, condition_mask)
        field = self.toolbox.uniform_mean_zero(field, owner="swe_height")
        current = scalar_history[:, -1]
        previous = scalar_history[:, -2]
        before_previous = scalar_history[:, -3]
        phase_velocity = current - previous
        phase_acceleration = current - 2.0 * previous + before_previous
        phase_chart = SWEPhaseAwareDetailHead.phase_chart(scalar_history)
        phase_flux = phase_chart[:, 4] * observable_step_scale(scalar_history)[:, 0]
        gates = self.phase_gate(decoded, scalar_history)
        learned_increment = field * observable_step_scale(scalar_history)[:, 0]
        phase_motion = self.toolbox.add_terms(
            gates[:, 0] * phase_velocity,
            gates[:, 1] * phase_acceleration,
            owner="swe_height",
        )
        wave_increment = self.toolbox.add_terms(
            phase_motion,
            gates[:, 2] * phase_flux,
            owner="swe_height",
        )
        increment = self.toolbox.add_terms(
            learned_increment, wave_increment, owner="swe_height"
        )
        increment = self.toolbox.uniform_mean_zero(increment, owner="swe_height")
        proposal = self.toolbox.affine_update(
            current, increment, 1.0, owner="swe_height"
        )
        prediction, alpha = global_positive_line_search(
            current, proposal, self.minimum_height
        )
        self.last_alpha = alpha.detach()
        if self.head.call_count != 1:
            raise RuntimeError("SWE wave-flux closure must be queried once")
        return prediction


class CNSProgram(_CNSProgram):
    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = CNSCurrentStateHead(config)


class CFDBProgram(_CFDBProgram):
    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = CFDBDirectObservationHead(config)

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
        next_velocity = self.head(decoded, current, program_codes, conditions, condition_mask)
        prediction = current.clone()
        prediction[..., :2] = next_velocity
        if self.head.call_count != 1:
            raise RuntimeError("CFDB direct observation closure must be queried once")
        return prediction


class CFDBMaskedVelocityProgram(_CFDBMaskedVelocityProgram):
    """Width-scaled CFDB finite-step action with public-mask execution."""

    def __init__(self, toolbox, config: FoundationScaleConfig) -> None:
        super().__init__(toolbox)
        self.head = CFDBDirectObservationHead(config)


__all__ = [
    "CFDBMaskedVelocityProgram",
    "CFDBProgram",
    "CNSProgram",
    "DRDirectObservationProgram",
    "DRProgram",
    "DRKnownSourceResidualProgram",
    "FNOInvariantTransportProgram",
    "FNOAnalyticGateProgram",
    "FNOProgram",
    "FNOSourceAlignedSubstep4Program",
    "FNOStageChartProgram",
    "SWEDirectObservationProgram",
    "SWEPhaseAwareProgram",
    "SWEPhaseCarryProgram",
    "SWELocalPhaseGateProgram",
    "SWEProgram",
]
