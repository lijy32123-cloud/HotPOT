from __future__ import annotations

import math
import weakref

import torch
from torch import Tensor, nn

from hotpot._internal.data_protocol.cns_specialist import (
    PeriodicMarkovConfig,
    conservative_to_primitive,
    positivity_segment,
    primitive_to_conservative,
)
from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
)
from hotpot._internal.data_protocol.native_charts import (
    fno_native64_exact_left_inverse,
)
from hotpot._internal.data_protocol.physics import (
    DRContract,
    exact_discrete_neumann_heat,
    kolmogorov_forcing,
    neumann_laplacian,
)
from hotpot._internal.data_protocol.swe_specialist import (
    global_positive_line_search,
    observable_step_scale,
)
from hotpot._internal.operators.heads import (
    CFDBObservationFlowHead,
    CNSConservativeHead,
    DRClosureHead,
    FNOTransportHead,
    SWEControlHead,
)

from .toolbox import CanonicalNumericalToolbox


_CONDITION = {name: index for index, name in enumerate(CONDITION_FIELDS)}


def _uniform(conditions: Tensor, name: str) -> float:
    values = conditions[:, _CONDITION[name]]
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError(f"missing finite condition {name}")
    reference = values[0]
    if not bool(
        torch.allclose(values, reference.expand_as(values), rtol=0.0, atol=1.0e-8)
    ):
        raise ValueError(f"condition {name} must be uniform within one program group")
    return float(reference.detach().cpu())


class _ToolboxBoundProgram(nn.Module):
    def __init__(self, toolbox: CanonicalNumericalToolbox) -> None:
        super().__init__()
        self._toolbox_ref = weakref.ref(toolbox)

    @property
    def toolbox(self) -> CanonicalNumericalToolbox:
        value = self._toolbox_ref()
        if value is None:
            raise RuntimeError("canonical toolbox owner no longer exists")
        return value


class FNOProgram(_ToolboxBoundProgram):
    """Exact V2 CN/Strang/midpoint program through shared combinators."""

    source_internal_dt = 1.0e-4

    def __init__(self, toolbox: CanonicalNumericalToolbox) -> None:
        super().__init__(toolbox)
        self.head = FNOTransportHead()
        self.register_buffer(
            "forcing",
            kolmogorov_forcing(64, dtype=torch.float32)[None, None],
            persistent=True,
        )
        wave = torch.fft.fftfreq(64, d=1.0 / 64.0)
        wave_r = torch.fft.rfftfreq(64, d=1.0 / 64.0)
        eigenvalue = (2.0 * math.pi) ** 2 * (
            wave[:, None].square() + wave_r[None, :].square()
        )
        self.register_buffer("minus_laplacian_eigenvalue", eigenvalue, persistent=True)

    def _diffuse(self, current: Tensor, viscosity: Tensor, duration: float) -> Tensor:
        microsteps = int(round(duration / self.source_internal_dt))
        if microsteps <= 0 or not math.isclose(
            microsteps * self.source_internal_dt,
            duration,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise ValueError("FNO duration must align with the source internal step")
        nu = viscosity.to(device=current.device, dtype=current.dtype)[:, None, None]
        eigenvalue = self.minus_laplacian_eigenvalue.to(current.dtype)[None]
        half = 0.5 * self.source_internal_dt * nu * eigenvalue
        if bool((half >= 1.0).any()):
            raise ValueError("source CN numerator is non-positive")
        multiplier = torch.exp(
            microsteps * (torch.log1p(-half) - torch.log1p(half))
        )
        spectrum = torch.fft.rfft2(current[:, 0], norm="ortho")
        return torch.fft.irfft2(
            spectrum * multiplier, s=current.shape[-2:], norm="ortho"
        )[:, None]

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
        dt = _uniform(conditions, "observation_dt")
        step = dt / 2.0
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


class DRProgram(_ToolboxBoundProgram):
    def __init__(self, toolbox: CanonicalNumericalToolbox) -> None:
        super().__init__(toolbox)
        self.head = DRClosureHead()
        self.contract = DRContract()
        self.diffusion_correction_limit = 0.05

    def _rhs(
        self,
        decoded: Tensor,
        state: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        diffusion_raw, reaction_raw = self.head(
            decoded, state, conditions, condition_mask
        )
        state_scale = state.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1.0e-3)
        diffusion_correction = (
            self.diffusion_correction_limit * state_scale * torch.tanh(diffusion_raw)
        )
        pointwise_amplitude = state.abs().amax(dim=-1, keepdim=True)
        reaction_scale = 1.0 + pointwise_amplitude + pointwise_amplitude.pow(3)
        reaction_rate = reaction_scale * torch.tanh(reaction_raw)
        diffusion_rate = neumann_laplacian(
            diffusion_correction, dx=self.contract.dx, dy=self.contract.dy
        ) * self.contract.diffusivity_tensor(state)
        return self.toolbox.add_terms(
            diffusion_rate, reaction_rate, owner="dr_reaction_diffusion"
        )

    def forward(
        self,
        history: Tensor,
        decoded: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        self.head.reset_calls()
        state = history[..., -1, :2].contiguous()
        diffusion = self.contract.diffusivity_tensor(state)

        def half_flow(value: Tensor) -> Tensor:
            return exact_discrete_neumann_heat(
                value,
                diffusion,
                0.5 * self.contract.observation_dt,
                dx=self.contract.dx,
                dy=self.contract.dy,
            )

        def nonlinear_flow(value: Tensor) -> Tensor:
            def rhs(stage: Tensor) -> Tensor:
                return self._rhs(decoded, stage, conditions, condition_mask)

            return self.toolbox.explicit_midpoint(
                value,
                self.contract.observation_dt,
                rhs,
                self.toolbox.affine_update,
                owner="dr_reaction_diffusion",
            )

        state = self.toolbox.strang_split(
            state,
            half_flow,
            nonlinear_flow,
            owner="dr_reaction_diffusion",
        )
        if self.head.call_count != 2:
            raise RuntimeError("DR closure must be queried twice")
        return state


class SWEProgram(_ToolboxBoundProgram):
    def __init__(self, toolbox: CanonicalNumericalToolbox) -> None:
        super().__init__(toolbox)
        self.head = SWEControlHead()
        self.minimum_height = 1.0e-6

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
        scale = observable_step_scale(scalar_history)[:, 0]
        proposal = self.toolbox.affine_update(
            current, field, scale, owner="swe_height"
        )
        prediction, _alpha = global_positive_line_search(
            current, proposal, self.minimum_height
        )
        if self.head.call_count != 1:
            raise RuntimeError("SWE closure must be queried once")
        return prediction


class CNSProgram(_ToolboxBoundProgram):
    def __init__(self, toolbox: CanonicalNumericalToolbox) -> None:
        super().__init__(toolbox)
        self.head = CNSConservativeHead()
        self.config = PeriodicMarkovConfig()

    def forward(
        self,
        history: Tensor,
        decoded: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        self.head.reset_calls()
        current_public = history[..., -1, :4].contiguous()
        action = self.head(decoded, current_public, conditions, condition_mask)
        action = self.toolbox.uniform_mean_zero(action, owner="cns_compressible")
        current = primitive_to_conservative(current_public, self.config.gamma)
        centered = current - current.mean(dim=(-2, -1), keepdim=True)
        state_scale = centered.square().mean(
            dim=(-2, -1), keepdim=True
        ).sqrt().clamp_min(self.config.chart_floor)
        candidate = self.toolbox.affine_update(
            current, action, state_scale, owner="cns_compressible"
        )
        candidate = self.toolbox.restore_uniform_mean(
            current, candidate, owner="cns_compressible"
        )
        next_state, _theta = positivity_segment(
            current.double(), candidate.double(), self.config
        )
        prediction = conservative_to_primitive(
            next_state, self.config.gamma, self.config.density_floor
        ).to(history.dtype)
        if self.head.call_count != 1:
            raise RuntimeError("CNS closure must be queried once")
        return prediction


class CFDBProgram(_ToolboxBoundProgram):
    def __init__(self, toolbox: CanonicalNumericalToolbox) -> None:
        super().__init__(toolbox)
        self.head = CFDBObservationFlowHead()

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
        rate = self.head(
            decoded, current, program_codes, conditions, condition_mask
        )
        dt = conditions[:, _CONDITION["observation_dt"]][:, None, None, None]
        prediction = current.clone()
        prediction[..., :2] = self.toolbox.affine_update(
            current[..., :2], rate, dt, owner="cfdb_observation_flow"
        )
        if self.head.call_count != 1:
            raise RuntimeError("CFDB closure must be queried once")
        return prediction


__all__ = ["CFDBProgram", "CNSProgram", "DRProgram", "FNOProgram", "SWEProgram"]
