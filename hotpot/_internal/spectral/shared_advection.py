from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn


FNO_PROGRAM: Final = "fno_vorticity_ns"
PDEARENA_PROGRAM: Final = "pdearena_smoke_ns"
SUPPORTED_PROGRAMS: Final = (FNO_PROGRAM, PDEARENA_PROGRAM)
MECHANISM_OWNER: Final = "mechanism::shared_scalar_advection"


@dataclass(frozen=True)
class SharedMechanismContract:
    identifier: str = "E_advect_scalar_by_velocity"
    mathematical_object: str = "-u_dot_grad_q"
    support: tuple[str, ...] = SUPPORTED_PROGRAMS
    learned_quantity: str = "dimensionless_local_advection_residual"
    boundary_owned_by_program_adapter: bool = True
    output_head_shared: bool = True
    learned_router_present: bool = False
    abstain: tuple[str, ...] = (
        "cns_compressible",
        "swe_height",
        "dr_reaction_diffusion",
        "cfdb_cavity",
        "cfdb_cylinder",
        "cfdb_tube",
    )


def _frequency(size: int, length: float, reference: Tensor) -> Tensor:
    values = 2.0 * torch.pi * torch.fft.fftfreq(
        size,
        d=float(length) / size,
        device=reference.device,
        dtype=reference.dtype,
    )
    # The Nyquist derivative of a real even-length grid is ambiguous.  The
    
    if size % 2 == 0:
        values[size // 2] = 0.0
    return values


def periodic_vorticity_geometry(
    vorticity: Tensor,
    *,
    length_x: float = 1.0,
    length_y: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return velocity and vorticity gradients for ``q=-Delta psi``.

    Inputs and outputs have shape ``[B,1,H,W]``.  Spatial axis conventions
    match the already-audited FNO vorticity geometry utility.
    """

    if vorticity.ndim != 4 or vorticity.shape[1] != 1:
        raise ValueError("periodic vorticity must be [B,1,H,W]")
    q = vorticity[:, 0]
    height, width = q.shape[-2:]
    spectrum = torch.fft.fft2(q, dim=(-2, -1), norm="ortho")
    # Tensor layout is [H(y), W(x)].  Keeping component and derivative names
    # in physical x/y order is essential because the same canonical channels
    # are consumed by the bounded PDEArena adapter.
    ky = _frequency(height, length_y, q)[:, None]
    kx = _frequency(width, length_x, q)[None, :]
    squared = kx.square() + ky.square()
    inverse = torch.where(squared > 0, squared.reciprocal(), torch.zeros_like(squared))
    stream = spectrum * inverse
    velocity_x = torch.fft.ifft2(
        stream * (1j * ky), dim=(-2, -1), norm="ortho"
    ).real
    velocity_y = torch.fft.ifft2(
        stream * (-1j * kx), dim=(-2, -1), norm="ortho"
    ).real
    gradient_x = torch.fft.ifft2(
        spectrum * (1j * kx), dim=(-2, -1), norm="ortho"
    ).real
    gradient_y = torch.fft.ifft2(
        spectrum * (1j * ky), dim=(-2, -1), norm="ortho"
    ).real
    return tuple(
        value[:, None]
        for value in (velocity_x, velocity_y, gradient_x, gradient_y)
    )  # type: ignore[return-value]


def bounded_scalar_geometry(
    scalar: Tensor,
    velocity_x: Tensor,
    velocity_y: Tensor,
    dx: Tensor,
    dy: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Program-owned closed-box derivatives for PDEArena smoke."""

    if scalar.ndim != 4 or scalar.shape[1] != 1:
        raise ValueError("bounded scalar must be [B,1,H,W]")
    if velocity_x.shape != scalar.shape or velocity_y.shape != scalar.shape:
        raise ValueError("bounded scalar and velocity shapes must agree")
    if dx.shape != (scalar.shape[0],) or dy.shape != (scalar.shape[0],):
        raise ValueError("dx and dy must be [B]")
    padded_x = F.pad(scalar, (1, 1, 0, 0), mode="replicate")
    padded_y = F.pad(scalar, (0, 0, 1, 1), mode="replicate")
    gradient_x = (padded_x[..., 2:] - padded_x[..., :-2]) / (
        2.0 * dx[:, None, None, None]
    )
    gradient_y = (padded_y[..., 2:, :] - padded_y[..., :-2, :]) / (
        2.0 * dy[:, None, None, None]
    )
    return velocity_x, velocity_y, gradient_x, gradient_y


def _rms(value: Tensor) -> Tensor:
    return value.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-8)


class SharedScalarAdvectionMechanism(nn.Module):
    """A single residual core called by two distinct physical programs."""

    def __init__(self, width: int = 128, correction_limit: float = 0.25) -> None:
        super().__init__()
        if width < 8 or correction_limit <= 0.0:
            raise ValueError("invalid shared advection configuration")
        self.width = int(width)
        self.correction_limit = float(correction_limit)
        self.residual_core = nn.Sequential(
            nn.Conv2d(6, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )
        nn.init.zeros_(self.residual_core[-1].weight)
        nn.init.zeros_(self.residual_core[-1].bias)
        self.program_gain = nn.ParameterDict(
            {
                FNO_PROGRAM: nn.Parameter(torch.zeros(())),
                PDEARENA_PROGRAM: nn.Parameter(torch.zeros(())),
            }
        )
        for parameter in self.residual_core.parameters():
            parameter._base12_owner = MECHANISM_OWNER  # type: ignore[attr-defined]
        for program, parameter in self.program_gain.items():
            parameter._base12_owner = f"program::{program}"  # type: ignore[attr-defined]

    @staticmethod
    def contract() -> dict[str, object]:
        payload = asdict(SharedMechanismContract())
        # The ledger is compared again after a JSON round trip at formal
        # launch, so emit one canonical representation rather than tuples that
        # serialize to lists and become spuriously unequal.
        payload["support"] = list(payload["support"])
        payload["abstain"] = list(payload["abstain"])
        return payload

    def _canonical_rate(
        self,
        scalar: Tensor,
        velocity_x: Tensor,
        velocity_y: Tensor,
        gradient_x: Tensor,
        gradient_y: Tensor,
    ) -> Tensor:
        if not all(
            value.shape == scalar.shape
            for value in (velocity_x, velocity_y, gradient_x, gradient_y)
        ):
            raise ValueError("canonical scalar-advection fields must have equal shapes")
        scalar_centered = scalar - scalar.mean((-2, -1), keepdim=True)
        scalar_scale = _rms(scalar_centered)
        velocity_scale = (velocity_x.square() + velocity_y.square()).mean(
            (-2, -1), keepdim=True
        ).sqrt().clamp_min(1.0e-8)
        gradient_scale = (gradient_x.square() + gradient_y.square()).mean(
            (-2, -1), keepdim=True
        ).sqrt().clamp_min(1.0e-8)
        analytic_rate = -(velocity_x * gradient_x + velocity_y * gradient_y)
        rate_scale = _rms(analytic_rate)
        features = torch.cat(
            (
                scalar_centered / scalar_scale,
                velocity_x / velocity_scale,
                velocity_y / velocity_scale,
                gradient_x / gradient_scale,
                gradient_y / gradient_scale,
                analytic_rate / rate_scale,
            ),
            dim=1,
        )
        correction = (
            self.correction_limit
            * rate_scale
            * torch.tanh(self.residual_core(features))
        )
        return analytic_rate + correction

    def fno_rate(self, current: Tensor) -> Tensor:
        velocity_x, velocity_y, gradient_x, gradient_y = periodic_vorticity_geometry(
            current
        )
        rate = self._canonical_rate(
            current, velocity_x, velocity_y, gradient_x, gradient_y
        )
        return self.program_gain[FNO_PROGRAM] * rate

    def pdearena_rate(self, current: Tensor, dx: Tensor, dy: Tensor) -> Tensor:
        if current.ndim != 4 or current.shape[1] != 3:
            raise ValueError("PDEArena state must be [B,3,H,W]")
        scalar = current[:, :1]
        geometry = bounded_scalar_geometry(
            scalar,
            current[:, 1:2],
            current[:, 2:3],
            dx,
            dy,
        )
        scalar_rate = self._canonical_rate(scalar, *geometry)
        output = torch.zeros_like(current)
        output[:, :1] = self.program_gain[PDEARENA_PROGRAM] * scalar_rate
        return output

    def shared_parameter_count(self) -> int:
        return sum(value.numel() for value in self.residual_core.parameters())


__all__ = [
    "FNO_PROGRAM",
    "MECHANISM_OWNER",
    "PDEARENA_PROGRAM",
    "SUPPORTED_PROGRAMS",
    "SharedMechanismContract",
    "SharedScalarAdvectionMechanism",
    "bounded_scalar_geometry",
    "periodic_vorticity_geometry",
]
