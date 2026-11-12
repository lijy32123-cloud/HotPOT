from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


_NEUMANN_HEAT_DECAY_CACHE: dict[tuple[object, ...], Tensor] = {}


@dataclass(frozen=True)
class DRContract:
    diffusivity_u: float = 1.0e-3
    diffusivity_v: float = 5.0e-3
    reaction_k: float = 5.0e-3
    observation_dt: float = 5.0e-2
    x_left: float = -1.0
    x_right: float = 1.0
    y_bottom: float = -1.0
    y_top: float = 1.0
    height: int = 128
    width: int = 128

    @property
    def dx(self) -> float:
        return (self.x_right - self.x_left) / self.width

    @property
    def dy(self) -> float:
        return (self.y_top - self.y_bottom) / self.height

    def diffusivity_tensor(self, reference: Tensor) -> Tensor:
        return reference.new_tensor((self.diffusivity_u, self.diffusivity_v))


def validate_dr_state(state: Tensor, contract: DRContract | None = None) -> None:
    if state.ndim != 4 or state.shape[-1] != 2 or not state.is_floating_point():
        raise ValueError("state must be a floating [B,H,W,2] tensor")
    if contract is not None and state.shape[1:3] != (contract.height, contract.width):
        raise ValueError("state grid and DR contract disagree")


def source_reaction_rhs(state: Tensor, reaction_k: float | Tensor) -> Tensor:
    """Authoritative FitzHugh--Nagumo pointwise source law."""

    validate_dr_state(state)
    k = torch.as_tensor(reaction_k, device=state.device, dtype=state.dtype)
    if k.numel() == 1:
        k = k.reshape(1, 1, 1)
    elif k.ndim == 1 and k.shape[0] == state.shape[0]:
        k = k[:, None, None]
    else:
        raise ValueError("reaction_k must be scalar or [B]")
    u, v = state[..., 0], state[..., 1]
    return torch.stack((u - u.pow(3) - k - v, u - v), dim=-1)


def neumann_laplacian(field: Tensor, *, dx: float, dy: float) -> Tensor:
    if field.ndim != 4 or not field.is_floating_point() or dx <= 0.0 or dy <= 0.0:
        raise ValueError("field must be floating [B,H,W,C] on a positive grid")
    value = field.permute(0, 3, 1, 2)
    padded = torch.nn.functional.pad(value, (1, 1, 1, 1), mode="replicate")
    laplacian = (
        (padded[..., 1:-1, :-2] - 2.0 * value + padded[..., 1:-1, 2:])
        / (dx * dx)
        + (padded[..., :-2, 1:-1] - 2.0 * value + padded[..., 2:, 1:-1])
        / (dy * dy)
    )
    return laplacian.permute(0, 2, 3, 1)


def exact_discrete_neumann_heat(
    state: Tensor,
    diffusivity: Tensor,
    duration: float | Tensor,
    *,
    dx: float,
    dy: float,
) -> Tensor:
    validate_dr_state(state)
    diffusion = torch.as_tensor(diffusivity, device=state.device, dtype=state.dtype)
    time = torch.as_tensor(duration, device=state.device, dtype=state.dtype)
    if diffusion.shape != (2,) or bool((diffusion < 0).any()):
        raise ValueError("diffusivity must be nonnegative [2]")
    if time.numel() != 1 or bool((time < 0).any()) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("invalid heat-flow duration or grid")
    value = state.permute(0, 3, 1, 2)
    height, width = value.shape[-2:]
    extended = torch.cat((value, value.flip(-1)), dim=-1)
    extended = torch.cat((extended, extended.flip(-2)), dim=-2)
    cacheable = not diffusion.requires_grad and not time.requires_grad
    cache_key: tuple[object, ...] | None = None
    decay: Tensor | None = None
    if cacheable:
        cache_key = (
            str(state.device),
            state.dtype,
            height,
            width,
            float(dx),
            float(dy),
            float(diffusion[0].detach().cpu()),
            float(diffusion[1].detach().cpu()),
            float(time.detach().cpu()),
        )
        decay = _NEUMANN_HEAT_DECAY_CACHE.get(cache_key)
    if decay is None:
        fy = torch.arange(2 * height, device=state.device, dtype=state.dtype)
        fx = torch.arange(2 * width, device=state.device, dtype=state.dtype)
        eigen_y = -4.0 * torch.sin(torch.pi * fy / (2.0 * height)).square() / (dy * dy)
        eigen_x = -4.0 * torch.sin(torch.pi * fx / (2.0 * width)).square() / (dx * dx)
        eigenvalue = eigen_y[:, None] + eigen_x[None, :]
        decay = torch.exp(
            time.reshape(()) * diffusion[:, None, None] * eigenvalue[None]
        )
        if cache_key is not None:
            _NEUMANN_HEAT_DECAY_CACHE[cache_key] = decay.detach()
    spectrum = torch.fft.fft2(extended, dim=(-2, -1))
    evolved = torch.fft.ifft2(spectrum * decay[None], dim=(-2, -1)).real
    return evolved[..., :height, :width].permute(0, 2, 3, 1)


def clear_physics_caches() -> None:
    _NEUMANN_HEAT_DECAY_CACHE.clear()


def kolmogorov_forcing(
    size: int,
    *,
    amplitude: float = 0.1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    if size < 2:
        raise ValueError("forcing grid must be at least 2x2")
    grid = torch.arange(size, device=device, dtype=dtype) / float(size)
    phase = 2.0 * math.pi * (grid[:, None] + grid[None, :])
    return float(amplitude) * (torch.sin(phase) + torch.cos(phase))


__all__ = [
    "DRContract",
    "clear_physics_caches",
    "exact_discrete_neumann_heat",
    "kolmogorov_forcing",
    "neumann_laplacian",
    "source_reaction_rhs",
    "validate_dr_state",
]
