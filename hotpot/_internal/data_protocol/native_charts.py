from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _axis(axis: int, ndim: int) -> int:
    value = axis + ndim if axis < 0 else axis
    if value < 0 or value >= ndim:
        raise IndexError(f"axis {axis} is invalid for rank {ndim}")
    return value


def _invert_official_2x_axis(field: Tensor, axis: int) -> Tensor:
    axis = _axis(axis, field.ndim)
    if field.shape[axis] != 128:
        raise ValueError("official FNO inverse requires a length-128 public axis")
    moved = field.movedim(axis, -1)
    first = moved[..., 1::4]
    second = moved[..., 2::4]
    even = 1.5 * first - 0.5 * second
    odd = -0.5 * first + 1.5 * second
    return torch.stack((even, odd), dim=-1).flatten(-2).movedim(-1, axis)


def fno_native64_exact_left_inverse(
    field: Tensor,
    *,
    spatial_dims: Sequence[int] = (-2, -1),
) -> Tensor:
    """Invert the official bilinear 64->128 lift on its exact range.

    With official post-lift training noise this remains a deterministic linear
    restriction of the exact same noisy public tensor seen by MoE-POT; it does
    not expose any extra sample information.
    """

    if len(spatial_dims) != 2:
        raise ValueError("FNO chart requires exactly two spatial axes")
    axes = tuple(_axis(value, field.ndim) for value in spatial_dims)
    if axes[0] == axes[1]:
        raise ValueError("FNO spatial axes must differ")
    sizes = tuple(field.shape[value] for value in axes)
    if sizes == (64, 64):
        return field
    if sizes != (128, 128):
        raise ValueError(f"FNO chart expects 64x64 or 128x128, received {sizes}")
    result = field
    for axis in axes:
        result = _invert_official_2x_axis(result, axis)
    return result


__all__ = ["fno_native64_exact_left_inverse"]
