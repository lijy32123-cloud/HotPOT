from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .factorized_core import FactorizedCoreConfig, HistoryContext
from .production_primitives import SpectralResidualBlock


def _retained_frequency_indices(
    size: int, radius: int, *, device: torch.device
) -> Tensor:
    """Return ``0..radius-1,-radius+1..-1`` in FFT storage order.

    The even-grid Nyquist frequency is deliberately omitted.  It has no unique
    sign under an even-to-even resize and is the only frequency that otherwise
    requires phase-dependent splitting.
    """

    positive = torch.arange(radius, device=device)
    negative = torch.arange(size - radius + 1, size, device=device)
    return torch.cat((positive, negative))


def periodic_spectral_resize(value: Tensor, size: tuple[int, int]) -> Tensor:
    """Band-limited periodic resize with amplitude-preserving FFT conventions.

    Frequencies represent the same integer Fourier modes before and after the
    resize.  ``norm='forward'`` makes the copied coefficients resolution
    independent, so constants and resolved sinusoid amplitudes are preserved.
    The target Nyquist row and column stay zero, making the result real without
    an arbitrary Nyquist phase convention.
    """

    if value.ndim != 4:
        raise ValueError("periodic_spectral_resize expects [B,C,H,W]")
    source_height, source_width = value.shape[-2:]
    target_height, target_width = (int(size[0]), int(size[1]))
    if (source_height, source_width) == (target_height, target_width):
        return value
    dimensions = (source_height, source_width, target_height, target_width)
    if any(dimension < 4 or dimension % 2 for dimension in dimensions):
        raise ValueError(f"periodic spectral resize requires even sizes >=4: {dimensions}")

    radius_y = min(source_height, target_height) // 2
    radius_x = min(source_width, target_width) // 2
    source_y = _retained_frequency_indices(
        source_height, radius_y, device=value.device
    )
    source_x = _retained_frequency_indices(
        source_width, radius_x, device=value.device
    )
    target_y = _retained_frequency_indices(
        target_height, radius_y, device=value.device
    )
    target_x = _retained_frequency_indices(
        target_width, radius_x, device=value.device
    )

    spectrum = torch.fft.fft2(value, dim=(-2, -1), norm="forward")
    retained = spectrum.index_select(-2, source_y).index_select(-1, source_x)
    resized = spectrum.new_zeros(
        (*spectrum.shape[:-2], target_height, target_width)
    )
    grid_y, grid_x = torch.meshgrid(target_y, target_x, indexing="ij")
    resized[..., grid_y, grid_x] = retained
    reconstructed = torch.fft.ifft2(
        resized, dim=(-2, -1), norm="forward"
    )
    return reconstructed.real.contiguous()


class _PeriodicSpectralAnalysisTower(nn.Module):
    """Canonical Fourier restriction followed by learned circular analysis."""

    def __init__(
        self,
        in_channels: int,
        config: FactorizedCoreConfig,
        *,
        depth: int,
    ) -> None:
        super().__init__()
        self.config = config
        # Same tensor shape and initialization order as the production
        # patch projection; only its geometric action is changed.
        self.input_projection = nn.Conv2d(
            in_channels,
            config.width,
            config.patch_size,
            stride=1,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            SpectralResidualBlock(
                config.width, config.num_blocks, config.ffn_hidden
            )
            for _ in range(depth)
        )
        groups = 8 if config.width % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(groups, config.width)

    def forward(self, value: Tensor) -> Tensor:
        value = periodic_spectral_resize(
            value, (self.config.latent_size, self.config.latent_size)
        )
        kernel = self.config.patch_size
        leading = (kernel - 1) // 2
        trailing = kernel - 1 - leading
        value = F.pad(
            value,
            (leading, trailing, leading, trailing),
            mode="circular",
        )
        value = self.input_projection(value)
        for block in self.blocks:
            value = block(value)
        return self.final_norm(value)


class PeriodicSpectralHistoryEncoder(nn.Module):
    """Drop-in history encoder with alias-safe periodic representation."""

    def __init__(self, config: FactorizedCoreConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.analysis = _PeriodicSpectralAnalysisTower(
            config.history_channels, config, depth=config.history_depth
        )

    def forward(self, history: Tensor) -> HistoryContext:
        expected = (
            history.shape[0],
            self.config.history_channels,
            self.config.native_size,
            self.config.native_size,
        )
        if tuple(history.shape) != expected:
            raise ValueError(
                f"packed history must be {expected}, got {tuple(history.shape)}"
            )
        if history.dtype != torch.float32:
            raise TypeError("shared production core is FP32-only")
        scale = history.square().mean((1, 2, 3)).sqrt().clamp_min(1.0e-6)
        latent = self.analysis(history / scale[:, None, None, None])
        return HistoryContext(latent, scale, self.config.native_size)


__all__ = ["PeriodicSpectralHistoryEncoder", "periodic_spectral_resize"]
