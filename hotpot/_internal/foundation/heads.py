from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hotpot._internal.data_protocol.cns_specialist import (
    PeriodicMarkovConfig,
    _current_chart,
)
from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
    ProgramId,
)
from hotpot._internal.data_protocol.native_charts import (
    fno_native64_exact_left_inverse,
)
from hotpot._internal.data_protocol.physics import (
    kolmogorov_forcing,
)
from hotpot._internal.data_protocol.swe_specialist import (
    observable_step_scale,
)
from hotpot._internal.data_protocol.production_primitives import (
    SpectralResidualBlock,
)
from hotpot._internal.spectral.shared_advection import (
    periodic_vorticity_geometry,
)
from hotpot._internal.operators.experts import PROGRAM_CODE

from .config import FoundationScaleConfig


def _coordinates(reference: Tensor) -> Tensor:
    batch, _channels, height, width = reference.shape
    y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0)[None].expand(batch, -1, -1, -1)


class ConditionFeature(nn.Module):
    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        self.channels = int(config.condition_feature_channels)
        self.net = nn.Sequential(
            nn.Linear(2 * len(CONDITION_FIELDS), config.condition_feature_hidden),
            nn.GELU(),
            nn.Linear(config.condition_feature_hidden, self.channels),
        )

    def forward(
        self,
        conditions: Tensor,
        mask: Tensor,
        *,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> Tensor:
        payload = torch.cat((conditions * mask, mask), dim=1).to(dtype)
        return self.net(payload)[:, :, None, None].expand(-1, -1, height, width)


class _SpatialHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden: int,
        padding_mode: str,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


def _head(config: FoundationScaleConfig, extra: int, out: int, padding: str) -> _SpatialHead:
    return _SpatialHead(
        config.decoded_channels + extra + config.condition_feature_channels,
        out,
        hidden=config.typed_head_hidden,
        padding_mode=padding,
    )


class FNOCurrentStateHead(nn.Module):
    reads_full_history = False

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        self.condition = ConditionFeature(config)
        self.head = _head(config, 1 + 2, 1, "circular")
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(self, decoded: Tensor, current: Tensor, conditions: Tensor, mask: Tensor) -> Tensor:
        self.call_count += 1
        context = fno_native64_exact_left_inverse(decoded)
        condition = self.condition(conditions, mask, height=64, width=64, dtype=decoded.dtype)
        return self.head(torch.cat((context, current, condition, _coordinates(current)), dim=1))


def fno_stage_transport_rate(current: Tensor) -> Tensor:
    """Evaluate the source-matched inviscid vorticity rate at one stage."""

    velocity_x, velocity_y, gradient_x, gradient_y = periodic_vorticity_geometry(
        current
    )
    return -(velocity_x * gradient_x + velocity_y * gradient_y)


class FNOSourceAlignedStageChartHead(FNOCurrentStateHead):
    """Instantaneous-rate head with source-aligned stage coordinates.

    The two generic absolute-coordinate channels are replaced, at identical
    width, by the current-stage Biot--Savart transport rate and the known
    Kolmogorov forcing. The shared H10 representation and action ABI remain
    unchanged.
    """

    action_abi = "instantaneous_transport_rate_v1"
    stage_chart_channels = 2

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__(config)
        self.register_buffer(
            "source_forcing_chart",
            kolmogorov_forcing(64, dtype=torch.float32)[None, None],
            persistent=False,
        )

    def stage_chart(self, current: Tensor) -> Tensor:
        transport = fno_stage_transport_rate(current)
        forcing = self.source_forcing_chart.to(
            device=current.device, dtype=current.dtype
        ).expand_as(current)
        return torch.cat((transport, forcing), dim=1)

    def forward(
        self,
        decoded: Tensor,
        current: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        context = fno_native64_exact_left_inverse(decoded)
        condition = self.condition(
            conditions,
            mask,
            height=64,
            width=64,
            dtype=decoded.dtype,
        )
        return self.head(
            torch.cat((context, current, condition, self.stage_chart(current)), dim=1)
        )


def _shared_dilated_update(
    value: Tensor,
    convolution: nn.Conv2d,
    *,
    dilation: int,
    padding_mode: str,
) -> Tensor:
    """Reuse one trained 3x3 kernel at a wider physical scale."""

    padded = F.pad(
        value,
        (dilation, dilation, dilation, dilation),
        mode=padding_mode,
    )
    return F.conv2d(
        padded,
        convolution.weight,
        convolution.bias,
        dilation=dilation,
    )


class FNORecurrentMultiscaleHead(FNOCurrentStateHead):
    """Parameter-tied periodic action depth with zero-start multiscale reuse."""

    dilations = (2, 4, 8, 16)

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__(config)
        self.scale_logits = nn.Parameter(torch.zeros(len(self.dilations)))

    def forward(
        self,
        decoded: Tensor,
        current: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        context = fno_native64_exact_left_inverse(decoded)
        condition = self.condition(
            conditions, mask, height=64, width=64, dtype=decoded.dtype
        )
        packed = torch.cat(
            (context, current, condition, _coordinates(current)), dim=1
        )
        first = self.head.net[0](packed)
        hidden = self.head.net[1](first)
        hidden = self.head.net[3](self.head.net[2](hidden))
        shared = self.head.net[2]
        for logit, dilation in zip(
            self.scale_logits, self.dilations, strict=True
        ):
            update = _shared_dilated_update(
                hidden, shared, dilation=dilation, padding_mode="circular"
            )
            hidden = hidden + torch.tanh(logit) * F.gelu(update)
        return self.head.net[4](hidden)


class _FNOStageSpectralRefiner(nn.Module):
    """Small native-grid spectral tower inside the typed midpoint action.

    The final projection is zero initialized, so inserting this tower leaves
    an existing checkpoint's function unchanged.  Unlike the one-shot latent
    mixer, the tower is recomputed for every numerical midpoint stage.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        width = max(32, hidden // 2)
        blocks = 4 if width % 4 == 0 else 1
        self.norm = nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden)
        self.down = nn.Conv2d(hidden, width, 1)
        self.blocks = nn.ModuleList(
            SpectralResidualBlock(width, blocks, 2 * width) for _ in range(3)
        )
        self.up = nn.Conv2d(width, hidden, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        value = F.gelu(self.down(self.norm(hidden)))
        for block in self.blocks:
            value = block(value)
        return hidden + self.up(value)


class FNODeepStageSpectralHead(FNOCurrentStateHead):
    """Legacy typed action with native-grid spectral depth at every stage."""

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__(config)
        self.deep_refiner = _FNOStageSpectralRefiner(config.typed_head_hidden)

    def forward(
        self,
        decoded: Tensor,
        current: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        context = fno_native64_exact_left_inverse(decoded)
        condition = self.condition(
            conditions, mask, height=64, width=64, dtype=decoded.dtype
        )
        packed = torch.cat(
            (context, current, condition, _coordinates(current)), dim=1
        )
        hidden = self.head.net[1](self.head.net[0](packed))
        hidden = self.head.net[3](self.head.net[2](hidden))
        hidden = self.deep_refiner(hidden)
        return self.head.net[4](hidden)


def _sample_rms(value: Tensor) -> Tensor:
    return value.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)


class FNOStageChartHead(nn.Module):
    """Stage-local transport head in source-aligned periodic coordinates.

    The shared H10 context remains the only learned history representation.
    At each midpoint query, the current vorticity uniquely determines a
    periodic streamfunction, velocity and vorticity gradient.  We express the
    gradient in the local velocity frame: its parallel component is the
    resolved advective derivative and its perpendicular component retains the
    complementary local geometry.  These two translation-equivariant fields
    replace the legacy absolute x/y coordinates, so the trainable parameter
    count and numerical program remain exactly unchanged.
    """

    reads_full_history = False
    stage_chart_channels = 2

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        self.condition = ConditionFeature(config)
        self.head = _head(
            config,
            1 + self.stage_chart_channels,
            1,
            "circular",
        )
        wave_y = 2.0 * torch.pi * torch.fft.fftfreq(64, d=1.0 / 64.0)
        wave_x = 2.0 * torch.pi * torch.fft.rfftfreq(64, d=1.0 / 64.0)
        self.register_buffer("wave_y", wave_y, persistent=False)
        self.register_buffer("wave_x", wave_x, persistent=False)
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def stage_chart(self, current: Tensor) -> Tensor:
        if current.ndim != 4 or tuple(current.shape[1:]) != (1, 64, 64):
            raise ValueError("FNO stage chart requires [B,1,64,64] vorticity")
        omega = current[:, 0]
        spectrum = torch.fft.rfft2(omega, norm="ortho")
        ky = self.wave_y.to(device=current.device, dtype=current.dtype)[:, None]
        kx = self.wave_x.to(device=current.device, dtype=current.dtype)[None, :]
        minus_laplacian = ky.square() + kx.square()
        inverse = torch.where(
            minus_laplacian > 0.0,
            minus_laplacian.reciprocal(),
            torch.zeros_like(minus_laplacian),
        )
        stream = spectrum * inverse[None]
        velocity_x = torch.fft.irfft2(
            1j * ky[None] * stream, s=omega.shape[-2:], norm="ortho"
        )
        velocity_y = torch.fft.irfft2(
            -1j * kx[None] * stream, s=omega.shape[-2:], norm="ortho"
        )
        gradient_x = torch.fft.irfft2(
            1j * kx[None] * spectrum, s=omega.shape[-2:], norm="ortho"
        )
        gradient_y = torch.fft.irfft2(
            1j * ky[None] * spectrum, s=omega.shape[-2:], norm="ortho"
        )
        parallel_transport = -(
            velocity_x * gradient_x + velocity_y * gradient_y
        )
        perpendicular_transport = -(
            velocity_x * gradient_y - velocity_y * gradient_x
        )
        fields = torch.stack(
            (
                parallel_transport,
                perpendicular_transport,
            ),
            dim=1,
        )
        return fields / _sample_rms(fields)

    def forward(
        self,
        decoded: Tensor,
        current: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        context = fno_native64_exact_left_inverse(decoded)
        condition = self.condition(
            conditions, mask, height=64, width=64, dtype=decoded.dtype
        )
        base = torch.cat((context, current, condition), dim=1)
        return self.head(torch.cat((base, self.stage_chart(current)), dim=1))


class FNOAnalyticGateHead(FNOCurrentStateHead):
    """Legacy neural action plus a zero-start audited transport mechanism."""

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__(config)
        self.analytic_gain = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        decoded: Tensor,
        current: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        learned = super().forward(decoded, current, conditions, mask)
        velocity_x, velocity_y, gradient_x, gradient_y = (
            periodic_vorticity_geometry(current)
        )
        analytic = -(velocity_x * gradient_x + velocity_y * gradient_y)
        return learned + torch.tanh(self.analytic_gain) * analytic


class FNOPhysicsClosureHead(FNOCurrentStateHead):
    """Jointly allocate typed action between source-matched transport and closure.

    The zero fraction exactly recovers the trained neural finite-time action.
    During task-only adaptation the existing head can become a closure while a
    bounded fraction of skew-symmetric, dealiased vorticity transport is moved
    into the numerical side of the typed expert.
    """

    maximum_physics_fraction = 0.5

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__(config)
        modes = torch.fft.fftfreq(64, d=1.0 / 64.0)
        wave = 2.0 * torch.pi * modes
        kx = wave[:, None].expand(64, 64).contiguous()
        ky = wave[None, :].expand(64, 64).contiguous()
        k2 = kx.square() + ky.square()
        dealias = (
            (modes.abs()[:, None] <= 64 // 3)
            & (modes.abs()[None, :] <= 64 // 3)
        )
        self.register_buffer("transport_kx", kx, persistent=False)
        self.register_buffer("transport_ky", ky, persistent=False)
        self.register_buffer("transport_k2", k2, persistent=False)
        self.register_buffer("transport_dealias", dealias, persistent=False)
        self.physics_fraction_logit = nn.Parameter(torch.zeros(()))

    def analytic_transport(self, current: Tensor) -> Tensor:
        omega = current[:, 0]
        omega_hat = torch.fft.fft2(omega)
        kx = self.transport_kx.to(omega)
        ky = self.transport_ky.to(omega)
        k2 = self.transport_k2.to(omega)
        safe = torch.where(k2 == 0.0, torch.ones_like(k2), k2)
        stream = omega_hat / safe
        stream = stream.clone()
        stream[..., 0, 0] = 0.0
        velocity_x = torch.fft.ifft2(1j * ky * stream).real
        velocity_y = torch.fft.ifft2(-1j * kx * stream).real
        derivative_x = torch.fft.ifft2(1j * kx * omega_hat).real
        derivative_y = torch.fft.ifft2(1j * ky * omega_hat).real
        advective = velocity_x * derivative_x + velocity_y * derivative_y
        conservative = (
            torch.fft.ifft2(1j * kx * torch.fft.fft2(velocity_x * omega)).real
            + torch.fft.ifft2(1j * ky * torch.fft.fft2(velocity_y * omega)).real
        )
        transport = -0.5 * (advective + conservative)
        filtered = torch.fft.ifft2(
            torch.fft.fft2(transport) * self.transport_dealias.to(omega)
        ).real
        return filtered[:, None]

    def forward(
        self,
        decoded: Tensor,
        current: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        learned = super().forward(decoded, current, conditions, mask)
        fraction = self.maximum_physics_fraction * torch.tanh(
            self.physics_fraction_logit
        )
        analytic = self.analytic_transport(current)
        return (1.0 - fraction) * learned + fraction * analytic


class FNODeepPhysicsClosureHead(FNODeepStageSpectralHead):
    """Stage-recomputed spectral closure plus source-matched transport split."""

    maximum_physics_fraction = 0.5

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__(config)
        modes = torch.fft.fftfreq(64, d=1.0 / 64.0)
        wave = 2.0 * torch.pi * modes
        kx = wave[:, None].expand(64, 64).contiguous()
        ky = wave[None, :].expand(64, 64).contiguous()
        k2 = kx.square() + ky.square()
        dealias = (
            (modes.abs()[:, None] <= 64 // 3)
            & (modes.abs()[None, :] <= 64 // 3)
        )
        self.register_buffer("transport_kx", kx, persistent=False)
        self.register_buffer("transport_ky", ky, persistent=False)
        self.register_buffer("transport_k2", k2, persistent=False)
        self.register_buffer("transport_dealias", dealias, persistent=False)
        self.physics_fraction_logit = nn.Parameter(torch.zeros(()))

    analytic_transport = FNOPhysicsClosureHead.analytic_transport

    def forward(
        self,
        decoded: Tensor,
        current: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        learned = super().forward(decoded, current, conditions, mask)
        fraction = self.maximum_physics_fraction * torch.tanh(
            self.physics_fraction_logit
        )
        analytic = self.analytic_transport(current)
        return (1.0 - fraction) * learned + fraction * analytic


class FNOSpectralActionHead(FNOCurrentStateHead):
    """Viscosity-conditioned Fourier calibration of the typed FNO action.

    The legacy head still produces the complete midpoint-stage action.  This
    module changes only its periodic spectral transfer function, so it cannot
    bypass the shared H10 representation or write an endpoint residual.  The
    zero parameters give the exact legacy action.  The deterministic Reynolds
    gate is one at nu=1e-5 and zero at nu=1e-3, with continuous interpolation
    between the two released regimes.
    """

    maximum_log_multiplier = 0.25

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__(config)
        self.spectral_real = nn.Parameter(torch.zeros(64, 33))
        self.spectral_imag = nn.Parameter(torch.zeros(64, 33))

    @staticmethod
    def _hermitian_boundary(
        real: Tensor, imaginary: Tensor
    ) -> tuple[Tensor, Tensor]:
        negative_y = torch.remainder(
            -torch.arange(real.shape[0], device=real.device), real.shape[0]
        )
        real_value = real.clone()
        imaginary_value = imaginary.clone()
        for column in (0, real.shape[1] - 1):
            real_value[:, column] = 0.5 * (
                real[:, column] + real[negative_y, column]
            )
            imaginary_value[:, column] = 0.5 * (
                imaginary[:, column] - imaginary[negative_y, column]
            )
        return real_value, imaginary_value

    def forward(
        self,
        decoded: Tensor,
        current: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        action = super().forward(decoded, current, conditions, mask)
        real, imaginary = self._hermitian_boundary(
            self.spectral_real, self.spectral_imag
        )
        viscosity = conditions[:, CONDITION_FIELDS.index("viscosity")].clamp_min(
            1.0e-12
        )
        low_reynolds_gate = ((-torch.log10(viscosity) - 3.0) / 2.0).clamp(0.0, 1.0)
        scale = low_reynolds_gate[:, None, None] * self.maximum_log_multiplier
        multiplier = torch.complex(
            1.0 + scale * torch.tanh(real)[None],
            scale * torch.tanh(imaginary)[None],
        )
        spectrum = torch.fft.rfft2(action[:, 0], norm="ortho")
        calibrated = torch.fft.irfft2(
            spectrum * multiplier, s=action.shape[-2:], norm="ortho"
        )
        return calibrated[:, None]


class FNOHighBandClosureHead(FNOCurrentStateHead):
    """Zero-start high-wavenumber closure for low-viscosity transport.

    The legacy typed action remains the complete learned transport term.  A
    single bounded coefficient corrects only modes above the diagnosed
    under-resolved band and is disabled continuously at the released
    high-viscosity regime.  This changes neither the shared analyzer nor the
    deterministic CN/Strang/midpoint program.
    """

    maximum_gain = 0.25

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__(config)
        self.high_band_logit = nn.Parameter(torch.zeros(()))
        wave_y = torch.fft.fftfreq(64) * 64.0
        wave_x = torch.fft.rfftfreq(64) * 64.0
        ky, kx = torch.meshgrid(wave_y, wave_x, indexing="ij")
        radius = (kx.square() + ky.square()).sqrt()
        self.register_buffer(
            "high_band_window",
            torch.sigmoid((radius - 16.0) / 2.0),
            persistent=False,
        )

    def forward(
        self,
        decoded: Tensor,
        current: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        action = super().forward(decoded, current, conditions, mask)
        viscosity = conditions[:, CONDITION_FIELDS.index("viscosity")].clamp_min(
            1.0e-12
        )
        low_viscosity = ((-torch.log10(viscosity) - 3.0) / 2.0).clamp(0.0, 1.0)
        gain = (
            self.maximum_gain
            * torch.tanh(self.high_band_logit)
            * low_viscosity
        )
        window = self.high_band_window.to(dtype=action.dtype, device=action.device)
        multiplier = 1.0 + gain[:, None, None] * window[None]
        spectrum = torch.fft.rfft2(action[:, 0], norm="ortho")
        corrected = torch.fft.irfft2(
            spectrum * multiplier, s=action.shape[-2:], norm="ortho"
        )
        return corrected[:, None]


class DRCurrentStateHead(nn.Module):
    reads_full_history = False

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        self.condition = ConditionFeature(config)
        self.head = _head(config, 2 + 2, 4, "replicate")
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(self, decoded: Tensor, current_bhwc: Tensor, conditions: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        self.call_count += 1
        current = current_bhwc.permute(0, 3, 1, 2).contiguous()
        condition = self.condition(conditions, mask, height=128, width=128, dtype=decoded.dtype)
        raw = self.head(torch.cat((decoded, current, condition, _coordinates(current)), dim=1))
        return raw[:, :2].permute(0, 2, 3, 1), raw[:, 2:].permute(0, 2, 3, 1)


class DRKnownSourceResidualHead(nn.Module):
    """Width-scalable endpoint discrepancy after the registered DR step."""

    reads_full_history = False

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        self.condition = ConditionFeature(config)
        self.head = _head(config, 2 + 2, 2, "replicate")
        # The deterministic registered source step is the exact initial
        # function; the neural endpoint correction starts identically at zero.
        nn.init.zeros_(self.head.net[-1].weight)
        nn.init.zeros_(self.head.net[-1].bias)
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(
        self,
        decoded: Tensor,
        current_bhwc: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        current = current_bhwc.permute(0, 3, 1, 2).contiguous()
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        return self.head(
            torch.cat((decoded, current, condition, _coordinates(current)), dim=1)
        ).permute(0, 2, 3, 1)


class SWECurrentStateHead(nn.Module):
    reads_full_history = False

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        self.condition = ConditionFeature(config)
        self.head = _head(config, 1 + 2, 1, "replicate")
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(self, decoded: Tensor, history_bthw: Tensor, conditions: Tensor, mask: Tensor) -> Tensor:
        self.call_count += 1
        current = history_bthw[:, -1:]
        condition = self.condition(conditions, mask, height=128, width=128, dtype=decoded.dtype)
        return self.head(torch.cat((decoded, current, condition, _coordinates(current)), dim=1))[:, 0]


class SWERecurrentMultiscaleHead(SWECurrentStateHead):
    """Parameter-tied boundary-aware depth for a saved-step height action."""

    dilations = (2, 4, 8, 16)

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__(config)
        self.scale_logits = nn.Parameter(torch.zeros(len(self.dilations)))

    def forward(
        self,
        decoded: Tensor,
        history_bthw: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        current = history_bthw[:, -1:]
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        packed = torch.cat(
            (decoded, current, condition, _coordinates(current)), dim=1
        )
        first = self.head.net[0](packed)
        hidden = self.head.net[1](first)
        hidden = self.head.net[3](self.head.net[2](hidden))
        shared = self.head.net[2]
        for logit, dilation in zip(
            self.scale_logits, self.dilations, strict=True
        ):
            update = _shared_dilated_update(
                hidden, shared, dilation=dilation, padding_mode="replicate"
            )
            hidden = hidden + torch.tanh(logit) * F.gelu(update)
        return self.head.net[4](hidden)[:, 0]


class SWEPhaseAwareDetailHead(nn.Module):
    """Height-action head with deterministic native-grid phase observables.

    Only the last three observed height fields are read outside the shared H10
    analyzer.  Their use is parameter-free and limited to the saved-step first
    and second differences plus closed-grid spatial differences.  The learned
    history representation still has no private bypass.
    """

    reads_full_history = False
    phase_chart_channels = 5

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        self.condition = ConditionFeature(config)
        self.head = _head(
            config,
            1 + 2 + self.phase_chart_channels,
            1,
            "replicate",
        )
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    @staticmethod
    def phase_chart(history_bthw: Tensor) -> Tensor:
        if history_bthw.ndim != 4 or history_bthw.shape[1] < 3:
            raise ValueError("SWE phase chart requires [B,T>=3,H,W]")
        current = history_bthw[:, -1:]
        previous = history_bthw[:, -2:-1]
        before_previous = history_bthw[:, -3:-2]
        first = current - previous
        second = current - 2.0 * previous + before_previous
        scale = observable_step_scale(history_bthw).clamp_min(1.0e-8)
        normalized_first = first / scale
        normalized_second = second / scale
        padded = F.pad(normalized_first, (1, 1, 1, 1), mode="replicate")
        left = padded[..., 1:-1, :-2]
        right = padded[..., 1:-1, 2:]
        down = padded[..., :-2, 1:-1]
        up = padded[..., 2:, 1:-1]
        gradient_x = 0.5 * (right - left)
        gradient_y = 0.5 * (up - down)
        laplacian = left + right + down + up - 4.0 * normalized_first
        return torch.cat(
            (
                normalized_first,
                normalized_second,
                gradient_x,
                gradient_y,
                laplacian,
            ),
            dim=1,
        )

    def forward(
        self,
        decoded: Tensor,
        history_bthw: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        current = history_bthw[:, -1:]
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        base = torch.cat(
            (decoded, current, condition, _coordinates(current)), dim=1
        )
        return self.head(torch.cat((base, self.phase_chart(history_bthw)), dim=1))[:, 0]


class _SWEFullResolutionWaveRefiner(nn.Module):
    """Boundary-aware native-grid tower conditioned on observable phase."""

    def __init__(self, hidden: int, phase_channels: int = 5) -> None:
        super().__init__()
        width = max(16, hidden // 4)
        groups = 8 if width % 8 == 0 else 1
        self.hidden_projection = nn.Conv2d(hidden, width, 1)
        self.phase_projection = nn.Conv2d(
            phase_channels,
            width,
            3,
            padding=1,
            padding_mode="replicate",
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(groups, width),
                nn.Conv2d(
                    width,
                    width,
                    3,
                    padding=1,
                    groups=width,
                    padding_mode="replicate",
                ),
                nn.GELU(),
                nn.Conv2d(width, width, 1),
            )
            for _ in range(12)
        )
        self.output = nn.Conv2d(width, hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: Tensor, phase: Tensor) -> Tensor:
        value = F.gelu(
            self.hidden_projection(hidden) + self.phase_projection(phase)
        )
        for block in self.blocks:
            value = value + 0.1 * F.gelu(block(value))
        return hidden + self.output(value)


class SWEDeepWaveActionHead(SWECurrentStateHead):
    """Deep full-resolution saved-step action with explicit wave phase."""

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__(config)
        self.deep_refiner = _SWEFullResolutionWaveRefiner(
            config.typed_head_hidden
        )

    def forward(
        self,
        decoded: Tensor,
        history_bthw: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        current = history_bthw[:, -1:]
        condition = self.condition(
            conditions, mask, height=128, width=128, dtype=decoded.dtype
        )
        packed = torch.cat(
            (decoded, current, condition, _coordinates(current)), dim=1
        )
        hidden = self.head.net[1](self.head.net[0](packed))
        hidden = self.head.net[3](self.head.net[2](hidden))
        phase = SWEPhaseAwareDetailHead.phase_chart(history_bthw)
        hidden = self.deep_refiner(hidden, phase)
        return self.head.net[4](hidden)[:, 0]


class SWELocalPhaseGate(nn.Module):
    """Small typed gate over observable saved-step velocity and acceleration."""

    reads_full_history = False

    def __init__(self, config: FoundationScaleConfig, width: int = 32) -> None:
        super().__init__()
        channels = config.decoded_channels + 1 + 2 + 5
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(width, 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, decoded: Tensor, history_bthw: Tensor) -> Tensor:
        current = history_bthw[:, -1:]
        phase = SWEPhaseAwareDetailHead.phase_chart(history_bthw)
        features = torch.cat(
            (decoded, current, _coordinates(current), phase), dim=1
        )
        return torch.tanh(self.net(features))


class SWEWaveFluxGate(nn.Module):
    """Local typed coefficients for velocity, acceleration, and wave flux."""

    reads_full_history = False

    def __init__(self, config: FoundationScaleConfig, width: int = 32) -> None:
        super().__init__()
        channels = config.decoded_channels + 1 + 2 + 5
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(width, 3, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, decoded: Tensor, history_bthw: Tensor) -> Tensor:
        current = history_bthw[:, -1:]
        phase = SWEPhaseAwareDetailHead.phase_chart(history_bthw)
        features = torch.cat(
            (decoded, current, _coordinates(current), phase), dim=1
        )
        return torch.tanh(self.net(features))


class SWEMultiscalePhaseGate(nn.Module):
    """Native-grid phase composer with a 33-cell multiscale receptive field."""

    reads_full_history = False

    def __init__(self, config: FoundationScaleConfig, width: int = 64) -> None:
        super().__init__()
        channels = config.decoded_channels + 1 + 2 + 5
        self.stem = nn.Conv2d(
            channels, width, 3, padding=1, padding_mode="replicate"
        )
        self.blocks = nn.ModuleList(
            nn.Conv2d(
                width,
                width,
                3,
                dilation=dilation,
                padding=dilation,
                padding_mode="replicate",
            )
            for dilation in (1, 2, 4, 8)
        )
        self.output = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, decoded: Tensor, history_bthw: Tensor) -> Tensor:
        current = history_bthw[:, -1:]
        phase = SWEPhaseAwareDetailHead.phase_chart(history_bthw)
        features = torch.cat(
            (decoded, current, _coordinates(current), phase), dim=1
        )
        value = F.gelu(self.stem(features))
        for block in self.blocks:
            value = value + 0.5 * F.gelu(block(value))
        return torch.tanh(self.output(value))


class CNSCurrentStateHead(nn.Module):
    reads_full_history = False

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        self.condition = ConditionFeature(config)
        self.config = PeriodicMarkovConfig()
        self.head = _head(config, 15, 4, "circular")
        self.call_count = 0

    def reset_calls(self) -> None:
        self.call_count = 0

    def forward(self, decoded: Tensor, current_bhwc: Tensor, conditions: Tensor, mask: Tensor) -> Tensor:
        self.call_count += 1
        chart, _scale = _current_chart(current_bhwc, self.config)
        condition = self.condition(conditions, mask, height=128, width=128, dtype=decoded.dtype)
        return self.head(torch.cat((decoded, chart, condition), dim=1))


class _CFDBFamilyObservationHead(nn.Module):
    reads_full_history = False

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        self.condition = ConditionFeature(config)
        self.head = _head(config, 4 + 2, 2, "replicate")

    def forward(self, decoded: Tensor, current_bhwc: Tensor, conditions: Tensor, mask: Tensor) -> Tensor:
        current = current_bhwc.permute(0, 3, 1, 2).contiguous()
        condition = self.condition(conditions, mask, height=128, width=128, dtype=decoded.dtype)
        return self.head(torch.cat((decoded, current, condition, _coordinates(current)), dim=1)).permute(0, 2, 3, 1)


class CFDBDirectObservationHead(nn.Module):
    reads_full_history = False

    def __init__(self, config: FoundationScaleConfig) -> None:
        super().__init__()
        self.family_heads = nn.ModuleDict(
            {family: _CFDBFamilyObservationHead(config) for family in ("cavity", "cylinder", "tube")}
        )
        self.call_count = 0
        self.last_route_counts = {family: 0 for family in self.family_heads}
        self._codes = {
            "cavity": PROGRAM_CODE[ProgramId.CFDB_CAVITY],
            "cylinder": PROGRAM_CODE[ProgramId.CFDB_CYLINDER],
            "tube": PROGRAM_CODE[ProgramId.CFDB_TUBE],
        }

    def reset_calls(self) -> None:
        self.call_count = 0
        self.last_route_counts = {family: 0 for family in self.family_heads}

    def forward(
        self,
        decoded: Tensor,
        current_bhwc: Tensor,
        program_codes: Tensor,
        conditions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.call_count += 1
        if program_codes.shape != (decoded.shape[0],):
            raise ValueError("CFDB program codes must be [B]")
        prediction = current_bhwc.new_empty(
            current_bhwc.shape[0], current_bhwc.shape[1], current_bhwc.shape[2], 2
        )
        for family, code in self._codes.items():
            index = (program_codes == code).nonzero(as_tuple=False).flatten()
            self.last_route_counts[family] = int(index.numel())
            if index.numel() == 0:
                continue
            value = self.family_heads[family](
                decoded.index_select(0, index),
                current_bhwc.index_select(0, index),
                conditions.index_select(0, index),
                mask.index_select(0, index),
            )
            prediction.index_copy_(0, index, value)
        return prediction


__all__ = [
    "CFDBDirectObservationHead",
    "CNSCurrentStateHead",
    "DRCurrentStateHead",
    "DRKnownSourceResidualHead",
    "FNOCurrentStateHead",
    "FNOSourceAlignedStageChartHead",
    "FNOHighBandClosureHead",
    "FNORecurrentMultiscaleHead",
    "SWECurrentStateHead",
    "SWERecurrentMultiscaleHead",
    "SWEWaveFluxGate",
    "fno_stage_transport_rate",
]
