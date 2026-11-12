from types import MethodType

RUNTIME_VERSION = "hotpot.nodiff_chart.skip_differences.v1"
RETAINED_FEATURE_INDICES = (0, 6, 7, 8, 9, 10)


def _assemble(retained, validity, *, resolution):
    """Mask/resample only the six retained channels, then restore the 11 ABI."""
    import torch
    import torch.nn.functional as F

    kept = torch.stack(retained, dim=3) * validity[:, :, :, None]
    batch, fields, steps, channels, height, width = kept.shape
    if (height, width) != (resolution, resolution):
        kept = F.interpolate(
            kept.reshape(batch * fields * steps, channels, height, width),
            size=(resolution, resolution), mode="bilinear", align_corners=False,
        ).reshape(batch, fields, steps, channels, resolution, resolution)
    features = kept.new_zeros(batch, fields, steps, 11, resolution, resolution)
    features[:, :, :, :1] = kept[:, :, :, :1]
    features[:, :, :, 6:] = kept[:, :, :, 1:]
    return features


def _summaries(history, config, dynamic_channels, mean, scale):
    import torch

    batch = history.shape[0]
    field_mean = history.new_zeros(batch, config.max_fields)
    field_log_scale = history.new_zeros(batch, config.max_fields)
    field_mean[:, :dynamic_channels] = mean[:, :, 0, 0, 0]
    field_log_scale[:, :dynamic_channels] = scale[:, :, 0, 0, 0].log()
    relative_lags = torch.linspace(
        -1.0, 0.0, config.history_steps, device=history.device, dtype=history.dtype,
    )[None].expand(batch, -1)
    return field_mean, field_log_scale, relative_lags


def _base6_forward(self, history):
    import torch
    from hotpot._internal.encoder.semantic_chart import SemanticHistory

    expected = (self.config.resolution, self.config.resolution, self.config.history_steps, 4)
    if history.ndim != 5 or tuple(history.shape[1:]) != expected:
        raise ValueError(f"public history must be [B,{expected}]")
    batch = history.shape[0]
    dynamic = history[..., :self.dynamic_channels].permute(0, 4, 3, 1, 2)
    dynamic = self._to_native(dynamic)
    mean = dynamic.mean(dim=(2, 3, 4), keepdim=True)
    scale = dynamic.std(dim=(2, 3, 4), keepdim=True).clamp_min(1.0e-6)
    normalized_active = (dynamic - mean) / scale
    normalized = history.new_zeros(
        batch, self.config.max_fields, self.config.history_steps, self.native_size, self.native_size,
    )
    normalized[:, :self.dynamic_channels] = normalized_active
    field_mask = history.new_zeros(batch, self.config.max_fields)
    field_mask[:, :self.dynamic_channels] = 1.0
    xx, yy, boundary_distance = self._coordinates(normalized)
    if self.geometry_channel is None:
        geometry = torch.ones_like(normalized)
    else:
        public_support = history[..., self.geometry_channel].permute(0, 3, 1, 2)
        support = self._to_native(public_support[:, None])[:, 0].mean(dim=1).clamp(0.0, 1.0)
        geometry = support[:, None, None].expand_as(normalized)
    validity = field_mask[:, :, None, None, None].expand_as(normalized)
    features = _assemble(
        (normalized, xx, yy, boundary_distance, geometry, validity), validity,
        resolution=self.config.resolution,
    )
    field_mean, field_log_scale, relative_lags = _summaries(
        history, self.config, self.dynamic_channels, mean, scale,
    )
    return SemanticHistory(features, field_mask, field_mean, field_log_scale, relative_lags)


def _general_forward(self, history):
    """Original GeneralSemanticFieldChart tuple ABI, including Wave's C=1."""
    import torch

    expected = (128, 128, self.config.history_steps, self.dynamic_channels)
    if history.ndim != 5 or tuple(history.shape[1:]) != expected:
        raise ValueError(f"history must be [B,{expected}]")
    batch = history.shape[0]
    dynamic = history.permute(0, 4, 3, 1, 2).contiguous()
    mean = dynamic.mean((2, 3, 4), keepdim=True)
    scale = dynamic.std((2, 3, 4), keepdim=True).clamp_min(1.0e-6)
    normalized_active = (dynamic - mean) / scale
    normalized = history.new_zeros(batch, self.config.max_fields, self.config.history_steps, 128, 128)
    normalized[:, :self.dynamic_channels] = normalized_active
    field_mask = history.new_zeros(batch, self.config.max_fields)
    field_mask[:, :self.dynamic_channels] = 1.0
    axis = torch.linspace(-1.0, 1.0, 128, device=history.device, dtype=history.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    distance = torch.ones_like(xx) if self.boundary == "periodic" else torch.minimum(
        1.0 - xx.abs(), 1.0 - yy.abs(),
    ).clamp_min(0.0)
    shape = (1, 1, 1, 128, 128)
    xx = xx.reshape(shape).expand(normalized.shape)
    yy = yy.reshape(shape).expand(normalized.shape)
    distance = distance.reshape(shape).expand(normalized.shape)
    geometry = torch.ones_like(normalized)
    validity = field_mask[:, :, None, None, None].expand_as(normalized)
    features = _assemble((normalized, xx, yy, distance, geometry, validity), validity, resolution=128)
    field_mean, field_log_scale, relative_lags = _summaries(
        history, self.config, self.dynamic_channels, mean, scale,
    )
    return features, field_mask, field_mean, field_log_scale, relative_lags


def install_no_difference_chart(wrapper):
    """Patch all supported chart instances in-place; return ``wrapper``.

    Supports Base6 wrappers/models, Wave's single GeneralSemanticFieldChart,
    and standalone chart modules. Installation is idempotent and validates all
    candidates before any mutation. Unknown chart subclasses/forward overrides
    fail instead of silently losing their semantics. Caller installs only in
    the new runtime process, before compiled execution or training.
    """
    from hotpot._internal.encoder.semantic_chart import SemanticFieldChart
    from hotpot._internal.general_fields.model import GeneralSemanticFieldChart

    replacements = {SemanticFieldChart: _base6_forward, GeneralSemanticFieldChart: _general_forward}
    candidates = []
    for module in wrapper.modules():
        if not isinstance(module, tuple(replacements)):
            continue
        replacement = replacements.get(type(module))
        if replacement is None:
            raise ValueError("Unsupported semantic chart subclass")
        if module.config.semantic_channels != 11 or list(module.parameters()) or list(module.buffers()):
            raise ValueError("Expected the parameter-free, buffer-free 11-channel chart")
        original = getattr(module.forward, "__func__", None)
        if original not in (type(module).forward, replacement):
            raise ValueError("Unknown chart forward override; refusing to bypass it")
        candidates.append((module, replacement))
    if not candidates:
        raise ValueError("No supported Base6 or General/Wave semantic chart found")
    for module, replacement in candidates:
        module.forward = MethodType(replacement, module)
    return wrapper


__all__ = ["RUNTIME_VERSION", "RETAINED_FEATURE_INDICES", "install_no_difference_chart"]
