from __future__ import annotations

import torch
from torch import Tensor, nn

from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
    ProgramId,
)
from hotpot._internal.operators.experts import (
    PROGRAM_CODE,
)

from .analyzer import AnalysisBlock, _official_afno
from .config import FoundationConfig


CFDB_FAMILY_CODES = {
    "cavity": PROGRAM_CODE[ProgramId.CFDB_CAVITY],
    "cylinder": PROGRAM_CODE[ProgramId.CFDB_CYLINDER],
    "tube": PROGRAM_CODE[ProgramId.CFDB_TUBE],
}


class PrivateTypedFieldMixer(nn.Module):
    """Interpret shared per-field memories inside one typed PDE family.

    The signature intentionally accepts no raw or canonical history.  This is
    the enforceable no-private-H10-reencoder boundary.
    """

    reads_full_history = False

    def __init__(self, config: FoundationConfig, *, dynamic_channels: int) -> None:
        super().__init__()
        config.validate()
        if not 1 <= dynamic_channels <= config.max_fields:
            raise ValueError("dynamic channel count outside field ABI")
        self.config = config
        self.dynamic_channels = int(dynamic_channels)
        self.field_projection = nn.Conv2d(
            self.dynamic_channels * config.shared_width,
            config.private_width,
            1,
        )
        condition_inputs = 2 * len(CONDITION_FIELDS) + 2 * config.max_fields
        self.condition = nn.Sequential(
            nn.Linear(condition_inputs, config.condition_hidden),
            nn.GELU(),
            nn.Linear(config.condition_hidden, config.private_width),
        )
        self.positional = nn.Parameter(
            torch.zeros(1, config.private_width, config.latent_size, config.latent_size)
        )
        nn.init.trunc_normal_(self.positional, std=0.02)
        self.blocks = nn.ModuleList(
            AnalysisBlock(
                config.private_width,
                num_blocks=config.afno_blocks,
                modes=config.afno_modes,
                ffn_hidden=config.private_ffn_hidden,
            )
            for _ in range(config.private_depth)
        )

    def forward(
        self,
        field_memory: Tensor,
        field_mask: Tensor,
        field_mean: Tensor,
        field_log_scale: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        batch = field_memory.shape[0]
        expected = (
            self.config.max_fields,
            self.config.shared_width,
            self.config.latent_size,
            self.config.latent_size,
        )
        if field_memory.ndim != 5 or tuple(field_memory.shape[1:]) != expected:
            raise ValueError("shared field memory ABI mismatch")
        if field_mask.shape != (batch, self.config.max_fields):
            raise ValueError("field mask ABI mismatch")
        if field_mean.shape != field_mask.shape or field_log_scale.shape != field_mask.shape:
            raise ValueError("field statistics ABI mismatch")
        if conditions.shape != (batch, len(CONDITION_FIELDS)):
            raise ValueError("condition ABI mismatch")
        if condition_mask.shape != conditions.shape:
            raise ValueError("condition mask ABI mismatch")
        required = field_mask[:, : self.dynamic_channels]
        if not bool((required == 1).all()):
            raise ValueError("typed mixer received a missing required scalar field")
        fields = field_memory[:, : self.dynamic_channels].flatten(1, 2)
        value = self.field_projection(fields)
        payload = torch.cat(
            (
                conditions * condition_mask,
                condition_mask,
                field_mean,
                field_log_scale,
            ),
            dim=1,
        ).to(value.dtype)
        value = value + self.condition(payload)[:, :, None, None] + self.positional
        for block in self.blocks:
            value = block(value)
        expected_latent = (
            batch,
            self.config.private_width,
            self.config.latent_size,
            self.config.latent_size,
        )
        if tuple(value.shape) != expected_latent:
            raise RuntimeError(f"private latent ABI drifted: {tuple(value.shape)}")
        return value


class CFDBFamilyAnalysisBlock(nn.Module):
    """Shared spatial analysis followed by one deterministic family FFN.

    This is the Semantic Field counterpart of the qualified V3 CFDB split:
    normalization, spectral mixing and local analysis are representation
    operations shared by the three flow datasets, while the nonlinear saved-
    frame evolution map is private to cavity, cylinder or tube.
    """

    def __init__(
        self,
        width: int,
        *,
        num_blocks: int,
        modes: int,
        ffn_hidden: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.spectral = _official_afno(width, num_blocks=num_blocks, modes=modes)
        self.local = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.norm2 = nn.GroupNorm(8, width)
        self.family_ffns = nn.ModuleDict(
            {
                family: nn.Sequential(
                    nn.Conv2d(width, ffn_hidden, 1),
                    nn.GELU(),
                    nn.Conv2d(ffn_hidden, width, 1),
                )
                for family in CFDB_FAMILY_CODES
            }
        )

    def forward(self, value: Tensor, program_codes: Tensor) -> Tensor:
        if program_codes.shape != (value.shape[0],):
            raise ValueError("CFDB program codes must be [B]")
        residual = value
        normalized = self.norm1(value)
        mixed = residual + self.spectral(normalized) + self.local(normalized)
        private_input = self.norm2(mixed)
        update = torch.empty_like(private_input)
        for family, code in CFDB_FAMILY_CODES.items():
            index = (program_codes == code).nonzero(as_tuple=False).flatten()
            if index.numel() == 0:
                continue
            family_update = self.family_ffns[family](
                private_input.index_select(0, index)
            )
            update.index_copy_(0, index, family_update)
        return mixed + update


class CFDBTypedFieldMixer(nn.Module):
    """CFDB carrier with shared analysis and family-private nonlinear flows."""

    reads_full_history = False

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.dynamic_channels = 2
        self.field_projection = nn.Conv2d(
            self.dynamic_channels * config.shared_width,
            config.private_width,
            1,
        )
        condition_inputs = 2 * len(CONDITION_FIELDS) + 2 * config.max_fields
        self.condition = nn.Sequential(
            nn.Linear(condition_inputs, config.condition_hidden),
            nn.GELU(),
            nn.Linear(config.condition_hidden, config.private_width),
        )
        self.positional = nn.Parameter(
            torch.zeros(1, config.private_width, config.latent_size, config.latent_size)
        )
        nn.init.trunc_normal_(self.positional, std=0.02)
        self.blocks = nn.ModuleList(
            CFDBFamilyAnalysisBlock(
                config.private_width,
                num_blocks=config.afno_blocks,
                modes=config.afno_modes,
                ffn_hidden=config.private_ffn_hidden,
            )
            for _ in range(config.private_depth)
        )

    def forward(
        self,
        field_memory: Tensor,
        field_mask: Tensor,
        field_mean: Tensor,
        field_log_scale: Tensor,
        conditions: Tensor,
        condition_mask: Tensor,
        program_codes: Tensor,
    ) -> Tensor:
        batch = field_memory.shape[0]
        expected = (
            self.config.max_fields,
            self.config.shared_width,
            self.config.latent_size,
            self.config.latent_size,
        )
        if field_memory.ndim != 5 or tuple(field_memory.shape[1:]) != expected:
            raise ValueError("shared field memory ABI mismatch")
        if field_mask.shape != (batch, self.config.max_fields):
            raise ValueError("field mask ABI mismatch")
        if field_mean.shape != field_mask.shape or field_log_scale.shape != field_mask.shape:
            raise ValueError("field statistics ABI mismatch")
        if conditions.shape != (batch, len(CONDITION_FIELDS)):
            raise ValueError("condition ABI mismatch")
        if condition_mask.shape != conditions.shape:
            raise ValueError("condition mask must match conditions")
        if program_codes.shape != (batch,):
            raise ValueError("CFDB program code ABI mismatch")
        required = field_mask[:, : self.dynamic_channels]
        if not bool((required == 1).all()):
            raise ValueError("CFDB mixer received a missing velocity field")
        fields = field_memory[:, : self.dynamic_channels].flatten(1, 2)
        value = self.field_projection(fields)
        payload = torch.cat(
            (
                conditions * condition_mask,
                condition_mask,
                field_mean,
                field_log_scale,
            ),
            dim=1,
        ).to(value.dtype)
        value = value + self.condition(payload)[:, :, None, None] + self.positional
        for block in self.blocks:
            value = block(value, program_codes)
        expected_latent = (
            batch,
            self.config.private_width,
            self.config.latent_size,
            self.config.latent_size,
        )
        if tuple(value.shape) != expected_latent:
            raise RuntimeError(f"CFDB private latent ABI drifted: {tuple(value.shape)}")
        return value


__all__ = [
    "CFDB_FAMILY_CODES",
    "CFDBFamilyAnalysisBlock",
    "CFDBTypedFieldMixer",
    "PrivateTypedFieldMixer",
]
