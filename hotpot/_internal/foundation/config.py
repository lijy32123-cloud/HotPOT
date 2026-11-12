from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FoundationScaleConfig:
    resolution: int = 128
    history_steps: int = 10
    max_fields: int = 4
    semantic_channels: int = 11
    patch_size: int = 8

    # Shared representation stays a minority of active capacity.  It analyzes
    # scalar-field history but never learns task-specific evolution semantics.
    shared_width: int = 512
    shared_depth: int = 3
    shared_ffn_hidden: int = 2048

    # Most added capacity is placed after the typed branch, where cross-field
    # coupling and task-specific actions are learned without cross-PDE mixing.
    private_width: int = 1024
    private_depth: int = 7
    private_ffn_hidden: int = 3968

    afno_blocks: int = 4
    afno_modes: int = 16

    # Widen the action bottleneck together with the latent network; otherwise
    
    decoded_channels: int = 64
    typed_head_hidden: int = 256
    condition_feature_channels: int = 32
    condition_feature_hidden: int = 128
    condition_hidden: int = 128

    def validate(self) -> None:
        invalid = {
            name: value
            for name, value in asdict(self).items()
            if int(value) < 1
        }
        if invalid:
            raise ValueError(f"configuration entries must be positive: {invalid}")
        if self.resolution % self.patch_size:
            raise ValueError("resolution must be divisible by patch_size")
        if self.shared_width % self.afno_blocks:
            raise ValueError("shared_width must be divisible by afno_blocks")
        if self.private_width % self.afno_blocks:
            raise ValueError("private_width must be divisible by afno_blocks")
        if self.shared_width % 8 or self.private_width % 8:
            raise ValueError("GroupNorm widths must be divisible by eight")

    @property
    def latent_size(self) -> int:
        return self.resolution // self.patch_size

    def payload(self) -> dict[str, int]:
        self.validate()
        return {name: int(value) for name, value in asdict(self).items()}


def tiny_test_config() -> FoundationScaleConfig:
    return FoundationScaleConfig(
        shared_width=32,
        shared_depth=1,
        shared_ffn_hidden=64,
        private_width=32,
        private_depth=1,
        private_ffn_hidden=64,
        afno_modes=8,
        decoded_channels=16,
        typed_head_hidden=32,
        condition_feature_channels=8,
        condition_feature_hidden=32,
        condition_hidden=32,
    )


def tiny_17m_config() -> FoundationScaleConfig:
    """Production Tiny scale with the same learned path as V1.1-S.

    This is not the unit-test configuration above.  It keeps the final typed
    programs and all interfaces identical to the S model while restoring the
    V1.1-T widths used for an approximately 17M activated-parameter budget.
    """

    return FoundationScaleConfig(
        shared_width=256,
        shared_depth=2,
        shared_ffn_hidden=512,
        private_width=512,
        private_depth=5,
        private_ffn_hidden=2048,
        decoded_channels=32,
        typed_head_hidden=128,
        condition_feature_channels=16,
        condition_feature_hidden=64,
        condition_hidden=128,
    )


__all__ = ["FoundationScaleConfig", "tiny_17m_config", "tiny_test_config"]
