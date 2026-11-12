from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FoundationConfig:
    resolution: int = 128
    history_steps: int = 10
    max_fields: int = 4
    semantic_channels: int = 11
    patch_size: int = 8
    shared_width: int = 256
    shared_depth: int = 2
    shared_ffn_hidden: int = 512
    private_width: int = 512
    private_depth: int = 5
    private_ffn_hidden: int = 2048
    afno_blocks: int = 4
    afno_modes: int = 16
    decoded_channels: int = 32
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
        if self.decoded_channels != 32:
            raise ValueError("typed action heads require 32 decoded channels")

    @property
    def latent_size(self) -> int:
        return self.resolution // self.patch_size

    def payload(self) -> dict[str, int]:
        self.validate()
        return {name: int(value) for name, value in asdict(self).items()}


def tiny_test_config() -> FoundationConfig:
    return FoundationConfig(
        shared_width=32,
        shared_depth=1,
        shared_ffn_hidden=64,
        private_width=32,
        private_depth=1,
        private_ffn_hidden=64,
        afno_blocks=4,
        afno_modes=8,
        condition_hidden=32,
    )


__all__ = ["FoundationConfig", "tiny_test_config"]
