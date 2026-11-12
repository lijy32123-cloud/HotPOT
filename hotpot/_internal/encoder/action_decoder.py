from __future__ import annotations

from torch import Tensor, nn

from .config import FoundationConfig


class ActionContextDecoder(nn.Module):
    """First stage of D_m; typed heads finish the physical action readout."""

    reads_full_history = False

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                config.private_width,
                config.decoded_channels,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            ),
            nn.GELU(),
            nn.Conv2d(config.decoded_channels, config.decoded_channels, 1),
        )

    def forward(self, private_latent: Tensor) -> Tensor:
        batch = private_latent.shape[0]
        expected = (
            self.config.private_width,
            self.config.latent_size,
            self.config.latent_size,
        )
        if private_latent.ndim != 4 or tuple(private_latent.shape[1:]) != expected:
            raise ValueError("private latent ABI mismatch")
        decoded = self.net(private_latent)
        expected_decoded = (
            batch,
            self.config.decoded_channels,
            self.config.resolution,
            self.config.resolution,
        )
        if tuple(decoded.shape) != expected_decoded:
            raise RuntimeError(f"action context ABI drifted: {tuple(decoded.shape)}")
        return decoded


__all__ = ["ActionContextDecoder"]
