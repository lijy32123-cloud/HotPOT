from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AFNO2D(nn.Module):
    def __init__(self, width: int, num_blocks: int) -> None:
        super().__init__()
        if width % num_blocks:
            raise ValueError("AFNO width must be divisible by num_blocks")
        self.width = int(width)
        self.num_blocks = int(num_blocks)
        self.block_size = self.width // self.num_blocks
        scale = 1.0 / max(1, self.block_size * self.block_size)
        shape = (2, self.num_blocks, self.block_size, self.block_size)
        bias_shape = (2, self.num_blocks, self.block_size)
        self.w1 = nn.Parameter(scale * torch.randn(shape))
        self.b1 = nn.Parameter(scale * torch.randn(bias_shape))
        self.w2 = nn.Parameter(scale * torch.randn(shape))
        self.b2 = nn.Parameter(scale * torch.randn(bias_shape))

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1] != self.width:
            raise ValueError("AFNO expects [B,width,H,W]")
        batch, _channels, height, width = value.shape
        spectrum = torch.fft.rfft2(
            value.permute(0, 2, 3, 1), dim=(1, 2), norm="ortho"
        ).reshape(
            batch,
            height,
            width // 2 + 1,
            self.num_blocks,
            self.block_size,
        )
        real, imag = spectrum.real, spectrum.imag
        hidden_real = F.gelu(
            torch.einsum("...bi,bio->...bo", real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", imag, self.w1[1])
            + self.b1[0]
        )
        hidden_imag = F.gelu(
            torch.einsum("...bi,bio->...bo", imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", real, self.w1[1])
            + self.b1[1]
        )
        out_real = (
            torch.einsum("...bi,bio->...bo", hidden_real, self.w2[0])
            - torch.einsum("...bi,bio->...bo", hidden_imag, self.w2[1])
            + self.b2[0]
        )
        out_imag = (
            torch.einsum("...bi,bio->...bo", hidden_imag, self.w2[0])
            + torch.einsum("...bi,bio->...bo", hidden_real, self.w2[1])
            + self.b2[1]
        )
        mixed = torch.complex(out_real, out_imag).reshape(
            batch, height, width // 2 + 1, self.width
        )
        result = torch.fft.irfft2(
            mixed, s=(height, width), dim=(1, 2), norm="ortho"
        )
        return result.permute(0, 3, 1, 2).contiguous()


class PointwiseFFN(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, hidden, 1), nn.GELU(), nn.Conv2d(hidden, width, 1)
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class SpectralResidualBlock(nn.Module):
    def __init__(self, width: int, num_blocks: int, ffn_hidden: int) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, width)
        self.mixer = AFNO2D(width, num_blocks)
        self.norm2 = nn.GroupNorm(groups, width)
        self.ffn = PointwiseFFN(width, ffn_hidden)

    def forward(self, value: Tensor) -> Tensor:
        value = value + self.mixer(self.norm1(value))
        return value + self.ffn(self.norm2(value))


class DepthwiseResidualBlock(nn.Module):
    def __init__(
        self, width: int, hidden: int, *, padding_mode: str = "circular"
    ) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, hidden, 1)
        self.depthwise = nn.Conv2d(
            hidden,
            hidden,
            3,
            padding=1,
            groups=hidden,
            bias=False,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(hidden, width, 1)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, value: Tensor) -> Tensor:
        hidden = F.gelu(self.conv1(self.norm(value)))
        hidden = F.gelu(self.depthwise(hidden))
        return value + torch.tanh(self.scale) * self.conv2(hidden)


__all__ = [
    "AFNO2D",
    "DepthwiseResidualBlock",
    "PointwiseFFN",
    "SpectralResidualBlock",
]
