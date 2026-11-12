from __future__ import annotations


import hashlib


from pathlib import Path


import h5py


import numpy as np


import torch


from torch import Tensor


from torch.utils.data import Dataset


from hotpot._internal.cfdb_schema.schema_data import ObservationSchema, resize_and_pad


SAMPLES = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class OfficialCFDBDataset(Dataset[tuple[int, Tensor, Tensor, Tensor]]):
    """Exact public chart plus the official original-grid evaluation mask."""

    def __init__(
        self,
        path: Path,
        schema: ObservationSchema,
        indices: tuple[int, ...] | None = None,
    ) -> None:
        self.path = path.resolve(strict=True)
        self.schema = schema
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            self.shape = tuple(int(value) for value in handle["data"].shape)
        expected = (schema.stored_height, schema.stored_width, 20, 3)
        if self.shape[0] < SAMPLES or self.shape[1:] != expected:
            raise ValueError(f"unexpected official CFDB shape {self.shape}")
        if (
            schema.public_height % schema.stored_height
            or schema.public_width % schema.stored_width
        ):
            raise ValueError("official public/stored grids are not integer aligned")
        self.stride = (
            schema.public_height // schema.stored_height,
            schema.public_width // schema.stored_width,
        )
        self.indices = tuple(range(SAMPLES)) if indices is None else indices
        if not self.indices or not set(self.indices).issubset(set(range(self.shape[0]))):
            raise ValueError("official CFDB subset indices drifted")

    def __len__(self) -> int:
        return len(self.indices)

    def _data(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle["data"]

    def __getitem__(self, index: int) -> tuple[int, Tensor, Tensor, Tensor]:
        source_index = self.indices[int(index)]
        sample = torch.from_numpy(
            np.asarray(self._data()[source_index], dtype=np.float32)
        )
        public = resize_and_pad(sample, self.schema)
        history = public[..., :10, :].contiguous()
        target = public[..., 10:20, :2].contiguous()
        mask = torch.zeros(
            self.schema.public_height,
            self.schema.public_width,
            1,
            2,
            dtype=torch.float32,
        )
        mask[:: self.stride[0], :: self.stride[1], :, :] = 1.0
        return source_index, history, target, mask


def per_sample_relative_l2(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None,
) -> Tensor:
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("velocity prediction/target contract drifted")
    if mask is not None:
        prediction = prediction * mask
        target = target * mask
    batch = prediction.shape[0]
    difference = torch.linalg.vector_norm(
        (prediction - target).reshape(batch, -1, 2), dim=1
    )
    denominator = (
        torch.linalg.vector_norm(target.reshape(batch, -1, 2), dim=1) + 1.0e-8
    )
    return (difference / denominator).mean(dim=-1)
