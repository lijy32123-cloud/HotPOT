from __future__ import annotations


import hashlib


import json


from pathlib import Path


from typing import Any


import h5py


import numpy as np


import torch


from torch import Tensor


from torch.utils.data import Dataset


from .evaluation_manifest import EvaluationSpec


from .manifest import ProgramId


CFDB_PROGRAM = {
    "cavity": ProgramId.CFDB_CAVITY,
    "cylinder": ProgramId.CFDB_CYLINDER,
    "tube": ProgramId.CFDB_TUBE,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path: Path, workspace: Path, label: str, *, strict: bool) -> Path:
    resolved = path.resolve(strict=strict)
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"{label} must remain below the workspace") from error
    return resolved


class CertifiedCFDBTestSidecar:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve(strict=True)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        entries = tuple(payload.get("entries", ()))
        if (
            payload.get("mapping_certified") is not True
            or payload.get("scope") != "OFFICIAL_TEST_ONLY"
            or len(entries) != 1000
            or [int(row.get("index", -1)) for row in entries]
            != list(range(len(entries)))
        ):
            raise ValueError("CFDB official-test sidecar contract drifted")
        families = tuple(str(row.get("regime")) for row in entries)
        if not set(families).issubset(CFDB_PROGRAM):
            raise ValueError("CFDB test sidecar contains an unsupported family")
        self.entries = entries
        self.families = families
        self.sha256 = sha256(self.path)

    def family(self, row: int) -> str:
        if row < 0 or row >= len(self.families):
            raise IndexError(row)
        return self.families[row]

    def program(self, row: int) -> ProgramId:
        return CFDB_PROGRAM[self.family(row)]


class OfficialRolloutDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, workspace: Path, spec: EvaluationSpec) -> None:
        self.spec = spec
        self.path = inside(workspace / spec.test_path, workspace, spec.key, strict=True)
        self._handle: h5py.File | None = None
        if spec.storage == "packed_hdf5":
            with h5py.File(self.path, "r") as handle:
                self.shape = tuple(int(value) for value in handle["data"].shape)
            # Public scalar FNO archives use [N,H,W,T], whereas packed
            # multi-channel archives use [N,H,W,T,C].
            expected = (
                (spec.native_size, spec.native_size, spec.total_frames)
                if spec.channels == 1
                else (
                    spec.native_size,
                    spec.native_size,
                    spec.total_frames,
                    spec.channels,
                )
            )
            if self.shape[0] < spec.samples or self.shape[1:] != expected:
                raise ValueError(f"{spec.key} official packed-test shape drifted: {self.shape}")
        elif spec.storage == "scatter_hdf5":
            first = self.path / "data_0.hdf5"
            last = self.path / f"data_{spec.samples - 1}.hdf5"
            for file in (first, last):
                with h5py.File(file.resolve(strict=True), "r") as handle:
                    shape = tuple(int(value) for value in handle["data"].shape)
                expected = (
                    spec.native_size,
                    spec.native_size,
                    spec.total_frames,
                    spec.channels,
                )
                if shape != expected:
                    raise ValueError(f"{spec.key} official scatter-test shape drifted: {shape}")
            self.shape = (spec.samples, *expected)
        else:
            raise ValueError(spec.storage)

    def __len__(self) -> int:
        return self.spec.samples

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def _load(self, row: int) -> np.ndarray:
        if self.spec.storage == "packed_hdf5":
            if self._handle is None:
                self._handle = h5py.File(self.path, "r")
            return np.asarray(self._handle["data"][row], dtype=np.float32)
        path = (self.path / f"data_{row}.hdf5").resolve(strict=True)
        with h5py.File(path, "r") as handle:
            return np.asarray(handle["data"][:], dtype=np.float32)

    @staticmethod
    def _resize(value: Tensor) -> Tensor:
        height, width, frames, channels = value.shape
        packed = value.reshape(height, width, frames * channels).permute(2, 0, 1)
        packed = torch.nn.functional.interpolate(
            packed.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        return packed.reshape(128, 128, frames, channels)

    def __getitem__(self, row: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        value = torch.from_numpy(self._load(int(row)))
        if value.ndim == 3:
            value = value[..., None]
        public = torch.ones(128, 128, self.spec.total_frames, 4, dtype=torch.float32)
        public[..., : self.spec.channels] = self._resize(value)
        history = public[..., :10, :].contiguous()
        target = public[..., 10:, :].contiguous()
        if self.spec.native_size > 128 or 128 % self.spec.native_size:
            raise ValueError(f"{self.spec.key} cannot form the released original-grid mask")
        stride = 128 // self.spec.native_size
        mask = torch.zeros(128, 128, 1, 4, dtype=torch.float32)
        mask[::stride, ::stride, :, : self.spec.scored_channels] = 1.0
        return torch.tensor(row, dtype=torch.int64), history, target, mask
