from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from hotpot._internal.data_protocol.manifest import (
    CONDITION_FIELDS,
)


HISTORY = 10
MODEL_RESOLUTION = 128
VALIDATION_ROWS = 256
SELECTION_SEED = 20260826


@dataclass(frozen=True)
class TargetSpec:
    key: str
    family: str
    program: str
    relative_train_path: str
    storage: str
    rows: int
    channels: int
    predicted_channels: int
    native_size: int
    total_frames: int
    condition: tuple[tuple[str, float], ...]

    @property
    def legal_starts(self) -> int:
        return self.total_frames - HISTORY


def _condition(**values: float) -> tuple[tuple[str, float], ...]:
    unknown = set(values).difference(CONDITION_FIELDS)
    if unknown:
        raise ValueError(f"unknown condition fields: {sorted(unknown)}")
    return tuple((name, float(values[name])) for name in CONDITION_FIELDS if name in values)


TARGETS = {
    "fno_1e-4": TargetSpec(
        key="ns2d_fno_1e-4",
        family="fno",
        program="fno_vorticity_ns",
        relative_train_path="data/large/ns2d_1e-4_train.hdf5",
        storage="packed_hdf5",
        rows=9800,
        channels=1,
        predicted_channels=1,
        native_size=64,
        total_frames=50,
        condition=_condition(
            observation_dt=1.0,
            dx=1.0 / 64.0,
            dy=1.0 / 64.0,
            viscosity=1.0e-4,
        ),
    ),
    "cns_m1_eta1e-2": TargetSpec(
        key="ns2d_pdb_M1_eta1e-2_zeta1e-2",
        family="cns",
        program="cns_compressible",
        relative_train_path="data/large/pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2/train",
        storage="scatter_hdf5",
        rows=9000,
        channels=4,
        predicted_channels=4,
        native_size=128,
        total_frames=21,
        condition=_condition(
            observation_dt=0.05,
            dx=1.0 / 128.0,
            dy=1.0 / 128.0,
            combined_dissipation=1.0e-2,
            mach_regime_tag=1.0,
        ),
    ),
}


@dataclass(frozen=True)
class Address:
    row: int
    start: int


def _hash64(label: str, *values: int) -> int:
    payload = ":".join((label, *(str(int(value)) for value in values))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def split_rows(
    spec: TargetSpec, shots: int, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if shots not in (16, 64, 256, 1000):
        raise ValueError("shots must be one of 16, 64, 256, 1000")
    if seed not in (0, 1, 2):
        raise ValueError("seed must be one of 0, 1, 2")
    validation = tuple(range(spec.rows - VALIDATION_ROWS, spec.rows))
    pool_size = spec.rows - VALIDATION_ROWS
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        SELECTION_SEED + 1009 * seed + _hash64(spec.key) % 1_000_003
    )
    training = tuple(
        int(value)
        for value in torch.randperm(pool_size, generator=generator)[:shots].tolist()
    )
    if set(training) & set(validation):
        raise RuntimeError("training and validation trajectories overlap")
    return training, validation


def phase_start(spec: TargetSpec, epoch: int, row: int, seed: int) -> int:
    legal = spec.legal_starts
    if legal < 1:
        raise RuntimeError("target has no legal H10-to-H1 window")
    if legal == 1:
        return 0
    slot = (epoch + _hash64("phase-offset", seed, row) % 10) % 10
    if slot == 0:
        return 0
    non_initial = list(range(1, legal))
    cut1 = (len(non_initial) + 2) // 3
    cut2 = (2 * len(non_initial) + 2) // 3
    bins = (non_initial[:cut1], non_initial[cut1:cut2], non_initial[cut2:])
    candidates = bins[min((slot - 1) // 3, 2)] or non_initial
    return candidates[_hash64("phase-choice", seed, epoch, row) % len(candidates)]


def fixed_validation_addresses(
    spec: TargetSpec, rows: Sequence[int]
) -> tuple[Address, ...]:
    return tuple(
        Address(int(row), _hash64("validation-start", row) % spec.legal_starts)
        for row in rows
    )


class EpochBatchSampler(Sampler[list[Address]]):
    def __init__(
        self,
        spec: TargetSpec,
        rows: Sequence[int],
        *,
        seed: int,
        batch_size: int,
    ) -> None:
        self.spec = spec
        self.rows = tuple(int(value) for value in rows)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.epoch = 0
        if not self.rows or self.batch_size < 1:
            raise ValueError("rows and batch size must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def addresses(self) -> tuple[Address, ...]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SELECTION_SEED + 10_000_019 * self.seed + self.epoch)
        order = torch.randperm(len(self.rows), generator=generator).tolist()
        return tuple(
            Address(
                self.rows[position],
                phase_start(self.spec, self.epoch, self.rows[position], self.seed),
            )
            for position in order
        )

    def __iter__(self) -> Iterator[list[Address]]:
        values = self.addresses()
        for begin in range(0, len(values), self.batch_size):
            yield list(values[begin : begin + self.batch_size])

    def __len__(self) -> int:
        return (len(self.rows) + self.batch_size - 1) // self.batch_size


class FixedBatchSampler(Sampler[list[Address]]):
    def __init__(self, addresses: Sequence[Address], batch_size: int) -> None:
        self.addresses = tuple(addresses)
        self.batch_size = int(batch_size)

    def __iter__(self) -> Iterator[list[Address]]:
        for begin in range(0, len(self.addresses), self.batch_size):
            yield list(self.addresses[begin : begin + self.batch_size])

    def __len__(self) -> int:
        return (len(self.addresses) + self.batch_size - 1) // self.batch_size


class TargetStorageDataset(Dataset):
    def __init__(self, workspace: str | Path, spec: TargetSpec) -> None:
        self.workspace = Path(workspace).resolve(strict=True)
        self.spec = spec
        self.root = (self.workspace / spec.relative_train_path).resolve(strict=True)
        if "test" in str(self.root).lower():
            raise RuntimeError("official test access is forbidden")
        self._pid: int | None = None
        self._packed: object | None = None

    def __len__(self) -> int:
        return self.spec.rows

    def _packed_handle(self):
        import h5py

        pid = os.getpid()
        if self._pid != pid:
            if self._packed is not None:
                self._packed.close()
            self._packed = h5py.File(self.root, "r")
            self._pid = pid
        assert self._packed is not None
        return self._packed

    def _load(self, row: int) -> np.ndarray:
        import h5py

        if row < 0 or row >= self.spec.rows:
            raise IndexError(row)
        if self.spec.storage == "packed_hdf5":
            value = np.asarray(self._packed_handle()["data"][row], dtype=np.float32)
        elif self.spec.storage == "scatter_hdf5":
            path = (self.root / f"data_{row}.hdf5").resolve(strict=True)
            with h5py.File(path, "r") as handle:
                value = np.asarray(handle["data"][:], dtype=np.float32)
        else:
            raise ValueError(self.spec.storage)
        if value.ndim == 3:
            value = value[..., None]
        expected = (
            self.spec.native_size,
            self.spec.native_size,
            self.spec.total_frames,
            self.spec.channels,
        )
        if tuple(value.shape) != expected:
            raise RuntimeError(f"{self.spec.key} tensor drifted: {value.shape} != {expected}")
        return value

    @staticmethod
    def _resize(value: Tensor) -> Tensor:
        height, width, frames, channels = value.shape
        packed = value.reshape(height, width, frames * channels).permute(2, 0, 1)
        packed = F.interpolate(
            packed.unsqueeze(0),
            size=(MODEL_RESOLUTION, MODEL_RESOLUTION),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        return packed.reshape(MODEL_RESOLUTION, MODEL_RESOLUTION, frames, channels)

    def __getitem__(self, address: Address) -> dict[str, Tensor]:
        if not isinstance(address, Address):
            raise TypeError("TargetStorageDataset requires Address indices")
        stop = address.start + HISTORY + 1
        if address.start < 0 or stop > self.spec.total_frames:
            raise IndexError((address.start, self.spec.total_frames))
        value = torch.from_numpy(self._load(address.row)[:, :, address.start:stop])
        value = self._resize(value)
        padded = torch.ones(MODEL_RESOLUTION, MODEL_RESOLUTION, HISTORY + 1, 4)
        padded[..., : self.spec.channels] = value
        condition = torch.zeros(len(CONDITION_FIELDS), dtype=torch.float32)
        condition_mask = torch.zeros_like(condition)
        for name, scalar in self.spec.condition:
            column = CONDITION_FIELDS.index(name)
            condition[column] = scalar
            condition_mask[column] = 1.0
        return {
            "history": padded[..., :HISTORY, :].contiguous(),
            "target": padded[..., HISTORY:, :].contiguous(),
            "condition": condition,
            "condition_mask": condition_mask,
            "row": torch.tensor(address.row, dtype=torch.int64),
            "start": torch.tensor(address.start, dtype=torch.int64),
        }


def address_hash(addresses: Sequence[Address]) -> str:
    payload = ";".join(f"{value.row}:{value.start}" for value in addresses).encode()
    return hashlib.sha256(payload).hexdigest()
