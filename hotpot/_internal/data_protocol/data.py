from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .cfdb_sidecar import CertifiedCFDBSidecar
from .manifest import BASE6, CONDITION_FIELDS, HISTORY, MODEL_RESOLUTION
from .sampler import Base6EpochPlan, EpochAddress


CFDB_CONDITION = {
    "cavity": {"observation_dt": 0.1, "dx": 1.0 / 64.0, "dy": 1.0 / 64.0},
    "cylinder": {"observation_dt": 0.001, "dx": 1.0 / 64.0, "dy": 1.0 / 64.0},
    "tube": {"observation_dt": 0.1, "dx": 1.0 / 64.0, "dy": 1.0 / 64.0},
}


class DistributedAddressBatchSampler(Sampler[list[EpochAddress]]):
    def __init__(
        self,
        plan: Base6EpochPlan,
        *,
        rank: int,
        world_size: int,
        local_batch_size: int = 20,
        epoch: int = 0,
    ) -> None:
        if rank < 0 or rank >= world_size or world_size < 1 or local_batch_size < 1:
            raise ValueError("invalid distributed batch sampler coordinates")
        self.plan = plan
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_batch_size = int(local_batch_size)
        self.epoch = int(epoch)
        self._cached_epoch: int | None = None
        self._cached_batches: tuple[tuple[EpochAddress, ...], ...] | None = None

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)
        if self._cached_epoch != self.epoch:
            self._cached_batches = None

    def _global_batches(self) -> tuple[tuple[EpochAddress, ...], ...]:
        if self._cached_batches is None or self._cached_epoch != self.epoch:
            global_size = self.local_batch_size * self.world_size
            self._cached_batches = self.plan.batches(
                self.epoch,
                global_batch_size=global_size,
                allow_remainder=True,
            )
            self._cached_epoch = self.epoch
        return self._cached_batches

    def fingerprint(self) -> str:
        import hashlib
        import struct

        digest = hashlib.sha256()
        digest.update(b"typed-sparse-evolution-base6-compatible-batches-v1\0")
        digest.update(
            struct.pack(
                ">qqq",
                self.plan.seed,
                self.epoch,
                self.local_batch_size * self.world_size,
            )
        )
        for batch_index, batch in enumerate(self._global_batches()):
            digest.update(struct.pack(">q", batch_index))
            for item in batch:
                digest.update(
                    struct.pack(
                        ">qqqqq",
                        item.epoch_position,
                        item.dataset_index,
                        item.data_index,
                        item.start,
                        item.noise_seed,
                    )
                )
        return digest.hexdigest()

    def __iter__(self) -> Iterator[list[EpochAddress]]:
        for batch in self._global_batches():
            if len(batch) % self.world_size:
                raise RuntimeError("global remainder batch cannot be partitioned evenly")
            replica_size = len(batch) // self.world_size
            begin = self.rank * replica_size
            end = begin + replica_size
            local = list(batch[begin:end])
            if len(local) != replica_size or replica_size > self.local_batch_size:
                raise RuntimeError("global address batch cannot be partitioned evenly")
            yield local

    def __len__(self) -> int:
        return len(self._global_batches())


class Base6StorageDataset(Dataset):
    """Read train-origin tensors only; the index is a fully specified address."""

    def __init__(
        self,
        workspace: str | Path,
        cfdb_sidecar: CertifiedCFDBSidecar,
    ) -> None:
        self.workspace = Path(workspace).resolve(strict=True)
        if "test" in str(self.workspace).lower():
            raise ValueError("workspace path unexpectedly contains test")
        self.cfdb_sidecar = cfdb_sidecar
        self._pid: int | None = None
        self._packed: dict[int, object] = {}

    def __len__(self) -> int:
        return sum(item.weighted_rows for item in BASE6)

    def _ensure_process(self) -> None:
        pid = os.getpid()
        if self._pid == pid:
            return
        for handle in self._packed.values():
            try:
                handle.close()
            except Exception:
                pass
        self._packed = {}
        self._pid = pid

    def _load(self, address: EpochAddress) -> np.ndarray:
        import h5py

        entry = BASE6[address.dataset_index]
        root = (self.workspace / entry.train_path).resolve(strict=True)
        if "test" in str(root).lower():
            raise RuntimeError("official test access is forbidden")
        if entry.storage == "packed_hdf5":
            self._ensure_process()
            if entry.index not in self._packed:
                self._packed[entry.index] = h5py.File(root, "r")
            value = np.asarray(
                self._packed[entry.index]["data"][address.data_index], dtype=np.float32
            )
        elif entry.storage == "scatter_hdf5":
            path = (root / f"data_{address.data_index}.hdf5").resolve(strict=True)
            try:
                with h5py.File(path, "r") as handle:
                    value = np.asarray(handle["data"][:], dtype=np.float32)
            except OSError as error:
                raise OSError(
                    f"{entry.key} train row {address.data_index} is unreadable: {path}"
                ) from error
        else:
            raise ValueError(entry.storage)
        if value.ndim == 3:
            value = value[..., None]
        expected = (entry.native_size, entry.native_size, entry.total_frames, entry.channels)
        if tuple(value.shape) != expected:
            raise RuntimeError(f"{entry.key} tensor drifted: {value.shape} != {expected}")
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

    def __getitem__(self, address: EpochAddress) -> dict[str, Tensor]:
        if not isinstance(address, EpochAddress):
            raise TypeError("Base6StorageDataset requires EpochAddress indices")
        entry = BASE6[address.dataset_index]
        value = self._load(address)
        stop = address.start + HISTORY + 1
        if address.start < 0 or stop > entry.total_frames:
            raise IndexError((entry.key, address.start, entry.total_frames))
        window = self._resize(torch.from_numpy(value[:, :, address.start:stop]))
        padded = torch.ones(MODEL_RESOLUTION, MODEL_RESOLUTION, HISTORY + 1, 4)
        padded[..., :entry.channels] = window
        conditions = entry.condition_mapping()
        if entry.key == "cfdbench":
            conditions = CFDB_CONDITION[self.cfdb_sidecar.family(address.data_index)]
        condition_value = torch.zeros(len(CONDITION_FIELDS), dtype=torch.float32)
        condition_mask = torch.zeros_like(condition_value)
        for name, scalar in conditions.items():
            column = CONDITION_FIELDS.index(name)
            condition_value[column] = float(scalar)
            condition_mask[column] = 1.0
        channel_mask = torch.zeros(4, dtype=torch.float32)
        channel_mask[:entry.channels] = 1.0
        return {
            "history": padded[..., :HISTORY, :].contiguous(),
            "target": padded[..., HISTORY:, :].contiguous(),
            "dataset_index": torch.tensor(address.dataset_index, dtype=torch.int64),
            "row_index": torch.tensor(address.data_index, dtype=torch.int64),
            "start": torch.tensor(address.start, dtype=torch.int64),
            "noise_seed": torch.tensor(address.noise_seed, dtype=torch.int64),
            "condition_value": condition_value,
            "condition_mask": condition_mask,
            "channel_mask": channel_mask,
        }


def matched_dpot_noise(
    history: Tensor,
    noise_seeds: Tensor,
    dataset_indices: Tensor,
    *,
    scale: float = 5.0e-4,
    positive_floor: float = 1.0e-6,
) -> Tensor:
    """DPOT-amplitude noise with rare positivity-compatible shortening.

    The exact same returned tensor must be passed to both model arms.  The
    shortening changes only directions that would make an observed positive
    primitive invalid.
    """

    if history.ndim != 5 or history.shape[-2:] != (HISTORY, 4):
        raise ValueError("history must be [B,H,W,H10,C4]")
    if noise_seeds.shape != (history.shape[0],) or dataset_indices.shape != noise_seeds.shape:
        raise ValueError("noise seeds and dataset indices must be [B]")
    if scale < 0.0:
        raise ValueError("noise scale must be nonnegative")
    if scale == 0.0:
        return history

    # Keep address seeds on CPU so generating train augmentation never inserts
    # one GPU->CPU synchronization per sample.  Each address still owns an
    # independent generator, hence its perturbation is invariant to model arm.
    seeds = noise_seeds.detach().to(device="cpu", dtype=torch.int64).tolist()
    datasets = dataset_indices.detach().to(device="cpu", dtype=torch.int64)
    sample_shape = history.shape[1:]
    noise = torch.empty_like(history)
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=history.device)
        generator.manual_seed(int(seed))
        torch.randn(
            sample_shape,
            device=history.device,
            dtype=history.dtype,
            generator=generator,
            out=noise[index],
        )
    amplitude = scale * history.square().sum((1, 2, 3), keepdim=True).sqrt()
    perturbation = amplitude * noise

    # The official Gaussian amplitude is unchanged.  Only directions that
    # would invalidate an observed positive primitive are shortened, once per
    # sample/channel and without host synchronization.
    protected = {
        # SWE height is the only positive scalar in Base6 channel zero.
        0: datasets == 3,
        # CNS density and pressure occupy primitive channels two and three.
        2: datasets == 2,
        3: datasets == 2,
    }
    for channel, selected_cpu in protected.items():
        selected = selected_cpu.to(device=history.device)
        state = history[..., channel]
        delta = perturbation[..., channel]
        admissible = torch.where(
            delta < 0,
            (state - positive_floor).clamp_min(0.0) / (-delta).clamp_min(1.0e-20),
            torch.full_like(delta, float("inf")),
        )
        bound = admissible.amin(dim=(1, 2, 3)).clamp(max=1.0)
        factor = torch.where(
            selected & (bound < 1.0), 0.99 * bound, torch.ones_like(bound)
        )
        perturbation[..., channel] = delta * factor[:, None, None, None]
    return history + perturbation


def operator_owned_augmentation(
    history: Tensor,
    noise_seeds: Tensor,
    dataset_indices: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Apply each learned operator's qualified training-chart augmentation.

    Raw sample addresses remain identical to the Base6 sampler.  Only the
    stochastic training transform is typed: FNO is perturbed after exact
    restriction to native64, DR/CFDB use released public-chart noise, and the
    qualified SWE/CNS actions receive no artificial input perturbation.
    """

    if history.ndim != 5 or history.shape[-2:] != (10, 4):
        raise ValueError("Base6 history must be [B,H,W,H10,C4]")
    if noise_seeds.shape != dataset_indices.shape or noise_seeds.shape != (
        history.shape[0],
    ):
        raise ValueError("noise seeds and dataset indices must be [B]")
    if scale < 0.0:
        raise ValueError("noise scale must be nonnegative")
    if scale == 0.0:
        return history

    augmented = history.clone()
    public_noisy = matched_dpot_noise(
        history, noise_seeds, dataset_indices, scale=scale
    )
    public_rows = (dataset_indices == 4) | (dataset_indices == 5)
    augmented[public_rows] = public_noisy[public_rows]

    fno_rows = (dataset_indices == 0) | (dataset_indices == 1)
    positions = fno_rows.nonzero(as_tuple=False).flatten().tolist()
    seeds = noise_seeds.detach().to(device="cpu", dtype=torch.int64).tolist()
    for position in positions:
        public = history[position, ..., 0].permute(2, 0, 1).contiguous()
        # Exact official bilinear 64->128 range inverse.
        first_y, second_y = public[:, 1::4], public[:, 2::4]
        native_y = torch.stack(
            (1.5 * first_y - 0.5 * second_y,
             -0.5 * first_y + 1.5 * second_y),
            dim=-2,
        ).flatten(-3, -2)
        first_x, second_x = native_y[..., 1::4], native_y[..., 2::4]
        native = torch.stack(
            (1.5 * first_x - 0.5 * second_x,
             -0.5 * first_x + 1.5 * second_x),
            dim=-1,
        ).flatten(-2, -1)
        generator = torch.Generator(device=history.device)
        generator.manual_seed(int(seeds[position]))
        noise = torch.randn(
            native.shape,
            device=native.device,
            dtype=native.dtype,
            generator=generator,
        )
        magnitude = native.square().sum().sqrt()
        native_noisy = native + scale * magnitude * noise
        public_noisy_scalar = F.interpolate(
            native_noisy[None],
            size=(history.shape[1], history.shape[2]),
            mode="bilinear",
            align_corners=False,
        )[0]
        augmented[position, ..., 0] = public_noisy_scalar.permute(1, 2, 0)
    return augmented


__all__ = [
    "Base6StorageDataset",
    "DistributedAddressBatchSampler",
    "matched_dpot_noise",
    "operator_owned_augmentation",
]
