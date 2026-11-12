from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import struct

import torch

from .manifest import BASE6, HISTORY, DatasetSpec, manifest_payload


@dataclass(frozen=True)
class EpochAddress:
    epoch_position: int
    dataset_index: int
    data_index: int
    start: int
    noise_seed: int


def _counter(seed: int, epoch: int, virtual_index: int) -> int:
    payload = f"base6-address-v1:{seed}:{epoch}:{virtual_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class Base6EpochPlan:
    """Deterministic realization of the released Base6 shuffled dataset.

    As in `MixedTemporalDataset`, each dataset row is repeated `data_weight`
    times and each occurrence samples a legal H10->H1 start.  Serializing the
    plan makes our model and MoE-POT consume exactly the same addresses.
    """

    def __init__(
        self,
        entries: tuple[DatasetSpec, ...] = BASE6,
        *,
        seed: int = 20260815,
    ) -> None:
        self.entries = entries
        self.seed = int(seed)
        self.weighted_sizes = tuple(item.weighted_rows for item in entries)
        cumulative = []
        total = 0
        for size in self.weighted_sizes:
            total += size
            cumulative.append(total)
        self.cumulative = tuple(cumulative)
        self.total = total

    def _decode(self, virtual_index: int, epoch_position: int, epoch: int) -> EpochAddress:
        if virtual_index < 0 or virtual_index >= self.total:
            raise IndexError(virtual_index)
        dataset_index = next(
            index for index, end in enumerate(self.cumulative) if virtual_index < end
        )
        begin = 0 if dataset_index == 0 else self.cumulative[dataset_index - 1]
        local = virtual_index - begin
        entry = self.entries[dataset_index]
        data_index = local // entry.data_weight
        legal_starts = max(entry.total_frames - (HISTORY + 1) + 1, 1)
        start = _counter(self.seed, epoch, virtual_index) % legal_starts
        noise_seed = _counter(self.seed + 1_000_003, epoch, virtual_index) % (2**63 - 1)
        return EpochAddress(epoch_position, dataset_index, data_index, start, noise_seed)

    def epoch(self, epoch: int) -> tuple[EpochAddress, ...]:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + int(epoch)) % (2**63 - 1))
        order = torch.randperm(self.total, generator=generator).tolist()
        return tuple(
            self._decode(virtual_index, position, int(epoch))
            for position, virtual_index in enumerate(order)
        )

    @staticmethod
    def lane(dataset_index: int) -> str:
        if dataset_index <= 1:
            return "fno2"
        if dataset_index == 2:
            return "cns"
        if dataset_index == 3:
            return "swe"
        if dataset_index == 4:
            return "dr"
        if dataset_index == 5:
            return "cfdb"
        raise IndexError(dataset_index)

    def batches(
        self,
        epoch: int,
        *,
        global_batch_size: int = 40,
        allow_remainder: bool = False,
    ) -> tuple[tuple[EpochAddress, ...], ...]:
        """Compatibility-packed global batches with exact Base6 exposure.

        The six released datasets, every address, global batch size, and the
        545-update epoch clock are unchanged.  Addresses are packed by
        executable-program lane so a hard-sparse operator library does not
        evaluate unrelated programs merely because storage rows happened to
        collide in one random batch.  The same serialized batches are consumed
        by our model and the matched MoE-POT arm.
        """

        if global_batch_size < 1:
            raise ValueError("global_batch_size must be positive")
        addresses = self.epoch(epoch)
        if len(addresses) % global_batch_size and not allow_remainder:
            raise RuntimeError(
                f"Base6 has {len(addresses)} rows, not divisible by global batch {global_batch_size}"
            )
        lanes: dict[str, list[EpochAddress]] = {}
        for address in addresses:
            lanes.setdefault(self.lane(address.dataset_index), []).append(address)
        packed: list[tuple[EpochAddress, ...]] = []
        remainder: list[EpochAddress] = []
        for lane in ("fno2", "cns", "swe", "dr", "cfdb"):
            values = lanes[lane]
            full = len(values) - (len(values) % global_batch_size)
            packed.extend(
                tuple(values[begin : begin + global_batch_size])
                for begin in range(0, full, global_batch_size)
            )
            remainder.extend(values[full:])
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + 10_000_019 + int(epoch)) % (2**63 - 1))
        if remainder:
            order = torch.randperm(len(remainder), generator=generator).tolist()
            remainder = [remainder[index] for index in order]
            if len(remainder) % global_batch_size and not allow_remainder:
                raise RuntimeError("compatibility-packed remainders do not form full batches")
            packed.extend(
                tuple(remainder[begin : begin + global_batch_size])
                for begin in range(0, len(remainder), global_batch_size)
            )
        batch_order = torch.randperm(len(packed), generator=generator).tolist()
        return tuple(packed[index] for index in batch_order)

    def write(self, path: str | Path, epoch: int) -> dict[str, object]:
        destination = Path(path)
        batches = self.batches(epoch)
        addresses = tuple(address for batch in batches for address in batch)
        header = {
            "schema": "typed_sparse_evolution.base6_compatibility_packed_epoch_plan.v1",
            "epoch": int(epoch),
            "seed": self.seed,
            "rows": len(addresses),
            "base6_manifest_sha256": manifest_payload()["manifest_sha256"],
            "official_test_accessed": False,
        }
        lines = [json.dumps({"header": header}, sort_keys=True)]
        lines.extend(json.dumps(asdict(item), sort_keys=True) for item in addresses)
        serialized = ("\n".join(lines) + "\n").encode()
        header["address_payload_sha256"] = hashlib.sha256(serialized).hexdigest()
        lines[0] = json.dumps({"header": header}, sort_keys=True)
        serialized = ("\n".join(lines) + "\n").encode()
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(serialized)
        os.replace(temporary, destination)
        return header

    def batch_fingerprint(
        self,
        epoch: int,
        *,
        global_batch_size: int = 40,
        allow_remainder: bool = False,
    ) -> str:
        """Hash the exact global batch order without storing a huge plan."""

        digest = hashlib.sha256()
        digest.update(b"typed-sparse-evolution-base6-compatible-batches-v1\0")
        digest.update(struct.pack(">qqq", self.seed, int(epoch), int(global_batch_size)))
        for batch_index, batch in enumerate(
            self.batches(
                epoch,
                global_batch_size=global_batch_size,
                allow_remainder=allow_remainder,
            )
        ):
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


__all__ = ["Base6EpochPlan", "EpochAddress"]
