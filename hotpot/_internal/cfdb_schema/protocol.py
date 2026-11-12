from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterator

import torch
from torch.utils.data import Sampler


SEED = 20260714
TRAIN_ROWS = 9_000
GLOBAL_BATCH = 40
UPDATES_PER_EPOCH = TRAIN_ROWS // GLOBAL_BATCH
FULL_EPOCHS = 1_000
FULL_UPDATES = FULL_EPOCHS * UPDATES_PER_EPOCH
PREFLIGHT_EPOCHS = 20
PREFLIGHT_UPDATES = PREFLIGHT_EPOCHS * UPDATES_PER_EPOCH
LEGAL_STARTS = 10

LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.9)
ADAM_EPS = 1.0e-8
WEIGHT_DECAY = 1.0e-6
GRADIENT_CLIP_NORM = 10_000.0
INPUT_NOISE_SCALE = 5.0e-4
SCHEDULER_TICKS_PER_UPDATE = 2
SCHEDULER_PCT_START = 0.2
DIV_FACTOR = 1.0e4
FINAL_DIV_FACTOR = 1.0e4


@dataclass(frozen=True)
class ObservationAddress:
    """A name-free address into one declared observation tensor."""

    epoch: int
    epoch_position: int
    row: int
    start: int


def _counter(namespace: str, *coordinates: int) -> int:
    payload = json.dumps(
        ["clean-observation-flow-v2", SEED, namespace, *coordinates],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def noise_seed(epoch: int, batch_index: int) -> int:
    return _counter("input-noise", int(epoch), int(batch_index)) % (2**63 - 1)


class ObservationEpochBatchSampler(Sampler[list[ObservationAddress]]):
    """One no-replacement permutation of the declared 9000 training rows."""

    def __init__(self, epoch: int = 0) -> None:
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return UPDATES_PER_EPOCH

    def __iter__(self) -> Iterator[list[ObservationAddress]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_counter("permutation", self.epoch) % (2**63 - 1))
        permutation = torch.randperm(TRAIN_ROWS, generator=generator).tolist()
        for batch_index in range(UPDATES_PER_EPOCH):
            begin = batch_index * GLOBAL_BATCH
            rows = permutation[begin : begin + GLOBAL_BATCH]
            yield [
                ObservationAddress(
                    epoch=self.epoch,
                    epoch_position=begin + slot,
                    row=int(row),
                    start=_counter("target-age", self.epoch, int(row)) % LEGAL_STARTS,
                )
                for slot, row in enumerate(rows)
            ]


def protocol_record() -> dict[str, object]:
    if TRAIN_ROWS % GLOBAL_BATCH or UPDATES_PER_EPOCH != 225:
        raise AssertionError("exposure no longer gives 225 updates per epoch")
    if PREFLIGHT_UPDATES != 4_500 or FULL_UPDATES != 225_000:
        raise AssertionError("optimizer clock drifted")
    return {
        "schema": "clean_observation_flow.protocol.v2",
        "seed": SEED,
        "address_fields": ["epoch", "epoch_position", "row", "start"],
        "name_or_integer_class_selector_present": False,
        "train_rows": TRAIN_ROWS,
        "global_batch": GLOBAL_BATCH,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "preflight_epochs": PREFLIGHT_EPOCHS,
        "preflight_updates": PREFLIGHT_UPDATES,
        "full_epochs": FULL_EPOCHS,
        "full_updates": FULL_UPDATES,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "adam_betas": list(ADAM_BETAS),
        "adam_eps": ADAM_EPS,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "input_noise_scale": INPUT_NOISE_SCALE,
        "scheduler": "OneCycleLR_two_raw_ticks_per_update",
        "scheduler_ticks_per_update": SCHEDULER_TICKS_PER_UPDATE,
        "scheduler_pct_start": SCHEDULER_PCT_START,
        "div_factor": DIV_FACTOR,
        "final_div_factor": FINAL_DIV_FACTOR,
        "held_out_split_accessed": False,
    }


__all__ = [
    "ObservationAddress",
    "ObservationEpochBatchSampler",
    "PREFLIGHT_EPOCHS",
    "PREFLIGHT_UPDATES",
    "FULL_UPDATES",
    "GLOBAL_BATCH",
    "noise_seed",
    "protocol_record",
]
