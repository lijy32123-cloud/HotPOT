from __future__ import annotations

import hashlib

from hotpot._internal.data_protocol.manifest import HISTORY
from hotpot._internal.data_protocol.sampler import (
    Base6EpochPlan,
    EpochAddress,
)


PHASE_SLOTS = 10


def _hash64(label: str, seed: int, epoch: int, virtual_index: int) -> int:
    payload = f"{label}:{seed}:{epoch}:{virtual_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class PhaseAwareBase6EpochPlan(Base6EpochPlan):
    """One initial-window anchor plus balanced early/middle/late starts.

    For every serialized trajectory occurrence, each ten-epoch cycle contains
    exactly one start=0 anchor and three visits to each non-initial phase.  The
    number of rows, update clock, lane packing and augmentation seeds remain
    unchanged.  This is a train-origin temporal design; no test data is read.
    """

    def _phase_start(self, legal_starts: int, epoch: int, virtual_index: int) -> int:
        if legal_starts <= 1:
            return 0
        offset = _hash64("phase-offset-v1", self.seed, 0, virtual_index) % PHASE_SLOTS
        slot = (int(epoch) + offset) % PHASE_SLOTS
        if slot == 0:
            return 0

        non_initial = list(range(1, legal_starts))
        # Integer thirds are explicit and exhaustive even for short FNO/CNS
        # trajectories.  Empty bins fall back to all non-initial starts.
        cut1 = (len(non_initial) + 2) // 3
        cut2 = (2 * len(non_initial) + 2) // 3
        bins = (
            non_initial[:cut1],
            non_initial[cut1:cut2],
            non_initial[cut2:],
        )
        phase = min((slot - 1) // 3, 2)
        candidates = bins[phase] or non_initial
        choice = _hash64("phase-choice-v1", self.seed, int(epoch), virtual_index)
        return candidates[choice % len(candidates)]

    def _decode(self, virtual_index: int, epoch_position: int, epoch: int) -> EpochAddress:
        base = super()._decode(virtual_index, epoch_position, epoch)
        entry = self.entries[base.dataset_index]
        legal_starts = max(entry.total_frames - (HISTORY + 1) + 1, 1)
        start = self._phase_start(legal_starts, epoch, virtual_index)
        return EpochAddress(
            epoch_position=base.epoch_position,
            dataset_index=base.dataset_index,
            data_index=base.data_index,
            start=start,
            noise_seed=base.noise_seed,
        )


__all__ = ["PHASE_SLOTS", "PhaseAwareBase6EpochPlan"]
