from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable


CHECKPOINT_LEDGER = "CHECKPOINT_REGISTRY.jsonl"
RETENTION_STATE = "CHECKPOINT_RETENTION.json"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint_e(?P<epoch>[0-9]{4})\.pt$")


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _metadata_id(
    *, completed_epoch: int, updates: int, preregistration: dict[str, object]
) -> str:
    payload = {
        "completed_epoch": int(completed_epoch),
        "updates": int(updates),
        "preregistration": preregistration,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def retained_epochs(
    completed_epoch: int,
    *,
    milestones: Iterable[int],
    rolling_keep: int,
) -> tuple[int, ...]:
    """Return permanent milestones plus the most recent completed epochs."""

    completed = int(completed_epoch)
    keep = int(rolling_keep)
    if completed < 1 or keep < 1:
        raise ValueError("completed_epoch and rolling_keep must be positive")
    retained = {
        int(epoch)
        for epoch in milestones
        if 1 <= int(epoch) <= completed
    }
    retained.update(range(max(1, completed - keep + 1), completed + 1))
    return tuple(sorted(retained))


def register_and_prune_checkpoint(
    output: Path,
    checkpoint: Path,
    *,
    completed_epoch: int,
    updates: int,
    preregistration: dict[str, object],
    milestones: Iterable[int],
    rolling_keep: int,
) -> dict[str, object]:
    """Register one saved checkpoint and evict only unprotected older files.

    The append-only ledger preserves both the creation and eviction history.
    A full byte hash is deliberately not reread every epoch: the atomic
    checkpoint payload already embeds the immutable training contract, while
    avoiding a second full checkpoint-sized I/O pass keeps registration cheap.
    """

    output = output.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.parent != output:
        raise ValueError("checkpoint must be directly inside its output directory")
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None or int(match.group("epoch")) != int(completed_epoch):
        raise ValueError("checkpoint filename and completed epoch disagree")
    milestone_set = {int(value) for value in milestones}
    record = {
        "schema": "typed_sparse_evolution.checkpoint_registry.event.v1",
        "event": "saved",
        "checkpoint": checkpoint.name,
        "completed_epoch": int(completed_epoch),
        "updates": int(updates),
        "bytes": checkpoint.stat().st_size,
        "checkpoint_metadata_id": _metadata_id(
            completed_epoch=completed_epoch,
            updates=updates,
            preregistration=preregistration,
        ),
        "permanent_milestone": int(completed_epoch) in milestone_set,
        "official_test_accessed": False,
    }
    ledger = output / CHECKPOINT_LEDGER
    _append_jsonl(ledger, record)

    protected = set(
        retained_epochs(
            completed_epoch,
            milestones=milestone_set,
            rolling_keep=rolling_keep,
        )
    )
    evicted: list[int] = []
    for candidate in sorted(output.glob("checkpoint_e*.pt")):
        candidate_match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if candidate_match is None:
            continue
        epoch = int(candidate_match.group("epoch"))
        if epoch in protected:
            continue
        candidate.unlink()
        evicted.append(epoch)
        _append_jsonl(
            ledger,
            {
                "schema": "typed_sparse_evolution.checkpoint_registry.event.v1",
                "event": "evicted_after_registration",
                "checkpoint": candidate.name,
                "completed_epoch": epoch,
                "retention_trigger_epoch": int(completed_epoch),
                "official_test_accessed": False,
            },
        )

    present = []
    for candidate in sorted(output.glob("checkpoint_e*.pt")):
        candidate_match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if candidate_match is not None:
            present.append(int(candidate_match.group("epoch")))
    state = {
        "schema": "typed_sparse_evolution.checkpoint_retention.v1",
        "completed_epoch": int(completed_epoch),
        "save_every_epoch": True,
        "rolling_keep": int(rolling_keep),
        "permanent_milestones": sorted(milestone_set),
        "retained_epochs": present,
        "evicted_this_update": evicted,
        "ledger": CHECKPOINT_LEDGER,
        "official_test_accessed": False,
    }
    _atomic_json(output / RETENTION_STATE, state)
    return state


def require_registered_checkpoint(
    output: Path,
    checkpoint: Path,
    *,
    completed_epoch: int,
    updates: int,
    preregistration: dict[str, object],
) -> dict[str, object]:
    """Fail closed unless ``checkpoint`` is the currently registered artifact.

    Registration is an append-only state machine.  A save activates a named
    checkpoint and a later eviction deactivates it.  Resume therefore cannot
    be authorized merely because a correctly named ``.pt`` file happens to be
    present in the output directory.
    """

    output = output.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.parent != output:
        raise RuntimeError("resume checkpoint is outside its registered output")
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None or int(match.group("epoch")) != int(completed_epoch):
        raise RuntimeError("resume checkpoint filename and payload epoch disagree")
    ledger = output / CHECKPOINT_LEDGER
    if not ledger.is_file():
        raise RuntimeError("resume checkpoint has no append-only registration ledger")

    active: dict[str, object] | None = None
    for line_number, raw in enumerate(
        ledger.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"checkpoint registration ledger is invalid at line {line_number}"
            ) from exc
        if not isinstance(event, dict) or event.get("checkpoint") != checkpoint.name:
            continue
        if event.get("event") == "saved":
            active = event
        elif event.get("event") == "evicted_after_registration":
            active = None

    if active is None:
        raise RuntimeError("resume checkpoint is not active in the registration ledger")
    expected = {
        "completed_epoch": int(completed_epoch),
        "updates": int(updates),
        "bytes": checkpoint.stat().st_size,
        "checkpoint_metadata_id": _metadata_id(
            completed_epoch=completed_epoch,
            updates=updates,
            preregistration=preregistration,
        ),
    }
    for key, value in expected.items():
        if active.get(key) != value:
            raise RuntimeError(f"resume checkpoint registration drifted at {key}")
    return active


__all__ = [
    "CHECKPOINT_LEDGER",
    "RETENTION_STATE",
    "register_and_prune_checkpoint",
    "require_registered_checkpoint",
    "retained_epochs",
]
