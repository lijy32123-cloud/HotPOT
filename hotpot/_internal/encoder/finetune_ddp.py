from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from hotpot._internal.data_protocol.cfdb_sidecar import (
    CertifiedCFDBSidecar,
)
from hotpot._internal.data_protocol.data import (
    Base6StorageDataset,
    DistributedAddressBatchSampler,
)
from hotpot._internal.data_protocol.distributed import (
    ActiveGradientSynchronizer,
    broadcast_module_state,
)
from hotpot._internal.data_protocol.manifest import BASE6
from hotpot._internal.data_protocol.objective import (
    operator_owned_relative_lp,
)

from .accounting import parameter_ledger
from .base6 import Base6SemanticFieldFoundation
from .sampler import PhaseAwareBase6EpochPlan


SEED = 20260815
SOURCE_CHECKPOINT_SCHEMA = "semantic_field_program_foundation.checkpoint.v1.1"
SOURCE_MODEL_SYSTEM = "semantic_field_program_foundation_v1_1"
FINETUNE_CHECKPOINT_SCHEMA = "semantic_field_program_foundation.finetune.checkpoint.v1"
FINETUNE_SNAPSHOT_SCHEMA = "semantic_field_program_foundation.finetune.active_snapshot.v1"
FINETUNE_MODEL_SYSTEM = "semantic_field_program_foundation_v1_1_known_task_finetune"
FIXED_UPDATE_CHECKPOINT_SCHEMA = (
    "semantic_field_program_foundation.finetune.fixed_update.checkpoint.v2"
)
FIXED_UPDATE_SNAPSHOT_SCHEMA = (
    "semantic_field_program_foundation.finetune.fixed_update.active_snapshot.v2"
)
MILESTONES = (20, 50, 100, 150, 200)
REFERENCE_FINETUNE_UPDATES = 5_000
REFERENCE_LOCAL_BATCH_SIZE = 20
FIXED_UPDATE_WARMUP_FRACTION = 0.2
CORRECTED_FINETUNE_MAX_LR = 1.0e-4
CORRECTED_E200_MAX_LR = 1.0e-4
LEGACY_E200_MAX_LR = 1.0e-3


@dataclass(frozen=True)
class FineTuneTarget:
    key: str
    dataset_index: int
    family: str

    @property
    def train_rows(self) -> int:
        return int(BASE6[self.dataset_index].weighted_rows)


TARGETS = {
    value.key: value
    for value in (
        FineTuneTarget("ns2d_fno_1e-5", 0, "fno"),
        FineTuneTarget("ns2d_fno_1e-3", 1, "fno"),
        FineTuneTarget("ns2d_pdb_M1e-1_eta1e-2_zeta1e-2", 2, "cns"),
        FineTuneTarget("swe_pdb", 3, "swe"),
        FineTuneTarget("dr_pdb", 4, "dr"),
        FineTuneTarget("cfdbench", 5, "cfdb"),
    )
}


class TargetPhaseAwareEpochPlan(PhaseAwareBase6EpochPlan):
    """One full target-dataset pass with V1.1 phase-aware H10 windows."""

    def __init__(self, dataset_index: int, *, seed: int = SEED) -> None:
        super().__init__(seed=seed)
        if dataset_index < 0 or dataset_index >= len(BASE6):
            raise ValueError(f"invalid Base6 dataset index: {dataset_index}")
        self.dataset_index = int(dataset_index)
        self.target_begin = (
            0 if self.dataset_index == 0 else self.cumulative[self.dataset_index - 1]
        )
        self.target_size = int(self.weighted_sizes[self.dataset_index])

    def epoch(self, epoch: int):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            (self.seed + 50_000_003 * self.dataset_index + int(epoch)) % (2**63 - 1)
        )
        order = torch.randperm(self.target_size, generator=generator).tolist()
        return tuple(
            self._decode(self.target_begin + local, position, int(epoch))
            for position, local in enumerate(order)
        )

    def batches(
        self,
        epoch: int,
        *,
        global_batch_size: int = 40,
        allow_remainder: bool = False,
    ):
        if global_batch_size < 1:
            raise ValueError("global_batch_size must be positive")
        addresses = self.epoch(epoch)
        if len(addresses) % global_batch_size and not allow_remainder:
            raise RuntimeError("target exposure is not divisible by the global batch size")
        return tuple(
            tuple(addresses[begin : begin + global_batch_size])
            for begin in range(0, len(addresses), global_batch_size)
        )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--cfdb-sidecar", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--local-batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--gradient-microbatch-size", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-updates", type=int, default=0)
    return parser.parse_args()


def _inside(path: Path, workspace: Path, label: str, *, strict: bool) -> Path:
    resolved = path.resolve(strict=strict)
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"{label} must stay below workspace: {resolved}") from error
    return resolved


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _init_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (1, 2):
        raise RuntimeError("known-task fine-tuning requires one or two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return local_rank, rank, world, device


def _fixed_update_clock(
    *, max_updates: int, updates_per_epoch: int
) -> dict[str, int | float]:
    if max_updates < 1 or updates_per_epoch < 1:
        raise ValueError("fixed-update clock requires positive update counts")
    return {
        "target_updates": int(max_updates),
        "loop_epochs": int(math.ceil(max_updates / updates_per_epoch)),
        "complete_full_epochs": int(max_updates // updates_per_epoch),
        "final_partial_epoch_updates": int(max_updates % updates_per_epoch),
        "warmup_fraction": FIXED_UPDATE_WARMUP_FRACTION,
    }


def _microbatch_slices(batch_size: int, microbatch_size: int) -> tuple[slice, ...]:
    if batch_size < 1 or microbatch_size < 1 or batch_size % microbatch_size:
        raise ValueError("physical batch must be divisible by the gradient microbatch")
    return tuple(
        slice(begin, begin + microbatch_size)
        for begin in range(0, batch_size, microbatch_size)
    )


def _e200_execution_profile(
    *,
    epochs: int,
    warmup_epochs: int,
    local_batch_size: int,
    gradient_microbatch_size: int,
    learning_rate: float,
) -> str:
    """Fail closed on the two registered E200 execution profiles."""

    if epochs != 200 or warmup_epochs != 40:
        raise ValueError("E200 fine-tuning requires exactly E200/W40")
    if (
        local_batch_size == 20
        and gradient_microbatch_size in (0, 20)
        and math.isclose(learning_rate, LEGACY_E200_MAX_LR)
    ):
        return "legacy_e200_ddp_lr1e-3"
    if (
        local_batch_size == 40
        and gradient_microbatch_size == REFERENCE_LOCAL_BATCH_SIZE
        and math.isclose(learning_rate, CORRECTED_E200_MAX_LR)
    ):
        return "corrected_e200_task_parallel_lr1e-4"
    raise ValueError(
        "E200 execution must be legacy DDP B40/lr1e-3 or corrected "
        "single-rank B40/micro-B20/lr1e-4"
    )


def _determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


def configure_target_trainability(
    model: Base6SemanticFieldFoundation, target: FineTuneTarget
) -> dict[str, object]:
    shared_prefix = "model.shared_analyzer."
    private_prefixes = (
        f"model.private_mixers.{target.family}.",
        f"model.action_context_decoders.{target.family}.",
        f"model.programs.{target.family}.",
    )
    names: list[str] = []
    shared = 0
    private = 0
    for name, parameter in model.named_parameters():
        is_shared = name.startswith(shared_prefix)
        is_private = name.startswith(private_prefixes)
        trainable = is_shared or is_private
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
            if is_shared:
                shared += parameter.numel()
            else:
                private += parameter.numel()
    if not names or shared == 0 or private == 0:
        raise RuntimeError("target trainability did not select one complete active path")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not (name.startswith(shared_prefix) or name.startswith(private_prefixes))
    ]
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected[:4]}")
    return {
        "schema": "semantic_field_program_foundation.finetune_trainability.v1",
        "target": target.key,
        "family": target.family,
        "shared_trainable_parameters": int(shared),
        "private_trainable_parameters": int(private),
        "trainable_parameters": int(shared + private),
        "trainable_parameter_tensors": len(names),
        "inactive_private_paths_frozen": True,
        "shared_analyzer_trainable": True,
    }


def _active_parameter_state(
    model: Base6SemanticFieldFoundation,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_active_parameter_state(
    model: Base6SemanticFieldFoundation, state: dict[str, torch.Tensor]
) -> None:
    named = dict(model.named_parameters())
    expected = {name for name, value in named.items() if value.requires_grad}
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise RuntimeError(
            f"active checkpoint keys drifted: missing={missing[:4]} extra={extra[:4]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def _load_base_checkpoint(
    model: Base6SemanticFieldFoundation,
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256.lower()
    ):
        raise ValueError("base SHA256 must be a lowercase hexadecimal digest")
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    if (
        checkpoint.get("schema") != SOURCE_CHECKPOINT_SCHEMA
        or checkpoint.get("system") != SOURCE_MODEL_SYSTEM
        or int(checkpoint.get("completed_epoch", -1)) != 1000
        or int(checkpoint.get("updates", -1)) != 545_000
        or checkpoint.get("official_test_accessed") is not False
    ):
        raise RuntimeError("source E1000 checkpoint contract mismatch")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise RuntimeError("source E1000 checkpoint has no model state")
    model.load_state_dict(state, strict=True)
    metadata = {
        "path": str(path),
        "sha256": expected_sha256,
        "bytes": path.stat().st_size,
        "completed_epoch": 1000,
        "updates": 545_000,
        "schema": SOURCE_CHECKPOINT_SCHEMA,
    }
    del state, checkpoint
    gc.collect()
    return metadata


def main() -> None:
    args = arguments()
    fixed_update_mode = args.max_updates > 0
    if (
        args.epochs < 1
        or args.warmup_epochs < 1
        or args.warmup_epochs >= args.epochs
        or args.local_batch_size < 1
        or args.checkpoint_every < 1
        or args.max_updates < 0
        or args.gradient_microbatch_size < 0
        or args.smoke_updates < 0
    ):
        raise ValueError("invalid fine-tuning schedule")
    e200_profile = None
    if fixed_update_mode:
        if args.smoke_updates == 0:
            if (
                args.max_updates != REFERENCE_FINETUNE_UPDATES
                or not math.isclose(args.learning_rate, CORRECTED_FINETUNE_MAX_LR)
            ):
                raise ValueError("formal fixed-update fine-tuning is locked to U5000/lr1e-4")
    else:
        e200_profile = _e200_execution_profile(
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            local_batch_size=args.local_batch_size,
            gradient_microbatch_size=args.gradient_microbatch_size,
            learning_rate=args.learning_rate,
        )

    _local_rank, rank, world, device = _init_distributed()
    if fixed_update_mode:
        expected_local_batch = 40 if world == 1 else 20
        expected_microbatch = REFERENCE_LOCAL_BATCH_SIZE
        if (
            args.local_batch_size != expected_local_batch
            or args.gradient_microbatch_size != expected_microbatch
            or args.local_batch_size * world != 40
        ):
            raise ValueError(
                "fixed-update execution requires global-B40 and gradient micro-B20"
            )
    elif e200_profile == "corrected_e200_task_parallel_lr1e-4":
        if world != 1:
            raise RuntimeError("corrected E200 task-parallel execution requires one rank")
    elif world != 2:
        raise RuntimeError("legacy E200 fine-tuning requires exactly two ranks")
    gradient_microbatch_size = (
        args.gradient_microbatch_size or args.local_batch_size
    )
    _determinism(SEED + rank)
    workspace = args.workspace.resolve(strict=True)
    sidecar_path = _inside(args.cfdb_sidecar, workspace, "CFDB sidecar", strict=True)
    base_checkpoint = _inside(
        args.base_checkpoint, workspace, "base checkpoint", strict=True
    )
    output = _inside(args.output, workspace, "fine-tune output", strict=False)
    resume = None
    if args.resume is not None:
        resume = _inside(args.resume, workspace, "fine-tune resume", strict=True)
        if resume.parent != output or not output.is_dir():
            raise ValueError("resume checkpoint must belong to the target output")
    elif rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    target = TARGETS[args.target]
    sidecar = CertifiedCFDBSidecar(sidecar_path)
    model = Base6SemanticFieldFoundation(workspace=workspace).to(device)
    source = _load_base_checkpoint(
        model, base_checkpoint, expected_sha256=args.base_sha256.lower()
    )
    broadcast_module_state(model, src=0)
    trainability = configure_target_trainability(model, target)
    full_ledger = parameter_ledger(model.model)

    plan = TargetPhaseAwareEpochPlan(target.dataset_index, seed=SEED)
    sampler = DistributedAddressBatchSampler(
        plan,
        rank=rank,
        world_size=world,
        local_batch_size=args.local_batch_size,
    )
    expected_steps = math.ceil(
        target.train_rows / (args.local_batch_size * world)
    )
    if len(sampler) != expected_steps:
        raise RuntimeError(f"target update clock drifted: {len(sampler)} != {expected_steps}")
    if fixed_update_mode:
        schedule = _fixed_update_clock(
            max_updates=args.max_updates, updates_per_epoch=len(sampler)
        )
        target_updates = int(schedule["target_updates"])
        loop_epochs = int(schedule["loop_epochs"])
    else:
        schedule = {
            "target_updates": int(args.epochs * len(sampler)),
            "loop_epochs": int(args.epochs),
            "complete_full_epochs": int(args.epochs),
            "final_partial_epoch_updates": 0,
            "warmup_fraction": float(args.warmup_epochs / args.epochs),
        }
        target_updates = int(schedule["target_updates"])
        loop_epochs = int(schedule["loop_epochs"])
    if args.smoke_updates > target_updates:
        raise ValueError("smoke update limit exceeds the registered schedule")
    update_limit = args.smoke_updates or target_updates
    dataset = Base6StorageDataset(workspace, sidecar)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    trainable_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if sum(value.numel() for value in trainable_parameters) != trainability["trainable_parameters"]:
        raise RuntimeError("optimizer parameter set does not match trainability ledger")
    optimizer = Adam(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.9),
        weight_decay=1.0e-6,
    )
    if fixed_update_mode:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            div_factor=1.0e4,
            pct_start=FIXED_UPDATE_WARMUP_FRACTION,
            final_div_factor=1.0e4,
            total_steps=target_updates,
        )
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            div_factor=1.0e4,
            pct_start=args.warmup_epochs / args.epochs,
            final_div_factor=1.0e4,
            steps_per_epoch=len(loader),
            epochs=args.epochs,
        )
    synchronizer = ActiveGradientSynchronizer(model)
    preregistration = {
        "schema": (
            "semantic_field_program_foundation.known_task_fixed_update_finetune.v2"
            if fixed_update_mode
            else (
                "semantic_field_program_foundation.known_task_finetune.corrected_e200.v2"
                if e200_profile == "corrected_e200_task_parallel_lr1e-4"
                else "semantic_field_program_foundation.known_task_finetune.v1"
            )
        ),
        "target": target.key,
        "dataset_index": target.dataset_index,
        "family": target.family,
        "source_checkpoint": source,
        "source_model_ledger": full_ledger,
        "trainability": trainability,
        "world_size": world,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.local_batch_size * world,
        "gradient_microbatch_size": gradient_microbatch_size,
        "optimizer_gradient_reference_local_batch_size": REFERENCE_LOCAL_BATCH_SIZE,
        "task_parallel_single_rank": bool(world == 1),
        "execution_profile": (
            "corrected_u5000_task_parallel_lr1e-4"
            if fixed_update_mode
            else e200_profile
        ),
        "train_rows_per_epoch": target.train_rows,
        "updates_per_epoch": len(loader),
        "epochs": loop_epochs,
        "complete_full_epochs": schedule["complete_full_epochs"],
        "final_partial_epoch_updates": schedule["final_partial_epoch_updates"],
        "target_updates": target_updates,
        "warmup_epochs": args.warmup_epochs if not fixed_update_mode else None,
        "warmup_fraction": schedule["warmup_fraction"],
        "optimizer": (
            f"Adam(lr_max={args.learning_rate:.12g},betas=(0.9,0.9),weight_decay=1e-6)"
        ),
        "scheduler": (
            "OneCycleLR(total_steps=5000,pct_start=0.2)"
            if fixed_update_mode
            else "OneCycleLR(pct_start=0.2)"
        ),
        "training_target": "H10_to_H1",
        "training_noise_scale": 0.0,
        "phase_aware_target_only_sampling": True,
        "full_target_train_exposure_each_epoch": not bool(
            schedule["final_partial_epoch_updates"]
        ),
        "final_epoch_may_be_partial": bool(schedule["final_partial_epoch_updates"]),
        "update_budget_equalized_to_reference_tasks": fixed_update_mode,
        "shared_analyzer_trainable": True,
        "other_private_paths_frozen": True,
        "checkpoint_selection": (
            "fixed_update_5000" if fixed_update_mode else "fixed_epoch_200"
        ),
        "official_test_accessed": False,
    }

    start_epoch = 0
    updates = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        expected_resume_schema = (
            FIXED_UPDATE_CHECKPOINT_SCHEMA
            if fixed_update_mode
            else FINETUNE_CHECKPOINT_SCHEMA
        )
        if (
            checkpoint.get("schema") != expected_resume_schema
            or checkpoint.get("target") != target.key
            or checkpoint.get("base_checkpoint", {}).get("sha256") != args.base_sha256.lower()
            or checkpoint.get("official_test_accessed") is not False
        ):
            raise RuntimeError("fine-tune resume contract mismatch")
        _restore_active_parameter_state(model, checkpoint["active_model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["completed_epoch"])
        updates = int(checkpoint["updates"])
        if (
            start_epoch < 1
            or start_epoch >= loop_epochs
            or updates != start_epoch * len(loader)
            or updates >= target_updates
        ):
            raise RuntimeError("fine-tune resume epoch/update clock drifted")
        del checkpoint
        broadcast_module_state(model, src=0)
    elif rank == 0:
        _atomic_json(output / "PREREGISTRATION.json", preregistration)

    stop = False
    completed_full_epochs = start_epoch
    last_epoch_number = start_epoch
    final_partial_epoch_updates = 0
    for epoch in range(start_epoch, loop_epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_started = time.perf_counter()
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        seen = torch.zeros((), dtype=torch.float64, device=device)
        sync_numel_max = 0
        sync_bytes_max = 0
        epoch_updates = 0
        for batch in loader:
            history = batch["history"].to(device, non_blocking=True)
            target_state = batch["target"].to(device, non_blocking=True)
            dataset_indices = batch["dataset_index"].to(device, non_blocking=True)
            row_indices = batch["row_index"].to(device, non_blocking=True)
            conditions = batch["condition_value"].to(device, non_blocking=True)
            condition_mask = batch["condition_mask"].to(device, non_blocking=True)
            if not bool((dataset_indices == target.dataset_index).all()):
                raise RuntimeError("target-only sampler emitted another dataset")

            optimizer.zero_grad(set_to_none=True)
            batch_size = int(history.shape[0])
            slices = _microbatch_slices(batch_size, gradient_microbatch_size)
            gradient_scale = REFERENCE_LOCAL_BATCH_SIZE / batch_size
            per_sample_parts: list[torch.Tensor] = []
            for selection in slices:
                prediction = model(
                    history[selection],
                    dataset_indices[selection],
                    row_indices[selection],
                    condition_values=conditions[selection],
                    condition_mask=condition_mask[selection],
                    cfdb_sidecar=sidecar,
                )
                per_sample_part = operator_owned_relative_lp(
                    prediction,
                    target_state[selection],
                    dataset_indices[selection],
                    reduction="none",
                )
                (per_sample_part.sum() * gradient_scale).backward()
                per_sample_parts.append(per_sample_part.detach())
            per_sample = torch.cat(per_sample_parts)
            prediction_loss = per_sample.sum()
            sync = synchronizer.synchronize()
            sync_numel_max = max(sync_numel_max, sync.active_parameter_numel)
            sync_bytes_max = max(sync_bytes_max, sync.communicated_bytes)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0e4)
            finite = (
                torch.isfinite(per_sample).all() & torch.isfinite(grad_norm)
            ).to(dtype=torch.int64)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if int(finite.item()) != 1:
                raise FloatingPointError("non-finite fine-tuning state")
            optimizer.step()
            scheduler.step()
            updates += 1
            epoch_updates += 1
            loss_sum += prediction_loss.detach().double()
            seen += history.shape[0]
            if updates >= update_limit:
                stop = True
                break

        elapsed = torch.tensor(
            time.perf_counter() - epoch_started, dtype=torch.float64, device=device
        )
        sync_max = torch.tensor(
            (sync_numel_max, sync_bytes_max), dtype=torch.int64, device=device
        )
        for value in (loss_sum, seen):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(sync_max, op=dist.ReduceOp.MAX)
        last_epoch_number = epoch + 1
        full_epoch_completed = epoch_updates == len(loader)
        completed_full_epochs = epoch + 1 if full_epoch_completed else epoch
        final_partial_epoch_updates = 0 if full_epoch_completed else epoch_updates
        metrics = {
            "target": target.key,
            "epoch": last_epoch_number,
            "completed_full_epochs": completed_full_epochs,
            "full_epoch_completed": full_epoch_completed,
            "partial_epoch_updates": final_partial_epoch_updates,
            "updates": updates,
            "target_updates": target_updates,
            "seconds_max_rank": float(elapsed),
            "seconds_per_update_max_rank": float(elapsed) / max(1, epoch_updates),
            "lr": optimizer.param_groups[0]["lr"],
            "train_relative_l2_per_sample": float(loss_sum / seen.clamp_min(1)),
            "seen_train_rows": int(seen.item()),
            "active_gradient_parameters_max": int(sync_max[0]),
            "communicated_bytes_per_batch_max": int(sync_max[1]),
            "training_noise_scale": 0.0,
            "official_test_accessed": False,
        }
        if rank == 0:
            _append_jsonl(output / "metrics.jsonl", metrics)
            _atomic_json(output / "HEARTBEAT.json", metrics)
            should_save = (
                stop
                or (
                    full_epoch_completed
                    and completed_full_epochs % args.checkpoint_every == 0
                )
                or updates == target_updates
            )
            if should_save:
                active_state = _active_parameter_state(model)
                resume_payload = {
                    "schema": (
                        FIXED_UPDATE_CHECKPOINT_SCHEMA
                        if fixed_update_mode
                        else FINETUNE_CHECKPOINT_SCHEMA
                    ),
                    "system": FINETUNE_MODEL_SYSTEM,
                    "target": target.key,
                    "completed_epoch": completed_full_epochs,
                    "last_epoch_number": last_epoch_number,
                    "partial_epoch_updates": final_partial_epoch_updates,
                    "updates": updates,
                    "target_updates": target_updates,
                    "base_checkpoint": source,
                    "active_model_state": active_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "preregistration": preregistration,
                    "official_test_accessed": False,
                }
                _atomic_save(output / "checkpoint_resume.pt", resume_payload)
                if (
                    (not fixed_update_mode and last_epoch_number in MILESTONES)
                    or updates == target_updates
                ):
                    snapshot = {
                        key: value
                        for key, value in resume_payload.items()
                        if key not in ("optimizer", "scheduler")
                    }
                    snapshot["schema"] = (
                        FIXED_UPDATE_SNAPSHOT_SCHEMA
                        if fixed_update_mode
                        else FINETUNE_SNAPSHOT_SCHEMA
                    )
                    snapshot["requires_base_checkpoint_for_materialization"] = True
                    snapshot_name = (
                        f"active_u{updates:06d}.pt"
                        if fixed_update_mode
                        else f"active_e{last_epoch_number:04d}.pt"
                    )
                    _atomic_save(
                        output / snapshot_name, snapshot
                    )
        dist.barrier()
        if stop:
            break

    if rank == 0:
        if args.smoke_updates:
            status = "SMOKE_COMPLETE"
            fixed_selection = None
        elif fixed_update_mode and updates == target_updates:
            status = f"U{target_updates:06d}_COMPLETE"
            fixed_selection = f"U{target_updates}"
        elif not fixed_update_mode and completed_full_epochs == args.epochs:
            status = "E0200_COMPLETE"
            fixed_selection = "E200"
        else:
            raise RuntimeError("fine-tune stopped before its registered endpoint")
        _atomic_json(
            output / "COMPLETE.json",
            {
                "status": status,
                "system": FINETUNE_MODEL_SYSTEM,
                "target": target.key,
                "completed_epoch": completed_full_epochs,
                "last_epoch_number": last_epoch_number,
                "partial_epoch_updates": final_partial_epoch_updates,
                "updates": updates,
                "target_updates": target_updates,
                "base_checkpoint_sha256": args.base_sha256.lower(),
                "fixed_checkpoint_selection": fixed_selection,
                "official_test_accessed": False,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
