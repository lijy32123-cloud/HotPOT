from __future__ import annotations

import argparse
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

from hotpot._internal.data_protocol.cfdb_sidecar import CertifiedCFDBSidecar
from hotpot._internal.data_protocol.checkpointing import register_and_prune_checkpoint
from hotpot._internal.data_protocol.data import (
    Base6StorageDataset,
    DistributedAddressBatchSampler,
    operator_owned_augmentation,
)
from hotpot._internal.data_protocol.distributed import (
    ActiveGradientSynchronizer,
    broadcast_module_state,
)
from hotpot._internal.data_protocol.evaluation_manifest import evaluation_protocol_payload
from hotpot._internal.data_protocol.manifest import manifest_payload
from hotpot._internal.data_protocol.objective import operator_owned_relative_lp

from .base6 import Base6SemanticFieldFoundation
from .sampler import PhaseAwareBase6EpochPlan


SEED = 20260815
CHECKPOINT_SCHEMA = "semantic_field_program_foundation.checkpoint.v1.1"
MODEL_SYSTEM = "semantic_field_program_foundation_v1_1"
PREREGISTRATION_SCHEMA = "semantic_field_program_foundation.preregistration.v1.1"
MODEL_CLASS = Base6SemanticFieldFoundation
MILESTONES = (20, 100, 300, 600, 800, 1000)
TASK_ORDER = (
    "fno",
    "cns",
    "swe",
    "dr",
    "cfdb_cavity",
    "cfdb_cylinder",
    "cfdb_tube",
)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--cfdb-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--local-batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--noise-scale", type=float, default=5.0e-4)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    return parser


def arguments() -> argparse.Namespace:
    return add_arguments(argparse.ArgumentParser()).parse_args()


def _inside(path: Path, workspace: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"{label} must stay below workspace: {resolved}") from error
    return resolved


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()


def _init_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2:
        raise RuntimeError("the formal realization requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return local_rank, rank, world, torch.device("cuda", local_rank)


def _determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


def _cfdb_metric_index_by_row(
    sidecar: CertifiedCFDBSidecar, device: torch.device
) -> torch.Tensor:
    task_by_family = {"cavity": 4, "cylinder": 5, "tube": 6}
    return torch.tensor(
        [task_by_family[sidecar.family(row)] for row in range(len(sidecar.entries))],
        dtype=torch.long,
        device=device,
    )


def _task_indices(
    dataset_indices: torch.Tensor,
    row_indices: torch.Tensor,
    cfdb_metric_index: torch.Tensor,
) -> torch.Tensor:
    # Base6 datasets 0/1 share FNO; datasets 2/3/4 are CNS/SWE/DR.  Dataset 5
    # is resolved into its three independently reported CFDB programs.
    result = torch.where(dataset_indices <= 1, 0, dataset_indices - 1)
    cfdb = dataset_indices == 5
    result[cfdb] = cfdb_metric_index[row_indices[cfdb]]
    return result


def semantic_field_training_augmentation(
    history: torch.Tensor,
    noise_seeds: torch.Tensor,
    dataset_indices: torch.Tensor,
    *,
    non_cfdb_scale: float,
) -> torch.Tensor:
    """Retain the V1 transform outside CFDB and keep CFDB exactly clean."""
    augmented = operator_owned_augmentation(
        history,
        noise_seeds,
        dataset_indices,
        scale=non_cfdb_scale,
    )
    cfdb = dataset_indices.to(device=history.device).view(-1, 1, 1, 1, 1) == 5
    return torch.where(cfdb, history, augmented)


def main(args=None) -> None:
    args = arguments() if args is None else args
    if args.epochs < 1 or args.local_batch_size < 1 or args.smoke_updates < 0:
        raise ValueError("epochs/batch must be positive")
    if args.smoke_updates == 0 and (
        args.epochs != 1000
        or args.local_batch_size != 20
        or not math.isclose(args.learning_rate, 1.0e-3)
        or not math.isclose(args.noise_scale, 5.0e-4)
    ):
        raise ValueError("formal run is locked to Base6 E1000/B20/lr/non-CFDB-noise")

    _local_rank, rank, world, device = _init_distributed()
    _determinism(SEED + rank)
    workspace = args.workspace.resolve(strict=True)
    sidecar_path = _inside(args.cfdb_sidecar.resolve(strict=True), workspace, "CFDB sidecar")
    sidecar = CertifiedCFDBSidecar(sidecar_path)
    cfdb_metric_index = _cfdb_metric_index_by_row(sidecar, device)
    output = _inside(args.output, workspace, "output")
    resume = None
    if args.resume is not None:
        resume = _inside(args.resume.resolve(strict=True), workspace, "resume checkpoint")
        if resume.parent != output or not output.is_dir():
            raise ValueError("resume checkpoint must belong to the declared output")
    elif rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    model = MODEL_CLASS(workspace=workspace).to(device)
    broadcast_module_state(model, src=0)
    ledger = model.parameter_ledger()
    synchronizer = ActiveGradientSynchronizer(model)
    plan = PhaseAwareBase6EpochPlan(seed=SEED)
    sampler = DistributedAddressBatchSampler(
        plan,
        rank=rank,
        world_size=world,
        local_batch_size=args.local_batch_size,
    )
    dataset = Base6StorageDataset(workspace, sidecar)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if len(loader) != 545:
        raise RuntimeError(f"matched Base6 update clock drifted: {len(loader)}")
    optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.9), weight_decay=1.0e-6)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        div_factor=1.0e4,
        pct_start=0.2,
        final_div_factor=1.0e4,
        steps_per_epoch=len(loader),
        epochs=args.epochs,
    )
    preregistration = {
        "schema": PREREGISTRATION_SCHEMA,
        "system": MODEL_SYSTEM,
        "base6_manifest": manifest_payload(),
        "model": ledger,
        "operator_registry": model.model.operator_registry(),
        "world_size": world,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.local_batch_size * world,
        "updates_per_epoch": len(loader),
        "epochs": args.epochs,
        "seed": SEED,
        "optimizer": "Adam(lr_max=1e-3,betas=(0.9,0.9),weight_decay=1e-6)",
        "scheduler": "OneCycleLR(pct_start=0.2)",
        "sampler": {
            "schema": "semantic_field_program_foundation.phase_aware_base6.v1",
            "exact_released_sample_exposure": True,
            "compatibility_homogeneous_batches": True,
            "phase_schedule": "one start0 anchor plus three early/middle/late visits per ten-epoch cycle",
            "epoch0_fingerprint": sampler.fingerprint(),
        },
        "gradient_synchronization": "global_active_union_bucketed_mean_allreduce",
        "backward_calls_per_batch": 1,
        "gradient_firewall": False,
        "gradient_surgery": False,
        "prediction_gradient_to_shared_history": True,
        "shared_history_supervision": {
            "objective": "joint H10-to-H1 prediction only",
            "auxiliary_reconstruction": False,
            "future_target_accessed": False,
        },
        "training_target": "H10_to_H1",
        "input_augmentation": {
            "non_cfdb_noise_scale": args.noise_scale,
            "cfdb_noise_scale": 0.0,
            "cfdb_clean_input_exact": True,
        },
        "fixed_e1000_rollout_protocol": evaluation_protocol_payload(),
        "rollout_training_loss": False,
        "official_test_accessed": False,
    }

    start_epoch = 0
    updates = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if (
            checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or checkpoint.get("system") != MODEL_SYSTEM
            or checkpoint.get("official_test_accessed") is not False
        ):
            raise RuntimeError("resume checkpoint contract mismatch")
        prior = checkpoint.get("preregistration")
        if not isinstance(prior, dict):
            raise RuntimeError("resume checkpoint lacks preregistration")
        for key in ("base6_manifest", "model", "world_size", "local_batch_size", "epochs", "seed"):
            if prior.get(key) != preregistration.get(key):
                raise RuntimeError(f"resume preregistration drifted at {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["completed_epoch"])
        updates = int(checkpoint["updates"])
        if start_epoch < 1 or start_epoch >= args.epochs or updates != start_epoch * len(loader):
            raise RuntimeError("resume epoch/update clock drifted")
        broadcast_module_state(model, src=0)
    elif rank == 0:
        _atomic_json(output / "PREREGISTRATION.json", preregistration)

    stop = False
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_started = time.perf_counter()
        task_sum = torch.zeros(len(TASK_ORDER), dtype=torch.float64, device=device)
        task_count = torch.zeros_like(task_sum)
        prediction_sum = torch.zeros((), dtype=torch.float64, device=device)
        seen = torch.zeros_like(prediction_sum)
        sync_numel_max = 0
        sync_bytes_max = 0
        epoch_updates = 0
        for batch in loader:
            history = batch["history"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            dataset_indices = batch["dataset_index"].to(device, non_blocking=True)
            row_indices = batch["row_index"].to(device, non_blocking=True)
            conditions = batch["condition_value"].to(device, non_blocking=True)
            condition_mask = batch["condition_mask"].to(device, non_blocking=True)
            noisy = semantic_field_training_augmentation(
                history,
                batch["noise_seed"],
                batch["dataset_index"],
                non_cfdb_scale=args.noise_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                noisy,
                dataset_indices,
                row_indices,
                condition_values=conditions,
                condition_mask=condition_mask,
                cfdb_sidecar=sidecar,
            )
            per_sample = operator_owned_relative_lp(
                prediction, target, dataset_indices, reduction="none"
            )
            prediction_loss = per_sample.sum()
            prediction_loss.backward()
            sync = synchronizer.synchronize()
            sync_numel_max = max(sync_numel_max, sync.active_parameter_numel)
            sync_bytes_max = max(sync_bytes_max, sync.communicated_bytes)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0e4)
            finite = (
                torch.isfinite(per_sample).all()
                & torch.isfinite(grad_norm)
            ).to(dtype=torch.int64)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if int(finite.item()) != 1:
                raise FloatingPointError("non-finite training state")
            optimizer.step()
            scheduler.step()
            updates += 1
            epoch_updates += 1

            prediction_sum += prediction_loss.detach().double()
            seen += history.shape[0]
            task_indices = _task_indices(
                dataset_indices, row_indices, cfdb_metric_index
            )
            task_sum.scatter_add_(0, task_indices, per_sample.detach().double())
            task_count += torch.bincount(
                task_indices, minlength=len(TASK_ORDER)
            ).to(dtype=torch.float64)
            if args.smoke_updates and updates >= args.smoke_updates:
                stop = True
                break

        elapsed = torch.tensor(time.perf_counter() - epoch_started, dtype=torch.float64, device=device)
        sync_max = torch.tensor((sync_numel_max, sync_bytes_max), dtype=torch.int64, device=device)
        for value in (task_sum, task_count, prediction_sum, seen):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(sync_max, op=dist.ReduceOp.MAX)
        metrics = {
            "epoch": epoch + 1,
            "updates": updates,
            "seconds_max_rank": float(elapsed),
            "seconds_per_update_max_rank": float(elapsed) / max(1, epoch_updates),
            "lr": optimizer.param_groups[0]["lr"],
            "prediction_relative_l2_per_sample": float(prediction_sum / seen.clamp_min(1)),
            "task_relative_l2": {
                name: float(task_sum[index] / task_count[index].clamp_min(1))
                for index, name in enumerate(TASK_ORDER)
            },
            "max_active_training_parameters_per_batch": int(sync_max[0]),
            "max_communicated_bytes_per_batch": int(sync_max[1]),
            "gradient_firewall": False,
            "cfdb_input_clean_exact": True,
            "backward_calls_per_batch": 1,
            "official_test_accessed": False,
        }
        if rank == 0:
            _append_jsonl(output / "metrics.jsonl", metrics)
            _atomic_json(output / "HEARTBEAT.json", metrics)
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "system": MODEL_SYSTEM,
                "completed_epoch": epoch + 1,
                "updates": updates,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "preregistration": preregistration,
                "official_test_accessed": False,
            }
            checkpoint_path = output / f"checkpoint_e{epoch + 1:04d}.pt"
            _atomic_save(checkpoint_path, checkpoint)
            register_and_prune_checkpoint(
                output,
                checkpoint_path,
                completed_epoch=epoch + 1,
                updates=updates,
                preregistration=preregistration,
                milestones=MILESTONES,
                rolling_keep=3,
            )
        dist.barrier()
        if stop:
            break

    if rank == 0:
        _atomic_json(
            output / "COMPLETE.json",
            {
                "status": "SMOKE_COMPLETE" if stop else "E1000_COMPLETE",
                "system": MODEL_SYSTEM,
                "completed_epoch": epoch + 1,
                "updates": updates,
                "official_test_accessed": False,
                "scientific_checkpoint_selected": False,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
