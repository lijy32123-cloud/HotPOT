from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn


DEFAULT_BUCKET_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class GradientSyncAudit:
    active_parameter_tensors: int
    active_parameter_numel: int
    communicated_bytes: int
    buckets: int
    world_size: int


def _communication_view(value: Tensor) -> Tensor:
    if value.is_complex():
        return torch.view_as_real(value).reshape(-1)
    return value.reshape(-1)


def _restore_view(flat: Tensor, reference: Tensor) -> Tensor:
    if reference.is_complex():
        # A bucket slice can be contiguous while retaining an odd storage
        # offset.  view_as_complex requires an even offset, so materialize a
        # fresh offset-zero real view before restoring the complex tensor.
        real = flat.reshape(*reference.shape, 2).clone()
        return torch.view_as_complex(real)
    return flat.reshape_as(reference)


def _bucketed_broadcast(tensors: list[Tensor], *, src: int, bucket_bytes: int) -> int:
    buckets = 0
    pending: list[Tensor] = []
    pending_bytes = 0
    pending_dtype: torch.dtype | None = None

    def flush() -> None:
        nonlocal buckets, pending, pending_bytes, pending_dtype
        if not pending:
            return
        views = [_communication_view(value) for value in pending]
        flat = torch.cat(views)
        dist.broadcast(flat, src=src)
        offset = 0
        with torch.no_grad():
            for value, view in zip(pending, views, strict=True):
                length = view.numel()
                value.copy_(_restore_view(flat[offset : offset + length], value))
                offset += length
        buckets += 1
        pending = []
        pending_bytes = 0
        pending_dtype = None

    for value in tensors:
        view = _communication_view(value)
        size = view.numel() * view.element_size()
        if pending and (view.dtype != pending_dtype or pending_bytes + size > bucket_bytes):
            flush()
        pending.append(value)
        pending_bytes += size
        pending_dtype = view.dtype
    flush()
    return buckets


def broadcast_module_state(
    module: nn.Module,
    *,
    src: int = 0,
    bucket_bytes: int = DEFAULT_BUCKET_BYTES,
) -> int:
    """Make every rank start from the exact rank-``src`` model state."""

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return 0
    tensors = list(module.parameters()) + list(module.buffers())
    return _bucketed_broadcast(tensors, src=src, bucket_bytes=int(bucket_bytes))


class ActiveGradientSynchronizer:
    """Mean-reduce the union of parameters used by the current lane."""

    def __init__(
        self,
        module: nn.Module,
        *,
        bucket_bytes: int = DEFAULT_BUCKET_BYTES,
    ) -> None:
        self.parameters = tuple(module.parameters())
        self.bucket_bytes = int(bucket_bytes)
        if self.bucket_bytes < 1:
            raise ValueError("bucket_bytes must be positive")

    def synchronize(self) -> GradientSyncAudit:
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if not self.parameters:
            return GradientSyncAudit(0, 0, 0, 0, world_size)
        device = self.parameters[0].device
        local_mask = torch.tensor(
            [parameter.grad is not None for parameter in self.parameters],
            device=device,
            dtype=torch.int32,
        )
        if world_size > 1:
            dist.all_reduce(local_mask, op=dist.ReduceOp.MAX)
        active_indices = local_mask.nonzero(as_tuple=False).flatten().tolist()
        if world_size == 1:
            active_numel = sum(self.parameters[index].numel() for index in active_indices)
            return GradientSyncAudit(
                len(active_indices), active_numel, 0, 0, world_size
            )

        buckets = 0
        communicated = 0
        active_numel = 0
        pending: list[tuple[nn.Parameter, Tensor, Tensor]] = []
        pending_bytes = 0
        pending_dtype: torch.dtype | None = None

        def flush() -> None:
            nonlocal buckets, communicated, pending, pending_bytes, pending_dtype
            if not pending:
                return
            flat = torch.cat([view for _parameter, _reference, view in pending])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
            offset = 0
            for parameter, reference, view in pending:
                length = view.numel()
                restored = _restore_view(flat[offset : offset + length], reference)
                if parameter.grad is None:
                    parameter.grad = restored
                else:
                    parameter.grad.copy_(restored)
                offset += length
            buckets += 1
            communicated += flat.numel() * flat.element_size()
            pending = []
            pending_bytes = 0
            pending_dtype = None

        for index in active_indices:
            parameter = self.parameters[index]
            reference = parameter.grad
            if reference is None:
                reference = torch.zeros_like(parameter)
            view = _communication_view(reference)
            size = view.numel() * view.element_size()
            if pending and (
                view.dtype != pending_dtype or pending_bytes + size > self.bucket_bytes
            ):
                flush()
            pending.append((parameter, reference, view))
            pending_bytes += size
            pending_dtype = view.dtype
            active_numel += parameter.numel()
        flush()
        return GradientSyncAudit(
            active_parameter_tensors=len(active_indices),
            active_parameter_numel=active_numel,
            communicated_bytes=communicated,
            buckets=buckets,
            world_size=world_size,
        )


def distributed_equivalence_probe(device: torch.device) -> dict[str, object]:
    """Compare sparse synchronization with the exact DDP mean-gradient rule."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed parity probe requires an initialized group")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    class _ToySparseBank(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Parameter(torch.tensor((0.3, -0.2, 0.7)))
            self.left = nn.Parameter(torch.tensor((0.4, -0.1)))
            self.right = nn.Parameter(torch.tensor((-0.6, 0.8)))
            self.spectral = nn.Parameter(
                torch.tensor((0.2 + 0.5j, -0.4 + 0.1j), dtype=torch.complex64)
            )
            self.idle = nn.Parameter(torch.tensor((1.0,)))

        def loss(self) -> Tensor:
            scale = float(rank + 1)
            value = scale * self.shared.square().sum()
            value = value + scale * self.spectral.abs().square().sum()
            value = value + (
                self.left.square().sum() if rank == 0 else self.right.square().sum()
            )
            return value

    torch.manual_seed(1701 + rank)
    model = _ToySparseBank().to(device)
    broadcast_module_state(model, src=0, bucket_bytes=64)
    model.loss().backward()
    expected: dict[str, Tensor | None] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            value = torch.zeros_like(parameter)
        else:
            value = parameter.grad.detach().clone()
        used = torch.tensor(
            int(parameter.grad is not None), device=device, dtype=torch.int32
        )
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
        if int(used.item()) == 0:
            expected[name] = None
            continue
        communication = _communication_view(value).clone()
        dist.all_reduce(communication, op=dist.ReduceOp.SUM)
        communication.div_(world_size)
        expected[name] = _restore_view(communication, value).clone()
    audit = ActiveGradientSynchronizer(model, bucket_bytes=64).synchronize()
    maximum = torch.zeros((), device=device)
    inactive_preserved = True
    for name, parameter in model.named_parameters():
        reference = expected[name]
        if reference is None:
            inactive_preserved &= parameter.grad is None
            continue
        if parameter.grad is None:
            maximum.fill_(float("inf"))
            continue
        difference = (parameter.grad - reference).abs().max().to(maximum.dtype)
        maximum = torch.maximum(maximum, difference)
    inactive = torch.tensor(int(inactive_preserved), device=device, dtype=torch.int32)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(inactive, op=dist.ReduceOp.MIN)
    return {
        "schema": "typed_sparse_evolution.active_gradient_parity.v1",
        "reference": "per_parameter_global_sum_div_world_size_DDP_rule",
        "maximum_absolute_difference": float(maximum.detach().cpu()),
        "inactive_parameters_remain_grad_none": bool(int(inactive.item())),
        "status": (
            "PASS"
            if float(maximum) == 0.0 and int(inactive.item()) == 1
            else "FAIL"
        ),
        "audit": {
            "active_parameter_tensors": audit.active_parameter_tensors,
            "active_parameter_numel": audit.active_parameter_numel,
            "communicated_bytes": audit.communicated_bytes,
            "buckets": audit.buckets,
            "world_size": audit.world_size,
        },
    }


__all__ = [
    "ActiveGradientSynchronizer",
    "GradientSyncAudit",
    "broadcast_module_state",
    "distributed_equivalence_probe",
]
