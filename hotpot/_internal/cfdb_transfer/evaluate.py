from __future__ import annotations


from typing import Any, Callable


import torch


from torch import Tensor


from torch.utils.data import DataLoader


from hotpot._internal.cfdb_conditions.evaluate_condition_space import HORIZONS, aggregate_arm, condition_audit, load_map


from hotpot._internal.cfdb_evaluation.evaluate_official_compare import OfficialCFDBDataset, per_sample_relative_l2


from hotpot._internal.cfdb_schema.schema_data import load_schema


from .sidecar import CertifiedTestSidecar


@torch.no_grad()
def evaluate_rows_indexed(
    name: str,
    loader: DataLoader,
    step: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    expected_samples: int,
):
    scores: dict[str, Any] = {
        "official_masked": {h: {} for h in HORIZONS},
        "legacy_dense": {h: {} for h in HORIZONS},
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    observed = 0
    for indices_cpu, history_cpu, target_cpu, mask_cpu in loader:
        source_indices = tuple(int(value) for value in indices_cpu.tolist())
        indices = indices_cpu.to(device=device, dtype=torch.long)
        history = history_cpu.to(device=device, dtype=torch.float32)
        target = target_cpu.to(device=device, dtype=torch.float32)
        mask = mask_cpu.to(device=device, dtype=torch.float32)
        current = history
        predictions = []
        for _ in range(10):
            prediction = step(current, indices)
            if prediction.shape != (*current.shape[:3], 1, 4):
                raise RuntimeError(f"{name} output contract drifted")
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError(f"{name} rollout became nonfinite")
            predictions.append(prediction[..., :2])
            current = torch.cat((current[..., 1:, :], prediction), dim=-2)
        rollout = torch.cat(predictions, dim=-2)
        for horizon in HORIZONS:
            pred_h = rollout[..., :horizon, :]
            target_h = target[..., :horizon, :]
            official = per_sample_relative_l2(pred_h, target_h, mask).cpu().tolist()
            dense = per_sample_relative_l2(pred_h, target_h, None).cpu().tolist()
            for row, source_index in enumerate(source_indices):
                scores["official_masked"][horizon][source_index] = float(official[row])
                scores["legacy_dense"][horizon][source_index] = float(dense[row])
        observed += len(source_indices)
    if observed != expected_samples:
        raise RuntimeError(f"{name} evaluated {observed}, expected {expected_samples}")
    scores["runtime"] = {0: {0: float(torch.cuda.max_memory_allocated(device) / 1024**3)}}
    return scores
