from __future__ import annotations

import torch
from torch import Tensor


def simple_relative_lp(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    *,
    reduction: str = "sum",
) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError("prediction and target must have equal temporal field shapes")
    if mask is not None:
        prediction = prediction * mask
        target = target * mask
        effective_channels = mask.sum(
            dim=tuple(range(1, mask.ndim - 1))
        ).count_nonzero(dim=-1).clamp_min(1)
    else:
        effective_channels = prediction.new_full(
            (prediction.shape[0],), prediction.shape[-1]
        )
    flattened_prediction = prediction.reshape(prediction.shape[0], -1, prediction.shape[-1])
    flattened_target = target.reshape(target.shape[0], -1, target.shape[-1])
    numerator = torch.linalg.vector_norm(
        flattened_prediction - flattened_target, ord=2, dim=1
    )
    denominator = torch.linalg.vector_norm(flattened_target, ord=2, dim=1) + 1.0e-8
    per_sample = (numerator / denominator).sum(dim=-1) / effective_channels
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "none":
        return per_sample
    raise ValueError(reduction)


def per_dataset_metrics(
    prediction: Tensor,
    target: Tensor,
    dataset_indices: Tensor,
) -> dict[int, float]:
    result = {}
    for dataset in dataset_indices.detach().cpu().unique(sorted=True).tolist():
        selected = dataset_indices == int(dataset)
        result[int(dataset)] = float(
            simple_relative_lp(
                prediction[selected], target[selected], reduction="mean"
            ).detach().cpu()
        )
    return result


PREDICTED_CHANNELS_BY_DATASET = (1, 1, 4, 1, 2, 2)


def operator_owned_relative_lp(
    prediction: Tensor,
    target: Tensor,
    dataset_indices: Tensor,
    *,
    reduction: str = "sum",
) -> Tensor:
    """Relative L2 on each operator's physical prediction channels only."""

    if prediction.shape != target.shape or dataset_indices.shape != (
        prediction.shape[0],
    ):
        raise ValueError("operator objective inputs are incompatible")
    per_sample = prediction.new_empty(prediction.shape[0])
    for dataset in dataset_indices.detach().cpu().unique(sorted=True).tolist():
        dataset = int(dataset)
        if dataset < 0 or dataset >= len(PREDICTED_CHANNELS_BY_DATASET):
            raise IndexError(dataset)
        selected = dataset_indices == dataset
        channels = PREDICTED_CHANNELS_BY_DATASET[dataset]
        per_sample[selected] = simple_relative_lp(
            prediction[selected, ..., :channels],
            target[selected, ..., :channels],
            reduction="none",
        )
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "none":
        return per_sample
    raise ValueError(reduction)


__all__ = [
    "operator_owned_relative_lp",
    "per_dataset_metrics",
    "simple_relative_lp",
]
