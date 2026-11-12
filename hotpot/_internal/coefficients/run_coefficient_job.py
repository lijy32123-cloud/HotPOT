from __future__ import annotations


from typing import Any


import torch


from torch import Tensor, nn


from hotpot._internal.data_protocol.objective import simple_relative_lp


def _predict(
    model: nn.Module,
    kind: str,
    batch: dict[str, Tensor],
    spec: Any,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    history = batch["history"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    if kind == "ours":
        condition = batch["condition"].to(device, non_blocking=True)
        condition_mask = batch["condition_mask"].to(device, non_blocking=True)
        prediction = model(
            history,
            [spec.program] * history.shape[0],
            condition,
            condition_mask,
        )
    elif kind == "moepot":
        prediction = model(history)[0]
    else:
        raise ValueError(kind)
    return prediction[..., : spec.predicted_channels], target[..., : spec.predicted_channels]
