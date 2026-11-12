from __future__ import annotations


from collections.abc import Sequence


import torch


from torch import Tensor


from hotpot._internal.data_protocol.manifest import ProgramId


from .program_specs import PROGRAM_CODE


def normalize_programs(
    programs: ProgramId | str | Sequence[ProgramId | str],
    *,
    batch: int,
    device: torch.device,
) -> tuple[tuple[ProgramId, ...], Tensor]:
    if isinstance(programs, (ProgramId, str)):
        raw = (programs,) * batch
    else:
        raw = tuple(programs)
    if len(raw) != batch:
        raise ValueError(f"expected {batch} program IDs, received {len(raw)}")
    normalized: list[ProgramId] = []
    for value in raw:
        try:
            normalized.append(value if isinstance(value, ProgramId) else ProgramId(value))
        except ValueError as error:
            raise ValueError(f"unknown physical program {value!r}") from error
    codes = torch.tensor(
        [PROGRAM_CODE[value] for value in normalized],
        dtype=torch.long,
        device=device,
    )
    return tuple(normalized), codes
