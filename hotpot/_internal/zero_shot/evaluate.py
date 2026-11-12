from __future__ import annotations


import torch


from torch import Tensor


from hotpot._internal.data_protocol.manifest import CONDITION_FIELDS


from hotpot._internal.data_protocol.evaluate_e1000 import OfficialRolloutDataset


from hotpot._internal.data_protocol.evaluation_manifest import EvaluationSpec


EVALUATION_BY_KEY = {
    "ns2d_fno_1e-4": EvaluationSpec(
        0,
        "ns2d_fno_1e-4",
        "data/large/ns2d_1e-4_test.hdf5",
        "packed_hdf5",
        200,
        64,
        50,
        1,
        1,
        20,
    ),
    "ns2d_pdb_M1_eta1e-2_zeta1e-2": EvaluationSpec(
        2,
        "ns2d_pdb_M1_eta1e-2_zeta1e-2",
        "data/large/pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2/test",
        "scatter_hdf5",
        200,
        128,
        21,
        4,
        4,
        11,
    ),
}


def conditions(target, batch: int, device: torch.device) -> tuple[Tensor, Tensor]:
    values = torch.zeros(batch, len(CONDITION_FIELDS), device=device)
    mask = torch.zeros_like(values)
    for name, scalar in target.condition:
        column = CONDITION_FIELDS.index(name)
        values[:, column] = float(scalar)
        mask[:, column] = 1.0
    return values, mask
