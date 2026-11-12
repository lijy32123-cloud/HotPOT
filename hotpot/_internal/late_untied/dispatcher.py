import copy

import torch
from torch import nn

from .contract import FAMILIES


class FamilyLateDispatcher(nn.Module):
    def __init__(self, original):
        super().__init__()
        # deepcopy consumes no RNG. The original is NOT registered or retained.
        self.blocks = nn.ModuleDict({family: copy.deepcopy(original) for family in FAMILIES})
        self.indices = None
        self.batch_size = None
        self.call_count = 0

    def reset_audit(self):
        self.call_count = 0
        self.clear_routes()
        for block in self.blocks.values():
            block.reset_audit()

    def clear_routes(self):
        self.indices = None
        self.batch_size = None

    def bind(self, indices, batch_size):
        """Use indices produced by the original semantic grouping, never values."""
        if not indices or set(indices) - set(FAMILIES):
            raise ValueError("Explicit known family indices are required")
        if any(index.ndim != 1 or index.dtype != torch.long or index.numel() == 0
               for index in indices.values()):
            raise ValueError("Family indices must be nonempty one-dimensional int64 tensors")
        packed = torch.cat(list(indices.values()))
        if not torch.equal(packed.sort().values,
                           torch.arange(batch_size, device=packed.device)):
            raise ValueError("Family indices must partition the batch exactly once")
        self.indices = dict(indices)
        self.batch_size = batch_size

    def forward(self, value):
        if self.indices is None or value.shape[0] != self.batch_size:
            raise RuntimeError("Bind explicit family indices before late dispatch")
        self.call_count += 1
        # The formal sampler is family homogeneous. Preserve the original batch
        # kernel shape and call exactly one same-width block in that common case.
        if len(self.indices) == 1:
            family, index = next(iter(self.indices.items()))
            output = self.blocks[family](value.index_select(0, index))
            return torch.zeros_like(value).index_copy(0, index, output)
        result = torch.zeros_like(value)
        for family, index in self.indices.items():
            result = result.index_copy(0, index,
                self.blocks[family](value.index_select(0, index)))
        return result
