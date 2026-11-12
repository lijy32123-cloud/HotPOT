"""Full-trajectory validation and evaluation for NS/CNS coefficient adaptation."""
import math

import torch
from torch.utils.data import Dataset

from ..coefficients.data import CONDITION_FIELDS, TargetStorageDataset
from ..data_protocol.objective import simple_relative_lp


class RolloutValidationDataset(Dataset):
    """First H10 frames followed by the full H20/H11 validation horizon."""

    def __init__(self, workspace, spec, rows):
        self.source = TargetStorageDataset(workspace, spec)
        self.spec = spec
        self.rows = tuple(rows)
        self.steps = 20 if spec.family == 'fno' else 11

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        values = torch.from_numpy(self.source._load(row)[..., :10 + self.steps, :])
        values = self.source._resize(values)
        state = torch.ones(128, 128, 10 + self.steps, 4)
        state[..., :self.spec.channels] = values
        mask = torch.zeros(128, 128, 1, 4)
        stride = 128 // self.spec.native_size
        mask[::stride, ::stride, :, :self.spec.predicted_channels] = 1
        return row, state[..., :10, :], state[..., 10:, :], mask


@torch.no_grad()
def rollout(model, spec, loader, *, device='cuda'):
    """Average per-trajectory, per-channel relative norms over the whole horizon."""
    model.eval()
    records = []
    horizon = 20 if spec.family == 'fno' else 11
    for rows, history, truth, mask in loader:
        history, truth, mask = history.to(device), truth.to(device), mask.to(device)
        condition = history.new_zeros(len(history), len(CONDITION_FIELDS))
        available = torch.zeros_like(condition)
        for name, value in spec.condition:
            column = CONDITION_FIELDS.index(name)
            condition[:, column] = value
            available[:, column] = 1
        predictions = []
        for _ in range(horizon):
            prediction = model(history, [spec.program] * len(history), condition, available)
            if prediction.shape != (*history.shape[:3], 1, 4) or not torch.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite or invalid coefficient rollout')
            predictions.append(prediction)
            feedback = prediction.clone()
            feedback[..., spec.predicted_channels:] = history[..., -1:, spec.predicted_channels:]
            history = torch.cat((history[..., 1:, :], feedback), dim=-2)
        prediction = torch.cat(predictions, dim=-2)
        # FP32 model execution; FP64 norm accumulation matches the final protocol.
        errors = simple_relative_lp(prediction.double(), truth[..., :horizon, :].double(),
                                    mask.double(), reduction='none').cpu().tolist()
        if not all(math.isfinite(value) for value in errors):
            raise FloatingPointError('Nonfinite coefficient rollout metric')
        records.extend({'row': int(row), 'l2re': value} for row, value in zip(rows.tolist(), errors))
    if not records:
        raise ValueError('Empty rollout population')
    return {'mean_l2re': sum(row['l2re'] for row in records) / len(records), 'rows': records}
