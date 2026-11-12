from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from .protocol import ObservationAddress, TRAIN_ROWS


@dataclass(frozen=True)
class ObservationSchema:
    variables: tuple[str, ...]
    stored_height: int
    stored_width: int
    public_height: int
    public_width: int
    history_frames: int
    target_frames: int
    available_frames: int
    public_channels: int
    padding_sentinel: float
    canonical_json: str


def load_schema(path: Path) -> ObservationSchema:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "schema",
        "observation_variables",
        "stored_grid",
        "public_grid",
        "temporal",
        "geometry_fields",
        "boundary_condition_fields",
        "physical_parameter_fields",
        "padding",
        "training_rows",
        "held_out_split_accessed",
    }
    if set(payload) != allowed:
        raise ValueError("observation schema contains undeclared top-level fields")
    if payload["training_rows"] != TRAIN_ROWS or payload["held_out_split_accessed"] is not False:
        raise ValueError("observation schema population or split contract drifted")
    schema = ObservationSchema(
        variables=tuple(payload["observation_variables"]),
        stored_height=int(payload["stored_grid"]["height"]),
        stored_width=int(payload["stored_grid"]["width"]),
        public_height=int(payload["public_grid"]["height"]),
        public_width=int(payload["public_grid"]["width"]),
        history_frames=int(payload["temporal"]["history_frames"]),
        target_frames=int(payload["temporal"]["target_frames"]),
        available_frames=int(payload["temporal"]["available_frames"]),
        public_channels=int(payload["padding"]["public_channels"]),
        padding_sentinel=float(payload["padding"]["sentinel"]),
        canonical_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )
    if len(schema.variables) != 3:
        raise ValueError("this bounded lane requires exactly three declared observations")
    if (schema.public_height, schema.public_width, schema.public_channels) != (128, 128, 4):
        raise ValueError("public tensor contract drifted")
    if (schema.history_frames, schema.target_frames, schema.available_frames) != (10, 1, 20):
        raise ValueError("temporal observation contract drifted")
    return schema


def resize_and_pad(sample: Tensor, schema: ObservationSchema) -> Tensor:
    if sample.ndim != 4:
        raise ValueError("sample must have [height,width,time,channel] layout")
    height, width, frames, channels = sample.shape
    if (height, width) != (schema.stored_height, schema.stored_width):
        raise ValueError("stored grid does not match declared schema")
    if frames < schema.history_frames + schema.target_frames:
        raise ValueError("insufficient saved frames")
    if channels != len(schema.variables):
        raise ValueError("observed channel count does not match declared variables")
    resized = sample.reshape(height, width, -1).permute(2, 0, 1)
    resized = F.interpolate(
        resized.unsqueeze(0),
        size=(schema.public_height, schema.public_width),
        mode="bilinear",
    ).squeeze(0).permute(1, 2, 0)
    resized = resized.reshape(
        schema.public_height, schema.public_width, frames, channels
    )
    public = torch.full(
        (schema.public_height, schema.public_width, frames, schema.public_channels),
        fill_value=schema.padding_sentinel,
        dtype=resized.dtype,
    )
    public[..., :channels] = resized
    return public


class DenseObservationDataset(Dataset[Mapping[str, Any]]):
    """Read explicit row/start addresses without class labels or integer selectors."""

    def __init__(self, source_file: Path, array_key: str, schema: ObservationSchema) -> None:
        if source_file.resolve(strict=True).is_dir():
            raise ValueError("dense observation source must be a file")
        self.source_file = source_file.absolute()
        self.array_key = str(array_key)
        self.schema = schema
        self._handle = None
        with h5py.File(self.source_file, "r") as handle:
            if self.array_key not in handle:
                raise KeyError("declared observation array is absent")
            shape = tuple(handle[self.array_key].shape)
        expected_tail = (
            schema.stored_height,
            schema.stored_width,
            schema.available_frames,
            len(schema.variables),
        )
        if len(shape) != 5 or shape[0] < TRAIN_ROWS or shape[1:] != expected_tail:
            raise ValueError(
                f"dense observation shape {shape} cannot supply declared "
                f"population/tail {(TRAIN_ROWS, *expected_tail)}"
            )

    def __len__(self) -> int:
        return TRAIN_ROWS

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def _array(self):
        if self._handle is None:
            self._handle = h5py.File(self.source_file, "r")
        return self._handle[self.array_key]

    def __getitem__(self, address: ObservationAddress) -> Mapping[str, Any]:
        if not isinstance(address, ObservationAddress):
            raise TypeError("explicit ObservationAddress required")
        if not 0 <= address.row < TRAIN_ROWS or not 0 <= address.start < 10:
            raise IndexError("observation address outside declared bounds")
        stop = address.start + self.schema.history_frames + self.schema.target_frames
        raw = np.asarray(self._array()[address.row, ..., address.start:stop, :], dtype=np.float32)
        public = resize_and_pad(torch.from_numpy(raw), self.schema)
        split = self.schema.history_frames
        return {
            "history": public[..., :split, :].contiguous(),
            "target": public[..., split : split + self.schema.target_frames, :].contiguous(),
            "mask": torch.ones_like(public[..., split : split + self.schema.target_frames, :]),
            "row": address.row,
            "start": address.start,
            "epoch_position": address.epoch_position,
        }


def load_panel(
    source_file: Path,
    array_key: str,
    schema: ObservationSchema,
    rows: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    histories, targets = [], []
    with h5py.File(source_file, "r") as handle:
        data = handle[array_key]
        for row in rows:
            sample = torch.from_numpy(np.asarray(data[row], dtype=np.float32))
            public = resize_and_pad(sample, schema)
            histories.append(public[..., : schema.history_frames, :])
            targets.append(public[..., schema.history_frames : schema.available_frames, :])
    return torch.stack(histories), torch.stack(targets)


__all__ = ["DenseObservationDataset", "ObservationSchema", "load_panel", "load_schema"]
