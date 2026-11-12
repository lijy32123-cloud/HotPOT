"""Build the array cache of the simulated combustion trajectories of RealPDEBench.

Input: the `numerical` Arrow dataset of the combustion scenario (30 trajectories, 2001 frames, 128x128), as released.
Output: <out>.npy, float32, shape (30, 2001, 16, 64, 64), and <out>.json with the trajectory names.
Space is subsampled by 2, as the official loader does for the simulated data. Channel 0 is the surrogate "observed"
field (never used by the experiment); channels 1..15 are the 15 solver fields in the released order.

usage: python prepare_data.py <combustion/hf_dataset/numerical> <out stem>
"""
import json
import sys

import numpy as np
from datasets import load_from_disk

SUB = 2


def decode(row, key, extra=()):
    shape = (row['shape_t'], row['shape_h'], row['shape_w']) + tuple(extra)
    return np.frombuffer(row[key], dtype=np.float32).reshape(shape)


def main(source, out):
    data = load_from_disk(source)
    first = data[0]
    shape = (len(data), first['shape_t'], 16, -(-first['shape_h'] // SUB), -(-first['shape_w'] // SUB))
    assert shape == (30, 2001, 16, 64, 64), shape
    array = np.lib.format.open_memmap(out + '.npy', mode='w+', dtype=np.float32, shape=shape)
    ids, nan = [], 0
    for i in range(len(data)):
        row = data[i]
        ids.append(row['sim_id'])
        observed = decode(row, 'observed')[:, ::SUB, ::SUB]
        numerical = decode(row, 'numerical', (15,))[:, ::SUB, ::SUB]
        frame = np.concatenate([observed[:, None], np.moveaxis(numerical, -1, 1)], 1)
        nan += int(np.isnan(frame).sum())
        array[i] = np.nan_to_num(frame)
        print(i + 1, len(data), row['sim_id'], flush=True)
    array.flush()
    json.dump({'shape': list(shape), 'sim_ids': ids, 'fields': ['observed', 'numerical'], 'sub': SUB, 'nan_values': nan}, open(out + '.json', 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
