"""Global modal basis for the real OH* intensity task.

Mean image and the leading 128 POD modes, computed from the start frames of the official
training windows only (no validation or test frame is read). Usage:
    python build_basis.py <task_dir> <out_dir>
<task_dir> is the directory holding CACHE_CHECK.json and metadata/ (see network/common.py).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

BASE = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0); KM = 128
check = json.loads((BASE / 'CACHE_CHECK.json').read_text()); ids = check['metadata']['sim_ids']
arr = torch.from_numpy(np.array(np.load(check['path'], mmap_mode='r').reshape(30, 2002, 64, 64), copy=True))
idx = {}
for s in ['train', 'val', 'test']:
    rows = json.loads((BASE / 'metadata' / f'{s}_index_real.json').read_text())
    idx[s] = torch.tensor([[ids.index(r['sim_id']), r['time_id']] for r in rows])

# ---- global basis from training-window start frames
tr = idx['train']
mu = torch.zeros(4096, dtype=torch.float64)
for j in range(0, len(tr), 4096):
    q = tr[j:j + 4096]; mu += arr[q[:, 0], q[:, 1]].double().flatten(1).sum(0)
mu /= len(tr)
g = torch.Generator().manual_seed(0)
sub = tr[torch.randperm(len(tr), generator=g)[:24000]]
X = arr[sub[:, 0], sub[:, 1]].double().flatten(1) - mu
ev, V = torch.linalg.eigh(X.T @ X / len(X)); ev = ev.flip(0); V = V.flip(1)[:, :KM].contiguous()
std = ev[:KM].clamp(min=1e-12).sqrt().float()
torch.save({'mu': mu, 'V': V, 'ev': ev[:KM]}, OUT / 'basis.pt')
print('saved', OUT / 'basis.pt')
