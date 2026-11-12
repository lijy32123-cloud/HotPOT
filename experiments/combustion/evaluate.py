"""Closed-loop evaluation at short horizons, on the validation conditions or on the two test groups.

Every checkpoint given is rolled out 20 frames from windows starting every 50 frames (40 per trajectory). Stored per
window and field: the normalised MSE over the first h predicted frames, h in {1, 5, 10, 20}; the same for two
predictors without training (repeat the last input frame; mean of the 10 input frames).

usage: python evaluate.py --arm <arm> --normalization <stage-1 run_dir>/PROTOCOL.json --split <validation|test> \
           --out <file.json> name=<checkpoint.pt> [name=<checkpoint.pt> ...]
Use it on validation for every saved checkpoint first; compare.py then picks, per model and horizon, the checkpoint
with the lowest validation error and reads only that one from the test file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from combustion import ARMS, CombustionData, build, predict

HORIZONS = (1, 5, 10, 20)
STARTS = list(range(0, 1951, 50))
BATCH = 8


def score(pred, z):
    """pred: (B, X, Y, 20, C), z: (B, X, Y, 30, C), both standardised -> {h: (B, C)}."""
    return {h: (pred[..., :h, :] - z[..., 10:10 + h, :]).square().mean((1, 2, 3)).cpu().numpy() for h in HORIZONS}


def windows(data, rows, device):
    for row in rows:
        for i in range(0, len(STARTS), BATCH):
            yield torch.stack([data.normalize(data.raw(row, s, 30).to(device)) for s in STARTS[i:i + BATCH]])


def collect(acc):
    return {str(h): np.concatenate(v).tolist() for h, v in acc.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-windows', type=int, default=None, help='debugging only: first N starts per trajectory')
    parser.add_argument('checkpoints', nargs='+', help='name=path')
    a = parser.parse_args()
    if a.max_windows:
        del STARTS[a.max_windows:]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    norm = json.loads(a.normalization.read_text())['normalization']
    data = CombustionData()
    data.set_normalization(norm)
    groups = {'validation': data.val} if a.split == 'validation' else data.test
    result = {'arm': a.arm, 'split': a.split, 'starts': STARTS, 'horizons': HORIZONS,
              'groups': {g: [data.ids[r] for r in rows] for g, rows in groups.items()}, 'trivial': {}, 'checkpoints': {}}
    with torch.inference_mode():
        for g, rows in groups.items():
            acc = {k: {h: [] for h in HORIZONS} for k in ('persistence', 'history_mean')}
            for z in windows(data, rows, a.device):
                trivial = {'persistence': z[..., 9:10, :].expand(-1, -1, -1, 20, -1), 'history_mean': z[..., :10, :].mean(3, keepdim=True).expand(-1, -1, -1, 20, -1)}
                for k, pred in trivial.items():
                    for h, v in score(pred, z).items():
                        acc[k][h].append(v)
            result['trivial'][g] = {k: collect(acc[k]) for k in acc}
        for item in a.checkpoints:
            name, path = item.split('=', 1)
            model, ledger = build(a.arm, normalization=norm)
            state = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            model.to(a.device).eval()
            entry = {'step': state.get('step'), 'stored_parameters': ledger['stored_parameters'], 'nmse': {}}
            for g, rows in groups.items():
                acc = {h: [] for h in HORIZONS}
                for z in windows(data, rows, a.device):
                    hist = z[..., :10, :]
                    preds = []
                    for _ in range(20):
                        p = predict(model, hist)
                        preds.append(p)
                        hist = torch.cat([hist[..., 1:, :], p.unsqueeze(-2)], -2)
                    for h, v in score(torch.stack(preds, 3), z).items():
                        acc[h].append(v)
                entry['nmse'][g] = collect(acc)
                print(a.arm, name, entry['step'], g, ' '.join('h%d %.4f' % (h, np.mean(entry['nmse'][g][str(h)])) for h in HORIZONS), flush=True)
            result['checkpoints'][name] = entry
            del model
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, 'w'))


if __name__ == '__main__':
    main()
