"""Stage 1: 5,000 one-step updates from the pre-training checkpoint. Identical for the three models.

usage: python train_stage1.py --arm <arm> --out <run_dir> [--smoke]
The run is resumable: start the same command again and it continues from <run_dir>/last.pt.
"""
import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from combustion import ARMS, SEED, CombustionData, atomic_json, build, predict, rollout_error

CONFIG = {'updates': 5000, 'warmup_updates': 100, 'lr': 1e-4, 'global_batch': 8, 'microbatch': 8,
          'optimizer': 'AdamW betas=(0.9, 0.99), weight_decay=1e-6, gradient norm clipped to 1',
          'schedule': 'linear warm-up, then cosine to 1e-6', 'loss': 'mean squared error of the standardised next frame, equal field weights',
          'validation_every': 250, 'validation_horizon': 20, 'validation_starts': [0, 600, 1200],
          'selection': 'lowest mean per-field relative L2 of the 20-frame validation rollout'}
STOP = False


def stop_handler(*_):
    global STOP
    STOP = True


def save(path, obj):
    q = path.with_suffix('.tmp')
    torch.save(obj, q)
    q.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda')
    a = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = CONFIG
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'TRAIN_COMPLETE.json').exists():
        print('ALREADY_COMPLETE', a.arm, flush=True)
        return
    data = CombustionData()
    norm = data.fit_normalization()
    data.set_normalization(norm)
    model, ledger = build(a.arm, normalization=norm)
    model.to(a.device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg['lr'], betas=(.9, .99), weight_decay=1e-6)
    start = time.time()
    done = 0
    best = float('inf')
    best_step = None
    past_seconds = 0.
    last = out / 'last.pt'
    atomic_json(out / 'PROTOCOL.json', {'config': cfg, 'source': ledger, 'normalization': norm, 'torch': str(torch.__version__), 'test_used_for_selection': False})

    def payload(step):
        return {'arm': a.arm, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'step': step, 'best': best, 'best_step': best_step,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if a.device == 'cuda' else [],
                'seconds': past_seconds + time.time() - start,
                'normalization': norm}

    if last.exists() and not a.smoke:
        state = torch.load(last, map_location='cpu', weights_only=False)
        assert state['arm'] == a.arm and state['normalization'] == norm
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer'])
        done, best, best_step, past_seconds = state['step'], state['best'], state['best_step'], state['seconds']
        torch.set_rng_state(state['torch_rng'])
        if a.device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    limit = 2 if a.smoke else cfg['updates']
    for step in range(done, limit):
        model.train()
        h, y, rows, starts = data.batch(step, cfg['global_batch'])
        opt.zero_grad(set_to_none=True)
        ramp = min(1., (step + 1) / cfg['warmup_updates'])
        decay = .5 * (1 + math.cos(math.pi * max(0, step - cfg['warmup_updates']) / (cfg['updates'] - cfg['warmup_updates'])))
        lr = cfg['lr'] * ramp * (.01 + .99 * decay)
        for group in opt.param_groups:
            group['lr'] = lr
        loss_sum = 0.
        for k in range(0, len(h), cfg['microbatch']):
            end = min(len(h), k + cfg['microbatch'])
            weight = (end - k) / len(h)
            pred = predict(model, h[k:end].to(a.device))
            loss = (pred - y[k:end].to(a.device)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            (loss * weight).backward()
            loss_sum += float(loss.detach()) * weight
        norm_grad = float(torch.nn.utils.clip_grad_norm_(params, 1.))
        assert math.isfinite(norm_grad)
        opt.step()
        done = step + 1
        if done % 10 == 0 or done <= 2:
            record = {'step': done, 'train_normalized_mse': loss_sum, 'gradient_norm_before_clip': norm_grad, 'lr': lr,
                      'seconds': past_seconds + time.time() - start, 'pid': os.getpid()}
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        if not a.smoke and (done % cfg['validation_every'] == 0 or done == limit):
            result = rollout_error(model, data, data.val, cfg['validation_starts'], cfg['validation_horizon'], a.device)
            atomic_json(out / f'validation_{done:06}.json', result)
            if result['mean_l2re'] < best:
                best, best_step = result['mean_l2re'], done
                save(out / 'best.pt', {'model': model.state_dict(), 'step': done, 'validation': result})
            print('VALIDATION', done, result['mean_l2re'], 'BEST', best, flush=True)
        if not a.smoke and (done % 100 == 0 or done % cfg['validation_every'] == 0 or done == limit or STOP):
            save(last, payload(done))
        if STOP:
            raise SystemExit(75)
    if a.smoke:
        model.eval()
        h, _, _, _ = data.batch(0, 1)
        with torch.no_grad():
            roll = h.to(a.device)
            for _ in range(3):
                y = predict(model, roll)
                assert torch.isfinite(y).all()
                roll = torch.cat([roll[..., 1:, :], y.unsqueeze(-2)], -2)
        print('SMOKE PASS', a.arm, ledger['stored_parameters'], flush=True)
        return
    train_seconds = past_seconds + time.time() - start
    atomic_json(out / 'TRAIN_COMPLETE.json', {'updates': done, 'best_validation': best, 'best_step': best_step, 'gpu_hours': train_seconds / 3600,
                                             'test_accessed': False})


if __name__ == '__main__':
    main()
