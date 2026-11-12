"""Stage 2: continue the one-step training of a stage-1 run under one plateau controller. Identical for the three models.

Restores the model, the AdamW moments and the random state of <stage1>/last.pt (update 5,000), keeps the same loss and
the same counter-based sampling, and validates every 2,500 updates with 100-frame closed-loop rollouts. Three
checkpoints are kept: best.pt (lowest relative L2), best_std_diagnostic.pt (lowest normalised MSE) and last.pt.
The controller halves the learning rate after six validations without a 0.5% improvement in any of three monitors and
stops after another six at the floor.

usage: python train_stage2.py --arm <arm> --stage1 <stage-1 run_dir> --out <run_dir> [--smoke]
"""
import argparse
import json
import math
import random
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from combustion import ARMS, SEED, CombustionData, atomic_json, build, predict, rollout_error
from convergence import Convergence
from stable_clip import clip_grad_norm

CONFIG = {'source_step': 5000, 'min_total_updates': 20000, 'global_batch': 8, 'lr_peak': 3e-5, 'lr_min': 1e-6, 'lr_factor': 0.5,
          'warmup_updates': 200, 'plateau_patience': 6, 'min_relative_improvement': 0.005, 'training_horizon': 1,
          'validation_horizon': 100, 'validation_starts': [0, 600, 1200], 'validation_every': 2500, 'checkpoint_every': 250,
          'fixed_training_panel': 'the 18 training trajectories at starts 600 and 1200 (36 one-step windows)'}
STOP = False


def stop(*args):
    global STOP
    STOP = True


def save(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def rng_state():
    return {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available() and len(state['cuda_rng']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    if 'numpy_rng' in state:
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])


@torch.inference_mode()
def fixed_train_error(model, data, device):
    model.eval()
    values = []
    for row in data.train:
        for start in [600, 1200]:
            z = data.normalize(data.raw(row, start, 11).to(device))
            pred = predict(model, z[None, ..., :10, :])[0]
            values.append(float((pred - z[..., 10, :]).square().mean()))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
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
    prior = json.loads((args.stage1 / 'PROTOCOL.json').read_text())
    data = CombustionData()
    data.set_normalization(prior['normalization'])
    model, ledger = build(args.arm, normalization=prior['normalization'])
    model.to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg['lr_peak'], betas=(.9, .99), weight_decay=1e-6)
    source = args.stage1 / 'last.pt'
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    last = out / 'last.pt'
    resumed = last.exists() and not args.smoke
    state = torch.load(last if resumed else source, map_location='cpu', weights_only=False)
    assert state['step'] >= cfg['source_step']
    if not resumed:
        assert state['step'] == cfg['source_step'] and state['normalization'] == prior['normalization']
    model.load_state_dict(state['model'], strict=True)
    opt.load_state_dict(state['optimizer'])
    restore_rng(state)
    step = state['step']
    elapsed = state.get('continuation_seconds', 0.)
    best = state.get('continuation_best', float('inf'))
    best_step = state.get('continuation_best_step')
    best_std = state.get('continuation_best_std', float('inf'))
    controller = Convergence(cfg, state.get('convergence_controller'))
    del state
    atomic_json(out / 'PROVENANCE.json', {'arm': args.arm, 'protocol': cfg, 'normalization': prior['normalization'], 'ledger': ledger,
                                          'torch': str(torch.__version__), 'resumed': resumed, 'start_step': step})
    start_time = time.time()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def payload():
        return {'arm': args.arm, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'step': step, 'continuation_best': best,
                'continuation_best_step': best_step, 'continuation_best_std': best_std, 'convergence_controller': controller.state,
                'continuation_seconds': elapsed + time.time() - start_time, **rng_state()}

    def select(value, candidate_step, initial=False):
        nonlocal best, best_step, best_std
        if value['mean_l2re'] < best:
            best, best_step = value['mean_l2re'], candidate_step
            save(out / 'best.pt', {'model': model.state_dict(), 'step': candidate_step, 'validation': value, 'initial_retained_candidate': initial})
        if value['normalized_mse'] < best_std:
            best_std = value['normalized_mse']
            save(out / 'best_std_diagnostic.pt', {'model': model.state_dict(), 'step': candidate_step, 'validation': value})

    def validate():
        saved_rng = rng_state()
        try:
            value = rollout_error(model, data, data.val, cfg['validation_starts'], cfg['validation_horizon'], args.device)
            value['fixed_train_nmse'] = fixed_train_error(model, data, args.device)
            atomic_json(out / f'validation_{step:06d}.json', value)
            select(value, step)
            event = controller.observe(step, {'rollout_l2re': value['mean_l2re'], 'rollout_nmse': value['normalized_mse'], 'fixed_train_nmse': value['fixed_train_nmse']})
            atomic_json(out / 'CONVERGENCE.json', controller.state)
            print(json.dumps({'validation_step': step, 'l2re': value['mean_l2re'], 'normalized_mse': value['normalized_mse'],
                              'fixed_train_nmse': value['fixed_train_nmse'], 'best_step': best_step, 'decision': event}), flush=True)
        finally:
            restore_rng(saved_rng)
        save(last, payload())

    try:
        if not args.smoke and not resumed:
            # The stage-1 best checkpoint stays a candidate under the 100-frame validation.
            saved_rng = rng_state()
            old_best = torch.load(args.stage1 / 'best.pt', map_location='cpu', weights_only=False)
            if old_best['step'] != step:
                model.load_state_dict(old_best['model'], strict=True)
                old_value = rollout_error(model, data, data.val, cfg['validation_starts'], cfg['validation_horizon'], args.device)
                select(old_value, old_best['step'], initial=True)
                full = torch.load(source, map_location='cpu', weights_only=False)
                model.load_state_dict(full['model'], strict=True)
                del full
            del old_best
            restore_rng(saved_rng)
        if not args.smoke and controller.validation_due(step):
            validate()
        smoke_end = step + 2
        while (step < smoke_end if args.smoke else not controller.state['converged']):
            if STOP:
                save(last, payload())
                raise SystemExit(75)
            model.train()
            history, target, rows, starts = data.batch(step, cfg['global_batch'])
            lr = controller.learning_rate(step)
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            prediction = predict(model, history.to(args.device))
            loss = (prediction - target.to(args.device)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward()
            grad_norm = float(clip_grad_norm(params, 1.))
            if not math.isfinite(grad_norm):
                raise FloatingPointError('nonfinite gradient norm')
            opt.step()
            step += 1
            if step % 10 == 0 or args.smoke:
                record = {'step': step, 'train_normalized_mse': float(loss.detach()), 'lr': lr, 'gradient_norm': grad_norm,
                          'seconds': elapsed + time.time() - start_time}
                with (out / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            if not args.smoke and controller.validation_due(step):
                validate()
            if not args.smoke and (step % cfg['checkpoint_every'] == 0 or STOP):
                save(last, payload())
        if args.smoke:
            print('SMOKE PASS', args.arm, flush=True)
            return
        atomic_json(out / 'TRAIN_COMPLETE.json', {'stop_reason': 'plateau controller', 'step': step, 'best_step': best_step, 'best_validation': best,
                                                 'convergence': controller.state, 'test_accessed': False})
    except Exception:
        atomic_json(out / 'FAILURE.json', {'completed_step': step, 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
