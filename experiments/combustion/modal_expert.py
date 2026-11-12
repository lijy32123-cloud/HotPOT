"""Modal state-space expert for real OH* combustion intensity (active-parameter-aligned with MoE-POT).

Usage: python modal_expert.py <network_run_dir> <moepot_run_dir> <modal_dir> <out_dir>

MoE-POT activates 19,645,751 parameters per sample on this task (top-4 of 16 routed experts in each of 4 blocks, 2 shared
experts, gating, 20-frame time aggregation; unused classification head excluded). The fine-tuned HotPOT predictor ("relaxed") activates 17,435,360.
  strict : observation step = network only                                   (17,435,360 active)
  aligned: observation step = network + compact propagator + 18 gains, total <= 19,645,751 active
The time step (5,313,024) runs alone on later blocks. Sweep on validation, then the unmodified official evaluate()."""
import copy, json, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
R = Path(sys.argv[1])   # network run directory (code + runs/hotpot_relaxed/lr_0.001/best.pt)
BM = Path(sys.argv[2])  # MoE-POT run directory with TEST_h{20,40,60}.json (paired comparison only)
HERE = Path(sys.argv[4]); HERE.mkdir(exist_ok=True); GO = Path(sys.argv[3]); sys.path.insert(0, str(R))  # GO: basis.pt + relaxed_modal.pt
import numpy as np, torch, torch.nn as nn
from common import IntensityData, evaluate
from intensity_models import build

t0 = time.time(); dev = torch.device('cuda'); torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
K, SY, DETECT, MOE_ACTIVE, NET = 128, 0.0252928975969553, 1e-5, 19645751, 17435360
CAP = MOE_ACTIVE - NET - 18
def log(*a): print('[%5.0fs]' % (time.time() - t0), *a, flush=True)
data = IntensityData(device='cuda'); arr = data.array
bs = torch.load(GO / 'basis.pt'); mu = bs['mu'].to(dev); V = bs['V'].to(dev); sd = bs['ev'][:K].sqrt().float().to(dev)
C = torch.empty(30, 2002, K, device=dev); E = torch.empty(30, 2002, dtype=torch.float64, device=dev); T2 = torch.empty_like(E)
for i in range(30):
    f = arr[i].flatten(1).double(); d = f - mu; C[i] = (d @ V).float(); E[i] = d.square().sum(1); T2[i] = f.square().sum(1)
idx = {s: data.indices[s] for s in ['train', 'val', 'test']}
def valid(s, h): return torch.nonzero(idx[s][:, 1] + 20 + h < 2001).flatten()
def gather(q, h):
    t = q[:, 1, None] + torch.arange(20 + h, device=dev); c = C[q[:, 0, None], t]; return c[:, :20], c[:, 20:], E[q[:, 0, None], t[:, 20:]], T2[q[:, 0, None], t[:, 20:]]
def metrics(P, Y, EY, TY):
    err = (Y - P).double().square().sum(-1) + (EY - Y.double().square().sum(-1)).clamp(min=0)
    return round(float((err.sum() / (err.numel() * 4096)).sqrt()), 6), round(float((err.sum(1) / TY.sum(1)).sqrt().mean()), 4)
AD = torch.load(GO / 'relaxed_modal.pt', map_location=dev)
Hs = {s: torch.cat([gather(idx[s][j:j + 4096], 20)[0] for j in range(0, len(idx[s]), 4096)]) for s in idx}
Ytr1 = torch.cat([gather(idx['train'][j:j + 4096], 20)[1] for j in range(0, len(idx['train']), 4096)])
v20, v60, tr60 = valid('val', 20), valid('val', 60), valid('train', 60)
_, Yv, EYv, TYv = gather(idx['val'][v20], 20); _, Yv60, EYv60, TYv60 = gather(idx['val'][v60], 60)
RW = {}
def ridge(Kp):
    if Kp not in RW:
        d_ = 20 * Kp; Gm = torch.zeros(d_ + 1, d_ + 1, dtype=torch.float64, device=dev); Rm = torch.zeros(d_ + 1, d_, dtype=torch.float64, device=dev)
        for j in range(0, len(Ytr1), 2048):
            x = torch.cat([Hs['train'][j:j + 2048, :, :Kp].flatten(1).double(), torch.ones(len(Hs['train'][j:j + 2048]), 1, dtype=torch.float64, device=dev)], 1); Gm += x.T @ x; Rm += x.T @ Ytr1[j:j + 2048, :, :Kp].flatten(1).double()
        reg = 1e-1 * Gm.diagonal()[:-1].mean() * torch.eye(d_ + 1, dtype=torch.float64, device=dev); reg[-1, -1] = 0; RW[Kp] = torch.linalg.solve(Gm + reg, Rm).float()
    return RW[Kp]


class Prop(nn.Module):
    """modal block state (first Kp modes) -> next block; linear skip from ridge (full, or rank-limited via its SVD) + MLP correction"""
    def __init__(s, Kp, hid, rank=None, drop=0.1):
        super().__init__(); s.Kp = Kp; d_ = 20 * Kp; W = ridge(Kp); M = W[:-1].T.contiguous()
        if rank is None:
            s.lin = nn.Linear(d_, d_)
            with torch.no_grad(): s.lin.weight.copy_(M); s.lin.bias.copy_(W[-1])
        else:
            U, S, Vh = torch.linalg.svd(M.double(), full_matrices=False); s.lin = nn.Sequential(nn.Linear(d_, rank, bias=False), nn.Linear(rank, d_))
            with torch.no_grad(): s.lin[0].weight.copy_((S[:rank].sqrt()[:, None] * Vh[:rank]).float()); s.lin[1].weight.copy_((U[:, :rank] * S[:rank].sqrt()).float()); s.lin[1].bias.copy_(W[-1])
        s.net = nn.Sequential(nn.Linear(d_, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, d_))
        nn.init.zeros_(s.net[-1].weight); nn.init.zeros_(s.net[-1].bias); s.register_buffer('sd', sd[:Kp].clone())
    def forward(s, A):
        A = A[..., :s.Kp]; b = len(A); return s.lin(A.flatten(1)).view(b, 20, s.Kp) + s.net((A / s.sd).flatten(1)).view(b, 20, s.Kp) * s.sd
def pad(P, base=None):
    out = torch.zeros(len(P), 20, K, device=dev) if base is None else base.clone(); out[..., :P.shape[-1]] = P; return out
npar = lambda m: sum(p.numel() for p in m.parameters())
mse = lambda P, Y: (P - Y).square().sum(-1).mean() / 4096 / SY ** 2
def fit(net, batch_fn, val_fn, seed):
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000); gen = torch.Generator().manual_seed(seed); best = (1e9, None)
    for it in range(1, 3001):
        net.train(); loss = batch_fn(gen); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if it % 250 == 0:
            net.eval()
            with torch.no_grad(): v = val_fn()
            if v < best[0]: best = (v, copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[1]); net.eval(); return net
MG = [(0, 1), (1, 4), (4, 8), (8, 16), (16, 32), (32, 64)]; LG = [(0, 5), (5, 10), (10, 20)]
def fit_gain(A, P, Y, Kp):
    G = torch.zeros(20, K, device=dev)
    for a, e in MG:
        if a >= Kp: continue
        for l0, l1 in LG:
            dd = (P - A)[:, l0:l1, a:e].double(); r = (Y - A)[:, l0:l1, a:e].double(); G[l0:l1, a:e] = float((dd * r).sum() / dd.square().sum().clamp(min=1e-30))
    return G
half = idx['val'][v20][:, 0] % 2 == 0


def make_obs(Kp, hid, rank):
    torch.manual_seed(0); go = Prop(Kp, hid, rank).to(dev)
    go = fit(go, lambda gen: (lambda sel: mse(go(Hs['train'][sel]), Ytr1[sel, :, :Kp]))(torch.randint(len(Ytr1), (512,), generator=gen).to(dev)), lambda: metrics(pad(go(Hs['val'][v20])), Yv, EYv, TYv)[0], 0)
    with torch.no_grad(): PO = {s: pad(torch.cat([go(Hs[s][j:j + 4096]) for j in range(0, len(Hs[s]), 4096)]), AD[s]) for s in idx}
    F = torch.empty_like(Yv)
    for m in (half, ~half): F[m] = AD['val'][v20][m] + fit_gain(AD['val'][v20][~m], PO['val'][v20][~m], Yv[~m], Kp) * (PO['val'][v20][m] - AD['val'][v20][m])
    G = fit_gain(AD['val'][v20], PO['val'][v20], Yv, Kp)
    return go, G, {s: AD[s] + G * (PO[s] - AD[s]) for s in idx}, metrics(F, Yv, EYv, TYv)


log('active-parameter room for the propagator next to the network: %d' % CAP)
base20 = metrics(AD['val'][v20], Yv, EYv, TYv); log('strict (network only, 128 modes): val 20 frames %s' % (base20,))
cands = {}
for name, (Kp, hid, rank) in {'32 modes, hidden 512': (32, 512, None), '32 modes, hidden 768': (32, 768, None), '40 modes, hidden 640': (40, 640, None), '48 modes, hidden 512': (48, 512, None),
                              '64 modes, hidden 512, rank 128': (64, 512, 128), '64 modes, hidden 384, rank 256': (64, 384, 256)}.items():
    go, G, S1, cfv = make_obs(Kp, hid, rank); ok = npar(go) <= CAP; cands[name] = (go, G, S1, cfv, Kp)
    log('propagator %-30s | %7d params %s | observation step active %d | val 20 frames, cross-fitted gains %s' % (name, npar(go), 'ok ' if ok else 'OVER', NET + npar(go) + 18, cfv))
pick = min([n for n in cands if npar(cands[n][0]) <= CAP], key=lambda n: cands[n][3][0]); log('selected on validation:', pick)

relaxed, _ = build(); relaxed.load_state_dict(torch.load(R / 'runs/hotpot_relaxed/lr_0.001/best.pt', map_location='cpu', weights_only=False)['model']); relaxed.to(dev).eval()
rng = np.random.default_rng(0); OUT = {}; RECS = {}
for tag, go, G, S1 in [('strict', None, None, AD), ('aligned', cands[pick][0], cands[pick][1], cands[pick][2])]:
    etr = (Ytr1 - S1['train']).square().mean((0, 1)); eva = (Yv - S1['val'][v20]).square().mean((0, 1)); cg = (eva / etr - 1).clamp(min=0).sqrt()[:64]; Rtr = (Ytr1 - S1['train'])[..., :64]
    torch.manual_seed(1); gp = Prop(64, 1024).to(dev)
    def roll(a1, n):
        outs = [a1]
        for _ in range(n - 1): outs.append(pad(gp(outs[-1])))
        return torch.cat(outs, 1)
    def bf(gen):
        sel = tr60[torch.randint(len(tr60), (512,), generator=gen).to(dev)]; Y = gather(idx['train'][sel], 60)[1][..., :64]
        a1 = S1['train'][sel][..., :64] + cg * Rtr[tr60[torch.randint(len(tr60), (512,), generator=gen).to(dev)]]; a2 = gp(a1); a3 = gp(a2); return mse(torch.cat([a2, a3], 1), Y[:, 20:])
    def vf():
        P = roll(S1['val'][v60], 3); return metrics(P[:, :40], Yv60[:, :40], EYv60[:, :40], TYv60[:, :40])[0] + metrics(P, Yv60, EYv60, TYv60)[0]
    gp = fit(gp, bf, vf, 1)

    class Expert(nn.Module):
        def __init__(s): super().__init__(); s.log = []
        def forward(s, history):
            d = (history[..., 0] * data.sx + data.mx).flatten(2).double() - mu; a64 = d @ V
            r = ((d - a64 @ V.T).square().sum((1, 2)) / (20 * 4096)).sqrt(); a = a64.float(); s.log.append(r); obs = r > DETECT; out = torch.zeros_like(a)
            if obs.any():
                ad = ((data.physical(relaxed(history[obs]))[..., 0].flatten(2).double() - mu) @ V).float(); out[obs] = ad if go is None else ad + G * (pad(go(a[obs]), ad) - ad)
            if (~obs).any(): out[~obs] = pad(gp(a[~obs]))
            return data.target((mu + out.double() @ V.T).float().view(len(history), 20, 64, 64)).unsqueeze(-1)
    ex = Expert().eval(); OFF = {}; n_obs = n_syn = 0; BOOT = {}
    for s in ['val', 'test']:
        for h in (20, 40, 60):
            r = evaluate(ex, data, s, h); OFF[f'{s}_{h}'] = {k: r[k] for k in ['rmse', 'mae', 'relative_l2', 'windows']}; n_obs += r['windows']; n_syn += r['windows'] * (h // 20 - 1)
            if s == 'test':
                e = {x['index']: (x['rmse'] ** 2, x['sim_id']) for x in r['records']}; m = {x['index']: x['rmse'] ** 2 for x in json.loads((BM / f'TEST_h{h}.json').read_text())['records']}; ids = sorted(e)
                RECS.setdefault(tag, {})[h] = {str(i): [float(e[i][0]), e[i][1]] for i in e}
                se_e = np.array([e[i][0] for i in ids]); se_m = np.array([m[i] for i in ids]); traj = np.array([e[i][1] for i in ids]); names = sorted(set(traj))
                Se = np.array([se_e[traj == t].sum() for t in names]); Sm = np.array([se_m[traj == t].sum() for t in names]); boots = np.empty(20000)
                for b in range(20000):
                    w = np.bincount(rng.integers(0, 30, 30), minlength=30); boots[b] = np.sqrt((w * Se).sum() / (w * Sm).sum()) - 1
                BOOT[h] = [float(np.sqrt(se_e.mean() / se_m.mean()) - 1), *[float(x) for x in np.percentile(boots, [2.5, 97.5])], int((Se < Sm).sum())]
        log('%-7s %-4s official evaluate(): rmse 20/40/60 = %s | relative L2 = %s | mae = %s' % (tag, s, [round(OFF[f'{s}_{h}']['rmse'], 6) for h in (20, 40, 60)],
            [round(OFF[f'{s}_{h}']['relative_l2'], 4) for h in (20, 40, 60)], [round(OFF[f'{s}_{h}']['mae'], 6) for h in (20, 40, 60)]))
    r = torch.cat(ex.log); act = NET + (0 if go is None else npar(go) + 18)
    log('%-7s state recognition: observed min %.2e in %d calls (expected %d) | synthesised max %.2e in %d calls (expected %d)' % (tag, float(r[r > DETECT].min()), int((r > DETECT).sum()), n_obs, float(r[r <= DETECT].max()), int((r <= DETECT).sum()), n_syn))
    log('%-7s active parameters: observation step %d | time step %d | MoE-POT %d' % (tag, act, npar(gp), MOE_ACTIVE))
    for h in (20, 40, 60): log('%-7s vs MoE-POT, test %d frames: %+.2f%%, 95%% CI [%+.2f%%, %+.2f%%], better on %d/30 trajectories' % (tag, h, 100 * BOOT[h][0], 100 * BOOT[h][1], 100 * BOOT[h][2], BOOT[h][3]))
    OUT[tag] = {'official': OFF, 'paired_vs_moepot_test': BOOT, 'active_observation_step': act, 'active_time_step': npar(gp), 'propagator': None if go is None else pick}
    torch.save({'propagator': None if go is None else go.state_dict(), 'G': G, 'gp': gp.state_dict(), 'pick': pick}, HERE / f'expert_{tag}_modules.pt')
json.dump(RECS, open(HERE / 'expert_aligned_test_records.json', 'w'))
json.dump(OUT, open(HERE / 'aligned_results.json', 'w'), indent=1); log('DONE')
