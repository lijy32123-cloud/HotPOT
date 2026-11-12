"""Official real-I task utilities. No numerical fields or condition labels enter models."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SHARED = ROOT.parent

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)

def save(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp); tmp.replace(path)

def resize(x, size):
    """Published DPOT FFT resize on the final two spatial axes, unchanged."""
    f = torch.fft.rfft2(x, norm='backward')
    z = torch.zeros((*x.shape[:-2], size[0], size[1]//2+1), dtype=f.dtype, device=f.device)
    a = min((f.shape[-2]+1)//2, (size[0]+1)//2)
    b = min(f.shape[-1], size[1]//2+1)
    c = min(f.shape[-2]//2, size[0]//2)
    z[..., :a, :b] = f[..., :a, :b]
    z[..., -c:, :b] = f[..., -c:, :b]
    return torch.fft.irfft2(z, s=size).real * (size[0]/x.shape[-2]) * (size[1]/x.shape[-1])

def predict(model, history):
    assert history.ndim == 5 and history.shape[1:] == (20,64,64,1)
    y = model(history)
    assert y.shape == (len(history),20,64,64,1)
    return y

class IntensityData:
    def __init__(self, device='cpu'):
        check = json.loads((BASE/'CACHE_CHECK.json').read_text())
        assert check['passed']
        self.ids = check['metadata']['sim_ids']
        a = np.load(check['path'], mmap_mode='r')
        assert a.shape[0] == 30 and a.shape[1] == 2002 and a.shape[-2:] == (64,64)
        # One observed channel, no simulated temperature/velocity/species.
        self.array = torch.from_numpy(np.array(a.reshape(30,2002,64,64), copy=True)).to(device)
        self.device = torch.device(device)
        values = torch.load(BASE/'metadata/mean_std.pt', map_location='cpu', weights_only=True)
        self.mx, self.my, self.sx, self.sy = [float(v[0]) for v in values]
        self.indices = {}
        self.rows = {}
        for split in ['train','val','test']:
            rows = json.loads((BASE/'metadata'/f'{split}_index_real.json').read_text())
            self.rows[split] = rows
            self.indices[split] = torch.tensor([[self.ids.index(r['sim_id']),r['time_id']] for r in rows], device=device)
        assert [len(self.rows[s]) for s in ['train','val','test']] == [46223,5718,5719]

    def valid_indices(self, split, horizon):
        # Match the official strict filter and n_sim_frame=2001, including the 2002-frame row.
        return torch.nonzero(self.indices[split][:,1] + 20 + horizon < 2001).flatten()

    def batch(self, split, selection, horizon=20):
        q = self.indices[split][selection]
        offsets = torch.arange(20+horizon, device=self.device)
        z = self.array[q[:,0,None],q[:,1,None]+offsets].unsqueeze(-1)
        return (z[:,:20]-self.mx)/self.sx, z[:,20:]

    def target(self, physical): return (physical-self.my)/self.sy
    def physical(self, target_normalized): return target_normalized*self.sy+self.my
    def feedback(self, target_normalized): return (self.physical(target_normalized)-self.mx)/self.sx

class Sampler:
    """Shuffled epochs with retained last partial batch; state is checkpointed."""
    def __init__(self, n, seed=0):
        self.n=n; self.generator=torch.Generator().manual_seed(seed)
        self.order=torch.randperm(n,generator=self.generator); self.cursor=0; self.epoch=0
    def next(self, batch=16):
        if self.cursor==self.n:
            self.order=torch.randperm(self.n,generator=self.generator);self.cursor=0;self.epoch+=1
        stop=min(self.n,self.cursor+batch);out=self.order[self.cursor:stop];self.cursor=stop
        return out
    def state_dict(self):
        return {'order':self.order,'cursor':self.cursor,'epoch':self.epoch,'rng':self.generator.get_state()}
    def load_state_dict(self,state):
        self.order=state['order'];self.cursor=state['cursor'];self.epoch=state['epoch'];self.generator.set_state(state['rng'])

@torch.inference_mode()
def evaluate(model, data, split='val', horizon=20, batch=64, limit=None, progress=None):
    model.eval(); valid=data.valid_indices(split,horizon)
    if limit is not None: valid=valid[:limit]
    squared=0.; absolute=0.; elements=0; relative=[]; records=[]; norm_batch_losses=[]
    for j in range(0,len(valid),batch):
        chosen=valid[j:j+batch];history,truth=data.batch(split,chosen,horizon)
        outputs=[]
        for _ in range(horizon//20):
            y=predict(model,history);outputs.append(data.physical(y));history=data.feedback(y)
        p=torch.cat(outputs,dim=1)
        if not torch.isfinite(p).all():raise FloatingPointError('nonfinite predicted I')
        delta=(p-truth).double();se=delta.square().flatten(1).sum(1)
        denominator=truth.double().square().flatten(1).sum(1)
        assert bool((denominator>0).all()),'official relative L2 denominator must be positive'
        rel=(se/denominator).sqrt(); rmse=(se/(horizon*64*64)).sqrt()
        squared+=float(se.sum());absolute+=float(delta.abs().sum());elements+=delta.numel()
        relative.extend(rel.cpu().tolist());norm_batch_losses.append(float(delta.square().mean())/data.sy**2)
        for k,idx in enumerate(chosen.cpu().tolist()):
            records.append({'index':idx,'sim_id':data.rows[split][idx]['sim_id'],'time_id':data.rows[split][idx]['time_id'],'relative_l2':float(rel[k]),'rmse':float(rmse[k])})
        if progress is not None:progress(min(j+batch,len(valid)),len(valid))
    return {'split':split,'horizon':horizon,'windows':len(valid),'rmse':(squared/elements)**.5,'mae':absolute/elements,'relative_l2':float(np.mean(relative)),'normalized_mse':float(np.mean(norm_batch_losses)),'records':records}
