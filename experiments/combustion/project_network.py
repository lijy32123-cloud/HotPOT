"""First-block predictions of the fine-tuned HotPOT predictor, projected on the global modal basis (inference only).

Usage: python project_network.py <out_dir> <network_run_dir>   (run from experiments/combustion/network)
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
R = Path(sys.argv[2])  # directory with the network code and runs/hotpot_relaxed/lr_0.001/best.pt
OUT = Path(sys.argv[1]); sys.path.insert(0, str(R))
import torch
from common import IntensityData, predict
from intensity_models import build

torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = torch.device('cuda'); data = IntensityData(device='cuda'); arr = data.array
bs = torch.load(OUT / 'basis.pt'); mu = bs['mu'].to(dev); V = bs['V'].to(dev)
model, _ = build(); ck = torch.load(R / 'runs/hotpot_relaxed/lr_0.001/best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ck['model']); model.cuda().eval(); AD = {}
with torch.inference_mode():
    for s in ['train', 'val', 'test']:
        q = data.indices[s]; out = torch.empty(len(q), 20, 128, device=dev); se = 0.
        for j in range(0, len(q), 64):
            c = q[j:j + 64]; z = arr[c[:, 0, None], c[:, 1, None] + torch.arange(40, device=dev)]
            p = data.physical(predict(model, ((z[:, :20] - data.mx) / data.sx).unsqueeze(-1)))[..., 0]
            out[j:j + 64] = ((p.flatten(2).double() - mu) @ V).float(); se += float((p - z[:, 20:]).double().square().sum())
        AD[s] = out.clone(); print('RELAXED step %d, %-5s 20 frames full-field rmse %.6f' % (ck['step'], s, (se / (len(q) * 20 * 4096)) ** .5), flush=True)
torch.save(AD, OUT / 'relaxed_modal.pt'); print('saved', flush=True)
