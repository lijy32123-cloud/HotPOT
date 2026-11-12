"""Autoregressive evaluation using the released grid/channel mask."""
from .config import ConfigArgumentParser
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .models import build_model, load_checkpoint, VARIANTS
from ._internal.data_protocol.evaluation_manifest import EVALUATION
from ._internal.data_protocol.evaluate_e1000 import OfficialRolloutDataset, CertifiedCFDBTestSidecar
from ._internal.data_protocol.objective import simple_relative_lp


def main():
    p = ConfigArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--scale', type=int, choices=(17,90), default=17)
    p.add_argument('--variant', choices=VARIANTS, default='full')
    p.add_argument('--task', choices=[s.key for s in EVALUATION], required=True)
    p.add_argument('--cfdb-test-sidecar', type=Path)
    p.add_argument('--target', action='store_true', help='Use the execution settings for target-adapted checkpoints')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    model = build_model(a.scale,a.variant,workspace=a.workspace,target=a.target)
    load_checkpoint(model,a.checkpoint)
    model.to(a.device).eval()
    spec = next(s for s in EVALUATION if s.key == a.task)
    if spec.dataset_index == 5 and a.cfdb_test_sidecar is None:
        p.error('--cfdb-test-sidecar is required for CFDBench')
    sidecar = CertifiedCFDBTestSidecar(a.cfdb_test_sidecar) if spec.dataset_index == 5 else None
    loader = DataLoader(OfficialRolloutDataset(a.workspace.resolve(),spec),batch_size=a.batch_size,
                        shuffle=False,num_workers=a.num_workers)
    scores = {str(h):[] for h in spec.report_horizons}
    with torch.inference_mode():
        for rows,history,truth,mask in loader:
            rows,history,truth,mask = [v.to(a.device) for v in (rows,history,truth,mask)]
            predictions=[]
            for _ in range(spec.rollout_steps):
                pred=model(history,torch.full_like(rows,spec.dataset_index),rows,cfdb_sidecar=sidecar)
                if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite rollout')
                predictions.append(pred)
                history=torch.cat((history[...,1:,:],pred),dim=-2)
            joined=torch.cat(predictions,dim=-2)
            for h in spec.report_horizons:
                scores[str(h)].extend(simple_relative_lp(joined[...,:h,:],truth[...,:h,:],mask,reduction='none').cpu().tolist())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result={'task':a.task,'variant':a.variant,'scale':a.scale,'samples':len(loader.dataset),
            'metric':'native-grid masked per-channel relative L2 over the rollout horizon',
            'mean':{h:sum(v)/len(v) for h,v in scores.items()},'per_trajectory':scores}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result['mean'],indent=2))


if __name__ == '__main__':
    main()
