"""Write a single-expert copy of a HotPOT checkpoint.

Keeps the shared history encoder and the expert of one PDE family and drops the experts of
the other families. Predictions for the kept family are unchanged because the dropped experts
are never executed for it.

    python tools/prune_checkpoint.py SOURCE.pt OUTPUT.pt --family {fno,cns,swe,dr,cfdb}
"""
import argparse
from pathlib import Path
import torch

GROUPS = ('private_mixers', 'action_context_decoders', 'programs')
FAMILY_OF_TASK = {'ns2d_fno_1e-5': 'fno', 'ns2d_fno_1e-3': 'fno', 'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2': 'cns',
                  'swe_pdb': 'swe', 'dr_pdb': 'dr', 'cfdbench': 'cfdb'}


def prune(state, family):
    def keep(name):
        return not any(name.startswith(f'model.{g}.') and not name.startswith(f'model.{g}.{family}.') for g in GROUPS)
    kept = {k: v for k, v in state.items() if keep(k)}
    assert any(k.startswith(f'model.programs.{family}.') for k in kept), family
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--family', required=True, choices=sorted(set(FAMILY_OF_TASK.values())))
    args = parser.parse_args()
    snapshot = torch.load(args.source, map_location='cpu', weights_only=True)
    state = snapshot['model'] if isinstance(snapshot, dict) and 'model' in snapshot else snapshot
    kept = prune(state, args.family)
    meta = dict(snapshot.get('meta', {})) if isinstance(snapshot, dict) else {}
    meta.update(single_expert=args.family, tensors=len(kept), elements=int(sum(v.numel() for v in kept.values())))
    torch.save({'model': kept, 'meta': meta}, args.output)
    print(f'{args.output}: {len(kept)}/{len(state)} tensors, {meta["elements"]:,} of '
          f'{sum(v.numel() for v in state.values()):,} elements')


if __name__ == '__main__':
    main()
