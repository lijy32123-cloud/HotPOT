"""Fine-tuning, coefficient generalization, and new-family transfer."""
from .config import ConfigArgumentParser
from .protocols import normalize_arguments, training_seed
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from .models import build_model, load_checkpoint
from ._internal.adaptation import transfer_runtime as runtime
from ._internal.adaptation.transfer import KNOWN, FEWSHOT, CFDB, WAVE, SHOTS, MODES, recipe
from ._internal.adaptation.core import sha256, atomic_json
from ._internal.history_features.nodiff_chart import install_no_difference_chart


def new_model(a, bound):
    if a.arm == 'transfer':
        return bound
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260831)
        return build_model(a.scale, config=bound.model.config, workspace=a.repo, target=True)


def wave_model(a, bound):
    from ._internal.wave import wave_models as wave
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260831)
        model = wave.WaveResidualModel()
    private = wave._state_digest(model, skip_shared=True)
    install_no_difference_chart(model)
    if a.arm == 'transfer':
        runtime.checked_state(model.shared_analyzer, bound.model.shared_analyzer.state_dict())
    if wave._state_digest(model, skip_shared=True) != private:
        raise RuntimeError('Loading the encoder changed the private Wave parameters')
    return model.cuda(), wave.predict, {
        'private_init_sha256': private,
        'encoder_init_sha256': wave._state_digest(model.shared_analyzer),
        'initialization_seed': 20260831,
        'source_load_scope': 'shared_analyzer' if a.arm == 'transfer' else 'none',
        'physical_metadata': 'unavailable; zero values and availability masks',
    }


def checkpoint_payload(model, prepared, epoch, updates, validation, data_identity):
    return {'schema':'hotpot.transfer.full_state.v1','recipe':prepared['recipe'],
            'completed_epoch':epoch,'updates':updates,'validation_l2re':validation,
            'data_identity':data_identity,'full_model_state':runtime.full_state(model),
            'test_used_for_selection':False}


def load_target(a, prepared):
    if a.target_checkpoint is None:
        raise ValueError('--target-checkpoint is required for a test mode')
    snap=torch.load(a.target_checkpoint,map_location='cpu',weights_only=True)
    if snap.get('schema') != 'hotpot.transfer.full_state.v1':
        raise ValueError('Expected a full-state checkpoint from hotpot.transfer')
    if snap.get('recipe') != prepared['recipe']:
        raise ValueError('Target checkpoint task, source, shots, or selection differs')
    if snap.get('test_used_for_selection') is not False:
        raise ValueError('Invalid checkpoint-selection provenance')
    return snap


class TransferArgumentParser(ConfigArgumentParser):
    def validate_args(self, args):
        try:
            normalize_arguments(args)
        except ValueError as error:
            self.error(str(error))


def main():
    p=TransferArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--paths',type=Path,required=True,help='JSON mapping of data dependencies to workspace-relative paths')
    p.add_argument('--checkpoint',type=Path,required=True,help='Full source model checkpoint')
    p.add_argument('--target-checkpoint',type=Path)
    p.add_argument('--scale',type=int,choices=(17,90),default=17)
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--task',choices=KNOWN+FEWSHOT+CFDB+WAVE,required=True)
    p.add_argument('--arm',choices=('transfer','scratch'),default='transfer')
    p.add_argument('--shots',type=int,choices=SHOTS)
    p.add_argument('--selection',choices=('fixed-final','original-val'),default='original-val')
    p.add_argument('--smoke-updates',type=int,choices=(0,1,2),default=0)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode.startswith('known-'):
        if a.task not in KNOWN or a.arm!='transfer':p.error('known modes require a pre-training task and transfer arm')
        a.selection='fixed-final'
    elif a.mode.startswith('fewshot-'):
        if a.task not in FEWSHOT or a.shots is None:p.error('fewshot modes require a coefficient task and --shots')
    elif a.mode.startswith('wave-'):
        if a.task not in WAVE or a.shots is None or a.scale!=17:p.error('Wave requires a Wave task, --shots, and scale 17')
    elif a.task not in FEWSHOT+CFDB or a.arm!='transfer':
        p.error('zero-shot requires a coefficient/CFDB task and transfer arm')
    if a.mode.endswith('-test') and a.target_checkpoint is None:p.error('Test modes require --target-checkpoint')
    a.repo=a.workspace.resolve(strict=True)
    a.checkpoint=a.checkpoint.resolve(strict=True)
    if a.target_checkpoint is not None:a.target_checkpoint=a.target_checkpoint.resolve(strict=True)
    a.source_key=f'full{a.scale}'
    raw=json.loads(a.paths.read_text(encoding='utf-8'))
    paths={}
    for key,value in raw.items():
        if key.startswith('_'):continue
        resolved=(a.repo/value).resolve()
        if not resolved.is_relative_to(a.repo):raise ValueError('Data paths must stay inside --workspace')
        paths[key]=resolved
    a.output=a.output.resolve()
    if not a.output.is_relative_to(a.repo):p.error('--output must be inside --workspace')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    seed=training_seed(a.mode)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=not a.mode.startswith("known-"))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    bound=build_model(a.scale,workspace=a.repo,target=True)
    load_checkpoint(bound,a.checkpoint)
    source_sha=sha256(a.checkpoint)
    protocol=recipe(a,source_sha)
    protocol['version']='hotpot.release.v2'
    if a.mode=='zero-shot':protocol['epochs']=0
    prepared={'repo':a.repo,'source_path':a.checkpoint,'source_entry':{'sha256':source_sha},
              'paths':paths,'recipe':protocol}
    runtime.new_ours=new_model
    runtime.wave_model=wave_model
    runtime.checkpoint_payload=checkpoint_payload
    runtime.load_target=load_target
    if a.mode=='known-train':result=runtime.known_train(a,{},prepared,a.output,bound)
    elif a.mode=='known-test':result=runtime.known_test(a,prepared,a.output,bound)
    elif a.mode.startswith('fewshot-'):result=runtime.fewshot(a,prepared,a.output,bound)
    elif a.mode.startswith('wave-'):result=runtime.wave(a,prepared,a.output,bound)
    elif a.task in CFDB:result=runtime.cfdb_evaluate(a,prepared,a.output,bound)
    else:result=runtime.ood_evaluate(a,prepared,a.output,bound.model)
    atomic_json(a.output/'COMPLETE.json',{'status':'PASS','mode':a.mode,'recipe':protocol,'result':result})


if __name__=='__main__':
    main()
