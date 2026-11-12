"""I-only 20->20 interfaces initialized from original six-PDE checkpoints."""
import os
from pathlib import Path
import hashlib
import sys
from dataclasses import replace
import torch
from torch import nn
import torch.nn.functional as F
from common import ROOT,digest

RELEASE=Path(os.environ['HOTPOT_CODE'])  # root of this repository
sys.path.insert(0,str(RELEASE));sys.path.insert(0,str(ROOT));sys.path.insert(0,str(RELEASE.parent))
HOT=Path(os.environ['HOTPOT_T_CHECKPOINT'])  # HotPOT-T_pretrain_E1000.pt
MOE=Path(os.environ['MOEPOT_T_CHECKPOINT'])  # six-dataset MoE-POT-T checkpoint
HASH={'hotpot_direct':'041408b1b02a944733147b5fabc2182382325d3305bd33f86f55a7783aa42684','moepot':'abbe4d6712953fdd489be8cc7a3fc27d9a3026f642d41196618b8615bad05d85'}

def temporal_resize(w, steps=20):
    old=w.shape[0]
    flat=w.reshape(old,-1).T.unsqueeze(0)
    return (F.interpolate(flat,size=steps,mode='linear',align_corners=True)[0].T*(old/steps)).reshape(steps,*w.shape[1:]).contiguous()

def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class DeterministicGlobalMaxPool(nn.Module):
    def forward(self,x):
        # Same first-maximum tie convention as AdaptiveMaxPool2d(1), with a
        # deterministic reduction backward instead of its CUDA atomic kernel.
        return x.flatten(2).max(dim=2).values[...,None,None]

def install_deterministic_pool(core):
    count=0
    for parent in core.modules():
        for name,module in list(parent.named_children()):
            if isinstance(module,nn.AdaptiveMaxPool2d):
                assert module.output_size in [1,(1,1)] and not module.return_indices
                setattr(parent,name,DeterministicGlobalMaxPool());count+=1
    assert count==4

class HotPOTDirect(nn.Module):
    def __init__(self):
        super().__init__()
        from hotpot._internal.foundation.config import tiny_17m_config
        from hotpot._internal.general_fields.model import GeneralSemanticFieldChart,TypedSavedStepHead
        from hotpot._internal.encoder.analyzer import SharedScalarFieldAnalyzer
        from hotpot._internal.encoder.private_mixer import PrivateTypedFieldMixer
        from hotpot._internal.encoder.action_decoder import ActionContextDecoder
        from hotpot._internal.history_features.nodiff_chart import install_no_difference_chart
        from hotpot._internal.data_protocol.manifest import CONDITION_FIELDS
        cfg=replace(tiny_17m_config(),history_steps=20,max_fields=1)
        self.chart=GeneralSemanticFieldChart(cfg,dynamic_channels=1,boundary='closed')
        self.shared_analyzer=SharedScalarFieldAnalyzer(cfg)
        self.private_mixer=PrivateTypedFieldMixer(cfg,dynamic_channels=1)
        self.decoder=ActionContextDecoder(cfg)
        self.head=TypedSavedStepHead(cfg,channels=1,boundary='closed')
        old=self.head.net[-1];self.head.net[-1]=nn.Conv2d(old.in_channels,20,1)
        nn.init.normal_(self.head.net[-1].weight,std=1e-4);nn.init.zeros_(self.head.net[-1].bias)
        self.nconditions=len(CONDITION_FIELDS)
        install_no_difference_chart(self)
    def forward(self,history):
        values=history.new_zeros(len(history),self.nconditions);mask=torch.zeros_like(values)
        features,field_mask,mean,log_scale,lags=self.chart(history)
        memory=self.shared_analyzer(features,field_mask,lags)
        private=self.private_mixer(memory,field_mask,mean,log_scale,values,mask)
        current=history[...,-1,:].permute(0,3,1,2).contiguous()
        # Absolute target-normalized intensity at each of the 20 future frames.
        return self.head(self.decoder(private),current,values,mask)

class MoEAdapter(nn.Module):
    def __init__(self,core):super().__init__();self.core=core
    def forward(self,history):return self.core(history)[0][...,0].permute(0,3,1,2).contiguous()

def build(arm,seed=0):
    assert arm in HASH
    torch.manual_seed(seed);source=HOT if arm=='hotpot_direct' else MOE
    assert digest(source)==HASH[arm],'original checkpoint hash mismatch'
    if arm=='hotpot_direct':
        model=HotPOTDirect()
        ck=torch.load(source,map_location='cpu',weights_only=True)
        state=next((ck[k] for k in ['model','model_state','state_dict','full_model_state'] if k in ck),ck)
        prefix='model.shared_analyzer.'
        encoder={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        assert encoder and encoder['temporal.weight'].shape[0]==10
        source_elements=sum(v.numel() for v in encoder.values())
        encoder['temporal.weight']=temporal_resize(encoder['temporal.weight'])
        model.shared_analyzer.load_state_dict(encoder,strict=True)
        reused='shared history encoder; temporal.weight deterministically adapted 10->20; new target mixer/decoder/output head'
    else:
        from experiments.moepot_mal_v15_official_fullrollout_eval_v1.official_loader import official_model_scope
        with official_model_scope(Path(os.environ['MOEPOT_CODE'])) as cls:
            core=cls(img_size=128,patch_size=8,mixing_type='afno',in_channels=4,out_channels=4,in_timesteps=10,out_timesteps=1,n_blocks=4,embed_dim=512,out_layer_dim=32,depth=4,modes=32,mlp_ratio=1,n_cls=6,normalize=False,act='gelu',time_agg='exp_mlp',is_finetune=True)
        ck=torch.load(source,map_location='cpu',weights_only=False)
        assert ck['epoch']==999
        core.load_state_dict(ck['model'],strict=True)
        old=core.patch_embed.proj[0]
        core.patch_embed.proj[0]=nn.Conv2d(4,old.out_channels,kernel_size=old.kernel_size,stride=old.stride,padding=old.padding)
        old=core.out_layer[-1]
        core.out_layer[-1]=nn.Conv2d(old.in_channels,20,kernel_size=old.kernel_size,stride=old.stride,padding=old.padding)
        assert core.time_agg_layer.w.shape[0]==10
        core.time_agg_layer.w=nn.Parameter(temporal_resize(core.time_agg_layer.w.detach()))
        core.time_agg_layer.n_timesteps=20;core.time_agg_layer.n_channels=1
        core.in_channels=1;core.out_channels=1;core.in_timesteps=20;core.out_timesteps=20
        core.requires_grad_(True);core.cls_head.requires_grad_(False)
        assert [float(b.MoE.gating.temperature) for b in core.blocks]==[.5]*4
        install_deterministic_pool(core)
        source_elements=sum(v.numel() for k,v in ck['model'].items() if not k.startswith(('patch_embed.proj.0.','out_layer.4.')))
        model=MoEAdapter(core)
        reused='pretrained temporal/spatial/MoE trunk; time_agg_layer.w deterministically adapted 10->20; new I input and 20-frame output interfaces; trained router temperature 0.5'
    ledger={'arm':arm,'source':str(source),'source_sha256':HASH[arm],'source_elements_reused_or_adapted':source_elements,'stored_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'initialization_sha256':state_hash(model),'adaptation':reused,'temporal_adaptation':'linear interpolation along normalized time, align_corners=True, multiplied by 10/20','input_information':['past 20 I frames','fixed grid coordinates'],'target':'20 absolute future I frames, target normalized','condition_inputs':'all zero and masked; no sim_id, fuel ratio, temperature, velocity or species'}
    return model,ledger
