"""CPU smoke checks; synthetic tensors are not scientific evaluations."""
import gc
import torch
import pytest
from hotpot import build_model, load_checkpoint
from hotpot._internal.foundation.config import tiny_test_config
from hotpot._internal.data_protocol.manifest import BASE6, ProgramId
from hotpot._internal.data_protocol.objective import simple_relative_lp

torch.set_num_threads(2)

class SyntheticSidecar:
    def __init__(self,family):self.name=family
    def family(self,row):return self.name
    def program(self,row):return ProgramId('cfdb_'+self.name)


@pytest.mark.parametrize('dataset,family',[(0,None),(1,None),(2,None),(3,None),(4,None),
                                           (5,'cavity'),(5,'cylinder'),(5,'tube')])
def test_routes(dataset,family):
    torch.manual_seed(11)
    model=build_model(config=tiny_test_config()).eval()
    history=torch.rand(1,128,128,10,4)+1
    with torch.inference_mode():
        prediction=model(history,torch.tensor([dataset]),torch.tensor([0]),
                         cfdb_sidecar=SyntheticSidecar(family) if family else None)
    assert prediction.shape==(1,128,128,1,4)
    assert torch.isfinite(prediction).all()
    for chart in model.model.charts.values():
        result=chart(history)
        assert torch.count_nonzero(result.features[:,:,:,1:6])==0


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(3)
    model=build_model(config=tiny_test_config())
    path=tmp_path/'state.pt'
    torch.save({'model':model.state_dict()},path)
    other=build_model(config=tiny_test_config())
    load_checkpoint(other,path)
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())


def test_loss_identity_and_scaling():
    target=torch.ones(2,8,8,3,4)
    mask=torch.ones(2,8,8,1,4)
    assert torch.allclose(simple_relative_lp(target,target,mask,reduction='none'),torch.zeros(2))
    assert torch.allclose(simple_relative_lp(target*2,target,mask,reduction='none'),torch.ones(2))


def test_transfer_context_and_metadata():
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from hotpot._internal.adaptation.transfer import patched, validate_wave_split
    obj=SimpleNamespace(value=1)
    with patched(obj,value=2):assert obj.value==2
    assert obj.value==1
    root=Path(__file__).resolve().parents[1]
    split=json.loads((root/'metadata/wave_split.json').read_text())
    for name in ('Wave-Gauss','Wave-Layer'):validate_wave_split(split,name)


def test_full_parameter_ledger():
    model=build_model(scale=17)
    assert sum(p.numel() for p in model.parameters())==97902720
    assert model.parameter_ledger()['stored_total']==97902720
    del model
    gc.collect()
