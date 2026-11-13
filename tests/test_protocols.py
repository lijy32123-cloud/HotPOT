"""Protocol regression checks; all tensors are synthetic CPU data."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hotpot.protocols import maximum_learning_rate, normalize_arguments, training_seed
from hotpot._internal.adaptation import transfer_runtime as runtime
from hotpot._internal.adaptation.coefficient_rollout import rollout
from hotpot._internal.coefficients.data import TARGETS, split_rows
from hotpot._internal.encoder.finetune_ddp import TargetPhaseAwareEpochPlan
from hotpot._internal.data_protocol.data import DistributedAddressBatchSampler

ROOT = Path(__file__).resolve().parents[1]


def test_source_and_target_substeps_reuse_identical_parameter_state():
    from hotpot import build_model
    from hotpot._internal.foundation.config import tiny_test_config
    torch.manual_seed(3)
    source = build_model(config=tiny_test_config())
    torch.manual_seed(3)
    target = build_model(config=tiny_test_config(), target=True)
    assert source.model.programs['fno'].substeps == 4
    assert target.model.programs['fno'].substeps == 8
    assert all(torch.equal(value, target.state_dict()[name])
               for name, value in source.state_dict().items())


def test_print_config_checks_task_specific_shot_budget(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(ROOT / 'finetune.py'), '--config',
                             str(ROOT / 'configs/few_shot.yaml'), '--shots', '32',
                             '--print-config'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert '16, 64, or 256' in result.stderr


@pytest.mark.parametrize('task,rate,updates', [
    ('ns2d_fno_1e-5', 1e-5, 5000), ('ns2d_fno_1e-3', 1e-3, 5000),
    ('ns2d_pdb_M1e-1_eta1e-2_zeta1e-2', 1e-4, 45000),
    ('swe_pdb', 1e-3, 4600), ('dr_pdb', 1e-3, 4600), ('cfdbench', 1e-5, 45000),
])
def test_known_task_final_schedule_and_exposure(task, rate, updates):
    from hotpot._internal.data_protocol.manifest import BASE6
    assert maximum_learning_rate('known-train', task) == rate
    assert training_seed('known-train') == 20260815
    # `finetune.py --mode known-train` runs one rank with batch 40 and microbatch 20
    # at the task's own maximum learning rate; the execution check must accept all six.
    from hotpot._internal.encoder.finetune_ddp import _e200_execution_profile
    assert _e200_execution_profile(epochs=200, warmup_epochs=40, local_batch_size=40,
                                   gradient_microbatch_size=20,
                                   learning_rate=rate) == 'single_rank_e200_b40_micro20'
    index = next(i for i, row in enumerate(BASE6) if row.key == task)
    plan = TargetPhaseAwareEpochPlan(index, seed=20260815)
    sampler = DistributedAddressBatchSampler(plan, rank=0, world_size=1, local_batch_size=40)
    assert 200 * len(sampler) == updates
    # Final refits batch the full phase-aware epoch in order, including the remainder.
    for epoch in (0, 1, 199):
        sampler.set_epoch(epoch)
        observed = [address for batch in sampler for address in batch]
        assert observed == list(plan.epoch(epoch))
        assert sorted(address.data_index for address in observed) == list(range(BASE6[index].ntrain))


@pytest.mark.parametrize('target', ['fno_1e-4', 'cns_m1_eta1e-2'])
def test_coefficient_subsets_and_rates(target):
    previous = ()
    for shots, ns_rate in ((16, 1e-4), (64, 1e-4), (256, 1e-5)):
        rows, validation = split_rows(TARGETS[target], shots, 0)
        assert rows[:len(previous)] == previous
        assert len(rows) == shots and len(validation) == 256
        assert not set(rows) & set(validation)
        assert validation == tuple(range(TARGETS[target].rows - 256, TARGETS[target].rows))
        assert maximum_learning_rate('fewshot-train', target, shots) == (ns_rate if target == 'fno_1e-4' else 1e-3)
        previous = rows


@pytest.mark.parametrize('mode,task,shots,allowed', [
    ('fewshot-train', 'fno_1e-4', 32, False),
    ('fewshot-train', 'fno_1e-4', 256, True),
    ('fewshot-train', 'cns_m1_eta1e-2', 128, False),
    ('wave-train', 'Wave-Gauss', 32, True),
    ('wave-train', 'Wave-Layer', 256, False),
])
def test_distinct_coefficient_and_wave_budgets(mode, task, shots, allowed):
    args = SimpleNamespace(mode=mode, task=task, shots=shots, scale=17, arm='transfer',
                           selection='original-val', target_checkpoint=None)
    if allowed:
        normalize_arguments(args)
    else:
        with pytest.raises(ValueError):
            normalize_arguments(args)


def test_rollout_preserves_support_channels_and_uses_whole_horizon_norm():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, x, programs, values, masks):
            assert torch.all(x[..., 1:] == 7)
            assert programs == ['fno_vorticity_ns']
            self.calls += 1
            p = torch.zeros_like(x[..., -1:, :])
            p[..., 0] = self.calls
            return p
    x = torch.ones(1, 2, 2, 10, 4)
    x[..., 1:] = 7
    y = torch.ones(1, 2, 2, 20, 4)
    y[..., 0] = torch.arange(1, 21).square()
    mask = torch.zeros(1, 2, 2, 1, 4)
    mask[..., 0] = 1
    model = Model()
    result = rollout(model, TARGETS['fno_1e-4'], [(torch.tensor([9]), x, y, mask)], device='cpu')
    t = torch.arange(1, 21, dtype=torch.float64)
    expected = torch.linalg.vector_norm(t - t.square()) / torch.linalg.vector_norm(t.square())
    assert model.calls == 20
    assert result['rows'][0]['row'] == 9
    assert result['mean_l2re'] == pytest.approx(float(expected), abs=1e-10)
    assert result['mean_l2re'] != pytest.approx(float(((t - t.square()).abs() / t.square()).mean()))


def test_coefficient_selection_skips_invalid_candidate_and_restores_full_state(tmp_path, monkeypatch):
    from hotpot.transfer import checkpoint_payload
    model = torch.nn.Linear(1, 1)
    model.register_buffer('completed', torch.tensor(0))
    args = SimpleNamespace(mode='fewshot-train', task='fno_1e-4', shots=16,
                           selection='original-val', smoke_updates=0)
    monkeypatch.setattr(runtime, 'training_identity', lambda *args: {})
    monkeypatch.setattr(runtime, 'checkpoint_payload', checkpoint_payload)
    def batches(epoch):
        model.completed.fill_(epoch)
        yield model(torch.ones(1, 1)).square().reshape(1), {'synthetic_epoch': epoch}
    def validate():
        if int(model.completed) == 10:
            raise FloatingPointError('synthetic rejected candidate')
        return float(abs(int(model.completed) - 20))
    result = runtime.fit(args, {'recipe': {'test': 'synthetic'}}, tmp_path, model,
                         batches, validate, {})
    assert result['updates'] == 500 and result['selected_epoch'] == 20
    assert int(model.completed) == 20
    checkpoint = torch.load(tmp_path / 'target_full_state.pt', weights_only=True)
    assert checkpoint['validation_l2re'] == 0
    assert not checkpoint['test_used_for_selection']
    log = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert log[9]['validation_l2re'] is None and log[9]['validation_invalid_reason']
