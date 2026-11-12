"""Check that documented configurations reach the real entry-point parsers."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hotpot.config import ConfigArgumentParser

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('entry,config,key,value', [
    ('train.py', 'pretrain_17m.yaml', 'scale', 17),
    ('train.py', 'pretrain_90m.yaml', 'scale', 90),
    ('evaluate.py', 'evaluate.yaml', 'task', 'ns2d_fno_1e-5'),
    ('finetune.py', 'finetune.yaml', 'mode', 'known-train'),
    ('finetune.py', 'zero_shot.yaml', 'mode', 'zero-shot'),
    ('finetune.py', 'few_shot.yaml', 'mode', 'fewshot-train'),
    ('finetune.py', 'wave.yaml', 'mode', 'wave-train'),
])
def test_entry_point_config_without_data_or_gpu(entry, config, key, value, tmp_path):
    output = tmp_path / 'must_not_be_created'
    result = subprocess.run(
        [sys.executable, str(ROOT / entry), '--config', str(ROOT / 'configs' / config),
         '--output', str(output), '--print-config'],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    resolved = json.loads(result.stdout)
    assert resolved[key] == value
    assert Path(resolved['output']) == output
    assert not output.exists()


def test_json_config_alias_and_cli_override(tmp_path):
    config = tmp_path / 'settings.json'
    config.write_text(json.dumps({'scale': 17, 'batch_size': 4, 'target': True}))
    parser = ConfigArgumentParser()
    parser.add_argument('--scale', type=int, choices=(17, 90), required=True)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--target', action='store_true')
    args = parser.parse_args(['--config_file', str(config), '--scale', '90'])
    assert (args.scale, args.batch_size, args.target) == (90, 4, True)


@pytest.mark.parametrize('content', ['batch_szie: 20', 'scale: 123'])
def test_invalid_configuration_fails_before_execution(content, tmp_path):
    config = tmp_path / 'invalid.yaml'
    config.write_text(content)
    parser = ConfigArgumentParser()
    parser.add_argument('--scale', type=int, choices=(17, 90))
    with pytest.raises(SystemExit) as error:
        parser.parse_args(['--config', str(config)])
    assert error.value.code == 2
