"""Load flat YAML/JSON command configurations with CLI overrides."""
import argparse
import json
from pathlib import Path
import sys

import yaml


class ConfigArgumentParser(argparse.ArgumentParser):
    """Accept --config (or --config_file), then validate through argparse."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('allow_abbrev', False)
        super().__init__(*args, **kwargs)
        self.add_argument('--config', '--config_file', type=Path,
                          help='YAML or JSON configuration; command-line values take priority')
        self.add_argument('--print-config', action='store_true',
                          help='Validate and print the resolved arguments without running')

    def validate_args(self, args):
        """Entry points may validate combinations before --print-config exits."""

    def parse_args(self, args=None, namespace=None):
        args = list(sys.argv[1:] if args is None else args)
        bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        bootstrap.add_argument('--config', '--config_file', type=Path)
        selected, _ = bootstrap.parse_known_args(args)
        defaults = []
        if selected.config is not None:
            try:
                content = yaml.safe_load(selected.config.read_text(encoding='utf-8'))
            except (OSError, yaml.YAMLError) as error:
                self.error(f'Cannot read configuration: {error}')
            if not isinstance(content, dict):
                self.error('Configuration must be a mapping of argument names to scalar values')
            seen = set()
            for key, value in content.items():
                if not isinstance(key, str):
                    self.error('Configuration keys must be strings')
                option = '--' + key.replace('_', '-')
                action = self._option_string_actions.get(option)
                if action is None or option in ('--config', '--config-file', '--print-config', '--help'):
                    self.error(f'Unknown or reserved configuration key: {key}')
                if option in seen:
                    self.error(f'Duplicate configuration key: {key}')
                seen.add(option)
                if value is None:
                    continue
                if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                    if not isinstance(value, bool):
                        self.error(f'{key} must be a boolean')
                    if value == action.const:
                        defaults.append(option)
                elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    defaults.append(f'{option}={value}')
                else:
                    self.error(f'{key} must be a scalar argument value')
        result = super().parse_args(defaults + args, namespace)
        self.validate_args(result)
        if result.print_config:
            print(json.dumps({k: v for k, v in vars(result).items()
                              if k not in ('config', 'print_config')}, indent=2, default=str))
            self.exit()
        return result
