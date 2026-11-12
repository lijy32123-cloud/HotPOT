# Configurations

Run commands from the repository root. YAML keys name CLI arguments with underscores or hyphens; CLI values take priority. `--config_file` is an alias for `--config`. Use `--help` for the available arguments and `--print-config` to validate argument names, types, and choices without starting a run.

| Configuration | Entry point | Purpose |
| --- | --- | --- |
| `pretrain_17m.yaml` | `train.py` | Two-GPU 17M pre-training |
| `pretrain_90m.yaml` | `train.py` | Two-GPU 90M pre-training |
| `evaluate.yaml` | `evaluate.py` | Source-checkpoint rollout evaluation |
| `finetune.yaml` | `finetune.py` | Known-task fine-tuning |
| `zero_shot.yaml` | `finetune.py` | Unseen-coefficient evaluation |
| `few_shot.yaml` | `finetune.py` | 64-shot coefficient adaptation |
| `wave.yaml` | `finetune.py` | 32-shot Wave-Gauss adaptation |

Paths in configurations are relative to the working directory. If you change `workspace`, also update checkpoint, metadata, and output paths as needed. `data_paths.json` maps transfer dependencies to paths **inside the workspace**.

For example, test a fine-tuned checkpoint in a separate invocation:

```bash
python finetune.py --config configs/finetune.yaml --mode known-test \
  --target-checkpoint ./workspace/runs/ns_finetuned/target_full_state.pt \
  --output ./workspace/results/ns_finetuned
```

The JSON files `pretrain.json` and `transfer.json` record the complete fixed experiment protocols, including settings implemented inside the trainer. They are reference manifests, **not CLI configurations**. The runnable configurations are the YAML files listed above.
