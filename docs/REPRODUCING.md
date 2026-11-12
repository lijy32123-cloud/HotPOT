# Reproducing the experiments

## Pre-training

Use `torchrun --standalone --nproc_per_node=2 train.py --config configs/pretrain_17m.yaml` (or `pretrain_90m.yaml`). The fixed recipe is two ranks, 20 samples per rank, 545 optimizer updates per epoch, and 1,000 epochs. The six storage entries use the same sample counts and unit weights as the matched MoE-POT protocol. Each sample contains 10 history frames and a one-frame target. The phase-aware sampler chooses history windows; training is not a multi-step rollout loss.

Adam uses a maximum learning rate of 1e-3, betas (0.9, 0.9), and weight decay 1e-6. OneCycle uses a 20% warm-up and initial/final division factors of 1e4. Input noise amplitude is 5e-4 for NS (on its native grid) and DR. CNS, SWE, and CFDBench inputs remain clean. The source seed is 20260815. `configs/pretrain.json` records the recipe. Use `--smoke-updates 2` to test the distributed training path without a full run; this is not a trained checkpoint.

Use `--resume workspace/runs/full17/checkpoint_eNNNN.pt` with the original run's output directory to resume. The trainer restores the optimizer and schedule and checks its recorded protocol.

## Evaluation

Use the fixed E1000 source checkpoint. `evaluate.py` reports a mean and all trajectory scores at the requested setting's horizons. It rolls forward using the model's own predictions. The scored horizons are 10 for NS 1e-5 and CFDBench, 20 for NS 1e-3, 11 for CNS, and 91 for SWE/DR. The evaluation computes relative error over the horizon using the released grid/channel mask; this differs from averaging independent per-step errors.

For CFDBench, add `--cfdb-test-sidecar workspace/metadata/cfdb_test.json`. Use `--target` when evaluating target-adapted full model states.

## Known-task fine-tuning

Use `finetune.py --mode known-train` as in the README. The protocol uses 200 epochs, a fixed final checkpoint, RNG seed 20260815, global batch 40, and microbatch 20. Maximum learning rates are NS 1e-5: 1e-5; NS 1e-3: 1e-3; CNS: 1e-4; SWE: 1e-3; DR: 1e-3; CFDBench: 1e-5. These rates were selected on held-out T-scale E200 rollout validation and reused at both scales; final E200 refits use the full original training populations. It trains the shared encoder and the target's private branch. One invocation uses one GPU with gradient accumulation. Other branches remain frozen.

Evaluate in a separate process:

```bash
python finetune.py \
  --workspace ./workspace --paths configs/data_paths.json \
  --checkpoint ./workspace/runs/full17/checkpoint_e1000.pt --scale 17 \
  --target-checkpoint ./workspace/runs/ns_finetuned/target_full_state.pt \
  --mode known-test --task ns2d_fno_1e-5 \
  --output ./workspace/results/ns_finetuned
```

## Few-shot coefficient adaptation

```bash
python finetune.py \
  --workspace ./workspace --paths configs/data_paths.json \
  --checkpoint ./workspace/runs/full17/checkpoint_e1000.pt --scale 17 \
  --mode fewshot-train --task fno_1e-4 --shots 64 \
  --arm transfer --selection original-val \
  --output ./workspace/runs/ns_64shot
```

Use `cns_m1_eta1e-2` for CNS. NS/CNS shot counts are 16, 64, and 256. Training runs for 500 epochs with batch size `min(20, shots)` and 20% OneCycle warm-up. The maximum learning rate is selected from 1e-3, 1e-4 and 1e-5 by the lowest full-rollout validation error for each target and budget. The selected values are 1e-4, 1e-4 and 1e-5 for NS at these budgets and 1e-3 for CNS. Replicate 0 uses RNG seed 20260831 and the bundled deterministic subset sampler. The final 256 trajectories of the target training source are reserved for validation. Every 10 epochs, validation rolls forward from their first 10 frames over 20 NS steps or 11 CNS steps. It averages per-trajectory, per-channel whole-horizon relative L2 norms, accumulated in FP64 with FP32 model execution. Nonfinite validation candidates are ineligible; they do not stop coefficient training or replace an earlier finite checkpoint. The minimum validation error selects a **full state, including buffers**. Test with the corresponding `fewshot-test` mode and `--target-checkpoint`; the saved validation error is reconstructed before test evaluation. Set `--arm scratch` to train the same target architecture without loading pretrained weights.

## New-family Wave transfer

```bash
python finetune.py \
  --workspace ./workspace --paths configs/data_paths.json \
  --checkpoint ./workspace/runs/full17/checkpoint_e1000.pt --scale 17 \
  --mode wave-train --task Wave-Gauss --shots 32 \
  --arm transfer --selection original-val \
  --output ./workspace/runs/wave_gauss_32shot
```

Wave shot counts are 16, 32, 64, and 128; its maximum learning rate is 1e-4. Only the pretrained history encoder is copied. The Wave-private modules are freshly initialized, and all target parameters can train. Use `Wave-Layer` for the other Wave dataset and `--arm scratch` for the matched randomly initialized control. Scratch and encoder transfer use the same target architecture and initial predictor weights, differing in encoder initialization. Train/test use the bundled split and file-identity checks. Evaluate with `wave-test` and the saved `--target-checkpoint`, using the same source checkpoint, task, shot count, and selection setting.

Wave validation/test metrics average relative L2 error across trajectories **and forecast steps**. Wave uses an observation-space residual predictor because the required physical coefficients and boundary metadata are not supplied. Array padding is not a claim about the physical boundary condition.

## CFDBench condition transfer

`--mode zero-shot --task cfdb_cavity`, `cfdb_cylinder`, or `cfdb_tube` evaluates the corresponding family with the fixed pretrained model. The condition-space audit uses the bundled case descriptions and separates boundary, parameter, and geometry changes. It reports both trajectory scores and the case-uniform aggregate.

## Baseline and compute

Use the [official MoE-POT implementation](https://github.com/haiyangxin/MoEPOT) for the baseline. Match dataset rows, checkpoint-selection rules, activated-parameter scale, and evaluation masks; comparing total stored HotPOT parameters to active MoE-POT parameters would use different definitions. The repository includes the HotPOT execution paths, not baseline training logs or pretrained weights.

The paper's Wave comparison uses full MoE-POT fine-tuning, including its feature extractor and routing parameters. Both methods use the same trajectory splits, training budgets, and full-rollout validation for checkpoint selection.

The source release was smoke-tested on CPU. Full pre-training, transfer, and the paper's reported values require the datasets, the stated GPU training budgets, and their resulting checkpoints. No new full training experiment was run as part of packaging this release.
