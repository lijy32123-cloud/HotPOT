# Simulated combustion: a 15-field reacting flow as a new family

This folder holds the coupled reacting-flow experiment added during the discussion period (appendix,
"Simulated combustion").

## Data and split

RealPDEBench, simulated combustion: NH3/CH4 flames at 30 operating conditions (fuel blend 0–80% NH3,
equivalence ratio 0.75–1.3), 2,001 frames each. Each frame has 15 coupled fields: absolute and gauge
pressure, heat release, seven species (CH4, CO, CO2, H2O, NH2, NH3, OH), temperature, three velocity
components and speed. The released 128×128 fields are subsampled by 2 to 64×64, as the official loader
does (`prepare_data.py`).

The published split files of this subset are empty, so `SPLIT_MANIFEST.json` fixes a whole-condition
split, written before any training:

| Part | Conditions |
| --- | --- |
| Training | 18 |
| Validation | 3 |
| Interpolation test | 3, inside the training ranges of fuel blend and equivalence ratio |
| Extrapolation test | 6, with 60% or 80% NH3 (training has 0–40%) |

## Task and models

Input: 10 frames; target: the next frame; longer horizons feed predictions back. No operating-condition
label enters any model. Fields are standardised with per-field training statistics; networks run at
128×128 (bilinear resize in and out) and are scored at 64×64.

| Arm | Model | Stored parameters |
| --- | --- | ---: |
| `moepot` | MoE-POT-T (official code), new 15-field input and output layers | 30,191,384 |
| `hotpot_direct` | shared pre-trained history encoder + new private predictor of the next state | 18,626,335 |
| `hotpot_increment` | same network; the output is the increment and the fixed update adds it to the current state | 18,626,335 |
| `hotpot_conservation` | same encoder; 22 typed coefficients per cell, executed by a fixed finite-volume, reaction and exchange program on eight molar pools, three momentum components and total energy (`physics.py`) | 18,627,249 |

All four start from six-dataset pre-training checkpoints: HotPOT-T (E1000, released) and a MoE-POT-T
trained by us with the official code and tiny configuration on the same six datasets for 1000 epochs.

## Training (one script per stage, identical for the four models)

1. `train_stage1.py`: 5,000 updates, AdamW (β = 0.9, 0.99; weight decay 1e-6), batch 8, learning rate
   1e-4 with 100 warm-up updates and cosine decay, gradient norm clipped to 1, mean squared error of the
   standardised next frame, FP32. All models see the same windows at the same update.
2. `train_stage2.py`: continuation from update 5,000 with the optimiser state, at learning rate 3e-5,
   under one plateau controller (`convergence.py`); validation every 2,500 updates with 100-frame
   rollouts. Kept checkpoints: lowest relative L2, lowest normalised MSE, last.

The reported runs were stopped before the controller's stopping criterion, at 111,000 (`moepot`),
84,000 (`hotpot_direct`), 100,888 (`hotpot_increment`) and 112,250 (`hotpot_conservation`) updates.

## Evaluation

`evaluate.py` rolls every saved checkpoint out for 20 frames from windows starting every 50 frames (40
per condition) and stores, per window and field, the mean squared error of the standardised fields over
the first *h* predicted frames. `compare.py` selects, for each model and horizon, the checkpoint with
the lowest validation error and reports its test error with a paired bootstrap over test windows.
The reported test numbers were computed once, after this rule was fixed. (The 5,000-update stage of each
run had earlier been scored on the test conditions with a relative-L2 metric; those scores were not
used for selection.)

### Interpolation test (3 conditions, 120 windows), normalised MSE

| Model | h = 1 | h = 5 | h = 10 | h = 20 |
| --- | ---: | ---: | ---: | ---: |
| Repeat the last input frame | 0.251 | 0.786 | 0.818 | 0.852 |
| Mean of the 10 input frames | 0.456 | 0.544 | 0.541 | 0.546 |
| MoE-POT-T | 0.0545 | 0.148 | 0.272 | 0.430 |
| HotPOT, direct-state predictor | 0.0346 | 0.125 | 0.287 | 0.474 |
| HotPOT, conservation-form expert | 0.0850 | 0.231 | 0.343 | 0.536 |
| HotPOT, increment expert | 0.0316 | 0.116 | 0.223 | 0.358 |
| Increment expert vs MoE-POT-T | −42% [−45, −40] | −22% [−26, −17] | −18% [−22, −14] | −17% [−20, −14] |

The increment expert is lower than MoE-POT-T on all three conditions at every horizon and on 15, 14, 14
and 11 of the 15 fields. With the last checkpoint of every model instead of the validation-selected one
the differences are −42%, −20%, −16% and −15%.

### Extrapolation test (6 conditions with 60% or 80% NH3, 240 windows), normalised MSE

| Model | h = 1 | h = 5 | h = 10 | h = 20 |
| --- | ---: | ---: | ---: | ---: |
| Repeat the last input frame | 2.08 | 5.66 | 7.05 | 7.66 |
| Mean of the 10 input frames | 4.32 | 5.24 | 5.49 | 5.53 |
| MoE-POT-T | 4.77 | 6.87 | 8.35 | 9.50 |
| HotPOT, direct-state predictor | 2.16 | 4.96 | 7.17 | 8.79 |
| HotPOT, conservation-form expert | 2.01 | 4.20 | 5.17 | 5.90 |
| HotPOT, increment expert | 2.75 | 7.63 | 10.96 | 13.62 |

Fields are standardised with training statistics, so values above 1 reflect states far outside the
training range. No model is better than the two predictors without training.

### What the experiment shows and what it does not

- Only the one-step gain is stable across splits. On the three validation conditions it is −39%, but
  MoE-POT-T is 11–12% lower there at 10 and 20 frames. Each split has three conditions.
- The predictable horizon is short: at 100 frames the errors are 0.86 (increment expert) and 0.97
  (MoE-POT-T), above the 0.52 of the input-mean predictor.
- No model extrapolates to the unseen fuel blends.
- The conservation-form expert is less accurate than the increment expert on the interpolation test.
  The physics-structured update tried for this system did not improve accuracy; on the unseen blends it
  degrades the least.

## Running

```bash
export COMBUSTION_CACHE=<dir>/combustion_numerical.npy
export HOTPOT_T_CHECKPOINT=<HotPOT-T_pretrain_E1000.pt>
export MOEPOT_CODE=<official MoE-POT repository> MOEPOT_T_CHECKPOINT=<six-dataset MoE-POT-T checkpoint>
python prepare_data.py <RealPDEBench>/combustion/hf_dataset/numerical <dir>/combustion_numerical
python train_stage1.py --arm hotpot_increment --out runs/stage1/hotpot_increment
python train_stage2.py --arm hotpot_increment --stage1 runs/stage1/hotpot_increment --out runs/stage2/hotpot_increment
python evaluate.py --arm hotpot_increment --normalization runs/stage1/hotpot_increment/PROTOCOL.json --split validation \
    --out results/validation_hotpot_increment.json best=runs/stage2/hotpot_increment/best.pt \
    best_std=runs/stage2/hotpot_increment/best_std_diagnostic.pt last=runs/stage2/hotpot_increment/last.pt
python compare.py results    # after validation_<arm>.json and test_<arm>.json exist for the four arms
```

Stage 2 runs until its controller stops it or the process is interrupted (SIGTERM saves a checkpoint);
starting the same command again resumes from `last.pt`. `prepare_data.py` additionally needs the
`datasets` package. Both training scripts accept `--smoke` (two updates and a three-step rollout) and
`--device cpu`.
