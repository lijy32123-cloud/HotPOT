# Real combustion (OH\* intensity): modal state-space expert

This folder holds the expert used for the RealPDEBench real-combustion task added during the
discussion period. The task observes one field only, the OH\* chemiluminescence intensity `I`
(64×64): 20 observed frames in, 20 frames out, and 40/60 frames by feeding predictions back,
with the official splits (46,223 / 5,718 / 5,719 windows from 30 trajectories), normalisation
and `evaluate()` routine. No velocity, temperature, species or operating-condition label
enters any model.

## Why a different update is needed here

`I` is a line-of-sight emission signal, not a closed state: it has no conservation law of its
own and no pointwise evolution equation in the observed variable. Its predictable content is a
low-dimensional global pulsation of the flame (one POD mode carries 56% of the fluctuation
energy, four carry 72%). A predictor that writes pixels and is fed its own pixels back loses
this leading mode over successive blocks, which is why the direct-state predictor degrades
from 20 to 60 frames.

## The expert

The expert keeps the paper's interface: a learned predictor supplies quantities, and a fixed
program with no trainable parameters turns them into the next state.

1. **Fixed basis.** Mean image and 128 POD modes from the start frames of the training windows
   (`build_basis.py`). Projection and reconstruction are fixed linear maps.
2. **Observation step (first block, real history).** The fine-tuned HotPOT predictor
   (`network/`, initialised from the six-PDE checkpoint) predicts the next 20 frames; the
   program projects them onto the basis (`project_network.py`). A compact modal propagator
   reads the modal history, and the two estimates are fused by 18 fixed scalar gains
   (6 mode groups × 3 lead-time groups).
3. **Time step (later blocks).** The state carried forward is the block of modal amplitudes,
   not pixels. A modal propagator (ridge-initialised linear map plus a small correction
   network on the first 64 modes) advances it one block; the program reconstructs the field.
4. **State recognition.** The official evaluation calls the model again on its own output.
   The program tells an observed history from a state it synthesised itself by the residual
   outside the basis (observed ≥ 4.4e-4, synthesised ≤ 5.9e-9; no misclassification in
   68,598 calls) and applies step 2 or step 3 accordingly.

`modal_expert.py` fits the propagators and gains and runs the official evaluation.

## Results (official `evaluate()`, test RMSE, lower is better)

| Model | Active parameters | 20 frames | 40 frames | 60 frames |
| --- | ---: | ---: | ---: | ---: |
| MoE-POT-T, fine-tuned | 19.65M | 0.020763 | 0.022462 | 0.024227 |
| HotPOT direct-state predictor | 17.43M | 0.021110 | 0.026594 | 0.031544 |
| HotPOT expert, network-only observation step | 17.44M + 5.31M | 0.020870 | 0.022009 | 0.023056 |
| HotPOT expert | 19.34M + 5.31M | 0.020712 | 0.021914 | 0.022974 |

"a + b" lists the observation step and the time step; only one of them runs on a given block.

Paired bootstrap over the 30 test trajectories, HotPOT expert against MoE-POT:
−0.25% [−0.97%, +0.37%] at 20 frames (not significant), −2.44% [−3.39%, −1.45%] at 40 and
−5.17% [−7.19%, −2.99%] at 60; the expert is better on 19, 19 and 22 of 30 trajectories.

### Controls

| Model | 20 frames | 40 frames | 60 frames |
| --- | ---: | ---: | ---: |
| Direct-state network + the same program | 0.020837 | 0.022066 | 0.023048 |
| MoE-POT-T + the same program | 0.020569 | 0.021738 | 0.022769 |

With the network held fixed, adding the program lowers the 40/60-frame error by 17% / 27%
for the HotPOT direct-state network and by 3.2% / 6.0% for MoE-POT. The gain therefore comes
from the predictor–update interface; on this task it is not specific to our backbone, and
MoE-POT with the same program is 0.7–0.9% lower than our expert.

## Protocol notes

- Learning rates {1e-4, 3e-4, 1e-3}, 10,000 updates, batch 16 and validation-RMSE selection
  are shared by all fine-tuned networks.
- The propagator size, the 18 gains and the checkpoint selection use the official validation
  split; the test split is evaluated with the unmodified official routine.
- This is a single observed channel. It tests the interface on a partially observed real
  system; it is not a demonstration of coupled multi-field combustion modelling.

## Running

```bash
export HOTPOT_CODE=<this repository> HOTPOT_T_CHECKPOINT=<HotPOT-T_pretrain_E1000.pt>
export MOEPOT_CODE=<MoE-POT repository> MOEPOT_T_CHECKPOINT=<six-dataset MoE-POT-T checkpoint>
python network/train.py --help                      # fine-tune the networks (task directory next to the code)
python build_basis.py <task_dir> <modal_dir>
python project_network.py <modal_dir> <network_run_dir>
python modal_expert.py <network_run_dir> <moepot_run_dir> <modal_dir> <out_dir>
```
