# Data preparation

## Sources and layout

Obtain the NS, PDEBench, CFDBench, and Wave arrays through the public sources listed by [MoE-POT](https://github.com/haiyangxin/MoEPOT#dataset-protocol). Use its `data_generation/preprocess.py` conversion for the same storage convention. Keep the original train/test row order: the bundled CFDBench routing maps refer to that order, not to a newly shuffled dataset. Respect the source datasets' terms.

All HDF5 files use a dataset named `data`. Packed scalar arrays have shape `[N,H,W,T]`; packed multivariate arrays use `[N,H,W,T,C]`. A scatter directory contains `data_0.hdf5`, `data_1.hdf5`, etc., each with `[H,W,T,C]`.

| Setting | Path relative to workspace | Training rows | Stored grid | Channels |
| --- | --- | ---: | --- | ---: |
| NS, viscosity 1e-5 | `data/large/ns2d_1e-5_train.hdf5` | 1,000 | 64 × 64 | 1 |
| NS, viscosity 1e-3 | `data/large/ns2d_1e-3_train.hdf5` | first 1,000 | 64 × 64 | 1 |
| CNS, Mach 0.1 | `data/large/pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/train/` | 9,000 | 128 × 128 | 4 |
| SWE | `data/large/pdebench/swe_pdb/train/` | 900 | 128 × 128 | 1 |
| DR | `data/large/pdebench/dr_pdb/train/` | 900 | 128 × 128 | 2 |
| CFDBench | `data/large/cfdbench/ns2d_cdb_train.hdf5` | first 9,000 | 64 × 64 | 3 stored; 2 scored |

Replace `train` with `test` for the six evaluation paths. The dataset reader resizes to 128 × 128 and pads to four channels. The loss mask scores the designated physical channels on the native grid.

## CFDBench metadata

Run `python prepare_data.py --workspace ./workspace` before training. It installs:

- `metadata/cfdb_train.json`: the 9,000 training rows (581 cavity, 8,231 cylinder, 188 tube).
- `metadata/cfdb_test.json`: the separate 1,000 test rows.
- `metadata/cfdb_schema.json`: field layout and observation conventions.
- `metadata/cases/`: 433 case descriptions, expanded from the bundled JSON, for condition-space evaluation.

The train/test maps and case identities are preserved; machine paths and operational provenance were removed. A row's family determines its fixed numerical program. These labels are data metadata and do not come from model predictions. Do not substitute arbitrary row maps.

## Coefficient changes and Wave

`configs/data_paths.json` lists the exact relative paths used by the transfer entry point. It includes NS viscosity 1e-4, CNS Mach 1, and `Wave_Gauss_{train,valid}.hdf5` / `Wave_Layer_{train,valid}.hdf5`. The Wave arrays use `[N,T,H,W]` and float32.

The bundled Wave split fixes disjoint trajectory rows, 100 validation trajectories, and nested 16/32/64/128-shot subsets. File hashes and shapes are retained for identity checks. Its test rows are a custom held-out subset of the converted training archive; they are not a separately supplied official test set. The original trajectory-parent identifiers and complete conversion provenance are unavailable, as described in the manuscript. A differently converted array must not be silently relabeled as this exact experiment.

## Check availability

```bash
python prepare_data.py --workspace ./workspace --check
```

This checks metadata and representative array shapes, and reports missing paths. It does not download large arrays or reconstruct trained weights.
