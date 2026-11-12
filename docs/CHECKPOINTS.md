# Checkpoints and Table 1 correspondence

The 14 files below reproduce the 24 HotPOT entries of Table 1 without any training. Each file stores only the model
tensors (`{"model": state_dict, "meta": {...}}`) and loads with `hotpot.load_checkpoint`. The tensors are bitwise
identical to the checkpoints evaluated for the paper; optimiser state and run metadata were removed.

Download location: [TODO: anonymous weights link]

## Files

| File | Scale | Stage | Bytes | SHA-256 |
| --- | ---: | --- | ---: | --- |
| `HotPOT-T_pretrain_E1000.pt` | 17M | pretrain E1000 | 391,847,761 | `4c600e1f7b0314bea980c6daa839ce66ad100f57467c5b22cdcab6d46f8cbc8d` |
| `HotPOT-S_pretrain_E1000.pt` | 90M | pretrain E1000 | 2,074,225,745 | `3800e428ef6d75798bf9fb06a6b9eb591d2f7e5eac27c00d9a733d0b78acc62a` |
| `HotPOT-T_ns2d_fno_1e-5_finetune_E200.pt` | 17M | finetune E200 | 391,855,807 | `8280ad5c502a73d7cd68cf9e7f59692890eb84ee3988ccc20a3c7d02eba4fd70` |
| `HotPOT-S_ns2d_fno_1e-5_finetune_E200.pt` | 90M | finetune E200 | 2,074,236,287 | `2701d6e1bda606739ff413cd7e60fffcb42f395c6e8478e0ba60dafde9e55447` |
| `HotPOT-T_ns2d_fno_1e-3_finetune_E200.pt` | 17M | finetune E200 | 391,855,807 | `8aef17d136b6166a9ce0b22d561deb54c448431a70fd7d7286a4de6e85719dfb` |
| `HotPOT-S_ns2d_fno_1e-3_finetune_E200.pt` | 90M | finetune E200 | 2,074,236,287 | `35165448e148261031df322e19e8a5c62aed7bcebf0d5b9fa02bdf634eb4bbe5` |
| `HotPOT-T_ns2d_pdb_M1e-1_eta1e-2_zeta1e-2_finetune_E200.pt` | 17M | finetune E200 | 391,867,115 | `046e2b3e65a362f97bf5cfb9cc5fee36681a81c1f6f4411c33da8990231ce9b6` |
| `HotPOT-S_ns2d_pdb_M1e-1_eta1e-2_zeta1e-2_finetune_E200.pt` | 90M | finetune E200 | 2,074,250,987 | `88d5bfa1cd6c91751916afbf8a05dfb0a05023be5954862637f149c3807880e1` |
| `HotPOT-T_swe_pdb_finetune_E200.pt` | 17M | finetune E200 | 391,852,123 | `4f62297100694ee37223a55336978591fb33c10fed094b6f270bc1a505b71d31` |
| `HotPOT-S_swe_pdb_finetune_E200.pt` | 90M | finetune E200 | 2,074,231,387 | `b37606009586b45494cf74b424fff37c716d864d2c7531723571c0ef56b9fc27` |
| `HotPOT-T_dr_pdb_finetune_E200.pt` | 17M | finetune E200 | 391,851,445 | `e319ed247c166a80854c9aa78479f48e14328fd6dd53cf9acf93e9ba7aa35776` |
| `HotPOT-S_dr_pdb_finetune_E200.pt` | 90M | finetune E200 | 2,074,230,581 | `9feb8ade90f5d170e790be2837deb5ccb289ad7b3d93264cb8cc370f8a7842d9` |
| `HotPOT-T_cfdbench_finetune_E200.pt` | 17M | finetune E200 | 391,852,737 | `e164bc3e6d52ef251545dfac8ead806a6725c75632c7c26e739a424affd5d06e` |
| `HotPOT-S_cfdbench_finetune_E200.pt` | 90M | finetune E200 | 2,074,232,193 | `6388299d9eaef0e210d6546e406e80d1990121f195b467b1a5b772b5305f2c85` |

## Table 1 entries

Test populations are the leading rows of each benchmark test file (`test_rows`); every prediction is fed back for the
full horizon and scored with the masked space–time relative L2 error described in the paper appendix.

| Model | Stage | Setting | Checkpoint | Test rows | Rollout steps | Expected L2RE | Paper |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| HotPOT-T | pretrain | NS 1e-5 | `HotPOT-T_pretrain_E1000.pt` | 0:200 | 10 | 0.0314921 | 0.0315 |
| HotPOT-T | pretrain | NS 1e-3 | `HotPOT-T_pretrain_E1000.pt` | 0:200 | 20 | 0.00324653 | 0.00325 |
| HotPOT-T | pretrain | CNS (0.1, 0.01) | `HotPOT-T_pretrain_E1000.pt` | 0:200 | 11 | 0.00670671 | 0.00671 |
| HotPOT-T | pretrain | SWE | `HotPOT-T_pretrain_E1000.pt` | 0:60 | 91 | 0.00176987 | 0.00177 |
| HotPOT-T | pretrain | DR | `HotPOT-T_pretrain_E1000.pt` | 0:60 | 91 | 0.014992 | 0.0150 |
| HotPOT-T | pretrain | CFDB | `HotPOT-T_pretrain_E1000.pt` | 0:1000 | 10 | 0.00218927 | 0.00219 |
| HotPOT-T | finetune | NS 1e-5 | `HotPOT-T_ns2d_fno_1e-5_finetune_E200.pt` | 0:200 | 10 | 0.0215203 | 0.0215 |
| HotPOT-T | finetune | NS 1e-3 | `HotPOT-T_ns2d_fno_1e-3_finetune_E200.pt` | 0:200 | 20 | 0.000480635 | 0.000481 |
| HotPOT-T | finetune | CNS (0.1, 0.01) | `HotPOT-T_ns2d_pdb_M1e-1_eta1e-2_zeta1e-2_finetune_E200.pt` | 0:200 | 11 | 0.00582883 | 0.00583 |
| HotPOT-T | finetune | SWE | `HotPOT-T_swe_pdb_finetune_E200.pt` | 0:60 | 91 | 0.00162415 | 0.00162 |
| HotPOT-T | finetune | DR | `HotPOT-T_dr_pdb_finetune_E200.pt` | 0:60 | 91 | 0.00163157 | 0.00163 |
| HotPOT-T | finetune | CFDB | `HotPOT-T_cfdbench_finetune_E200.pt` | 0:1000 | 10 | 0.00207672 | 0.00208 |
| HotPOT-S | pretrain | NS 1e-5 | `HotPOT-S_pretrain_E1000.pt` | 0:200 | 10 | 0.029867 | 0.0299 |
| HotPOT-S | pretrain | NS 1e-3 | `HotPOT-S_pretrain_E1000.pt` | 0:200 | 20 | 0.00323227 | 0.00323 |
| HotPOT-S | pretrain | CNS (0.1, 0.01) | `HotPOT-S_pretrain_E1000.pt` | 0:200 | 11 | 0.00622873 | 0.00623 |
| HotPOT-S | pretrain | SWE | `HotPOT-S_pretrain_E1000.pt` | 0:60 | 91 | 0.0016573 | 0.00166 |
| HotPOT-S | pretrain | DR | `HotPOT-S_pretrain_E1000.pt` | 0:60 | 91 | 0.0326083 | 0.0326 |
| HotPOT-S | pretrain | CFDB | `HotPOT-S_pretrain_E1000.pt` | 0:1000 | 10 | 0.00225445 | 0.00225 |
| HotPOT-S | finetune | NS 1e-5 | `HotPOT-S_ns2d_fno_1e-5_finetune_E200.pt` | 0:200 | 10 | 0.0192568 | 0.0193 |
| HotPOT-S | finetune | NS 1e-3 | `HotPOT-S_ns2d_fno_1e-3_finetune_E200.pt` | 0:200 | 20 | 0.000482719 | 0.000483 |
| HotPOT-S | finetune | CNS (0.1, 0.01) | `HotPOT-S_ns2d_pdb_M1e-1_eta1e-2_zeta1e-2_finetune_E200.pt` | 0:200 | 11 | 0.00536218 | 0.00536 |
| HotPOT-S | finetune | SWE | `HotPOT-S_swe_pdb_finetune_E200.pt` | 0:60 | 91 | 0.00153732 | 0.00154 |
| HotPOT-S | finetune | DR | `HotPOT-S_dr_pdb_finetune_E200.pt` | 0:60 | 91 | 0.00436468 | 0.00436 |
| HotPOT-S | finetune | CFDB | `HotPOT-S_cfdbench_finetune_E200.pt` | 0:1000 | 10 | 0.00202275 | 0.00202 |

## Command

Place the files under `workspace/checkpoints/` and the data as described in [DATA.md](DATA.md), then for example:

```bash
python evaluate.py --workspace ./workspace --checkpoint ./workspace/checkpoints/HotPOT-T_pretrain_E1000.pt --scale 17 --task ns2d_fno_1e-5 --batch-size 4 --num-workers 0 --output ./workspace/results/HotPOT-T_pretrain_ns2d_fno_1e-5.json
```

The exact command for every entry is stored in `metadata/checkpoints.json` (`table1[*].command`).
