# Third-party notices

`hotpot/_vendor/afno.py` contains the `AFNO2D` class and its activation table from `models/moepot.py` in [MoE-POT](https://github.com/haiyangxin/MoEPOT), commit `c36ec22aa492b9460d2e9f24cf4f0354d0f07540`. The extraction removes unrelated model classes and imports; it preserves the AFNO computation and initialization. The original header is retained:

> Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

MoE-POT credits DPOT for its architecture. The data conventions and baseline evaluation protocol also build on MoE-POT. This anonymous release does not claim authorship of the AFNO component or grant additional rights to third-party code or datasets. Please consult the upstream projects and dataset providers for their applicable terms.

PyTorch, NumPy, h5py, einops, and PyYAML are installed as dependencies and retain their respective licenses. Paper figures and the original HotPOT implementation are provided as anonymous research artifacts; this package does not add a new blanket license to third-party material.

`experiments/combustion/prepare_data.py` reads the simulated combustion trajectories of RealPDEBench with the `datasets` package; neither the data nor that package is redistributed here.
