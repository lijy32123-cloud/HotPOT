"""Pre-train HotPOT: torchrun --nproc_per_node=2 train.py --config CONFIG."""
from hotpot.train import main

if __name__ == '__main__':
    main()
