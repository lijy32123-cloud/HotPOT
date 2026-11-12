"""Pre-train HotPOT with the paper's two-GPU protocol."""
from .config import ConfigArgumentParser
from .models import build_model, VARIANTS


def main():
    from ._internal.encoder import train_ddp as trainer
    parser = ConfigArgumentParser(description=__doc__)
    parser.add_argument('--scale', type=int, choices=(17, 90), default=17)
    parser.add_argument('--variant', choices=VARIANTS, default='full')
    selected = trainer.add_arguments(parser).parse_args()
    trainer.MODEL_CLASS = lambda **kw: build_model(selected.scale, selected.variant, **kw)
    trainer.MODEL_SYSTEM = f'hotpot.{selected.variant}.{selected.scale}'
    trainer.CHECKPOINT_SCHEMA = 'hotpot.pretraining.v1'
    trainer.main(selected)


if __name__ == '__main__':
    main()
