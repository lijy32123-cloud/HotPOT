"""Table of the combustion experiment from the files written by evaluate.py.

For each model and horizon the checkpoint with the lowest validation normalised MSE is selected; its test error is
reported per test group, with a paired bootstrap (20,000 resamples of test windows) against MoE-POT-T.

usage: python compare.py <dir>     # <dir>/validation_<arm>.json and <dir>/test_<arm>.json for the four arms
"""
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ('moepot', 'hotpot_direct', 'hotpot_conservation', 'hotpot_increment')
HORIZONS = ('1', '5', '10', '20')


def main(folder):
    folder = Path(folder)
    rng = np.random.default_rng(0)
    val = {a: json.load(open(folder / f'validation_{a}.json')) for a in ARMS}
    test = {a: json.load(open(folder / f'test_{a}.json')) for a in ARMS}
    pick = {(a, h): min(val[a]['checkpoints'], key=lambda c: np.mean(val[a]['checkpoints'][c]['nmse']['validation'][h])) for a in ARMS for h in HORIZONS}
    for group, names in test['moepot']['groups'].items():
        print('=====', group, len(names), 'trajectories')
        trivial = test['moepot']['trivial'][group]
        for k in trivial:
            print('  %-22s' % k + ' '.join('h=%s %.4f' % (h, np.mean(trivial[k][h])) for h in HORIZONS))
        for h in HORIZONS:
            err = {a: np.array(test[a]['checkpoints'][pick[a, h]]['nmse'][group][h]) for a in ARMS}  # (windows, fields)
            print('  h=%s ' % h + ' | '.join('%s[%s] %.4f' % (a, pick[a, h], err[a].mean()) for a in ARMS))
            base = err['moepot'].mean(1)
            for a in ARMS[1:]:
                x = err[a].mean(1)
                idx = rng.integers(0, len(x), (20000, len(x)))
                low, high = np.percentile(100 * (x[idx].mean(1) / base[idx].mean(1) - 1), [2.5, 97.5])
                per = x.reshape(len(names), -1).mean(1) / base.reshape(len(names), -1).mean(1) - 1
                print('      %-20s vs moepot %+.1f%% [%+.1f, %+.1f]  windows %d/%d  trajectories %d/%d  fields %d/15' % (
                    a, 100 * (x.mean() / base.mean() - 1), low, high, (x < base).sum(), len(x), (per < 0).sum(), len(names),
                    (err[a].mean(0) < err['moepot'].mean(0)).sum()))


if __name__ == '__main__':
    main(sys.argv[1])
