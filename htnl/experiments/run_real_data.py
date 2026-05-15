"""Run all 4 HTNL methods on real-world datasets and report AUC.

Usage
-----
    python -m htnl.experiments.run_real_data \
        --data /path/to/hkl_combined \
        --datasets argentina colombia flu \
        --seeds 0 1 2 \
        --out results/

Or set the HTNL_DATA_ROOT environment variable and omit --data.

Output
------
results/
  summary.json   — per-dataset / per-method mean AUC over seeds
  summary.csv    — same in CSV format
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))

from htnl.data.real_data_loader import load_country, load_flu, COUNTRIES
from htnl.models.htnl_real_fast import run_method_fast

ALL_DATASETS = list(COUNTRIES) + ['flu']

METHODS = ['Original', 'VQF', 'ADMM-Bisect', 'WGL-Cont']
# WGL-Cont(lam=1): pass lam_init=1.0 to avoid scale-collapse on imbalanced data
METHOD_KWARGS = {
    'WGL-Cont': {'lam_init': 1.0, 'lam_target': 1.0},
}


def load_dataset(name: str, data_root=None) -> dict:
    kw = {'root': data_root} if data_root else {}
    if name == 'flu':
        return load_flu(**kw)
    return load_country(name, **kw)


def run(datasets, seeds, out_dir, data_root=None):
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for ds in datasets:
        print(f"\n=== Dataset: {ds} ===")
        data = load_dataset(ds, data_root=data_root)
        X_tr, y_tr = data['X_tr'], data['Y_tr']
        X_te, y_te = data['X_te'], data['Y_te']
        V_total = X_tr.shape[1]
        n_pos = int((y_tr == 1).sum())
        print(f"  n_tr={len(y_tr)}  n_te={len(y_te)}  V={V_total}  "
              f"pos_frac={n_pos/len(y_tr)*100:.2f}%")

        results[ds] = {}
        for mname in METHODS:
            extra = METHOD_KWARGS.get(mname, {})
            aucs, sels = [], []
            for seed in seeds:
                t0 = time.time()
                rec = run_method_fast(
                    mname, X_tr, y_tr, X_te, y_te,
                    V_total=V_total, seed=seed, **extra
                )
                elapsed = time.time() - t0
                aucs.append(rec['auc'])
                sels.append(rec.get('sel_size_rel040', 0))
                print(f"  {mname:20s} seed={seed}  AUC={rec['auc']:.4f}  "
                      f"sel={rec.get('sel_size_rel040','?')}  t={elapsed:.1f}s")
            results[ds][mname] = {
                'auc_mean':       float(np.mean(aucs)),
                'auc_std':        float(np.std(aucs)),
                'auc_per_seed':   [float(a) for a in aucs],
                'sel_mean':       float(np.mean(sels)),
            }

    # Save JSON
    out_json = os.path.join(out_dir, 'summary.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Save CSV
    out_csv = os.path.join(out_dir, 'summary.csv')
    with open(out_csv, 'w') as f:
        f.write('dataset,method,auc_mean,auc_std,sel_mean\n')
        for ds, ms in results.items():
            for mname, v in ms.items():
                f.write(f"{ds},{mname},{v['auc_mean']:.4f},"
                        f"{v['auc_std']:.4f},{v['sel_mean']:.1f}\n")
    print(f"Saved: {out_csv}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run HTNL on real-world datasets')
    parser.add_argument('--data', default=None,
                        help='Path to hkl_combined/ directory '
                             '(default: HTNL_DATA_ROOT env or ./data/hkl_combined)')
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS,
                        help=f'Datasets to run. Available: {ALL_DATASETS}')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Random seeds (default: 0 1 2)')
    parser.add_argument('--out', default='results',
                        help='Output directory (default: results/)')
    args = parser.parse_args()

    if args.data:
        os.environ['HTNL_DATA_ROOT'] = args.data

    run(args.datasets, args.seeds, args.out, data_root=args.data)
