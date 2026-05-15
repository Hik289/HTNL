"""Run all 4 methods on synthetic seeds, print P/R/F1 table, save plots."""
import os
import sys
import time
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.synthetic import generate_synthetic
from models.ncf import ConjunctionLattice
from models.htnl_original import solve_htnl_original
from models.htnl_vqf import solve_vqf
from models.htnl_admm import solve_admm
from models.htnl_wgl import solve_wgl

METHODS = {
    'HTNL-Original': solve_htnl_original,
    'HTNL-VQF':      solve_vqf,
    'HTNL-ADMM':     solve_admm,
    'HTNL-WGL':      solve_wgl,
}


def evaluate_ncf(selected_U, true_U):
    inter = selected_U & true_U
    tp = len(inter)
    p = tp / max(len(selected_U), 1)
    r = tp / max(len(true_U), 1)
    f1 = (2 * p * r / max(p + r, 1e-9)) if (p + r) > 0 else 0.0
    return p, r, f1


def selected_set(W_dict, groups=None, threshold=None, rel_threshold=0.05):
    """Select NCFs based on the per-NCF group-norm sum r_u, the quantity the
    bi-space regularizer acts on. An NCF u is 'selected' iff r_u exceeds
    rel_threshold * max_u r_u (or an absolute floor `threshold`).
    """
    # collect NCFs and per-task weights
    all_us = sorted({u for wd in W_dict.values() for u in wd},
                    key=lambda u: (len(u), sorted(u)))
    if groups is None:
        # fall back to absolute threshold across all weights
        sel = set()
        thr = threshold if threshold is not None else 1e-3
        for s, wd in W_dict.items():
            for u, w in wd.items():
                if abs(w) > thr:
                    sel.add(u)
        return sel
    r_u = {}
    for u in all_us:
        r = 0.0
        for g in groups:
            r += float(np.sqrt(sum(W_dict[s].get(u, 0.0) ** 2 for s in g)))
        r_u[u] = r
    if not r_u:
        return set()
    max_r = max(r_u.values())
    abs_floor = 1e-3 if threshold is None else threshold
    cutoff = max(rel_threshold * max_r, abs_floor)
    return {u for u, r in r_u.items() if r > cutoff}


def run(n_seeds=5, V=10, S=12, T=100, C=1.0, p=1.5, save_dir=None, verbose=False):
    save_dir = save_dir or os.path.join(ROOT, 'results')
    os.makedirs(save_dir, exist_ok=True)
    results = {m: {'P': [], 'R': [], 'F1': [], 'time': [], 'obj_hist': [], 'sel_size': []}
               for m in METHODS}

    for seed in range(n_seeds):
        X, Y, W_true, U_true, groups = generate_synthetic(V=V, S=S, T=T, seed=seed)
        lattice = ConjunctionLattice(range(V), max_order=3)
        if verbose:
            print(f"\n=== seed {seed} | |U*|={len(U_true)} ===")
            print(f"   U_true = {sorted(map(sorted, U_true))}")
            pos_frac = np.mean([np.mean(Y[s] == 1) for s in range(S)])
            print(f"   label balance (avg fraction +1): {pos_frac:.2f}")

        for name, solver in METHODS.items():
            t0 = time.time()
            try:
                W_dict, history = solver(X, Y, groups, lattice, C=C, p=p)
            except Exception as exc:
                print(f"!! {name} on seed {seed} failed: {exc}")
                raise
            elapsed = time.time() - t0
            sel = selected_set(W_dict, groups=groups, rel_threshold=0.20)
            P, R, F = evaluate_ncf(sel, U_true)
            results[name]['P'].append(P)
            results[name]['R'].append(R)
            results[name]['F1'].append(F)
            results[name]['time'].append(elapsed)
            results[name]['obj_hist'].append(history.get('obj', []))
            results[name]['sel_size'].append(len(sel))
            if verbose:
                print(f"   {name:<14} P={P:.3f} R={R:.3f} F1={F:.3f} "
                      f"|sel|={len(sel)} t={elapsed:.1f}s")

    # ---- Print summary table ----
    print(f"\n{'Method':<16} {'Precision':>14} {'Recall':>14} {'F1':>14} "
          f"{'|U_sel|':>10} {'Time(s)':>10}")
    print('-' * 86)
    for m, res in results.items():
        print(f"{m:<16} "
              f"{np.mean(res['P']):.3f}±{np.std(res['P']):.3f}  "
              f"{np.mean(res['R']):.3f}±{np.std(res['R']):.3f}  "
              f"{np.mean(res['F1']):.3f}±{np.std(res['F1']):.3f}  "
              f"{np.mean(res['sel_size']):>9.1f} "
              f"{np.mean(res['time']):>9.1f}")

    # ---- Save convergence plot (first seed) ----
    # Each method has its own scale (squared vs linear Omega form).
    # Plot running-min objective (best so far) to remove transient spikes
    # caused by active-set expansion / alternating updates.
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    styles = ['-', '--', '-', ':']
    for (m, res), color, style in zip(results.items(), colors, styles):
        if not (res['obj_hist'] and len(res['obj_hist'][0])):
            continue
        hist = np.asarray(res['obj_hist'][0], dtype=float)
        hist = np.maximum(hist, 1e-6)
        running_min = np.minimum.accumulate(hist)
        ax.semilogy(running_min, label=m, color=color, linewidth=2.0,
                    linestyle=style)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective value (log scale)')
    ax.set_title('Convergence (best-so-far): HTNL Original vs. Three New Optimizers\n'
                 '(seed 0; each method tracks its own objective form)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(save_dir, 'convergence.png')
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_png}")

    out_json = os.path.join(save_dir, 'ncf_selection.json')
    summary = {}
    for m, res in results.items():
        summary[m] = {
            'P_mean': float(np.mean(res['P'])), 'P_std': float(np.std(res['P'])),
            'R_mean': float(np.mean(res['R'])), 'R_std': float(np.std(res['R'])),
            'F1_mean': float(np.mean(res['F1'])), 'F1_std': float(np.std(res['F1'])),
            'sel_size_mean': float(np.mean(res['sel_size'])),
            'time_mean': float(np.mean(res['time'])),
        }
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")
    return results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--V', type=int, default=10)
    ap.add_argument('--S', type=int, default=12)
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    run(n_seeds=args.seeds, V=args.V, S=args.S, T=args.T, verbose=args.verbose)
