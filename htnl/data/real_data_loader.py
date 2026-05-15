"""Load the 10-country (and flu) real-data binary classification datasets
from `data/Archive 4/hkl_combined/{country}_data.mat`.

Each .mat file contains:
  X_tr : (n_tr, 100) uint8 binary feature matrix
  X_te : (n_te, 100) uint8 binary
  Y_tr : (n_tr, 1) int16 ∈ {-1, +1}
  Y_te : (n_te, 1) int16 ∈ {-1, +1}

We expose them in two shapes:
  (a) "flat" shape — (X, y) numpy arrays — used by sklearn / our v4 port
  (b) "multi-task" shape — dict {0: X_subset, ...} — wraps a single-task
       problem in our v1-v3 multi-task API so we can re-use the existing
       solvers without modification.
"""
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat


DATA_ROOT = os.environ.get(
    'HTNL_DATA_ROOT',
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'hkl_combined')
)

COUNTRIES = [
    'argentina', 'brazil', 'chile', 'colombia', 'ecuador',
    'el salvador', 'mexico', 'paraguay', 'uruguay', 'venezuela',
]
# Also available: 'flu' (182 features instead of 100). Keep separate.


def list_countries() -> List[str]:
    return list(COUNTRIES)


def load_country(name: str, root: str = DATA_ROOT) -> Dict[str, np.ndarray]:
    """Load a country .mat file and return numpy arrays.

    Returns dict with keys: X_tr, Y_tr, X_te, Y_te. Y reshaped to (n,).
    """
    fn = os.path.join(root, f'{name}_data.mat')
    d = loadmat(fn)
    X_tr = np.asarray(d['X_tr'], dtype=np.float32)
    Y_tr = np.asarray(d['Y_tr'], dtype=np.int8).ravel()
    X_te = np.asarray(d['X_te'], dtype=np.float32)
    Y_te = np.asarray(d['Y_te'], dtype=np.int8).ravel()
    return {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_te': X_te, 'Y_te': Y_te, 'name': name}


def stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int = 0,
                          keep_all_pos: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to size `n` keeping all positive examples.

    Real data is extremely imbalanced (~1% positive). With `keep_all_pos`,
    we keep ALL +1 examples and randomly subsample -1 examples to fill
    the remaining (n - n_pos) slots.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]
    if keep_all_pos:
        n_neg = max(0, n - len(pos_idx))
        if n_neg > len(neg_idx):
            n_neg = len(neg_idx)
        neg_sel = rng.choice(neg_idx, n_neg, replace=False)
        idx = np.concatenate([pos_idx, neg_sel])
    else:
        n_pos = max(1, int(n * len(pos_idx) / len(y)))
        n_neg = n - n_pos
        pos_sel = rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False)
        neg_sel = rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)
        idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    return X[idx], y[idx]


def to_singletask_dict(X: np.ndarray, y: np.ndarray,
                        n_msg_per_point: int = 1):
    """Wrap (X, y) in the v1-v3 multi-task dict-of-bags shape with T=1.

    Our v1-v3 solvers expect:
      X = dict {s: list of T arrays, each (n_msg, V) binary}
      Y = dict {s: ndarray(T,) ∈ {-1, +1}}
      groups = list of lists of task indices
    Here we use S=1 "tasks" with T=n data points, each "data point" is a
    bag of (n_msg_per_point=1) message (the single binary row).
    """
    n, V = X.shape
    X_bags = []
    for i in range(n):
        # one message per "bag", row i
        msg = X[i:i+1].astype(np.int32)  # (1, V)
        X_bags.append(msg)
    X_dict = {0: X_bags}
    Y_dict = {0: y.astype(np.int32)}
    groups = [[0]]   # 1 geographic group containing the only task
    return X_dict, Y_dict, groups, V


def load_flu(root: str = DATA_ROOT) -> Dict[str, np.ndarray]:
    """Load the flu surveillance dataset (182 binary features)."""
    fn = os.path.join(root, 'flu_data.mat')
    d = loadmat(fn)
    X_tr = np.asarray(d['X_tr'], dtype=np.float32)
    Y_tr = np.asarray(d['Y_tr'], dtype=np.int8).ravel()
    X_te = np.asarray(d['X_te'], dtype=np.float32)
    Y_te = np.asarray(d['Y_te'], dtype=np.int8).ravel()
    return {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_te': X_te, 'Y_te': Y_te, 'name': 'flu'}


def summary():
    rows = []
    for c in COUNTRIES:
        try:
            d = load_country(c)
            X_tr, y_tr, X_te, y_te = d['X_tr'], d['Y_tr'], d['X_te'], d['Y_te']
            rows.append({
                'country': c,
                'n_tr': len(y_tr), 'n_te': len(y_te),
                'V': X_tr.shape[1],
                'pos_tr': int((y_tr == 1).sum()),
                'pos_te': int((y_te == 1).sum()),
                'pos_frac_tr': float((y_tr == 1).mean()),
                'density': float(X_tr.mean()),
            })
        except Exception as e:
            rows.append({'country': c, 'error': str(e)})
    return rows


if __name__ == '__main__':
    for r in summary():
        print(r)
