"""Synthetic multi-task data for the HTNL experiment driver."""
from __future__ import annotations

import numpy as np

from htnl.models.ncf import ConjunctionLattice

SyntheticData = tuple[
    dict[int, list[np.ndarray]],
    dict[int, np.ndarray],
    dict[int, dict[frozenset[int], float]],
    set[frozenset[int]],
    list[list[int]],
]


def _balanced_groups(S: int, n_groups: int = 3) -> list[list[int]]:
    n_groups = max(1, min(n_groups, S))
    splits = np.array_split(np.arange(S), n_groups)
    return [split.astype(int).tolist() for split in splits if len(split)]


def _true_ncfs(V: int) -> list[frozenset[int]]:
    singletons = [frozenset([i]) for i in range(min(V, 3))]
    pairs = []
    if V >= 2:
        pairs.append(frozenset([0, 1]))
    if V >= 4:
        pairs.append(frozenset([2, 3]))
    return singletons + pairs


def generate_synthetic(
    V: int = 10,
    S: int = 12,
    T: int = 100,
    seed: int = 0,
    n_messages: int = 8,
) -> SyntheticData:
    """Generate binary bag-of-message tasks with sparse conjunctive signals.

    The return shape matches the four HTNL solvers:
      X: dict {task: list of T binary matrices, each (n_messages, V)}
      Y: dict {task: ndarray(T,) with labels in {-1, +1}}
      W_true: per-task weights for the planted NCFs
      U_true: set of planted NCFs
      groups: list of task-index lists
    """
    if V < 1:
        raise ValueError("V must be positive")
    if S < 1:
        raise ValueError("S must be positive")
    if T < 1:
        raise ValueError("T must be positive")

    rng = np.random.default_rng(seed)
    groups = _balanced_groups(S)
    U_true_list = _true_ncfs(V)
    U_true = set(U_true_list)

    W_true: dict[int, dict[frozenset[int], float]] = {}
    for s in range(S):
        sign = 1.0 if s % 2 == 0 else -1.0
        scale = 1.0 + 0.15 * rng.normal()
        W_true[s] = {}
        for j, u in enumerate(U_true_list):
            base = 0.9 if len(u) == 1 else 1.3
            W_true[s][u] = sign * scale * base / (j + 1)

    X: dict[int, list[np.ndarray]] = {}
    Y: dict[int, np.ndarray] = {}
    for s in range(S):
        task_X = []
        labels = np.empty(T, dtype=np.int32)
        bias = -0.15 if s % 2 == 0 else 0.15
        for t in range(T):
            density = rng.uniform(0.18, 0.42)
            bag = rng.binomial(1, density, size=(n_messages, V)).astype(np.int32)
            score = bias
            for u, w in W_true[s].items():
                score += w * ConjunctionLattice.phi(bag, u) / max(n_messages, 1)
            score += rng.normal(0.0, 0.35)
            labels[t] = 1 if score >= 0.0 else -1
            task_X.append(bag)
        X[s] = task_X
        Y[s] = labels

    return X, Y, W_true, U_true, groups
