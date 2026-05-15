"""Numerical Conjunctive Feature (NCF) lattice utilities (TKDE Sec. 4)."""
import numpy as np
from itertools import combinations


class ConjunctionLattice:
    def __init__(self, primitives, max_order=3):
        self.primitives = list(primitives)
        self.V = len(self.primitives)
        self.max_order = max_order
        self._all = self._build_all(max_order)

    def _build_all(self, max_order):
        all_ncfs = []
        for k in range(1, max_order + 1):
            for combo in combinations(self.primitives, k):
                all_ncfs.append(frozenset(combo))
        return all_ncfs

    def all_ncf(self, max_order=None):
        if max_order is None or max_order == self.max_order:
            return list(self._all)
        return [v for v in self._all if len(v) <= max_order]

    @staticmethod
    def phi(X, v):
        """phi_v(X) = number of rows where ALL columns in v equal 1.
        X: (n_msg, V) binary matrix; v: frozenset of column indices.
        """
        if len(v) == 0:
            return float(X.shape[0])
        cols = list(v)
        return float(np.prod(X[:, cols], axis=1).sum())

    def compute_all_phi(self, X, ncf_set):
        return {v: self.phi(X, v) for v in ncf_set}

    @staticmethod
    def descendants(v, all_ncfs):
        return [u for u in all_ncfs if v.issubset(u)]

    @staticmethod
    def ancestors(v, all_ncfs):
        return [u for u in all_ncfs if u.issubset(v)]

    @staticmethod
    def kernel_matrix(Phi_u_s, Phi_u_t):
        return np.outer(Phi_u_s, Phi_u_t)


def precompute_phi_matrix(X, lattice, ncf_list):
    """For each task s, build a (T, |U|) matrix of phi values.

    X: dict {s: list of T arrays (n_msg_t, V)}
    Returns: dict {s: ndarray(T, |U|)}
    """
    Phi = {}
    for s, msgs in X.items():
        T = len(msgs)
        P = np.zeros((T, len(ncf_list)))
        for t, m in enumerate(msgs):
            for i, u in enumerate(ncf_list):
                P[t, i] = ConjunctionLattice.phi(m, u)
        Phi[s] = P
    return Phi


def group_of_task(groups, S):
    g_of_s = np.zeros(S, dtype=int)
    for g_idx, g in enumerate(groups):
        for s in g:
            g_of_s[s] = g_idx
    return g_of_s
