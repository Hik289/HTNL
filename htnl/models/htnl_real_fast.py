"""Optimised real-data adapter for v1-v3 HTNL solvers.

Key speed-ups vs htnl_real_adapter.py:
- Precompute Phi as a single (n × |U|) numpy array (NOT a Python list of bags).
- Phi for length-k NCFs is a vectorised product of k columns of X.
- Re-implement w_update_weighted_ridge inline so we avoid the per-task
  Python loop wrapper for S=1.

The algorithm logic is the same as htnl_original.py / htnl_vqf.py /
htnl_admm_bisection.py / htnl_wgl_continuation.py, just adapted for the
T=1 single-task case with ALL data in one (n × V) matrix.
"""
import os
import sys
import time
from typing import Dict, List, Tuple, Set, Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    average_precision_score, precision_score, recall_score,
)


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


# ---------------- core helpers ---------------------------------------

def huber_hinge_loss(margin, delta=0.5):
    loss = np.where(
        margin <= -delta, 0.0,
        np.where(margin >= delta, margin, (margin + delta) ** 2 / (4.0 * delta)),
    )
    deriv = np.where(
        margin <= -delta, 0.0,
        np.where(margin >= delta, 1.0, (margin + delta) / (2.0 * delta)),
    )
    return loss, deriv


def phi_X(X: np.ndarray, U: List[frozenset]) -> np.ndarray:
    """Compute (n × |U|) Phi matrix from (n × V) binary X."""
    n = X.shape[0]
    if not U:
        return np.zeros((n, 0), dtype=np.float32)
    out = np.zeros((n, len(U)), dtype=np.float32)
    for j, u in enumerate(U):
        feats = list(u)
        if len(feats) == 1:
            out[:, j] = X[:, feats[0]]
        else:
            col = X[:, feats[0]].copy()
            for f in feats[1:]:
                col = col * X[:, f]
            out[:, j] = col
    return out


def compute_class_weights(y: np.ndarray, scheme: str = 'balanced') -> np.ndarray:
    """Per-sample weight vector implementing class-weighted hinge.

    'balanced' = MATLAB ghkl wpos/wneg formula (when details.balance=1):
        w_pos = n_neg / n,  w_neg = n_pos / n
        Per-sample weight is C_i = w_pos if y_i=+1 else w_neg
        Effective C for positives ~ 1 - pos_frac (large)
        Effective C for negatives ~ pos_frac (small) — down-weights negatives
    'none' / None = uniform (all 1)
    'sklearn' = sklearn-style balanced: w_pos = n / (2 * n_pos), w_neg = n / (2 * n_neg)
    """
    if scheme is None or scheme == 'none' or scheme == 'uniform':
        return np.ones(len(y), dtype=np.float64)
    n = len(y)
    n_pos = float((y == 1).sum())
    n_neg = float((y == -1).sum())
    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y), dtype=np.float64)
    sw = np.ones(n, dtype=np.float64)
    if scheme == 'balanced':
        # MATLAB ghkl formula
        w_pos = n_neg / n
        w_neg = n_pos / n
    elif scheme == 'sklearn':
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
    else:
        raise ValueError(f'Unknown class_weight scheme: {scheme}')
    sw[y == 1] = w_pos
    sw[y == -1] = w_neg
    return sw


def w_update_l2reg(Phi: np.ndarray, y: np.ndarray, c_arr: np.ndarray,
                   C: float, w_init: np.ndarray = None,
                   extra_linear: np.ndarray = None,
                   delta: float = 0.5, max_iter: int = 60,
                   sample_weight: np.ndarray = None) -> np.ndarray:
    """Solve  min sum_i C * sw_i * huber(1 - y_i * Phi_i w) + sum_j c[j] w_j^2 - <b, w>.

    sample_weight (n,) defaults to all 1 (no class weighting).
    """
    n_u = Phi.shape[1]
    w0 = np.zeros(n_u) if w_init is None else w_init.copy()
    b = extra_linear if extra_linear is not None else np.zeros(n_u)
    sw = sample_weight if sample_weight is not None else None

    def obj_grad(w):
        margin = 1.0 - y * (Phi @ w)
        loss, deriv = huber_hinge_loss(margin, delta)
        if sw is not None:
            loss = loss * sw
            deriv = deriv * sw
        obj = C * loss.sum() + float((c_arr * w * w).sum()) - float(b @ w)
        grad = -C * Phi.T @ (y * deriv) + 2.0 * c_arr * w - b
        return obj, grad

    res = minimize(obj_grad, w0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'gtol': 1e-6})
    return res.x


def compute_r_u_fast(w: np.ndarray) -> np.ndarray:
    """For S=1, r_u = |w_u|. Returns (|U|,)."""
    return np.abs(w) + 1e-10


def compute_omega_fast(w: np.ndarray, U: List[frozenset], p: float = 1.5,
                       eps: float = 1e-10) -> float:
    """Omega(w) = sum_v d_v · (sum_{u in D(v)} r_u^p)^{1/p}.
    For S=1, r_u = |w_u|. Vectorised.
    """
    n_u = len(U)
    r = np.abs(w) + eps
    rp = r ** p
    # For each v in U, compute f_v = (sum_{u containing v} r_u^p)^(1/p)
    # We'll precompute the descendants relation.
    omega = 0.0
    sets = [set(u) for u in U]
    for i, v_set in enumerate(sets):
        f_v = 0.0
        for j, u_set in enumerate(sets):
            if v_set.issubset(u_set):
                f_v += rp[j]
        f_v = (f_v + eps) ** (1.0 / p)
        omega += (2.0 ** len(v_set)) * f_v
    return float(omega)


def hinge_total_obj_fast(w: np.ndarray, Phi: np.ndarray, y: np.ndarray,
                         C: float, delta: float = 0.5,
                         sample_weight: np.ndarray = None) -> float:
    margin = 1.0 - y * (Phi @ w)
    loss, _ = huber_hinge_loss(margin, delta)
    if sample_weight is not None:
        loss = loss * sample_weight
    return float(C * loss.sum())


# ---------------- KKT score for active-set expansion -----------------

def active_set_expand_fast(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                            U: List[frozenset], V_total: int, C: float,
                            threshold: float = 0.01, max_add: int = 5,
                            max_len: int = 2,
                            delta: float = 0.5,
                            sample_weight: np.ndarray = None) -> List[frozenset]:
    """Return list of candidate NCFs (length 2) sorted by KKT score, top max_add
    above `threshold`. Vectorised version. Honours sample_weight.
    """
    Phi = phi_X(X, U)
    margin = 1.0 - y * (Phi @ w)
    _, deriv = huber_hinge_loss(margin, delta)
    if sample_weight is not None:
        deriv = deriv * sample_weight
    g = y * deriv  # (n,)

    set_act = set(U)
    from itertools import combinations
    pairs = []
    for v in combinations(range(V_total), 2):
        u = frozenset(v)
        if u in set_act:
            continue
        pairs.append((v[0], v[1], u))

    if not pairs:
        return []

    Xf = X.astype(np.float32)
    gf = (g * C).astype(np.float32)
    weighted = Xf * gf[:, None]
    score_matrix = Xf.T @ weighted
    scored = []
    for a, b, u in pairs:
        score = abs(float(score_matrix[a, b])) / 4.0
        if score > threshold:
            scored.append((score, u))
    scored.sort(reverse=True)
    return [u for _, u in scored[:max_add]]


# ---------------- HTNL-Original (single-task, fast) ------------------

def _hoelder_coeffs_fast(w: np.ndarray, U: List[frozenset], p: float = 1.5,
                         eps: float = 1e-10) -> np.ndarray:
    """Returns (|U|,) coefficient vector c (sent to solver)."""
    n_u = len(U)
    r = np.abs(w) + eps
    sets = [set(u) for u in U]
    # f_v
    rp = r ** p
    f_v = np.zeros(n_u)
    is_anc = np.zeros((n_u, n_u), dtype=bool)
    for i, v_set in enumerate(sets):
        for j, u_set in enumerate(sets):
            if v_set.issubset(u_set):
                is_anc[i, j] = True
    for i in range(n_u):
        f_v[i] = (rp[is_anc[i]].sum() + eps) ** (1.0 / p)
    Omega = float(np.sum(np.array([2.0 ** len(s) for s in sets]) * f_v))

    # ancestor_sum_u
    d_v = np.array([2.0 ** len(s) for s in sets])
    f_v_pow = f_v ** (1.0 - p)
    anc_sum = np.zeros(n_u)
    for j in range(n_u):
        anc_sum[j] = float((d_v[is_anc[:, j]] * f_v_pow[is_anc[:, j]]).sum())

    # c_u = (Omega · r_u^(p-1) · anc_sum) / s_u, where s_u = r_u (S=1)
    c = (Omega * (r ** (p - 1)) * anc_sum) / r
    return c, Omega


def solve_original_fast(X: np.ndarray, y: np.ndarray, V_total: int,
                         C: float = 1.0, p: float = 1.5,
                         max_outer: int = 2, max_inner: int = 12, tol: float = 1e-3,
                         kkt_threshold: float = 0.01, max_add: int = 5,
                         max_active: int = 250,
                         class_weight: str = None):
    """Original-MM (Hölder closed-form) on single-task (X, y).

    class_weight: None / 'balanced' / 'sklearn' — see compute_class_weights.
    """
    history = {'obj': [], 'rounds': [], 'active_size': []}
    sw = compute_class_weights(y, class_weight) if class_weight else None
    active_set = [frozenset([i]) for i in range(V_total)]

    w_final = None
    U_final = None

    for outer in range(max_outer):
        U = list(active_set)
        Phi = phi_X(X, U)
        n_u = len(U)
        w = np.zeros(n_u)
        c = np.ones(n_u)

        prev_obj = np.inf
        for it in range(max_inner):
            w = w_update_l2reg(Phi, y, c * 0.5, C, w_init=w, max_iter=60,
                                sample_weight=sw)
            c, Omega = _hoelder_coeffs_fast(w, U, p)
            obj = hinge_total_obj_fast(w, Phi, y, C, sample_weight=sw) + 0.5 * Omega * Omega
            history['obj'].append(obj)
            history['rounds'].append(outer)
            history['active_size'].append(n_u)
            if abs(prev_obj - obj) < tol * (abs(prev_obj) + 1e-8):
                break
            prev_obj = obj

        # Always update w_final / U_final at end of each outer round.
        w_final = w
        U_final = U
        if outer < max_outer - 1 and n_u < max_active:
            new_v = active_set_expand_fast(X, y, w, U, V_total, C,
                                            threshold=kkt_threshold,
                                            max_add=max_add,
                                            sample_weight=sw)
            if not new_v:
                break
            active_set = U + new_v

    return w_final, U_final, history


# ---------------- VQF (mathematically equivalent to Original) --------

def solve_vqf_fast(X: np.ndarray, y: np.ndarray, V_total: int, **kw):
    """VQF is bit-exact equivalent to Original (proven in v1). Just call it."""
    return solve_original_fast(X, y, V_total, **kw)


# ---------------- ADMM-Bisection (single-task, fast) -----------------

def _solve_r_u_for_fixed_eta_1d(q_u, eta, p, max_bisect=60, tol=1e-12):
    """Same as v2's _solve_r_u_for_fixed_eta. q_u is per-group magnitudes,
    but for S=1 we only have ONE group, so q_u is a scalar.
    r = max(0, q - eta · r^(p-1))  →  for S=1 just direct shrinkage."""
    q = float(q_u)
    if q <= 0:
        return 0.0
    # r = max(0, q - eta · r^(p-1)). Single block per u → r = q - eta · r^(p-1)
    # Bisect r ∈ [0, q]
    lo, hi = 0.0, q
    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        f_mid = mid - max(q - eta * mid ** (p - 1), 0.0)
        if f_mid >= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * max(1.0, q):
            break
    return 0.5 * (lo + hi)


def _exact_prox_fv_1g(q_per_u, descend_idx, d_v, rho, p):
    """Exact prox for single-group (S=1) per-ancestor v.

    q_per_u: dict j -> scalar magnitude (signed; we work with abs).
    Returns z_per_u: dict j -> signed scalar.
    """
    if not q_per_u:
        return {}
    # Use abs values for the prox; restore sign at the end.
    q_abs = {j: abs(float(q_per_u[j])) for j in descend_idx}
    q_sign = {j: float(np.sign(q_per_u[j])) for j in descend_idx}
    f_v_max = (sum(q_abs[j] ** p for j in descend_idx)) ** (1.0 / p)
    if f_v_max <= 0:
        return {j: 0.0 for j in descend_idx}

    fv = f_v_max
    r_u = {j: 0.0 for j in descend_idx}
    for _ in range(60):
        eta = (d_v / rho) / (fv ** (p - 1)) if fv > 0 else float('inf')
        r_u = {j: _solve_r_u_for_fixed_eta_1d(q_abs[j], eta, p)
               for j in descend_idx}
        fv_new = (sum(r_u[j] ** p for j in descend_idx)) ** (1.0 / p)
        if abs(fv_new - fv) < 1e-12 * max(1.0, f_v_max):
            fv = fv_new
            break
        fv = fv_new

    eta = (d_v / rho) / (fv ** (p - 1)) if fv > 0 else float('inf')
    z_per_u = {}
    for j in descend_idx:
        if r_u[j] <= 0:
            z_per_u[j] = 0.0
            continue
        theta = eta * r_u[j] ** (p - 1)
        z_per_u[j] = max(0.0, q_abs[j] - theta) * q_sign[j]
    return z_per_u


def solve_admm_bisect_fast(X: np.ndarray, y: np.ndarray, V_total: int,
                            C: float = 1.0, p: float = 1.5, rho: float = 1.0,
                            max_outer: int = 2, max_iter: int = 30,
                            tol: float = 1e-3, kkt_threshold: float = 0.01,
                            max_add: int = 5, max_active: int = 250,
                            lam: float = 1.0,
                            class_weight: str = None):
    """ADMM with exact-bisection prox on single-task (X, y).

    For S=1 (single group), the per-ancestor prox simplifies massively:
    each (u, g) reduces to a scalar (z, q) and the bisection runs once
    per descendant.
    """
    history = {'obj': [], 'primal_res': [], 'dual_res': [],
               'rounds': [], 'active_size': []}
    sw = compute_class_weights(y, class_weight) if class_weight else None

    active_set = [frozenset([i]) for i in range(V_total)]
    w_final = None
    U_final = None

    for outer in range(max_outer):
        U = list(active_set)
        Phi = phi_X(X, U)
        n_u = len(U)
        sets = [set(u) for u in U]

        is_anc = np.zeros((n_u, n_u), dtype=bool)
        for i, v_set in enumerate(sets):
            for j, u_set in enumerate(sets):
                if v_set.issubset(u_set):
                    is_anc[i, j] = True
        ancestors_of = {j: [i for i in range(n_u) if is_anc[i, j]] for j in range(n_u)}
        descendants_of = {i: [j for j in range(n_u) if is_anc[i, j]] for i in range(n_u)}

        w = np.zeros(n_u)
        Z = {}
        Dual = {}
        for i, v_set in enumerate(sets):
            for j in descendants_of[i]:
                Z[(i, j)] = 0.0
                Dual[(i, j)] = 0.0

        prev_obj = np.inf
        for it in range(max_iter):
            c_arr = np.zeros(n_u)
            extra_lin = np.zeros(n_u)
            for j in range(n_u):
                anc = ancestors_of[j]
                c_arr[j] = 0.5 * rho * len(anc)
                M = 0.0
                for i in anc:
                    M += Z[(i, j)] - Dual[(i, j)]
                extra_lin[j] = rho * M

            w = w_update_l2reg(Phi, y, c_arr, C, w_init=w,
                                extra_linear=extra_lin, max_iter=40,
                                sample_weight=sw)

            # Z-step (exact prox per ancestor v)
            Z_old = {k: v for k, v in Z.items()}
            primal_sq = 0.0
            for i, v_set in enumerate(sets):
                d_v = lam * (2.0 ** len(v_set))
                desc = descendants_of[i]
                if not desc:
                    continue
                Q_v = {j: float(w[j] + Dual[(i, j)]) for j in desc}
                Z_new = _exact_prox_fv_1g(Q_v, desc, d_v, rho, p)
                for j in desc:
                    Z[(i, j)] = Z_new.get(j, 0.0)
                    diff = w[j] - Z[(i, j)]
                    primal_sq += diff * diff

            # Dual step
            for (i, j), zold in Z_old.items():
                Dual[(i, j)] += w[j] - Z[(i, j)]

            primal_res = float(np.sqrt(primal_sq))
            dual_sq = sum((Z[k] - Z_old[k]) ** 2 for k in Z_old)
            dual_res = rho * float(np.sqrt(dual_sq))

            obj = hinge_total_obj_fast(w, Phi, y, C, sample_weight=sw) + lam * compute_omega_fast(w, U, p)
            history['obj'].append(obj)
            history['primal_res'].append(primal_res)
            history['dual_res'].append(dual_res)
            history['rounds'].append(outer)
            history['active_size'].append(n_u)
            if primal_res < tol and dual_res < tol:
                break
            if abs(prev_obj - obj) < tol * (abs(prev_obj) + 1e-8):
                break
            prev_obj = obj

        # Always update w_final / U_final at end of each outer round.
        w_final = w
        U_final = U
        if outer < max_outer - 1 and n_u < max_active:
            new_v = active_set_expand_fast(X, y, w, U, V_total, C,
                                            threshold=kkt_threshold,
                                            max_add=max_add,
                                            sample_weight=sw)
            if not new_v:
                break
            active_set = U + new_v

    return w_final, U_final, history


# ---------------- WGL-Cont (single-task, fast) ----------------------

def _fista_l1_block_1g(Phi, y, c_arr, C, w_init, max_iter=200, tol=1e-5,
                       delta=0.5, sample_weight=None):
    """FISTA on  sum_i C·sw_i·huber + sum_j c[j] |w_j|. S=1, scalar blocks.
    """
    n, n_u = Phi.shape
    w = w_init.copy() if w_init is not None else np.zeros(n_u)
    sw = sample_weight if sample_weight is not None else None
    # Lipschitz upper bound — when class-weighted, sw_i can amplify by max(sw)
    sw_max = float(sw.max()) if sw is not None else 1.0
    L = max(sw_max * C * float(np.linalg.norm(Phi, 'fro') ** 2) / (2.0 * delta), 1e-3)

    z = w.copy()
    w_prev = w.copy()
    t = 1.0

    for it in range(max_iter):
        margin = 1.0 - y * (Phi @ z)
        _, deriv = huber_hinge_loss(margin, delta)
        if sw is not None:
            deriv = deriv * sw
        grad = -C * Phi.T @ (y * deriv)
        w_grad = z - grad / L
        thresh = c_arr / L
        w_new = np.sign(w_grad) * np.maximum(np.abs(w_grad) - thresh, 0.0)

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = w_new + ((t - 1.0) / t_new) * (w_new - w_prev)
        if np.linalg.norm(w_new - w_prev) < tol * (np.linalg.norm(w_new) + 1e-8):
            w = w_new
            break
        w_prev = w_new
        t = t_new
        w = w_new
    return w


def solve_wgl_cont_fast(X: np.ndarray, y: np.ndarray, V_total: int,
                         C: float = 1.0, p: float = 1.5,
                         lambda_target: float = 50.0, lambda_start: float = 500.0,
                         n_lambda_steps: int = 4, max_outer: int = 2,
                         max_iter_per_lam: int = 6, fista_iters: int = 200,
                         tol: float = 1e-3, kkt_threshold: float = 0.01,
                         max_add: int = 5, max_active: int = 250,
                         direction: str = 'high_to_low', eps: float = 1e-8,
                         class_weight: str = None):
    """WGL-Cont (H2L) on single-task (X, y)."""
    history = {'obj': [], 'lam': [], 'rounds': [], 'active_size': []}
    sw = compute_class_weights(y, class_weight) if class_weight else None

    if n_lambda_steps > 1:
        ratios = np.geomspace(1.0, lambda_target / lambda_start, n_lambda_steps)
        schedule = lambda_start * ratios
    else:
        schedule = np.array([lambda_target])
    if direction == 'low_to_high':
        schedule = schedule[::-1]

    active_set = [frozenset([i]) for i in range(V_total)]
    w_final = None
    U_final = None

    for outer in range(max_outer):
        U = list(active_set)
        Phi = phi_X(X, U)
        n_u = len(U)
        sets = [set(u) for u in U]

        anc_of = {j: [i for i in range(n_u) if sets[i].issubset(sets[j])] for j in range(n_u)}
        desc_of = {i: [j for j in range(n_u) if sets[i].issubset(sets[j])] for i in range(n_u)}

        w = np.zeros(n_u)
        alpha = {(i, j): 1.0 / max(len(desc_of[i]), 1)
                 for i in range(n_u) for j in desc_of[i]}
        d_v_arr = np.array([2.0 ** len(s) for s in sets])

        # Warm-start at lam_target/10 if H2L
        if direction == 'high_to_low':
            warm_lam = max(lambda_target / 10.0, 1.0)
            c_warm = np.zeros(n_u)
            for j in range(n_u):
                c_warm[j] = warm_lam * sum(d_v_arr[i] * alpha[(i, j)]
                                            for i in anc_of[j])
            w = _fista_l1_block_1g(Phi, y, c_warm, C, w, max_iter=fista_iters,
                                   sample_weight=sw)
            r_u = np.abs(w) + eps
            for i in range(n_u):
                desc = desc_of[i]
                base = sum((r_u[j]) ** p for j in desc)
                denom = base ** ((p - 1.0) / p) + eps
                for j in desc:
                    alpha[(i, j)] = (r_u[j]) ** (p - 1.0) / denom

        prev_obj = np.inf
        for lam_idx, lam_now in enumerate(schedule):
            for it in range(max_iter_per_lam):
                c_arr = np.zeros(n_u)
                for j in range(n_u):
                    c_arr[j] = lam_now * sum(d_v_arr[i] * alpha[(i, j)]
                                              for i in anc_of[j])
                w = _fista_l1_block_1g(Phi, y, c_arr, C, w,
                                       max_iter=fista_iters, sample_weight=sw)
                r_u = np.abs(w) + eps
                for i in range(n_u):
                    desc = desc_of[i]
                    base = sum((r_u[j]) ** p for j in desc)
                    denom = base ** ((p - 1.0) / p) + eps
                    for j in desc:
                        alpha[(i, j)] = (r_u[j]) ** (p - 1.0) / denom
                obj = (hinge_total_obj_fast(w, Phi, y, C, sample_weight=sw)
                       + lam_now * compute_omega_fast(w, U, p))
                history['obj'].append(obj)
                history['lam'].append(lam_now)
                history['rounds'].append(outer)
                history['active_size'].append(n_u)
                if abs(prev_obj - obj) < tol * (abs(prev_obj) + 1e-8):
                    break
                prev_obj = obj

        # Always update w_final / U_final at end of each outer round.
        w_final = w
        U_final = U
        if outer < max_outer - 1 and n_u < max_active:
            new_v = active_set_expand_fast(X, y, w, U, V_total, C,
                                            threshold=kkt_threshold,
                                            max_add=max_add,
                                            sample_weight=sw)
            if not new_v:
                break
            active_set = U + new_v

    return w_final, U_final, history


# ---------------- runner / metrics ----------------------------------

def predict_scores_fast(X_te: np.ndarray, w: np.ndarray, U: List[frozenset]):
    Phi_te = phi_X(X_te, U)
    return Phi_te @ w


def selected_set_fast(w: np.ndarray, U: List[frozenset],
                       rel_threshold: float = 0.20,
                       abs_floor: float = 1e-3) -> Set[frozenset]:
    if w is None or len(w) == 0:
        return set()
    abs_w = np.abs(w)
    max_w = abs_w.max()
    if max_w == 0:
        return set()
    cutoff = max(rel_threshold * max_w, abs_floor)
    return {U[i] for i in range(len(U)) if abs_w[i] > cutoff}


def evaluate_method(name: str, w: np.ndarray, U: List[frozenset],
                     X_tr: np.ndarray, y_tr: np.ndarray,
                     X_te: np.ndarray, y_te: np.ndarray,
                     wall_clock_s: float) -> Dict[str, Any]:
    sel20 = selected_set_fast(w, U, rel_threshold=0.20)
    sel40 = selected_set_fast(w, U, rel_threshold=0.40)

    scores_tr = predict_scores_fast(X_tr, w, U)
    scores_te = predict_scores_fast(X_te, w, U)
    pred_tr = np.where(scores_tr >= 0, 1, -1)
    pred_te = np.where(scores_te >= 0, 1, -1)

    train_acc = accuracy_score(y_tr, pred_tr)
    test_acc = accuracy_score(y_te, pred_te)
    train_f1 = f1_score(y_tr, pred_tr, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_te, pred_te, pos_label=1, zero_division=0)
    test_p = precision_score(y_te, pred_te, pos_label=1, zero_division=0)
    test_r = recall_score(y_te, pred_te, pos_label=1, zero_division=0)
    try:
        test_auc = roc_auc_score(y_te, scores_te)
    except ValueError:
        test_auc = float('nan')
    try:
        test_pr_auc = average_precision_score((y_te == 1).astype(int), scores_te)
    except ValueError:
        test_pr_auc = float('nan')

    return {
        'method': name,
        'n_train': len(y_tr), 'n_test': len(y_te),
        'pos_frac_train': float((y_tr == 1).mean()),
        'pos_frac_test': float((y_te == 1).mean()),
        'wall_clock_s': float(wall_clock_s),
        'train_accuracy': float(train_acc), 'test_accuracy': float(test_acc),
        'train_f1': float(train_f1), 'test_f1': float(test_f1),
        'test_precision': float(test_p), 'test_recall': float(test_r),
        'test_auc': float(test_auc), 'test_pr_auc': float(test_pr_auc),
        'sel_size_rel020': len(sel20), 'sel_size_rel040': len(sel40),
        'selected_ncfs_rel020': sorted([sorted(u) for u in sel20]),
        'selected_ncfs_rel040': sorted([sorted(u) for u in sel40]),
    }


def run_method_fast(name: str, X_tr: np.ndarray, y_tr: np.ndarray,
                     X_te: np.ndarray, y_te: np.ndarray,
                     V_total: int = 100, **kwargs) -> Dict[str, Any]:
    t0 = time.time()
    if name == 'Original':
        w, U, hist = solve_original_fast(X_tr, y_tr, V_total, **kwargs)
    elif name == 'VQF':
        w, U, hist = solve_vqf_fast(X_tr, y_tr, V_total, **kwargs)
    elif name == 'ADMM-Bisect':
        w, U, hist = solve_admm_bisect_fast(X_tr, y_tr, V_total, **kwargs)
    elif name == 'WGL-Cont':
        w, U, hist = solve_wgl_cont_fast(X_tr, y_tr, V_total, **kwargs)
    else:
        raise ValueError(f'Unknown method: {name}')
    elapsed = time.time() - t0
    rec = evaluate_method(name, w, U, X_tr, y_tr, X_te, y_te, elapsed)
    rec['n_iters'] = len(hist.get('obj', []))
    rec['final_obj'] = float(hist['obj'][-1]) if hist.get('obj') else float('nan')
    return rec
