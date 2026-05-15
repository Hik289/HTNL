"""Shared core: Huber hinge, per-task weighted-ridge SVM, KKT active-set check."""
import numpy as np
from scipy.optimize import minimize

from .ncf import (
    ConjunctionLattice,
    precompute_phi_matrix,
    group_of_task,
)


# --------------------------- losses ---------------------------------

def huber_hinge_loss(margin, delta=0.5):
    """Smooth approximation of hinge max(0, m).
    Returns (loss, deriv) elementwise.
    """
    loss = np.where(
        margin <= -delta, 0.0,
        np.where(margin >= delta, margin, (margin + delta) ** 2 / (4.0 * delta)),
    )
    deriv = np.where(
        margin <= -delta, 0.0,
        np.where(margin >= delta, 1.0, (margin + delta) / (2.0 * delta)),
    )
    return loss, deriv


def hinge_obj_grad_per_task(w, Phi_s, Y_s, C, delta=0.5):
    """Huber-hinge sum over t with margin = 1 - y * Phi w."""
    margin = 1.0 - Y_s * (Phi_s @ w)
    loss, deriv = huber_hinge_loss(margin, delta)
    obj = C * loss.sum()
    grad = -C * Phi_s.T @ (Y_s * deriv)
    return obj, grad


def hinge_total_obj(W, Phi, Y, C, delta=0.5):
    """Total Huber hinge across all tasks."""
    total = 0.0
    for s, Ps in Phi.items():
        margin = 1.0 - Y[s] * (Ps @ W[s])
        loss, _ = huber_hinge_loss(margin, delta)
        total += loss.sum()
    return C * total


# ------------------------ W subproblem ------------------------------

def w_update_weighted_ridge(Phi, Y, U, c_dict, groups, S, C,
                            W_init=None, delta=0.5, max_iter=80, gtol=1e-6,
                            extra_linear=None):
    """Solve per-task: min C huber_hinge + sum_u c[(u,g(s))] * w_{s,u}^2 - sum_u b[s,u]*w_{s,u}

    c_dict : (u, g_idx) -> non-negative scalar (already includes 1/2 if any).
    extra_linear : optional (S, |U|) array of linear coefficients (added as -W*extra).
    """
    g_of_s = group_of_task(groups, S)
    n_u = len(U)
    W = np.zeros((S, n_u)) if W_init is None else W_init.copy()

    c_arr = np.zeros((len(groups), n_u))
    for (u, g_idx), val in c_dict.items():
        i = U.index(u)
        c_arr[g_idx, i] = val

    for s in range(S):
        c_s = c_arr[g_of_s[s]]
        Ps = Phi[s]
        Ys = Y[s]
        b_s = (extra_linear[s] if extra_linear is not None else None)

        def obj_grad(w):
            o, g = hinge_obj_grad_per_task(w, Ps, Ys, C, delta)
            o += float(np.sum(c_s * w * w))
            g = g + 2.0 * c_s * w
            if b_s is not None:
                o += -float(np.dot(b_s, w))
                g = g - b_s
            return o, g

        res = minimize(obj_grad, W[s], jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol})
        W[s] = res.x
    return W


# ------------------------ Omega evaluation ---------------------------

def compute_r_u(W, U, groups, eps=1e-10):
    """r_u = sum_g ||W_{j in g, u}||_2 for each u in U."""
    n_g = len(groups)
    n_u = len(U)
    s_ug = np.zeros((n_u, n_g))
    for i in range(n_u):
        for g_idx, g in enumerate(groups):
            s_ug[i, g_idx] = np.linalg.norm(W[g, i])
    r_u = s_ug.sum(axis=1) + eps
    return r_u, s_ug


def compute_omega(W, U, groups, p=1.5, ancestors_of=None, eps=1e-10):
    """Omega(W) = sum_v d_v * (sum_{u in D(v)} r_u^p)^{1/p}."""
    r_u, _ = compute_r_u(W, U, groups, eps)
    omega = 0.0
    for v in U:
        d_v = 2.0 ** len(v)
        # u must contain v
        f_v = 0.0
        for i, u in enumerate(U):
            if v.issubset(u):
                f_v += r_u[i] ** p
        f_v = (f_v + eps) ** (1.0 / p)
        omega += d_v * f_v
    return omega


def compute_objective(W, Phi, Y, U, groups, C=1.0, p=1.5, mode='squared'):
    """C * total hinge + (1/2) Omega^2  (mode='squared') or + Omega (mode='linear')."""
    L = hinge_total_obj(W, Phi, Y, C)
    om = compute_omega(W, U, groups, p)
    if mode == 'squared':
        return L + 0.5 * om * om
    return L + om


# ------------------------ Active-set / KKT --------------------------

def kkt_violation_score(W, Phi, Y, U_active, lattice, candidate_v,
                        groups, C, delta=0.5):
    """Score for adding v to active set: max_g ||grad_v on group g|| / d_v.

    Computes phi_v on the fly for each task using lattice.phi.
    """
    grads_per_g = np.zeros(len(groups))
    for g_idx, g in enumerate(groups):
        block = []
        for s in g:
            margin = 1.0 - Y[s] * (Phi[s] @ W[s])
            _, deriv = huber_hinge_loss(margin, delta)
            # phi_v for this task for each t
            phi_v_t = np.array([
                ConjunctionLattice.phi(m, candidate_v) for m in _msgs_for_task(s)
            ]) if False else None
            # We can't recompute phi without raw X; the caller must pass phi_v_per_task.
            raise RuntimeError("kkt_violation_score requires precomputed phi_v")
    return grads_per_g.max() / (2.0 ** len(candidate_v))


def kkt_score_with_phi(W, Phi_dict, Y, candidate_v, phi_v_dict, groups, C,
                       delta=0.5):
    """Compute KKT violation score given precomputed phi_v_dict {s: array(T)}.
    Score = max_g ||{<phi_v, hinge_deriv*y>_s}_{s in g}||_2 / d_v
    """
    grads = {}
    for s, Ps in Phi_dict.items():
        margin = 1.0 - Y[s] * (Ps @ W[s])
        _, deriv = huber_hinge_loss(margin, delta)
        grads[s] = -C * float(np.dot(phi_v_dict[s], Y[s] * deriv))
    best = 0.0
    for g in groups:
        v = np.array([grads[s] for s in g])
        n = float(np.linalg.norm(v))
        if n > best:
            best = n
    return best / (2.0 ** len(candidate_v))


def active_set_expand(W, X, Y, Phi_dict, U_active, lattice, groups, C,
                      threshold=0.01, max_add=5, max_len=2, delta=0.5):
    """Return list of new NCFs to add; phi_v_dict for each candidate.

    Considers all NCFs of length 2 not in active set.
    """
    candidates = []
    all_ncfs = lattice.all_ncf(max_order=max_len)
    set_act = set(U_active)
    scores = []
    for v in all_ncfs:
        if v in set_act:
            continue
        if len(v) > max_len or len(v) < 2:
            continue
        # compute phi_v for each task
        phi_v_dict = {}
        for s, msgs in X.items():
            phi_v_dict[s] = np.array(
                [ConjunctionLattice.phi(m, v) for m in msgs]
            )
        score = kkt_score_with_phi(W, Phi_dict, Y, v, phi_v_dict, groups, C, delta)
        if score > threshold:
            scores.append((score, v))
    scores.sort(reverse=True)
    return [v for _, v in scores[:max_add]]


# ----------------------- helpers (W <-> dict) -----------------------

def W_array_to_dict(W, U, S):
    return {s: {u: float(W[s, i]) for i, u in enumerate(U)} for s in range(S)}
