"""Method 4: Weighted One-Level Group Lasso (Song & Zhao NeurIPS 2024).

Reformulates the bi-space penalty as a single-level weighted group lasso,
alternating between
  (1) FISTA on  C * hinge(W) + sum_{u,g} c_{u,g} ||W_{u,g}||_2,
  (2) closed-form dual updates of alpha:
        alpha_{v,u} = (r_{v,u} + eps)^{p-1} / (sum_{u' in D(v)} (r_{v,u'}+eps)^p)^{(p-1)/p}
        c_{u,g}    = lambda * sum_{v in A(u)} d_v * alpha_{v,u}.
"""
import numpy as np

from .ncf import precompute_phi_matrix, group_of_task
from ._core import (
    huber_hinge_loss,
    compute_objective,
    compute_r_u,
    active_set_expand,
    W_array_to_dict,
)


def _hinge_grad(W, Phi, Y, C, delta=0.5):
    """Gradient of total Huber-hinge loss wrt W (S x |U|)."""
    G = np.zeros_like(W)
    for s, Ps in Phi.items():
        margin = 1.0 - Y[s] * (Ps @ W[s])
        _, deriv = huber_hinge_loss(margin, delta)
        G[s] = -C * Ps.T @ (Y[s] * deriv)
    return G


def fista_group_lasso(Phi, Y, U, c_dict, groups, S, C, W_init=None,
                      max_iter=120, tol=1e-5, delta=0.5):
    """FISTA for min_W C * huber_hinge + sum_{u,g} c_{u,g} ||W_{u,g}||_2.

    Block soft-thresholding proximal per (u, g) block where
    W_{u,g} = (W[s, u_idx])_{s in g}.
    """
    n_u = len(U)
    W = np.zeros((S, n_u)) if W_init is None else W_init.copy()

    # Lipschitz: huber-hinge derivative is at most 1/(2*delta), then chain through Phi
    L = 0.0
    for s, Ps in Phi.items():
        # ||Ps^T Ps|| upper bound by ||Ps||_F^2
        L = max(L, C * float(np.linalg.norm(Ps, 'fro') ** 2) / (2.0 * delta))
    L = max(L, 1e-3)

    Z = W.copy()
    W_prev = W.copy()
    t = 1.0

    for it in range(max_iter):
        G = _hinge_grad(Z, Phi, Y, C, delta)
        W_new = Z - G / L

        # Block soft-thresh per (g, u)
        for g_idx, g in enumerate(groups):
            for j, u in enumerate(U):
                w_block = W_new[g, j]
                norm = float(np.linalg.norm(w_block))
                thresh = c_dict[(u, g_idx)] / L
                if norm > thresh:
                    W_new[g, j] = w_block * (1.0 - thresh / norm)
                else:
                    W_new[g, j] = 0.0

        # FISTA momentum
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = W_new + ((t - 1.0) / t_new) * (W_new - W_prev)

        delta_W = float(np.linalg.norm(W_new - W_prev))
        W_prev = W_new
        t = t_new
        if delta_W < tol * (float(np.linalg.norm(W_new)) + 1e-8):
            W = W_new
            break
        W = W_new

    return W


def solve_wgl(X, Y, groups, lattice, C=1.0, p=1.5, lambda_reg=5.0,
              max_outer=2, max_iter=25, fista_iters=200, tol=1e-3,
              kkt_threshold=0.01, max_add=5, verbose=False, eps=1e-8):
    S = len(X)
    history = {'obj': [], 'rounds': [], 'active_size': []}
    active_set = [frozenset([i]) for i in range(lattice.V)]
    W_dict_final = None
    n_g = len(groups)

    for outer in range(max_outer):
        U = list(active_set)
        Phi = precompute_phi_matrix(X, lattice, U)
        n_u = len(U)
        W = np.zeros((S, n_u))

        # Build ancestor structure
        anc_of = {j: [i for i, v in enumerate(U) if v.issubset(U[j])] for j in range(n_u)}
        desc_of = {i: [j for j, u in enumerate(U) if U[i].issubset(u)] for i in range(n_u)}

        # Initialise alpha uniformly per v: alpha_{v,u} = 1/|D(v)|
        alpha = {}
        for i, v in enumerate(U):
            n_desc = len(desc_of[i])
            for j in desc_of[i]:
                alpha[(i, j)] = 1.0 / n_desc

        d_v_arr = np.array([2.0 ** len(v) for v in U])

        prev_obj = np.inf
        for it in range(max_iter):
            # Step 1: collapse alpha into per-(u, g) weights
            c_dict = {}
            for j, u in enumerate(U):
                w_u = lambda_reg * sum(d_v_arr[i] * alpha[(i, j)] for i in anc_of[j])
                for g_idx in range(n_g):
                    c_dict[(u, g_idx)] = w_u

            # Step 2: FISTA on weighted group lasso
            W = fista_group_lasso(Phi, Y, U, c_dict, groups, S, C,
                                  W_init=W, max_iter=fista_iters, tol=tol * 0.1)

            # Step 3: compute r_{v,u} = sum_g ||W_{u,g}|| (independent of v actually,
            # = r_u). The 1-level WGL closed-form per Song&Zhao uses r_u.
            r_u, _ = compute_r_u(W, U, groups, eps=eps)

            # Step 4: closed-form alpha update
            for i, v in enumerate(U):
                desc = desc_of[i]
                base = sum((r_u[j] + eps) ** p for j in desc)
                denom = base ** ((p - 1.0) / p) + eps
                for j in desc:
                    alpha[(i, j)] = (r_u[j] + eps) ** (p - 1.0) / denom

            # WGL minimizes  C * hinge + lambda_reg * Omega(W)   (linear form);
            # report that objective so the trajectory is monotone.
            from ._core import hinge_total_obj as _h, compute_omega as _o
            obj = _h(W, Phi, Y, C) + lambda_reg * _o(W, U, groups, p)
            history['obj'].append(obj)
            history['rounds'].append(outer)
            history['active_size'].append(n_u)
            if verbose:
                print(f"[WGL] round {outer} it {it}: obj={obj:.4f}")
            if abs(prev_obj - obj) < tol * (abs(prev_obj) + 1e-8):
                break
            prev_obj = obj

        if outer < max_outer - 1:
            new_v = active_set_expand(W, X, Y, Phi, U, lattice, groups, C,
                                      threshold=kkt_threshold,
                                      max_add=max_add, max_len=2)
            if not new_v:
                W_dict_final = W_array_to_dict(W, U, S)
                break
            active_set = U + new_v
        W_dict_final = W_array_to_dict(W, U, S)

    return W_dict_final, history
