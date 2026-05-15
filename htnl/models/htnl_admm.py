"""Method 3: Direct ADMM with consensus copies Z_{v,u,g}.

Reformulation:
   min_{W, Z}  C * hinge(W) + sum_v d_v * f_v(Z_v)
   s.t. for each v in A(u) and each g:  W_{g,u} = Z_{v,u,g}

with f_v(Z_v) = (sum_{u in D(v)} (sum_g ||Z_{v,u,g}||_2)^p)^{1/p}.

Augmented-Lagrangian (scaled-dual) ADMM:
  - W-step : per-task L-BFGS-B (Huber hinge + quadratic term).
  - Z-step : per ancestor v, run a few iterations of FISTA on the smoothed prox.
  - Dual    : D_{v,u,g} += W_{g,u} - Z_{v,u,g}.
"""
import numpy as np

from .ncf import precompute_phi_matrix, group_of_task
from ._core import (
    w_update_weighted_ridge,
    compute_objective,
    active_set_expand,
    W_array_to_dict,
)


def _grad_fv(Z_v_blocks, d_v, p, descend_idx, n_g, eps=1e-6):
    """Smoothed gradient of d_v * f_v(Z_v).

    Z_v_blocks: dict (u_idx, g_idx) -> ndarray(|g|,)
    descend_idx : list of u indices (within global U) that are descendants of v
    """
    # smoothed group norms
    s_smooth = {}
    for j in descend_idx:
        for g_idx in range(n_g):
            z = Z_v_blocks[(j, g_idx)]
            s_smooth[(j, g_idx)] = float(np.sqrt(z @ z + eps * eps))
    r_u = {j: sum(s_smooth[(j, g_idx)] for g_idx in range(n_g))
           for j in descend_idx}
    f_v = (sum(r_u[j] ** p for j in descend_idx) + eps) ** (1.0 / p)

    grad = {}
    for j in descend_idx:
        # df/dr_u = r_u^{p-1} / f_v^{p-1}
        a = (r_u[j] ** (p - 1)) / (f_v ** (p - 1) + eps)
        for g_idx in range(n_g):
            z = Z_v_blocks[(j, g_idx)]
            grad[(j, g_idx)] = d_v * a * (z / s_smooth[(j, g_idx)])
    return grad, f_v


def _z_update_v(Q_v_blocks, descend_idx, d_v, rho, p, n_g,
                fista_iters=8, eps=1e-8):
    """Iterative Majorize-Minimize prox of d_v * f_v at point Q_v.

    Solves: min_Z d_v * f_v(Z) + (rho/2) * sum ||Z_{u,g} - Q_{u,g}||^2.

    Iteration: linearize the OUTER lp norm at the current Z to get a per-(u, g)
    weighted block-l2 penalty; apply block soft-thresholding which produces
    exact zeros.
    """
    # Initial block magnitudes from Q (a sensible starting Z)
    Z = {k: q.copy() for k, q in Q_v_blocks.items()}

    for _ in range(fista_iters):
        # current group norms
        s = {k: float(np.linalg.norm(z) + eps) for k, z in Z.items()}
        r_u = {j: sum(s[(j, g)] for g in range(n_g)) + eps for j in descend_idx}
        f_v = (sum(r_u[j] ** p for j in descend_idx) + eps) ** (1.0 / p)
        Z_new = {}
        for j in descend_idx:
            # threshold per (j, g): d_v * r_u^{p-1} / f_v^{p-1}
            thr = d_v * (r_u[j] ** (p - 1)) / (f_v ** (p - 1) + eps)
            for g_idx in range(n_g):
                q = Q_v_blocks[(j, g_idx)]
                qn = float(np.linalg.norm(q))
                if qn < eps:
                    Z_new[(j, g_idx)] = np.zeros_like(q)
                    continue
                shrink = thr / rho / qn
                if shrink >= 1.0:
                    Z_new[(j, g_idx)] = np.zeros_like(q)
                else:
                    Z_new[(j, g_idx)] = (1.0 - shrink) * q
        Z = Z_new
    return Z


def solve_admm(X, Y, groups, lattice, C=1.0, p=1.5, rho=5.0,
               max_outer=2, max_iter=50, fista_iters=8, tol=1e-3,
               kkt_threshold=0.01, max_add=5, verbose=False, lam=1.0):
    S = len(X)
    history = {'obj': [], 'primal_res': [], 'dual_res': [],
               'rounds': [], 'active_size': []}

    active_set = [frozenset([i]) for i in range(lattice.V)]
    W_dict_final = None
    g_of_s = group_of_task(groups, S)
    n_g = len(groups)

    for outer in range(max_outer):
        U = list(active_set)
        Phi = precompute_phi_matrix(X, lattice, U)
        n_u = len(U)

        # Indexing structures
        is_anc = np.zeros((n_u, n_u), dtype=bool)
        for i, v in enumerate(U):
            for j, u in enumerate(U):
                if v.issubset(u):
                    is_anc[i, j] = True

        ancestors_of = {j: [i for i in range(n_u) if is_anc[i, j]] for j in range(n_u)}
        descendants_of = {i: [j for j in range(n_u) if is_anc[i, j]] for i in range(n_u)}

        # Initialise W, Z, D
        W = np.zeros((S, n_u))

        Z = {}
        Dual = {}
        for i, v in enumerate(U):
            for j in descendants_of[i]:
                for g_idx, g in enumerate(groups):
                    Z[(i, j, g_idx)] = np.zeros(len(g))
                    Dual[(i, j, g_idx)] = np.zeros(len(g))

        prev_obj = np.inf
        for it in range(max_iter):
            # ---- W-step ----
            # quadratic c_{u,g} on ||W_{g,u}||^2 = sum_{s in g} W_{s,u}^2
            # coefficient = (rho/2) * |A(u)|
            c_dict = {}
            extra_lin = np.zeros((S, n_u))
            for j, u in enumerate(U):
                anc = ancestors_of[j]
                for g_idx, g in enumerate(groups):
                    c_dict[(u, g_idx)] = 0.5 * rho * len(anc)
                    M = np.zeros(len(g))
                    for i in anc:
                        M += Z[(i, j, g_idx)] - Dual[(i, j, g_idx)]
                    # Add linear term: -rho * <W_{g,u}, M>  →  per task s in g:
                    # extra_lin[s, j] = rho * M[pos_in_g(s)]
                    for pos, s in enumerate(g):
                        extra_lin[s, j] = rho * M[pos]

            W = w_update_weighted_ridge(Phi, Y, U, c_dict, groups, S, C,
                                        W_init=W, max_iter=40,
                                        extra_linear=extra_lin)

            # ---- Z-step (per ancestor v) ----
            Z_old = {k: z.copy() for k, z in Z.items()}
            primal_sq = 0.0
            for i, v in enumerate(U):
                d_v = lam * (2.0 ** len(v))
                desc = descendants_of[i]
                Q_v = {}
                for j in desc:
                    for g_idx, g in enumerate(groups):
                        W_block = W[g, j]
                        Q_v[(j, g_idx)] = W_block + Dual[(i, j, g_idx)]
                Z_v_new = _z_update_v(Q_v, desc, d_v, rho, p, n_g,
                                      fista_iters=fista_iters)
                for j in desc:
                    for g_idx, g in enumerate(groups):
                        Z[(i, j, g_idx)] = Z_v_new[(j, g_idx)]
                        # primal residual
                        diff = W[g, j] - Z[(i, j, g_idx)]
                        primal_sq += float(diff @ diff)

            # ---- Dual step ----
            for (i, j, g_idx), zold in Z_old.items():
                g = groups[g_idx]
                Dual[(i, j, g_idx)] += W[g, j] - Z[(i, j, g_idx)]

            # Residuals
            primal_res = float(np.sqrt(primal_sq))
            dual_sq = 0.0
            for k, zold in Z_old.items():
                d = Z[k] - zold
                dual_sq += float(d @ d)
            dual_res = rho * float(np.sqrt(dual_sq))

            # ADMM optimizes the linear-Omega form (lam * Omega).
            from ._core import hinge_total_obj as _h, compute_omega as _o
            obj = _h(W, Phi, Y, C) + lam * _o(W, U, groups, p)
            history['obj'].append(obj)
            history['primal_res'].append(primal_res)
            history['dual_res'].append(dual_res)
            history['rounds'].append(outer)
            history['active_size'].append(n_u)
            if verbose:
                print(f"[ADMM] round {outer} it {it}: obj={obj:.4f} "
                      f"prim={primal_res:.4e} dual={dual_res:.4e}")

            if (primal_res < tol and dual_res < tol):
                break
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
