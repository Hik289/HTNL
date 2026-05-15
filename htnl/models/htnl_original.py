"""Method 0 (baseline): Original HTNL — variational form (Lemma 2) with
explicit gamma / lambda / mu auxiliary updates and active-set expansion.

Faithful to TKDE 2019 Section 5: alternating MM between W and (gamma, lambda, mu)
plus KKT-based active set growth.
"""
import numpy as np

from .ncf import ConjunctionLattice, precompute_phi_matrix, group_of_task
from ._core import (
    w_update_weighted_ridge,
    compute_objective,
    compute_r_u,
    active_set_expand,
    W_array_to_dict,
)


def _hoelder_coeffs(W, U, groups, p, eps=1e-8):
    """Collapsed weighted-ridge coefficients c_{u,g} from Hölder optimum.

    Equivalent to setting (gamma, lambda, mu) to their analytic optima and
    plugging into c_{u,g} = sum_{v in A(u)} d_v^2 / (gamma_v * lambda_{v,u} * mu_{u,v,g}).
    """
    n_u = len(U)
    n_g = len(groups)
    r_u, s_ug = compute_r_u(W, U, groups, eps=eps)  # r_u: (n_u,), s_ug: (n_u, n_g)

    # f_v = (sum_{u in D(v)} r_u^p)^{1/p}
    f_v = np.zeros(n_u)  # f_v[i] for v = U[i]
    for i, v in enumerate(U):
        s = 0.0
        for j, u in enumerate(U):
            if v.issubset(u):
                s += r_u[j] ** p
        f_v[i] = (s + eps) ** (1.0 / p)

    d_v = np.array([2.0 ** len(v) for v in U])
    Omega = float(np.sum(d_v * f_v))

    # c_{u,g} = (Omega / s_{u,g}) * r_u^{p-1} * sum_{v in A(u)} d_v * f_v^{1-p}
    c = {}
    for j, u in enumerate(U):
        ancestor_sum = 0.0
        for i, v in enumerate(U):
            if v.issubset(u):
                ancestor_sum += d_v[i] * f_v[i] ** (1.0 - p)
        for g_idx in range(n_g):
            denom = max(s_ug[j, g_idx], eps)
            c[(u, g_idx)] = (Omega * r_u[j] ** (p - 1) * ancestor_sum) / denom
    return c, Omega


def solve_htnl_original(X, Y, groups, lattice, C=1.0, p=1.5,
                        max_outer=2, max_inner=20, tol=1e-3, kkt_threshold=0.01,
                        max_add=5, verbose=False):
    """Original HTNL with explicit MM + active-set expansion.

    Returns (W_dict, history) with history['obj'] -> list of objective values.
    """
    S = len(X)
    history = {'obj': [], 'rounds': [], 'active_size': []}

    # Round 0: singletons
    active_set = [frozenset([i]) for i in range(lattice.V)]

    W_dict_final = None

    for outer in range(max_outer):
        U = list(active_set)
        Phi = precompute_phi_matrix(X, lattice, U)

        # Normalise Phi to keep numerics tame (column-wise max scaling)
        col_scale = np.ones(len(U))

        n_u = len(U)
        W = np.zeros((S, n_u))

        # Initial uniform coefficients (cold start)
        c = {}
        for u in U:
            for g_idx in range(len(groups)):
                c[(u, g_idx)] = 1.0

        prev_obj = np.inf
        for it in range(max_inner):
            # W-step: pass c/2 because objective has (1/2)*Omega^2 = (1/2)*sum c||W||^2
            c_half = {k: v * 0.5 for k, v in c.items()}
            W = w_update_weighted_ridge(Phi, Y, U, c_half, groups, S, C,
                                        W_init=W, max_iter=60)

            # Auxiliary update via Hölder closed form (gamma, lambda, mu collapsed)
            c, omega_val = _hoelder_coeffs(W, U, groups, p)

            obj = compute_objective(W, Phi, Y, U, groups, C=C, p=p, mode='squared')
            history['obj'].append(obj)
            history['rounds'].append(outer)
            history['active_size'].append(n_u)
            if verbose:
                print(f"[Original] round {outer} it {it}: obj={obj:.4f} Omega={omega_val:.4f}")
            if abs(prev_obj - obj) < tol * (abs(prev_obj) + 1e-8):
                break
            prev_obj = obj

        # KKT check / active-set expansion
        if outer < max_outer - 1:
            new_v = active_set_expand(W, X, Y, Phi, U, lattice, groups, C,
                                      threshold=kkt_threshold, max_add=max_add,
                                      max_len=2)
            if not new_v:
                W_dict_final = W_array_to_dict(W, U, S)
                break
            active_set = U + new_v
        W_dict_final = W_array_to_dict(W, U, S)

    return W_dict_final, history
