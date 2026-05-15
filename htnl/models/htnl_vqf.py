"""Method 2: VQF + MM. Same variational form as HTNL Original but with a
clean modular implementation: collapsed Hoelder closed-form for the
(gamma, lambda, mu) auxiliaries is hidden inside a single helper.

Two layers of variational reweighting:
  - Outer:  Omega(W) -> reweighted ridge with coefficients c_{u,g}.
  - Inner:  per-task L-BFGS-B on Huber-hinge + sum_u c_{u,g(s)} w_{s,u}^2.
"""
import numpy as np

from .ncf import precompute_phi_matrix, group_of_task
from ._core import (
    w_update_weighted_ridge,
    compute_objective,
    compute_r_u,
    active_set_expand,
    W_array_to_dict,
)


def _vqf_update_aux(W, U, groups, p, eps=1e-8):
    """Collapsed VQF coefficient update (Holder equality conditions).

    Returns:
      c : dict (u, g_idx) -> scalar  (already absorbing the 1/2 factor;
                                       passed straight to weighted-ridge solver)
      diag : dict with intermediate values for diagnostics.
    """
    r_u, s_ug = compute_r_u(W, U, groups, eps=eps)
    n_u = len(U)
    n_g = len(groups)

    # f_v
    is_anc = np.zeros((n_u, n_u), dtype=bool)
    for i, v in enumerate(U):
        for j, u in enumerate(U):
            if v.issubset(u):
                is_anc[i, j] = True
    rp = r_u ** p
    f_v = np.array([(rp[is_anc[i]].sum() + eps) ** (1.0 / p) for i in range(n_u)])
    d_v = np.array([2.0 ** len(v) for v in U])
    Omega = float(np.sum(d_v * f_v))

    # ancestor_sum_u = sum_{v in A(u)} d_v * f_v^{1-p}
    ancestor_sum = np.zeros(n_u)
    fv_pow = f_v ** (1.0 - p)
    for j in range(n_u):
        anc_idx = is_anc[:, j]  # rows v that are ancestors of u
        ancestor_sum[j] = float((d_v[anc_idx] * fv_pow[anc_idx]).sum())

    # Coefficient already includes the 1/2 from (1/2)*Omega^2:
    # (1/2)*Omega^2 = (1/2) * sum_{u,g} c_{u,g} * ||W_{u,g}||^2 at optimum
    # So pass c_{u,g}/2 to the solver. We bake it in here.
    c = {}
    for j, u in enumerate(U):
        rp_u = r_u[j] ** (p - 1)
        for g_idx in range(n_g):
            denom = max(s_ug[j, g_idx], eps)
            full = (Omega * rp_u * ancestor_sum[j]) / denom
            c[(u, g_idx)] = 0.5 * full
    return c, {'omega': Omega, 'r_u': r_u, 's_ug': s_ug, 'f_v': f_v}


def solve_vqf(X, Y, groups, lattice, C=1.0, p=1.5,
              max_outer=2, max_inner=25, tol=1e-3, kkt_threshold=0.01,
              max_add=5, verbose=False):
    S = len(X)
    history = {'obj': [], 'rounds': [], 'active_size': []}

    active_set = [frozenset([i]) for i in range(lattice.V)]
    W_dict_final = None

    for outer in range(max_outer):
        U = list(active_set)
        Phi = precompute_phi_matrix(X, lattice, U)
        n_u = len(U)
        W = np.zeros((S, n_u))

        # warm-start coefficients: uniform tiny ridge
        c = {(u, g): 0.5 for u in U for g in range(len(groups))}

        prev_obj = np.inf
        for it in range(max_inner):
            W = w_update_weighted_ridge(Phi, Y, U, c, groups, S, C,
                                        W_init=W, max_iter=60)
            c, diag = _vqf_update_aux(W, U, groups, p)

            obj = compute_objective(W, Phi, Y, U, groups, C=C, p=p, mode='squared')
            history['obj'].append(obj)
            history['rounds'].append(outer)
            history['active_size'].append(n_u)
            if verbose:
                print(f"[VQF] round {outer} it {it}: obj={obj:.4f} Omega={diag['omega']:.4f}")
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
