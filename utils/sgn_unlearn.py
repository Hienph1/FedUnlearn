"""utils/sgn_unlearn.py

SGN (Subspace Gauss-Newton) unlearning step.

This file implements Algorithm 1 of the proposal:

    1. aggregate_sketches()   -- build g_U and H̃_U from cached approximators
    2. solve_damped_system()  -- solve α_SGN = -H̃_U⁻¹ g_U
                                 (Cholesky + retry damping, mirrors repo)
    3. sgn_unlearn_step()     -- apply θ̃ = θ̂ + U α_SGN to the model
                                 with optional reversible bookkeeping

All heavy maths are first-order only — no create_graph, no HVP.
The solver follows ``solve_damped_projected_system`` in fusg_utils.py
exactly: Cholesky with up to 6 damping retries, falling back to
``torch.linalg.solve`` if every Cholesky attempt fails.
"""

import copy
import math
import torch

from utils.subspace import (
    _flatten_selected,
    expand_delta,
    copy_vector_to_model,
    snapshot_selected,
    restore_selected,
)


###############################################################################
#                        CONDITION-NUMBER ESTIMATOR                           #
###############################################################################

def _estimate_condition_number(matrix: torch.Tensor) -> float:
    """Estimate condition number via eigenvalues (mirrors fusg_utils.py)."""
    if matrix.numel() == 0:
        return math.inf
    try:
        eigs = torch.linalg.eigvalsh(matrix)
    except RuntimeError:
        return math.inf
    max_val = float(torch.max(torch.abs(eigs)).item())
    min_val = float(torch.min(torch.abs(eigs)).item())
    if min_val <= 1e-12:
        return math.inf
    return max_val / min_val


###############################################################################
#                          STEP 1 — AGGREGATE SKETCHES                       #
###############################################################################

def aggregate_sketches(approximators: dict, forget_indices: list) -> tuple:
    """Aggregate cached per-sample sketches into (g_U, H~_U).

    Implements Algorithm 1 of the proposal (Section IV.D.b) exactly:

        g_U  = U^T grad_L_ret(theta_hat)
             = -U^T grad_L_for(theta_hat)          (stationarity: eq. 2)
             = -(1/|F|) sum_{i in F} s_i

        H~_U = U^T G_ret(theta_hat) U + gamma*I    (eq. 6)
             = projected_curvature_retain           (precomputed in Approximator.py)

    IMPORTANT — two different curvatures:
      g_U   uses FORGET gradient   (negated via stationarity identity)
      H~_U  uses RETAIN curvature  (stored as "projected_curvature_retain")

    The key field name distinction in the approximators dict:
      "projected_gradient"        -> s_i = U^T grad_L_for sample i  (forget)
      "projected_curvature_retain"-> H~_U = U^T G_ret U + gamma*I   (retain, shared)
      "projected_curv_forget"     -> U^T G_for_i U  (per-sample forget, diagnostic only)

    Parameters
    ----------
    approximators : dict[int -> {...}]
        Output of getapproximator(). Keys are forget sample indices.
    forget_indices : list[int]

    Returns
    -------
    g_U     : Tensor float64 cpu  shape (r,)   — retain projected gradient
    H_tilde : Tensor float64 cpu  shape (r,r)  — retain projected curvature + damping
    U       : Tensor float64 cpu  shape (d,r)  — basis matrix
    """
    forget_set = set(forget_indices)

    if not approximators:
        raise ValueError("approximators dict is empty.")
    missing = forget_set - set(approximators.keys())
    if missing:
        raise ValueError(
            "approximators missing entries for forget indices: {}".format(missing))

    first = approximators[next(iter(forget_set))]
    r = int(first["projected_gradient"].shape[0])
    U = first["basis_matrix"].to(dtype=torch.float64, device="cpu")

    # ── g_U: negate mean forget gradient (stationarity identity, proposal eq.2) ──
    # grad_L_ret(theta_hat) = -grad_L_for(theta_hat)  at a stationary point of L
    # => g_U = U^T grad_L_ret = -U^T grad_L_for = -(1/|F|) sum_i s_i
    grad_sum = torch.zeros(r, dtype=torch.float64)
    for i in forget_set:
        grad_sum += approximators[i]["projected_gradient"].to(
            dtype=torch.float64, device="cpu")
    n_forget = float(len(forget_set))
    g_U = -grad_sum / n_forget                  # shape (r,)

    # ── H~_U: retain curvature — precomputed in Approximator.py (eq. 6) ─────────
    # H~_U = U^T G_ret(theta_hat) U + gamma*I
    # This is stored as "projected_curvature_retain" in every forget-sample entry
    # (all entries share the same value because it is a global aggregate).
    H_tilde = first["projected_curvature_retain"].to(dtype=torch.float64, device="cpu")

    return g_U, H_tilde, U


###############################################################################
#                          STEP 2 — SOLVE DAMPED SYSTEM                      #
###############################################################################

def solve_damped_system(
    projected_curvature: torch.Tensor,
    projected_gradient:  torch.Tensor,
    damping:             float,
) -> tuple:
    """Solve the damped projected GN system α = -H̃_U⁻¹ g_U.

    Mirrors ``solve_damped_projected_system`` in fusg_utils.py exactly:

    • Symmetrise H̃_U  →  sym = 0.5*(H + Hᵀ)
    • Try Cholesky decomposition with up to 6 damping retries:
          damping_used = base_damping × 10^retry_idx
      On success: solve via torch.cholesky_solve (fast, numerically stable)
    • If all 6 Cholesky attempts fail, fallback to torch.linalg.solve
      on the most-damped matrix.
    • Raises RuntimeError only if the fallback also fails.

    Parameters
    ----------
    projected_curvature : Tensor float64 cpu  shape (r,r)
    projected_gradient  : Tensor float64 cpu  shape (r,)
    damping             : float   base LM damping γ (from args.gamma)

    Returns
    -------
    alpha     : Tensor float64 cpu  shape (r,)   the SGN correction
    solve_info: dict   {"status", "damping_used", "solver",
                        "retries", "condition_number", [warning]}
    """
    projected_curvature = projected_curvature.to(dtype=torch.float64, device="cpu")
    projected_gradient  = projected_gradient.to( dtype=torch.float64, device="cpu")

    # Symmetrise to guard against tiny floating-point asymmetries
    sym_curv     = 0.5 * (projected_curvature + projected_curvature.t())
    identity     = torch.eye(sym_curv.size(0), dtype=torch.float64, device="cpu")
    base_damping = max(float(damping), 1e-8)
    last_error   = None

    # ------------------------------------------------------------------
    # Cholesky with progressive damping  (up to 6 retries: γ×1 … γ×1e5)
    # ------------------------------------------------------------------
    for retry_idx in range(6):
        damping_used    = base_damping * (10 ** retry_idx)
        damped_curv     = sym_curv + damping_used * identity
        try:
            L     = torch.linalg.cholesky(damped_curv)
            alpha = torch.cholesky_solve(
                (-projected_gradient).unsqueeze(1), L
            ).squeeze(1)
            solve_info = {
                "status":           "success",
                "damping_used":     damping_used,
                "solver":           "cholesky",
                "retries":          retry_idx,
                "condition_number": _estimate_condition_number(damped_curv),
            }
            return alpha, solve_info
        except RuntimeError as err:
            last_error = err

    # ------------------------------------------------------------------
    # Fallback: torch.linalg.solve on the most-damped matrix
    # ------------------------------------------------------------------
    final_damping = base_damping * (10 ** 5)
    final_matrix  = sym_curv + final_damping * identity
    try:
        alpha = torch.linalg.solve(
            final_matrix, (-projected_gradient).unsqueeze(1)
        ).squeeze(1)
        solve_info = {
            "status":           "success",
            "damping_used":     final_damping,
            "solver":           "solve",
            "retries":          6,
            "condition_number": _estimate_condition_number(final_matrix),
            "warning":          "cholesky failed on all retries; used linalg.solve",
        }
        return alpha, solve_info
    except RuntimeError as err:
        raise RuntimeError(
            f"SGN system solve failed after all damping retries: {err}"
        ) from last_error


###############################################################################
#                          STEP 3 — APPLY SGN UPDATE                         #
###############################################################################

def sgn_unlearn_step(
    net,
    approximators: dict,
    forget_indices: list,
    args,
) -> tuple:
    """Apply one SGN unlearning step to the model.

    Implements Algorithm 1 of the proposal end-to-end:

        g_U      =  aggregate forget-set projected gradients
        H̃_U     =  aggregate forget-set projected curvatures
        α_SGN   =  -H̃_U⁻¹ g_U          (Cholesky solve)
        θ̃       =  θ̂  +  U α_SGN        (only selected layer changes)

    With reversible bookkeeping enabled (always on), the function also
    stores before/after snapshots of the selected layer so the update
    can be undone later (relearn_unlearning_knowledge).

    Parameters
    ----------
    net             : nn.Module   trained model θ̂  (modified in-place
                                  on a deep copy — original is untouched)
    approximators   : dict        output of getapproximator()
    forget_indices  : list[int]
    args            : Namespace   must contain args.gamma

    Returns
    -------
    updated_net     : nn.Module   θ̃  (deep copy of net with SGN applied)
    bookkeeping     : dict        reversible bookkeeping payload:
                                  {"before_snapshot", "after_snapshot",
                                   "delta_vector", "selection_prefix",
                                   "effective_rank", "solve_info",
                                   "correction_norm"}
    """
    # ------------------------------------------------------------------
    # Step 1: aggregate sketches → (g_U, H̃_U, U)
    # ------------------------------------------------------------------
    g_U, H_tilde, U = aggregate_sketches(approximators, forget_indices)

    # ------------------------------------------------------------------
    # Step 2: solve α_SGN = -H̃_U⁻¹ g_U
    # ------------------------------------------------------------------
    damping = float(getattr(args, "gamma", 1e-2))
    alpha, solve_info = solve_damped_system(H_tilde, g_U, damping)

    correction_norm = float(torch.norm(alpha).item())
    print("[SGN] solve: solver={solver}, retries={retries}, "
          "damping_used={damping_used:.2e}, "
          "condition_number={condition_number:.4g}, "
          "correction_norm={cn:.4e}".format(
              cn=correction_norm, **solve_info))

    # ------------------------------------------------------------------
    # Step 3: apply θ̃ = θ̂ + U α_SGN
    # ------------------------------------------------------------------
    # Build a minimal spec-like object from the first approximator entry
    # so we can call the subspace helpers without a full spec rebuild.
    first = approximators[forget_indices[0]]
    U     = first["basis_matrix"].to(dtype=torch.float64, device="cpu")

    # We need the spec to know which layer params to touch.
    # Re-use build_spec so layer selection stays consistent with
    # Approximator.py (same args.fusg_subspace / fusg_layer_name).
    from utils.subspace import build_spec
    spec = build_spec(net, args)

    # Work on a deep copy — never mutate the caller's model
    updated_net = copy.deepcopy(net)

    # --- reversible bookkeeping: snapshot BEFORE update ---------------
    before_snapshot = snapshot_selected(updated_net, spec)

    # --- expand α into full-layer delta and apply ---------------------
    base_selected = _flatten_selected(updated_net, spec).detach().to(
        dtype=torch.float64)
    updated_selected, delta_full = expand_delta(base_selected, alpha, U)
    copy_vector_to_model(updated_net, spec, updated_selected)

    # --- reversible bookkeeping: snapshot AFTER update ----------------
    after_snapshot = snapshot_selected(updated_net, spec)

    bookkeeping = {
        "before_snapshot": before_snapshot,
        "after_snapshot":  after_snapshot,
        "delta_vector":    delta_full.detach().cpu().clone(),
        "selection_prefix": spec.selection_prefix,
        "effective_rank":   spec.effective_rank,
        "solve_info":       solve_info,
        "correction_norm":  correction_norm,
    }

    return updated_net, bookkeeping


###############################################################################
#                       OPTIONAL: UNDO / RELEARN                             #
###############################################################################

def undo_sgn_step(net, bookkeeping: dict, args) -> object:
    """Restore the model to its state before sgn_unlearn_step().

    Uses the ``before_snapshot`` stored in the bookkeeping dict to
    write back the original selected-layer weights.  Mirrors the
    ``relearn_unlearning_knowledge`` / restore path in fusg_unlearning.py.

    Parameters
    ----------
    net          : nn.Module   the *unlearned* model θ̃
    bookkeeping  : dict        returned by sgn_unlearn_step()
    args         : Namespace

    Returns
    -------
    restored_net : nn.Module   deep copy with original weights restored
    """
    from utils.subspace import build_spec
    spec = build_spec(net, args)

    before_snapshot = bookkeeping.get("before_snapshot")
    if before_snapshot is None:
        raise ValueError("bookkeeping dict has no 'before_snapshot' — "
                         "cannot undo SGN step.")

    restored_net = copy.deepcopy(net)
    restore_selected(restored_net, spec, before_snapshot)
    print("[SGN] undo: restored selected layer '{}' from before_snapshot."
          .format(bookkeeping.get("selection_prefix", "?")))
    return restored_net