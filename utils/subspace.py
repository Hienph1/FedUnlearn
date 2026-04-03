"""utils/subspace.py

Public subspace utilities for FUSG unlearning.

Design rationale
----------------
All core logic (SelectedParameterSpec, build_spec, flatten helpers,
per-sample GGN kernel) lives in utils/Approximator.py so it can be
used both during the approximator pre-computation phase AND here.
This file re-exports those symbols and adds two higher-level helpers
that the rest of the pipeline (sgn_unlearn.py, main_proposed.py) can
call without importing Approximator directly:

    snapshot_selected   -- save a copy of the chosen layer's weights
    restore_selected    -- write a saved copy back into the model
    expand_delta        -- convert subspace correction α → full delta
    copy_vector_to_model-- write a flattened vector back into the model

These four functions mirror the identically-named helpers in the
reference repo (fusg_utils.py: snapshot_selected_parameters,
restore_selected_parameters, expand_basis_delta,
copy_selected_vector_into_model).
"""

import torch

# ---------------------------------------------------------------------------
# Re-export everything that downstream files need from Approximator.py.
# Single source of truth: all subspace logic lives in Approximator.py.
# ---------------------------------------------------------------------------
from utils.Approximator import (       # noqa: F401  (re-exported)
    SelectedParameterSpec,
    build_spec,
    _get_root_name,
    _select_layer_params,
    _flatten_selected,
    _build_state_with_vector,
    _compute_per_sample_ggn,
)


###############################################################################
#                    REVERSIBLE BOOKKEEPING HELPERS                           #
###############################################################################

def snapshot_selected(model, spec) -> dict:
    """Save a detached copy of the selected layer's parameters.

    Mirrors ``snapshot_selected_parameters`` in fusg_utils.py.
    Used by sgn_unlearn.py to enable reversible unlearning: store the
    model state *before* applying the SGN correction so it can be
    restored later if needed.

    Parameters
    ----------
    model : nn.Module   -- current model (weights already loaded)
    spec  : SelectedParameterSpec

    Returns
    -------
    dict[param_name → Tensor]   detached CPU clones of selected params
    """
    named = dict(model.named_parameters())
    return {name: named[name].detach().clone() for name in spec.names}


def restore_selected(model, spec, snapshot: dict) -> None:
    """Write a snapshot back into the model in-place.

    Mirrors ``restore_selected_parameters`` in fusg_utils.py.

    Parameters
    ----------
    model    : nn.Module
    spec     : SelectedParameterSpec
    snapshot : dict returned by snapshot_selected()
    """
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name in spec.names:
            named[name].copy_(
                snapshot[name].to(
                    device=named[name].device,
                    dtype=named[name].dtype,
                )
            )


def expand_delta(base_selected, alpha_vector, basis_matrix) -> tuple:
    """Expand a subspace correction α into the full selected-layer delta.

    Mirrors ``expand_basis_delta`` in fusg_utils.py.

    Computes:
        delta_full      = U @ α              shape (d_layer,)
        updated_selected = base_selected + delta_full

    Parameters
    ----------
    base_selected : Tensor  float64  shape (d_layer,)
        Flattened current weights of the selected layer.
    alpha_vector  : Tensor  float64  shape (r,)
        Subspace correction returned by solve_damped_system().
    basis_matrix  : Tensor  float64  shape (d_layer, r)
        Basis U stored in the approximator dict or spec.

    Returns
    -------
    updated_selected : Tensor  shape (d_layer,)
    delta_full       : Tensor  shape (d_layer,)
    """
    # Normalise everything to CPU float64 before arithmetic.
    # base_selected may arrive from CUDA (flattened directly from model
    # parameters); alpha_vector and basis_matrix are always on CPU
    # (the solver runs on CPU for numerical stability).
    # We do the update on CPU; copy_vector_to_model() moves the result
    # back to the correct device when writing into the model.
    base_cpu  = base_selected.to(dtype=torch.float64, device="cpu")
    alpha_cpu = alpha_vector.to(dtype=torch.float64,  device="cpu")

    if basis_matrix is None:
        delta_full = alpha_cpu
    else:
        U_cpu      = basis_matrix.to(dtype=torch.float64, device="cpu")
        delta_full = U_cpu @ alpha_cpu

    updated_selected = base_cpu + delta_full
    return updated_selected, delta_full


def copy_vector_to_model(model, spec, selected_vector) -> None:
    """Write a flattened selected-layer vector back into the model in-place.

    Mirrors ``copy_selected_vector_into_model`` in fusg_utils.py.

    Slices ``selected_vector`` according to spec.offsets and writes each
    slice into the corresponding named parameter, respecting the
    original device and dtype.

    Parameters
    ----------
    model           : nn.Module
    spec            : SelectedParameterSpec
    selected_vector : Tensor  shape (d_layer,)
    """
    named = dict(model.named_parameters())
    vec   = selected_vector.reshape(-1)
    with torch.no_grad():
        for (start, end), pname, shape, ref in zip(
            spec.offsets,
            spec.names,
            spec.shapes,
            [named[n] for n in spec.names],
        ):
            named[pname].copy_(
                vec[start:end].to(
                    device=ref.device,
                    dtype=ref.dtype,
                ).view(shape)
            )