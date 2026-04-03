import torch
import time
import random
import numpy as np
import joblib
import os
from dataclasses import dataclass, field
from typing import Any

from utils.options import args_parser
from models.load_datasets import load_dataset
from models.load_models import load_model
from utils.power_iteration import spectral_radius


###############################################################################
#                          SUBSPACE SPEC  (inline)                            #
###############################################################################
# Mirrors the SelectedParameterSpec / helpers from the reference repo
# (fusg_utils.py) but kept self-contained so no extra dependency is needed.

@dataclass
class SelectedParameterSpec:
    """Describe the selected parameter block and its basis matrix U."""
    selection_mode: str = "layer_name"
    selection_prefix: str = ""
    names: list = field(default_factory=list)
    shapes: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    offsets: list = field(default_factory=list)
    selected_dim: int = 0          # d_layer  (full dim of chosen layer)
    effective_rank: int = 0        # r        (projected / subspace dim)
    basis_matrix: Any = None       # U  ∈ R^{d_layer × r}


def _get_root_name(param_name: str) -> str:
    """Return the module root from a dotted parameter name."""
    if "." not in param_name:
        return param_name
    return param_name.rsplit(".", 1)[0]


def _select_layer_params(model, mode: str, layer_name: str):
    """Return (root_name, [(param_name, param), ...]) for the chosen layer."""
    named = list(model.named_parameters())
    if not named:
        raise ValueError("Model has no trainable parameters.")

    if mode == "first_layer":
        root = _get_root_name(named[0][0])
    elif mode == "layer_name":
        if not layer_name:
            # Supervisor advice: default to the SECOND named parameter block
            # when fusg_layer_name is left empty.
            roots_seen = []
            for pname, _ in named:
                r = _get_root_name(pname)
                if r not in roots_seen:
                    roots_seen.append(r)
                if len(roots_seen) == 2:
                    break
            if len(roots_seen) < 2:
                # Only one block → fall back to it
                root = roots_seen[0]
            else:
                root = roots_seen[1]
        else:
            root = layer_name
    else:
        raise NotImplementedError(f"Unsupported fusg_subspace mode: {mode!r}")

    selected = [
        (pname, p) for pname, p in named
        if _get_root_name(pname) == root or pname.startswith(root + ".")
    ]
    if not selected:
        raise ValueError(f"No parameters matched layer prefix {root!r}")
    return root, selected


def build_spec(model, args) -> SelectedParameterSpec:
    """Build SelectedParameterSpec from model + args.

    Maps directly onto the logic in ``build_selected_parameter_spec``
    from the reference repo (fusg_utils.py).

    Subspace basis U is a coordinate-truncated identity:
        U = I_{d_layer}[:, :r]   shape (d_layer, r)
    i.e. U simply selects the first r coordinates of the chosen layer's
    flattened parameter vector.  This is Construction (1) from the
    proposal (Section IV.C) — the simplest and fastest option.
    """
    mode       = getattr(args, "fusg_subspace",   "layer_name")
    layer_name = getattr(args, "fusg_layer_name", "")
    rank       = int(getattr(args, "subspace_dim", 64))

    root, selected = _select_layer_params(model, mode, layer_name)

    names, shapes, sizes, offsets = [], [], [], []
    total_dim = 0
    for pname, p in selected:
        sz = int(p.numel())
        names.append(pname)
        shapes.append(tuple(p.shape))
        sizes.append(sz)
        offsets.append((total_dim, total_dim + sz))
        total_dim += sz

    effective_rank = min(rank, total_dim)
    # U = first `effective_rank` columns of identity  →  shape (total_dim, r)
    basis_matrix = torch.eye(total_dim, dtype=torch.float64)[:, :effective_rank]

    print(f"[Approximator] subspace: layer='{root}', "
          f"d_layer={total_dim}, r={effective_rank}")

    return SelectedParameterSpec(
        selection_mode=mode,
        selection_prefix=root,
        names=names,
        shapes=shapes,
        sizes=sizes,
        offsets=offsets,
        selected_dim=total_dim,
        effective_rank=effective_rank,
        basis_matrix=basis_matrix,
    )


def _flatten_selected(model, spec) -> torch.Tensor:
    """Flatten the selected layer parameters into a 1-D vector."""
    named = dict(model.named_parameters())
    parts = [named[n].detach().reshape(-1) for n in spec.names]
    return torch.cat(parts, dim=0)


def _build_state_with_vector(base_state, spec, vector):
    """Return a state-dict copy with the selected block replaced by vector."""
    updated = {n: t.detach().clone() for n, t in base_state.items()}
    vec = vector.reshape(-1)
    for (start, end), pname, shape, ref in zip(
        spec.offsets, spec.names, spec.shapes,
        [base_state[n] for n in spec.names]
    ):
        updated[pname] = vec[start:end].to(
            device=ref.device, dtype=ref.dtype).view(shape)
    return updated


###############################################################################
#               PER-SAMPLE GGN SKETCH  (core of the refactor)                #
###############################################################################

def _compute_per_sample_ggn(
    model, base_state, spec, image, label, device, loss_func
):
    """Compute gradient and GGN curvature for ONE sample.

    Follows exactly the technique in ``compute_projected_client_statistics``
    from the reference repo (fusg_utils.py lines 569-596):

    1.  selected_vector  =  flatten(θ_layer),  requires_grad=True
    2.  forward pass via functional_call so grad flows through selected_vector
    3.  gradient  g  =  ∂L/∂selected_vector          shape (d_layer,)
    4.  jacobian  J  =  [∂z_c/∂selected_vector]_c     shape (C, d_layer)
        — computed by C separate backward passes on each logit z_c
    5.  H_ℓ  =  diag(p) − p p⊤                       shape (C, C)
        — exact softmax Hessian (= GGN curvature for cross-entropy)
    6.  sample_curvature  =  J⊤ H_ℓ J                 shape (d_layer, d_layer)

    Returns
    -------
    grad_vec   : torch.Tensor  float64  cpu  shape (d_layer,)
    curvature  : torch.Tensor  float64  cpu  shape (d_layer, d_layer)
    """
    selected_vector = _flatten_selected(model, spec).detach().to(device)
    selected_vector = selected_vector.to(dtype=torch.float32)
    selected_vector.requires_grad_(True)

    # Functional forward: replace selected layer params with selected_vector
    state_mapping = _build_state_with_vector(base_state, spec, selected_vector)
    try:
        from torch.func import functional_call
    except ImportError:
        from torch.nn.utils.stateless import functional_call

    logits = functional_call(model, state_mapping, (image,))
    if hasattr(logits, "logits"):     # HuggingFace-style output
        logits = logits.logits
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits_vec = logits.reshape(-1)   # shape (C,)

    # --- gradient of loss w.r.t. selected_vector -------------------------
    loss = loss_func(logits, label)
    grad_vec = torch.autograd.grad(
        loss, selected_vector, retain_graph=True, allow_unused=False
    )[0].reshape(-1)                  # shape (d_layer,)

    # --- Jacobian J: one backward per logit ------------------------------
    jacobian_rows = []
    for c in range(int(logits_vec.numel())):
        row = torch.autograd.grad(
            logits_vec[c], selected_vector,
            retain_graph=True, allow_unused=False
        )[0].reshape(-1)              # shape (d_layer,)
        jacobian_rows.append(row)
    J = torch.stack(jacobian_rows, dim=0)   # shape (C, d_layer)

    # --- softmax Hessian H_ℓ = diag(p) - p p⊤ ---------------------------
    # Move to CPU + float64 immediately so all subsequent ops stay on CPU.
    probs = torch.softmax(
        logits_vec.detach().to(dtype=torch.float64, device="cpu"), dim=0)  # (C,)
    H_ell = torch.diag(probs) - torch.outer(probs, probs)                  # (C, C) cpu

    # --- GGN = J⊤ H_ℓ J --------------------------------------------------
    J64       = J.detach().to(dtype=torch.float64, device="cpu")  # (C, d_layer) cpu
    curvature = J64.t() @ H_ell @ J64                             # (d_layer, d_layer) cpu
    grad_cpu  = grad_vec.detach().to(dtype=torch.float64, device="cpu")

    return grad_cpu, curvature


###############################################################################
#               BATCH GGN SKETCH — vectorized via vmap (fast path)           #
###############################################################################

def _compute_batch_sketch(model, base_state, spec, images, labels,
                           device, loss_func, U):
    """Compute projected gradient and curvature for a FULL BATCH efficiently.

    Instead of calling _compute_per_sample_ggn B times (B × C backward passes),
    this function uses torch.func.vmap + jacrev to vectorize the Jacobian
    computation across the entire batch in a single vmapped call.

    Cost: O(C) vmapped forward passes instead of O(B × C) separate passes.
    Speedup: ~B× faster (batch_size = 64 → ~60x speedup over the loop).

    Follows the same math as _compute_per_sample_ggn:
        J_i   = jacrev(f)(x_i)           shape (C, d)
        H_ell = diag(p_i) - p_i p_i^T    shape (C, C)
        GGN_i = J_i^T H_ell J_i          shape (d, d)
        s_k   = U^T mean(grad_i)          shape (r,)
        C_k   = U^T mean(GGN_i) U         shape (r, r)

    Parameters
    ----------
    model      : nn.Module   current model (eval mode)
    base_state : dict        state_dict snapshot
    spec       : SelectedParameterSpec
    images     : Tensor  float32  shape (B, ...)  on device
    labels     : Tensor  int64    shape (B,)       on device
    device     : torch.device
    loss_func  : CrossEntropyLoss
    U          : Tensor  float64  shape (d, r)  on cpu

    Returns
    -------
    s_k : Tensor float64 cpu  shape (r,)    mean projected gradient
    C_k : Tensor float64 cpu  shape (r, r)  mean projected curvature
    """
    try:
        from torch.func import functional_call, vmap, jacrev
    except ImportError:
        # PyTorch < 2.0 fallback: per-sample loop
        B = images.shape[0]
        s_acc = torch.zeros(U.shape[1], dtype=torch.float64)
        C_acc = torch.zeros(U.shape[1], U.shape[1], dtype=torch.float64)
        for b in range(B):
            g, curv = _compute_per_sample_ggn(
                model, base_state, spec,
                images[b:b+1], labels[b:b+1], device, loss_func)
            s_acc += U.t() @ g
            C_acc += U.t() @ curv @ U
        return s_acc / B, C_acc / B

    # ── Extract selected parameters as a single vector ────────────────────────
    selected_vector = _flatten_selected(model, spec).detach().to(
        device=device, dtype=torch.float32)
    d = selected_vector.shape[0]
    r = U.shape[1]

    # ── Define per-sample function: x -> logits (selected params as input) ───
    def f_single(sel_vec, x):
        """Forward pass with sel_vec replacing selected layer params."""
        state = _build_state_with_vector(base_state, spec, sel_vec)
        out   = functional_call(model, state, (x.unsqueeze(0),))
        if hasattr(out, "logits"):
            out = out.logits
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(-1)   # shape (C,)

    # ── Vectorized Jacobian: vmap(jacrev(f))(images) ─────────────────────────
    # jacrev differentiates f w.r.t. its FIRST argument (sel_vec).
    # vmap maps it over the batch dimension of images.
    # Result J: shape (B, C, d)
    try:
        jac_fn  = jacrev(f_single)   # grad w.r.t. sel_vec
        J_batch = vmap(jac_fn, in_dims=(None, 0))(
            selected_vector, images)   # (B, C, d)
    except Exception:
        # vmap/jacrev unavailable or failed — fall back to loop
        B = images.shape[0]
        s_acc = torch.zeros(r, dtype=torch.float64)
        C_acc = torch.zeros(r, r, dtype=torch.float64)
        for b in range(B):
            g, curv = _compute_per_sample_ggn(
                model, base_state, spec,
                images[b:b+1], labels[b:b+1], device, loss_func)
            s_acc += U.t() @ g
            C_acc += U.t() @ curv @ U
        return s_acc / B, C_acc / B

    # ── Per-sample gradient of loss ──────────────────────────────────────────
    # g_i = J_i^T * (d_loss/d_logits)_i   shape (B, d)
    # Use mean across batch for s_k
    B = J_batch.shape[0]
    J64 = J_batch.detach().to(dtype=torch.float64, device="cpu")   # (B, C, d)

    # Compute softmax probs and H_ell per sample
    with torch.no_grad():
        logits_batch = vmap(lambda x: f_single(selected_vector, x))(images)
        # shape (B, C)

    logits_cpu = logits_batch.detach().to(dtype=torch.float64, device="cpu")
    probs_batch = torch.softmax(logits_cpu, dim=-1)   # (B, C)

    # ── Gradient of cross-entropy w.r.t. logits: p - one_hot(y) ─────────────
    labels_cpu = labels.cpu()
    one_hot    = torch.zeros_like(probs_batch)
    one_hot.scatter_(1, labels_cpu.unsqueeze(1).long(), 1.0)
    dL_dlogits = probs_batch - one_hot                 # (B, C)

    # g_i = J_i^T dL_dlogits_i   shape (B, d)
    # einsum: b=batch, c=class, d=param
    grad_batch_d = torch.einsum('bcd,bc->bd', J64, dL_dlogits)   # (B, d)
    grad_mean_d  = grad_batch_d.mean(dim=0)                       # (d,)
    s_k          = U.t().to(dtype=torch.float64) @ grad_mean_d    # (r,)

    # ── GGN = J^T H_ell J per sample, then mean ──────────────────────────────
    # H_ell_i = diag(p_i) - p_i p_i^T
    # GGN_i   = J_i^T H_ell_i J_i   shape (d, d)
    # Efficient: GGN_i = J_i^T (diag(p) J_i) - J_i^T (p p^T J_i)
    #          = J_i^T diag(p) J_i - (J_i^T p)(p^T J_i)

    # (B, d, d) via einsum
    # Step 1: J^T diag(p) J  = einsum('bcd,bc,bce->bde', J, p, J)
    # Step 2: (J^T p)(p^T J) = einsum('bcd,bc->bd', J, p) outer product
    p = probs_batch                                          # (B, C)
    JtDJ  = torch.einsum('bcd,bc,bce->bde', J64, p, J64)   # (B, d, d)
    Jtp   = torch.einsum('bcd,bc->bd', J64, p)             # (B, d)
    JtppJ = torch.einsum('bd,be->bde', Jtp, Jtp)           # (B, d, d)

    GGN_batch = JtDJ - JtppJ                                # (B, d, d)
    GGN_mean  = GGN_batch.mean(dim=0)                       # (d, d)

    U64 = U.to(dtype=torch.float64)
    C_k = U64.t() @ GGN_mean @ U64                          # (r, r)

    return s_k, C_k


###############################################################################
#                         MAIN APPROXIMATOR FUNCTION                          #
###############################################################################

def getapproximator(args, img_size, client_all_loaders, indices_to_unlearn):
    """Compute SGN approximators using cached per-client sketches.

    Implements the Cached SGN mode (proposal Section IV.D.b):

        During training (Update.train with spec != None), each client k caches:
            s_k = U^T grad_L_k(theta_t)   shape (r,)   projected gradient
            C_k = U^T G_k(theta_t) U      shape (r, r) projected curvature

        At unlearn time, the server aggregates over ALL clients, then subtracts
        the forget-client contributions to recover retain statistics:

            g_U  = Σ_{k ∉ F} p_k s_k  =  Σ_all p_k s_k  -  Σ_{k∈F} p_k s_k
            H~_U = Σ_{k ∉ F} p_k C_k  =  Σ_all p_k C_k  -  Σ_{k∈F} p_k C_k

        This requires ZERO extra communication rounds after training.
        Storage cost: r + r^2 floats per client per round (vs full state_dict).

    Falls back to trajectory replay (legacy mode) if sketch cache files are
    detected to contain "model_list" entries instead of "s_k"/"C_k" entries.

    Parameters
    ----------
    args               : Namespace  (warmup_rounds, subspace_dim, fusg_subspace,
                                     fusg_layer_name, gamma, regularization,
                                     lr, lr_decay, clip, seed, device, num_user)
    img_size           : tuple      image shape — used to rebuild the model
    client_all_loaders : list[DataLoader]  one per client
    indices_to_unlearn : list[int]  sample-level forget indices (used to infer
                                    forget clients in client-level paradigm)

    Returns
    -------
    approximators : dict[int -> {"projected_gradient":         Tensor (r,),
                                 "projected_curv_forget":      Tensor (r,r),
                                 "projected_curvature_retain": Tensor (r,r),
                                 "basis_matrix":               Tensor (d,r)}]
    rho           : float   spectral radius for DP calibration
    """
    ###########################################################################
    # Setup
    ###########################################################################
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )

    # img_size is passed in — no need to reload the dataset.
    net = load_model(img_size).to(args.device)
    net.eval()

    loss_func = torch.nn.CrossEntropyLoss()
    gamma     = float(getattr(args, "gamma", 1e-2))

    # Build subspace spec — depends only on architecture, not weights.
    spec = build_spec(net, args)
    r    = spec.effective_rank
    d    = spec.selected_dim
    U    = spec.basis_matrix       # shape (d, r), float64, cpu

    forget_set = set(indices_to_unlearn)
    gamma_I    = gamma * torch.eye(r, dtype=torch.float64)

    ###########################################################################
    # Detect cache mode by inspecting first available checkpoint
    ###########################################################################
    from models.Update import get_checkpoint_path

    use_sketch_mode = False
    for client_id in range(args.num_user):
        fp = get_checkpoint_path(args, client_id)
        if os.path.exists(fp):
            probe = joblib.load(fp)
            if probe and isinstance(probe[0], dict) and "s_k" in probe[0]:
                use_sketch_mode = True
            break

    print("[Approximator] cache mode: {}".format(
        "sketch (s_k, C_k)" if use_sketch_mode else "legacy trajectory replay"))

    ###########################################################################
    # MODE A — Sketch cache (proposal Section IV.D.b)
    #
    # Each checkpoint file contains a list of per-epoch dicts:
    #   {"s_k": Tensor(r,), "C_k": Tensor(r,r)}
    #
    # Aggregation:
    #   s_all = (1/K) Σ_k  mean_over_epochs(s_k)
    #   C_all = (1/K) Σ_k  mean_over_epochs(C_k)
    #
    # Then subtract forget clients:
    #   g_U  = -(s_all - Σ_{k∈F} p_k s_k_mean)   (stationarity: negate for retain)
    #   H~_U = (C_all - Σ_{k∈F} p_k C_k_mean) + gamma*I
    ###########################################################################
    if use_sketch_mode:
        # Per-client averages
        s_per_client = {}   # client_id -> Tensor(r,)
        C_per_client = {}   # client_id -> Tensor(r,r)
        loaded_clients = []

        for client_id in range(args.num_user):
            fp = get_checkpoint_path(args, client_id)
            if not os.path.exists(fp):
                print("[Approximator] WARNING: sketch not found for "
                      "client {} -- skipping".format(client_id))
                continue

            sketches = joblib.load(fp)   # list of {"s_k", "C_k"} per epoch

            if not sketches:
                continue

            # Average across cached epochs
            s_mean = torch.stack([e["s_k"].to(dtype=torch.float64)
                                  for e in sketches]).mean(dim=0)   # (r,)
            C_mean = torch.stack([e["C_k"].to(dtype=torch.float64)
                                  for e in sketches]).mean(dim=0)   # (r, r)

            s_per_client[client_id] = s_mean
            C_per_client[client_id] = C_mean
            loaded_clients.append(client_id)

        if not loaded_clients:
            raise RuntimeError("[Approximator] No sketch checkpoints found. "
                               "Run training with spec passed to train().")

        n_loaded = float(len(loaded_clients))

        # Global aggregate (uniform client weights p_k = 1/K)
        s_all = torch.stack(list(s_per_client.values())).mean(dim=0)  # (r,)
        C_all = torch.stack(list(C_per_client.values())).mean(dim=0)  # (r, r)

        # Identify forget clients — clients whose data overlaps forget_set
        # For client-level paradigm: directly from args.forget_client_idx
        # For other paradigms: approximate by checking which client loaders
        # contain any forget index
        forget_client_ids = set(getattr(args, "forget_client_idx", []))
        if not forget_client_ids:
            # Sample/class paradigm: find clients that contain forget indices
            for kid in range(args.num_user):
                ds = client_all_loaders[kid].dataset
                for local_idx in range(len(ds)):
                    if local_idx in forget_set:
                        forget_client_ids.add(kid)
                        break

        # Subtract forget-client contributions
        s_forget = torch.zeros(r, dtype=torch.float64)
        C_forget = torch.zeros((r, r), dtype=torch.float64)
        n_forget_clients = 0
        for kid in forget_client_ids:
            if kid in s_per_client:
                s_forget += s_per_client[kid]
                C_forget += C_per_client[kid]
                n_forget_clients += 1

        if n_forget_clients > 0:
            p_forget = float(n_forget_clients) / n_loaded
            s_ret = s_all - p_forget * (s_forget / float(n_forget_clients))
            C_ret = C_all - p_forget * (C_forget / float(n_forget_clients))
        else:
            # No forget client found in loaded checkpoints — use all
            s_ret = s_all
            C_ret = C_all

        # Build H~_U from retain curvature (eq. 6)
        proj_curv_ret_damped = C_ret + gamma_I          # (r, r) PD guaranteed

        # aggregate_sketches() computes:
        #   g_U = -mean("projected_gradient")   (negate for stationarity)
        # So "projected_gradient" must store the FORGET gradient s_forget_mean,
        # NOT s_ret. aggregate_sketches() will negate it to get g_U = -s_for = s_ret.
        #
        # If no forget client was identified, fall back to negating s_all
        # (approximation: assume s_all ≈ s_for at stationarity).
        if n_forget_clients > 0:
            s_for_store = (s_forget / float(n_forget_clients)).detach().clone()
        else:
            s_for_store = s_all.detach().clone()

        # Build approximators dict: one entry per forget index.
        # aggregate_sketches() reads "projected_gradient" and negates it → g_U.
        # aggregate_sketches() reads "projected_curvature_retain" directly → H~_U.
        approximators = {}
        for i in forget_set:
            approximators[i] = {
                "projected_gradient":         s_for_store,           # U^T grad_L_for
                "projected_curv_forget":      torch.zeros(r, r, dtype=torch.float64),
                "projected_curvature_retain": proj_curv_ret_damped.detach().clone(),
                "basis_matrix":               U.detach().clone(),
            }

        rho = 0.0   # spectral radius not computed in sketch mode
        print("[Approximator] Sketch mode done. "
              "Loaded {} clients, {} forget clients subtracted.".format(
                  len(loaded_clients), n_forget_clients))
        return approximators, rho

    ###########################################################################
    # MODE B — Legacy trajectory replay (fallback)
    #
    # Cache file contains {"batch_idx_list", "model_list"} entries.
    # Load state_dict, replay forward/backward to compute GGN per sample.
    # Uses G_all - G_for subtraction to recover G_ret.
    ###########################################################################
    lr    = args.lr

    gradient_sum    = {i: torch.zeros(d, dtype=torch.float64) for i in forget_set}
    curv_sum_forget = {i: torch.zeros((d, d), dtype=torch.float64) for i in forget_set}
    step_count_for  = {i: 0 for i in forget_set}

    curv_sum_all   = torch.zeros((d, d), dtype=torch.float64)
    step_count_all = 0

    computed_rho = False
    rho          = 0.0

    for client_id in range(args.num_user):
        file_path = get_checkpoint_path(args, client_id)
        if not os.path.exists(file_path):
            print("[Approximator] WARNING: checkpoint not found for "
                  "client {} -- skipping".format(client_id))
            continue

        info    = joblib.load(file_path)
        dataset = client_all_loaders[client_id].dataset

        for step, snapshot in enumerate(info):
            model_state = snapshot["model_list"]
            batch_idx   = snapshot["batch_idx_list"]

            net.load_state_dict(model_state)
            net.eval()

            base_state = {
                name: tensor.detach().clone()
                for name, tensor in net.state_dict().items()
            }

            if not computed_rho:
                batch_imgs, batch_lbls = [], []
                for i in batch_idx:
                    try:
                        img_i, lbl_i = dataset[i]
                    except IndexError:
                        continue
                    batch_imgs.append(img_i.unsqueeze(0).to(args.device))
                    batch_lbls.append(torch.tensor([lbl_i]).to(args.device))
                if batch_imgs:
                    batch_imgs = torch.cat(batch_imgs, dim=0)
                    batch_lbls = torch.cat(batch_lbls, dim=0)
                    with torch.enable_grad():
                        log_probs  = net(batch_imgs)
                        loss_batch = loss_func(log_probs, batch_lbls)
                        for param in net.parameters():
                            loss_batch += (0.5 * args.regularization
                                           * (param * param).sum())
                    rho = spectral_radius(args, loss_batch=loss_batch,
                                         net=net, t=len(info))
                    computed_rho = True
                    del batch_imgs, batch_lbls, log_probs, loss_batch

            t_start = time.time()
            forget_in_batch = [i for i in batch_idx if i in forget_set]

            for i in batch_idx:
                try:
                    img_i, lbl_i = dataset[i]
                except IndexError:
                    continue
                image = img_i.unsqueeze(0).to(args.device)
                label = torch.tensor([lbl_i]).to(args.device)

                grad_cpu, curv_cpu = _compute_per_sample_ggn(
                    net, base_state, spec, image, label, args.device, loss_func
                )

                weight = lr * (args.lr_decay ** step) / max(len(batch_idx), 1)

                curv_sum_all   += weight * curv_cpu
                step_count_all += 1

                if i in forget_set:
                    gradient_sum[i]    += weight * grad_cpu
                    curv_sum_forget[i] += weight * curv_cpu
                    step_count_for[i]  += 1

            t_end = time.time()
            print("Client {:2d} | Step {:3d} | batch={} forget={} | "
                  "GGN time: {:.4f}s".format(
                      client_id, step,
                      len(batch_idx), len(forget_in_batch),
                      t_end - t_start))

        del info

    # G_ret = G_all - G_for
    curv_for_total       = sum(curv_sum_forget[i] for i in forget_set)
    curv_ret             = curv_sum_all - curv_for_total
    n_all                = max(step_count_all, 1)
    n_for_total          = sum(step_count_for[i] for i in forget_set)
    n_ret                = max(n_all - n_for_total, 1)
    curv_ret_mean        = curv_ret / float(n_ret)
    proj_curv_ret        = U.t() @ curv_ret_mean @ U
    proj_curv_ret_damped = proj_curv_ret + gamma_I

    approximators = {}
    for i in forget_set:
        n_for         = max(step_count_for[i], 1)
        grad_mean     = gradient_sum[i]    / float(n_for)
        curv_for_mean = curv_sum_forget[i] / float(n_for)
        proj_grad     = U.t() @ grad_mean
        proj_curv_for = U.t() @ curv_for_mean @ U

        approximators[i] = {
            "projected_gradient":         proj_grad.detach().clone(),
            "projected_curv_forget":      proj_curv_for.detach().clone(),
            "projected_curvature_retain": proj_curv_ret_damped.detach().clone(),
            "basis_matrix":               U.detach().clone(),
        }

    print("[Approximator] Legacy mode done. Approximators for {} forget samples."
          .format(len(approximators)))
    return approximators, rho