import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import copy
import random
import matplotlib
matplotlib.use('Agg')

torch.set_printoptions(threshold=np.inf)


class DatasetSplit(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.all_indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label, self.all_indices[item]


def train(step, args, net, client_loader, learning_rate, info,
          current_epoch, client_id=0, spec=None):
    """Train one local epoch and cache per-client SGN sketches (s_k, C_k).

    Cached SGN mode (proposal Section IV.D.b):
    ------------------------------------------
    Instead of saving full model state_dicts (expensive), we compute and
    cache the projected sufficient statistics directly during training:

        s_k = U^T grad_L_k(theta_t)     shape (r,)   — projected gradient
        C_k = U^T G_k(theta_t) U        shape (r, r) — projected curvature

    These are accumulated (weighted by lr schedule) across all batches
    after warmup_rounds, then averaged and stored per-client.  At unlearn
    time the server subtracts the forget-client contributions:

        g_U  = Σ_{k ∉ S} p_k s_k  =  (Σ_all p_k s_k) - Σ_{k∈S} p_k s_k
        H~_U = Σ_{k ∉ S} p_k C_k  =  (Σ_all p_k C_k) - Σ_{k∈S} p_k C_k

    Storage cost: r + r^2 floats per client per round — orders of magnitude
    smaller than a full state_dict.

    Parameters
    ----------
    step         : int       global step counter
    args         : Namespace must include warmup_rounds, lr_decay, clip,
                             regularization, seed, device
    net          : nn.Module current model (deepcopy from caller)
    client_loader: DataLoader DataLoader of client k
    learning_rate: float
    info         : list      accumulated sketch dicts for this client.
                             Each entry: {"s_k": Tensor(r,), "C_k": Tensor(r,r)}
                             Only appended when current_epoch >= warmup_rounds.
    current_epoch: int       0-indexed global round
    client_id    : int       client index (used in log messages)
    spec         : SelectedParameterSpec or None
                             Subspace spec built from model architecture.
                             If None, falls back to legacy state_dict caching
                             (kept for backward compatibility).

    Returns
    -------
    state_dict, loss, lr, step, info
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    loss_func = nn.CrossEntropyLoss()
    lr = learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=args.lr_decay)

    from torch.utils.data import DataLoader as DL
    base_dataset = client_loader.dataset
    wrapped      = DatasetSplit(base_dataset)
    dataloader   = DL(wrapped, batch_size=client_loader.batch_size, shuffle=True)

    cache_this_epoch = (current_epoch >= args.warmup_rounds)

    # ── Decide caching mode ───────────────────────────────────────────────────
    use_sketch_cache = (spec is not None) and cache_this_epoch

    # Accumulators for sketch-mode (reset each epoch, averaged at end)
    if use_sketch_cache:
        from utils.Approximator import _compute_per_sample_ggn, _compute_batch_sketch
        r = spec.effective_rank
        s_k_sum  = torch.zeros(r, dtype=torch.float64)   # Σ weighted grad projections
        C_k_sum  = torch.zeros((r, r), dtype=torch.float64)  # Σ weighted curv projections
        U        = spec.basis_matrix                          # shape (d, r), cpu float64
        n_sketch = 0

    loss = 0
    for batch_idx, (images, labels, indices) in enumerate(dataloader):
        optimizer.zero_grad()
        net.eval()

        # ── Sketch-mode caching — batch Jacobian via vmap ───────────────────
        # Thay vì loop từng sample (64 × 12 backward = 768 passes/batch),
        # dùng torch.func.vmap + jacrev để vectorize toàn bộ batch:
        #   J_batch = vmap(jacrev(f))(images)   shape (B, C, d)
        #   1 pass thay vì B×C passes → ~60x nhanh hơn
        if use_sketch_cache:
            base_state = {
                name: tensor.detach().clone()
                for name, tensor in net.state_dict().items()
            }
            weight = lr * (args.lr_decay ** step) / max(len(indices), 1)

            # Dùng thẳng images/labels đã có trong batch — không cần load lại
            # từ base_dataset (tránh B lần dataset[idx] thừa mỗi batch).
            batch_imgs = images.to(args.device)   # (B, ...)
            batch_lbls = labels.to(args.device)   # (B,)
            valid_count = batch_imgs.shape[0]

            if valid_count > 0:
                grad_batch, C_batch = _compute_batch_sketch(
                    net, base_state, spec, batch_imgs, batch_lbls,
                    args.device, loss_func, U,
                )
                # grad_batch: (r,)  — mean projected gradient over batch
                # C_batch:    (r,r) — mean projected curvature over batch
                s_k_sum  += weight * valid_count * grad_batch
                C_k_sum  += weight * valid_count * C_batch
                n_sketch += valid_count

        # ── Legacy caching (fallback when spec=None) ──────────────────────────
        elif cache_this_epoch and spec is None:
            info.append({
                "batch_idx_list": indices.tolist(),
                "model_list":     copy.deepcopy(net).state_dict(),
            })

        # ── Forward + backward pass (training) ───────────────────────────────
        images, labels = images.to(args.device), labels.to(args.device)
        net.zero_grad()
        log_probs = net(images)

        loss = loss_func(log_probs, labels)
        for param in net.parameters():
            loss += 0.5 * args.regularization * (param * param).sum()

        net.train()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=net.parameters(), max_norm=args.clip, norm_type=2)
        optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        del images, labels, log_probs
        print("     Step {:3d}     Batch {:3d}, Batch Size: {:3d}, "
              "Training Loss: {:.2f}  [cache={}, sketch={}]".format(
                  step, batch_idx, dataloader.batch_size,
                  loss, cache_this_epoch, use_sketch_cache))
        step += 1

    # ── After epoch: store averaged sketch for this client ────────────────────
    if use_sketch_cache and n_sketch > 0:
        info.append({
            "s_k": (s_k_sum / float(n_sketch)).detach().clone(),   # shape (r,)
            "C_k": (C_k_sum / float(n_sketch)).detach().clone(),   # shape (r, r)
        })

    return net.state_dict(), loss, lr, step, info


def get_checkpoint_path(args, client_id):
    """Return checkpoint file path for client_id."""
    path1 = "./Checkpoint/model_{}_checkpoints".format(args.model)
    file_name = (
        "check_{}_client{}_epoch_{}_lr_{}"
        "_lrdecay_{}_clip_{}_seed{}.dat".format(
            args.dataset, client_id, args.global_epoch,
            args.lr, args.lr_decay, args.clip, args.seed)
    )
    return os.path.join(path1, file_name)
