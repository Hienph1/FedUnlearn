import torch
import time
import joblib
import torch.nn as nn
import random
import numpy as np
import os
from torchvision.models import resnet18

from models.Nets import MLP, CNNMnist, CNNCifar, Logistic, LeNet, FashionCNN4
from utils.power_iteration import spectral_radius

# Re-use all subspace helpers and the per-sample GGN kernel
# that were defined in Approximator.py — single source of truth.
from utils.Approximator import (
    SelectedParameterSpec,
    build_spec,
    _flatten_selected,
    _build_state_with_vector,
    _compute_per_sample_ggn,
)


###############################################################################
#                         MODEL BUILDER  (resnet variant)                     #
###############################################################################

def _build_model(args, img_size):
    """Rebuild the model from args — mirrors load_models.py + resnet branches."""
    net = None
    if args.model == 'cnn' and args.dataset == 'cifar':
        net = CNNCifar(args=args)
    elif args.model == 'cnn' and args.dataset in ('mnist', 'fashion-mnist'):
        net = CNNMnist(args=args)
    elif args.model == 'cnn4' and args.dataset in ('mnist', 'fashion-mnist'):
        net = FashionCNN4()
    elif args.model == 'lenet' and args.dataset == 'fashion-mnist':
        net = LeNet()
    elif args.model == 'resnet18' and args.dataset == 'celeba':
        net = resnet18(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, 2)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net = resnet18(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, 10)
    elif args.model == 'resnet18' and args.dataset == 'svhn':
        net = resnet18(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, 10)
    elif args.model == 'resnet18' and args.dataset == 'lfw':
        net = resnet18(pretrained=True)
        net.fc = nn.Sequential(
            nn.Linear(net.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 29),
        )
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    elif args.model == 'logistic':
        len_in = 1
        for x in img_size:
            len_in *= x
        net = Logistic(dim_in=len_in, dim_out=args.num_classes)
    else:
        raise ValueError(f"Unrecognized model: {args.model!r}")

    return net.to(args.device)


###############################################################################
#                         MAIN APPROXIMATOR FUNCTION                          #
###############################################################################

def getapproximator_resnet(args, img_size, Dataset2recollect, indices_to_unlearn):
    """Compute per-sample SGN approximators for the forget set (resnet variant).

    Differences from ``getapproximator`` in Approximator.py
    --------------------------------------------------------
    • Checkpoints are stored **per-epoch** in separate files
      (one ``.dat`` per ``iter``), so we load inside the epoch loop.
    • Model is rebuilt manually via ``_build_model()`` rather than
      ``load_model()``, to support the full resnet / CNN / MLP zoo.

    Everything else — per-sample GGN kernel, subspace spec, projection,
    LM damping, accumulator structure — is identical to Approximator.py.
    The GGN technique follows the reference repo (fusg_utils.py) exactly:

        grad_vec  = ∂L/∂θ_layer          (first-order only)
        J         = [∂z_c/∂θ_layer]_c    (C backward passes per sample)
        H_ℓ       = diag(p) − p p⊤       (softmax Hessian, exact)
        curvature = J⊤ H_ℓ J

    Projection is applied AFTER accumulation (same as fusg_utils lines 612-613):
        projected_gradient  = U⊤ · gradient_mean
        projected_curvature = U⊤ · curvature_mean · U  +  γI

    Parameters
    ----------
    args               : parsed args
    img_size           : image shape tuple
    Dataset2recollect  : DatasetSplit wrapping training data
    indices_to_unlearn : list[int]

    Returns
    -------
    approximators : dict[int → {"projected_gradient":  Tensor (r,),
                                "projected_curvature": Tensor (r,r),
                                "basis_matrix":        Tensor (d,r)}]
    rho           : float
    """
    ###########################################################################
    # Setup
    ###########################################################################
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.empty_cache()

    net = _build_model(args, img_size)
    net.eval()

    dataset   = Dataset2recollect
    loss_func = torch.nn.CrossEntropyLoss()
    lr        = args.lr
    gamma     = float(getattr(args, "gamma", 1e-2))

    # Build subspace spec once from the (randomly initialised) model
    # architecture — the spec only depends on layer names/shapes, not weights.
    spec = build_spec(net, args)
    r    = spec.effective_rank
    d    = spec.selected_dim
    U    = spec.basis_matrix       # shape (d, r), float64, cpu

    forget_set = set(indices_to_unlearn)

    ###########################################################################
    # Accumulators  (only for forget samples)
    ###########################################################################
    gradient_sum = {i: torch.zeros(d, dtype=torch.float64) for i in forget_set}
    curvature_sum = {i: torch.zeros((d, d), dtype=torch.float64) for i in forget_set}
    step_count    = {i: 0 for i in forget_set}

    computed_rho = False
    rho          = 0.0

    ###########################################################################
    # Replay training trajectory — per-epoch checkpoint files
    #
    # The resnet variant saves one checkpoint file per epoch (iter), each
    # containing all batch snapshots for that epoch.  We load them one by
    # one to avoid holding every epoch in RAM simultaneously.
    ###########################################################################
    global_step = 0   # absolute step index across all epochs and batches

    for iter in range(args.epochs):
        path1 = "./Checkpoint/Resnet/model_{}_checkpoints".format(args.model)
        file_name = "check_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}_iter_{}.dat".format(
            args.dataset, args.epochs, args.lr, args.lr_decay,
            args.clip, args.seed, iter)
        file_path = os.path.join(path1, file_name)

        if not os.path.exists(file_path):
            print(f"[Approximator_resnet] WARNING: checkpoint not found: {file_path} — skipping epoch {iter}")
            continue

        info = joblib.load(file_path)

        for b, snapshot in enumerate(info):
            model_state = snapshot["model_list"]
            batch_idx   = snapshot["batch_idx_list"]

            # ResNet18 uses BatchNorm → keep in train() mode so BN stats
            # are consistent with how it was trained; all others use eval().
            if args.model == 'resnet18':
                net.train()
            else:
                net.eval()
            net.load_state_dict(model_state)

            base_state = {
                name: tensor.detach().clone()
                for name, tensor in net.state_dict().items()
            }

            # --- spectral radius (once, first batch of first epoch) -------
            if not computed_rho:
                batch_imgs, batch_lbls = [], []
                for i in batch_idx:
                    img_i, lbl_i, _ = dataset[i]
                    batch_imgs.append(img_i.unsqueeze(0).to(args.device))
                    batch_lbls.append(torch.tensor([lbl_i]).to(args.device))
                batch_imgs = torch.cat(batch_imgs, dim=0)
                batch_lbls = torch.cat(batch_lbls, dim=0)
                with torch.enable_grad():
                    log_probs  = net(batch_imgs)
                    loss_batch = loss_func(log_probs, batch_lbls)
                    for param in net.parameters():
                        loss_batch += 0.5 * args.regularization * (param * param).sum()
                rho = spectral_radius(
                    args, loss_batch=loss_batch, net=net,
                    t=len(info) * iter + b)
                computed_rho = True
                del batch_imgs, batch_lbls, log_probs, loss_batch

            # --- per-sample GGN for forget samples in this batch ---------
            t_start = time.time()
            forget_in_batch = [i for i in batch_idx if i in forget_set]

            for i in forget_in_batch:
                img_i, lbl_i, _ = dataset[i]
                image = img_i.unsqueeze(0).to(args.device)
                label = torch.tensor([lbl_i]).to(args.device)

                grad_cpu, curv_cpu = _compute_per_sample_ggn(
                    net, base_state, spec, image, label,
                    args.device, loss_func,
                )

                # lr schedule: lr * lr_decay^(global_step) / batch_size
                weight = lr * (args.lr_decay ** global_step) / len(batch_idx)
                gradient_sum[i]  += weight * grad_cpu
                curvature_sum[i] += weight * curv_cpu
                step_count[i]    += 1

            t_end = time.time()
            print("Epoch {:3d} Batch {:3d} | forget in batch: {} | "
                  "GGN time: {:.4f}s".format(
                      iter, b, len(forget_in_batch), t_end - t_start))

            global_step += 1

        del info   # free epoch checkpoint from RAM before loading next one

    ###########################################################################
    # Project onto subspace U and apply LM damping
    # (identical to Approximator.py — fusg_utils lines 612-613)
    ###########################################################################
    approximators = {}
    gamma_I = gamma * torch.eye(r, dtype=torch.float64)

    for i in forget_set:
        n = max(step_count[i], 1)
        grad_mean = gradient_sum[i]  / float(n)
        curv_mean = curvature_sum[i] / float(n)

        proj_grad        = U.t() @ grad_mean          # shape (r,)
        proj_curv        = U.t() @ curv_mean @ U      # shape (r, r)
        proj_curv_damped = proj_curv + gamma_I        # PD guaranteed

        approximators[i] = {
            "projected_gradient":  proj_grad.detach().clone(),
            "projected_curvature": proj_curv_damped.detach().clone(),
            "basis_matrix":        U.detach().clone(),
        }

    del dataset
    print("[Approximator_resnet] Done. Approximators built for {} forget samples."
          .format(len(approximators)))

    return approximators, rho