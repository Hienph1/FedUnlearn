import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import time
import copy
import numpy as np
import torch
import os
import shutil
import joblib

from models.load_models import load_model
from models.Update import train, get_checkpoint_path
from models.test import test_img
from torch.utils.data import DataLoader, ConcatDataset

from utils.Approximator import getapproximator, build_spec
from utils.options import args_parser
from utils.perturbation import NoisedNetReturn
from utils.sgn_unlearn import sgn_unlearn_step
from utils.data_utils import data_set, separate_data, split_data


###############################################################################
#                               HELPER: FEDAVG                                #
###############################################################################

def fedavg(client_weights):
    """Average a list of state_dicts — standard FedAvg aggregation."""
    avg = copy.deepcopy(client_weights[0])
    for key in avg:
        for w in client_weights[1:]:
            avg[key] = avg[key] + w[key]
        if 'num_batches_tracked' in key:
            avg[key] = (avg[key] / len(client_weights)).long()
        else:
            avg[key] = avg[key] / len(client_weights)
    return avg


###############################################################################
#                               HELPER: TEST ALL CLIENTS                      #
###############################################################################

def test_all_clients(net, test_loaders, args):
    """Return (avg_acc, avg_loss) averaged across all clients."""
    total_acc  = 0.0
    total_loss = 0.0
    for loader in test_loaders:
        acc_k, loss_k = test_img(net, loader.dataset, args)
        total_acc  += float(acc_k)
        total_loss += float(loss_k)
    n = len(test_loaders)
    return total_acc / n, total_loss / n


###############################################################################
#                               MAIN                                          #
###############################################################################

if __name__ == '__main__':

###############################################################################
#                               SETUP                                         #
###############################################################################
    pycache_folder = "__pycache__"
    if os.path.exists(pycache_folder):
        shutil.rmtree(pycache_folder)

    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs("./data", exist_ok=True)
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # ── Load dataset và phân chia cho K clients ───────────────────────────────
    trainset, testset = data_set(args.data_name)

    train_data    = np.array(trainset.data)
    train_targets = np.array(trainset.targets)

    X, y, statistic = separate_data(
        data=(train_data, train_targets),
        num_clients=args.num_user,
        num_classes=args.num_classes,
        args=args,
        niid=args.niid,
        balance=args.balance,
        partition=args.partition,
    )

    client_all_loaders, test_loaders = split_data(X, y, args)

    img_size = trainset[0][0].shape
    net      = load_model(img_size)

    # Build subspace spec BEFORE training — depends only on architecture.
    # Passed into train() to enable sketch caching (s_k, C_k) instead of
    # saving full state_dicts (proposal Section IV.D.b).
    spec = build_spec(net, args)
    print("[Setup] Subspace spec built: layer='{}', d={}, r={}".format(
        spec.selection_prefix, spec.selected_dim, spec.effective_rank))

    print("\n[Setup] device={}, model={}, data_name={}, "
          "num_user={}, global_epoch={}, local_epoch={}, "
          "warmup_rounds={}, subspace_dim={}, "
          "forget_paradigm={}, gamma={}".format(
              args.device, args.model, args.data_name,
              args.num_user, args.global_epoch, args.local_epoch,
              args.warmup_rounds, args.subspace_dim,
              args.forget_paradigm, args.gamma))

###############################################################################
#                               FL TRAINING                                   #
###############################################################################
    print("\n" + 5*"#" + "  Federated Training Start  " + 5*"#")

    # ── Đường dẫn mid-training checkpoint ────────────────────────────────────
    resume_dir  = "./Checkpoint/resume"
    os.makedirs(resume_dir, exist_ok=True)
    resume_path = os.path.join(
        resume_dir,
        "resume_{}_data_{}_epoch_{}_lr_{}_seed{}.pth".format(
            args.model, args.data_name, args.global_epoch,
            args.lr, args.seed))

    # ── Cố gắng resume từ checkpoint bị gián đoạn ────────────────────────────
    acc_test        = []
    loss_test       = []
    lr              = args.lr
    step            = 0
    start_round     = 0
    info_per_client = [[] for _ in range(args.num_user)]

    if os.path.exists(resume_path):
        print("[Resume] Found checkpoint: {}".format(resume_path))
        ckpt = torch.load(resume_path, map_location=args.device)

        net.load_state_dict(ckpt["model_state"])
        lr              = ckpt["lr"]
        step            = ckpt["step"]
        start_round     = ckpt["global_round"] + 1   # bắt đầu từ round tiếp theo
        acc_test        = ckpt["acc_test"]
        loss_test       = ckpt["loss_test"]
        info_per_client = ckpt["info_per_client"]

        print("[Resume] Resuming from round {} (lr={:.6f}, step={})".format(
            start_round, lr, step))
    else:
        print("[Resume] No checkpoint found — starting from round 0.")

    # ── Vòng lặp FL chính ────────────────────────────────────────────────────
    for global_round in range(start_round, args.global_epoch):
        torch.cuda.synchronize()
        t_start = time.time()

        # ── Client selection ──────────────────────────────────────────────────
        # Dùng seed cố định theo round để đảm bảo resume cho ra cùng clients
        round_rng = random.Random(args.seed + global_round)
        selected_clients = round_rng.sample(
            range(args.num_user),
            k=max(1, int(args.num_user * args.fraction)))

        client_weights = []

        for client_id in selected_clients:
            w, loss, lr, step, info_per_client[client_id] = train(
                step=step,
                args=args,
                net=copy.deepcopy(net).to(args.device),
                client_loader=client_all_loaders[client_id],
                learning_rate=lr,
                info=info_per_client[client_id],
                current_epoch=global_round,
                client_id=client_id,
                spec=spec,   # enables sketch caching (s_k, C_k) per client
            )
            client_weights.append(w)

        # ── FedAvg ───────────────────────────────────────────────────────────
        global_state = fedavg(client_weights)
        net.load_state_dict(global_state)
        net.eval()

        torch.cuda.synchronize()
        t_end = time.time()

        avg_acc, avg_loss = test_all_clients(net, test_loaders, args)
        phase = "warmup" if global_round < args.warmup_rounds else "cache"
        print(" Round {:3d} [{}] | Avg acc: {:.2f} | "
              "Time: {:.4f}s".format(
                  global_round, phase, avg_acc, t_end - t_start))

        acc_test.append(avg_acc)
        loss_test.append(avg_loss)

        # ── Lưu mid-training checkpoint sau mỗi round ─────────────────────────
        torch.save({
            "global_round":    global_round,
            "model_state":     net.state_dict(),
            "lr":              lr,
            "step":            step,
            "acc_test":        acc_test,
            "loss_test":       loss_test,
            "info_per_client": info_per_client,
        }, resume_path)

    print(5*"#" + "  Federated Training End  " + 5*"#")

    # ── Lưu checkpoint per-client (sketches) vào disk ────────────────────────
    for client_id in range(args.num_user):
        ckpt_path = get_checkpoint_path(args, client_id)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        joblib.dump(info_per_client[client_id], ckpt_path)
        print("[Checkpoint] Client {:2d} saved ({} snapshots): {}".format(
            client_id, len(info_per_client[client_id]), ckpt_path))

    # ── Xóa mid-training checkpoint khi training hoàn tất ────────────────────
    if os.path.exists(resume_path):
        os.remove(resume_path)
        print("[Resume] Training complete — resume checkpoint deleted.")

    # ── Lưu model gốc (trước unlearning) ─────────────────────────────────────
    rootpath = './log/Original/Model/'
    os.makedirs(rootpath, exist_ok=True)
    torch.save(net.state_dict(),
               rootpath + 'Original_model_{}_data_{}_epoch_{}_lr_{}'
               '_lrdecay_{}_clip_{}_seed{}.pth'.format(
                   args.model, args.data_name, args.global_epoch,
                   args.lr, args.lr_decay, args.clip, args.seed))

###############################################################################
#                         ĐỊNH NGHĨA FORGET / RETAIN SET                      #
###############################################################################

    if args.forget_paradigm == 'client':
        # Quên toàn bộ dữ liệu của các forget clients
        forget_client_set = set(args.forget_client_idx)

        indices_to_unlearn = []
        for k in args.forget_client_idx:
            ds = client_all_loaders[k].dataset
            indices_to_unlearn.extend(list(range(len(ds))))

        forget_dataset = ConcatDataset(
            [client_all_loaders[k].dataset for k in args.forget_client_idx])
        remain_dataset = ConcatDataset(
            [client_all_loaders[k].dataset
             for k in range(args.num_user)
             if k not in forget_client_set])

    elif args.forget_paradigm == 'class':
        # Quên tất cả samples thuộc forget_class_idx trên mọi clients
        forget_class_set = set(args.forget_class_idx)
        forget_items, remain_items = [], []
        indices_to_unlearn = []

        for k in range(args.num_user):
            ds = client_all_loaders[k].dataset
            for local_idx in range(len(ds)):
                img, lbl = ds[local_idx]
                if int(lbl) in forget_class_set:
                    forget_items.append((img, lbl))
                    indices_to_unlearn.append(local_idx)
                else:
                    remain_items.append((img, lbl))

        forget_dataset = forget_items   # list of (img, lbl)
        remain_dataset = remain_items

    else:
        # 'sample': random sample num_forget mẫu từ toàn bộ dữ liệu
        all_items = []
        for k in range(args.num_user):
            ds = client_all_loaders[k].dataset
            for local_idx in range(len(ds)):
                all_items.append((k, local_idx))

        forget_pairs   = random.sample(all_items, k=args.num_forget)
        forget_indices_set = set(map(tuple, forget_pairs))
        remain_pairs   = [p for p in all_items if tuple(p) not in forget_indices_set]

        # indices_to_unlearn: dùng local index của từng client
        indices_to_unlearn = [local_idx for _, local_idx in forget_pairs]

        forget_dataset = [client_all_loaders[k].dataset[i]
                          for k, i in forget_pairs]
        remain_dataset = [client_all_loaders[k].dataset[i]
                          for k, i in remain_pairs]

    print("\n[Info] forget_paradigm={}, forget set size={}, "
          "retain set size={}".format(
              args.forget_paradigm,
              len(indices_to_unlearn),
              len(remain_dataset) if hasattr(remain_dataset, '__len__') else '?'))

###############################################################################
#                         PRECOMPUTATION — APPROXIMATORS                      #
###############################################################################
    rho = 0.0

    save_path = (
        './log/Proposed/statistics/'
        'Approximators_FL_model_{}_data_{}_paradigm_{}_epoch_{}'
        '_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
            args.model, args.data_name, args.forget_paradigm,
            args.global_epoch, args.lr, args.lr_decay,
            args.clip, args.seed))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        print("\n[Approximator] Computing GGN sketches across {} clients ..."
              .format(args.num_user))
        Approximators, rho = getapproximator(
            args,
            img_size,
            client_all_loaders=client_all_loaders,
            indices_to_unlearn=indices_to_unlearn,
        )
        torch.save({'Approximators': Approximators, 'rho': rho}, save_path)
        print("[Approximator] Saved to", save_path)
    else:
        print("\n[Approximator] Loading cached approximators from", save_path)
        ckpt          = torch.load(save_path)
        Approximators = ckpt['Approximators']
        rho           = ckpt['rho']

###############################################################################
#                               UNLEARNING  (SGN)                             #
###############################################################################
    print("\n[SGN] Begin unlearning ...")
    torch.cuda.synchronize()
    unlearn_t_start = time.time()

    # One-shot SGN update:
    #   1. aggregate_sketches()  — build g_U, H̃_U from cached approximators
    #   2. solve_damped_system() — α_SGN = -H̃_U⁻¹ g_U  (Cholesky + retry)
    #   3. apply θ̃ = θ̂ + U α_SGN  (only selected layer changes)
    net, bookkeeping = sgn_unlearn_step(
        net=net,
        approximators=Approximators,
        forget_indices=indices_to_unlearn,
        args=args,
    )

    # Optional: add DP noise after SGN update
    if args.application:
        w = NoisedNetReturn(
            args,
            net=copy.deepcopy(net).to(args.device),
            rho=rho,
            epsilon=args.epsilon,
            delta=args.delta,
            n=sum(len(client_all_loaders[k].dataset)
                  for k in range(args.num_user)),
            m=len(indices_to_unlearn),
        )
        net.load_state_dict(w)

    torch.cuda.synchronize()
    unlearn_t_end = time.time()

    net.eval()
    avg_acc_post, avg_loss_post = test_all_clients(net, test_loaders, args)
    acc_test.append(avg_acc_post)
    loss_test.append(avg_loss_post)

    print("[SGN] Unlearning done. Avg test accuracy: {:.2f}, "
          "Time: {:.6f}s".format(avg_acc_post, unlearn_t_end - unlearn_t_start))
    print("[SGN] solver={solver}, retries={retries}, "
          "damping_used={damping_used:.2e}, "
          "correction_norm={cn:.4e}".format(
              cn=bookkeeping['correction_norm'],
              **bookkeeping['solve_info']))

###############################################################################
#                               SAVE                                          #
###############################################################################
    # ── Unlearned model ───────────────────────────────────────────────────────
    rootpath1 = './log/Proposed/Model/'
    os.makedirs(rootpath1, exist_ok=True)
    torch.save(net.state_dict(),
               rootpath1 + 'Proposed_model_{}_data_{}_paradigm_{}'
               '_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
                   args.model, args.data_name, args.forget_paradigm,
                   len(indices_to_unlearn), args.global_epoch,
                   args.lr, args.lr_decay, args.clip, args.seed))

    # ── Bookkeeping (reversible) ───────────────────────────────────────────────
    rootpath2 = './log/Proposed/Bookkeeping/'
    os.makedirs(rootpath2, exist_ok=True)
    torch.save(bookkeeping,
               rootpath2 + 'Bookkeeping_model_{}_data_{}_paradigm_{}'
               '_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
                   args.model, args.data_name, args.forget_paradigm,
                   len(indices_to_unlearn), args.global_epoch,
                   args.lr, args.lr_decay, args.clip, args.seed))

    # ── Test accuracy log ──────────────────────────────────────────────────────
    rootpath3 = './log/Proposed/acctest/'
    os.makedirs(rootpath3, exist_ok=True)
    fname3 = (rootpath3
              + 'Proposed_accfile_model_{}_data_{}_paradigm_{}'
                '_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.format(
                    args.model, args.data_name, args.forget_paradigm,
                    len(indices_to_unlearn), args.global_epoch,
                    args.lr, args.lr_decay, args.clip, args.seed))
    with open(fname3, "w") as f:
        for ac in acc_test:
            f.write(str(ac) + '\n')

    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.xlabel('round')
    plt.ylabel('avg test accuracy')
    plt.title('FL + FUSG unlearning')
    plt.savefig(fname3.replace('.dat', '.png'))
    plt.close()

    # ── Test loss log ──────────────────────────────────────────────────────────
    rootpath4 = './log/Proposed/losstest/'
    os.makedirs(rootpath4, exist_ok=True)
    fname4 = (rootpath4
              + 'Proposed_lossfile_model_{}_data_{}_paradigm_{}'
                '_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.format(
                    args.model, args.data_name, args.forget_paradigm,
                    len(indices_to_unlearn), args.global_epoch,
                    args.lr, args.lr_decay, args.clip, args.seed))
    with open(fname4, "w") as f:
        for l in loss_test:
            f.write(str(l) + '\n')

    plt.figure()
    plt.plot(range(len(loss_test)), loss_test)
    plt.xlabel('round')
    plt.ylabel('avg test loss')
    plt.savefig(fname4.replace('.dat', '.png'))
    plt.close()

    # ── Forget set accuracy ───────────────────────────────────────────────────
    # Xây dựng DataLoader tạm để test_img có thể dùng
    if hasattr(forget_dataset, '__len__') and len(forget_dataset) > 0:
        if isinstance(forget_dataset, list):
            forget_loader_tmp = DataLoader(
                forget_dataset, batch_size=args.bs, shuffle=False)
            forget_acc, forget_loss = test_img(net, forget_loader_tmp.dataset, args)
        else:
            forget_acc, forget_loss = test_img(net, forget_dataset, args)
        print("[Eval] Forget set  accuracy: {:.2f}".format(forget_acc))
    else:
        forget_acc = float('nan')
        print("[Eval] Forget set  accuracy: N/A (empty)")

    rootpath5 = './log/Proposed/accforget/'
    os.makedirs(rootpath5, exist_ok=True)
    with open(rootpath5
              + 'Proposed_accfile_model_{}_data_{}_paradigm_{}'
                '_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.format(
                    args.model, args.data_name, args.forget_paradigm,
                    len(indices_to_unlearn), args.global_epoch,
                    args.lr, args.lr_decay, args.clip, args.seed), 'w') as f:
        f.write(str(forget_acc))

    # ── Retain set accuracy ───────────────────────────────────────────────────
    if hasattr(remain_dataset, '__len__') and len(remain_dataset) > 0:
        if isinstance(remain_dataset, list):
            remain_loader_tmp = DataLoader(
                remain_dataset, batch_size=args.bs, shuffle=False)
            remain_acc, remain_loss = test_img(net, remain_loader_tmp.dataset, args)
        else:
            remain_acc, remain_loss = test_img(net, remain_dataset, args)
        print("[Eval] Retain set  accuracy: {:.2f}".format(remain_acc))
    else:
        remain_acc = float('nan')
        print("[Eval] Retain set  accuracy: N/A (empty)")

    rootpath6 = './log/Proposed/accremain/'
    os.makedirs(rootpath6, exist_ok=True)
    with open(rootpath6
              + 'Proposed_accfile_model_{}_data_{}_paradigm_{}'
                '_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.dat'.format(
                    args.model, args.data_name, args.forget_paradigm,
                    len(indices_to_unlearn), args.global_epoch,
                    args.lr, args.lr_decay, args.clip, args.seed), 'w') as f:
        f.write(str(remain_acc))

###############################################################################
#                               SUMMARY                                       #
###############################################################################
    print("\n[Done]")
    print("  Original model   : {}".format(rootpath))
    print("  Unlearned model  : {}".format(rootpath1))
    print("  Bookkeeping      : {}".format(rootpath2))
    print("  Forget paradigm  : {}".format(args.forget_paradigm))
    print("  Forget set size  : {}".format(len(indices_to_unlearn)))
    print("  Forget acc       : {:.2f}".format(forget_acc)
          if not (isinstance(forget_acc, float) and
                  forget_acc != forget_acc) else "  Forget acc       : N/A")
    print("  Retain acc       : {:.2f}".format(remain_acc)
          if not (isinstance(remain_acc, float) and
                  remain_acc != remain_acc) else "  Retain acc       : N/A")
    print("  Test  acc (post) : {:.2f}".format(avg_acc_post))