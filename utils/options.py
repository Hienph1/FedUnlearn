import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Hello')

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    parser.add_argument('--epochs', type=int, default=100,
                        help="rounds of training (centralized mode; "
                             "in FL mode use --global_epoch instead)")
    parser.add_argument('--lr', type=float, default=0.05,
                        help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995,
                        help="learning rate decay each round")
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
    parser.add_argument('--regularization', type=float, default=1e-6,
                        help="l2 regularization")
    parser.add_argument('--batch_size', type=int, default=1000,
                        help="batch size (centralized mode; "
                             "in FL mode use --local_batch_size instead)")

    # -------------------------------------------------------------------------
    # Dataset / Model
    # -------------------------------------------------------------------------
    parser.add_argument('--model', type=str, default='logistic',
                        help='model name')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help="name of dataset")
    parser.add_argument('--num_dataset', type=int, default=1000,
                        help="number of train dataset")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1,
                        help="number of channels of images")
    parser.add_argument('--test_train_rate', type=float, default=0.4,
                        help="ratio of test set to training set")

    # -------------------------------------------------------------------------
    # Federated Learning
    # -------------------------------------------------------------------------
    parser.add_argument('--num_user', type=int, default=10,
                        help="number of FL clients")
    parser.add_argument('--global_epoch', type=int, default=100,
                        help="number of FL global communication rounds")
    parser.add_argument('--local_epoch', type=int, default=1,
                        help="number of local training epochs per client "
                             "per global round")
    parser.add_argument('--local_batch_size', type=int, default=64,
                        help="batch size for local training at each client")
    parser.add_argument('--fraction', type=float, default=1.0,
                        help="fraction of clients selected each round, "
                             "must be in (0, 1]")
    parser.add_argument('--data_name', type=str, default='mnist',
                        help="dataset name used by data_utils.py: "
                             "mnist, fashionmnist, cifar10, cifar100")
    parser.add_argument('--niid', type=lambda x: x.lower() != 'false',
                        default=True,
                        help="True = non-IID data distribution across clients")
    parser.add_argument('--balance', type=lambda x: x.lower() != 'false',
                        default=True,
                        help="True = balanced class distribution per client")
    parser.add_argument('--partition', type=str, default='dir',
                        choices=['dir', 'pat'],
                        help="data partition strategy: "
                             "'dir' = Dirichlet distribution; "
                             "'pat' = pathological (class-based) partition")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Dirichlet concentration parameter alpha. "
                             "Smaller alpha = more heterogeneous distribution "
                             "across clients")

    # -------------------------------------------------------------------------
    # Unlearning
    # -------------------------------------------------------------------------
    parser.add_argument('--forget_paradigm', type=str, default='client',
                        choices=['client', 'class', 'sample'],
                        help="unlearning granularity: "
                             "'client' = forget all data of selected clients; "
                             "'class'  = forget all samples of selected classes; "
                             "'sample' = forget a random subset of samples")
    parser.add_argument('--forget_client_idx', type=int, nargs='+', default=[0],
                        help="list of client indices to forget, "
                             "used when forget_paradigm='client'. "
                             "Example: --forget_client_idx 0 1")
    parser.add_argument('--forget_class_idx', type=int, nargs='+', default=[0],
                        help="list of class indices to forget, "
                             "used when forget_paradigm='class'. "
                             "Example: --forget_class_idx 3 7")
    parser.add_argument('--num_forget', type=int, default=10,
                        help="number of samples to forget, "
                             "used when forget_paradigm='sample'")
    parser.add_argument('--damping_factor', type=float, default=1e-2,
                        help="damping to make Hessian invertible (legacy HVP)")

    # -- SGN (Subspace Gauss-Newton) ------------------------------------------
    parser.add_argument('--warmup_rounds', type=int, default=30,
                        help="number of global rounds to run before caching "
                             "unlearning statistics. Model must converge "
                             "before s_k and C_k are meaningful")
    parser.add_argument('--subspace_dim', type=int, default=64,
                        help="subspace dimension r for SGN unlearning "
                             "(U in R^{d x r}). Smaller r is faster; "
                             "larger r is more accurate")
    parser.add_argument('--fusg_subspace', type=str, default='layer_name',
                        choices=['first_layer', 'layer_name'],
                        help="how to select the unlearning subspace: "
                             "'first_layer' = first named parameter block; "
                             "'layer_name'  = block specified by --fusg_layer_name "
                             "(supervisor recommends second layer)")
    parser.add_argument('--fusg_layer_name', type=str, default='',
                        help="name prefix of the layer whose parameters span "
                             "the unlearning subspace U, e.g. 'fc1' or 'layer2'. "
                             "Empty string = use the second named parameter block "
                             "in the model (supervisor advice). "
                             "Used when fusg_subspace='layer_name'")
    parser.add_argument('--gamma', type=float, default=1e-2,
                        help="Levenberg-Marquardt damping added to the "
                             "projected GGN: H_tilde_U = U^T G_ret U + gamma*I. "
                             "Ensures H_tilde_U is positive definite and "
                             "invertible even when G_ret is rank-deficient")

    # -------------------------------------------------------------------------
    # Differential Privacy
    # -------------------------------------------------------------------------
    parser.add_argument('--application', action='store_true',
                        help="enable DP noise (NoisedNetReturn), default: False")
    parser.add_argument('--std', type=float, default=0,
                        help="fixed noise std (0 = calibrate automatically)")
    parser.add_argument('--epsilon', type=float, default=1000,
                        help="DP privacy budget epsilon")
    parser.add_argument('--delta', type=float, default=1,
                        help="DP relaxation level delta")

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    parser.add_argument('--gpu', type=int, default=1,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--bs', type=int, default=2048,
                        help="test batch size")

    # -------------------------------------------------------------------------
    # Membership Inference Attack (MIA evaluation)
    # -------------------------------------------------------------------------
    parser.add_argument('--attack_model', type=str, default='LR',
                        help="attack model type: 'LR', 'MLP'")
    parser.add_argument('--method', type=str, default='direct_diff',
                        help="attack feature method: 'direct_diff', "
                             "'sorted_diff', 'direct_concat', "
                             "'sorted_concat', 'l2_distance'")

    args = parser.parse_args()
    return args