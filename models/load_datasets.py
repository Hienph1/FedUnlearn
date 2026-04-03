import math
import numpy as np
import torch
from torchvision import datasets, transforms
from utils.options import args_parser
from utils.subset import reduce_dataset_size


def load_dataset():
    args = args_parser()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu'
    )

    # ── Load MNIST ─────────────────────────────────────────────────────────
    # PyTorch expects: <root>/MNIST/raw/<files>
    # Cấu trúc hiện tại:  ./data/mnist/MNIST/raw/<files>  => root='./data/mnist'
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_train = datasets.MNIST(
        root='./data', train=True, download=False, transform=trans
    )
    dataset_test = datasets.MNIST(
        root='./data', train=False, download=False, transform=trans
    )
    args.num_channels = 1
    # ───────────────────────────────────────────────────────────────────────

    dataset_train = reduce_dataset_size(dataset_train, args.num_dataset, random_seed=args.seed)
    testsize      = math.floor(args.num_dataset * args.test_train_rate)
    dataset_test  = reduce_dataset_size(dataset_test, testsize, random_seed=args.seed)

    print(f'[load_dataset] Train size : {len(dataset_train)}')
    print(f'[load_dataset] Test  size : {len(dataset_test)}')

    return dataset_train, dataset_test, args.num_classes