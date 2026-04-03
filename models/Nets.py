import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
from contextlib import contextmanager

class SubspaceMixin:

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_subspace_params(self, subspace: str = "first"):
        
        names = set(self._subspace_param_names(subspace))
        return [(n, p) for n, p in self.named_parameters() if n in names]

    @contextmanager
    def freeze_outside_subspace(self, subspace: str = "first"):
        
        subspace_names = set(self._subspace_param_names(subspace))

        # Save original state
        saved = {n: p.requires_grad for n, p in self.named_parameters()}

        # Freeze outside subspace
        for n, p in self.named_parameters():
            if n not in subspace_names:
                p.requires_grad_(False)

        try:
            yield self
        finally:
            # Restore original state
            for n, p in self.named_parameters():
                p.requires_grad_(saved[n])

    def _subspace_param_names(self, subspace: str):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _subspace_param_names()"
        )


# ===========================================================================
# Model definitions
# ===========================================================================

class Logistic(SubspaceMixin, nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.float()
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        x = self.layer(x)
        return x

    def _subspace_param_names(self, subspace: str):
        # Single layer: it is both the feature extractor and the head.
        if subspace in ("first", "last", "all"):
            return [n for n, _ in self.named_parameters()]
        raise ValueError(f"Unknown subspace '{subspace}'")


class MLP(SubspaceMixin, nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.float()
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

    def _subspace_param_names(self, subspace: str):
        mapping = {
            # "first" = input feature projection (sensitive representation)
            "first": ["layer_input.weight", "layer_input.bias"],
            # "last"  = classification head
            "last":  ["layer_hidden.weight", "layer_hidden.bias"],
            # "all"   = every parameter
            "all":   [n for n, _ in self.named_parameters()],
        }
        if subspace not in mapping:
            raise ValueError(f"Unknown subspace '{subspace}'. Choose from {list(mapping)}")
        return mapping[subspace]


class CNNMnist(SubspaceMixin, nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # output cố định 4×4
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = x.float()
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.adaptive_pool(x)          # → [B, 20, 4, 4] bất kể ảnh size
        x = x.view(x.size(0), -1)         # → [B, 320]
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def _subspace_param_names(self, subspace: str):
        mapping = {
            # First conv layer = raw feature detectors (most sensitive for FU)
            "first": ["conv1.weight", "conv1.bias"],
            # Classification head
            "last":  ["fc2.weight", "fc2.bias"],
            "all":   [n for n, _ in self.named_parameters()],
        }
        if subspace not in mapping:
            raise ValueError(f"Unknown subspace '{subspace}'")
        return mapping[subspace]


class Swish(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + " ()"


class FashionCNN4(SubspaceMixin, nn.Module):
    def __init__(self):
        super(FashionCNN4, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 5)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv1_drop = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3)
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv2_drop = nn.Dropout(0.25)

        self.fc1 = nn.Linear(576, 256)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.float()
        out = self.conv1(x); out = self.relu1(out); out = self.batch1(out)
        out = self.conv2(out); out = self.relu2(out); out = self.batch2(out)
        out = self.maxpool1(out); out = self.conv1_drop(out)

        out = self.conv3(out); out = self.relu3(out); out = self.batch3(out)
        out = self.conv4(out); out = self.relu4(out); out = self.batch4(out)
        out = self.maxpool2(out); out = self.conv2_drop(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out); out = self.fc1_relu(out); out = self.dp1(out)
        out = self.fc2(out)
        return out

    def _subspace_param_names(self, subspace: str):
        # "first" block: conv1 + conv2 (first feature extraction stage)
        first_block = [
            "conv1.weight", "conv1.bias",
            "batch1.weight", "batch1.bias",
            "conv2.weight", "conv2.bias",
            "batch2.weight", "batch2.bias",
        ]
        mapping = {
            "first": first_block,
            "last":  ["fc2.weight", "fc2.bias"],
            "all":   [n for n, _ in self.named_parameters()],
        }
        if subspace not in mapping:
            raise ValueError(f"Unknown subspace '{subspace}'")
        return mapping[subspace]


class LeNet(SubspaceMixin, nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ("Conv1", nn.Conv2d(1, 6, (5, 5), padding=2)),
            ("Relu1", nn.ReLU()),
            ("Pool1", nn.MaxPool2d((2, 2), stride=2)),
        ]))
        self.c2 = nn.Sequential(OrderedDict([
            ("Conv2", nn.Conv2d(6, 16, (5, 5))),
            ("Relu2", nn.ReLU()),
            ("Pool2", nn.MaxPool2d((2, 2), stride=2)),
        ]))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))  # output cố định 5×5 → 400
        self.c3 = nn.Sequential(OrderedDict([
            ("FullCon3", nn.Linear(400, 120)),
            ("Relu3", nn.ReLU()),
        ]))
        self.c4 = nn.Sequential(OrderedDict([
            ("FullCon4", nn.Linear(120, 84)),
            ("Relu4", nn.ReLU()),
        ]))
        self.c5 = nn.Sequential(OrderedDict([
            ("FullCon5", nn.Linear(84, 10)),
            ("Sig5", nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, img):
        img = img.float()
        output = self.c1(img)
        output = self.c2(output)
        output = self.adaptive_pool(output)    # → [B, 16, 5, 5] bất kể ảnh size
        output = output.view(output.size(0), -1)  # → [B, 400]
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)
        return output

    def _subspace_param_names(self, subspace: str):
        mapping = {
            # c1 = first conv stage (most sensitive low-level features)
            "first": ["c1.Conv1.weight", "c1.Conv1.bias"],
            # c5 = classification head (LogSoftmax layer)
            "last":  ["c5.FullCon5.weight", "c5.FullCon5.bias"],
            "all":   [n for n, _ in self.named_parameters()],
        }
        if subspace not in mapping:
            raise ValueError(f"Unknown subspace '{subspace}'")
        return mapping[subspace]


class CNNCifar(SubspaceMixin, nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # output cố định 4×4 → 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, args.num_classes)

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)          # → [B, 64, 4, 4] bất kể ảnh size
        x = x.view(x.size(0), -1)         # → [B, 1024]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _subspace_param_names(self, subspace: str):
        mapping = {
            "first": ["conv1.weight", "conv1.bias"],
            "last":  ["fc3.weight", "fc3.bias"],
            "all":   [n for n, _ in self.named_parameters()],
        }
        if subspace not in mapping:
            raise ValueError(f"Unknown subspace '{subspace}'")
        return mapping[subspace]


# ---------------------------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(SubspaceMixin, nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        in_dim = 1 if grayscale else 3
        super(ResNet, self).__init__()

        # Stem
        self.conv1 = nn.Conv2d(in_dim, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.fc = nn.Linear(25088 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

    def _subspace_param_names(self, subspace: str):
        # "first" = stem conv + bn  (low-level feature extractor – FUSG default)
        first_names = ["conv1.weight", "bn1.weight", "bn1.bias"]
        # "last"  = linear classification head
        last_names  = ["fc.weight", "fc.bias"]

        mapping = {
            "first": first_names,
            "last":  last_names,
            # Both stem + head: a practical middle ground
            "first+last": first_names + last_names,
            "all":   [n for n, _ in self.named_parameters()],
        }
        if subspace not in mapping:
            raise ValueError(
                f"Unknown subspace '{subspace}'. Choose from {list(mapping)}"
            )
        return mapping[subspace]


def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, grayscale=False)


# ===========================================================================
# Quick sanity-check (run `python Nets.py` to verify)
# ===========================================================================

if __name__ == "__main__":
    import types

    print("=" * 60)
    print("FUSG subspace sanity checks")
    print("=" * 60)

    # --- ResNet-18 ---
    model = resnet18(num_classes=10)
    model.eval()

    for sp in ("first", "last", "first+last", "all"):
        params = model.get_subspace_params(sp)
        total = sum(p.numel() for _, p in params)
        print(f"ResNet-18 | subspace='{sp:10s}' | #tensors={len(params):3d} | #params={total:,}")

    # Verify freeze context manager does NOT break forward pass
    x = torch.randn(2, 3, 32, 32)
    with model.freeze_outside_subspace("first"):
        out = model(x)
        # Check: parameters outside subspace have requires_grad=False
        frozen = [n for n, p in model.named_parameters()
                  if not p.requires_grad and n not in
                  {"conv1.weight", "bn1.weight", "bn1.bias"}]
        assert len(frozen) > 0, "Nothing was frozen!"

    # After context: all params restored
    all_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(all_grad) == len(list(model.parameters())), "requires_grad not restored!"

    print("\nForward pass inside freeze_outside_subspace('first'): OK")
    print("requires_grad fully restored after context:          OK")
    print("Output shape:", out.shape)

    # --- LeNet ---
    lenet = LeNet()
    for sp in ("first", "last", "all"):
        params = lenet.get_subspace_params(sp)
        print(f"LeNet     | subspace='{sp:5s}' | #tensors={len(params)} | "
              f"#params={sum(p.numel() for _, p in params):,}")