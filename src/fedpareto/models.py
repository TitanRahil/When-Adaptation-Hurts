import torch
import torch.nn as nn
from torchvision import models as tv_models


def _in_channels(dataset_name: str) -> int:
    return 1 if dataset_name.lower() in {"mnist", "fashionmnist", "kmnist", "emnist"} else 3


class ChannelRepeat(nn.Module):
    def forward(self, x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


class WrappedModel(nn.Module):
    def __init__(self, pre, model):
        super().__init__()
        self.pre = pre
        self.model = model

    def forward(self, x):
        return self.model(self.pre(x))


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    def __init__(self, in_features=28 * 28, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_resnet18(in_channels: int, num_classes: int, image_size: int):
    model = tv_models.resnet18(weights=None)
    if image_size <= 64:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_mobilenet_v3_small(in_channels: int, num_classes: int):
    model = tv_models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    if in_channels == 1:
        return WrappedModel(ChannelRepeat(), model)
    return model


def make_efficientnet_b0(in_channels: int, num_classes: int):
    model = tv_models.efficientnet_b0(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    if in_channels == 1:
        return WrappedModel(ChannelRepeat(), model)
    return model


def make_shufflenet_v2(in_channels: int, num_classes: int):
    model = tv_models.shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if in_channels == 1:
        return WrappedModel(ChannelRepeat(), model)
    return model


def get_model(name: str, dataset_name: str, num_classes: int, image_size: int):
    name = name.lower()
    in_channels = _in_channels(dataset_name)

    if name == "simple_cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    if name == "mlp":
        in_features = in_channels * image_size * image_size
        return MLP(in_features=in_features, num_classes=num_classes)

    if name == "resnet18":
        return make_resnet18(in_channels, num_classes, image_size)

    if name == "mobilenet_v3_small":
        return make_mobilenet_v3_small(in_channels, num_classes)

    if name == "efficientnet_b0":
        return make_efficientnet_b0(in_channels, num_classes)

    if name == "shufflenet_v2":
        return make_shufflenet_v2(in_channels, num_classes)

    raise ValueError(f"Unsupported model: {name}")