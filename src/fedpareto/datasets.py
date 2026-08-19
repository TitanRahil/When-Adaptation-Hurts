from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from fedpareto.partition import dirichlet_partition, pathological_partition


def _stats(name: str):
    name = name.lower()
    stats = {
        "mnist": ((0.1307,), (0.3081,)),
        "fashionmnist": ((0.5,), (0.5,)),
        "kmnist": ((0.5,), (0.5,)),
        "emnist": ((0.5,), (0.5,)),
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        "gtsrb": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    }
    return stats[name]


def get_transforms(name: str, image_size: int, normalize: bool, train: bool, augment: bool):
    from torchvision import transforms

    tfms = []
    if image_size is not None:
        tfms.append(transforms.Resize((image_size, image_size)))

    color_datasets = {"cifar10", "cifar100", "svhn", "gtsrb"}
    if train and augment and name in color_datasets:
        tfms += [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

    tfms.append(transforms.ToTensor())

    if normalize:
        mean, std = _stats(name)
        tfms.append(transforms.Normalize(mean, std))

    return transforms.Compose(tfms)


@dataclass
class FederatedDataBundle:
    train_dataset: Dataset
    test_dataset: Dataset
    anchor_dataset: Dataset
    root_dataset: Dataset
    client_train_subsets: Dict[int, Dataset]
    client_eval_subsets: Dict[int, Dataset]
    num_classes: int


def _num_classes(cfg) -> int:
    name = cfg["dataset"]["name"].lower()
    if name in {"mnist", "fashionmnist", "kmnist", "svhn", "cifar10"}:
        return 10
    if name == "cifar100":
        return 100
    if name == "gtsrb":
        return 43
    if name == "emnist":
        split = cfg["dataset"].get("emnist_split", "balanced").lower()
        return {
            "byclass": 62,
            "bymerge": 47,
            "balanced": 47,
            "letters": 26,
            "digits": 10,
            "mnist": 10,
        }[split]
    raise ValueError(name)


def _targets(dataset):
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return np.array(t if isinstance(t, list) else t)
    if hasattr(dataset, "labels"):
        t = dataset.labels
        return np.array(t if isinstance(t, list) else t)
    if hasattr(dataset, "_samples"):
        return np.array([y for _, y in dataset._samples])
    if hasattr(dataset, "samples"):
        return np.array([y for _, y in dataset.samples])
    raise AttributeError("Dataset has no targets/labels/samples attribute")


def _build_dataset(name: str, root: Path, train: bool, transform, cfg):
    from torchvision import datasets as tv_datasets

    name = name.lower()

    if name == "svhn":
        return tv_datasets.SVHN(
            root=root,
            split="train" if train else "test",
            download=True,
            transform=transform,
        )

    if name == "gtsrb":
        return tv_datasets.GTSRB(
            root=root,
            split="train" if train else "test",
            download=True,
            transform=transform,
        )

    if name == "emnist":
        return tv_datasets.EMNIST(
            root=root,
            split=cfg["dataset"].get("emnist_split", "balanced"),
            train=train,
            download=True,
            transform=transform,
        )

    registry = {
        "mnist": tv_datasets.MNIST,
        "fashionmnist": tv_datasets.FashionMNIST,
        "kmnist": tv_datasets.KMNIST,
        "cifar10": tv_datasets.CIFAR10,
        "cifar100": tv_datasets.CIFAR100,
    }

    ds_cls = registry[name]
    return ds_cls(root=root, train=train, download=True, transform=transform)


def build_federated_data(cfg) -> FederatedDataBundle:
    name = cfg["dataset"]["name"].lower()
    root = Path(cfg["dataset"]["root"])
    image_size = int(cfg["dataset"].get("image_size", 32))
    normalize = bool(cfg["dataset"].get("normalize", True))
    augment = bool(cfg["dataset"].get("augment", False))

    train_transform = get_transforms(name, image_size, normalize, train=True, augment=augment)
    test_transform = get_transforms(name, image_size, normalize, train=False, augment=False)

    train_dataset = _build_dataset(name, root, True, train_transform, cfg)
    test_dataset = _build_dataset(name, root, False, test_transform, cfg)

    targets = _targets(train_dataset)
    num_clients = cfg["partition"]["num_clients"]
    part_type = cfg["partition"]["type"]

    if part_type == "dirichlet":
        client_indices = dirichlet_partition(
            targets, num_clients, cfg["partition"]["dirichlet_alpha"], cfg["seed"]
        )
    elif part_type == "pathological":
        client_indices = pathological_partition(
            targets, num_clients, cfg["partition"]["pathological_classes_per_client"], cfg["seed"]
        )
    else:
        raise ValueError(f"Unsupported partition type: {part_type}")

    rng = np.random.default_rng(cfg["seed"])
    anchor_size = int(cfg["anchor"]["size"])
    all_indices = np.arange(len(train_dataset))
    rng.shuffle(all_indices)

    anchor_idx = all_indices[:anchor_size].tolist()
    root_idx = all_indices[anchor_size: anchor_size + min(anchor_size, 256)].tolist()

    anchor_dataset = Subset(train_dataset, anchor_idx)
    root_dataset = Subset(train_dataset, root_idx)

    anchor_set = set(anchor_idx) | set(root_idx)
    client_train_subsets = {}
    client_eval_subsets = {}

    for cid, idxs in enumerate(client_indices):
        idxs = [i for i in idxs if i not in anchor_set]
        if len(idxs) < 8:
            idxs = idxs + idxs
        cut = max(1, int(0.8 * len(idxs)))
        train_ids = idxs[:cut]
        eval_ids = idxs[cut:] if cut < len(idxs) else idxs[: max(1, len(idxs) // 5)]
        client_train_subsets[cid] = Subset(train_dataset, train_ids)
        client_eval_subsets[cid] = Subset(train_dataset, eval_ids)

    return FederatedDataBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        anchor_dataset=anchor_dataset,
        root_dataset=root_dataset,
        client_train_subsets=client_train_subsets,
        client_eval_subsets=client_eval_subsets,
        num_classes=_num_classes(cfg),
    )


def build_loaders(bundle: FederatedDataBundle, cfg):
    bs = cfg["federated"]["batch_size"]
    eval_bs = cfg["evaluation"]["batch_size"]

    client_train_loaders = {
        cid: DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
        for cid, ds in bundle.client_train_subsets.items()
    }
    client_eval_loaders = {
        cid: DataLoader(ds, batch_size=eval_bs, shuffle=False, num_workers=2, pin_memory=True)
        for cid, ds in bundle.client_eval_subsets.items()
    }
    anchor_loader = DataLoader(
        bundle.anchor_dataset,
        batch_size=cfg["anchor"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    root_loader = DataLoader(
        bundle.root_dataset,
        batch_size=cfg["anchor"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return client_train_loaders, client_eval_loaders, anchor_loader, root_loader, test_loader