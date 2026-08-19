from __future__ import annotations
import copy
import json
from pathlib import Path
import yaml

OUT_DIR = Path("benchmark_configs/attack_only")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "MNIST": {
        "name": "mnist",
        "image_size": 28,
        "normalize": True,
        "augment": False,
        "num_clients": 20,
        "dirichlet_alpha": 1.0,
        "pathological_classes_per_client": 2,
        "anchor_size": 2048,
    },
    "CIFAR-10": {
        "name": "cifar10",
        "image_size": 32,
        "normalize": True,
        "augment": True,
        "num_clients": 20,
        "dirichlet_alpha": 1.0,
        "pathological_classes_per_client": 2,
        "anchor_size": 4096,
    },
    "CIFAR-100": {
        "name": "cifar100",
        "image_size": 32,
        "normalize": True,
        "augment": True,
        "num_clients": 20,
        "dirichlet_alpha": 1.0,
        "pathological_classes_per_client": 5,
        "anchor_size": 4096,
    },
    "SVHN": {
        "name": "svhn",
        "image_size": 32,
        "normalize": True,
        "augment": True,
        "num_clients": 20,
        "dirichlet_alpha": 1.0,
        "pathological_classes_per_client": 2,
        "anchor_size": 4096,
    },
    "GTSRB": {
        "name": "gtsrb",
        "image_size": 64,
        "normalize": True,
        "augment": True,
        "num_clients": 20,
        "dirichlet_alpha": 1.0,
        "pathological_classes_per_client": 5,
        "anchor_size": 4096,
    },
}

ARCHS = {
    "SimpleCNN": {"model": "simple_cnn", "lr": 0.01, "batch_size": 64},
    "ResNet-18": {"model": "resnet18", "lr": 0.003, "batch_size": 64},
    "MobileNetV3-Small": {"model": "mobilenet_v3_small", "lr": 0.003, "batch_size": 64},
    "EfficientNet-B0": {"model": "efficientnet_b0", "lr": 0.0025, "batch_size": 48},
    "ShuffleNetV2": {"model": "shufflenet_v2", "lr": 0.003, "batch_size": 64},
}

METHODS = {
    "FedAvg": {"method_name": "fedavg"},
    "TrimmedMean": {"method_name": "trimmed_mean", "trim_ratio": 0.2},
    "Krum": {"method_name": "krum"},
    "FLTrust": {"method_name": "fltrust"},
    "FedPARETO": {"method_name": "fedpareto"},
}

ATTACKS = {
    "sign_flip": {"name": "sign_flip", "malicious_fraction": 0.25},
    "gaussian": {"name": "gaussian", "malicious_fraction": 0.25, "std": 0.5},
    "badnets": {
        "name": "badnets",
        "malicious_fraction": 0.25,
        "target_label": 0,
        "poison_fraction": 0.30,
        "patch_value": 1.0,
        "trigger_size": 4,
    },
}

BASE = {
    "seed": 1,
    "device": "cuda",
    "dataset": {"root": "data/raw"},
    "partition": {"type": "dirichlet"},
    "federated": {
        "rounds": 500,
        "clients_per_round": 10,
        "local_epochs": 1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
    },
    "attack": {},
    "anchor": {"batch_size": 256, "server_root_epochs": 1},
    "fedpareto": {
        "objective_weights": {
            "accuracy": 0.50,
            "calibration": 0.20,
            "fairness": 0.10,
            "robustness": 0.20,
        },
        "temperature": 1.0,
        "reliability_momentum": 0.9,
        "trust_penalty": 0.10,
        "fairness_strength": 0.15,
        "entropy_reg": 0.25,
        "pareto_bonus": 0.05,
    },
    "evaluation": {"batch_size": 256, "ece_bins": 15},
    "output": {"root_dir": "runs"},
    "benchmark_meta": {
        "attack_only": True,
        "max_rounds": 500,
        "early_stopping_supported": False,
        "early_stopping_used": False,
        "early_stopping_patience_requested": 8,
        "notes": (
            "Repo-level standardized early stopping not enabled by default; "
            "benchmark uses fixed max rounds and skip-if-summary-exists resume semantics."
        ),
    },
}

rows = []
for attack_name, attack_cfg in ATTACKS.items():
    for arch_label, arch in ARCHS.items():
        for ds_label, ds in DATASETS.items():
            for method_label, method in METHODS.items():
                cfg = copy.deepcopy(BASE)
                cfg["experiment_name"] = f"{ds['name']}_{arch['model']}_{method['method_name']}_{attack_name}_seed1"
                cfg["dataset"].update({
                    "name": ds["name"],
                    "image_size": ds["image_size"],
                    "normalize": ds["normalize"],
                    "augment": ds["augment"],
                })
                cfg["partition"].update({
                    "num_clients": ds["num_clients"],
                    "dirichlet_alpha": ds["dirichlet_alpha"],
                    "pathological_classes_per_client": ds["pathological_classes_per_client"],
                })
                cfg["federated"].update({
                    "batch_size": arch["batch_size"],
                    "lr": arch["lr"],
                })
                cfg["model"] = {"name": arch["model"]}
                cfg["method"] = {"name": method["method_name"]}
                if "trim_ratio" in method:
                    cfg["method"]["trim_ratio"] = method["trim_ratio"]
                cfg["attack"] = copy.deepcopy(attack_cfg)
                cfg["anchor"]["size"] = ds["anchor_size"]
                cfg["benchmark_meta"].update({
                    "architecture_label": arch_label,
                    "dataset_label": ds_label,
                    "method_label": method_label,
                    "attack_label": attack_name,
                })
                fn = OUT_DIR / f"{cfg['experiment_name']}.yaml"
                with open(fn, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                rows.append({
                    "attack_config_name": attack_name,
                    "architecture": arch_label,
                    "dataset": ds_label,
                    "method": method_label,
                    "seed": cfg["seed"],
                    "config_path": str(fn),
                    "run_dir": f"runs/{cfg['experiment_name']}",
                })

manifest = Path("outputs/benchmark_results/attack_suite_manifest.json")
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"Generated {len(rows)} configs")
print(manifest)
