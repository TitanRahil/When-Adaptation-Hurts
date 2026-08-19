from __future__ import annotations
import importlib
import inspect
import json
from pathlib import Path

REPORT = {
    "ok": True,
    "checks": [],
    "warnings": [],
    "errors": [],
}

def add_check(name, ok, details=""):
    REPORT["checks"].append({"name": name, "ok": bool(ok), "details": details})
    if not ok:
        REPORT["ok"] = False

try:
    datasets_mod = importlib.import_module("fedpareto.datasets")
    models_mod = importlib.import_module("fedpareto.models")
    attacks_mod = importlib.import_module("fedpareto.attacks")
    utils_mod = importlib.import_module("fedpareto.utils")
    add_check("import_modules", True, "Imported fedpareto datasets/models/attacks/utils")
except Exception as e:
    add_check("import_modules", False, repr(e))
    print(json.dumps(REPORT, indent=2))
    raise SystemExit(1)

source_datasets = inspect.getsource(datasets_mod).lower()
for ds in ["mnist", "svhn", "gtsrb", "cifar100", "cifar10"]:
    add_check(f"dataset_support_{ds}", ds in source_datasets, f"Search term '{ds}' in fedpareto.datasets")

source_models = inspect.getsource(models_mod).lower()
for m in ["simple_cnn", "resnet18", "mobilenet_v3_small", "efficientnet_b0", "shufflenet_v2"]:
    add_check(f"model_support_{m}", m in source_models, f"Search term '{m}' in fedpareto.models")

source_attacks = inspect.getsource(attacks_mod)
attack_safe = "is_floating_point" in source_attacks
add_check("attack_nonfloat_guard", attack_safe, "apply_model_attack should guard non-floating tensors")

source_utils = inspect.getsource(utils_mod)
vector_safe = "is_floating_point" in source_utils
add_check("utils_nonfloat_guard", vector_safe, "flatten/vector helpers should guard non-floating tensors")

try:
    from fedpareto.experiments import ExperimentRunner
    add_check("experiment_runner_import", True, "ExperimentRunner import ok")
except Exception as e:
    add_check("experiment_runner_import", False, repr(e))

REPORT["warnings"].append(
    "The current repo as shared earlier in chat does not implement standardized early stopping/resume in ExperimentRunner. "
    "This harness records early_stopping_used=false and uses skip-if-summary-exists resume semantics at the benchmark level."
)

out = Path("outputs/benchmark_results/preflight_report.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
print(json.dumps(REPORT, indent=2))
