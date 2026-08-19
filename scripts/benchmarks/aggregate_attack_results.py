from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

MASTER_CSV = Path("outputs/benchmark_results/attack_suite_master_log.csv")
OUT_DIR = Path("outputs/benchmark_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["FedAvg", "TrimmedMean", "Krum", "FLTrust", "FedPARETO"]
ARCH_ORDER = ["SimpleCNN", "ResNet-18", "MobileNetV3-Small", "EfficientNet-B0", "ShuffleNetV2"]
DATASET_ORDER = ["GTSRB", "SVHN", "MNIST", "CIFAR-10", "CIFAR-100"]
ATTACK_ORDER = ["sign_flip", "gaussian", "badnets"]

rows = list(csv.DictReader(MASTER_CSV.open("r", encoding="utf-8"))) if MASTER_CSV.exists() else []

best = {}
failures = []
for r in rows:
    key = (r["attack_config_name"], r["architecture"], r["dataset"], r["method"], r["seed"])
    if r["status"] == "ok":
        best[key] = r
    else:
        failures.append(r)

(OUT_DIR / "attack_suite_results.json").write_text(json.dumps(list(best.values()), indent=2), encoding="utf-8")

for attack in ATTACK_ORDER:
    lines = []
    lines.append(f"# Attack: {attack}\n")
    lines.append("| Architecture | Dataset | FedAvg | TrimmedMean | Krum | FLTrust | FedPARETO |")
    lines.append("|--------------|---------|--------|-------------|------|---------|-----------|")
    for arch in ARCH_ORDER:
        for ds in DATASET_ORDER:
            vals = []
            for method in METHOD_ORDER:
                key = (attack, arch, ds, method, "1")
                r = best.get(key)
                if not r:
                    vals.append("NA")
                else:
                    acc = r.get("final_test_accuracy_under_attack")
                    asr = r.get("attack_success_rate")
                    try:
                        accf = float(acc)
                        if asr not in ("", None):
                            asrf = float(asr)
                            vals.append(f"{accf:.4f} / ASR {asrf:.4f}")
                        else:
                            vals.append(f"{accf:.4f}")
                    except Exception:
                        vals.append("NA")
            lines.append(f"| {arch} | {ds} | " + " | ".join(vals) + " |")
    (OUT_DIR / f"table_{attack}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

summary = defaultdict(dict)
for attack in ATTACK_ORDER:
    for arch in ARCH_ORDER:
        for ds in DATASET_ORDER:
            cand = []
            for method in METHOD_ORDER:
                key = (attack, arch, ds, method, "1")
                r = best.get(key)
                if r and r.get("final_test_accuracy_under_attack") not in (None, ""):
                    cand.append((float(r["final_test_accuracy_under_attack"]), method, r))
            if cand:
                cand.sort(reverse=True)
                summary[attack][f"{arch} / {ds}"] = {
                    "best_method": cand[0][1],
                    "final_test_accuracy_under_attack": cand[0][0]
                }

(OUT_DIR / "best_method_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(OUT_DIR / "failed_or_incomplete.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
print("Wrote markdown tables, raw json, best-method summary, and failure log to outputs/benchmark_results")
