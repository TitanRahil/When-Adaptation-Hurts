from __future__ import annotations
import csv
import json
from pathlib import Path

MASTER_CSV = Path("outputs/benchmark_results_clean/clean_suite_master_log.csv")
OUT_DIR = Path("outputs/benchmark_results_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["FedAvg", "TrimmedMean", "Krum", "FLTrust", "FedPARETO"]
ARCH_ORDER = ["SimpleCNN", "ResNet-18", "MobileNetV3-Small", "EfficientNet-B0", "ShuffleNetV2"]
DATASET_ORDER = ["GTSRB", "SVHN", "MNIST", "CIFAR-10", "CIFAR-100"]

rows = list(csv.DictReader(MASTER_CSV.open("r", encoding="utf-8"))) if MASTER_CSV.exists() else []

best = {}
failures = []
for r in rows:
    key = (r["architecture"], r["dataset"], r["method"], r["seed"])
    if r["status"] == "ok":
        best[key] = r
    else:
        failures.append(r)

(OUT_DIR / "clean_suite_results.json").write_text(json.dumps(list(best.values()), indent=2), encoding="utf-8")

lines = []
lines.append("# Clean benchmark\n")
lines.append("| Architecture | Dataset | FedAvg | TrimmedMean | Krum | FLTrust | FedPARETO |")
lines.append("|--------------|---------|--------|-------------|------|---------|-----------|")
for arch in ARCH_ORDER:
    for ds in DATASET_ORDER:
        vals = []
        for method in METHOD_ORDER:
            key = (arch, ds, method, "1")
            r = best.get(key)
            if not r:
                vals.append("NA")
            else:
                try:
                    vals.append(f"{float(r['final_test_accuracy']):.4f}")
                except Exception:
                    vals.append("NA")
        lines.append(f"| {arch} | {ds} | " + " | ".join(vals) + " |")

(OUT_DIR / "table_clean.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT_DIR / "failed_or_incomplete.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
print("Wrote clean benchmark table and raw outputs to outputs/benchmark_results_clean")
