from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

MANIFEST_PATH = Path("outputs/benchmark_results_clean/clean_suite_manifest.json")
MASTER_JSONL = Path("outputs/benchmark_results_clean/clean_suite_master_log.jsonl")
MASTER_CSV = Path("outputs/benchmark_results_clean/clean_suite_master_log.csv")
LOG_DIR = Path("outputs/benchmark_logs_clean")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = [
    "timestamp", "status", "attack_config_name", "architecture", "dataset", "method", "seed",
    "config_path", "run_dir", "max_rounds", "actual_stopped_round", "early_stopping_used",
    "best_validation_metric", "final_validation_metric", "best_test_accuracy",
    "final_test_accuracy", "attack_success_rate", "total_runtime", "notes_errors"
]

def append_csv(row):
    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = MASTER_CSV.exists()
    with open(MASTER_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})

def append_jsonl(obj):
    MASTER_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(MASTER_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def load_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

def read_summary(run_dir: Path):
    p = run_dir / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def read_metrics(run_dir: Path):
    p = run_dir / "metrics_round.csv"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows

def classify_needed(manifest_row, force=False):
    run_dir = Path(manifest_row["run_dir"])
    if force:
        return True
    summary = read_summary(run_dir)
    return summary is None

def do_dry_run(rows, force=False):
    pending, completed = [], []
    for r in rows:
        (pending if classify_needed(r, force=force) else completed).append(r)
    report = {
        "total": len(rows),
        "pending": len(pending),
        "completed": len(completed),
        "pending_examples": pending[:10],
    }
    out = Path("outputs/benchmark_results_clean/clean_suite_dry_run.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

def run_one(row, python_bin=sys.executable):
    cfg = row["config_path"]
    run_dir = Path(row["run_dir"])
    log_path = LOG_DIR / (Path(cfg).stem + ".log")
    cmd = [python_bin, "scripts/run_experiment.py", "--config", cfg]

    start = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write("\n===== START {} =====\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        logf.write("COMMAND: {}\n".format(" ".join(cmd)))
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    summary = read_summary(run_dir)
    metrics = read_metrics(run_dir)
    actual_round = len(metrics) if metrics else None
    best_val = None
    final_val = None
    if metrics:
        if "worst_client_accuracy" in metrics[0]:
            vals = [float(x["worst_client_accuracy"]) for x in metrics]
            best_val = max(vals)
            final_val = vals[-1]
        elif "test_accuracy" in metrics[0]:
            vals = [float(x["test_accuracy"]) for x in metrics]
            best_val = max(vals)
            final_val = vals[-1]

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ok" if proc.returncode == 0 and summary else "failed",
        "attack_config_name": "clean",
        "architecture": row["architecture"],
        "dataset": row["dataset"],
        "method": row["method"],
        "seed": row["seed"],
        "config_path": cfg,
        "run_dir": row["run_dir"],
        "max_rounds": 500,
        "actual_stopped_round": actual_round,
        "early_stopping_used": False,
        "best_validation_metric": best_val,
        "final_validation_metric": final_val,
        "best_test_accuracy": summary.get("best_test_accuracy") if summary else None,
        "final_test_accuracy": summary.get("final_test_accuracy") if summary else None,
        "attack_success_rate": summary.get("final_attack_success_rate") if summary else None,
        "total_runtime": summary.get("total_runtime_sec") if summary else elapsed,
        "notes_errors": "" if proc.returncode == 0 else f"return_code={proc.returncode}; see {log_path}",
    }
    append_jsonl(result)
    append_csv(result)
    print(json.dumps(result, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--python-bin", default=sys.executable)
    args = ap.parse_args()

    rows = load_manifest()
    if args.dry_run:
        do_dry_run(rows, force=args.force)
        return

    for row in rows:
        if not classify_needed(row, force=args.force):
            continue
        try:
            run_one(row, python_bin=args.python_bin)
        except Exception as e:
            fail = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "failed",
                "attack_config_name": "clean",
                "architecture": row["architecture"],
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": row["seed"],
                "config_path": row["config_path"],
                "run_dir": row["run_dir"],
                "max_rounds": 500,
                "actual_stopped_round": None,
                "early_stopping_used": False,
                "best_validation_metric": None,
                "final_validation_metric": None,
                "best_test_accuracy": None,
                "final_test_accuracy": None,
                "attack_success_rate": None,
                "total_runtime": None,
                "notes_errors": repr(e),
            }
            append_jsonl(fail)
            append_csv(fail)
            print(json.dumps(fail, indent=2))

if __name__ == "__main__":
    main()
