import argparse
from pathlib import Path
import json
import pandas as pd

def collect_summaries(runs_root: Path):
    rows = []
    for summary_path in runs_root.rglob("summary.json"):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(data)
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    df = collect_summaries(runs_root)
    if df.empty:
        raise SystemExit("No summary.json files found.")

    keep = [
        c for c in [
            "experiment_name",
            "method",
            "attack",
            "final_test_accuracy",
            "best_test_accuracy",
            "final_test_ece",
            "final_worst_client_accuracy",
            "final_attack_success_rate",
            "total_runtime_sec",
        ] if c in df.columns
    ]
    df = df[keep].sort_values(by=["method", "experiment_name"])

    latex = df.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved LaTeX: {out_path}")
    print(f"Saved CSV: {csv_path}")

if __name__ == "__main__":
    main()
