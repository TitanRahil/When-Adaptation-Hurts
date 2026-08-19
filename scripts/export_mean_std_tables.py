import argparse
import json
import re
from pathlib import Path

import pandas as pd


SEED_PATTERN = re.compile(r"(.*)_seed(\d+)(.*)$")


def canonical_name(exp_name: str) -> str:
    """
    Convert names like:
      mnist_fedpareto_seed1
      mnist_fedpareto_seed2_ablation_no_fairness
    into:
      mnist_fedpareto
      mnist_fedpareto_ablation_no_fairness
    """
    m = SEED_PATTERN.match(exp_name)
    if not m:
        return exp_name
    prefix, _seed_num, suffix = m.groups()
    return f"{prefix}{suffix}"


def collect_summaries(runs_root: Path) -> pd.DataFrame:
    rows = []
    for summary_path in runs_root.rglob("summary.json"):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_summary_path"] = str(summary_path)
        data["_group_name"] = canonical_name(data.get("experiment_name", summary_path.parent.name))
        rows.append(data)
    return pd.DataFrame(rows)


def flatten_stats(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    agg = {}
    for col in metric_cols:
        agg[col] = ["mean", "std", "count"]

    grouped = df.groupby(["_group_name", "method", "attack"], dropna=False).agg(agg)
    grouped.columns = ["_".join(x) for x in grouped.columns]
    grouped = grouped.reset_index()

    # Optional pretty mean±std strings for LaTeX/paper tables
    for col in metric_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        pretty_col = f"{col}_mean_std"
        grouped[pretty_col] = grouped.apply(
            lambda r: f"{r[mean_col]:.4f} $\\pm$ {0.0 if pd.isna(r[std_col]) else r[std_col]:.4f}",
            axis=1,
        )
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate summary.json files across seeds and export mean±std tables."
    )
    parser.add_argument("--runs_root", required=True, type=str)
    parser.add_argument("--output", required=True, type=str, help="Output LaTeX path")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    df = collect_summaries(runs_root)
    if df.empty:
        raise SystemExit("No summary.json files found under the given runs_root.")

    metric_candidates = [
        "final_test_accuracy",
        "best_test_accuracy",
        "final_test_ece",
        "final_worst_client_accuracy",
        "final_attack_success_rate",
        "total_runtime_sec",
    ]
    metric_cols = [c for c in metric_candidates if c in df.columns]

    grouped = flatten_stats(df, metric_cols)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_path.with_suffix(".csv")
    grouped.to_csv(csv_path, index=False)

    # LaTeX table with pretty mean±std columns only
    pretty_keep = ["_group_name", "method", "attack"] + [f"{c}_mean_std" for c in metric_cols]
    latex_df = grouped[pretty_keep].copy()
    latex_df = latex_df.rename(columns={"_group_name": "experiment_group"})

    latex = latex_df.to_latex(index=False, escape=False)
    out_path.write_text(latex, encoding="utf-8")

    print(f"Saved aggregated CSV: {csv_path}")
    print(f"Saved aggregated LaTeX: {out_path}")


if __name__ == "__main__":
    main()
