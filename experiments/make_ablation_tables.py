#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_KEY_COLUMNS = [
    "benchmark_name",
    "mutation_type",
    "burden",
    "removed_signature",
    "removal_selection_groups",
]

DEFAULT_METRICS = [
    "sample_f1",
    "sample_jaccard",
    "exposure_tvd",
    "exposure_cosine",
    "reconstruction_cosine",
    "catalog_insufficiency_auroc",
    "catalog_insufficiency_auprc",
    "catalog_insufficiency_probability_ece",
    "catalog_insufficiency_probability_brier",
    "assignment_confidence_ece",
    "exposure_interval_coverage",
    "exposure_interval_mean_width",
]

LOWER_IS_BETTER = {
    "exposure_tvd",
    "catalog_insufficiency_probability_ece",
    "catalog_insufficiency_probability_brier",
    "assignment_confidence_ece",
    "exposure_interval_mean_width",
}


def _step_label(step_name: str) -> str:
    marker = "_ablation_"
    if marker in step_name:
        return step_name.split(marker, 1)[1]
    return step_name


def _available_columns(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def _metric_direction(metric_name: str) -> str:
    return "lower_is_better" if metric_name in LOWER_IS_BETTER else "higher_is_better"


def make_ablation_tables(
    root: Path,
    output_dir: Path,
    *,
    baseline_step: str,
    target_expert: str = "rule_fusion",
    metrics: list[str] | None = None,
    key_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_path = root / "metrics" / "aggregate_metrics.tsv"
    if not aggregate_path.exists():
        raise SystemExit(f"Missing aggregate metrics: {aggregate_path}")
    aggregate = pd.read_csv(aggregate_path, sep="\t")
    if "step_name" not in aggregate.columns or "expert_name" not in aggregate.columns:
        raise SystemExit("aggregate_metrics.tsv must include step_name and expert_name columns.")

    selected = aggregate.loc[aggregate["expert_name"].astype(str) == target_expert].copy()
    baseline = selected.loc[selected["step_name"].astype(str) == baseline_step].copy()
    comparisons = selected.loc[selected["step_name"].astype(str) != baseline_step].copy()
    if baseline.empty:
        raise SystemExit(f"Baseline step not found for expert {target_expert}: {baseline_step}")
    if comparisons.empty:
        raise SystemExit(f"No comparison rows found for expert {target_expert}.")

    available_keys = _available_columns(aggregate, key_columns or DEFAULT_KEY_COLUMNS)
    available_metrics = _available_columns(aggregate, metrics or DEFAULT_METRICS)
    if not available_keys:
        raise SystemExit("No shared key columns were available for ablation comparison.")
    if not available_metrics:
        raise SystemExit("No requested metric columns were available for ablation comparison.")

    baseline = baseline.loc[:, available_keys + available_metrics].copy()
    baseline = baseline.rename(columns={metric: f"{metric}__baseline" for metric in available_metrics})
    merged = comparisons.merge(baseline, on=available_keys, how="inner")

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        for metric in available_metrics:
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            baseline_value = pd.to_numeric(pd.Series([row.get(f"{metric}__baseline")]), errors="coerce").iloc[0]
            if pd.isna(value) or pd.isna(baseline_value):
                continue
            direction = _metric_direction(metric)
            delta = float(value - baseline_value)
            improvement = float(baseline_value - value) if direction == "lower_is_better" else delta
            rows.append(
                {
                    "baseline_step": baseline_step,
                    "comparison_step": row["step_name"],
                    "baseline_label": _step_label(baseline_step),
                    "comparison_label": _step_label(str(row["step_name"])),
                    "target_expert": target_expert,
                    **{key: row.get(key) for key in available_keys},
                    "metric_name": metric,
                    "metric_direction": direction,
                    "baseline_value": float(baseline_value),
                    "comparison_value": float(value),
                    "delta_vs_baseline": delta,
                    "improvement_vs_baseline": improvement,
                }
            )

    deltas = pd.DataFrame.from_records(rows)
    if deltas.empty:
        raise SystemExit("Ablation comparison produced no numeric metric rows.")

    summary_group_columns = ["baseline_label", "comparison_label", "target_expert", "metric_name", "metric_direction"]
    summary = (
        deltas.groupby(summary_group_columns, dropna=False)
        .agg(
            mean_delta_vs_baseline=("delta_vs_baseline", "mean"),
            mean_improvement_vs_baseline=("improvement_vs_baseline", "mean"),
            slice_count=("delta_vs_baseline", "size"),
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(output_dir / "ablation_metric_deltas.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / "ablation_summary.tsv", sep="\t", index=False)
    return deltas, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paper-ready ablation delta tables from collected metrics.")
    parser.add_argument("root", help="Paper-suite output root.")
    parser.add_argument("--baseline-step", required=True)
    parser.add_argument("--target-expert", default="rule_fusion")
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/tables.")
    parser.add_argument("--metrics", default=None, help="Optional comma-separated metric list.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()] if args.metrics else None
    output_dir = Path(args.output_dir) if args.output_dir else root / "tables"
    make_ablation_tables(
        root,
        output_dir,
        baseline_step=args.baseline_step,
        target_expert=args.target_expert,
        metrics=metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
