#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLUMNS = [
    "burden_group",
    "flatness_group",
    "risk_group",
    "removal_selection_groups",
]


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _infer_benchmark_name(step_name: str) -> str:
    if "insuff" in step_name:
        return "catalog_insufficiency"
    if "known" in step_name:
        return "known_catalog"
    return "unknown"


def _coalesce_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    aliases = {
        "mutation_type": ["mutation_type", "mutation_type_x", "mutation_type_y", "removal_mutation_type"],
        "benchmark_name": ["benchmark_name"],
    }
    for canonical, candidates in aliases.items():
        if canonical not in working.columns:
            working[canonical] = np.nan
        for candidate in candidates:
            if candidate in working.columns and candidate != canonical:
                working[canonical] = working[canonical].combine_first(working[candidate])
    if "benchmark_name" in working.columns and "step_name" in working.columns:
        missing = working["benchmark_name"].isna()
        working.loc[missing, "benchmark_name"] = working.loc[missing, "step_name"].astype(str).map(_infer_benchmark_name)
    return working


def _explode_group_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    working = frame.copy()
    working[column] = working[column].fillna("").astype(str).str.split(",")
    working = working.explode(column)
    working[column] = working[column].astype(str).str.strip()
    return working.loc[working[column] != ""].copy()


def _base_context(row: pd.Series, task_name: str, score_column: str, label_column: str) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "score_column": score_column,
        "label_column": label_column,
        "step_name": row.get("step_name"),
        "benchmark_name": row.get("benchmark_name"),
        "mutation_type": row.get("mutation_type"),
        "expert_name": row.get("expert_name"),
    }


def _bin_one_group(
    group_frame: pd.DataFrame,
    *,
    task_name: str,
    score_column: str,
    label_column: str,
    n_bins: int,
    group_dimension: str,
    group_value: str,
) -> list[dict[str, Any]]:
    valid = group_frame.dropna(subset=[score_column, label_column]).copy()
    if valid.empty:
        return []
    valid[score_column] = pd.to_numeric(valid[score_column], errors="coerce")
    valid[label_column] = pd.to_numeric(valid[label_column], errors="coerce")
    valid = valid.dropna(subset=[score_column, label_column]).copy()
    valid = valid.loc[(valid[score_column] >= 0.0) & (valid[score_column] <= 1.0)].copy()
    if valid.empty:
        return []
    valid[label_column] = valid[label_column].astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    context_source = valid.iloc[0]
    for bin_index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:]), start=1):
        if upper == 1.0:
            mask = (valid[score_column] >= lower) & (valid[score_column] <= upper)
        else:
            mask = (valid[score_column] >= lower) & (valid[score_column] < upper)
        bin_frame = valid.loc[mask]
        if bin_frame.empty:
            continue
        mean_score = float(bin_frame[score_column].mean())
        observed = float(bin_frame[label_column].mean())
        rows.append(
            {
                **_base_context(context_source, task_name, score_column, label_column),
                "group_dimension": group_dimension,
                "group_value": group_value,
                "bin_index": bin_index,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "n_samples": int(len(bin_frame)),
                "mean_predicted_probability": mean_score,
                "observed_positive_fraction": observed,
                "calibration_gap": float(observed - mean_score),
                "abs_calibration_gap": float(abs(observed - mean_score)),
                "brier": float(np.mean(np.square(bin_frame[score_column].to_numpy(dtype=float) - bin_frame[label_column].to_numpy(dtype=float)))),
            }
        )
    return rows


def _summarize_bins(bins: pd.DataFrame) -> pd.DataFrame:
    if bins.empty:
        return pd.DataFrame()
    group_columns = [
        "task_name",
        "score_column",
        "label_column",
        "step_name",
        "benchmark_name",
        "mutation_type",
        "expert_name",
        "group_dimension",
        "group_value",
    ]
    rows = []
    for key, group in bins.groupby(group_columns, dropna=False):
        weight = group["n_samples"].astype(float)
        total = float(weight.sum())
        ece = float((group["abs_calibration_gap"] * weight).sum() / total) if total > 0 else np.nan
        brier = float((group["brier"] * weight).sum() / total) if total > 0 else np.nan
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_samples": int(total),
                "n_bins_nonempty": int(len(group)),
                "ece": ece,
                "brier": brier,
                "mean_predicted_probability": float((group["mean_predicted_probability"] * weight).sum() / total) if total > 0 else np.nan,
                "observed_positive_fraction": float((group["observed_positive_fraction"] * weight).sum() / total) if total > 0 else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _task_frame(
    per_sample: pd.DataFrame,
    *,
    task_name: str,
    score_column: str,
    label_column: str,
    benchmark_name: str,
    assignment_f1_threshold: float,
) -> pd.DataFrame:
    if score_column not in per_sample.columns:
        return pd.DataFrame()
    frame = per_sample.loc[per_sample["benchmark_name"].astype(str) == benchmark_name].copy()
    if frame.empty:
        return frame
    if label_column not in frame.columns:
        if task_name == "assignment_confidence" and "active_set_f1" in frame.columns:
            frame[label_column] = (pd.to_numeric(frame["active_set_f1"], errors="coerce") >= assignment_f1_threshold).astype(int)
        else:
            return pd.DataFrame()
    return frame


def reliability_tables_from_per_sample(
    per_sample: pd.DataFrame,
    *,
    n_bins: int = 10,
    assignment_f1_threshold: float = 0.8,
    group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if per_sample.empty:
        return pd.DataFrame(), pd.DataFrame()
    per_sample = _coalesce_columns(per_sample)
    group_columns = group_columns or DEFAULT_GROUP_COLUMNS
    task_specs = [
        {
            "task_name": "assignment_confidence",
            "score_column": "assignment_confidence_probability",
            "label_column": "assignment_confidence_label",
            "benchmark_name": "known_catalog",
        },
        {
            "task_name": "catalog_insufficiency",
            "score_column": "catalog_insufficiency_probability",
            "label_column": "catalog_insufficient_label",
            "benchmark_name": "catalog_insufficiency",
        },
    ]
    rows: list[dict[str, Any]] = []
    base_group_columns = ["task_name", "step_name", "benchmark_name", "mutation_type", "expert_name"]
    for spec in task_specs:
        task_frame = _task_frame(
            per_sample,
            task_name=spec["task_name"],
            score_column=spec["score_column"],
            label_column=spec["label_column"],
            benchmark_name=spec["benchmark_name"],
            assignment_f1_threshold=assignment_f1_threshold,
        )
        if task_frame.empty:
            continue
        task_frame = task_frame.copy()
        task_frame["task_name"] = spec["task_name"]
        for _, group in task_frame.groupby(base_group_columns, dropna=False):
            rows.extend(
                _bin_one_group(
                    group,
                    task_name=spec["task_name"],
                    score_column=spec["score_column"],
                    label_column=spec["label_column"],
                    n_bins=n_bins,
                    group_dimension="overall",
                    group_value="all",
                )
            )
        for group_column in group_columns:
            if group_column not in task_frame.columns:
                continue
            grouped_frame = _explode_group_column(task_frame, group_column)
            for _, group in grouped_frame.groupby(base_group_columns + [group_column], dropna=False):
                group_value = str(group[group_column].iloc[0])
                rows.extend(
                    _bin_one_group(
                        group,
                        task_name=spec["task_name"],
                        score_column=spec["score_column"],
                        label_column=spec["label_column"],
                        n_bins=n_bins,
                        group_dimension=group_column,
                        group_value=group_value,
                    )
                )
    bins = pd.DataFrame.from_records(rows)
    return bins, _summarize_bins(bins)


def make_calibration_tables(root: Path, output_dir: Path, *, n_bins: int = 10, assignment_f1_threshold: float = 0.8) -> None:
    per_sample = _read_optional(root / "metrics" / "per_sample_metrics.tsv")
    bins, summary = reliability_tables_from_per_sample(
        per_sample,
        n_bins=n_bins,
        assignment_f1_threshold=assignment_f1_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not bins.empty:
        bins.to_csv(output_dir / "reliability_bins.tsv", sep="\t", index=False)
    if not summary.empty:
        summary.to_csv(output_dir / "reliability_summary.tsv", sep="\t", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create reliability-curve tables from paper-suite per-sample metrics.")
    parser.add_argument("root", help="Paper-suite output root.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/tables.")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--assignment-f1-threshold", type=float, default=0.8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "tables"
    make_calibration_tables(
        root,
        output_dir,
        n_bins=args.n_bins,
        assignment_f1_threshold=args.assignment_f1_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
