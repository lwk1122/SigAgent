#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ASSIGNMENT_F1_THRESHOLDS = [0.20, 0.30, 0.40, 0.50]
CATALOG_PROBABILITY_THRESHOLDS = [0.45, 0.55, 0.65, 0.75]


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return pd.read_csv(path, sep="\t")


def _ece(probabilities: pd.Series, labels: pd.Series, *, bins: int = 10) -> tuple[float, int]:
    p = pd.to_numeric(probabilities, errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
    y = pd.to_numeric(labels, errors="coerce").fillna(0.0).to_numpy()
    total = len(p)
    if total == 0:
        return float("nan"), 0
    ece = 0.0
    nonempty = 0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if not mask.any():
            continue
        nonempty += 1
        ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return ece, nonempty


def _assignment_label_sensitivity(per_sample: pd.DataFrame) -> list[dict[str, object]]:
    known = per_sample.loc[
        per_sample["benchmark_name"].astype(str).eq("known_catalog")
        & per_sample["expert_name"].astype(str).eq("rule_fusion")
    ].copy()
    rows: list[dict[str, object]] = []
    for threshold in ASSIGNMENT_F1_THRESHOLDS:
        labels = (pd.to_numeric(known["active_set_f1"], errors="coerce").fillna(0.0) >= threshold).astype(int)
        probabilities = pd.to_numeric(known["assignment_confidence_probability"], errors="coerce").fillna(0.0)
        ece, nonempty_bins = _ece(probabilities, labels)
        brier = float(((probabilities - labels) ** 2).mean()) if len(known) else float("nan")
        rows.append(
            {
                "analysis_block": "assignment_confidence_label_sensitivity",
                "operating_point": f"active_set_f1_ge_{threshold:.2f}",
                "n_rows": int(len(known)),
                "positive_fraction": float(labels.mean()) if len(labels) else float("nan"),
                "ece": ece,
                "brier": brier,
                "nonempty_bins": int(nonempty_bins),
                "active_removal_capture": np.nan,
                "inactive_removal_escalation": np.nan,
                "precision": np.nan,
                "balanced_accuracy": np.nan,
                "interpretation": "reported label" if threshold == 0.30 else "sensitivity label",
            }
        )
    return rows


def _catalog_threshold_sensitivity(per_sample: pd.DataFrame) -> list[dict[str, object]]:
    catalog = per_sample.loc[
        per_sample["benchmark_name"].astype(str).eq("catalog_insufficiency")
        & per_sample["expert_name"].astype(str).eq("rule_fusion")
    ].copy()
    labels = pd.to_numeric(catalog["catalog_insufficient_label"], errors="coerce").fillna(0).astype(int)
    probabilities = pd.to_numeric(catalog["catalog_insufficiency_probability"], errors="coerce").fillna(0.0)
    rows: list[dict[str, object]] = []
    for threshold in CATALOG_PROBABILITY_THRESHOLDS:
        predicted = (probabilities >= threshold).astype(int)
        true_positive = int(((predicted == 1) & (labels == 1)).sum())
        false_positive = int(((predicted == 1) & (labels == 0)).sum())
        true_negative = int(((predicted == 0) & (labels == 0)).sum())
        false_negative = int(((predicted == 0) & (labels == 1)).sum())
        sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan")
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else float("nan")
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else float("nan")
        rows.append(
            {
                "analysis_block": "catalog_insufficiency_threshold_sensitivity",
                "operating_point": f"probability_ge_{threshold:.2f}",
                "n_rows": int(len(catalog)),
                "positive_fraction": float(labels.mean()) if len(labels) else float("nan"),
                "ece": np.nan,
                "brier": np.nan,
                "nonempty_bins": np.nan,
                "active_removal_capture": float(sensitivity),
                "inactive_removal_escalation": float(false_positive / (false_positive + true_negative))
                if false_positive + true_negative
                else float("nan"),
                "precision": float(precision),
                "balanced_accuracy": float((sensitivity + specificity) / 2.0),
                "interpretation": {
                    0.55: "catalog reassessment threshold",
                    0.75: "cohort-level discovery threshold",
                }.get(threshold, "sensitivity threshold"),
            }
        )
    return rows


def build_sensitivity_table(per_sample_path: Path) -> pd.DataFrame:
    per_sample = _read_tsv(per_sample_path)
    rows = _assignment_label_sensitivity(per_sample)
    rows.extend(_catalog_threshold_sensitivity(per_sample))
    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supplementary sensitivity checks for the SigAgent paper.")
    parser.add_argument(
        "--per-sample",
        type=Path,
        default=Path("results/paper/paper_review_response_sbs96/metrics/per_sample_metrics.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper/paper_review_response_sbs96/tables/sensitivity_checks.tsv"),
    )
    args = parser.parse_args()
    table = build_sensitivity_table(args.per_sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
