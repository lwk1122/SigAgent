#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "catalog_feature_mutation_count",
    "catalog_feature_failed_expert_fraction",
    "catalog_feature_disagreement_score",
    "catalog_feature_exposure_disagreement_score",
    "catalog_feature_mean_reconstruction_cosine",
    "catalog_feature_best_reconstruction_cosine",
    "catalog_feature_mean_relative_l1_pct",
    "catalog_feature_mean_residual_structure_score",
    "catalog_feature_max_residual_structure_score",
    "catalog_feature_missing_catalog_probability_mass",
    "catalog_feature_classifier_entropy",
]


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return pd.read_csv(path, sep="\t")


def _seed_label(step_name: object) -> str:
    match = re.search(r"seed(\d+)", str(step_name))
    return f"seed{match.group(1)}" if match else str(step_name)


def _prepare_catalog_frame(feature_path: Path, per_sample_path: Path) -> pd.DataFrame:
    features = _read_tsv(feature_path)
    features = features.loc[
        features["benchmark_name"].astype(str).eq("catalog_insufficiency")
        & features["step_name"].astype(str).str.contains("insuff")
    ].copy()

    per_sample = _read_tsv(per_sample_path)
    labels = per_sample.loc[
        per_sample["benchmark_name"].astype(str).eq("catalog_insufficiency")
        & per_sample["expert_name"].astype(str).eq("rule_fusion"),
        ["step_name", "sample_id", "removed_signature", "burden", "catalog_insufficient_label"],
    ].copy()

    frame = features.merge(
        labels,
        on=["step_name", "sample_id", "removed_signature", "burden"],
        how="inner",
        validate="one_to_one",
    )
    if frame.empty:
        raise ValueError("No merged catalog-insufficiency rows were found.")
    frame["label"] = pd.to_numeric(frame["catalog_insufficient_label"], errors="coerce").fillna(0).astype(int)
    frame["seed"] = frame["step_name"].map(_seed_label)
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def _fold_metric(frame: pd.DataFrame, holdout_column: str, holdout_value: str) -> dict[str, object] | None:
    train = frame.loc[frame[holdout_column].astype(str).ne(str(holdout_value))].copy()
    test = frame.loc[frame[holdout_column].astype(str).eq(str(holdout_value))].copy()
    if train.empty or test.empty:
        return None
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        return None

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(train.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float), train["label"].to_numpy(dtype=int))
    scores = model.predict_proba(test.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]
    labels = test["label"].to_numpy(dtype=int)
    return {
        "holdout_type": holdout_column,
        "holdout_value": str(holdout_value),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_positive_fraction": float(np.mean(labels)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for holdout_type, group in folds.groupby("holdout_type", sort=False):
        rows.append(
            {
                "holdout_type": holdout_type,
                "n_folds": int(len(group)),
                "test_rows_min": int(group["test_rows"].min()),
                "test_rows_max": int(group["test_rows"].max()),
                "auroc_mean": float(group["auroc"].mean()),
                "auroc_min": float(group["auroc"].min()),
                "auroc_max": float(group["auroc"].max()),
                "auprc_mean": float(group["auprc"].mean()),
                "auprc_min": float(group["auprc"].min()),
                "auprc_max": float(group["auprc"].max()),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_robustness_tables(feature_path: Path, per_sample_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _prepare_catalog_frame(feature_path, per_sample_path)
    rows: list[dict[str, object]] = []
    for seed in sorted(frame["seed"].astype(str).unique()):
        metric = _fold_metric(frame, "seed", seed)
        if metric is not None:
            rows.append(metric)
    for removal_group in sorted(frame["removal_selection_groups"].astype(str).unique()):
        metric = _fold_metric(frame, "removal_selection_groups", removal_group)
        if metric is not None:
            rows.append(metric)
    folds = pd.DataFrame.from_records(rows)
    if folds.empty:
        raise ValueError("No valid robustness folds were produced.")
    return folds, _summarize(folds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build catalog-insufficiency robustness tables for the SigAgent paper.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("results/paper/paper_review_response_sbs96/metrics/fusion_evidence_features.tsv"),
    )
    parser.add_argument(
        "--per-sample",
        type=Path,
        default=Path("results/paper/paper_review_response_sbs96/metrics/per_sample_metrics.tsv"),
    )
    parser.add_argument(
        "--fold-output",
        type=Path,
        default=Path("results/paper/paper_review_response_sbs96/tables/catalog_insufficiency_robustness.tsv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/paper/paper_review_response_sbs96/tables/catalog_insufficiency_robustness_summary.tsv"),
    )
    args = parser.parse_args()

    folds, summary = build_robustness_tables(args.features, args.per_sample)
    args.fold_output.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.fold_output, sep="\t", index=False, float_format="%.6f")
    summary.to_csv(args.summary_output, sep="\t", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
