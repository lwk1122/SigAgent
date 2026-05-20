#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _write(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def _numeric_mean_summary(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    available_groups = [column for column in group_columns if column in frame.columns]
    available_values = [column for column in value_columns if column in frame.columns]
    if frame.empty or not available_groups or not available_values:
        return pd.DataFrame()
    working = frame.loc[:, available_groups + available_values].copy()
    for column in available_values:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    summary = working.groupby(available_groups, dropna=False)[available_values].mean(numeric_only=True).reset_index()
    count = working.groupby(available_groups, dropna=False).size().rename("row_count").reset_index()
    return summary.merge(count, on=available_groups, how="left")


def _explode_group_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    working = frame.copy()
    working[column] = working[column].fillna("").astype(str).str.split(",")
    working = working.explode(column)
    working[column] = working[column].astype(str).str.strip()
    return working.loc[working[column] != ""].copy()


def make_tables(root: Path, output_dir: Path) -> None:
    metrics_dir = root / "metrics"
    aggregate = _read_optional(metrics_dir / "aggregate_metrics.tsv")
    evidence = _read_optional(metrics_dir / "fusion_evidence_features.tsv")

    known = aggregate.loc[aggregate.get("benchmark_name", pd.Series(dtype=str)).eq("known_catalog")].copy() if "benchmark_name" in aggregate.columns else aggregate.loc[
        aggregate.get("step_name", pd.Series(dtype=str)).astype(str).str.contains("known", na=False)
    ].copy()
    _write(
        _numeric_mean_summary(
            known,
            group_columns=["mutation_type", "expert_name", "burden"],
            value_columns=[
                "sample_f1",
                "sample_jaccard",
                "exposure_tvd",
                "exposure_cosine",
                "reconstruction_cosine",
                "assignment_confidence_ece",
                "exposure_interval_coverage",
                "exposure_interval_mean_width",
            ],
        ),
        output_dir / "known_catalog_summary.tsv",
    )

    insuff = aggregate.loc[
        aggregate.get("step_name", pd.Series(dtype=str)).astype(str).str.contains("insuff", na=False)
    ].copy()
    insuff = _explode_group_column(insuff, "removal_selection_groups")
    _write(
        _numeric_mean_summary(
            insuff,
            group_columns=["mutation_type", "expert_name", "burden", "removal_selection_groups"],
            value_columns=[
                "catalog_insufficiency_auroc",
                "catalog_insufficiency_auprc",
                "catalog_insufficiency_probability_ece",
                "catalog_insufficiency_probability_brier",
                "sample_f1",
                "exposure_tvd",
                "reconstruction_cosine",
            ],
        ),
        output_dir / "catalog_insufficiency_by_group.tsv",
    )

    if not evidence.empty:
        evidence_by_group = _explode_group_column(evidence, "removal_selection_groups")
        _write(
            _numeric_mean_summary(
                evidence_by_group,
                group_columns=["benchmark_name", "mutation_type", "burden_group", "removal_selection_groups"],
                value_columns=[
                    "catalog_insufficiency_proxy_score",
                    "catalog_insufficiency_probability",
                    "residual_structure_score",
                    "agreement_score",
                    "disagreement_score",
                    "mean_reconstruction_cosine",
                    "catalog_feature_missing_catalog_probability_mass",
                    "catalog_feature_classifier_entropy",
                ],
            ),
            output_dir / "fusion_evidence_by_group.tsv",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paper-ready summary TSVs from collected metrics.")
    parser.add_argument("root", help="Paper-suite output root.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/tables.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "tables"
    make_tables(root, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

