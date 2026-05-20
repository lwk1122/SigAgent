#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signature_decision import CatalogInsufficiencyModel


GROUP_COLUMNS = [
    "mutation_type",
    "burden",
    "removed_signature",
    "split_partition",
]

CALIBRATION_GROUP_COLUMNS = [
    "mutation_type",
    "burden_group",
    "flatness_group",
    "disagreement_group",
    "split_partition",
]


def _artifact_id(path: Path) -> str:
    return path.stem


def coefficient_frame(artifact_path: Path) -> pd.DataFrame:
    model = CatalogInsufficiencyModel.load(artifact_path)
    rows = []
    for feature_name, coefficient, scaler_mean, scaler_scale in zip(
        model.feature_names,
        model.coefficients,
        model.scaler_mean,
        model.scaler_scale,
    ):
        rows.append(
            {
                "artifact_id": _artifact_id(artifact_path),
                "artifact_path": str(artifact_path),
                "mutation_type": (model.metadata or {}).get("mutation_type"),
                "feature_name": feature_name,
                "standardized_log_odds_coefficient": float(coefficient),
                "abs_standardized_log_odds_coefficient": abs(float(coefficient)),
                "odds_ratio_per_standard_deviation": float(np.exp(float(coefficient))),
                "scaler_mean": float(scaler_mean),
                "scaler_scale": float(scaler_scale),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["abs_standardized_log_odds_coefficient", "feature_name"],
        ascending=[False, True],
    ).reset_index(drop=True)
    frame.insert(3, "importance_rank", frame.index + 1)
    return frame


def _balance_summary(
    training_frame: pd.DataFrame,
    *,
    artifact_path: Path,
    group_columns: list[str],
) -> pd.DataFrame:
    available_groups = [column for column in group_columns if column in training_frame.columns]
    if training_frame.empty or "label" not in training_frame.columns or not available_groups:
        return pd.DataFrame()
    working = training_frame.loc[:, available_groups + ["label"]].copy()
    working["label"] = pd.to_numeric(working["label"], errors="coerce").fillna(0).astype(int)
    summary = (
        working.groupby(available_groups, dropna=False)
        .agg(n_samples=("label", "size"), n_positive=("label", "sum"))
        .reset_index()
    )
    summary["n_negative"] = summary["n_samples"] - summary["n_positive"]
    summary["positive_fraction"] = summary["n_positive"] / summary["n_samples"].replace(0, np.nan)
    summary.insert(0, "artifact_path", str(artifact_path))
    summary.insert(0, "artifact_id", _artifact_id(artifact_path))
    return summary


def summarize_artifacts(
    artifact_paths: list[Path],
    output_dir: Path,
    *,
    training_source: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_frames = []
    training_balance_frames = []
    group_balance_frames = []

    for artifact_path in artifact_paths:
        coefficient_frames.append(coefficient_frame(artifact_path))
        resolved_training_source = training_source if training_source is not None else artifact_path.with_suffix(".training.tsv")
        if resolved_training_source.exists():
            training_frame = pd.read_csv(resolved_training_source, sep="\t")
            training_balance_frames.append(
                _balance_summary(
                    training_frame,
                    artifact_path=artifact_path,
                    group_columns=GROUP_COLUMNS,
                )
            )
            group_balance_frames.append(
                _balance_summary(
                    training_frame,
                    artifact_path=artifact_path,
                    group_columns=CALIBRATION_GROUP_COLUMNS,
                )
            )

    if coefficient_frames:
        pd.concat(coefficient_frames, ignore_index=True).to_csv(
            output_dir / "catalog_assessor_coefficients.tsv",
            sep="\t",
            index=False,
        )
    if training_balance_frames:
        pd.concat(training_balance_frames, ignore_index=True).to_csv(
            output_dir / "catalog_assessor_training_balance.tsv",
            sep="\t",
            index=False,
        )
    if group_balance_frames:
        pd.concat(group_balance_frames, ignore_index=True).to_csv(
            output_dir / "catalog_assessor_group_balance.tsv",
            sep="\t",
            index=False,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize trained catalog-insufficiency assessor artifacts.")
    parser.add_argument("artifacts", nargs="+", help="Catalog assessor artifact JSON files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--training-source",
        default=None,
        help="Optional training TSV. Only use this when summarizing a single artifact; otherwise each artifact uses <artifact>.training.tsv.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact_paths = [Path(path) for path in args.artifacts]
    if args.training_source and len(artifact_paths) != 1:
        raise SystemExit("--training-source can only be used with one artifact.")
    summarize_artifacts(
        artifact_paths,
        Path(args.output_dir),
        training_source=None if args.training_source is None else Path(args.training_source),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
