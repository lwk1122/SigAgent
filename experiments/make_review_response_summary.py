#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "sample_f1",
    "exposure_tvd",
    "reconstruction_cosine",
    "catalog_insufficiency_auroc",
    "catalog_insufficiency_auprc",
    "ece",
    "brier",
    "n_bins_nonempty",
]


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _numeric_summary(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    groups = [column for column in group_columns if column in frame.columns]
    values = [column for column in value_columns if column in frame.columns]
    if not groups or not values:
        return pd.DataFrame()
    working = frame.loc[:, groups + values].copy()
    for column in values:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    summary = working.groupby(groups, dropna=False)[values].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    for column in values:
        count_col = f"{column}_count"
        std_col = f"{column}_std"
        sem_col = f"{column}_sem"
        if count_col in summary.columns and std_col in summary.columns:
            summary[sem_col] = summary[std_col] / summary[count_col].pow(0.5)
    return summary


def _write(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def _headline_value(frame: pd.DataFrame, *, row_filter: pd.Series, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame.loc[row_filter, column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _available_values(*frames: pd.DataFrame, column: str) -> list[str]:
    values: set[str] = set()
    for frame in frames:
        if not frame.empty and column in frame.columns:
            values.update(str(value) for value in frame[column].dropna().unique())
    return sorted(values)


def make_review_response_summary(root: Path, output_dir: Path) -> None:
    aggregate = _read_tsv(root / "metrics" / "aggregate_metrics.tsv")
    catalog_by_group = _read_tsv(root / "tables" / "catalog_insufficiency_by_group.tsv")
    reliability = _read_tsv(root / "tables" / "reliability_summary.tsv")
    known_summary = _read_tsv(root / "tables" / "known_catalog_summary.tsv")

    if not aggregate.empty:
        known = aggregate.loc[aggregate["benchmark_name"].astype(str).eq("known_catalog")].copy()
        _write(
            _numeric_summary(
                known,
                ["mutation_type", "expert_name", "burden"],
                ["sample_f1", "exposure_tvd", "reconstruction_cosine"],
            ),
            output_dir / "known_catalog_by_burden_with_uncertainty.tsv",
        )
        _write(
            _numeric_summary(
                known,
                ["mutation_type", "expert_name"],
                ["sample_f1", "exposure_tvd", "reconstruction_cosine"],
            ),
            output_dir / "known_catalog_overall_with_uncertainty.tsv",
        )

    if not catalog_by_group.empty:
        _write(
            _numeric_summary(
                catalog_by_group,
                ["mutation_type", "expert_name", "burden", "removal_selection_groups"],
                ["catalog_insufficiency_auroc", "catalog_insufficiency_auprc"],
            ),
            output_dir / "catalog_insufficiency_by_group_with_uncertainty.tsv",
        )
        _write(
            _numeric_summary(
                catalog_by_group,
                ["mutation_type", "expert_name"],
                ["catalog_insufficiency_auroc", "catalog_insufficiency_auprc"],
            ),
            output_dir / "catalog_insufficiency_overall_with_uncertainty.tsv",
        )

    if not reliability.empty:
        overall = reliability.loc[
            reliability["group_dimension"].astype(str).eq("overall")
            & reliability["group_value"].astype(str).eq("all")
        ].copy()
        _write(
            _numeric_summary(
                overall,
                ["task_name", "benchmark_name", "mutation_type", "expert_name"],
                ["ece", "brier", "n_bins_nonempty", "n_samples"],
            ),
            output_dir / "calibration_overall_with_uncertainty.tsv",
        )

    mutation_types = _available_values(aggregate, catalog_by_group, reliability, known_summary, column="mutation_type")
    expert_names = _available_values(aggregate, catalog_by_group, reliability, known_summary, column="expert_name")
    suite_label = "/".join(mutation_types) if mutation_types else root.name

    lines = [f"# Review-Response {suite_label} Headline Metrics", ""]
    if not known_summary.empty:
        for expert in expert_names:
            mask = known_summary["expert_name"].astype(str).eq(expert)
            sample_f1 = _headline_value(known_summary, row_filter=mask, column="sample_f1")
            exposure_tvd = _headline_value(known_summary, row_filter=mask, column="exposure_tvd")
            reconstruction_cosine = _headline_value(known_summary, row_filter=mask, column="reconstruction_cosine")
            if sample_f1 is not None:
                lines.append(
                    f"- Known catalog {expert}: mean sample F1 `{sample_f1:.6f}`, "
                    f"mean exposure TVD `{exposure_tvd:.6f}`, mean reconstruction cosine `{reconstruction_cosine:.6f}`."
                )
    if not catalog_by_group.empty:
        for expert in expert_names:
            mask = catalog_by_group["expert_name"].astype(str).eq(expert)
            auroc = _headline_value(catalog_by_group, row_filter=mask, column="catalog_insufficiency_auroc")
            auprc = _headline_value(catalog_by_group, row_filter=mask, column="catalog_insufficiency_auprc")
            if auroc is not None:
                lines.append(
                    f"- Catalog insufficiency {expert}: mean AUROC `{auroc:.6f}`, mean AUPRC `{auprc:.6f}`."
                )
    if not reliability.empty:
        for expert in expert_names:
            overall = reliability.loc[
                reliability["task_name"].astype(str).eq("assignment_confidence")
                & reliability["group_dimension"].astype(str).eq("overall")
                & reliability["group_value"].astype(str).eq("all")
                & reliability["expert_name"].astype(str).eq(expert)
            ].copy()
            if not overall.empty:
                ece = float(pd.to_numeric(overall["ece"], errors="coerce").mean())
                brier = float(pd.to_numeric(overall["brier"], errors="coerce").mean())
                bins = float(pd.to_numeric(overall["n_bins_nonempty"], errors="coerce").mean())
                samples = int(pd.to_numeric(overall["n_samples"], errors="coerce").sum())
                lines.append(
                    f"- Assignment-confidence calibration {expert}: mean ECE `{ece:.6f}`, "
                    f"mean Brier `{brier:.6f}`, mean non-empty bins `{bins:.1f}`, total evaluated rows `{samples}`."
                )
    (output_dir / "headline_metrics.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize review-response paper-scale metrics.")
    parser.add_argument("root", help="Review-response suite root.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/tables.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "tables"
    make_review_response_summary(root, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
