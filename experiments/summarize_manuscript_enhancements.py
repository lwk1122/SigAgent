#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


METRICS = [
    "sample_f1",
    "sample_precision",
    "sample_recall",
    "exposure_tvd",
    "reconstruction_cosine",
    "catalog_insufficiency_auroc",
    "catalog_insufficiency_auprc",
]


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _split_signatures(value: object, *, limit: int = 5) -> list[str]:
    if pd.isna(value):
        return []
    signatures = [item.strip() for item in str(value).split(",") if item.strip()]
    return signatures[:limit]


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _overlap(left: list[str], right: list[str]) -> int:
    return len(set(left) & set(right))


def _summarize_numeric(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    available_metrics = [metric for metric in metrics if metric in frame.columns]
    if not available_metrics or frame.empty:
        return pd.DataFrame()
    for key, group in frame.groupby(group_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(group_columns, key_values))
        row["n_cells"] = int(len(group))
        for metric in available_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_sem"] = float(values.sem()) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _format_float(value: object) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(value_float):
        return ""
    return f"{value_float:.3f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    working = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    if max_rows is not None:
        working = working.head(max_rows)
    header = "| " + " | ".join(working.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(working.columns)) + " |"
    rows = []
    for _, row in working.iterrows():
        rows.append("| " + " | ".join(_format_float(row[column]) for column in working.columns) + " |")
    return "\n".join([header, separator, *rows])


def summarize_real_data(real_data_root: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    tables_dir = real_data_root / "tables"
    stress = _read_tsv(tables_dir / "real_data_stress_design_summary.tsv")
    delta = _read_tsv(tables_dir / "real_data_catalog_stress_delta.tsv")
    samples = _read_tsv(tables_dir / "real_data_sample_summary.tsv")

    sensitive = pd.DataFrame()
    tumor_sensitivity = pd.DataFrame()
    if not delta.empty:
        delta_column = "catalog_insufficiency_probability_delta_reduced_minus_full"
        rec_changed = delta["primary_recommendation_full"].astype(str).ne(
            delta["primary_recommendation_reduced"].astype(str)
        )
        large_delta = pd.to_numeric(delta[delta_column], errors="coerce") >= 0.10
        sensitive = delta.loc[rec_changed | large_delta].copy()
        sensitive = sensitive.sort_values(
            ["stress_design", delta_column, "sample_id"],
            ascending=[True, False, True],
        )
        sensitive_columns = [
            "stress_design",
            "sample_id",
            "source_tumor_type",
            "source_mutation_count",
            "primary_recommendation_full",
            "primary_recommendation_reduced",
            "catalog_insufficiency_level_full",
            "catalog_insufficiency_level_reduced",
            "catalog_insufficiency_probability_full",
            "catalog_insufficiency_probability_reduced",
            "catalog_insufficiency_probability_delta_reduced_minus_full",
            "mean_reconstruction_cosine_delta_reduced_minus_full",
            "top_signatures_full",
            "top_signatures_reduced",
        ]
        sensitive = sensitive.loc[:, [column for column in sensitive_columns if column in sensitive.columns]]

        rows: list[dict[str, object]] = []
        for key, group in delta.groupby(["stress_design", "source_tumor_type"], dropna=False):
            stress_design, tumor_type = key
            values = pd.to_numeric(group[delta_column], errors="coerce")
            changed = group["primary_recommendation_full"].astype(str).ne(
                group["primary_recommendation_reduced"].astype(str)
            )
            rows.append(
                {
                    "stress_design": stress_design,
                    "source_tumor_type": tumor_type,
                    "n_samples": int(len(group)),
                    "n_delta_ge_0_10": int((values >= 0.10).sum()),
                    "n_primary_recommendation_changed": int(changed.sum()),
                    "mean_catalog_insufficiency_probability_delta": float(values.mean()),
                    "max_catalog_insufficiency_probability_delta": float(values.max()),
                }
            )
        tumor_sensitivity = pd.DataFrame.from_records(rows).sort_values(
            ["stress_design", "n_delta_ge_0_10", "max_catalog_insufficiency_probability_delta"],
            ascending=[True, False, False],
        )

    tool_disagreement = pd.DataFrame()
    tool_summary = pd.DataFrame()
    full_steps = [
        path
        for path in sorted((real_data_root / "raw").glob("*full_catalog*"))
        if (path / "experts" / "summary.tsv").exists()
    ]
    if full_steps:
        experts = _read_tsv(full_steps[0] / "experts" / "summary.tsv")
        fusion = _read_tsv(full_steps[0] / "fusion" / "summary.tsv")
        if not experts.empty:
            plain = experts.loc[experts["expert_name"] == "plain_nnls"].copy()
            musical = experts.loc[experts["expert_name"] == "musical"].copy()
            merged = plain.merge(musical, on="sample_id", suffixes=("_plain_nnls", "_musical"), how="inner")
            rows = []
            for _, row in merged.iterrows():
                plain_top = _split_signatures(row.get("top_signatures_plain_nnls"))
                musical_top = _split_signatures(row.get("top_signatures_musical"))
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "top5_jaccard_plain_nnls_vs_musical": _jaccard(plain_top, musical_top),
                        "top5_overlap_plain_nnls_vs_musical": _overlap(plain_top, musical_top),
                        "plain_nnls_active_signature_count": row.get("active_signature_count_plain_nnls"),
                        "musical_active_signature_count": row.get("active_signature_count_musical"),
                        "active_signature_count_delta_musical_minus_plain_nnls": pd.to_numeric(
                            pd.Series([row.get("active_signature_count_musical")]), errors="coerce"
                        ).iloc[0]
                        - pd.to_numeric(
                            pd.Series([row.get("active_signature_count_plain_nnls")]), errors="coerce"
                        ).iloc[0],
                        "plain_nnls_reconstruction_cosine": row.get("reconstruction_cosine_plain_nnls"),
                        "musical_reconstruction_cosine": row.get("reconstruction_cosine_musical"),
                        "top_signatures_plain_nnls": row.get("top_signatures_plain_nnls"),
                        "top_signatures_musical": row.get("top_signatures_musical"),
                    }
                )
            tool_disagreement = pd.DataFrame.from_records(rows)
            if not fusion.empty:
                context = fusion.loc[
                    :,
                    [
                        column
                        for column in [
                            "sample_id",
                            "primary_recommendation",
                            "catalog_insufficiency_level",
                            "catalog_insufficiency_probability",
                            "mean_reconstruction_cosine",
                            "top_signatures",
                        ]
                        if column in fusion.columns
                    ],
                ]
                tool_disagreement = tool_disagreement.merge(context, on="sample_id", how="left")
            if not samples.empty and "source_tumor_type" in samples.columns:
                sample_context = samples.loc[
                    samples["step_name"].astype(str).str.contains("full_catalog", na=False),
                    ["sample_id", "source_tumor_type", "source_mutation_count"],
                ].drop_duplicates("sample_id")
                tool_disagreement = tool_disagreement.merge(sample_context, on="sample_id", how="left")
            tool_disagreement = tool_disagreement.sort_values(
                ["top5_jaccard_plain_nnls_vs_musical", "sample_id"], ascending=[True, True]
            )
            values = pd.to_numeric(
                tool_disagreement["top5_jaccard_plain_nnls_vs_musical"], errors="coerce"
            )
            tool_summary = pd.DataFrame.from_records(
                [
                    {
                        "n_samples": int(len(tool_disagreement)),
                        "mean_top5_jaccard": float(values.mean()),
                        "median_top5_jaccard": float(values.median()),
                        "n_top5_jaccard_below_0_50": int((values < 0.50).sum()),
                        "n_top5_jaccard_below_0_25": int((values < 0.25).sum()),
                        "mean_plain_nnls_active_signature_count": float(
                            pd.to_numeric(
                                tool_disagreement["plain_nnls_active_signature_count"],
                                errors="coerce",
                            ).mean()
                        ),
                        "mean_musical_active_signature_count": float(
                            pd.to_numeric(
                                tool_disagreement["musical_active_signature_count"],
                                errors="coerce",
                            ).mean()
                        ),
                    }
                ]
            )

    outputs = {
        "real_data_stress_design_summary.tsv": stress,
        "real_data_catalog_sensitive_samples.tsv": sensitive,
        "real_data_tumor_type_sensitivity.tsv": tumor_sensitivity,
        "real_data_tool_disagreement.tsv": tool_disagreement,
        "real_data_tool_disagreement_summary.tsv": tool_summary,
    }
    for filename, frame in outputs.items():
        if not frame.empty:
            frame.to_csv(output_dir / filename, sep="\t", index=False)
    return outputs


def _seed_from_step(step_name: str) -> str:
    match = re.search(r"seed_(\d+)$", step_name)
    return match.group(1) if match else ""


def summarize_comparator(comparator_root: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    frames = []
    for path in sorted((comparator_root / "raw").glob("*/aggregate_metrics.tsv")):
        step_name = path.parent.name
        frame = pd.read_csv(path, sep="\t")
        frame.insert(0, "step_name", step_name)
        frame.insert(1, "seed", _seed_from_step(step_name))
        frame.insert(2, "suite_benchmark", "known_catalog" if step_name.startswith("known") else "catalog_insufficiency")
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if all_rows.empty:
        return {}

    known = all_rows.loc[all_rows["suite_benchmark"] == "known_catalog"].copy()
    insuff = all_rows.loc[all_rows["suite_benchmark"] == "catalog_insufficiency"].copy()
    known_by_burden = _summarize_numeric(
        known,
        ["burden", "expert_name"],
        ["sample_f1", "sample_precision", "sample_recall", "exposure_tvd", "reconstruction_cosine"],
    )
    known_overall = _summarize_numeric(
        known,
        ["expert_name"],
        ["sample_f1", "sample_precision", "sample_recall", "exposure_tvd", "reconstruction_cosine"],
    )
    insuff_overall = _summarize_numeric(
        insuff,
        ["expert_name"],
        [
            "sample_f1",
            "sample_precision",
            "sample_recall",
            "exposure_tvd",
            "reconstruction_cosine",
            "catalog_insufficiency_auroc",
            "catalog_insufficiency_auprc",
        ],
    )
    insuff_by_group = _summarize_numeric(
        insuff,
        ["removal_selection_groups", "expert_name"],
        ["catalog_insufficiency_auroc", "catalog_insufficiency_auprc", "sample_f1", "exposure_tvd"],
    )
    outputs = {
        "musical_comparator_all_aggregate_rows.tsv": all_rows,
        "musical_known_catalog_by_burden.tsv": known_by_burden,
        "musical_known_catalog_overall.tsv": known_overall,
        "musical_catalog_insufficiency_overall.tsv": insuff_overall,
        "musical_catalog_insufficiency_by_removal_group.tsv": insuff_by_group,
    }
    for filename, frame in outputs.items():
        if not frame.empty:
            frame.to_csv(output_dir / filename, sep="\t", index=False)
    return outputs


def write_report(output_dir: Path, real_outputs: dict[str, pd.DataFrame], comparator_outputs: dict[str, pd.DataFrame]) -> None:
    stress = real_outputs.get("real_data_stress_design_summary.tsv", pd.DataFrame())
    sensitive = real_outputs.get("real_data_catalog_sensitive_samples.tsv", pd.DataFrame())
    tool_summary = real_outputs.get("real_data_tool_disagreement_summary.tsv", pd.DataFrame())
    known = comparator_outputs.get("musical_known_catalog_by_burden.tsv", pd.DataFrame())
    insuff = comparator_outputs.get("musical_catalog_insufficiency_overall.tsv", pd.DataFrame())

    lines = [
        "# Manuscript Enhancement Results",
        "",
        "Purpose: strengthen the paper with real-data decision reanalysis and multi-seed optional-comparator evidence.",
        "",
        "## Expanded PCAWG Decision Reanalysis",
        "",
        _markdown_table(
            stress,
            [
                "stress_design",
                "n_samples",
                "n_primary_recommendation_changed",
                "n_catalog_level_changed",
                "mean_catalog_insufficiency_probability_delta",
                "max_catalog_insufficiency_probability_delta",
                "n_catalog_insufficiency_delta_ge_0_10",
                "mean_reconstruction_cosine_delta",
            ],
        ),
        "",
        "Catalog-sensitive samples are rows with a primary-recommendation change or catalog-insufficiency delta >= 0.10.",
        "",
        _markdown_table(
            sensitive,
            [
                "stress_design",
                "sample_id",
                "source_tumor_type",
                "source_mutation_count",
                "primary_recommendation_full",
                "primary_recommendation_reduced",
                "catalog_insufficiency_probability_delta_reduced_minus_full",
                "mean_reconstruction_cosine_delta_reduced_minus_full",
            ],
            max_rows=12,
        ),
        "",
        "## Full-Catalog Tool Disagreement",
        "",
        _markdown_table(
            tool_summary,
            [
                "n_samples",
                "mean_top5_jaccard",
                "median_top5_jaccard",
                "n_top5_jaccard_below_0_50",
                "n_top5_jaccard_below_0_25",
                "mean_plain_nnls_active_signature_count",
                "mean_musical_active_signature_count",
            ],
        ),
        "",
        "## Multi-Seed Optional MuSiCal Comparator",
        "",
        "Known-catalog summaries by burden:",
        "",
        _markdown_table(
            known,
            [
                "burden",
                "expert_name",
                "n_cells",
                "sample_f1_mean",
                "exposure_tvd_mean",
                "reconstruction_cosine_mean",
            ],
        ),
        "",
        "Controlled-removal summaries:",
        "",
        _markdown_table(
            insuff,
            [
                "expert_name",
                "n_cells",
                "sample_f1_mean",
                "exposure_tvd_mean",
                "reconstruction_cosine_mean",
                "catalog_insufficiency_auroc_mean",
                "catalog_insufficiency_auprc_mean",
            ],
        ),
        "",
        "## Manuscript Use",
        "",
        "- Use the expanded PCAWG analysis as a real-data decision-reanalysis supplement, not as biological validation of new signatures.",
        "- Use tool-disagreement summaries to show why a decision layer adds value above exposure fitting.",
        "- Use MuSiCal as an optional user-installed comparator/adaptor result; do not claim that rule fusion outperforms MuSiCal.",
        "- Promote catalog-sensitive samples and stress-design deltas as the main new empirical finding from existing public data.",
    ]
    (output_dir / "manuscript_enhancement_results.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize manuscript-strengthening analyses.")
    parser.add_argument("--real-data-root", required=True)
    parser.add_argument("--comparator-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_outputs = summarize_real_data(Path(args.real_data_root), output_dir)
    comparator_outputs = summarize_comparator(Path(args.comparator_root), output_dir)
    write_report(output_dir, real_outputs, comparator_outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
