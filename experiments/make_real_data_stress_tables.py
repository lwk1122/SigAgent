#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def _read_optional_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision_step_dirs(root: Path) -> list[Path]:
    raw_root = root / "raw"
    if not raw_root.exists():
        return []
    return sorted(path for path in raw_root.iterdir() if path.is_dir() and (path / "fusion" / "summary.tsv").exists())


def _with_step(frame: pd.DataFrame, step_dir: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    working.insert(0, "step_name", step_dir.name)
    working.insert(1, "source_path", str(step_dir))
    return working


def _sample_manifest_from_source(sample_source: Path | None) -> pd.DataFrame:
    if sample_source is None or not sample_source.exists():
        return pd.DataFrame()
    frame = pd.read_csv(sample_source)
    metadata = {"Mutation.type", "Mutation type", "Trinucleotide"}
    sample_columns = [
        column
        for column in frame.columns
        if column not in metadata and pd.to_numeric(frame[column], errors="coerce").notna().all()
    ]
    numeric = frame.loc[:, sample_columns].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "source_tumor_type": sample_id.split("::", 1)[0] if "::" in sample_id else "unknown",
                "source_mutation_count": int(numeric[sample_id].sum()),
            }
            for sample_id in sample_columns
        ]
    )


def _merge_sample_context(frame: pd.DataFrame, sample_context: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or sample_context.empty or "sample_id" not in frame.columns:
        return frame
    return frame.merge(sample_context, on="sample_id", how="left")


def _recommendation_counts(samples: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_columns = [
        "assignment_confidence_probability",
        "catalog_insufficiency_probability",
        "catalog_insufficiency_proxy_score",
        "mean_reconstruction_cosine",
        "residual_structure_score",
        "mutation_count",
    ]
    for dimension in ["primary_recommendation", "catalog_insufficiency_level", "source_tumor_type"]:
        if dimension not in samples.columns:
            continue
        group_columns = ["step_name", dimension] if "step_name" in samples.columns else [dimension]
        for key, group in samples.groupby(group_columns, dropna=False):
            if len(group_columns) == 2:
                step_name, value = key
            else:
                step_name, value = "", key
            row = {
                "step_name": step_name,
                "summary_dimension": dimension,
                "summary_value": value,
                "n_samples": int(len(group)),
            }
            for column in numeric_columns:
                if column in group.columns:
                    row[f"mean_{column}"] = float(pd.to_numeric(group[column], errors="coerce").mean())
            rows.append(row)
    if not candidates.empty and "candidate_type" in candidates.columns:
        group_columns = ["step_name", "candidate_type"] if "step_name" in candidates.columns else ["candidate_type"]
        for key, group in candidates.groupby(group_columns, dropna=False):
            if len(group_columns) == 2:
                step_name, value = key
            else:
                step_name, value = "", key
            rows.append(
                {
                    "step_name": step_name,
                    "summary_dimension": "candidate_type",
                    "summary_value": value,
                    "n_samples": int(group["sample_id"].nunique()) if "sample_id" in group.columns else int(len(group)),
                }
            )
    return pd.DataFrame.from_records(rows)


def _expert_stability(experts: pd.DataFrame) -> pd.DataFrame:
    if experts.empty:
        return pd.DataFrame()
    working = experts.copy()
    for column in ["active_signature_count", "reconstruction_cosine", "rss", "runtime_seconds"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    group_columns = [column for column in ["step_name", "expert_name", "status"] if column in working.columns]
    if not group_columns:
        return pd.DataFrame()
    summary = working.groupby(group_columns, dropna=False).size().rename("n_rows").reset_index()
    for column in ["active_signature_count", "reconstruction_cosine", "rss", "runtime_seconds"]:
        if column in working.columns:
            values = working.groupby(group_columns, dropna=False)[column].mean(numeric_only=True).reset_index()
            values = values.rename(columns={column: f"mean_{column}"})
            summary = summary.merge(values, on=group_columns, how="left")
    return summary


def _catalog_stress_delta(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty or "step_name" not in samples.columns or "sample_id" not in samples.columns:
        return pd.DataFrame()
    full = samples.loc[samples["step_name"].astype(str).str.contains("full_catalog", na=False)].copy()
    reduced_steps = sorted(
        samples.loc[samples["step_name"].astype(str).str.contains("reduced_catalog", na=False), "step_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if full.empty or not reduced_steps:
        return pd.DataFrame()
    full = full.sort_values("step_name").drop_duplicates("sample_id", keep="first")

    rows = []
    numeric_columns = [
        "assignment_confidence_probability",
        "catalog_insufficiency_probability",
        "catalog_insufficiency_proxy_score",
        "mean_reconstruction_cosine",
        "residual_structure_score",
        "mutation_count",
    ]
    for reduced_step in reduced_steps:
        reduced = samples.loc[samples["step_name"].astype(str) == reduced_step].copy()
        reduced = reduced.sort_values("step_name").drop_duplicates("sample_id", keep="first")
        merged = full.merge(reduced, on="sample_id", suffixes=("_full", "_reduced"), how="inner")
        if merged.empty:
            continue
        for _, row in merged.iterrows():
            output = {
                "sample_id": row["sample_id"],
                "stress_step_name": reduced_step,
                "stress_design": (
                    "data_driven_active_signature_removal"
                    if "active_reduced" in reduced_step
                    else "fixed_sbs1_sbs5_removal"
                    if "reduced_catalog" in reduced_step
                    else "reduced_catalog"
                ),
                "source_tumor_type": row.get("source_tumor_type_full", row.get("source_tumor_type_reduced")),
                "source_mutation_count": row.get("source_mutation_count_full", row.get("source_mutation_count_reduced")),
                "full_step_name": row.get("step_name_full"),
                "reduced_step_name": row.get("step_name_reduced"),
                "primary_recommendation_full": row.get("primary_recommendation_full"),
                "primary_recommendation_reduced": row.get("primary_recommendation_reduced"),
                "catalog_insufficiency_level_full": row.get("catalog_insufficiency_level_full"),
                "catalog_insufficiency_level_reduced": row.get("catalog_insufficiency_level_reduced"),
                "top_signatures_full": row.get("top_signatures_full"),
                "top_signatures_reduced": row.get("top_signatures_reduced"),
            }
            for column in numeric_columns:
                full_value = pd.to_numeric(pd.Series([row.get(f"{column}_full")]), errors="coerce").iloc[0]
                reduced_value = pd.to_numeric(pd.Series([row.get(f"{column}_reduced")]), errors="coerce").iloc[0]
                output[f"{column}_full"] = full_value
                output[f"{column}_reduced"] = reduced_value
                output[f"{column}_delta_reduced_minus_full"] = reduced_value - full_value
            rows.append(output)
    return pd.DataFrame.from_records(rows)


def _stress_design_summary(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty:
        return pd.DataFrame()
    working = delta.copy()
    numeric_columns = [
        "catalog_insufficiency_probability_delta_reduced_minus_full",
        "mean_reconstruction_cosine_delta_reduced_minus_full",
        "residual_structure_score_delta_reduced_minus_full",
    ]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    group_columns = [column for column in ["stress_design", "stress_step_name"] if column in working.columns]
    if not group_columns:
        return pd.DataFrame()
    rows = []
    for key, group in working.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        catalog_delta = group["catalog_insufficiency_probability_delta_reduced_minus_full"]
        reconstruction_delta = group["mean_reconstruction_cosine_delta_reduced_minus_full"]
        residual_delta = group["residual_structure_score_delta_reduced_minus_full"]
        row.update(
            {
                "n_samples": int(len(group)),
                "n_primary_recommendation_changed": int(
                    (group["primary_recommendation_full"].astype(str) != group["primary_recommendation_reduced"].astype(str)).sum()
                )
                if {"primary_recommendation_full", "primary_recommendation_reduced"}.issubset(group.columns)
                else 0,
                "n_catalog_level_changed": int(
                    (group["catalog_insufficiency_level_full"].astype(str) != group["catalog_insufficiency_level_reduced"].astype(str)).sum()
                )
                if {"catalog_insufficiency_level_full", "catalog_insufficiency_level_reduced"}.issubset(group.columns)
                else 0,
                "mean_catalog_insufficiency_probability_delta": float(catalog_delta.mean()),
                "max_catalog_insufficiency_probability_delta": float(catalog_delta.max()),
                "n_catalog_insufficiency_delta_ge_0_10": int((catalog_delta >= 0.10).sum()),
                "mean_reconstruction_cosine_delta": float(reconstruction_delta.mean()),
                "min_reconstruction_cosine_delta": float(reconstruction_delta.min()),
                "mean_residual_structure_score_delta": float(residual_delta.mean()),
                "max_residual_structure_score_delta": float(residual_delta.max()),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _public_data_manifest(
    sample_source: Path | None,
    source_manifest: Path | None,
    sample_context: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if source_manifest is not None and source_manifest.exists():
        source_frame = pd.read_csv(source_manifest, sep="\t")
        rows.extend(source_frame.to_dict("records"))
    if sample_source is not None and sample_source.exists():
        frame = pd.read_csv(sample_source)
        metadata = {"Mutation.type", "Mutation type", "Trinucleotide"}
        sample_columns = [
            column
            for column in frame.columns
            if column not in metadata and pd.to_numeric(frame[column], errors="coerce").notna().all()
        ]
        rows.append(
            {
                "source_label": "analysis_sample_source",
                "source_url": "",
                "source_reference": "",
                "raw_path": "",
                "sample_source": str(sample_source),
                "sample_manifest": "",
                "raw_sha256": "",
                "sample_source_sha256": _sha256(sample_source),
                "n_contexts": int(len(frame)),
                "n_selected_samples": int(len(sample_columns)),
                "selection_mode": "",
                "tumor_type_filter": "",
                "total_selected_mutations": int(sample_context["source_mutation_count"].sum()) if "source_mutation_count" in sample_context else 0,
                "generated_at": "",
            }
        )
    return pd.DataFrame.from_records(rows)


def make_real_data_stress_tables(
    root: Path,
    output_dir: Path,
    *,
    sample_source: Path | None = None,
    source_manifest: Path | None = None,
) -> None:
    step_dirs = _decision_step_dirs(root)
    sample_context = _sample_manifest_from_source(sample_source)
    sample_frames = []
    candidate_frames = []
    expert_frames = []
    for step_dir in step_dirs:
        fusion = _read_optional_tsv(step_dir / "fusion" / "summary.tsv")
        cohort = _read_optional_tsv(step_dir / "cohort" / "summary.tsv")
        candidates = _read_optional_tsv(step_dir / "cohort" / "candidates.tsv")
        experts = _read_optional_tsv(step_dir / "experts" / "summary.tsv")
        sample_frame = fusion if not fusion.empty else cohort
        if not sample_frame.empty:
            sample_frames.append(_with_step(_merge_sample_context(sample_frame, sample_context), step_dir))
        if not candidates.empty:
            candidate_frames.append(_with_step(candidates, step_dir))
        if not experts.empty:
            expert_frames.append(_with_step(experts, step_dir))

    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    experts = pd.concat(expert_frames, ignore_index=True) if expert_frames else pd.DataFrame()
    recommendation_counts = _recommendation_counts(samples, candidates)
    expert_stability = _expert_stability(experts)
    catalog_stress_delta = _catalog_stress_delta(samples)
    stress_design_summary = _stress_design_summary(catalog_stress_delta)
    public_manifest = _public_data_manifest(sample_source, source_manifest, sample_context)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "real_data_sample_summary.tsv": samples,
        "real_data_recommendation_counts.tsv": recommendation_counts,
        "real_data_expert_stability.tsv": expert_stability,
        "real_data_catalog_stress_delta.tsv": catalog_stress_delta,
        "real_data_stress_design_summary.tsv": stress_design_summary,
        "real_data_candidate_summary.tsv": candidates,
        "public_data_manifest.tsv": public_manifest,
    }
    for filename, frame in outputs.items():
        if not frame.empty:
            frame.to_csv(output_dir / filename, sep="\t", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paper-ready tables for public real-data decision stress tests.")
    parser.add_argument("root", help="Paper-suite output root.")
    parser.add_argument("--sample-source", default=None)
    parser.add_argument("--source-manifest", default=None)
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/tables.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    make_real_data_stress_tables(
        root,
        Path(args.output_dir) if args.output_dir else root / "tables",
        sample_source=Path(args.sample_source) if args.sample_source else None,
        source_manifest=Path(args.source_manifest) if args.source_manifest else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
