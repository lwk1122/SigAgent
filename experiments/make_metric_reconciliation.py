#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signature_decision import (  # noqa: E402
    ConfidenceArtifacts,
    build_default_registry,
    fuse_expert_runs,
    load_expert_request,
)
from signature_decision.benchmark import load_truth_exposures  # noqa: E402
from signature_decision.metrics import (  # noqa: E402
    active_set_metrics,
    exposure_error_metrics,
    exposure_frame_from_run,
    reconstruction_metrics,
)
from signature_decision.simulation import (  # noqa: E402
    scaled_truth_exposures,
    simulate_counts_from_truth,
    subset_samples,
)


DEFAULT_THRESHOLDS = (0.0, 0.01, 0.05, 0.10)
DEFAULT_TOP_K = (3, 5)


def _parse_csv(value: str | None, *, cast=str) -> list:
    if value is None:
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def _retain_top_k(exposures: pd.DataFrame, k: int) -> pd.DataFrame:
    retained = pd.DataFrame(0.0, index=exposures.index, columns=exposures.columns)
    normalized = exposures.divide(exposures.sum(axis=0).replace(0.0, np.nan), axis=1).fillna(0.0)
    for sample_id in exposures.columns:
        positive = normalized.loc[:, sample_id]
        keep = positive.loc[positive > 0.0].sort_values(ascending=False).head(k).index
        retained.loc[keep, sample_id] = exposures.loc[keep, sample_id]
    return retained


def _metric_record(
    *,
    run_name: str,
    seed: int,
    burden: int,
    n_samples: int,
    operating_point: str,
    active_threshold: float | None,
    top_k: int | None,
    truth_exposures: pd.DataFrame,
    predicted_exposures: pd.DataFrame,
    sample_matrix: pd.DataFrame,
    signature_matrix: pd.DataFrame,
    vector_error_scope: str,
    interpretation: str,
) -> dict[str, Any]:
    active, per_sample = active_set_metrics(
        truth_exposures,
        predicted_exposures,
        threshold=float(active_threshold or 0.0),
    )
    exposure, _ = exposure_error_metrics(truth_exposures, predicted_exposures)
    reconstruction, _ = reconstruction_metrics(sample_matrix, signature_matrix, predicted_exposures)
    return {
        "expert_name": run_name,
        "seed": int(seed),
        "burden": int(burden),
        "n_samples": int(n_samples),
        "operating_point": operating_point,
        "active_threshold": active_threshold,
        "top_k": top_k,
        "sample_precision": active["sample_precision"],
        "sample_recall": active["sample_recall"],
        "sample_f1": active["sample_f1"],
        "sample_jaccard": active["sample_jaccard"],
        "mean_true_active_count": float(per_sample["true_active_count"].mean()),
        "mean_pred_active_count": float(per_sample["pred_active_count"].mean()),
        "exposure_tvd": exposure["exposure_tvd"],
        "exposure_cosine": exposure["exposure_cosine"],
        "reconstruction_cosine": reconstruction["reconstruction_cosine"],
        "reconstruction_tvd": reconstruction["reconstruction_tvd"],
        "vector_error_scope": vector_error_scope,
        "interpretation": interpretation,
    }


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    group_columns = [
        "expert_name",
        "operating_point",
        "active_threshold",
        "top_k",
        "vector_error_scope",
        "interpretation",
    ]
    value_columns = [
        "sample_precision",
        "sample_recall",
        "sample_f1",
        "sample_jaccard",
        "mean_true_active_count",
        "mean_pred_active_count",
        "exposure_tvd",
        "exposure_cosine",
        "reconstruction_cosine",
        "reconstruction_tvd",
    ]
    working = frame.copy()
    for column in value_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    grouped = working.groupby(group_columns, dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        record = dict(zip(group_columns, keys, strict=False))
        record["n_cells"] = int(len(group))
        record["n_sample_rows"] = int(pd.to_numeric(group["n_samples"], errors="coerce").sum())
        for column in value_columns:
            record[f"{column}_mean"] = float(group[column].mean())
            record[f"{column}_sem"] = _sem(group[column])
        rows.append(record)
    return pd.DataFrame.from_records(rows).sort_values(
        ["expert_name", "operating_point"],
        ignore_index=True,
    )


def run_metric_reconciliation(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    request_base = load_expert_request(
        sample_source=args.sample_source,
        signature_source=args.signature_source,
        mutation_type=args.mutation_type,
    )
    truth = load_truth_exposures(
        args.exposure_source,
        mutation_type=request_base.mutation_type,
        sample_ids=request_base.sample_ids,
        signature_names=request_base.signature_names,
    )
    confidence_artifacts = ConfidenceArtifacts.load(args.confidence_artifact) if args.confidence_artifact else None
    registry = build_default_registry(REPO_ROOT, confidence_artifacts=confidence_artifacts)
    expert_names = _parse_csv(args.expert_names) or ["plain_nnls"]
    burdens = _parse_csv(args.burdens, cast=int) or [100, 200, 500, 2000]
    seeds = _parse_csv(args.seeds, cast=int) or [0, 1, 2, 3, 4]
    thresholds = _parse_csv(args.thresholds, cast=float) or list(DEFAULT_THRESHOLDS)
    top_ks = _parse_csv(args.top_k, cast=int) or list(DEFAULT_TOP_K)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for burden in burdens:
            available_sample_ids = request_base.sample_ids
            chosen_sample_ids = available_sample_ids
            if args.max_samples_per_burden and len(available_sample_ids) > args.max_samples_per_burden:
                chosen_sample_ids = sorted(
                    rng.choice(available_sample_ids, size=args.max_samples_per_burden, replace=False).tolist()
                )
            truth_subset = subset_samples(truth.exposures, chosen_sample_ids)
            sample_matrix = simulate_counts_from_truth(
                request_base.signature_matrix,
                truth_subset,
                burden=burden,
                rng=rng,
            )
            scaled_truth = scaled_truth_exposures(truth_subset, burden)
            request = request_base.with_samples(list(sample_matrix.columns))
            request.sample_matrix = sample_matrix
            request.request_id = f"metric_reconciliation_{args.mutation_type}_{seed}_{burden}"
            runs = registry.run_all(request, expert_names)
            if not args.skip_rule_fusion:
                fusion = fuse_expert_runs(
                    runs,
                    request,
                    confidence_artifacts=confidence_artifacts,
                    catalog_assessor_model=None,
                )
                runs = runs + [fusion.fused_run]

            for run in runs:
                if run.status != "success" or not run.sample_results:
                    continue
                predicted = exposure_frame_from_run(run)
                for threshold in thresholds:
                    label = "strict_gt_0" if threshold == 0.0 else f"threshold_ge_{int(round(threshold * 100))}pct"
                    rows.append(
                        _metric_record(
                            run_name=run.expert_name,
                            seed=seed,
                            burden=burden,
                            n_samples=len(chosen_sample_ids),
                            operating_point=label,
                            active_threshold=threshold,
                            top_k=None,
                            truth_exposures=scaled_truth,
                            predicted_exposures=predicted,
                            sample_matrix=sample_matrix,
                            signature_matrix=request.signature_matrix,
                            vector_error_scope="full_exposure_vector",
                            interpretation="Active-set membership is re-thresholded; exposure and reconstruction errors use the full fitted vector.",
                        )
                    )
                for k in top_ks:
                    top_predicted = _retain_top_k(predicted, k)
                    rows.append(
                        _metric_record(
                            run_name=run.expert_name,
                            seed=seed,
                            burden=burden,
                            n_samples=len(chosen_sample_ids),
                            operating_point=f"predicted_top{k}_vs_truth_gt_0",
                            active_threshold=0.0,
                            top_k=k,
                            truth_exposures=scaled_truth,
                            predicted_exposures=top_predicted,
                            sample_matrix=sample_matrix,
                            signature_matrix=request.signature_matrix,
                            vector_error_scope="top_k_truncated_prediction",
                            interpretation="Only the top-k fitted signatures are retained before active-set and exposure-error scoring.",
                        )
                    )

    per_cell = pd.DataFrame.from_records(rows)
    summary = _summarize(per_cell)
    return {
        "metric_reconciliation_per_cell.tsv": per_cell,
        "metric_reconciliation_summary.tsv": summary,
    }


def _write_report(output_dir: Path, tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    summary = tables.get("metric_reconciliation_summary.tsv", pd.DataFrame())
    lines = [
        "# Metric Reconciliation Report",
        "",
        "Purpose: compare strict active-set F1 with thresholded and top-k operating definitions for the SBS96 known-catalog support check.",
        "",
        "This analysis is interpretive. It does not replace the locked headline benchmark; it explains how active-set metric definitions change the apparent F1 while preserving exposure-vector and reconstruction context.",
        "",
        "## Configuration",
        "",
        f"- mutation_type: `{args.mutation_type}`",
        f"- seeds: `{args.seeds}`",
        f"- burdens: `{args.burdens}`",
        f"- expert_names: `{args.expert_names}`",
        f"- confidence_artifact: `{args.confidence_artifact or ''}`",
        "",
    ]
    if not summary.empty:
        display_columns = [
            "expert_name",
            "operating_point",
            "sample_f1_mean",
            "sample_precision_mean",
            "sample_recall_mean",
            "mean_pred_active_count_mean",
            "exposure_tvd_mean",
            "reconstruction_cosine_mean",
        ]
        lines.extend(["## Summary", ""])
        lines.append(summary.loc[:, display_columns].round(3).to_markdown(index=False))
        lines.append("")
    (output_dir / "metric_reconciliation_report.md").write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build active-set metric reconciliation tables for the SigAgent paper.")
    parser.add_argument("--sample-source", default="Data/test_sbs_catalog.csv")
    parser.add_argument("--signature-source", default="Data/ground.truth.syn.sigs.SBS96.csv")
    parser.add_argument("--exposure-source", default="Data/test_sbs_exposures.csv")
    parser.add_argument("--mutation-type", default="SBS96")
    parser.add_argument("--burdens", default="100,200,500,2000")
    parser.add_argument("--max-samples-per-burden", type=int, default=80)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--thresholds", default="0,0.01,0.05,0.10")
    parser.add_argument("--top-k", default="3,5")
    parser.add_argument("--expert-names", default="plain_nnls")
    parser.add_argument("--confidence-artifact", default="results/paper/paper_review_response_sbs96/artifacts/confidence_sbs96_v2.json")
    parser.add_argument("--skip-rule-fusion", action="store_true")
    parser.add_argument("--output-dir", default="results/paper/metric_reconciliation_20260516")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = run_metric_reconciliation(args)
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, sep="\t", index=False)
    _write_report(output_dir, tables, args)
    (output_dir / "manifest.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
