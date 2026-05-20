#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signature_decision.benchmark import load_truth_exposures
from signature_decision.experts.registry import build_default_registry
from signature_decision.experts.schema import ExpertRequest
from signature_decision.fusion import RuleFusionConfig, fuse_expert_runs
from signature_decision.metrics import binary_probability_metrics, evaluate_expert_run
from signature_decision.simulation import scaled_truth_exposures, simulate_counts_from_truth, subset_samples
from signature_decision.experts.io import load_expert_request


def _write(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def _mean_summary(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    values = [col for col in value_cols if col in frame.columns]
    groups = [col for col in group_cols if col in frame.columns]
    if not values or not groups:
        return pd.DataFrame()
    working = frame.loc[:, groups + values].copy()
    for col in values:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    out = working.groupby(groups, dropna=False)[values].mean(numeric_only=True).reset_index()
    out["n_cells"] = working.groupby(groups, dropna=False).size().to_numpy()
    return out


def _build_known_request(
    base_request,
    truth_exposures: pd.DataFrame,
    *,
    sample_source: str,
    signature_source: str,
    burden: int,
    sample_ids: list[str],
    rng: np.random.Generator,
) -> tuple[ExpertRequest, pd.DataFrame]:
    truth_subset = subset_samples(truth_exposures, sample_ids)
    simulated_samples = simulate_counts_from_truth(
        base_request.signature_matrix,
        truth_subset,
        burden=burden,
        rng=rng,
    )
    scaled_truth = scaled_truth_exposures(truth_subset, burden)
    request = ExpertRequest(
        mutation_type=base_request.mutation_type,
        sample_matrix=simulated_samples,
        signature_matrix=base_request.signature_matrix.copy(),
        channel_metadata=base_request.channel_metadata.copy() if base_request.channel_metadata is not None else None,
        sample_source=str(sample_source),
        signature_source=str(signature_source),
        reference_name=base_request.reference_name,
        request_id=f"risk_opt_known_{base_request.mutation_type}_{burden}",
        alignment_strategy=base_request.alignment_strategy,
    )
    return request, scaled_truth


def run_complete_catalog_grid(
    *,
    output_dir: Path,
    sample_source: str,
    signature_source: str,
    exposure_source: str,
    mutation_type: str,
    seeds: list[int],
    burdens: list[int],
    max_samples_per_burden: int,
    max_reported_grid: list[int],
    active_thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_request = load_expert_request(
        sample_source=sample_source,
        signature_source=signature_source,
        mutation_type=mutation_type,
    )
    truth = load_truth_exposures(
        exposure_source,
        mutation_type=base_request.mutation_type,
        sample_ids=base_request.sample_ids,
        signature_names=base_request.signature_names,
    )
    registry = build_default_registry(REPO_ROOT)
    rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for burden in burdens:
            available_sample_ids = base_request.sample_ids
            if max_samples_per_burden and len(available_sample_ids) > max_samples_per_burden:
                sample_ids = sorted(
                    rng.choice(available_sample_ids, size=max_samples_per_burden, replace=False).tolist()
                )
            else:
                sample_ids = list(available_sample_ids)
            request, scaled_truth = _build_known_request(
                base_request,
                truth.exposures,
                sample_source=sample_source,
                signature_source=signature_source,
                burden=burden,
                sample_ids=sample_ids,
                rng=rng,
            )
            runs = registry.run_all(request, ["plain_nnls"])

            run_variants = [("plain_nnls", "plain_nnls", runs[0], {})]
            for max_reported in max_reported_grid:
                config = RuleFusionConfig(max_reported_signatures=max_reported)
                fusion_output = fuse_expert_runs(runs, request, config=config)
                run_variants.append(
                    (
                        f"rule_fusion_top{max_reported}",
                        "rule_fusion",
                        fusion_output.fused_run,
                        {"max_reported_signatures": max_reported, **asdict(config)},
                    )
                )

            for variant_name, expert_family, run, config_values in run_variants:
                for active_threshold in active_thresholds:
                    aggregate, per_sample = evaluate_expert_run(
                        run,
                        sample_matrix=request.sample_matrix,
                        signature_matrix=request.signature_matrix,
                        truth_exposures=scaled_truth,
                        active_threshold=active_threshold,
                    )
                    row = {
                        "seed": seed,
                        "burden": burden,
                        "variant_name": variant_name,
                        "expert_family": expert_family,
                        "active_threshold": active_threshold,
                        "n_samples": len(sample_ids),
                        **config_values,
                        **aggregate,
                    }
                    rows.append(row)
                    if not per_sample.empty and active_threshold in {0.0, 0.01, 0.05}:
                        per_sample = per_sample.copy()
                        per_sample["seed"] = seed
                        per_sample["burden"] = burden
                        per_sample["variant_name"] = variant_name
                        per_sample["active_threshold"] = active_threshold
                        detail_frames.append(per_sample)

    detailed = pd.DataFrame.from_records(rows)
    per_sample_detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    _write(detailed, output_dir / "complete_catalog_fusion_grid.tsv")
    _write(per_sample_detail, output_dir / "complete_catalog_fusion_grid_per_sample.tsv")

    summary = _mean_summary(
        detailed,
        ["variant_name", "expert_family", "active_threshold"],
        [
            "sample_precision",
            "sample_recall",
            "sample_f1",
            "sample_jaccard",
            "signature_f1",
            "exposure_tvd",
            "exposure_cosine",
            "reconstruction_cosine",
            "reconstruction_tvd",
            "pred_active_count",
        ],
    )
    _write(summary, output_dir / "complete_catalog_fusion_grid_summary.tsv")
    return detailed, summary


def _threshold_operating_points(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    labels = pd.to_numeric(frame["catalog_insufficient_label"], errors="coerce").fillna(0).astype(int)
    scores = pd.to_numeric(frame[score_column], errors="coerce")
    out: dict[str, float] = {}
    for threshold in (0.55, 0.75):
        pred = scores.ge(threshold)
        pos = labels.eq(1)
        neg = labels.eq(0)
        out[f"capture_at_{threshold:.2f}"] = float((pred & pos).sum() / pos.sum()) if pos.sum() else np.nan
        out[f"inactive_escalation_at_{threshold:.2f}"] = float((pred & neg).sum() / neg.sum()) if neg.sum() else np.nan
    return out


def run_proxy_stress_audit(*, output_dir: Path, paper_results_root: Path) -> pd.DataFrame:
    source = paper_results_root / "paper_review_response_sbs96" / "metrics" / "per_sample_metrics.tsv"
    if not source.exists():
        return pd.DataFrame()
    frame = pd.read_csv(source, sep="\t")
    working = frame.loc[
        frame.get("benchmark_name", pd.Series(dtype=str)).astype(str).eq("catalog_insufficiency")
        & frame.get("expert_name", pd.Series(dtype=str)).astype(str).eq("rule_fusion")
    ].copy()
    if working.empty:
        return pd.DataFrame()
    score_column = "catalog_insufficiency_probability"
    if score_column not in working.columns or working[score_column].isna().all():
        score_column = "catalog_insufficiency_score"
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working["catalog_insufficient_label"] = pd.to_numeric(
        working["catalog_insufficient_label"], errors="coerce"
    ).fillna(0).astype(int)

    groups: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=working.index)),
        ("high_similarity", working["removal_selection_groups"].fillna("").astype(str).str.contains("high_similarity")),
        ("flat_signature", working["removal_selection_groups"].fillna("").astype(str).str.contains("flat_signature")),
        ("peaky_signature", working["removal_selection_groups"].fillna("").astype(str).str.contains("peaky_signature")),
        (
            "high_prevalence_active",
            working["removal_selection_groups"].fillna("").astype(str).str.contains("high_prevalence_active"),
        ),
        (
            "low_prevalence_active",
            working["removal_selection_groups"].fillna("").astype(str).str.contains("low_prevalence_active"),
        ),
    ]
    rows: list[dict[str, object]] = []
    for group_name, mask in groups:
        subset = working.loc[mask].copy()
        if subset.empty:
            continue
        metrics = binary_probability_metrics(
            subset["catalog_insufficient_label"],
            subset[score_column],
            prefix="catalog_insufficiency",
        )
        rows.append(
            {
                "group_name": group_name,
                "n_rows": int(len(subset)),
                "n_positive": int(subset["catalog_insufficient_label"].sum()),
                "n_negative": int((subset["catalog_insufficient_label"] == 0).sum()),
                "score_column": score_column,
                **metrics,
                **_threshold_operating_points(subset, score_column),
            }
        )

    out = pd.DataFrame.from_records(rows)
    _write(out, output_dir / "controlled_removal_proxy_stress.tsv")
    return out


def summarize_comparator_pilot(*, output_dir: Path, paper_results_root: Path) -> pd.DataFrame:
    paths = [
        paper_results_root / "comparator_audit_smoke" / "raw" / "toy_sbs96_plain_musical_spa" / "aggregate_metrics.tsv",
        paper_results_root / "comparator_audit_smoke" / "raw" / "insuff_sbs96_plain_musical_spa_pilot" / "aggregate_metrics.tsv",
    ]
    frames = []
    for path in paths:
        if path.exists():
            frame = pd.read_csv(path, sep="\t")
            frame["source_suite"] = path.parent.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    summary = _mean_summary(
        data,
        ["source_suite", "benchmark_name", "expert_name"],
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
    _write(summary, output_dir / "external_comparator_pilot_summary.tsv")
    return summary


def write_report(
    *,
    output_dir: Path,
    complete_summary: pd.DataFrame,
    proxy_summary: pd.DataFrame,
    comparator_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Review Risk Optimization Pilot",
        "",
        "Purpose: exploratory tests for peer-review risks 2, 3, and 4. These results do not replace the locked paper suite unless explicitly promoted after integrity review.",
        "",
        "## Risk 4: complete-catalog active-set F1",
        "",
    ]
    strict = complete_summary.loc[complete_summary["active_threshold"].eq(0.0)].copy() if not complete_summary.empty else pd.DataFrame()
    if not strict.empty:
        best = strict.sort_values(["sample_f1", "reconstruction_cosine"], ascending=[False, False]).head(8)
        lines.append(best[[
            "variant_name",
            "sample_f1",
            "sample_precision",
            "sample_recall",
            "exposure_tvd",
            "reconstruction_cosine",
            "n_cells",
        ]].to_markdown(index=False, floatfmt=".3f"))
        baseline = strict.loc[strict["variant_name"].eq("rule_fusion_top12")]
        if not baseline.empty:
            b = baseline.iloc[0]
            lines.append("")
            lines.append(
                f"Locked manuscript analogue: rule_fusion_top12 strict sample F1 {b['sample_f1']:.3f}, "
                f"exposure TVD {b['exposure_tvd']:.3f}, reconstruction cosine {b['reconstruction_cosine']:.3f}."
            )
    else:
        lines.append("No complete-catalog grid results were generated.")

    lines.extend(["", "## Risk 3: proxy-task stress subsets", ""])
    if not proxy_summary.empty:
        lines.append(proxy_summary.to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("No proxy stress summary was generated.")

    lines.extend(["", "## Risk 2: external comparator pilot", ""])
    if not comparator_summary.empty:
        lines.append(comparator_summary.to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("No comparator pilot summary was found.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A top-K rule-fusion cap can be considered if it improves strict active-set F1 without a large reconstruction or TVD penalty.",
            "- Proxy-task subset summaries should be used to describe benchmark boundaries, not as real unknown-process validation.",
            "- External comparator rows are adapter-feasibility evidence unless scaled and run under locked, license-compatible conditions.",
        ]
    )
    (output_dir / "review_risk_optimization_report.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exploratory optimization tests for manuscript review risks.")
    parser.add_argument("--output-dir", default="results/paper/review_risk_optimization_20260516")
    parser.add_argument("--sample-source", default="Data/test_sbs_catalog.csv")
    parser.add_argument("--signature-source", default="Data/ground.truth.syn.sigs.SBS96.csv")
    parser.add_argument("--exposure-source", default="Data/test_sbs_exposures.csv")
    parser.add_argument("--mutation-type", default="SBS96")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--burdens", default="100,200,500,2000")
    parser.add_argument("--max-samples-per-burden", type=int, default=80)
    parser.add_argument("--max-reported-grid", default="3,4,5,6,8,10,12,16")
    parser.add_argument("--active-thresholds", default="0,0.005,0.01,0.02,0.05")
    parser.add_argument("--paper-results-root", default="results/paper")
    return parser


def _parse_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    complete_detail, complete_summary = run_complete_catalog_grid(
        output_dir=output_dir,
        sample_source=args.sample_source,
        signature_source=args.signature_source,
        exposure_source=args.exposure_source,
        mutation_type=args.mutation_type,
        seeds=_parse_list(args.seeds, int),
        burdens=_parse_list(args.burdens, int),
        max_samples_per_burden=args.max_samples_per_burden,
        max_reported_grid=_parse_list(args.max_reported_grid, int),
        active_thresholds=_parse_list(args.active_thresholds, float),
    )
    proxy_summary = run_proxy_stress_audit(
        output_dir=output_dir,
        paper_results_root=REPO_ROOT / args.paper_results_root,
    )
    comparator_summary = summarize_comparator_pilot(
        output_dir=output_dir,
        paper_results_root=REPO_ROOT / args.paper_results_root,
    )
    write_report(
        output_dir=output_dir,
        complete_summary=complete_summary,
        proxy_summary=proxy_summary,
        comparator_summary=comparator_summary,
    )
    print(f"Wrote exploratory results to {output_dir}")
    print(f"Complete-catalog grid rows: {len(complete_detail)}")
    print(f"Proxy stress rows: {len(proxy_summary)}")
    print(f"Comparator summary rows: {len(comparator_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

