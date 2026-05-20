from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .conformal_groups import ASSIGNMENT_GROUP_COLUMNS, CATALOG_ASSESSOR_GROUP_COLUMNS, EXPOSURE_GROUP_COLUMNS, group_context_from_sample_counts
from .catalog_insufficiency import (
    catalog_insufficiency_level_from_sample_result,
    catalog_insufficiency_probability_from_sample_result,
    catalog_insufficiency_score_from_sample_result,
)
from .experts.io import load_expert_request
from .experts.registry import ExpertRegistry, build_default_registry
from .experts.schema import ExpertRequest, ExpertSampleResult
from .fusion import fuse_expert_runs
from .metrics import (
    binary_probability_metrics,
    catalog_insufficiency_metrics,
    evaluate_expert_run,
    exposure_interval_metrics,
    groupwise_interval_metrics,
    groupwise_probability_metrics,
    normalize_exposures,
    summarize_group_metric_frame,
)
from .removal_design import load_catalog_removal_design
from .schemas import BenchmarkSliceResult, BenchmarkSuiteResult, GroundTruthSet
from .simulation import scaled_truth_exposures, simulate_counts_from_truth, subset_samples


def load_truth_exposures(
    exposure_source: str | Path,
    *,
    mutation_type: str,
    sample_ids: Sequence[str] | None = None,
    signature_names: Sequence[str] | None = None,
    source_label: str | None = None,
) -> GroundTruthSet:
    exposures = pd.read_csv(exposure_source, index_col=0)
    if sample_ids is not None:
        exposures = exposures.reindex(columns=list(sample_ids), fill_value=0.0)
    if signature_names is not None:
        exposures = exposures.reindex(index=list(signature_names), fill_value=0.0)
    return GroundTruthSet(
        mutation_type=mutation_type,
        exposures=exposures.astype(float),
        source=source_label or str(exposure_source),
    )

def _default_catalog_insufficiency_score(sample_result: ExpertSampleResult) -> float:
    return catalog_insufficiency_score_from_sample_result(sample_result)


def _sample_group_frame(
    request: ExpertRequest,
    *,
    report_by_sample: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_id in request.sample_ids:
        report = None if report_by_sample is None else report_by_sample.get(sample_id)
        agreement_score = None
        risk_level = None
        if report is not None:
            agreement_score = float(report.metadata.get("agreement_score", 0.0))
            risk_level = report.catalog_insufficiency_level
        context = group_context_from_sample_counts(
            request.sample_matrix.loc[:, sample_id],
            mutation_type=request.mutation_type,
            disagreement_score=None if agreement_score is None else 1.0 - agreement_score,
            risk_level=risk_level,
        ).to_dict()
        rows.append(
            {
                "sample_id": sample_id,
                **context,
            }
        )
    return pd.DataFrame.from_records(rows)


def _evaluate_runs_for_slice(
    *,
    runs,
    request: ExpertRequest,
    truth_exposures: pd.DataFrame,
    benchmark_name: str,
    slice_parameters: dict[str, Any],
    extra_sample_fields: pd.DataFrame | None = None,
) -> BenchmarkSliceResult:
    aggregate_rows: list[dict[str, Any]] = []
    per_sample_frames: list[pd.DataFrame] = []

    for run in runs:
        aggregate, sample_metrics = evaluate_expert_run(
            run,
            sample_matrix=request.sample_matrix,
            signature_matrix=request.signature_matrix,
            truth_exposures=truth_exposures,
        )
        aggregate["benchmark_name"] = benchmark_name
        aggregate.update(slice_parameters)
        aggregate_rows.append(aggregate)
        if not sample_metrics.empty:
            sample_metrics.insert(0, "expert_name", run.expert_name)
            sample_metrics.insert(1, "benchmark_name", benchmark_name)
            for key, value in slice_parameters.items():
                sample_metrics[key] = value
            if extra_sample_fields is not None and not extra_sample_fields.empty:
                sample_metrics = sample_metrics.merge(extra_sample_fields, on="sample_id", how="left")
            per_sample_frames.append(sample_metrics)

    return BenchmarkSliceResult(
        benchmark_name=benchmark_name,
        mutation_type=request.mutation_type,
        aggregate_metrics=pd.DataFrame.from_records(aggregate_rows),
        per_sample_metrics=pd.concat(per_sample_frames, ignore_index=True) if per_sample_frames else pd.DataFrame(),
        parameters=slice_parameters,
    )


def _fusion_evidence_frame(
    *,
    fusion_output,
    benchmark_name: str,
    slice_parameters: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_result in fusion_output.fused_run.sample_results:
        evidence = (sample_result.diagnostics or {}).get("fusion_evidence_features") or {}
        if not evidence:
            continue
        rows.append(
            {
                "benchmark_name": benchmark_name,
                **slice_parameters,
                **dict(evidence),
            }
        )
    return pd.DataFrame.from_records(rows)


def _removal_metadata_by_signature(removal_manifest_source: str | Path | None) -> dict[str, dict[str, Any]]:
    if not removal_manifest_source:
        return {}
    manifest = load_catalog_removal_design(removal_manifest_source)
    if manifest.empty or "signature_name" not in manifest.columns:
        return {}
    metadata_by_signature: dict[str, dict[str, Any]] = {}
    excluded_columns = {"signature_name"}
    for signature_name, group in manifest.groupby("signature_name", sort=False):
        metadata: dict[str, Any] = {}
        if "selection_group" in group.columns:
            metadata["removal_selection_groups"] = ",".join(
                sorted({str(value) for value in group["selection_group"].dropna().tolist()})
            )
        for column in group.columns:
            if column in excluded_columns or column == "selection_group":
                continue
            values = group[column].dropna()
            if values.empty:
                continue
            metadata[f"removal_{column}"] = values.iloc[0]
        metadata_by_signature[str(signature_name)] = metadata
    return metadata_by_signature


def _candidate_removed_signatures(
    base_signature_names: Sequence[str],
    *,
    removed_signatures: Sequence[str] | None,
    removal_manifest_source: str | Path | None,
) -> list[str]:
    if removed_signatures is not None:
        return list(removed_signatures)
    if removal_manifest_source:
        manifest = load_catalog_removal_design(removal_manifest_source, benchmarkable_only=True)
        if not manifest.empty and "signature_name" in manifest.columns:
            return list(dict.fromkeys(str(value) for value in manifest["signature_name"].tolist()))
    return list(base_signature_names)


def run_known_catalog_benchmark(
    *,
    sample_source: str | Path,
    signature_source: str | Path,
    exposure_source: str | Path,
    mutation_type: str,
    burdens: Sequence[int] = (100, 200, 500, 2000, 50000),
    max_samples_per_burden: int = 100,
    random_seed: int = 0,
    expert_names: Sequence[str] | None = None,
    registry: ExpertRegistry | None = None,
    include_rule_fusion: bool = False,
    confidence_artifacts: Any | None = None,
    catalog_assessor_model: Any | None = None,
    bootstrap_config: Any | None = None,
    assignment_f1_threshold: float = 0.8,
) -> BenchmarkSuiteResult:
    registry = registry or build_default_registry(".", confidence_artifacts=confidence_artifacts)
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
    rng = np.random.default_rng(random_seed)
    slices: list[BenchmarkSliceResult] = []

    available_sample_ids = base_request.sample_ids
    for burden in burdens:
        chosen_sample_ids = available_sample_ids
        if max_samples_per_burden and len(available_sample_ids) > max_samples_per_burden:
            chosen_sample_ids = sorted(rng.choice(available_sample_ids, size=max_samples_per_burden, replace=False).tolist())
        truth_subset = subset_samples(truth.exposures, chosen_sample_ids)
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
            request_id=f"known_catalog_{base_request.mutation_type}_{burden}",
            alignment_strategy=base_request.alignment_strategy,
        )
        runs = registry.run_all(request, list(expert_names) if expert_names is not None else None)
        fusion_output = None
        if include_rule_fusion:
            fusion_output = fuse_expert_runs(
                runs,
                request,
                confidence_artifacts=confidence_artifacts,
                catalog_assessor_model=catalog_assessor_model,
                bootstrap_config=bootstrap_config,
            )
            runs = runs + [fusion_output.fused_run]
        report_by_sample = None if fusion_output is None else {report.sample_id: report for report in fusion_output.reports}
        extra_sample_fields = _sample_group_frame(request, report_by_sample=report_by_sample)
        slice_result = _evaluate_runs_for_slice(
            runs=runs,
            request=request,
            truth_exposures=scaled_truth,
            benchmark_name="known_catalog",
            slice_parameters={
                "mutation_type": request.mutation_type,
                "burden": int(burden),
                "n_samples": int(len(chosen_sample_ids)),
            },
            extra_sample_fields=extra_sample_fields,
        )
        if fusion_output is not None and fusion_output.reports:
            evidence_frame = _fusion_evidence_frame(
                fusion_output=fusion_output,
                benchmark_name="known_catalog",
                slice_parameters={
                    "mutation_type": request.mutation_type,
                    "burden": int(burden),
                    "n_samples": int(len(chosen_sample_ids)),
                },
            )
            if not evidence_frame.empty:
                slice_result.artifacts["fusion_evidence_features"] = evidence_frame
            fused_exposures = pd.DataFrame(
                {
                    sample_result.sample_id: sample_result.exposures
                    for sample_result in fusion_output.fused_run.sample_results
                }
            ).reindex(index=request.signature_names, fill_value=0.0)
            _, active_per_sample = evaluate_expert_run(
                fusion_output.fused_run,
                sample_matrix=request.sample_matrix,
                signature_matrix=request.signature_matrix,
                truth_exposures=scaled_truth,
            )
            labels = (active_per_sample.set_index("sample_id")["active_set_f1"] >= assignment_f1_threshold).astype(int)
            scores = pd.Series(
                {
                    report.sample_id: report.assignment_confidence_probability
                    for report in fusion_output.reports
                }
            )
            assignment_metrics = binary_probability_metrics(labels, scores, prefix="assignment_confidence")
            interval_metrics, interval_per_sample = exposure_interval_metrics(
                scaled_truth,
                fusion_output.fused_run,
                prefix="exposure_interval",
            )
            conformal_metrics, conformal_per_sample = exposure_interval_metrics(
                scaled_truth,
                fusion_output.fused_run,
                prefix="conformal_interval",
                source_filter="bootstrap_conformal",
            )
            mask = slice_result.aggregate_metrics["expert_name"] == "rule_fusion"
            for key, value in assignment_metrics.items():
                slice_result.aggregate_metrics.loc[mask, key] = value
            for key, value in interval_metrics.items():
                slice_result.aggregate_metrics.loc[mask, key] = value
            for key, value in conformal_metrics.items():
                slice_result.aggregate_metrics.loc[mask, key] = value
            if not slice_result.per_sample_metrics.empty:
                expert_mask = slice_result.per_sample_metrics["expert_name"] == "rule_fusion"
                slice_result.per_sample_metrics.loc[expert_mask, "assignment_confidence_probability"] = slice_result.per_sample_metrics.loc[
                    expert_mask, "sample_id"
                ].map(scores)
                if not interval_per_sample.empty:
                    interval_map = interval_per_sample.set_index("sample_id")
                    for column in [
                        "exposure_interval_coverage",
                        "exposure_interval_active_coverage",
                        "exposure_interval_mean_width",
                    ]:
                        slice_result.per_sample_metrics.loc[expert_mask, column] = slice_result.per_sample_metrics.loc[
                            expert_mask, "sample_id"
                        ].map(interval_map[column])
                if not conformal_per_sample.empty:
                    conformal_map = conformal_per_sample.set_index("sample_id")
                    for column in [
                        "conformal_interval_coverage",
                        "conformal_interval_active_coverage",
                        "conformal_interval_mean_width",
                    ]:
                        slice_result.per_sample_metrics.loc[expert_mask, column] = slice_result.per_sample_metrics.loc[
                            expert_mask, "sample_id"
                        ].map(conformal_map[column])
                rule_frame = slice_result.per_sample_metrics.loc[expert_mask].copy()
                if not rule_frame.empty:
                    rule_frame["assignment_confidence_label"] = rule_frame["sample_id"].map(labels)
                    assignment_group_metrics = groupwise_probability_metrics(
                        rule_frame,
                        label_column="assignment_confidence_label",
                        score_column="assignment_confidence_probability",
                        group_columns=list(ASSIGNMENT_GROUP_COLUMNS),
                        prefix="assignment_confidence",
                    )
                    interval_group_metrics = groupwise_interval_metrics(
                        rule_frame,
                        coverage_column="conformal_interval_coverage",
                        width_column="conformal_interval_mean_width",
                        active_coverage_column="conformal_interval_active_coverage",
                        group_columns=list(EXPOSURE_GROUP_COLUMNS),
                        prefix="conformal_interval",
                        target_coverage=1.0 - (bootstrap_config.alpha if bootstrap_config is not None else 0.1),
                    )
                    if not assignment_group_metrics.empty:
                        assignment_group_metrics.insert(0, "expert_name", "rule_fusion")
                        assignment_group_metrics["burden"] = int(burden)
                        assignment_group_metrics["benchmark_name"] = "known_catalog"
                    if not interval_group_metrics.empty:
                        interval_group_metrics.insert(0, "expert_name", "rule_fusion")
                        interval_group_metrics["burden"] = int(burden)
                        interval_group_metrics["benchmark_name"] = "known_catalog"
                    slice_result.artifacts["assignment_group_metrics"] = assignment_group_metrics
                    slice_result.artifacts["interval_group_metrics"] = interval_group_metrics
                    for key, value in summarize_group_metric_frame(
                        assignment_group_metrics,
                        ece_column="assignment_confidence_ece",
                        prefix="assignment_confidence",
                    ).items():
                        slice_result.aggregate_metrics.loc[mask, key] = value
                    for key, value in summarize_group_metric_frame(
                        interval_group_metrics,
                        coverage_gap_column="conformal_interval_coverage_gap",
                        width_column="conformal_interval_mean_width",
                        prefix="conformal_interval",
                    ).items():
                        slice_result.aggregate_metrics.loc[mask, key] = value
        slices.append(slice_result)

    return BenchmarkSuiteResult(
        benchmark_name="known_catalog",
        mutation_type=base_request.mutation_type,
        slices=slices,
        metadata={
            "sample_source": str(sample_source),
            "signature_source": str(signature_source),
            "exposure_source": str(exposure_source),
            "random_seed": random_seed,
            "include_rule_fusion": include_rule_fusion,
            "assignment_f1_threshold": assignment_f1_threshold,
            "confidence_artifacts_loaded": confidence_artifacts is not None,
            "catalog_assessor_loaded": catalog_assessor_model is not None,
            "expert_names": list(expert_names) if expert_names is not None else registry.default_names(),
        },
    )


def run_catalog_insufficiency_benchmark(
    *,
    sample_source: str | Path,
    signature_source: str | Path,
    exposure_source: str | Path,
    mutation_type: str,
    burdens: Sequence[int] = (200, 2000),
    removed_signatures: Sequence[str] | None = None,
    removal_manifest_source: str | Path | None = None,
    max_positive_per_signature: int = 50,
    max_negative_per_signature: int = 50,
    active_threshold: float = 0.0,
    random_seed: int = 0,
    expert_names: Sequence[str] | None = None,
    registry: ExpertRegistry | None = None,
    score_extractor: Callable[[ExpertSampleResult], float] | None = None,
    include_rule_fusion: bool = False,
    confidence_artifacts: Any | None = None,
    catalog_assessor_model: Any | None = None,
    bootstrap_config: Any | None = None,
) -> BenchmarkSuiteResult:
    registry = registry or build_default_registry(".", confidence_artifacts=confidence_artifacts)
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
    truth_norm = normalize_exposures(truth.exposures)
    rng = np.random.default_rng(random_seed)
    score_extractor = score_extractor or _default_catalog_insufficiency_score
    candidate_removed_signatures = _candidate_removed_signatures(
        base_request.signature_names,
        removed_signatures=removed_signatures,
        removal_manifest_source=removal_manifest_source,
    )
    removal_metadata_by_signature = _removal_metadata_by_signature(removal_manifest_source)
    slices: list[BenchmarkSliceResult] = []

    for burden in burdens:
        for removed_signature in candidate_removed_signatures:
            if removed_signature not in truth_norm.index:
                continue
            positive_mask = truth_norm.loc[removed_signature] > active_threshold
            negative_mask = ~positive_mask
            positive_ids = truth_norm.columns[positive_mask].tolist()
            negative_ids = truth_norm.columns[negative_mask].tolist()
            if not positive_ids or not negative_ids:
                continue
            if max_positive_per_signature and len(positive_ids) > max_positive_per_signature:
                positive_ids = sorted(rng.choice(positive_ids, size=max_positive_per_signature, replace=False).tolist())
            if max_negative_per_signature and len(negative_ids) > max_negative_per_signature:
                negative_ids = sorted(rng.choice(negative_ids, size=max_negative_per_signature, replace=False).tolist())
            chosen_sample_ids = positive_ids + negative_ids
            if not chosen_sample_ids:
                continue
            removal_metadata = removal_metadata_by_signature.get(str(removed_signature), {})

            truth_subset_full = subset_samples(truth.exposures, chosen_sample_ids)
            simulated_samples = simulate_counts_from_truth(
                base_request.signature_matrix,
                truth_subset_full,
                burden=burden,
                rng=rng,
            )
            incomplete_catalog = base_request.signature_matrix.drop(columns=[removed_signature])
            truth_subset_available = truth_subset_full.drop(index=[removed_signature])
            scaled_truth_available = scaled_truth_exposures(truth_subset_available, burden)
            request = ExpertRequest(
                mutation_type=base_request.mutation_type,
                sample_matrix=simulated_samples,
                signature_matrix=incomplete_catalog,
                channel_metadata=base_request.channel_metadata.copy() if base_request.channel_metadata is not None else None,
                sample_source=str(sample_source),
                signature_source=str(signature_source),
                reference_name=f"{base_request.reference_name}_without_{removed_signature}",
                request_id=f"catalog_insufficiency_{base_request.mutation_type}_{removed_signature}_{burden}",
                alignment_strategy=base_request.alignment_strategy,
            )
            labels = pd.Series(
                {
                    sample_id: int(sample_id in positive_ids)
                    for sample_id in chosen_sample_ids
                },
                name="catalog_insufficient_label",
            )
            runs = registry.run_all(request, list(expert_names) if expert_names is not None else None)
            fusion_output = None
            if include_rule_fusion:
                fusion_output = fuse_expert_runs(
                    runs,
                    request,
                    confidence_artifacts=confidence_artifacts,
                    catalog_assessor_model=catalog_assessor_model,
                    bootstrap_config=bootstrap_config,
                )
                runs = runs + [fusion_output.fused_run]
            report_by_sample = None if fusion_output is None else {report.sample_id: report for report in fusion_output.reports}
            extra_sample_fields = _sample_group_frame(request, report_by_sample=report_by_sample).merge(
                labels.rename_axis("sample_id").reset_index(),
                on="sample_id",
                how="left",
            )
            slice_result = _evaluate_runs_for_slice(
                runs=runs,
                request=request,
                truth_exposures=scaled_truth_available,
                benchmark_name="catalog_insufficiency",
                slice_parameters={
                    "mutation_type": request.mutation_type,
                    "burden": int(burden),
                    "removed_signature": removed_signature,
                    "n_positive": int(len(positive_ids)),
                    "n_negative": int(len(negative_ids)),
                    **removal_metadata,
                },
                extra_sample_fields=extra_sample_fields,
            )

            aggregate_rows = []
            for run in runs:
                score_map = {}
                for sample_result in run.sample_results:
                    score_map[sample_result.sample_id] = float(score_extractor(sample_result))
                scores = pd.Series(score_map, name="catalog_insufficiency_score")
                insuff_metrics = catalog_insufficiency_metrics(labels, scores)
                aggregate_rows.append(
                    {
                        "expert_name": run.expert_name,
                        "burden": int(burden),
                        "removed_signature": removed_signature,
                        **insuff_metrics,
                    }
                )
                if not slice_result.per_sample_metrics.empty:
                    expert_mask = slice_result.per_sample_metrics["expert_name"] == run.expert_name
                    slice_result.per_sample_metrics.loc[expert_mask, "catalog_insufficiency_score"] = slice_result.per_sample_metrics.loc[
                        expert_mask, "sample_id"
                    ].map(scores)
                    slice_result.per_sample_metrics.loc[expert_mask, "catalog_insufficiency_level"] = slice_result.per_sample_metrics.loc[
                        expert_mask, "sample_id"
                    ].map(
                        {
                            sample_result.sample_id: catalog_insufficiency_level_from_sample_result(sample_result)
                            for sample_result in run.sample_results
                        }
                    )
                    slice_result.per_sample_metrics.loc[expert_mask, "catalog_insufficiency_probability"] = slice_result.per_sample_metrics.loc[
                        expert_mask, "sample_id"
                    ].map(
                        {
                            sample_result.sample_id: catalog_insufficiency_probability_from_sample_result(sample_result)
                            for sample_result in run.sample_results
                        }
                    )

            if aggregate_rows:
                insuff_df = pd.DataFrame.from_records(aggregate_rows)
                slice_result.aggregate_metrics = slice_result.aggregate_metrics.merge(
                    insuff_df,
                    on=["expert_name", "burden", "removed_signature"],
                    how="left",
                )
            if fusion_output is not None and fusion_output.reports:
                evidence_frame = _fusion_evidence_frame(
                    fusion_output=fusion_output,
                    benchmark_name="catalog_insufficiency",
                    slice_parameters={
                        "mutation_type": request.mutation_type,
                        "burden": int(burden),
                        "removed_signature": removed_signature,
                        "n_positive": int(len(positive_ids)),
                        "n_negative": int(len(negative_ids)),
                        **removal_metadata,
                    },
                )
                if not evidence_frame.empty:
                    slice_result.artifacts["fusion_evidence_features"] = evidence_frame
                scores = pd.Series(
                    {
                        report.sample_id: report.catalog_insufficiency_probability
                        for report in fusion_output.reports
                    }
                )
                probability_metrics = binary_probability_metrics(
                    labels,
                    scores,
                    prefix="catalog_insufficiency_probability",
                )
                mask = slice_result.aggregate_metrics["expert_name"] == "rule_fusion"
                for key, value in probability_metrics.items():
                    slice_result.aggregate_metrics.loc[mask, key] = value
                interval_metrics, interval_per_sample = exposure_interval_metrics(
                    scaled_truth_available,
                    fusion_output.fused_run,
                    prefix="exposure_interval",
                )
                conformal_metrics, conformal_per_sample = exposure_interval_metrics(
                    scaled_truth_available,
                    fusion_output.fused_run,
                    prefix="conformal_interval",
                    source_filter="bootstrap_conformal",
                )
                for key, value in interval_metrics.items():
                    slice_result.aggregate_metrics.loc[mask, key] = value
                for key, value in conformal_metrics.items():
                    slice_result.aggregate_metrics.loc[mask, key] = value
                if not slice_result.per_sample_metrics.empty:
                    expert_mask = slice_result.per_sample_metrics["expert_name"] == "rule_fusion"
                    if not interval_per_sample.empty:
                        interval_map = interval_per_sample.set_index("sample_id")
                        for column in [
                            "exposure_interval_coverage",
                            "exposure_interval_active_coverage",
                            "exposure_interval_mean_width",
                        ]:
                            slice_result.per_sample_metrics.loc[expert_mask, column] = slice_result.per_sample_metrics.loc[
                                expert_mask, "sample_id"
                            ].map(interval_map[column])
                    if not conformal_per_sample.empty:
                        conformal_map = conformal_per_sample.set_index("sample_id")
                        for column in [
                            "conformal_interval_coverage",
                            "conformal_interval_active_coverage",
                            "conformal_interval_mean_width",
                        ]:
                            slice_result.per_sample_metrics.loc[expert_mask, column] = slice_result.per_sample_metrics.loc[
                                expert_mask, "sample_id"
                            ].map(conformal_map[column])
                    rule_frame = slice_result.per_sample_metrics.loc[expert_mask].copy()
                    if not rule_frame.empty:
                        probability_group_metrics = groupwise_probability_metrics(
                            rule_frame,
                            label_column="catalog_insufficient_label",
                            score_column="catalog_insufficiency_probability",
                            group_columns=list(CATALOG_ASSESSOR_GROUP_COLUMNS),
                            prefix="catalog_insufficiency_probability",
                        )
                        interval_group_metrics = groupwise_interval_metrics(
                            rule_frame,
                            coverage_column="conformal_interval_coverage",
                            width_column="conformal_interval_mean_width",
                            active_coverage_column="conformal_interval_active_coverage",
                            group_columns=list(EXPOSURE_GROUP_COLUMNS),
                            prefix="conformal_interval",
                            target_coverage=1.0 - (bootstrap_config.alpha if bootstrap_config is not None else 0.1),
                        )
                        if not probability_group_metrics.empty:
                            probability_group_metrics.insert(0, "expert_name", "rule_fusion")
                            probability_group_metrics["burden"] = int(burden)
                            probability_group_metrics["removed_signature"] = removed_signature
                            probability_group_metrics["benchmark_name"] = "catalog_insufficiency"
                        if not interval_group_metrics.empty:
                            interval_group_metrics.insert(0, "expert_name", "rule_fusion")
                            interval_group_metrics["burden"] = int(burden)
                            interval_group_metrics["removed_signature"] = removed_signature
                            interval_group_metrics["benchmark_name"] = "catalog_insufficiency"
                        slice_result.artifacts["catalog_probability_group_metrics"] = probability_group_metrics
                        slice_result.artifacts["interval_group_metrics"] = interval_group_metrics
                        for key, value in summarize_group_metric_frame(
                            probability_group_metrics,
                            ece_column="catalog_insufficiency_probability_ece",
                            prefix="catalog_insufficiency_probability",
                        ).items():
                            slice_result.aggregate_metrics.loc[mask, key] = value
                        for key, value in summarize_group_metric_frame(
                            interval_group_metrics,
                            coverage_gap_column="conformal_interval_coverage_gap",
                            width_column="conformal_interval_mean_width",
                            prefix="conformal_interval",
                        ).items():
                            slice_result.aggregate_metrics.loc[mask, key] = value
            slices.append(slice_result)

    return BenchmarkSuiteResult(
        benchmark_name="catalog_insufficiency",
        mutation_type=base_request.mutation_type,
        slices=slices,
        metadata={
            "sample_source": str(sample_source),
            "signature_source": str(signature_source),
            "exposure_source": str(exposure_source),
            "random_seed": random_seed,
            "active_threshold": active_threshold,
            "include_rule_fusion": include_rule_fusion,
            "removal_manifest_source": None if removal_manifest_source is None else str(removal_manifest_source),
            "confidence_artifacts_loaded": confidence_artifacts is not None,
            "catalog_assessor_loaded": catalog_assessor_model is not None,
            "expert_names": list(expert_names) if expert_names is not None else registry.default_names(),
        },
    )


__all__ = [
    "load_truth_exposures",
    "run_catalog_insufficiency_benchmark",
    "run_known_catalog_benchmark",
]
