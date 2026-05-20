#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signature_decision import (  # noqa: E402
    BootstrapConfig,
    CatalogInsufficiencyModel,
    ConfidenceArtifacts,
    CORE_EXPERT_NAMES,
    DiscoveryTriggerConfig,
    ExperienceStore,
    LocalExtractionConfig,
    ReviewDecision,
    aggregate_cohort_reports,
    build_default_registry,
    build_decision_experience_records,
    build_experience_dataset,
    build_review_queue_output,
    fit_catalog_insufficiency_model_from_benchmark,
    fit_confidence_artifacts_from_known_catalog,
    fuse_expert_runs,
    load_catalog_removal_design,
    load_expert_request,
    reports_to_frame,
    run_catalog_insufficiency_benchmark,
    run_known_catalog_benchmark,
    run_conservative_discovery_workflow,
    utc_now_iso,
    write_experience_dataset,
    write_review_queues,
    write_runs,
)


def _parse_csv_list(value: str | None, *, cast=str) -> list:
    if value is None:
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _write_benchmark_result(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "result.json", result.to_dict())
    result.aggregate_metrics.to_csv(output_dir / "aggregate_metrics.tsv", sep="\t", index=False)
    result.per_sample_metrics.to_csv(output_dir / "per_sample_metrics.tsv", sep="\t", index=False)
    artifact_frames: dict[str, list[pd.DataFrame]] = {}
    for slice_result in getattr(result, "slices", []):
        for artifact_name, artifact_value in getattr(slice_result, "artifacts", {}).items():
            if isinstance(artifact_value, pd.DataFrame) and not artifact_value.empty:
                artifact_frames.setdefault(artifact_name, []).append(artifact_value)
    for artifact_name, frames in artifact_frames.items():
        pd.concat(frames, ignore_index=True).to_csv(output_dir / f"{artifact_name}.tsv", sep="\t", index=False)


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    if normalized in {"none", "null", "unknown"}:
        return None
    raise ValueError(f"Unsupported optional bool value: {value}")


def _default_expert_names() -> list[str]:
    return list(CORE_EXPERT_NAMES)


def _load_confidence_artifacts(path: str | None) -> ConfidenceArtifacts | None:
    if not path:
        return None
    return ConfidenceArtifacts.load(path)


def _load_catalog_assessor(path: str | None) -> CatalogInsufficiencyModel | None:
    if not path:
        return None
    return CatalogInsufficiencyModel.load(path)


def _removed_signatures_from_manifest(path: str | None) -> list[str]:
    if not path:
        return []
    frame = load_catalog_removal_design(path, benchmarkable_only=True)
    if frame.empty or "signature_name" not in frame.columns:
        return []
    return list(dict.fromkeys(str(value) for value in frame["signature_name"].tolist()))


def _resolve_removed_signatures(args: argparse.Namespace) -> list[str] | None:
    explicit = _parse_csv_list(getattr(args, "removed_signatures", None))
    if explicit:
        return explicit
    manifest_signatures = _removed_signatures_from_manifest(getattr(args, "removal_manifest", None))
    return manifest_signatures or None


def _discovery_run_token(value: str) -> str:
    return value.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")


def _load_discovery_request(
    *,
    records,
    sample_source: str | None,
    signature_source: str | None,
) -> tuple[object | None, list[str]]:
    if not records:
        return None, ["No experience records available for discovery."]
    source_record = records[0]
    resolved_sample_source = sample_source or source_record.source_context.get("sample_source")
    resolved_signature_source = signature_source or source_record.source_context.get("signature_source")
    if not resolved_sample_source or not resolved_signature_source:
        warnings = ["Could not resolve sample_source/signature_source from arguments or experience records."]
        return None, warnings
    try:
        request = load_expert_request(
            sample_source=resolved_sample_source,
            signature_source=resolved_signature_source,
            mutation_type=source_record.mutation_type,
        )
        return request, []
    except Exception as exc:
        return None, [f"Failed to load discovery reference inputs: {exc}"]


def _flatten_discovery_packets(packets) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    packet_rows = []
    component_rows = []
    fit_rows = []
    for packet in packets:
        packet_rows.append(packet.to_index_row())
        for component in packet.extracted_components:
            component_rows.append(
                {
                    "packet_id": packet.packet_id,
                    "component_id": component.get("component_id"),
                    "recurrence_count": component.get("recurrence_count"),
                    "stability_score": component.get("stability_score"),
                    "catalog_match_name": component.get("catalog_match_name"),
                    "catalog_match_cosine": component.get("catalog_match_cosine"),
                    "mean_residual_mass": component.get("mean_residual_mass"),
                }
            )
        fit_summary = packet.fit_improvement_summary or {}
        aggregate = fit_summary.get("aggregate") or {}
        if aggregate:
            fit_rows.append(
                {
                    "packet_id": packet.packet_id,
                    "row_type": "aggregate",
                    **aggregate,
                }
            )
        for row in fit_summary.get("per_sample") or []:
            fit_rows.append(
                {
                    "packet_id": packet.packet_id,
                    "row_type": "per_sample",
                    **row,
                }
            )
    return (
        pd.DataFrame.from_records(packet_rows),
        pd.DataFrame.from_records(component_rows),
        pd.DataFrame.from_records(fit_rows),
    )


def run_decision(args: argparse.Namespace) -> int:
    request = load_expert_request(
        sample_source=args.sample_source,
        signature_source=args.signature_source,
        mutation_type=args.mutation_type,
    )
    sample_ids = _parse_csv_list(args.sample_ids)
    if sample_ids:
        request = request.with_samples(sample_ids)

    confidence_artifacts = _load_confidence_artifacts(args.confidence_artifact)
    catalog_assessor_model = _load_catalog_assessor(args.catalog_assessor_artifact)
    expert_names = _parse_csv_list(args.expert_names) or _default_expert_names()
    registry = build_default_registry(REPO_ROOT, confidence_artifacts=confidence_artifacts)
    runs = registry.run_all(request, expert_names)
    bootstrap_config = None
    if args.bootstrap_replicates > 0:
        bootstrap_config = BootstrapConfig(
            n_replicates=args.bootstrap_replicates,
            alpha=args.bootstrap_alpha,
            random_seed=args.bootstrap_random_seed,
            use_conformal=not args.disable_bootstrap_conformal,
        )
    fusion_output = fuse_expert_runs(
        runs,
        request,
        confidence_artifacts=confidence_artifacts,
        catalog_assessor_model=catalog_assessor_model,
        bootstrap_config=bootstrap_config,
    )

    output_dir = Path(args.output_dir)
    expert_dir = output_dir / "experts"
    fusion_dir = output_dir / "fusion"
    cohort_dir = output_dir / "cohort"
    write_runs(runs, expert_dir)
    _write_json(fusion_dir / "fused_run.json", fusion_output.fused_run.to_dict())
    _write_json(fusion_dir / "reports.json", [report.to_dict() for report in fusion_output.reports])
    reports_to_frame(fusion_output.reports).to_csv(fusion_dir / "reports.tsv", sep="\t", index=False)
    fusion_output.summary_frame.to_csv(fusion_dir / "summary.tsv", sep="\t", index=False)
    cohort_output = aggregate_cohort_reports(fusion_output.reports)
    _write_json(cohort_dir / "summary.json", cohort_output.report.to_dict())
    cohort_output.summary_frame.to_csv(cohort_dir / "summary.tsv", sep="\t", index=False)
    cohort_output.candidates_frame.to_csv(cohort_dir / "candidates.tsv", sep="\t", index=False)
    experience_dir = Path(args.experience_dir) if args.experience_dir else output_dir / "experience"
    experience_store = ExperienceStore(experience_dir)
    records = build_decision_experience_records(
        request=request,
        runs=runs,
        fusion_output=fusion_output,
        cohort_output=cohort_output,
        confidence_artifact_path=args.confidence_artifact,
        catalog_assessor_artifact_path=args.catalog_assessor_artifact,
        run_label=args.run_label or output_dir.name,
        output_dir=output_dir,
    )
    experience_store.append_records(records)
    queue_output = build_review_queue_output(
        experience_store.load_records(),
        experience_store.load_review_decisions(),
    )
    write_review_queues(queue_output, experience_dir)
    dataset_frame = build_experience_dataset(
        experience_store.load_records(),
        experience_store.load_review_decisions(),
    )
    write_experience_dataset(dataset_frame, experience_store.datasets_dir / "high_quality_dataset.tsv")
    return 0


def run_append_review(args: argparse.Namespace) -> int:
    experience_store = ExperienceStore(args.experience_dir)
    labels = json.loads(args.labels_json) if args.labels_json else {}
    review = ReviewDecision(
        review_id="__".join(
            [
                utc_now_iso().replace("-", "").replace(":", "").replace("T", "_").replace("Z", ""),
                args.record_id,
                args.reviewer,
            ]
        ),
        record_id=args.record_id,
        created_at=utc_now_iso(),
        reviewer=args.reviewer,
        review_outcome=args.review_outcome,
        evidence_type=args.evidence_type,
        validated_recommendation=args.validated_recommendation,
        catalog_insufficiency_confirmed=_parse_optional_bool(args.catalog_insufficiency_confirmed),
        reference_reassessment_confirmed=_parse_optional_bool(args.reference_reassessment_confirmed),
        preprocessing_issue_flag=_parse_optional_bool(args.preprocessing_issue_flag),
        notes=args.notes,
        labels=labels,
    )
    experience_store.append_review_decisions([review])
    records = experience_store.load_records()
    reviews = experience_store.load_review_decisions()
    queue_output = build_review_queue_output(records, reviews)
    write_review_queues(queue_output, experience_store.root_dir)
    dataset_frame = build_experience_dataset(records, reviews, include_all_reviewed=args.include_all_reviewed)
    write_experience_dataset(dataset_frame, experience_store.datasets_dir / "high_quality_dataset.tsv")
    if args.include_all_reviewed:
        write_experience_dataset(dataset_frame, experience_store.datasets_dir / "reviewed_dataset.tsv")
    return 0


def run_export_experience_dataset(args: argparse.Namespace) -> int:
    experience_store = ExperienceStore(args.experience_dir)
    frame = build_experience_dataset(
        experience_store.load_records(),
        experience_store.load_review_decisions(),
        include_all_reviewed=args.include_all_reviewed,
    )
    write_experience_dataset(frame, args.output_path)
    return 0


def run_discovery(args: argparse.Namespace) -> int:
    experience_store = ExperienceStore(args.experience_dir)
    records = experience_store.load_records()
    reviews = experience_store.load_review_decisions()
    request, load_warnings = _load_discovery_request(
        records=records,
        sample_source=args.sample_source,
        signature_source=args.signature_source,
    )
    trigger_config = DiscoveryTriggerConfig(
        probability_threshold=args.probability_threshold,
        residual_structure_threshold=args.residual_structure_threshold,
        min_recurrence_count=args.min_recurrence_count,
        mutation_count_threshold=args.mutation_count_threshold,
        similarity_threshold=args.similarity_threshold,
        min_cluster_size=args.min_cluster_size,
        require_cohort_candidate=not args.disable_cohort_candidate_gate,
        require_review_confirmation=not args.disable_review_confirmation,
        allow_pending_manual_review=args.allow_pending_manual_review,
    )
    extraction_config = LocalExtractionConfig(
        max_components=args.max_components,
        min_records=args.min_records,
        min_mean_residual_mass=args.min_mean_residual_mass,
        min_error_gain=args.min_error_gain,
        min_component_weight_fraction=args.min_component_weight_fraction,
        bootstrap_repeats=args.bootstrap_repeats,
        max_iter=args.max_iter,
        random_seed=args.random_seed,
    )
    created_at = utc_now_iso()
    run_token = _discovery_run_token(args.run_label or created_at)
    workflow = run_conservative_discovery_workflow(
        records,
        reviews=reviews,
        trigger_config=trigger_config,
        extraction_config=extraction_config,
        created_at=created_at,
        signature_matrix=None if request is None else request.signature_matrix,
        channel_ids=None if request is None else request.channel_ids,
        packet_id_prefix=f"discovery_packet__{run_token}",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters_frame = pd.DataFrame.from_records(
        [cluster.to_dict() for cluster in workflow.trigger_output.recurrence_clusters]
    )
    packets_frame, components_frame, fit_frame = _flatten_discovery_packets(workflow.packets)
    summary = {
        **workflow.trigger_output.summary,
        "n_packets": len(workflow.packets),
        "load_warnings": load_warnings,
        "persisted_packets": bool(workflow.packets) and not args.skip_experience_persist,
        "resolved_sample_source": None if request is None else request.sample_source,
        "resolved_signature_source": None if request is None else request.signature_source,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "trigger_output.json", workflow.trigger_output.to_dict())
    workflow.trigger_output.candidates_frame.to_csv(output_dir / "trigger_candidates.tsv", sep="\t", index=False)
    clusters_frame.to_csv(output_dir / "recurrence_clusters.tsv", sep="\t", index=False)
    _write_json(output_dir / "packets.json", [packet.to_dict() for packet in workflow.packets])
    packets_frame.to_csv(output_dir / "packets.tsv", sep="\t", index=False)
    components_frame.to_csv(output_dir / "extracted_components.tsv", sep="\t", index=False)
    fit_frame.to_csv(output_dir / "fit_improvements.tsv", sep="\t", index=False)
    if workflow.packets and not args.skip_experience_persist:
        experience_store.append_discovery_packets(workflow.packets)
    return 0


def run_fit_confidence(args: argparse.Namespace) -> int:
    burdens = tuple(_parse_csv_list(args.burdens, cast=int))
    expert_names = _parse_csv_list(args.expert_names) or _default_expert_names()
    artifacts = fit_confidence_artifacts_from_known_catalog(
        sample_source=args.sample_source,
        signature_source=args.signature_source,
        exposure_source=args.exposure_source,
        mutation_type=args.mutation_type,
        burdens=burdens,
        max_samples_per_burden=args.max_samples_per_burden,
        random_seed=args.random_seed,
        registry=build_default_registry(REPO_ROOT),
        expert_names=expert_names,
        active_threshold=args.active_threshold,
        assignment_f1_threshold=args.assignment_f1_threshold,
        amusa_method=args.amusa_method,
        assignment_method=args.assignment_method,
        conformal_alpha=args.conformal_alpha,
        calibration_fraction=args.calibration_fraction,
    )
    artifacts.save(args.output_artifact)
    return 0


def run_fit_catalog_assessor(args: argparse.Namespace) -> int:
    burdens = tuple(_parse_csv_list(args.burdens, cast=int))
    removed_signatures = _resolve_removed_signatures(args)
    expert_names = _parse_csv_list(args.expert_names) or _default_expert_names()
    confidence_artifacts = _load_confidence_artifacts(args.confidence_artifact)
    registry = build_default_registry(REPO_ROOT, confidence_artifacts=confidence_artifacts)
    model, training_frame = fit_catalog_insufficiency_model_from_benchmark(
        sample_source=args.sample_source,
        signature_source=args.signature_source,
        exposure_source=args.exposure_source,
        mutation_type=args.mutation_type,
        burdens=burdens,
        removed_signatures=removed_signatures,
        max_positive_per_signature=args.max_positive_per_signature,
        max_negative_per_signature=args.max_negative_per_signature,
        active_threshold=args.active_threshold,
        random_seed=args.random_seed,
        expert_names=expert_names,
        registry=registry,
        calibration_method=args.calibration_method,
        calibration_fraction=args.calibration_fraction,
    )
    output_artifact = Path(args.output_artifact)
    model.save(output_artifact)
    training_frame.to_csv(output_artifact.with_suffix(".training.tsv"), sep="\t", index=False)
    return 0


def run_known_benchmark(args: argparse.Namespace) -> int:
    expert_names = _parse_csv_list(args.expert_names) or _default_expert_names()
    burdens = _parse_csv_list(args.burdens, cast=int)
    confidence_artifacts = _load_confidence_artifacts(args.confidence_artifact)
    catalog_assessor_model = _load_catalog_assessor(args.catalog_assessor_artifact)
    bootstrap_config = None
    if args.bootstrap_replicates > 0:
        bootstrap_config = BootstrapConfig(
            n_replicates=args.bootstrap_replicates,
            alpha=args.bootstrap_alpha,
            random_seed=args.bootstrap_random_seed,
            use_conformal=not args.disable_bootstrap_conformal,
        )
    result = run_known_catalog_benchmark(
        sample_source=args.sample_source,
        signature_source=args.signature_source,
        exposure_source=args.exposure_source,
        mutation_type=args.mutation_type,
        burdens=burdens,
        max_samples_per_burden=args.max_samples_per_burden,
        expert_names=expert_names,
        include_rule_fusion=not args.skip_rule_fusion,
        random_seed=args.random_seed,
        confidence_artifacts=confidence_artifacts,
        catalog_assessor_model=catalog_assessor_model,
        bootstrap_config=bootstrap_config,
        assignment_f1_threshold=args.assignment_f1_threshold,
    )
    _write_benchmark_result(result, Path(args.output_dir))
    return 0


def run_catalog_insufficiency(args: argparse.Namespace) -> int:
    expert_names = _parse_csv_list(args.expert_names) or _default_expert_names()
    burdens = _parse_csv_list(args.burdens, cast=int)
    removed_signatures = _resolve_removed_signatures(args)
    confidence_artifacts = _load_confidence_artifacts(args.confidence_artifact)
    catalog_assessor_model = _load_catalog_assessor(args.catalog_assessor_artifact)
    bootstrap_config = None
    if args.bootstrap_replicates > 0:
        bootstrap_config = BootstrapConfig(
            n_replicates=args.bootstrap_replicates,
            alpha=args.bootstrap_alpha,
            random_seed=args.bootstrap_random_seed,
            use_conformal=not args.disable_bootstrap_conformal,
        )
    result = run_catalog_insufficiency_benchmark(
        sample_source=args.sample_source,
        signature_source=args.signature_source,
        exposure_source=args.exposure_source,
        mutation_type=args.mutation_type,
        burdens=burdens,
        removed_signatures=removed_signatures,
        removal_manifest_source=args.removal_manifest,
        max_positive_per_signature=args.max_positive_per_signature,
        max_negative_per_signature=args.max_negative_per_signature,
        active_threshold=args.active_threshold,
        expert_names=expert_names,
        include_rule_fusion=not args.skip_rule_fusion,
        random_seed=args.random_seed,
        confidence_artifacts=confidence_artifacts,
        catalog_assessor_model=catalog_assessor_model,
        bootstrap_config=bootstrap_config,
    )
    _write_benchmark_result(result, Path(args.output_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mutational signature decision workflows for this repository.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    decision_parser = subparsers.add_parser("decision", help="Run experts and version-1 rule fusion.")
    decision_parser.add_argument("--sample-source", required=True)
    decision_parser.add_argument("--signature-source", required=True)
    decision_parser.add_argument("--mutation-type", required=True, choices=["SBS96", "DBS78", "ID83"])
    decision_parser.add_argument("--sample-ids", default=None, help="Comma-separated sample ids. Omit for cohort mode.")
    decision_parser.add_argument("--expert-names", default=None, help="Comma-separated expert names. Defaults to the release-core plain_nnls path.")
    decision_parser.add_argument("--confidence-artifact", default=None, help="Optional confidence calibration artifact JSON.")
    decision_parser.add_argument("--catalog-assessor-artifact", default=None, help="Optional trained catalog assessor artifact JSON.")
    decision_parser.add_argument("--bootstrap-replicates", type=int, default=0)
    decision_parser.add_argument("--bootstrap-alpha", type=float, default=0.1)
    decision_parser.add_argument("--bootstrap-random-seed", type=int, default=0)
    decision_parser.add_argument("--disable-bootstrap-conformal", action="store_true")
    decision_parser.add_argument("--experience-dir", default=None)
    decision_parser.add_argument("--run-label", default=None)
    decision_parser.add_argument("--output-dir", required=True)
    decision_parser.set_defaults(func=run_decision)

    known_parser = subparsers.add_parser("known-benchmark", help="Run the complete-reference benchmark.")
    known_parser.add_argument("--sample-source", required=True)
    known_parser.add_argument("--signature-source", required=True)
    known_parser.add_argument("--exposure-source", required=True)
    known_parser.add_argument("--mutation-type", required=True, choices=["SBS96", "DBS78", "ID83"])
    known_parser.add_argument("--burdens", default="100,200,500,2000,50000")
    known_parser.add_argument("--max-samples-per-burden", type=int, default=100)
    known_parser.add_argument("--expert-names", default=None, help="Comma-separated expert names. Defaults to the release-core plain_nnls path.")
    known_parser.add_argument("--skip-rule-fusion", action="store_true")
    known_parser.add_argument("--confidence-artifact", default=None)
    known_parser.add_argument("--catalog-assessor-artifact", default=None)
    known_parser.add_argument("--assignment-f1-threshold", type=float, default=0.8)
    known_parser.add_argument("--bootstrap-replicates", type=int, default=0)
    known_parser.add_argument("--bootstrap-alpha", type=float, default=0.1)
    known_parser.add_argument("--bootstrap-random-seed", type=int, default=0)
    known_parser.add_argument("--disable-bootstrap-conformal", action="store_true")
    known_parser.add_argument("--random-seed", type=int, default=0)
    known_parser.add_argument("--output-dir", required=True)
    known_parser.set_defaults(func=run_known_benchmark)

    insuff_parser = subparsers.add_parser(
        "catalog-insufficiency-benchmark",
        help="Run the incomplete-reference benchmark by removing signatures from the catalog.",
    )
    insuff_parser.add_argument("--sample-source", required=True)
    insuff_parser.add_argument("--signature-source", required=True)
    insuff_parser.add_argument("--exposure-source", required=True)
    insuff_parser.add_argument("--mutation-type", required=True, choices=["SBS96", "DBS78", "ID83"])
    insuff_parser.add_argument("--burdens", default="200,2000")
    insuff_parser.add_argument("--removed-signatures", default=None, help="Comma-separated signatures to remove.")
    insuff_parser.add_argument("--removal-manifest", default=None, help="Optional TSV generated by experiments/make_removal_manifest.py.")
    insuff_parser.add_argument("--max-positive-per-signature", type=int, default=50)
    insuff_parser.add_argument("--max-negative-per-signature", type=int, default=50)
    insuff_parser.add_argument("--active-threshold", type=float, default=0.0)
    insuff_parser.add_argument("--expert-names", default=None, help="Comma-separated expert names. Defaults to the release-core plain_nnls path.")
    insuff_parser.add_argument("--skip-rule-fusion", action="store_true")
    insuff_parser.add_argument("--confidence-artifact", default=None)
    insuff_parser.add_argument("--catalog-assessor-artifact", default=None)
    insuff_parser.add_argument("--bootstrap-replicates", type=int, default=0)
    insuff_parser.add_argument("--bootstrap-alpha", type=float, default=0.1)
    insuff_parser.add_argument("--bootstrap-random-seed", type=int, default=0)
    insuff_parser.add_argument("--disable-bootstrap-conformal", action="store_true")
    insuff_parser.add_argument("--random-seed", type=int, default=0)
    insuff_parser.add_argument("--output-dir", required=True)
    insuff_parser.set_defaults(func=run_catalog_insufficiency)

    confidence_parser = subparsers.add_parser("fit-confidence", help="Fit confidence calibration artifacts from known-catalog synthetic benchmark data.")
    confidence_parser.add_argument("--sample-source", required=True)
    confidence_parser.add_argument("--signature-source", required=True)
    confidence_parser.add_argument("--exposure-source", required=True)
    confidence_parser.add_argument("--mutation-type", required=True, choices=["SBS96", "DBS78", "ID83"])
    confidence_parser.add_argument("--burdens", default="100,200,500,2000,50000")
    confidence_parser.add_argument("--max-samples-per-burden", type=int, default=100)
    confidence_parser.add_argument("--expert-names", default=None, help="Comma-separated expert names. Defaults to the release-core plain_nnls path.")
    confidence_parser.add_argument("--active-threshold", type=float, default=0.0)
    confidence_parser.add_argument("--assignment-f1-threshold", type=float, default=0.8)
    confidence_parser.add_argument("--amusa-method", default="temperature", choices=["temperature", "isotonic"])
    confidence_parser.add_argument("--assignment-method", default="isotonic", choices=["temperature", "isotonic"])
    confidence_parser.add_argument("--conformal-alpha", type=float, default=0.1)
    confidence_parser.add_argument("--calibration-fraction", type=float, default=0.25)
    confidence_parser.add_argument("--random-seed", type=int, default=0)
    confidence_parser.add_argument("--output-artifact", required=True)
    confidence_parser.set_defaults(func=run_fit_confidence)

    assessor_parser = subparsers.add_parser("fit-catalog-assessor", help="Fit a trainable catalog-insufficiency assessor from synthetic incomplete-catalog benchmark data.")
    assessor_parser.add_argument("--sample-source", required=True)
    assessor_parser.add_argument("--signature-source", required=True)
    assessor_parser.add_argument("--exposure-source", required=True)
    assessor_parser.add_argument("--mutation-type", required=True, choices=["SBS96", "DBS78", "ID83"])
    assessor_parser.add_argument("--burdens", default="200,2000")
    assessor_parser.add_argument("--removed-signatures", default=None)
    assessor_parser.add_argument("--removal-manifest", default=None, help="Optional TSV generated by experiments/make_removal_manifest.py.")
    assessor_parser.add_argument("--max-positive-per-signature", type=int, default=50)
    assessor_parser.add_argument("--max-negative-per-signature", type=int, default=50)
    assessor_parser.add_argument("--active-threshold", type=float, default=0.0)
    assessor_parser.add_argument("--expert-names", default=None, help="Comma-separated expert names. Defaults to the release-core plain_nnls path.")
    assessor_parser.add_argument("--confidence-artifact", default=None)
    assessor_parser.add_argument("--calibration-method", default="isotonic", choices=["temperature", "isotonic"])
    assessor_parser.add_argument("--calibration-fraction", type=float, default=0.25)
    assessor_parser.add_argument("--random-seed", type=int, default=0)
    assessor_parser.add_argument("--output-artifact", required=True)
    assessor_parser.set_defaults(func=run_fit_catalog_assessor)

    review_parser = subparsers.add_parser("append-review", help="Append one review decision into the experience store and refresh queues/datasets.")
    review_parser.add_argument("--experience-dir", required=True)
    review_parser.add_argument("--record-id", required=True)
    review_parser.add_argument("--reviewer", required=True)
    review_parser.add_argument("--review-outcome", required=True, choices=["confirmed", "corrected", "rejected", "deferred"])
    review_parser.add_argument("--evidence-type", required=True)
    review_parser.add_argument("--validated-recommendation", default=None)
    review_parser.add_argument("--catalog-insufficiency-confirmed", default=None)
    review_parser.add_argument("--reference-reassessment-confirmed", default=None)
    review_parser.add_argument("--preprocessing-issue-flag", default=None)
    review_parser.add_argument("--labels-json", default=None)
    review_parser.add_argument("--notes", default=None)
    review_parser.add_argument("--include-all-reviewed", action="store_true")
    review_parser.set_defaults(func=run_append_review)

    export_parser = subparsers.add_parser("export-experience-dataset", help="Export reviewed experience records into a reusable dataset TSV.")
    export_parser.add_argument("--experience-dir", required=True)
    export_parser.add_argument("--output-path", required=True)
    export_parser.add_argument("--include-all-reviewed", action="store_true")
    export_parser.set_defaults(func=run_export_experience_dataset)

    discovery_parser = subparsers.add_parser(
        "discovery-run",
        help="Run conservative local constrained discovery on the experience store and emit discovery packets.",
    )
    discovery_parser.add_argument("--experience-dir", required=True)
    discovery_parser.add_argument("--sample-source", default=None)
    discovery_parser.add_argument("--signature-source", default=None)
    discovery_parser.add_argument("--probability-threshold", type=float, default=0.8)
    discovery_parser.add_argument("--residual-structure-threshold", type=float, default=0.45)
    discovery_parser.add_argument("--min-recurrence-count", type=int, default=2)
    discovery_parser.add_argument("--mutation-count-threshold", type=float, default=500.0)
    discovery_parser.add_argument("--similarity-threshold", type=float, default=0.9)
    discovery_parser.add_argument("--min-cluster-size", type=int, default=2)
    discovery_parser.add_argument("--disable-cohort-candidate-gate", action="store_true")
    discovery_parser.add_argument("--disable-review-confirmation", action="store_true")
    discovery_parser.add_argument("--allow-pending-manual-review", action="store_true")
    discovery_parser.add_argument("--max-components", type=int, default=2)
    discovery_parser.add_argument("--min-records", type=int, default=2)
    discovery_parser.add_argument("--min-mean-residual-mass", type=float, default=0.05)
    discovery_parser.add_argument("--min-error-gain", type=float, default=0.08)
    discovery_parser.add_argument("--min-component-weight-fraction", type=float, default=0.10)
    discovery_parser.add_argument("--bootstrap-repeats", type=int, default=10)
    discovery_parser.add_argument("--max-iter", type=int, default=800)
    discovery_parser.add_argument("--random-seed", type=int, default=0)
    discovery_parser.add_argument("--run-label", default=None)
    discovery_parser.add_argument("--skip-experience-persist", action="store_true")
    discovery_parser.add_argument("--output-dir", required=True)
    discovery_parser.set_defaults(func=run_discovery)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
