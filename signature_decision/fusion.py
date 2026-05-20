from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd

from .confidence import BootstrapConfig, ConfidenceArtifacts, bootstrap_exposure_intervals
from .experts.base import build_sample_results
from .experts.schema import ExpertRequest, ExpertRunResult, ExpertSampleResult
from .fusion_features import extract_fusion_evidence
from .schemas import FinalSampleReport


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))

def _reconstruction_score(mean_reconstruction_cosine: float) -> float:
    return _clamp((mean_reconstruction_cosine - 0.7) / 0.25)


def _mutation_count_profile(mutation_count: float) -> tuple[float, float]:
    if mutation_count < 200:
        return 2.0, 0.35
    if mutation_count < 500:
        return 1.5, 0.55
    if mutation_count < 2000:
        return 1.2, 0.75
    return 1.0, 1.0


@dataclass(slots=True)
class RuleFusionConfig:
    active_probability_threshold: float = 0.5
    report_probability_threshold: float = 0.35
    unstable_probability_threshold: float = 0.2
    consensus_jaccard_threshold: float = 0.65
    good_reconstruction_threshold: float = 0.9
    high_disagreement_threshold: float = 0.55
    structured_residual_threshold: float = 0.45
    amusa_musical_alignment_threshold: float = 0.6
    spa_deviation_threshold: float = 0.4
    min_interval_width: float = 0.04
    stable_interval_width: float = 0.12
    max_reported_signatures: int = 12


@dataclass(slots=True)
class RuleFusionOutput:
    fused_run: ExpertRunResult
    reports: list[FinalSampleReport]
    summary_frame: pd.DataFrame


def _build_known_signature_record(
    *,
    signature_name: str,
    fused_proxy_score: float,
    fused_probability: float | None,
    fused_proportion: float,
    interval: tuple[float, float],
    interval_source: str,
    stability: str,
    supporters: list[str],
    dissenters: list[str],
    trusted_experts: list[str],
) -> dict[str, Any]:
    return {
        "name": signature_name,
        "active_proxy_score": round(float(fused_proxy_score), 6),
        "active_probability": None if fused_probability is None else round(float(fused_probability), 6),
        "stability": stability,
        "exposure": round(float(fused_proportion), 6),
        "exposure_interval": [round(float(interval[0]), 6), round(float(interval[1]), 6)],
        "exposure_interval_source": interval_source,
        "supporting_experts": supporters,
        "dissenting_experts": dissenters,
        "trusted_experts": trusted_experts,
    }


def _build_unstable_record(
    *,
    signature_name: str,
    fused_proxy_score: float,
    fused_probability: float | None,
    interval: tuple[float, float],
    interval_source: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "name": signature_name,
        "active_proxy_score": round(float(fused_proxy_score), 6),
        "active_probability": None if fused_probability is None else round(float(fused_probability), 6),
        "exposure_interval": [round(float(interval[0]), 6), round(float(interval[1]), 6)],
        "exposure_interval_source": interval_source,
        "reason": reason,
    }


def _choose_recommendation(
    *,
    assignment_confidence_score: float,
    catalog_insufficiency_score: float,
    unstable_count: int,
    mutation_count: float,
) -> tuple[str, list[str], list[str]]:
    rationale: list[str] = []
    secondary: list[str] = []
    if catalog_insufficiency_score >= 0.75:
        rationale.append("High catalog-insufficiency suspicion after rule fusion.")
        return "cohort_level_discovery", ["manual_review"], rationale
    if catalog_insufficiency_score >= 0.55:
        rationale.append("Reference catalog may be insufficient for this sample.")
        secondary.append("manual_review")
        return "reassess_reference_catalog", secondary, rationale
    if assignment_confidence_score < 0.45 or unstable_count > 0:
        rationale.append("Expert disagreement or wide uncertainty remains after fusion.")
        if mutation_count < 500:
            rationale.append("Low mutation count weakens exposure interpretation.")
        return "manual_review", secondary, rationale
    rationale.append("Experts are sufficiently consistent for downstream use.")
    return "direct_downstream_analysis", secondary, rationale


def _find_sample_result(run: ExpertRunResult, sample_id: str) -> ExpertSampleResult | None:
    for sample_result in run.sample_results:
        if sample_result.sample_id == sample_id:
            return sample_result
    return None


def fuse_expert_runs(
    runs: list[ExpertRunResult],
    request: ExpertRequest,
    *,
    config: RuleFusionConfig | None = None,
    confidence_artifacts: ConfidenceArtifacts | None = None,
    catalog_assessor_model: Any | None = None,
    bootstrap_config: BootstrapConfig | None = None,
) -> RuleFusionOutput:
    config = config or RuleFusionConfig()
    successful_runs = [run for run in runs if run.status == "success" and run.sample_results]
    failed_runs = [run for run in runs if run.status != "success"]

    exposure_table: dict[str, dict[str, float]] = {}
    raw_score_table: dict[str, dict[str, float]] = {}
    probability_table: dict[str, dict[str, float | None]] = {}
    diagnostics_by_sample: dict[str, dict[str, Any]] = {}
    warnings_by_sample: dict[str, list[str]] = {}
    reports: list[FinalSampleReport] = []
    summary_rows: list[dict[str, Any]] = []

    if not successful_runs:
        fused_run = ExpertRunResult.failed(
            expert_name="rule_fusion",
            request=request,
            runtime_seconds=0.0,
            parameters={"config": asdict(config)},
            error="No successful expert runs available for fusion.",
        )
        return RuleFusionOutput(fused_run=fused_run, reports=[], summary_frame=pd.DataFrame())

    for sample_id in request.sample_ids:
        sample_results_by_expert = {
            run.expert_name: _find_sample_result(run, sample_id)
            for run in successful_runs
        }
        sample_results_by_expert = {
            expert_name: sample_result
            for expert_name, sample_result in sample_results_by_expert.items()
            if sample_result is not None
        }
        evidence = extract_fusion_evidence(
            sample_id=sample_id,
            request=request,
            sample_results_by_expert=sample_results_by_expert,
            failed_expert_names=[run.expert_name for run in failed_runs],
            catalog_assessor_model=catalog_assessor_model,
        )
        expert_names = evidence.expert_names
        mutation_count = evidence.mutation_count
        interval_multiplier, burden_confidence = _mutation_count_profile(mutation_count)

        trusted_experts = expert_names[:]
        fusion_mode = "fallback"
        rationale = []
        active_sets = evidence.active_sets
        agreement_score = evidence.agreement_score
        disagreement_score = evidence.disagreement_score
        pairwise_jaccard_scores = evidence.pairwise_active_set_jaccard
        mean_reconstruction_cosine = evidence.mean_reconstruction_cosine

        if (
            len(expert_names) >= 2
            and agreement_score >= config.consensus_jaccard_threshold
            and mean_reconstruction_cosine >= config.good_reconstruction_threshold
        ):
            fusion_mode = "consensus"
            rationale.append("Multiple experts agree on active signatures with strong reconstruction.")
        elif {"amusa", "musical"}.issubset(sample_results_by_expert):
            amusa_set = active_sets["amusa"]
            musical_set = active_sets["musical"]
            amusa_musical_union = len(amusa_set | musical_set)
            amusa_musical_jaccard = 1.0 if amusa_musical_union == 0 else len(amusa_set & musical_set) / amusa_musical_union
            spa_set = active_sets.get("sigprofiler_assignment", set())
            amusa_spa_union = len(amusa_set | spa_set)
            musical_spa_union = len(musical_set | spa_set)
            amusa_spa_jaccard = 1.0 if amusa_spa_union == 0 else len(amusa_set & spa_set) / amusa_spa_union
            musical_spa_jaccard = 1.0 if musical_spa_union == 0 else len(musical_set & spa_set) / musical_spa_union
            if (
                amusa_musical_jaccard >= config.amusa_musical_alignment_threshold
                and max(amusa_spa_jaccard, musical_spa_jaccard) <= config.spa_deviation_threshold
            ):
                trusted_experts = ["amusa", "musical"]
                constrained_set = active_sets.get("classifier_guided_refit")
                if constrained_set is not None:
                    constrained_union = len(constrained_set | amusa_set)
                    constrained_jaccard = (
                        1.0 if constrained_union == 0 else len(constrained_set & amusa_set) / constrained_union
                    )
                    if constrained_jaccard >= config.amusa_musical_alignment_threshold:
                        trusted_experts.append("classifier_guided_refit")
                fusion_mode = "amusa_musical_priority"
                rationale.append("AMuSa and MuSiCal agree while SigProfilerAssignment deviates.")

        catalog_assessment = evidence.catalog_assessment
        catalog_insufficiency_proxy_score = evidence.catalog_insufficiency_proxy_score
        catalog_insufficiency_probability = evidence.catalog_insufficiency_probability
        catalog_insufficiency_level = evidence.catalog_insufficiency_level
        residual_structure_score = evidence.residual_structure_score
        sample_group_context = evidence.group_context
        if (
            disagreement_score >= config.high_disagreement_threshold
            and residual_structure_score >= config.structured_residual_threshold
        ):
            fusion_mode = "disagreement_review"
            rationale.append("Large expert disagreement coincides with structured residual signal.")

        reconstruction_component = _reconstruction_score(mean_reconstruction_cosine)
        assignment_confidence = _clamp(
            0.45 * agreement_score
            + 0.3 * reconstruction_component
            + 0.2 * burden_confidence
            + (0.05 if fusion_mode == "consensus" else 0.0)
            - 0.3 * catalog_insufficiency_proxy_score
        )
        assignment_confidence_probability = None
        assignment_calibration_group = "none"
        if confidence_artifacts is not None:
            if confidence_artifacts.final_assignment_group_calibrator is not None:
                assignment_confidence_probability, assignment_calibration_group = (
                    confidence_artifacts.final_assignment_group_calibrator.transform_one(
                        assignment_confidence,
                        context=sample_group_context,
                        return_source=True,
                    )
                )
            elif confidence_artifacts.final_assignment_calibrator is not None:
                assignment_confidence_probability = float(
                    confidence_artifacts.final_assignment_calibrator.transform([assignment_confidence])[0]
                )
                assignment_calibration_group = "global"

        normalized_exposures_by_expert = evidence.normalized_exposures_by_expert
        fused_probability_by_signature: dict[str, float] = {}
        fused_proportion_by_signature: dict[str, float] = {}
        interval_by_signature: dict[str, tuple[float, float]] = {}
        known_signatures: list[dict[str, Any]] = []
        unstable_conclusions: list[dict[str, Any]] = []

        for signature_name in request.signature_names:
            expert_scores = []
            calibrated_probabilities = []
            proportion_values = []
            supporters = []
            dissenters = []
            for expert_name, sample_result in sample_results_by_expert.items():
                normalized_exposure = normalized_exposures_by_expert[expert_name].get(signature_name, 0.0)
                if expert_name in trusted_experts:
                    proportion_values.append(normalized_exposure)
                    score = sample_result.signature_scores.get(signature_name)
                    if score is None:
                        score = 1.0 if signature_name in sample_result.active_signatures else 0.0
                    expert_scores.append(float(score))
                    probability = sample_result.signature_probabilities.get(signature_name)
                    if probability is not None:
                        calibrated_probabilities.append(float(probability))
                if signature_name in sample_result.active_signatures:
                    supporters.append(expert_name)
                else:
                    dissenters.append(expert_name)

            fused_proxy_score = float(np.mean(expert_scores)) if expert_scores else 0.0
            fused_probability = float(np.mean(calibrated_probabilities)) if calibrated_probabilities else None
            mean_proportion = float(np.mean(proportion_values)) if proportion_values else 0.0
            spread = max(proportion_values) - min(proportion_values) if proportion_values else 0.0
            interval_width = max(spread, config.min_interval_width if fused_proxy_score >= config.unstable_probability_threshold else 0.0)
            interval_width *= interval_multiplier
            interval = (
                max(0.0, mean_proportion - interval_width / 2.0),
                min(1.0, mean_proportion + interval_width / 2.0),
            )

            fused_probability_by_signature[signature_name] = fused_probability
            fused_proportion_by_signature[signature_name] = mean_proportion
            interval_by_signature[signature_name] = interval
            raw_score_table.setdefault(sample_id, {})[signature_name] = fused_proxy_score
            probability_table.setdefault(sample_id, {})[signature_name] = fused_probability

            if fused_proxy_score < config.report_probability_threshold and mean_proportion <= 0.0:
                continue

            if fused_proxy_score >= 0.8 and (interval[1] - interval[0]) <= config.stable_interval_width and agreement_score >= 0.6:
                stability = "high"
            elif fused_proxy_score >= config.active_probability_threshold:
                stability = "medium"
            else:
                stability = "low"

            if fused_proxy_score >= config.report_probability_threshold:
                known_signatures.append(
                    _build_known_signature_record(
                        signature_name=signature_name,
                        fused_proxy_score=fused_proxy_score,
                        fused_probability=fused_probability,
                        fused_proportion=mean_proportion,
                        interval=interval,
                        interval_source="heuristic",
                        stability=stability,
                        supporters=supporters,
                        dissenters=dissenters,
                        trusted_experts=trusted_experts,
                    )
                )
            if (
                fused_proxy_score >= config.unstable_probability_threshold
                and (
                    stability == "low"
                    or len(set(supporters) & set(trusted_experts)) not in (0, len(trusted_experts))
                )
            ):
                unstable_conclusions.append(
                    _build_unstable_record(
                        signature_name=signature_name,
                        fused_proxy_score=fused_proxy_score,
                        fused_probability=fused_probability,
                        interval=interval,
                        interval_source="heuristic",
                        reason="Mixed expert support or wide interval after rule fusion.",
                    )
                )

        known_signatures.sort(key=lambda item: (item["active_proxy_score"], item["exposure"]), reverse=True)
        unstable_conclusions.sort(key=lambda item: item["active_proxy_score"], reverse=True)
        known_signatures = known_signatures[: config.max_reported_signatures]
        unstable_conclusions = unstable_conclusions[: config.max_reported_signatures]

        selected_signature_names = [
            item["name"]
            for item in known_signatures
            if item["active_proxy_score"] >= config.active_probability_threshold
        ]
        if not selected_signature_names and known_signatures:
            selected_signature_names = [known_signatures[0]["name"]]

        selected_proportions = {
            signature_name: fused_proportion_by_signature.get(signature_name, 0.0)
            for signature_name in selected_signature_names
        }
        total_selected = sum(selected_proportions.values())
        if total_selected > 0.0:
            selected_proportions = {
                signature_name: proportion / total_selected
                for signature_name, proportion in selected_proportions.items()
            }
        elif selected_signature_names:
            uniform = 1.0 / len(selected_signature_names)
            selected_proportions = {signature_name: uniform for signature_name in selected_signature_names}

        bootstrap_artifact = {"interval_source": "heuristic"}
        if bootstrap_config is not None and bootstrap_config.n_replicates > 0 and selected_signature_names:
            conformal_margin = None
            conformal_group = "none"
            if confidence_artifacts is not None:
                if confidence_artifacts.exposure_group_conformal is not None:
                    conformal_margin, conformal_group = confidence_artifacts.exposure_group_conformal.resolve(
                        sample_group_context
                    )
                else:
                    conformal_margin = confidence_artifacts.exposure_conformal_margin
                    conformal_group = "global" if conformal_margin is not None else "none"
            bootstrap_intervals, bootstrap_artifact = bootstrap_exposure_intervals(
                sample_counts=request.sample_matrix.loc[:, sample_id],
                signature_matrix=request.signature_matrix,
                selected_signature_names=selected_signature_names,
                config=bootstrap_config,
                conformal_margin=conformal_margin,
            )
            bootstrap_artifact["conformal_group"] = conformal_group
            for signature_name, interval in bootstrap_intervals.items():
                interval_by_signature[signature_name] = interval
            for item in known_signatures:
                signature_name = item["name"]
                if signature_name in bootstrap_intervals:
                    item["exposure_interval"] = [
                        round(float(bootstrap_intervals[signature_name][0]), 6),
                        round(float(bootstrap_intervals[signature_name][1]), 6),
                    ]
                    item["exposure_interval_source"] = bootstrap_artifact["interval_source"]
            for item in unstable_conclusions:
                signature_name = item["name"]
                if signature_name in bootstrap_intervals:
                    item["exposure_interval"] = [
                        round(float(bootstrap_intervals[signature_name][0]), 6),
                        round(float(bootstrap_intervals[signature_name][1]), 6),
                    ]
                    item["exposure_interval_source"] = bootstrap_artifact["interval_source"]

        fused_counts = {
            signature_name: selected_proportions.get(signature_name, 0.0) * mutation_count
            for signature_name in request.signature_names
        }
        exposure_table[sample_id] = fused_counts
        raw_score_table[sample_id] = {
            signature_name: raw_score_table.get(sample_id, {}).get(signature_name, 0.0)
            for signature_name in request.signature_names
        }
        probability_table[sample_id] = {
            signature_name: fused_probability_by_signature.get(signature_name)
            for signature_name in request.signature_names
        }

        primary_recommendation, secondary_recommendations, recommendation_rationale = _choose_recommendation(
            assignment_confidence_score=assignment_confidence,
            catalog_insufficiency_score=catalog_insufficiency_proxy_score,
            unstable_count=len(unstable_conclusions),
            mutation_count=mutation_count,
        )
        rationale.extend(recommendation_rationale)
        if mutation_count < 500:
            rationale.append("Low mutation count widens fusion intervals and lowers exposure certainty.")
        for assessment_reason in catalog_assessment.rationale:
            if assessment_reason not in rationale:
                rationale.append(assessment_reason)

        diagnostics_by_sample[sample_id] = {
            "fusion_mode": fusion_mode,
            "trusted_experts": trusted_experts,
            "successful_experts": expert_names,
            "failed_experts": [run.expert_name for run in failed_runs],
            "pairwise_active_set_jaccard": pairwise_jaccard_scores,
            "agreement_score": float(agreement_score),
            "mean_reconstruction_cosine": mean_reconstruction_cosine,
            "residual_structure_score": residual_structure_score,
            "mutation_count": mutation_count,
            "catalog_insufficiency_proxy_score": catalog_insufficiency_proxy_score,
            "catalog_insufficiency_probability": catalog_insufficiency_probability,
            "catalog_insufficiency_level": catalog_insufficiency_level,
            "catalog_insufficiency_features": catalog_assessment.features.to_dict(),
            "catalog_insufficiency_component_scores": catalog_assessment.component_scores,
            "fusion_evidence_features": evidence.to_feature_row(),
            "assignment_confidence_raw_score": assignment_confidence,
            "assignment_confidence_probability": assignment_confidence_probability,
            "assignment_confidence_calibration_group": assignment_calibration_group,
            "primary_recommendation": primary_recommendation,
            "secondary_recommendations": secondary_recommendations,
            "recommendation_rationale": rationale,
            "group_context": sample_group_context,
            "exposure_intervals": {
                signature_name: [float(interval[0]), float(interval[1])]
                for signature_name, interval in interval_by_signature.items()
            },
            "exposure_interval_source": bootstrap_artifact["interval_source"],
            "exposure_interval_conformal_group": bootstrap_artifact.get("conformal_group"),
            "rationale": rationale,
        }
        sample_warnings = []
        if failed_runs:
            sample_warnings.append(
                "Some experts failed and were excluded from fusion: "
                + ", ".join(run.expert_name for run in failed_runs)
            )
        warnings_by_sample[sample_id] = sample_warnings

        reports.append(
            FinalSampleReport(
                sample_id=sample_id,
                mutation_type=request.mutation_type,
                known_signatures=known_signatures,
                unstable_conclusions=unstable_conclusions,
                catalog_insufficiency_proxy_score=round(catalog_insufficiency_proxy_score, 6),
                catalog_insufficiency_probability=(
                    None if catalog_insufficiency_probability is None else round(float(catalog_insufficiency_probability), 6)
                ),
                catalog_insufficiency_level=catalog_insufficiency_level,
                assignment_confidence_raw_score=round(assignment_confidence, 6),
                assignment_confidence_probability=(
                    None if assignment_confidence_probability is None else round(float(assignment_confidence_probability), 6)
                ),
                assignment_confidence=(
                    None if assignment_confidence_probability is None else round(float(assignment_confidence_probability), 6)
                ),
                primary_recommendation=primary_recommendation,
                secondary_recommendations=secondary_recommendations,
                recommendation_rationale=rationale,
                metadata={
                    "fusion_mode": fusion_mode,
                    "trusted_experts": trusted_experts,
                    "agreement_score": round(float(agreement_score), 6),
                    "mean_reconstruction_cosine": round(float(mean_reconstruction_cosine), 6),
                    "residual_structure_score": round(float(residual_structure_score), 6),
                    "mutation_count": round(float(mutation_count), 6),
                },
            )
        )

        summary_rows.append(
            {
                "sample_id": sample_id,
                "fusion_mode": fusion_mode,
                "catalog_insufficiency_level": catalog_insufficiency_level,
                "primary_recommendation": primary_recommendation,
                "secondary_recommendations": ",".join(secondary_recommendations),
                "recommendation_rationale": " | ".join(rationale),
                "assignment_confidence_raw_score": assignment_confidence,
                "assignment_confidence_probability": assignment_confidence_probability,
                "catalog_insufficiency_proxy_score": catalog_insufficiency_proxy_score,
                "catalog_insufficiency_probability": catalog_insufficiency_probability,
                "agreement_score": agreement_score,
                "mean_reconstruction_cosine": mean_reconstruction_cosine,
                "residual_structure_score": residual_structure_score,
                "mutation_count": mutation_count,
                "top_signatures": ",".join(selected_signature_names[:5]),
            }
        )

    exposure_df = pd.DataFrame(exposure_table).reindex(index=request.signature_names, columns=request.sample_ids, fill_value=0.0)
    raw_score_df = pd.DataFrame(raw_score_table).T.reindex(index=request.sample_ids, columns=request.signature_names, fill_value=0.0)
    probability_df = pd.DataFrame(probability_table).T.reindex(index=request.sample_ids, columns=request.signature_names)
    fused_run = ExpertRunResult(
        expert_name="rule_fusion",
        mutation_type=request.mutation_type,
        request_id=request.request_id or "request",
        status="success",
        signature_names=request.signature_names,
        channel_ids=request.channel_ids,
        sample_results=build_sample_results(
            request=request,
            exposures=exposure_df,
            signature_scores=raw_score_df,
            signature_probabilities=probability_df,
            diagnostics_by_sample=diagnostics_by_sample,
            warnings_by_sample=warnings_by_sample,
        ),
        parameters={
            "config": asdict(config),
            "confidence_artifacts_loaded": confidence_artifacts is not None,
            "catalog_assessor_loaded": catalog_assessor_model is not None,
            "bootstrap_config": None if bootstrap_config is None else asdict(bootstrap_config),
        },
        artifacts={"source_experts": [run.expert_name for run in runs]},
        warnings=[
            "Rule-based fusion is intended for version 1 baseline behavior."
        ],
    )
    return RuleFusionOutput(
        fused_run=fused_run,
        reports=reports,
        summary_frame=pd.DataFrame.from_records(summary_rows),
    )


def reports_to_frame(reports: list[FinalSampleReport]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report in reports:
        rows.append(
            {
                "sample_id": report.sample_id,
                "mutation_type": report.mutation_type,
                "primary_recommendation": report.primary_recommendation,
                "secondary_recommendations": ",".join(report.secondary_recommendations),
                "recommendation_rationale": " | ".join(report.recommendation_rationale),
                "assignment_confidence_raw_score": report.assignment_confidence_raw_score,
                "assignment_confidence_probability": report.assignment_confidence_probability,
                "catalog_insufficiency_proxy_score": report.catalog_insufficiency_proxy_score,
                "catalog_insufficiency_probability": report.catalog_insufficiency_probability,
                "catalog_insufficiency_level": report.catalog_insufficiency_level,
                "top_signatures": ",".join(item["name"] for item in report.known_signatures[:5]),
                "unstable_count": len(report.unstable_conclusions),
                "fusion_mode": report.metadata.get("fusion_mode"),
            }
        )
    return pd.DataFrame.from_records(rows)


__all__ = [
    "RuleFusionConfig",
    "RuleFusionOutput",
    "fuse_expert_runs",
    "reports_to_frame",
]
