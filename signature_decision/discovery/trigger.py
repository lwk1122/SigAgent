from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..experience import ExperienceRecord, ReviewDecision, latest_reviews_by_record
from ..experts.schema import json_ready
from .local_extract import LocalExtractionConfig, LocalExtractionResult, extract_local_residual_components
from .packet import DiscoveryPacket, build_discovery_packet
from .recurrence import RecurrenceCluster, build_recurrence_clusters, cluster_map


@dataclass(slots=True)
class DiscoveryTriggerConfig:
    probability_threshold: float = 0.80
    residual_structure_threshold: float = 0.45
    min_recurrence_count: int = 2
    mutation_count_threshold: float = 500.0
    similarity_threshold: float = 0.90
    min_cluster_size: int = 2
    require_cohort_candidate: bool = True
    require_review_confirmation: bool = True
    allow_pending_manual_review: bool = False
    accepted_review_outcomes: tuple[str, ...] = ("confirmed", "corrected")


@dataclass(slots=True)
class TriggeredDiscoveryCandidate:
    record_id: str
    sample_id: str
    mutation_type: str
    trigger_status: str
    cluster_id: str | None
    recurrence_count: int
    priority_score: float
    calibrated_catalog_insufficiency_probability: float | None
    residual_structure_score: float | None
    mutation_count: float
    review_gate_status: str
    passed_conditions: list[str] = field(default_factory=list)
    blocked_conditions: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "record_id": self.record_id,
                "sample_id": self.sample_id,
                "mutation_type": self.mutation_type,
                "trigger_status": self.trigger_status,
                "cluster_id": self.cluster_id,
                "recurrence_count": self.recurrence_count,
                "priority_score": self.priority_score,
                "calibrated_catalog_insufficiency_probability": self.calibrated_catalog_insufficiency_probability,
                "residual_structure_score": self.residual_structure_score,
                "mutation_count": self.mutation_count,
                "review_gate_status": self.review_gate_status,
                "passed_conditions": self.passed_conditions,
                "blocked_conditions": self.blocked_conditions,
                "rationale": self.rationale,
                "metadata": self.metadata,
            }
        )

    def to_index_row(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "sample_id": self.sample_id,
            "mutation_type": self.mutation_type,
            "trigger_status": self.trigger_status,
            "cluster_id": self.cluster_id,
            "recurrence_count": self.recurrence_count,
            "priority_score": self.priority_score,
            "catalog_insufficiency_probability": self.calibrated_catalog_insufficiency_probability,
            "residual_structure_score": self.residual_structure_score,
            "mutation_count": self.mutation_count,
            "review_gate_status": self.review_gate_status,
            "passed_conditions": ",".join(self.passed_conditions),
            "blocked_conditions": ",".join(self.blocked_conditions),
            "rationale": " | ".join(self.rationale),
        }


@dataclass(slots=True)
class DiscoveryTriggerOutput:
    candidates: list[TriggeredDiscoveryCandidate]
    recurrence_clusters: list[RecurrenceCluster]
    summary: dict[str, Any]

    @property
    def candidates_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([candidate.to_index_row() for candidate in self.candidates])

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "recurrence_clusters": [cluster.to_dict() for cluster in self.recurrence_clusters],
                "summary": self.summary,
            }
        )


@dataclass(slots=True)
class DiscoveryRunOutput:
    trigger_output: DiscoveryTriggerOutput
    packets: list[DiscoveryPacket]
    extraction_results_by_cluster: dict[str, LocalExtractionResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "trigger_output": self.trigger_output.to_dict(),
                "packets": [packet.to_dict() for packet in self.packets],
                "extraction_results_by_cluster": {
                    cluster_id: result.to_dict()
                    for cluster_id, result in self.extraction_results_by_cluster.items()
                },
            }
        )


def _review_gate_status(
    record: ExperienceRecord,
    review: ReviewDecision | None,
    *,
    config: DiscoveryTriggerConfig,
) -> tuple[bool, str]:
    if not config.require_review_confirmation:
        return True, "disabled"
    if review is not None and review.review_outcome in config.accepted_review_outcomes:
        return True, "confirmed_review"
    if (
        config.allow_pending_manual_review
        and "manual_review" in record.queue_types
        and "cohort_level_discovery" in record.queue_types
    ):
        return True, "pending_high_priority_manual_review"
    return False, "review_required"


def _priority_score(
    probability: float | None,
    residual_structure_score: float | None,
    recurrence_count: int,
    mutation_count: float,
    *,
    config: DiscoveryTriggerConfig,
) -> float:
    probability_component = 0.0 if probability is None else min(1.0, float(probability))
    residual_component = 0.0 if residual_structure_score is None else min(1.0, float(residual_structure_score))
    recurrence_component = min(1.0, recurrence_count / max(float(config.min_recurrence_count), 1.0))
    burden_component = min(1.0, mutation_count / max(config.mutation_count_threshold, 1.0))
    return float(0.35 * probability_component + 0.25 * residual_component + 0.25 * recurrence_component + 0.15 * burden_component)


def build_discovery_trigger_output(
    records: list[ExperienceRecord],
    *,
    reviews: list[ReviewDecision] | None = None,
    config: DiscoveryTriggerConfig | None = None,
) -> DiscoveryTriggerOutput:
    config = config or DiscoveryTriggerConfig()
    reviews = reviews or []
    latest_reviews = latest_reviews_by_record(reviews)
    candidate_pool = [
        record
        for record in records
        if record.recommendation.get("primary_recommendation") == "cohort_level_discovery"
        or "cohort_level_discovery" in record.queue_types
    ]
    clusters = build_recurrence_clusters(
        candidate_pool,
        similarity_threshold=config.similarity_threshold,
        min_cluster_size=config.min_cluster_size,
    )
    clusters_by_record = cluster_map(clusters)
    candidates: list[TriggeredDiscoveryCandidate] = []
    for record in records:
        probability = record.fusion_report.get("catalog_insufficiency_probability")
        residual_structure_score = (
            (record.fused_sample_result.get("diagnostics") or {}).get("residual_structure_score")
        )
        mutation_count = float((record.input_summary or {}).get("mutation_count", 0.0))
        cluster = clusters_by_record.get(record.record_id)
        recurrence_count = 0 if cluster is None else cluster.recurrence_count
        review = latest_reviews.get(record.record_id)
        review_passed, review_status = _review_gate_status(record, review, config=config)
        passed_conditions: list[str] = []
        blocked_conditions: list[str] = []
        rationale: list[str] = []

        if probability is not None and float(probability) >= config.probability_threshold:
            passed_conditions.append("high_catalog_insufficiency_probability")
        else:
            blocked_conditions.append("catalog_probability_below_threshold")
        if residual_structure_score is not None and float(residual_structure_score) >= config.residual_structure_threshold:
            passed_conditions.append("structured_residual")
        else:
            blocked_conditions.append("structured_residual_not_stable")
        if recurrence_count >= config.min_recurrence_count:
            passed_conditions.append("cohort_recurrence")
        else:
            blocked_conditions.append("insufficient_cohort_recurrence")
        if mutation_count >= config.mutation_count_threshold:
            passed_conditions.append("mutation_burden_sufficient")
        else:
            blocked_conditions.append("mutation_burden_below_threshold")
        if not config.require_cohort_candidate or "cohort_level_discovery" in record.queue_types:
            passed_conditions.append("cohort_candidate_gate")
        else:
            blocked_conditions.append("not_in_cohort_discovery_queue")
        if review_passed:
            passed_conditions.append("review_gate")
        else:
            blocked_conditions.append("review_gate_not_satisfied")

        if "high_catalog_insufficiency_probability" in passed_conditions:
            rationale.append("Calibrated catalog-insufficiency probability is high enough for secondary analysis.")
        if "structured_residual" in passed_conditions:
            rationale.append("Residual structure remains after known-catalog explanation.")
        if "cohort_recurrence" in passed_conditions:
            rationale.append("Residual pattern recurs in a cohort cluster rather than a single isolated sample.")
        if review_passed:
            rationale.append(f"Review gate satisfied via {review_status}.")

        status = "ready" if not blocked_conditions else "blocked"
        candidates.append(
            TriggeredDiscoveryCandidate(
                record_id=record.record_id,
                sample_id=record.sample_id,
                mutation_type=record.mutation_type,
                trigger_status=status,
                cluster_id=None if cluster is None else cluster.cluster_id,
                recurrence_count=recurrence_count,
                priority_score=_priority_score(
                    None if probability is None else float(probability),
                    None if residual_structure_score is None else float(residual_structure_score),
                    recurrence_count,
                    mutation_count,
                    config=config,
                ),
                calibrated_catalog_insufficiency_probability=(
                    None if probability is None else float(probability)
                ),
                residual_structure_score=(
                    None if residual_structure_score is None else float(residual_structure_score)
                ),
                mutation_count=mutation_count,
                review_gate_status=review_status,
                passed_conditions=passed_conditions,
                blocked_conditions=blocked_conditions,
                rationale=rationale,
                metadata={
                    "queue_types": record.queue_types,
                    "primary_recommendation": record.recommendation.get("primary_recommendation"),
                },
            )
        )
    summary = {
        "n_records": len(records),
        "n_candidate_pool": len(candidate_pool),
        "n_clusters": len(clusters),
        "n_ready": int(sum(candidate.trigger_status == "ready" for candidate in candidates)),
        "n_blocked": int(sum(candidate.trigger_status != "ready" for candidate in candidates)),
        "config": json_ready(asdict(config)),
    }
    return DiscoveryTriggerOutput(
        candidates=sorted(candidates, key=lambda item: (item.trigger_status != "ready", -item.priority_score)),
        recurrence_clusters=clusters,
        summary=summary,
    )


def build_discovery_packets(
    trigger_output: DiscoveryTriggerOutput,
    *,
    created_at: str,
    packet_id_prefix: str = "discovery_packet",
) -> list[DiscoveryPacket]:
    cluster_lookup = {cluster.cluster_id: cluster for cluster in trigger_output.recurrence_clusters}
    packets: list[DiscoveryPacket] = []
    ready_by_cluster: dict[str, list[TriggeredDiscoveryCandidate]] = {}
    for candidate in trigger_output.candidates:
        if candidate.trigger_status != "ready" or candidate.cluster_id is None:
            continue
        ready_by_cluster.setdefault(candidate.cluster_id, []).append(candidate)
    for cluster_id, cluster_candidates in ready_by_cluster.items():
        cluster = cluster_lookup[cluster_id]
        packet = build_discovery_packet(
            packet_id=f"{packet_id_prefix}__{cluster_id}",
            created_at=created_at,
            mutation_type=cluster.mutation_type,
            trigger_summary={
                "n_ready_candidates": len(cluster_candidates),
                "top_candidate": cluster_candidates[0].to_dict(),
            },
            recurrence_summary=cluster.to_dict(),
            candidate_records=[
                {
                    "record_id": candidate.record_id,
                    "sample_id": candidate.sample_id,
                    "priority_score": candidate.priority_score,
                }
                for candidate in cluster_candidates
            ],
            recommended_actions=["manual_research_review", "cohort_level_validation"],
            rationale=[
                "Trigger conditions were satisfied after calibration, recurrence filtering, burden thresholding, and review gating.",
                "Packet generation is advisory only and must not update the main catalog automatically.",
            ],
            metadata={
                "cluster_id": cluster.cluster_id,
                "recurrence_count": cluster.recurrence_count,
            },
        )
        packets.append(packet)
    return packets


def run_conservative_discovery_workflow(
    records: list[ExperienceRecord],
    *,
    reviews: list[ReviewDecision] | None = None,
    trigger_config: DiscoveryTriggerConfig | None = None,
    extraction_config: LocalExtractionConfig | None = None,
    created_at: str,
    signature_matrix: pd.DataFrame | None = None,
    channel_ids: list[str] | None = None,
    packet_id_prefix: str = "discovery_packet",
) -> DiscoveryRunOutput:
    trigger_output = build_discovery_trigger_output(
        records,
        reviews=reviews,
        config=trigger_config,
    )
    extraction_config = extraction_config or LocalExtractionConfig()
    extraction_results_by_cluster: dict[str, LocalExtractionResult] = {}
    cluster_lookup = {cluster.cluster_id: cluster for cluster in trigger_output.recurrence_clusters}
    record_lookup = {record.record_id: record for record in records}
    ready_clusters: dict[str, list[TriggeredDiscoveryCandidate]] = {}
    for candidate in trigger_output.candidates:
        if candidate.trigger_status == "ready" and candidate.cluster_id is not None:
            ready_clusters.setdefault(candidate.cluster_id, []).append(candidate)

    packets: list[DiscoveryPacket] = []
    for cluster_id, cluster_candidates in ready_clusters.items():
        cluster = cluster_lookup[cluster_id]
        cluster_records = [record_lookup[record_id] for record_id in cluster.record_ids if record_id in record_lookup]
        extraction_result = extract_local_residual_components(
            cluster_records,
            config=extraction_config,
            channel_ids=channel_ids,
            signature_matrix=signature_matrix,
        )
        extraction_results_by_cluster[cluster_id] = extraction_result
        packet = build_discovery_packet(
            packet_id=f"{packet_id_prefix}__{cluster_id}",
            created_at=created_at,
            mutation_type=cluster.mutation_type,
            trigger_summary={
                "n_ready_candidates": len(cluster_candidates),
                "top_candidate": cluster_candidates[0].to_dict(),
            },
            recurrence_summary=cluster.to_dict(),
            candidate_records=[
                {
                    "record_id": candidate.record_id,
                    "sample_id": candidate.sample_id,
                    "priority_score": candidate.priority_score,
                    "review_gate_status": candidate.review_gate_status,
                }
                for candidate in cluster_candidates
            ],
            extracted_components=[component.to_dict() for component in extraction_result.components],
            catalog_match_summary=[
                {
                    "component_id": component.component_id,
                    "best_match_name": component.catalog_match_name,
                    "best_match_cosine": component.catalog_match_cosine,
                    "top_hits": component.catalog_top_hits,
                }
                for component in extraction_result.components
            ],
            fit_improvement_summary=extraction_result.fit_improvement_summary,
            recommended_actions=(
                ["manual_research_review", "cohort_level_validation"]
                if extraction_result.components
                else ["manual_research_review"]
            ),
            rationale=[
                "Trigger conditions were satisfied after calibrated risk, residual structure, recurrence, burden, and review gating.",
                "Local constrained discovery is limited to residual-only extraction plus known-signature refit improvement analysis.",
                "Packets are evidence artifacts only and must not update the main catalog automatically.",
            ],
            metadata={
                "cluster_id": cluster.cluster_id,
                "recurrence_count": cluster.recurrence_count,
                "extraction_status": extraction_result.metadata.get("status"),
                "extraction_metadata": extraction_result.metadata,
            },
        )
        packets.append(packet)

    return DiscoveryRunOutput(
        trigger_output=trigger_output,
        packets=packets,
        extraction_results_by_cluster=extraction_results_by_cluster,
    )


__all__ = [
    "DiscoveryRunOutput",
    "DiscoveryTriggerConfig",
    "DiscoveryTriggerOutput",
    "TriggeredDiscoveryCandidate",
    "build_discovery_packets",
    "build_discovery_trigger_output",
    "run_conservative_discovery_workflow",
]
