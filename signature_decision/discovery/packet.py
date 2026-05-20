from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..experts.schema import json_ready


@dataclass(slots=True)
class DiscoveryPacket:
    packet_id: str
    created_at: str
    mutation_type: str
    packet_status: str
    trigger_summary: dict[str, Any]
    recurrence_summary: dict[str, Any]
    candidate_records: list[dict[str, Any]] = field(default_factory=list)
    extracted_components: list[dict[str, Any]] = field(default_factory=list)
    catalog_match_summary: list[dict[str, Any]] = field(default_factory=list)
    fit_improvement_summary: dict[str, Any] = field(default_factory=dict)
    recommended_actions: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    writeback_policy: str = "manual_review_only"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "packet_id": self.packet_id,
                "created_at": self.created_at,
                "mutation_type": self.mutation_type,
                "packet_status": self.packet_status,
                "trigger_summary": self.trigger_summary,
                "recurrence_summary": self.recurrence_summary,
                "candidate_records": self.candidate_records,
                "extracted_components": self.extracted_components,
                "catalog_match_summary": self.catalog_match_summary,
                "fit_improvement_summary": self.fit_improvement_summary,
                "recommended_actions": self.recommended_actions,
                "rationale": self.rationale,
                "writeback_policy": self.writeback_policy,
                "metadata": self.metadata,
            }
        )

    def to_index_row(self) -> dict[str, Any]:
        aggregate = self.fit_improvement_summary.get("aggregate") or {}
        return {
            "packet_id": self.packet_id,
            "created_at": self.created_at,
            "mutation_type": self.mutation_type,
            "packet_status": self.packet_status,
            "n_candidate_records": len(self.candidate_records),
            "n_extracted_components": len(self.extracted_components),
            "recommended_actions": ",".join(self.recommended_actions),
            "writeback_policy": self.writeback_policy,
            "recurrence_count": self.recurrence_summary.get("recurrence_count"),
            "mean_pairwise_similarity": self.recurrence_summary.get("mean_pairwise_similarity"),
            "mean_delta_reconstruction_cosine_vs_current": aggregate.get("mean_delta_reconstruction_cosine_vs_current"),
            "mean_delta_reconstruction_cosine_vs_known_only": aggregate.get("mean_delta_reconstruction_cosine_vs_known_only"),
            "mean_delta_relative_l1_pct_vs_current": aggregate.get("mean_delta_relative_l1_pct_vs_current"),
            "mean_delta_relative_l1_pct_vs_known_only": aggregate.get("mean_delta_relative_l1_pct_vs_known_only"),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiscoveryPacket":
        return cls(
            packet_id=str(payload["packet_id"]),
            created_at=str(payload["created_at"]),
            mutation_type=str(payload["mutation_type"]),
            packet_status=str(payload["packet_status"]),
            trigger_summary=dict(payload.get("trigger_summary") or {}),
            recurrence_summary=dict(payload.get("recurrence_summary") or {}),
            candidate_records=list(payload.get("candidate_records") or []),
            extracted_components=list(payload.get("extracted_components") or []),
            catalog_match_summary=list(payload.get("catalog_match_summary") or []),
            fit_improvement_summary=dict(payload.get("fit_improvement_summary") or {}),
            recommended_actions=[str(value) for value in payload.get("recommended_actions") or []],
            rationale=[str(value) for value in payload.get("rationale") or []],
            writeback_policy=str(payload.get("writeback_policy") or "manual_review_only"),
            metadata=dict(payload.get("metadata") or {}),
        )


def build_discovery_packet(
    *,
    packet_id: str,
    created_at: str,
    mutation_type: str,
    trigger_summary: dict[str, Any],
    recurrence_summary: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    extracted_components: list[dict[str, Any]] | None = None,
    catalog_match_summary: list[dict[str, Any]] | None = None,
    fit_improvement_summary: dict[str, Any] | None = None,
    recommended_actions: list[str] | None = None,
    rationale: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DiscoveryPacket:
    actions = list(recommended_actions or [])
    if not actions:
        actions = ["manual_research_review"]
    packet_rationale = list(rationale or [])
    if not packet_rationale:
        packet_rationale = [
            "This packet is a secondary analysis artifact and must not update the reference catalog automatically.",
        ]
    return DiscoveryPacket(
        packet_id=packet_id,
        created_at=created_at,
        mutation_type=mutation_type,
        packet_status="ready_for_review",
        trigger_summary=trigger_summary,
        recurrence_summary=recurrence_summary,
        candidate_records=list(candidate_records),
        extracted_components=list(extracted_components or []),
        catalog_match_summary=list(catalog_match_summary or []),
        fit_improvement_summary=dict(fit_improvement_summary or {}),
        recommended_actions=actions,
        rationale=packet_rationale,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "DiscoveryPacket",
    "build_discovery_packet",
]
