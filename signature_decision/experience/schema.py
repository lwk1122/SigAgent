from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..experts.schema import json_ready


@dataclass(slots=True)
class ExperienceRecord:
    record_id: str
    created_at: str
    sample_id: str
    mutation_type: str
    request_id: str | None = None
    source_context: dict[str, Any] = field(default_factory=dict)
    input_summary: dict[str, Any] = field(default_factory=dict)
    expert_outputs: dict[str, Any] = field(default_factory=dict)
    fusion_report: dict[str, Any] = field(default_factory=dict)
    fused_sample_result: dict[str, Any] = field(default_factory=dict)
    cohort_context: dict[str, Any] = field(default_factory=dict)
    recommendation: dict[str, Any] = field(default_factory=dict)
    artifact_versions: dict[str, Any] = field(default_factory=dict)
    review_status: str = "pending_review"
    queue_types: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "record_id": self.record_id,
                "created_at": self.created_at,
                "sample_id": self.sample_id,
                "mutation_type": self.mutation_type,
                "request_id": self.request_id,
                "source_context": self.source_context,
                "input_summary": self.input_summary,
                "expert_outputs": self.expert_outputs,
                "fusion_report": self.fusion_report,
                "fused_sample_result": self.fused_sample_result,
                "cohort_context": self.cohort_context,
                "recommendation": self.recommendation,
                "artifact_versions": self.artifact_versions,
                "review_status": self.review_status,
                "queue_types": self.queue_types,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperienceRecord":
        return cls(
            record_id=str(payload["record_id"]),
            created_at=str(payload["created_at"]),
            sample_id=str(payload["sample_id"]),
            mutation_type=str(payload["mutation_type"]),
            request_id=payload.get("request_id"),
            source_context=dict(payload.get("source_context") or {}),
            input_summary=dict(payload.get("input_summary") or {}),
            expert_outputs=dict(payload.get("expert_outputs") or {}),
            fusion_report=dict(payload.get("fusion_report") or {}),
            fused_sample_result=dict(payload.get("fused_sample_result") or {}),
            cohort_context=dict(payload.get("cohort_context") or {}),
            recommendation=dict(payload.get("recommendation") or {}),
            artifact_versions=dict(payload.get("artifact_versions") or {}),
            review_status=str(payload.get("review_status") or "pending_review"),
            queue_types=[str(value) for value in payload.get("queue_types") or []],
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_index_row(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "created_at": self.created_at,
            "sample_id": self.sample_id,
            "mutation_type": self.mutation_type,
            "request_id": self.request_id,
            "primary_recommendation": self.recommendation.get("primary_recommendation"),
            "catalog_insufficiency_level": self.fusion_report.get("catalog_insufficiency_level"),
            "catalog_insufficiency_probability": self.fusion_report.get("catalog_insufficiency_probability"),
            "assignment_confidence_probability": self.fusion_report.get("assignment_confidence_probability"),
            "review_status": self.review_status,
            "queue_types": ",".join(self.queue_types),
            "sample_hash": self.input_summary.get("sample_hash"),
        }


@dataclass(slots=True)
class ReviewDecision:
    review_id: str
    record_id: str
    created_at: str
    reviewer: str
    review_outcome: str
    evidence_type: str
    validated_recommendation: str | None = None
    catalog_insufficiency_confirmed: bool | None = None
    reference_reassessment_confirmed: bool | None = None
    preprocessing_issue_flag: bool | None = None
    notes: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "review_id": self.review_id,
                "record_id": self.record_id,
                "created_at": self.created_at,
                "reviewer": self.reviewer,
                "review_outcome": self.review_outcome,
                "evidence_type": self.evidence_type,
                "validated_recommendation": self.validated_recommendation,
                "catalog_insufficiency_confirmed": self.catalog_insufficiency_confirmed,
                "reference_reassessment_confirmed": self.reference_reassessment_confirmed,
                "preprocessing_issue_flag": self.preprocessing_issue_flag,
                "notes": self.notes,
                "labels": self.labels,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReviewDecision":
        return cls(
            review_id=str(payload["review_id"]),
            record_id=str(payload["record_id"]),
            created_at=str(payload["created_at"]),
            reviewer=str(payload["reviewer"]),
            review_outcome=str(payload["review_outcome"]),
            evidence_type=str(payload["evidence_type"]),
            validated_recommendation=payload.get("validated_recommendation"),
            catalog_insufficiency_confirmed=payload.get("catalog_insufficiency_confirmed"),
            reference_reassessment_confirmed=payload.get("reference_reassessment_confirmed"),
            preprocessing_issue_flag=payload.get("preprocessing_issue_flag"),
            notes=payload.get("notes"),
            labels=dict(payload.get("labels") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_index_row(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "record_id": self.record_id,
            "created_at": self.created_at,
            "reviewer": self.reviewer,
            "review_outcome": self.review_outcome,
            "evidence_type": self.evidence_type,
            "validated_recommendation": self.validated_recommendation,
            "catalog_insufficiency_confirmed": self.catalog_insufficiency_confirmed,
            "reference_reassessment_confirmed": self.reference_reassessment_confirmed,
            "preprocessing_issue_flag": self.preprocessing_issue_flag,
        }


__all__ = [
    "ExperienceRecord",
    "ReviewDecision",
]
