from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .review_queue import latest_reviews_by_record
from .schema import ExperienceRecord, ReviewDecision


HIGH_QUALITY_OUTCOMES = {"confirmed", "corrected"}
HIGH_QUALITY_EVIDENCE_TYPES = {
    "manual_review",
    "cohort_discovery",
    "reference_reassessment",
    "external_validation",
}


def is_high_quality_review(review: ReviewDecision) -> bool:
    if review.review_outcome not in HIGH_QUALITY_OUTCOMES:
        return False
    if review.preprocessing_issue_flag is True:
        return False
    if review.evidence_type not in HIGH_QUALITY_EVIDENCE_TYPES:
        return False
    return True


def build_experience_dataset(
    records: list[ExperienceRecord],
    reviews: list[ReviewDecision],
    *,
    include_all_reviewed: bool = False,
) -> pd.DataFrame:
    latest_reviews = latest_reviews_by_record(reviews)
    rows: list[dict[str, Any]] = []
    for record in records:
        review = latest_reviews.get(record.record_id)
        if review is None:
            continue
        high_quality = is_high_quality_review(review)
        if not include_all_reviewed and not high_quality:
            continue
        labels = review.labels or {}
        active_signatures = labels.get("active_signatures") or []
        if isinstance(active_signatures, list):
            active_signatures = ",".join(str(value) for value in active_signatures)
        rows.append(
            {
                "record_id": record.record_id,
                "sample_id": record.sample_id,
                "mutation_type": record.mutation_type,
                "created_at": record.created_at,
                "sample_hash": record.input_summary.get("sample_hash"),
                "mutation_count": record.input_summary.get("mutation_count"),
                "primary_recommendation": record.recommendation.get("primary_recommendation"),
                "secondary_recommendations": ",".join(record.recommendation.get("secondary_recommendations") or []),
                "catalog_insufficiency_proxy_score": record.fusion_report.get("catalog_insufficiency_proxy_score"),
                "catalog_insufficiency_probability": record.fusion_report.get("catalog_insufficiency_probability"),
                "assignment_confidence_raw_score": record.fusion_report.get("assignment_confidence_raw_score"),
                "assignment_confidence_probability": record.fusion_report.get("assignment_confidence_probability"),
                "queue_types": ",".join(record.queue_types),
                "review_id": review.review_id,
                "review_outcome": review.review_outcome,
                "evidence_type": review.evidence_type,
                "validated_recommendation": review.validated_recommendation,
                "catalog_insufficiency_confirmed": review.catalog_insufficiency_confirmed,
                "reference_reassessment_confirmed": review.reference_reassessment_confirmed,
                "preprocessing_issue_flag": review.preprocessing_issue_flag,
                "validated_active_signatures": active_signatures,
                "review_notes": review.notes,
                "is_high_quality": high_quality,
            }
        )
    return pd.DataFrame.from_records(rows)


def write_experience_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, sep="\t", index=False)
    return output_path


__all__ = [
    "HIGH_QUALITY_EVIDENCE_TYPES",
    "HIGH_QUALITY_OUTCOMES",
    "build_experience_dataset",
    "is_high_quality_review",
    "write_experience_dataset",
]
