from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import ExperienceRecord, ReviewDecision


QUEUE_TYPES = [
    "manual_review",
    "cohort_level_discovery",
    "reassess_reference_catalog",
]


@dataclass(slots=True)
class ReviewQueueOutput:
    combined_frame: pd.DataFrame
    frames_by_queue: dict[str, pd.DataFrame]
    summary: dict[str, Any]


def latest_reviews_by_record(reviews: list[ReviewDecision]) -> dict[str, ReviewDecision]:
    latest: dict[str, ReviewDecision] = {}
    for review in sorted(reviews, key=lambda item: (item.created_at, item.review_id)):
        latest[review.record_id] = review
    return latest


def _queue_reason(record: ExperienceRecord, queue_type: str) -> str:
    queue_reasons = record.cohort_context.get("queue_reasons") or {}
    if queue_type in queue_reasons:
        return str(queue_reasons[queue_type])
    rationale = record.recommendation.get("recommendation_rationale") or []
    if isinstance(rationale, list):
        return " | ".join(str(value) for value in rationale)
    return str(rationale)


def _is_unresolved_review(review: ReviewDecision | None) -> bool:
    if review is None:
        return True
    return review.review_outcome == "deferred"


def build_review_queue_output(
    records: list[ExperienceRecord],
    reviews: list[ReviewDecision] | None = None,
) -> ReviewQueueOutput:
    reviews = reviews or []
    latest_reviews = latest_reviews_by_record(reviews)
    combined_rows: list[dict[str, Any]] = []
    for record in records:
        latest_review = latest_reviews.get(record.record_id)
        if not _is_unresolved_review(latest_review):
            continue
        for queue_type in record.queue_types:
            if queue_type not in QUEUE_TYPES:
                continue
            combined_rows.append(
                {
                    "queue_type": queue_type,
                    "record_id": record.record_id,
                    "sample_id": record.sample_id,
                    "mutation_type": record.mutation_type,
                    "primary_recommendation": record.recommendation.get("primary_recommendation"),
                    "validated_recommendation": None if latest_review is None else latest_review.validated_recommendation,
                    "catalog_insufficiency_level": record.fusion_report.get("catalog_insufficiency_level"),
                    "catalog_insufficiency_probability": record.fusion_report.get("catalog_insufficiency_probability"),
                    "assignment_confidence_probability": record.fusion_report.get("assignment_confidence_probability"),
                    "reason": _queue_reason(record, queue_type),
                    "review_status": "pending_review" if latest_review is None else latest_review.review_outcome,
                    "created_at": record.created_at,
                }
            )
    combined_frame = pd.DataFrame.from_records(
        combined_rows,
        columns=[
            "queue_type",
            "record_id",
            "sample_id",
            "mutation_type",
            "primary_recommendation",
            "validated_recommendation",
            "catalog_insufficiency_level",
            "catalog_insufficiency_probability",
            "assignment_confidence_probability",
            "reason",
            "review_status",
            "created_at",
        ],
    )
    frames_by_queue = {
        queue_type: combined_frame.loc[combined_frame["queue_type"] == queue_type].copy()
        if not combined_frame.empty
        else pd.DataFrame(columns=combined_frame.columns)
        for queue_type in QUEUE_TYPES
    }
    summary = {
        "n_records": len(records),
        "n_reviews": len(reviews),
        "queue_counts": {
            queue_type: int(len(frame))
            for queue_type, frame in frames_by_queue.items()
        },
    }
    return ReviewQueueOutput(
        combined_frame=combined_frame,
        frames_by_queue=frames_by_queue,
        summary=summary,
    )


def write_review_queues(output: ReviewQueueOutput, root_dir: str | Path) -> Path:
    root = Path(root_dir)
    queue_dir = root / "queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    output.combined_frame.to_csv(queue_dir / "combined.tsv", sep="\t", index=False)
    for queue_type, frame in output.frames_by_queue.items():
        frame.to_csv(queue_dir / f"{queue_type}.tsv", sep="\t", index=False)
    (queue_dir / "summary.json").write_text(json.dumps(output.summary, indent=2, ensure_ascii=False))
    return queue_dir


__all__ = [
    "QUEUE_TYPES",
    "ReviewQueueOutput",
    "build_review_queue_output",
    "latest_reviews_by_record",
    "write_review_queues",
]
