from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .schemas import CohortSummaryReport, FinalSampleReport


@dataclass(slots=True)
class CohortAggregationOutput:
    report: CohortSummaryReport
    summary_frame: pd.DataFrame
    candidates_frame: pd.DataFrame


def _recommendation_present(report: FinalSampleReport, recommendation: str) -> bool:
    return report.primary_recommendation == recommendation or recommendation in report.secondary_recommendations


def aggregate_cohort_reports(reports: list[FinalSampleReport]) -> CohortAggregationOutput:
    if not reports:
        empty_report = CohortSummaryReport(
            mutation_type="unknown",
            sample_ids=[],
            recommendation_counts={},
            catalog_insufficiency_level_counts={},
        )
        return CohortAggregationOutput(
            report=empty_report,
            summary_frame=pd.DataFrame(),
            candidates_frame=pd.DataFrame(),
        )

    recommendation_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    manual_review_candidates: list[str] = []
    cohort_discovery_candidates: list[str] = []
    reference_reassessment_candidates: list[str] = []
    direct_downstream_candidates: list[str] = []

    for report in reports:
        recommendation = report.primary_recommendation or "unknown"
        recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1
        insufficiency_level = report.catalog_insufficiency_level or "unknown"
        level_counts[insufficiency_level] = level_counts.get(insufficiency_level, 0) + 1

        summary_rows.append(
            {
                "sample_id": report.sample_id,
                "mutation_type": report.mutation_type,
                "primary_recommendation": report.primary_recommendation,
                "secondary_recommendations": ",".join(report.secondary_recommendations),
                "catalog_insufficiency_proxy_score": report.catalog_insufficiency_proxy_score,
                "catalog_insufficiency_probability": report.catalog_insufficiency_probability,
                "catalog_insufficiency_level": report.catalog_insufficiency_level,
                "assignment_confidence_raw_score": report.assignment_confidence_raw_score,
                "assignment_confidence_probability": report.assignment_confidence_probability,
                "top_signatures": ",".join(item["name"] for item in report.known_signatures[:5]),
                "unstable_count": len(report.unstable_conclusions),
            }
        )

        if _recommendation_present(report, "manual_review"):
            manual_review_candidates.append(report.sample_id)
            candidate_rows.append(
                {
                    "sample_id": report.sample_id,
                    "candidate_type": "manual_review",
                    "reason": " | ".join(report.recommendation_rationale),
                }
            )
        if _recommendation_present(report, "cohort_level_discovery"):
            cohort_discovery_candidates.append(report.sample_id)
            candidate_rows.append(
                {
                    "sample_id": report.sample_id,
                    "candidate_type": "cohort_level_discovery",
                    "reason": " | ".join(report.recommendation_rationale),
                }
            )
        if _recommendation_present(report, "reassess_reference_catalog"):
            reference_reassessment_candidates.append(report.sample_id)
            candidate_rows.append(
                {
                    "sample_id": report.sample_id,
                    "candidate_type": "reassess_reference_catalog",
                    "reason": " | ".join(report.recommendation_rationale),
                }
            )
        if report.primary_recommendation == "direct_downstream_analysis":
            direct_downstream_candidates.append(report.sample_id)

    mutation_types = {report.mutation_type for report in reports}
    mutation_type = sorted(mutation_types)[0] if len(mutation_types) == 1 else "mixed"
    report = CohortSummaryReport(
        mutation_type=mutation_type,
        sample_ids=[report.sample_id for report in reports],
        recommendation_counts=recommendation_counts,
        catalog_insufficiency_level_counts=level_counts,
        manual_review_candidates=manual_review_candidates,
        cohort_discovery_candidates=cohort_discovery_candidates,
        reference_reassessment_candidates=reference_reassessment_candidates,
        direct_downstream_candidates=direct_downstream_candidates,
        metadata={
            "mean_assignment_confidence": float(
                pd.Series([report.assignment_confidence_probability for report in reports], dtype=float).dropna().mean()
            )
            if any(report.assignment_confidence_probability is not None for report in reports)
            else None,
            "mean_catalog_insufficiency_probability": float(
                pd.Series([report.catalog_insufficiency_probability for report in reports], dtype=float).dropna().mean()
            )
            if any(report.catalog_insufficiency_probability is not None for report in reports)
            else None,
        },
    )
    return CohortAggregationOutput(
        report=report,
        summary_frame=pd.DataFrame.from_records(summary_rows),
        candidates_frame=pd.DataFrame.from_records(
            candidate_rows,
            columns=["sample_id", "candidate_type", "reason"],
        ),
    )


__all__ = [
    "CohortAggregationOutput",
    "aggregate_cohort_reports",
]
