from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .experts.schema import ExpertRequest, ExpertRunResult, ExpertSampleResult, json_ready


@dataclass(slots=True)
class SampleBatch:
    mutation_type: str
    sample_matrix: pd.DataFrame
    channel_metadata: pd.DataFrame | None = None
    source: str | None = None
    batch_id: str | None = None

    @property
    def sample_ids(self) -> list[str]:
        return [str(value) for value in self.sample_matrix.columns.tolist()]

    @property
    def channel_ids(self) -> list[str]:
        return [str(value) for value in self.sample_matrix.index.tolist()]


@dataclass(slots=True)
class ReferenceCatalog:
    mutation_type: str
    signature_matrix: pd.DataFrame
    channel_metadata: pd.DataFrame | None = None
    source: str | None = None
    reference_name: str | None = None

    @property
    def signature_names(self) -> list[str]:
        return [str(value) for value in self.signature_matrix.columns.tolist()]

    @property
    def channel_ids(self) -> list[str]:
        return [str(value) for value in self.signature_matrix.index.tolist()]


@dataclass(slots=True)
class GroundTruthSet:
    mutation_type: str
    exposures: pd.DataFrame
    active_threshold: float = 0.0
    catalog_insufficient_labels: pd.Series | None = None
    source: str | None = None

    def align(self, sample_ids: list[str], signature_names: list[str]) -> "GroundTruthSet":
        exposures = self.exposures.reindex(index=signature_names, columns=sample_ids, fill_value=0.0)
        labels = None
        if self.catalog_insufficient_labels is not None:
            labels = self.catalog_insufficient_labels.reindex(sample_ids).fillna(0).astype(int)
        return GroundTruthSet(
            mutation_type=self.mutation_type,
            exposures=exposures,
            active_threshold=self.active_threshold,
            catalog_insufficient_labels=labels,
            source=self.source,
        )


@dataclass(slots=True)
class FinalSampleReport:
    sample_id: str
    mutation_type: str
    known_signatures: list[dict[str, Any]]
    unstable_conclusions: list[dict[str, Any]] = field(default_factory=list)
    catalog_insufficiency_proxy_score: float | None = None
    catalog_insufficiency_probability: float | None = None
    catalog_insufficiency_level: str | None = None
    assignment_confidence_raw_score: float | None = None
    assignment_confidence_probability: float | None = None
    assignment_confidence: float | None = None
    primary_recommendation: str | None = None
    secondary_recommendations: list[str] = field(default_factory=list)
    recommendation_rationale: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "sample_id": self.sample_id,
                "mutation_type": self.mutation_type,
                "known_signatures": self.known_signatures,
                "unstable_conclusions": self.unstable_conclusions,
                "catalog_insufficiency_proxy_score": self.catalog_insufficiency_proxy_score,
                "catalog_insufficiency_probability": self.catalog_insufficiency_probability,
                "catalog_insufficiency_level": self.catalog_insufficiency_level,
                "assignment_confidence_raw_score": self.assignment_confidence_raw_score,
                "assignment_confidence_probability": self.assignment_confidence_probability,
                "assignment_confidence": self.assignment_confidence,
                "primary_recommendation": self.primary_recommendation,
                "secondary_recommendations": self.secondary_recommendations,
                "recommendation_rationale": self.recommendation_rationale,
                "rationale": self.recommendation_rationale,
                "metadata": self.metadata,
            }
        )


@dataclass(slots=True)
class CohortSummaryReport:
    mutation_type: str
    sample_ids: list[str]
    recommendation_counts: dict[str, int]
    catalog_insufficiency_level_counts: dict[str, int]
    manual_review_candidates: list[str] = field(default_factory=list)
    cohort_discovery_candidates: list[str] = field(default_factory=list)
    reference_reassessment_candidates: list[str] = field(default_factory=list)
    direct_downstream_candidates: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "mutation_type": self.mutation_type,
                "sample_ids": self.sample_ids,
                "n_samples": self.n_samples,
                "recommendation_counts": self.recommendation_counts,
                "catalog_insufficiency_level_counts": self.catalog_insufficiency_level_counts,
                "manual_review_candidates": self.manual_review_candidates,
                "cohort_discovery_candidates": self.cohort_discovery_candidates,
                "reference_reassessment_candidates": self.reference_reassessment_candidates,
                "direct_downstream_candidates": self.direct_downstream_candidates,
                "metadata": self.metadata,
            }
        )


@dataclass(slots=True)
class BenchmarkSliceResult:
    benchmark_name: str
    mutation_type: str
    aggregate_metrics: pd.DataFrame
    per_sample_metrics: pd.DataFrame
    parameters: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "benchmark_name": self.benchmark_name,
                "mutation_type": self.mutation_type,
                "aggregate_metrics": self.aggregate_metrics,
                "per_sample_metrics": self.per_sample_metrics,
                "parameters": self.parameters,
                "artifacts": self.artifacts,
            }
        )


@dataclass(slots=True)
class BenchmarkSuiteResult:
    benchmark_name: str
    mutation_type: str
    slices: list[BenchmarkSliceResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregate_metrics(self) -> pd.DataFrame:
        if not self.slices:
            return pd.DataFrame()
        return pd.concat([slice_result.aggregate_metrics for slice_result in self.slices], ignore_index=True)

    @property
    def per_sample_metrics(self) -> pd.DataFrame:
        if not self.slices:
            return pd.DataFrame()
        return pd.concat([slice_result.per_sample_metrics for slice_result in self.slices], ignore_index=True)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "benchmark_name": self.benchmark_name,
                "mutation_type": self.mutation_type,
                "slices": [slice_result.to_dict() for slice_result in self.slices],
                "metadata": self.metadata,
            }
        )


__all__ = [
    "BenchmarkSliceResult",
    "BenchmarkSuiteResult",
    "CohortSummaryReport",
    "ExpertRequest",
    "ExpertRunResult",
    "ExpertSampleResult",
    "FinalSampleReport",
    "GroundTruthSet",
    "ReferenceCatalog",
    "SampleBatch",
]
