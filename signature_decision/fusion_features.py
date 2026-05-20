from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .catalog_insufficiency import (
    CatalogInsufficiencyAssessment,
    assess_catalog_insufficiency_from_expert_results,
    normalize_exposure_dict,
    pairwise_jaccard,
)
from .conformal_groups import group_context_from_sample_counts
from .experts.schema import ExpertRequest, ExpertSampleResult


@dataclass(slots=True)
class FusionEvidence:
    sample_id: str
    mutation_type: str
    mutation_count: float
    expert_names: list[str]
    failed_expert_names: list[str]
    active_sets: dict[str, set[str]]
    normalized_exposures_by_expert: dict[str, dict[str, float]]
    agreement_score: float
    disagreement_score: float
    pairwise_active_set_jaccard: dict[str, float]
    mean_reconstruction_cosine: float
    catalog_assessment: CatalogInsufficiencyAssessment
    catalog_insufficiency_proxy_score: float
    catalog_insufficiency_probability: float | None
    catalog_insufficiency_level: str
    residual_structure_score: float
    group_context: dict[str, Any] = field(default_factory=dict)

    def to_feature_row(self) -> dict[str, Any]:
        features = self.catalog_assessment.features
        row = {
            "sample_id": self.sample_id,
            "mutation_type": self.mutation_type,
            "mutation_count": self.mutation_count,
            "expert_count": len(self.expert_names),
            "failed_expert_count": len(self.failed_expert_names),
            "agreement_score": self.agreement_score,
            "disagreement_score": self.disagreement_score,
            "mean_reconstruction_cosine": self.mean_reconstruction_cosine,
            "catalog_insufficiency_proxy_score": self.catalog_insufficiency_proxy_score,
            "catalog_insufficiency_probability": self.catalog_insufficiency_probability,
            "catalog_insufficiency_level": self.catalog_insufficiency_level,
            "residual_structure_score": self.residual_structure_score,
            **self.group_context,
        }
        for key, value in asdict(features).items():
            if isinstance(value, dict):
                continue
            row[f"catalog_feature_{key}"] = value
        return row


def extract_fusion_evidence(
    *,
    sample_id: str,
    request: ExpertRequest,
    sample_results_by_expert: dict[str, ExpertSampleResult],
    failed_expert_names: list[str] | None = None,
    catalog_assessor_model: Any | None = None,
) -> FusionEvidence:
    failed_expert_names = list(failed_expert_names or [])
    expert_names = list(sample_results_by_expert.keys())
    mutation_count = float(request.sample_matrix.loc[:, sample_id].sum())
    active_sets = {
        expert_name: set(sample_result.active_signatures)
        for expert_name, sample_result in sample_results_by_expert.items()
    }
    agreement_score, pairwise_scores = pairwise_jaccard(active_sets)
    disagreement_score = float(1.0 - agreement_score)
    mean_reconstruction_cosine = float(
        np.mean(
            [
                sample_result.metrics.get("reconstruction_cosine", 0.0)
                for sample_result in sample_results_by_expert.values()
            ]
        )
    ) if sample_results_by_expert else 0.0
    normalized_exposures_by_expert = {
        expert_name: normalize_exposure_dict(sample_result.exposures)
        for expert_name, sample_result in sample_results_by_expert.items()
    }
    base_group_context = group_context_from_sample_counts(
        request.sample_matrix.loc[:, sample_id],
        mutation_type=request.mutation_type,
        disagreement_score=disagreement_score,
    ).to_dict()
    if catalog_assessor_model is not None:
        catalog_assessment = catalog_assessor_model.assess(
            sample_results_by_expert,
            failed_expert_count=len(failed_expert_names),
            group_context=base_group_context,
        )
    else:
        catalog_assessment = assess_catalog_insufficiency_from_expert_results(
            sample_results_by_expert,
            failed_expert_count=len(failed_expert_names),
        )
    catalog_probability = None if catalog_assessor_model is None else getattr(catalog_assessment, "probability", None)
    catalog_level = str(catalog_assessment.level)
    group_context = group_context_from_sample_counts(
        request.sample_matrix.loc[:, sample_id],
        mutation_type=request.mutation_type,
        disagreement_score=disagreement_score,
        risk_level=catalog_level,
    ).to_dict()
    return FusionEvidence(
        sample_id=sample_id,
        mutation_type=request.mutation_type,
        mutation_count=mutation_count,
        expert_names=expert_names,
        failed_expert_names=failed_expert_names,
        active_sets=active_sets,
        normalized_exposures_by_expert=normalized_exposures_by_expert,
        agreement_score=float(agreement_score),
        disagreement_score=disagreement_score,
        pairwise_active_set_jaccard=pairwise_scores,
        mean_reconstruction_cosine=mean_reconstruction_cosine,
        catalog_assessment=catalog_assessment,
        catalog_insufficiency_proxy_score=float(catalog_assessment.score),
        catalog_insufficiency_probability=None if catalog_probability is None else float(catalog_probability),
        catalog_insufficiency_level=catalog_level,
        residual_structure_score=float(catalog_assessment.features.mean_residual_structure_score),
        group_context=group_context,
    )


__all__ = [
    "FusionEvidence",
    "extract_fusion_evidence",
]

