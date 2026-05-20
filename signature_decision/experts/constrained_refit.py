from __future__ import annotations

from typing import Any

from ..conformal_groups import group_context_from_sample_counts
from .amusa_support import run_amusa_support
from .base import BaseExpert, build_sample_results
from .nnls_utils import solve_nnls_matrix
from .schema import ExpertRequest, ExpertRunResult


class ClassifierGuidedRefitExpert(BaseExpert):
    expert_name = "classifier_guided_refit"

    def __init__(
        self,
        *,
        prediction_profile: str | None = None,
        max_active_signatures: int | None = None,
        fallback_top_k: int = 3,
        probability_calibrator: Any | None = None,
        repo_root=None,
    ) -> None:
        super().__init__(repo_root=repo_root)
        self.prediction_profile = prediction_profile
        self.max_active_signatures = max_active_signatures
        self.fallback_top_k = fallback_top_k
        self.probability_calibrator = probability_calibrator

    def parameter_snapshot(self) -> dict:
        return {
            "prediction_profile": self.prediction_profile,
            "max_active_signatures": self.max_active_signatures,
            "fallback_top_k": self.fallback_top_k,
            "probability_calibrator": None if self.probability_calibrator is None else self.probability_calibrator.to_dict(),
        }

    def _run_impl(self, request: ExpertRequest) -> ExpertRunResult:
        support = run_amusa_support(
            request=request,
            repo_root=self.repo_root,
            max_active_signatures=self.max_active_signatures,
            prediction_profile=self.prediction_profile,
        )

        candidate_signatures_by_sample: dict[str, list[str]] = {}
        for sample_id in support.sample_ids:
            active_names = [
                signature_name
                for signature_name, is_active in support.active_prediction_df.loc[sample_id].items()
                if int(is_active) == 1
            ]
            if not active_names:
                ranked_signatures = (
                    support.probability_df.loc[sample_id]
                    .sort_values(ascending=False)
                    .head(self.fallback_top_k)
                    .index
                    .tolist()
                )
                active_names = [str(signature_name) for signature_name in ranked_signatures]
            candidate_signatures_by_sample[sample_id] = active_names

        exposures_df, nnls_diagnostics = solve_nnls_matrix(
            sample_matrix=request.sample_matrix.loc[:, support.sample_ids],
            signature_matrix=support.aligned_signature_matrix,
            signature_names=support.available_signature_names,
            candidate_signatures_by_sample=candidate_signatures_by_sample,
            fallback_top_k=self.fallback_top_k,
        )

        diagnostics_by_sample = {}
        for sample_id in support.sample_ids:
            diagnostics_by_sample[sample_id] = {
                **support.diagnostics_by_sample.get(sample_id, {}),
                **nnls_diagnostics.get(sample_id, {}),
                "core_method": "amusa_support_plus_constrained_nnls",
                "candidate_source": "amusa_classifier_support",
                "prediction_profile": support.prediction_profile,
            }
        calibrated_probability_df = None
        if self.probability_calibrator is not None:
            calibrated_probability_df = support.probability_df.copy()
            if hasattr(self.probability_calibrator, "transform_frame"):
                contexts_by_row = {
                    sample_id: group_context_from_sample_counts(
                        request.sample_matrix.loc[:, sample_id],
                        mutation_type=request.mutation_type,
                    ).to_dict()
                    for sample_id in support.sample_ids
                }
                calibrated_probability_df = self.probability_calibrator.transform_frame(
                    calibrated_probability_df,
                    contexts_by_row=contexts_by_row,
                )
            else:
                calibrated_probability_df.loc[:, :] = self.probability_calibrator.transform(
                    calibrated_probability_df.to_numpy(dtype=float).ravel()
                ).reshape(calibrated_probability_df.shape)

        return ExpertRunResult(
            expert_name=self.expert_name,
            mutation_type=request.mutation_type,
            request_id=request.request_id or "request",
            status="success",
            signature_names=request.signature_names,
            channel_ids=request.channel_ids,
            sample_results=build_sample_results(
                request=request,
                exposures=exposures_df,
                signature_scores=support.probability_df,
                signature_probabilities=calibrated_probability_df,
                diagnostics_by_sample=diagnostics_by_sample,
            ),
            parameters=self.parameter_snapshot(),
            artifacts={
                **support.artifacts,
                "implementation": "amusa_support_plus_constrained_nnls",
            },
        )


__all__ = [
    "ClassifierGuidedRefitExpert",
]
