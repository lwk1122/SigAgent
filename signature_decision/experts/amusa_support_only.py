from __future__ import annotations

from typing import Any

from ..conformal_groups import group_context_from_sample_counts
from .amusa_support import run_amusa_support
from .base import BaseExpert, build_sample_results, ensure_runtime_env
from .schema import ExpertRequest, ExpertRunResult


class AMuSaSupportOnlyExpert(BaseExpert):
    expert_name = "amusa_support_only"

    def __init__(
        self,
        *,
        prediction_profile: str = "checkpoint_thresholds",
        exposure_profile: str = "bidirectional",
        min_exposure: float = 0.01,
        max_active_signatures: int | None = 6,
        probability_calibrator: Any | None = None,
        repo_root=None,
    ) -> None:
        super().__init__(repo_root=repo_root)
        self.prediction_profile = prediction_profile
        self.exposure_profile = exposure_profile
        self.min_exposure = min_exposure
        self.max_active_signatures = max_active_signatures
        self.probability_calibrator = probability_calibrator

    def parameter_snapshot(self) -> dict:
        return {
            "prediction_profile": self.prediction_profile,
            "exposure_profile": self.exposure_profile,
            "min_exposure": self.min_exposure,
            "max_active_signatures": self.max_active_signatures,
            "probability_calibrator": None if self.probability_calibrator is None else self.probability_calibrator.to_dict(),
        }

    def _run_impl(self, request: ExpertRequest) -> ExpertRunResult:
        ensure_runtime_env(self.repo_root)
        support = run_amusa_support(
            request=request,
            repo_root=self.repo_root,
            max_active_signatures=self.max_active_signatures,
            prediction_profile=self.prediction_profile,
        )
        try:
            from AMuSa.runtime import estimate_exposures_runtime
        except Exception as exc:
            raise ImportError(
                "AMuSA is not bundled with the minimal SigAgent release. "
                "Provide it on PYTHONPATH or set SIGAGENT_AMUSA_PATH before "
                "requesting AMuSA-derived experts."
            ) from exc

        exposures_df = estimate_exposures_runtime(
            sample_matrix=request.sample_matrix.loc[:, support.sample_ids].set_axis(support.feature_names, axis=0),
            signature_matrix=support.aligned_signature_matrix,
            active_prediction_df=support.active_prediction_df,
            mutation_type=request.mutation_type,
            repo_root=self.repo_root,
            exposure_profile=self.exposure_profile,
            min_exposure=self.min_exposure,
        )

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

        diagnostics_by_sample = {}
        for sample_id in support.sample_ids:
            diagnostics_by_sample[sample_id] = {
                **support.diagnostics_by_sample.get(sample_id, {}),
                "core_method": "amusa_support_only",
                "prediction_profile": support.prediction_profile,
                "exposure_profile": self.exposure_profile,
            }

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
                "implementation": "amusa_support_only",
            },
        )


__all__ = [
    "AMuSaSupportOnlyExpert",
]
