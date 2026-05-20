from __future__ import annotations

from typing import Any

from ..conformal_groups import group_context_from_sample_counts
from .base import BaseExpert, build_sample_results, ensure_runtime_env
from .optional_paths import add_optional_import_path
from .schema import ExpertRequest, ExpertRunResult


class AMuSaExpert(BaseExpert):
    expert_name = "amusa"

    def __init__(
        self,
        *,
        prediction_profile: str | None = None,
        exposure_profile: str = "sig_exposure_plus",
        max_active_signatures: int | None = None,
        min_exposure: float = 0.01,
        probability_calibrator: Any | None = None,
        repo_root=None,
    ) -> None:
        super().__init__(repo_root=repo_root)
        self.prediction_profile = prediction_profile
        self.exposure_profile = exposure_profile
        self.max_active_signatures = max_active_signatures
        self.min_exposure = min_exposure
        self.probability_calibrator = probability_calibrator

    def parameter_snapshot(self) -> dict:
        return {
            "prediction_profile": self.prediction_profile,
            "exposure_profile": self.exposure_profile,
            "max_active_signatures": self.max_active_signatures,
            "min_exposure": self.min_exposure,
            "probability_calibrator": None if self.probability_calibrator is None else self.probability_calibrator.to_dict(),
        }

    def _calibrate_probability_frame(self, request: ExpertRequest, support):
        if self.probability_calibrator is None:
            return None
        calibrated_probability_df = support.probability_df.copy()
        if hasattr(self.probability_calibrator, "transform_frame"):
            contexts_by_row = {
                sample_id: group_context_from_sample_counts(
                    request.sample_matrix.loc[:, sample_id],
                    mutation_type=request.mutation_type,
                ).to_dict()
                for sample_id in support.sample_ids
            }
            return self.probability_calibrator.transform_frame(
                calibrated_probability_df,
                contexts_by_row=contexts_by_row,
            )
        calibrated_probability_df.loc[:, :] = self.probability_calibrator.transform(
            calibrated_probability_df.to_numpy(dtype=float).ravel()
        ).reshape(calibrated_probability_df.shape)
        return calibrated_probability_df

    def _run_impl(self, request: ExpertRequest) -> ExpertRunResult:
        ensure_runtime_env(self.repo_root)
        add_optional_import_path(
            env_var="SIGAGENT_AMUSA_PATH",
            repo_root=self.repo_root,
            default_local_dir="AMuSa",
            package_name="AMuSa",
        )
        try:
            from AMuSa.runtime import AMuSaRuntimeConfig, run_official_pipeline
        except Exception as exc:
            raise ImportError(
                "AMuSA is not bundled with the minimal SigAgent release. "
                "Provide it on PYTHONPATH or set SIGAGENT_AMUSA_PATH before "
                "requesting AMuSA-derived experts."
            ) from exc

        pipeline = run_official_pipeline(
            sample_matrix=request.sample_matrix.loc[:, request.sample_ids],
            signature_matrix=request.signature_matrix.loc[:, request.signature_names],
            requested_signature_names=request.signature_names,
            mutation_type=request.mutation_type,
            repo_root=self.repo_root,
            config=AMuSaRuntimeConfig(
                mutation_type=request.mutation_type,
                prediction_profile=self.prediction_profile,
                exposure_profile=self.exposure_profile,
                max_active_signatures=self.max_active_signatures,
                min_exposure=self.min_exposure,
            ),
        )
        calibrated_probability_df = self._calibrate_probability_frame(request, pipeline.support)
        diagnostics_by_sample = {}
        for sample_id in pipeline.support.sample_ids:
            diagnostics_by_sample[sample_id] = {
                **pipeline.support.diagnostics_by_sample.get(sample_id, {}),
                "core_method": "amusa_official_pipeline",
                "prediction_profile": pipeline.support.profile.name,
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
                exposures=pipeline.exposures_df,
                signature_scores=pipeline.support.probability_df,
                signature_probabilities=calibrated_probability_df,
                diagnostics_by_sample=diagnostics_by_sample,
            ),
            parameters=self.parameter_snapshot(),
            artifacts={
                **pipeline.artifacts,
                "implementation": "amusa_official_pipeline",
            },
        )


__all__ = [
    "AMuSaExpert",
]
