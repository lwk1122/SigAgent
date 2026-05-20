from __future__ import annotations

from .base import BaseExpert, build_sample_results
from .nnls_utils import solve_nnls_matrix
from .schema import ExpertRequest, ExpertRunResult


class PlainNNLSExpert(BaseExpert):
    expert_name = "plain_nnls"

    def __init__(
        self,
        *,
        fallback_top_k: int = 1,
        repo_root=None,
    ) -> None:
        super().__init__(repo_root=repo_root)
        self.fallback_top_k = fallback_top_k

    def parameter_snapshot(self) -> dict:
        return {
            "fallback_top_k": self.fallback_top_k,
        }

    def _run_impl(self, request: ExpertRequest) -> ExpertRunResult:
        exposures, diagnostics_by_sample = solve_nnls_matrix(
            sample_matrix=request.sample_matrix,
            signature_matrix=request.signature_matrix,
            signature_names=request.signature_names,
            fallback_top_k=self.fallback_top_k,
        )
        for sample_id, diagnostics in diagnostics_by_sample.items():
            diagnostics["core_method"] = "scipy.optimize.nnls"

        return ExpertRunResult(
            expert_name=self.expert_name,
            mutation_type=request.mutation_type,
            request_id=request.request_id or "request",
            status="success",
            signature_names=request.signature_names,
            channel_ids=request.channel_ids,
            sample_results=build_sample_results(
                request=request,
                exposures=exposures,
                diagnostics_by_sample=diagnostics_by_sample,
            ),
            parameters=self.parameter_snapshot(),
            artifacts={"implementation": "scipy.optimize.nnls"},
        )


__all__ = [
    "PlainNNLSExpert",
]
