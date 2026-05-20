from __future__ import annotations

from .base import BaseExpert, build_sample_results, ensure_runtime_env
from .optional_paths import add_optional_import_path
from .schema import ExpertRequest, ExpertRunResult


class MuSiCalExpert(BaseExpert):
    expert_name = "musical"

    def __init__(
        self,
        *,
        method: str = "likelihood_bidirectional",
        thresh: float | None = 0.001,
        connected_sigs: bool = False,
        repo_root=None,
    ) -> None:
        super().__init__(repo_root=repo_root)
        self.method = method
        self.thresh = thresh
        self.connected_sigs = connected_sigs

    def parameter_snapshot(self) -> dict:
        return {
            "method": self.method,
            "thresh": self.thresh,
            "connected_sigs": self.connected_sigs,
        }

    def _run_impl(self, request: ExpertRequest) -> ExpertRunResult:
        ensure_runtime_env(self.repo_root)
        add_optional_import_path(
            env_var="SIGAGENT_MUSICAL_PATH",
            repo_root=self.repo_root,
            default_local_dir="MuSiCal",
            package_name="musical",
        )
        try:
            from musical.refit import refit
        except Exception as exc:
            raise ImportError(
                "MuSiCal is not bundled with the minimal SigAgent release. "
                "Install MuSiCal according to its upstream instructions, or set "
                "SIGAGENT_MUSICAL_PATH to a local MuSiCal checkout before requesting "
                "the 'musical' expert."
            ) from exc

        exposures, model = refit(
            request.sample_matrix,
            request.signature_matrix,
            method=self.method,
            thresh=self.thresh,
            connected_sigs=self.connected_sigs,
        )
        diagnostics_by_sample = {}
        for sample_id, cosine_similarity in zip(request.sample_ids, getattr(model, "cos_similarities", [])):
            diagnostics_by_sample[sample_id] = {
                "core_method": "musical.refit.refit",
                "toolkit_cosine_similarity": float(cosine_similarity),
                "reduced_signature_count": int(exposures.loc[:, sample_id].gt(0).sum()),
                "method": self.method,
            }
        reduced_signatures = getattr(model, "sigs_reduced", [])
        if hasattr(reduced_signatures, "tolist"):
            reduced_signatures = reduced_signatures.tolist()

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
            artifacts={
                "implementation": "musical.refit.refit",
                "official_api_equivalent": True,
                "reduced_signatures": [str(v) for v in reduced_signatures],
            },
        )
