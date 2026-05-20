from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import types

from .base import BaseExpert, build_sample_results, ensure_runtime_env, make_temp_dir
from .optional_paths import add_optional_import_path, resolve_optional_package_dir
from .schema import ExpertRequest, ExpertRunResult


class SigProfilerAssignmentExpert(BaseExpert):
    expert_name = "sigprofiler_assignment"

    def __init__(
        self,
        *,
        connected_sigs: bool = True,
        add_background_signatures: bool = False,
        cpu: int = 1,
        repo_root=None,
    ) -> None:
        super().__init__(repo_root=repo_root)
        self.connected_sigs = connected_sigs
        self.add_background_signatures = add_background_signatures
        self.cpu = cpu

    def parameter_snapshot(self) -> dict:
        return {
            "connected_sigs": self.connected_sigs,
            "add_background_signatures": self.add_background_signatures,
            "cpu": self.cpu,
        }

    def _background_signature_indices(self, request: ExpertRequest) -> list[int]:
        if not self.add_background_signatures:
            return []
        if request.mutation_type != "SBS96":
            return []
        background_names = {"SBS1", "SBS5"}
        return [
            index
            for index, signature_name in enumerate(request.signature_names)
            if signature_name in background_names
        ]

    def _load_local_single_sample_module(self, package_dir: Path):
        package_module = sys.modules.get("SigProfilerAssignment")
        if package_module is None or not getattr(package_module, "__path__", None):
            package_module = types.ModuleType("SigProfilerAssignment")
            package_module.__path__ = [str(package_dir)]
            package_module.__file__ = str(package_dir / "__init__.py")
            package_module.__version__ = "local"
            sys.modules["SigProfilerAssignment"] = package_module

        submodule_name = "SigProfilerAssignment.decompose_subroutines"
        if submodule_name not in sys.modules:
            shim = types.ModuleType(submodule_name)

            def get_items_from_index(sequence, indices):
                sequence_list = list(sequence)
                return [sequence_list[index] for index in indices]

            def get_indeces(sequence, items):
                sequence_list = list(sequence)
                used = set()
                result = []
                for item in items:
                    for index, candidate in enumerate(sequence_list):
                        if index in used:
                            continue
                        if candidate == item:
                            result.append(index)
                            used.add(index)
                            break
                return result

            def make_letter_ids(idlenth, mtype="Signature "):
                return [f"{mtype}{index + 1}" for index in range(idlenth)]

            shim.get_items_from_index = get_items_from_index
            shim.get_indeces = get_indeces
            shim.make_letter_ids = make_letter_ids
            sys.modules[submodule_name] = shim
            setattr(package_module, "decompose_subroutines", shim)

        module_name = "SigProfilerAssignment.single_sample"
        module = sys.modules.get(module_name)
        if module is not None:
            return module

        spec = importlib.util.spec_from_file_location(module_name, package_dir / "single_sample.py")
        if spec is None or spec.loader is None:
            raise ImportError("Could not load SigProfilerAssignment.single_sample")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        setattr(package_module, "single_sample", module)
        return module

    def _load_single_sample_module(self):
        ensure_runtime_env(self.repo_root)
        add_optional_import_path(
            env_var="SIGAGENT_SIGPROFILERASSIGNMENT_PATH",
            repo_root=self.repo_root,
            default_local_dir="SigProfilerAssignment",
            package_name="SigProfilerAssignment",
        )
        package_dir = resolve_optional_package_dir(
            env_var="SIGAGENT_SIGPROFILERASSIGNMENT_PATH",
            repo_root=self.repo_root,
            default_local_dir="SigProfilerAssignment",
            package_name="SigProfilerAssignment",
        )
        if package_dir is not None and (package_dir / "single_sample.py").exists():
            return self._load_local_single_sample_module(package_dir)
        try:
            return importlib.import_module("SigProfilerAssignment.single_sample")
        except Exception as exc:
            raise ImportError(
                "SigProfilerAssignment is not bundled with the minimal SigAgent release. "
                "Install it according to the upstream instructions, or set "
                "SIGAGENT_SIGPROFILERASSIGNMENT_PATH to a local checkout before requesting "
                "the 'sigprofiler_assignment' expert."
            ) from exc

    def _run_impl(self, request: ExpertRequest) -> ExpertRunResult:
        single_sample = self._load_single_sample_module()
        temp_dir = make_temp_dir("spa_", self.repo_root)
        exposures_by_sample = {}
        diagnostics_by_sample = {}
        signature_matrix = request.signature_matrix.to_numpy(dtype=float)
        background_signature_indices = self._background_signature_indices(request)

        for sample_id in request.sample_ids:
            sample = request.sample_matrix.loc[:, sample_id].to_numpy(dtype=float)
            logfile = temp_dir / f"{sample_id}.log"
            selected_signature_indices, selected_activities, original_distance, cosine_similarity, kldiv, correlation, cosine_similarity_with_four_signatures = single_sample.add_remove_signatures(
                signature_matrix,
                sample,
                metric="l2",
                solver="nnls",
                background_sigs=background_signature_indices,
                permanent_sigs=[],
                candidate_sigs="all",
                add_penalty=0.05,
                remove_penalty=0.01,
                check_rule_negatives=[],
                checkrule_penalty=1.0,
                allsigids=request.signature_names,
                directory=str(logfile),
                connected_sigs=self.connected_sigs,
                verbose=False,
            )
            exposure_vector = [0.0] * len(request.signature_names)
            if len(selected_signature_indices) > 0:
                for signature_index, exposure in zip(selected_signature_indices, selected_activities):
                    exposure_vector[int(signature_index)] = float(exposure)

            if len(selected_signature_indices) == 0 or sum(exposure_vector) == 0.0:
                fitted_activities, fallback_distance = single_sample.fit_signatures(signature_matrix, sample, metric="l2")
                exposure_vector = [float(v) for v in fitted_activities]
                diagnostics_by_sample[sample_id] = {
                    "core_method": "single_sample.fit_signatures_fallback",
                    "logfile": str(logfile),
                    "toolkit_l2_error": float(fallback_distance),
                    "degenerate_add_remove_solution": True,
                    "background_signatures": [request.signature_names[index] for index in background_signature_indices],
                }
            else:
                diagnostics_by_sample[sample_id] = {
                    "core_method": "single_sample.add_remove_signatures",
                    "logfile": str(logfile),
                    "toolkit_l2_error": float(original_distance),
                    "toolkit_cosine_similarity": float(cosine_similarity),
                    "toolkit_kl_divergence": float(kldiv),
                    "toolkit_correlation": float(correlation),
                    "toolkit_cosine_similarity_with_four_signatures": float(cosine_similarity_with_four_signatures),
                    "background_signatures": [request.signature_names[index] for index in background_signature_indices],
                }
            exposures_by_sample[sample_id] = exposure_vector

        import pandas as pd

        exposures = pd.DataFrame(exposures_by_sample, index=request.signature_names)

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
                "temp_dir": str(temp_dir),
                "implementation": "single_sample.add_remove_signatures_core",
                "background_signatures": [request.signature_names[index] for index in background_signature_indices],
                "official_api_equivalent": False,
            },
            warnings=[
                "This wrapper uses the single_sample add/remove core rather than Analyzer.cosmic_fit; "
                "COSMIC catalog loading, subgroup exclusion, plotting, and multi-sample output generation are not invoked."
            ],
        )
