from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import math
import os
import tempfile
import time

import numpy as np
import pandas as pd

from .schema import ExpertRequest, ExpertRunResult, ExpertSampleResult


def ensure_runtime_env(repo_root: Path) -> Path:
    cache_root = repo_root / ".runtime_cache"
    mpl_cache = cache_root / "mpl"
    font_cache = cache_root / "fontconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("FONTCONFIG_PATH", str(font_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    return cache_root


def make_temp_dir(prefix: str, repo_root: Path) -> Path:
    scratch_root = repo_root / ".expert_runs"
    scratch_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=scratch_root))


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return float(np.dot(x, y) / (x_norm * y_norm))


def build_sample_results(
    *,
    request: ExpertRequest,
    exposures: pd.DataFrame,
    signature_scores: pd.DataFrame | None = None,
    signature_probabilities: pd.DataFrame | None = None,
    diagnostics_by_sample: dict[str, dict] | None = None,
    warnings_by_sample: dict[str, list[str]] | None = None,
) -> list[ExpertSampleResult]:
    aligned_exposures = exposures.reindex(index=request.signature_names, columns=request.sample_ids, fill_value=0.0)
    aligned_scores = None
    if signature_scores is not None:
        aligned_scores = signature_scores.reindex(index=request.sample_ids, columns=request.signature_names)
    aligned_probabilities = None
    if signature_probabilities is not None:
        aligned_probabilities = signature_probabilities.reindex(index=request.sample_ids, columns=request.signature_names)

    w = request.signature_matrix.loc[:, request.signature_names].to_numpy(dtype=float)
    sample_results: list[ExpertSampleResult] = []

    for sample_id in request.sample_ids:
        h = aligned_exposures.loc[:, sample_id].to_numpy(dtype=float)
        x = request.sample_matrix.loc[:, sample_id].to_numpy(dtype=float)
        reconstructed = w @ h
        residual = x - reconstructed
        l1_residual = float(np.linalg.norm(residual, ord=1))
        l2_residual = float(np.linalg.norm(residual, ord=2))
        rss = float(np.sum(np.square(residual)))
        total_mutations = float(np.sum(x))
        active_signatures = [
            signature_name
            for signature_name, value in aligned_exposures.loc[:, sample_id].sort_values(ascending=False).items()
            if float(value) > 0.0
        ]
        exposure_dict = {
            signature_name: float(value)
            for signature_name, value in aligned_exposures.loc[:, sample_id].items()
        }
        score_dict: dict[str, float | None] = {}
        if aligned_scores is not None:
            for signature_name, value in aligned_scores.loc[sample_id].items():
                if pd.isna(value):
                    score_dict[str(signature_name)] = None
                else:
                    score_dict[str(signature_name)] = float(value)
        probability_dict: dict[str, float | None] = {}
        if aligned_probabilities is not None:
            for signature_name, value in aligned_probabilities.loc[sample_id].items():
                if pd.isna(value):
                    probability_dict[str(signature_name)] = None
                else:
                    probability_dict[str(signature_name)] = float(value)

        metrics = {
            "mutation_count": total_mutations,
            "active_signature_count": float(len(active_signatures)),
            "reconstruction_cosine": _cosine_similarity(x, reconstructed),
            "rss": rss,
            "l1_residual": l1_residual,
            "l2_residual": l2_residual,
            "relative_l1_pct": float((l1_residual / total_mutations) * 100.0) if total_mutations > 0 else 0.0,
            "relative_l2_pct": float((l2_residual / (np.linalg.norm(x) + 1e-12)) * 100.0),
            "explained_fraction": float(1.0 - (rss / (float(np.sum(np.square(x))) + 1e-12))),
        }
        diagnostics = (diagnostics_by_sample or {}).get(sample_id, {})
        warnings = (warnings_by_sample or {}).get(sample_id, [])
        sample_results.append(
            ExpertSampleResult(
                sample_id=sample_id,
                active_signatures=active_signatures,
                exposures=exposure_dict,
                signature_scores=score_dict,
                signature_probabilities=probability_dict,
                reconstructed_counts=[float(v) for v in reconstructed.tolist()],
                residual_counts=[float(v) for v in residual.tolist()],
                metrics=metrics,
                diagnostics=diagnostics,
                warnings=warnings,
            )
        )
    return sample_results


class BaseExpert(ABC):
    expert_name: str

    def __init__(self, *, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()

    @abstractmethod
    def parameter_snapshot(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _run_impl(self, request: ExpertRequest) -> ExpertRunResult:
        raise NotImplementedError

    def run(self, request: ExpertRequest) -> ExpertRunResult:
        start = time.time()
        try:
            result = self._run_impl(request)
            result.runtime_seconds = time.time() - start
            return result
        except Exception as exc:
            return ExpertRunResult.failed(
                expert_name=self.expert_name,
                request=request,
                runtime_seconds=time.time() - start,
                parameters=self.parameter_snapshot(),
                error=str(exc),
            )
