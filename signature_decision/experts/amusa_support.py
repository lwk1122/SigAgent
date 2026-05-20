from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .base import ensure_runtime_env
from .optional_paths import add_optional_import_path
from .schema import ExpertRequest


@dataclass(slots=True)
class AMuSaSupportResult:
    sample_ids: list[str]
    full_signature_names: list[str]
    available_signature_names: list[str]
    missing_catalog_signatures: list[str]
    aligned_signature_matrix: pd.DataFrame
    probability_df: pd.DataFrame
    full_probability_df: pd.DataFrame
    threshold_series: pd.Series
    active_prediction_df: pd.DataFrame
    full_active_prediction_df: pd.DataFrame
    diagnostics_by_sample: dict[str, dict[str, Any]]
    artifacts: dict[str, Any]
    prediction_profile: str
    feature_names: list[str]


def run_amusa_support(
    *,
    request: ExpertRequest,
    repo_root: Path,
    max_active_signatures: int | None,
    prediction_profile: str | None = None,
) -> AMuSaSupportResult:
    ensure_runtime_env(repo_root)
    add_optional_import_path(
        env_var="SIGAGENT_AMUSA_PATH",
        repo_root=repo_root,
        default_local_dir="AMuSa",
        package_name="AMuSa",
    )
    try:
        from AMuSa.runtime import AMuSaRuntimeConfig, predict_support
    except Exception as exc:
        raise ImportError(
            "AMuSA is not bundled with the minimal SigAgent release. "
            "Provide it on PYTHONPATH or set SIGAGENT_AMUSA_PATH before "
            "requesting AMuSA-derived experts."
        ) from exc

    support = predict_support(
        sample_matrix=request.sample_matrix.loc[:, request.sample_ids],
        signature_matrix=request.signature_matrix.loc[:, request.signature_names],
        requested_signature_names=request.signature_names,
        mutation_type=request.mutation_type,
        repo_root=repo_root,
        config=AMuSaRuntimeConfig(
            mutation_type=request.mutation_type,
            prediction_profile=prediction_profile,
            max_active_signatures=max_active_signatures,
        ),
    )
    return AMuSaSupportResult(
        sample_ids=support.sample_ids,
        full_signature_names=support.full_signature_names,
        available_signature_names=support.available_signature_names,
        missing_catalog_signatures=support.missing_catalog_signatures,
        aligned_signature_matrix=support.aligned_signature_matrix,
        probability_df=support.probability_df,
        full_probability_df=support.full_probability_df,
        threshold_series=support.threshold_series,
        active_prediction_df=support.active_prediction_df,
        full_active_prediction_df=support.full_active_prediction_df,
        diagnostics_by_sample=support.diagnostics_by_sample,
        artifacts=support.artifacts,
        prediction_profile=support.profile.name,
        feature_names=support.feature_names,
    )


__all__ = [
    "AMuSaSupportResult",
    "run_amusa_support",
]
