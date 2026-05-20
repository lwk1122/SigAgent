from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def _top_signature_indices(signature_matrix: np.ndarray, sample: np.ndarray, top_k: int) -> list[int]:
    if top_k <= 0:
        return []
    scores = []
    for index in range(signature_matrix.shape[1]):
        signature = signature_matrix[:, index]
        denominator = float(np.linalg.norm(signature) * np.linalg.norm(sample))
        cosine = float(np.dot(signature, sample) / denominator) if denominator > 0 else 0.0
        scores.append((index, cosine))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [index for index, _ in scores[:top_k]]


def solve_nnls_matrix(
    *,
    sample_matrix: pd.DataFrame,
    signature_matrix: pd.DataFrame,
    signature_names: list[str] | None = None,
    candidate_signatures_by_sample: dict[str, list[str]] | None = None,
    fallback_top_k: int = 1,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    signature_names = signature_names or [str(value) for value in signature_matrix.columns.tolist()]
    aligned_signature_matrix = signature_matrix.loc[:, signature_names]
    exposures = pd.DataFrame(0.0, index=signature_names, columns=sample_matrix.columns)
    diagnostics_by_sample: dict[str, dict[str, Any]] = {}

    full_matrix = aligned_signature_matrix.to_numpy(dtype=float)
    for sample_id in sample_matrix.columns:
        sample = sample_matrix.loc[:, sample_id].to_numpy(dtype=float)
        candidate_names = list(dict.fromkeys(candidate_signatures_by_sample.get(sample_id, signature_names))) if candidate_signatures_by_sample else list(signature_names)
        candidate_names = [name for name in candidate_names if name in signature_names]
        fallback_used = False
        if not candidate_names:
            fallback_indices = _top_signature_indices(full_matrix, sample, fallback_top_k)
            candidate_names = [signature_names[index] for index in fallback_indices]
            fallback_used = True
        if not candidate_names:
            diagnostics_by_sample[str(sample_id)] = {
                "nnls_residual_norm": 0.0,
                "candidate_signature_count": 0,
                "candidate_signatures": [],
                "fallback_used": True,
            }
            continue

        candidate_matrix = aligned_signature_matrix.loc[:, candidate_names].to_numpy(dtype=float)
        coefficients, residual_norm = nnls(candidate_matrix, sample)
        exposures.loc[candidate_names, sample_id] = coefficients
        diagnostics_by_sample[str(sample_id)] = {
            "nnls_residual_norm": float(residual_norm),
            "candidate_signature_count": int(len(candidate_names)),
            "candidate_signatures": [str(name) for name in candidate_names],
            "fallback_used": fallback_used,
        }

    return exposures, diagnostics_by_sample


__all__ = [
    "solve_nnls_matrix",
]
