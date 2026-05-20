from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .metrics import normalize_exposures


def simulate_counts_from_truth(
    signature_matrix: pd.DataFrame,
    truth_exposures: pd.DataFrame,
    *,
    burden: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    truth_norm = normalize_exposures(truth_exposures.reindex(index=signature_matrix.columns, fill_value=0.0))
    signature_values = signature_matrix.to_numpy(dtype=float)
    counts = np.zeros((signature_matrix.shape[0], truth_norm.shape[1]), dtype=int)

    for sample_index, sample_id in enumerate(truth_norm.columns):
        weights = truth_norm.loc[:, sample_id].to_numpy(dtype=float)
        profile = signature_values @ weights
        if np.sum(profile) == 0:
            profile = np.full(signature_matrix.shape[0], 1.0 / signature_matrix.shape[0], dtype=float)
        else:
            profile = profile / np.sum(profile)
        counts[:, sample_index] = rng.multinomial(int(burden), profile)
    return pd.DataFrame(counts, index=signature_matrix.index, columns=truth_norm.columns)


def scaled_truth_exposures(truth_exposures: pd.DataFrame, burden: int) -> pd.DataFrame:
    truth_norm = normalize_exposures(truth_exposures)
    return truth_norm * float(burden)


def subset_samples(truth_exposures: pd.DataFrame, sample_ids: Sequence[str]) -> pd.DataFrame:
    return truth_exposures.loc[:, list(sample_ids)].copy()


__all__ = [
    "scaled_truth_exposures",
    "simulate_counts_from_truth",
    "subset_samples",
]
