from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


AMUSA_GROUP_COLUMNS = ("mutation_type", "burden_group", "flatness_group")
ASSIGNMENT_GROUP_COLUMNS = ("mutation_type", "burden_group", "flatness_group", "risk_group")
EXPOSURE_GROUP_COLUMNS = ("mutation_type", "burden_group", "flatness_group", "risk_group")
CATALOG_ASSESSOR_GROUP_COLUMNS = ("mutation_type", "burden_group", "flatness_group", "disagreement_group")


@dataclass(frozen=True, slots=True)
class GroupContext:
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _normalize_group_value(value: Any) -> str:
    if _is_missing(value):
        return "missing"
    if isinstance(value, float):
        if math.isnan(value):
            return "missing"
        return f"{value:.6g}"
    return str(value)


def sample_profile_entropy(sample_counts: pd.Series | np.ndarray | list[float]) -> float:
    values = np.asarray(sample_counts, dtype=float)
    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    probabilities = values / total
    nonzero = probabilities[probabilities > 0.0]
    if nonzero.size <= 1:
        return 0.0
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    max_entropy = float(np.log(len(probabilities))) if len(probabilities) > 1 else 1.0
    if max_entropy <= 0.0:
        return 0.0
    return float(entropy / max_entropy)


def burden_group_from_count(mutation_count: float) -> str:
    if mutation_count < 200:
        return "lt200"
    if mutation_count < 500:
        return "200_499"
    if mutation_count < 2000:
        return "500_1999"
    return "ge2000"


def flatness_group_from_entropy(sample_entropy: float) -> str:
    if sample_entropy < 0.82:
        return "peaked"
    if sample_entropy < 0.90:
        return "mid"
    return "flat"


def disagreement_group_from_score(disagreement_score: float | None) -> str:
    if disagreement_score is None or not np.isfinite(disagreement_score):
        return "unknown"
    if disagreement_score < 0.25:
        return "low"
    if disagreement_score < 0.50:
        return "medium"
    return "high"


def risk_group_from_value(risk_value: float | str | None) -> str:
    if risk_value is None:
        return "unknown"
    if isinstance(risk_value, str):
        normalized = risk_value.strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        return normalized or "unknown"
    if not np.isfinite(float(risk_value)):
        return "unknown"
    if float(risk_value) >= 0.75:
        return "high"
    if float(risk_value) >= 0.55:
        return "medium"
    return "low"


def build_group_context(
    *,
    mutation_type: str,
    mutation_count: float,
    sample_entropy: float | None = None,
    disagreement_score: float | None = None,
    risk_level: str | None = None,
    risk_score: float | None = None,
) -> GroupContext:
    entropy = 0.0 if sample_entropy is None or not np.isfinite(sample_entropy) else float(sample_entropy)
    return GroupContext(
        values={
            "mutation_type": str(mutation_type),
            "mutation_count": float(mutation_count),
            "sample_entropy": entropy,
            "burden_group": burden_group_from_count(float(mutation_count)),
            "flatness_group": flatness_group_from_entropy(entropy),
            "disagreement_score": None if disagreement_score is None else float(disagreement_score),
            "disagreement_group": disagreement_group_from_score(disagreement_score),
            "risk_group": risk_group_from_value(risk_level if risk_level is not None else risk_score),
        }
    )


def group_context_from_sample_counts(
    sample_counts: pd.Series | np.ndarray | list[float],
    *,
    mutation_type: str,
    disagreement_score: float | None = None,
    risk_level: str | None = None,
    risk_score: float | None = None,
) -> GroupContext:
    values = np.asarray(sample_counts, dtype=float)
    mutation_count = float(np.sum(values))
    entropy = sample_profile_entropy(values)
    return build_group_context(
        mutation_type=mutation_type,
        mutation_count=mutation_count,
        sample_entropy=entropy,
        disagreement_score=disagreement_score,
        risk_level=risk_level,
        risk_score=risk_score,
    )


def annotate_group_columns(
    frame: pd.DataFrame,
    *,
    mutation_type: str | None = None,
    mutation_type_column: str = "mutation_type",
    burden_column: str = "burden",
    sample_entropy_column: str = "sample_entropy",
    disagreement_column: str = "disagreement_score",
    risk_level_column: str = "risk_level",
    risk_score_column: str = "risk_score",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    annotated = frame.copy()
    if mutation_type is not None and mutation_type_column not in annotated.columns:
        annotated[mutation_type_column] = str(mutation_type)
    if burden_column in annotated.columns and "burden_group" not in annotated.columns:
        annotated["burden_group"] = annotated[burden_column].astype(float).map(burden_group_from_count)
    if sample_entropy_column in annotated.columns and "flatness_group" not in annotated.columns:
        annotated["flatness_group"] = annotated[sample_entropy_column].astype(float).map(flatness_group_from_entropy)
    if disagreement_column in annotated.columns and "disagreement_group" not in annotated.columns:
        annotated["disagreement_group"] = annotated[disagreement_column].astype(float).map(disagreement_group_from_score)
    if "risk_group" not in annotated.columns:
        if risk_level_column in annotated.columns:
            annotated["risk_group"] = annotated[risk_level_column].map(risk_group_from_value)
        elif risk_score_column in annotated.columns:
            annotated["risk_group"] = annotated[risk_score_column].astype(float).map(risk_group_from_value)
    return annotated


def build_group_key(context: Mapping[str, Any] | GroupContext | None, columns: Sequence[str]) -> str:
    if not columns:
        return "global"
    values = context.values if isinstance(context, GroupContext) else (context or {})
    return "|".join(f"{column}={_normalize_group_value(values.get(column))}" for column in columns)


def fallback_group_keys(
    context: Mapping[str, Any] | GroupContext | None,
    columns: Sequence[str],
) -> list[str]:
    values = context.values if isinstance(context, GroupContext) else (context or {})
    available_columns: list[str] = []
    for column in columns:
        if _is_missing(values.get(column)):
            break
        available_columns.append(column)
    keys = [build_group_key(values, available_columns[:level]) for level in range(len(available_columns), 0, -1)]
    keys.append("global")
    seen: list[str] = []
    for key in keys:
        if key not in seen:
            seen.append(key)
    return seen


__all__ = [
    "AMUSA_GROUP_COLUMNS",
    "ASSIGNMENT_GROUP_COLUMNS",
    "CATALOG_ASSESSOR_GROUP_COLUMNS",
    "EXPOSURE_GROUP_COLUMNS",
    "GroupContext",
    "annotate_group_columns",
    "build_group_context",
    "build_group_key",
    "burden_group_from_count",
    "disagreement_group_from_score",
    "fallback_group_keys",
    "flatness_group_from_entropy",
    "group_context_from_sample_counts",
    "risk_group_from_value",
    "sample_profile_entropy",
]
