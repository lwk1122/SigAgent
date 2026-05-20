from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from .conformal_groups import build_group_key, fallback_group_keys


def _clip_probabilities(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)


@dataclass(slots=True)
class ProbabilityCalibrator:
    method: str
    temperature: float | None = None
    x_thresholds: list[float] | None = None
    y_thresholds: list[float] | None = None

    def transform(self, scores: np.ndarray | list[float]) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if values.size == 0:
            return values.astype(float)
        if self.method == "identity":
            return np.clip(values, 0.0, 1.0)
        if self.method == "temperature":
            if self.temperature is None:
                raise ValueError("Temperature calibrator is missing temperature.")
            clipped = _clip_probabilities(values)
            return expit(logit(clipped) / float(self.temperature))
        if self.method == "isotonic":
            if self.x_thresholds is None or self.y_thresholds is None:
                raise ValueError("Isotonic calibrator is missing thresholds.")
            calibrated = np.interp(
                values,
                np.asarray(self.x_thresholds, dtype=float),
                np.asarray(self.y_thresholds, dtype=float),
                left=float(self.y_thresholds[0]),
                right=float(self.y_thresholds[-1]),
            )
            return np.clip(calibrated, 0.0, 1.0)
        raise ValueError(f"Unsupported calibrator method: {self.method}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProbabilityCalibrator":
        return cls(
            method=str(payload["method"]),
            temperature=payload.get("temperature"),
            x_thresholds=payload.get("x_thresholds"),
            y_thresholds=payload.get("y_thresholds"),
        )


def fit_probability_calibrator(
    scores: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    *,
    method: str = "temperature",
) -> ProbabilityCalibrator:
    y_score = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    valid_mask = np.isfinite(y_score)
    y_score = y_score[valid_mask]
    y_true = y_true[valid_mask]
    if y_score.size == 0 or len(np.unique(y_true)) < 2:
        return ProbabilityCalibrator(method="identity")

    if method == "temperature":
        clipped = _clip_probabilities(y_score)
        logits = logit(clipped)

        def objective(temperature: float) -> float:
            calibrated = expit(logits / temperature)
            return float(
                -np.mean(
                    y_true * np.log(np.clip(calibrated, 1e-9, 1.0))
                    + (1 - y_true) * np.log(np.clip(1.0 - calibrated, 1e-9, 1.0))
                )
            )

        result = minimize_scalar(objective, bounds=(0.05, 20.0), method="bounded")
        return ProbabilityCalibrator(method="temperature", temperature=float(result.x))

    if method == "isotonic":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(y_score, y_true.astype(float))
        return ProbabilityCalibrator(
            method="isotonic",
            x_thresholds=[float(value) for value in model.X_thresholds_.tolist()],
            y_thresholds=[float(value) for value in model.y_thresholds_.tolist()],
        )

    raise ValueError(f"Unsupported calibration method: {method}")


@dataclass(slots=True)
class GroupedProbabilityCalibrator:
    group_columns: list[str]
    global_calibrator: ProbabilityCalibrator | None = None
    group_calibrators: dict[str, ProbabilityCalibrator] | None = None
    group_counts: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None

    def resolve(self, context: Mapping[str, Any] | None = None) -> tuple[ProbabilityCalibrator | None, str]:
        group_calibrators = self.group_calibrators or {}
        for key in fallback_group_keys(context, self.group_columns):
            if key == "global":
                break
            calibrator = group_calibrators.get(key)
            if calibrator is not None:
                return calibrator, key
        return self.global_calibrator, "global"

    def transform(self, scores: np.ndarray | list[float], *, context: Mapping[str, Any] | None = None) -> np.ndarray:
        calibrator, _ = self.resolve(context)
        if calibrator is None:
            return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        return calibrator.transform(scores)

    def transform_one(
        self,
        score: float,
        *,
        context: Mapping[str, Any] | None = None,
        return_source: bool = False,
    ) -> float | tuple[float, str]:
        calibrator, source = self.resolve(context)
        if calibrator is None:
            value = float(np.clip(score, 0.0, 1.0))
        else:
            value = float(calibrator.transform([score])[0])
        return (value, source) if return_source else value

    def transform_frame(
        self,
        score_frame: pd.DataFrame,
        *,
        contexts_by_row: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> pd.DataFrame:
        calibrated = score_frame.copy().astype(float)
        contexts = contexts_by_row or {}
        for row_name in calibrated.index:
            context = contexts.get(str(row_name))
            calibrated.loc[row_name, :] = self.transform(
                calibrated.loc[row_name, :].to_numpy(dtype=float),
                context=context,
            )
        return calibrated

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_columns": self.group_columns,
            "global_calibrator": None if self.global_calibrator is None else self.global_calibrator.to_dict(),
            "group_calibrators": {
                key: calibrator.to_dict()
                for key, calibrator in (self.group_calibrators or {}).items()
            },
            "group_counts": self.group_counts or {},
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroupedProbabilityCalibrator":
        return cls(
            group_columns=[str(value) for value in payload.get("group_columns", [])],
            global_calibrator=(
                None
                if payload.get("global_calibrator") is None
                else ProbabilityCalibrator.from_dict(payload["global_calibrator"])
            ),
            group_calibrators={
                str(key): ProbabilityCalibrator.from_dict(value)
                for key, value in (payload.get("group_calibrators") or {}).items()
            },
            group_counts={str(key): int(value) for key, value in (payload.get("group_counts") or {}).items()},
            metadata=payload.get("metadata") or {},
        )


def _split_conformal_quantile(errors: np.ndarray | list[float], *, alpha: float) -> float | None:
    values = np.sort(np.asarray(errors, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    rank = int(math.ceil((values.size + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), values.size - 1)
    return float(values[rank])


@dataclass(slots=True)
class GroupedConformalMargin:
    group_columns: list[str]
    alpha: float
    global_margin: float | None = None
    group_margins: dict[str, float] | None = None
    group_counts: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None

    def resolve(self, context: Mapping[str, Any] | None = None) -> tuple[float | None, str]:
        margins = self.group_margins or {}
        for key in fallback_group_keys(context, self.group_columns):
            if key == "global":
                break
            margin = margins.get(key)
            if margin is not None:
                return float(margin), key
        return self.global_margin, "global"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_columns": self.group_columns,
            "alpha": self.alpha,
            "global_margin": self.global_margin,
            "group_margins": self.group_margins or {},
            "group_counts": self.group_counts or {},
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroupedConformalMargin":
        return cls(
            group_columns=[str(value) for value in payload.get("group_columns", [])],
            alpha=float(payload["alpha"]),
            global_margin=payload.get("global_margin"),
            group_margins={
                str(key): float(value)
                for key, value in (payload.get("group_margins") or {}).items()
            },
            group_counts={str(key): int(value) for key, value in (payload.get("group_counts") or {}).items()},
            metadata=payload.get("metadata") or {},
        )


def fit_grouped_probability_calibrator(
    frame: pd.DataFrame,
    *,
    score_column: str,
    label_column: str,
    group_columns: Sequence[str],
    method: str,
    min_group_size: int = 50,
) -> GroupedProbabilityCalibrator:
    if frame.empty:
        return GroupedProbabilityCalibrator(group_columns=list(group_columns), global_calibrator=None)

    valid = frame.dropna(subset=[score_column, label_column]).copy()
    global_calibrator = fit_probability_calibrator(
        valid[score_column].to_numpy(dtype=float),
        valid[label_column].to_numpy(dtype=int),
        method=method,
    )
    group_calibrators: dict[str, ProbabilityCalibrator] = {}
    group_counts: dict[str, int] = {}
    for level in range(1, len(group_columns) + 1):
        columns = list(group_columns[:level])
        if any(column not in valid.columns for column in columns):
            continue
        for group_values, group_frame in valid.groupby(columns, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            if len(group_frame) < min_group_size or group_frame[label_column].nunique() < 2:
                continue
            context = {column: value for column, value in zip(columns, group_values)}
            key = build_group_key(context, columns)
            group_calibrators[key] = fit_probability_calibrator(
                group_frame[score_column].to_numpy(dtype=float),
                group_frame[label_column].to_numpy(dtype=int),
                method=method,
            )
            group_counts[key] = int(len(group_frame))
    return GroupedProbabilityCalibrator(
        group_columns=list(group_columns),
        global_calibrator=global_calibrator,
        group_calibrators=group_calibrators,
        group_counts=group_counts,
        metadata={
            "method": method,
            "min_group_size": min_group_size,
            "global_rows": int(len(valid)),
            "group_count": int(len(group_calibrators)),
        },
    )


def fit_grouped_conformal_margin(
    frame: pd.DataFrame,
    *,
    error_column: str,
    group_columns: Sequence[str],
    alpha: float,
    min_group_size: int = 100,
) -> GroupedConformalMargin:
    if frame.empty:
        return GroupedConformalMargin(group_columns=list(group_columns), alpha=alpha, global_margin=None)

    valid = frame.dropna(subset=[error_column]).copy()
    global_margin = _split_conformal_quantile(valid[error_column].to_numpy(dtype=float), alpha=alpha)
    group_margins: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    for level in range(1, len(group_columns) + 1):
        columns = list(group_columns[:level])
        if any(column not in valid.columns for column in columns):
            continue
        for group_values, group_frame in valid.groupby(columns, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            if len(group_frame) < min_group_size:
                continue
            margin = _split_conformal_quantile(group_frame[error_column].to_numpy(dtype=float), alpha=alpha)
            if margin is None:
                continue
            context = {column: value for column, value in zip(columns, group_values)}
            key = build_group_key(context, columns)
            group_margins[key] = float(margin)
            group_counts[key] = int(len(group_frame))
    return GroupedConformalMargin(
        group_columns=list(group_columns),
        alpha=alpha,
        global_margin=global_margin,
        group_margins=group_margins,
        group_counts=group_counts,
        metadata={
            "min_group_size": min_group_size,
            "global_rows": int(len(valid)),
            "group_count": int(len(group_margins)),
        },
    )


__all__ = [
    "GroupedConformalMargin",
    "GroupedProbabilityCalibrator",
    "ProbabilityCalibrator",
    "fit_grouped_conformal_margin",
    "fit_grouped_probability_calibrator",
    "fit_probability_calibrator",
]
