from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .experts.schema import ExpertRunResult


def _ensure_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(float).fillna(0.0)


def align_exposure_frames(
    truth_exposures: pd.DataFrame,
    predicted_exposures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = [sample_id for sample_id in truth_exposures.columns if sample_id in predicted_exposures.columns]
    signature_names = sorted(set(truth_exposures.index).union(predicted_exposures.index))
    truth = _ensure_dataframe(truth_exposures.reindex(index=signature_names, columns=sample_ids, fill_value=0.0))
    predicted = _ensure_dataframe(predicted_exposures.reindex(index=signature_names, columns=sample_ids, fill_value=0.0))
    return truth, predicted


def normalize_exposures(exposures: pd.DataFrame) -> pd.DataFrame:
    exposures = _ensure_dataframe(exposures)
    column_sums = exposures.sum(axis=0).replace(0.0, np.nan)
    normalized = exposures.divide(column_sums, axis=1).fillna(0.0)
    return normalized


def exposure_frame_from_run(run: ExpertRunResult) -> pd.DataFrame:
    records = {
        sample_result.sample_id: sample_result.exposures
        for sample_result in run.sample_results
    }
    return pd.DataFrame(records).reindex(index=run.signature_names, fill_value=0.0)


def score_frame_from_run(run: ExpertRunResult) -> pd.DataFrame | None:
    if not run.sample_results:
        return None
    has_probabilities = any(bool(sample_result.signature_probabilities) for sample_result in run.sample_results)
    has_scores = any(bool(sample_result.signature_scores) for sample_result in run.sample_results)
    if not has_probabilities and not has_scores:
        return None
    if has_probabilities:
        records = {
            sample_result.sample_id: sample_result.signature_probabilities
            for sample_result in run.sample_results
        }
    else:
        records = {
            sample_result.sample_id: sample_result.signature_scores
            for sample_result in run.sample_results
        }
    return pd.DataFrame(records).reindex(index=run.signature_names).T.reindex(columns=run.signature_names)


def _active_mask(exposures: pd.DataFrame, threshold: float) -> pd.DataFrame:
    normalized = normalize_exposures(exposures)
    return normalized.gt(threshold)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if np.sum(y_true) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if np.sum(y_true) == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def active_set_metrics(
    truth_exposures: pd.DataFrame,
    predicted_exposures: pd.DataFrame,
    *,
    threshold: float = 0.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    truth, predicted = align_exposure_frames(truth_exposures, predicted_exposures)
    truth_mask = _active_mask(truth, threshold)
    predicted_mask = _active_mask(predicted, threshold)

    sample_records: list[dict[str, Any]] = []
    for sample_id in truth.columns:
        y_true = truth_mask.loc[:, sample_id].astype(int).to_numpy()
        y_pred = predicted_mask.loc[:, sample_id].astype(int).to_numpy()
        precision, recall, f1 = _binary_metrics(y_true, y_pred)
        union = float(np.sum((y_true == 1) | (y_pred == 1)))
        intersection = float(np.sum((y_true == 1) & (y_pred == 1)))
        jaccard = intersection / union if union > 0 else 1.0
        sample_records.append(
            {
                "sample_id": sample_id,
                "active_set_precision": precision,
                "active_set_recall": recall,
                "active_set_f1": f1,
                "active_set_jaccard": jaccard,
                "true_active_count": int(np.sum(y_true)),
                "pred_active_count": int(np.sum(y_pred)),
            }
        )

    flat_true = truth_mask.to_numpy(dtype=int).ravel()
    flat_pred = predicted_mask.to_numpy(dtype=int).ravel()
    signature_precision, signature_recall, signature_f1 = _binary_metrics(flat_true, flat_pred)
    aggregate = {
        "sample_precision": float(np.mean([record["active_set_precision"] for record in sample_records])),
        "sample_recall": float(np.mean([record["active_set_recall"] for record in sample_records])),
        "sample_f1": float(np.mean([record["active_set_f1"] for record in sample_records])),
        "sample_jaccard": float(np.mean([record["active_set_jaccard"] for record in sample_records])),
        "signature_precision": signature_precision,
        "signature_recall": signature_recall,
        "signature_f1": signature_f1,
    }
    return aggregate, pd.DataFrame.from_records(sample_records)


def exposure_error_metrics(
    truth_exposures: pd.DataFrame,
    predicted_exposures: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    truth, predicted = align_exposure_frames(truth_exposures, predicted_exposures)
    truth_norm = normalize_exposures(truth)
    predicted_norm = normalize_exposures(predicted)

    sample_records: list[dict[str, Any]] = []
    for sample_id in truth.columns:
        y_true = truth_norm.loc[:, sample_id].to_numpy(dtype=float)
        y_pred = predicted_norm.loc[:, sample_id].to_numpy(dtype=float)
        diff = y_true - y_pred
        tvd = 0.5 * float(np.sum(np.abs(diff)))
        mae = float(np.mean(np.abs(diff)))
        rmse = float(math.sqrt(np.mean(np.square(diff))))
        numerator = float(np.dot(y_true, y_pred))
        denominator = float(np.linalg.norm(y_true) * np.linalg.norm(y_pred))
        cosine = numerator / denominator if denominator > 0 else 0.0
        sample_records.append(
            {
                "sample_id": sample_id,
                "exposure_tvd": tvd,
                "exposure_mae": mae,
                "exposure_rmse": rmse,
                "exposure_cosine": cosine,
            }
        )
    aggregate = {
        "exposure_tvd": float(np.mean([record["exposure_tvd"] for record in sample_records])),
        "exposure_mae": float(np.mean([record["exposure_mae"] for record in sample_records])),
        "exposure_rmse": float(np.mean([record["exposure_rmse"] for record in sample_records])),
        "exposure_cosine": float(np.mean([record["exposure_cosine"] for record in sample_records])),
    }
    return aggregate, pd.DataFrame.from_records(sample_records)


def reconstruction_metrics(
    sample_matrix: pd.DataFrame,
    signature_matrix: pd.DataFrame,
    predicted_exposures: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    signature_names = [signature_name for signature_name in signature_matrix.columns if signature_name in predicted_exposures.index]
    predicted = _ensure_dataframe(predicted_exposures.reindex(index=signature_names, columns=sample_matrix.columns, fill_value=0.0))
    normalized_predicted = normalize_exposures(predicted)
    reconstructed = signature_matrix.loc[:, signature_names].to_numpy(dtype=float) @ normalized_predicted.to_numpy(dtype=float)
    reconstructed_df = pd.DataFrame(reconstructed, index=sample_matrix.index, columns=sample_matrix.columns)

    sample_profiles = sample_matrix.divide(sample_matrix.sum(axis=0).replace(0.0, np.nan), axis=1).fillna(0.0)
    sample_records: list[dict[str, Any]] = []
    for sample_id in sample_matrix.columns:
        x = sample_profiles.loc[:, sample_id].to_numpy(dtype=float)
        y = reconstructed_df.loc[:, sample_id].to_numpy(dtype=float)
        numerator = float(np.dot(x, y))
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        cosine = numerator / denominator if denominator > 0 else 0.0
        tvd = 0.5 * float(np.sum(np.abs(x - y)))
        sample_records.append(
            {
                "sample_id": sample_id,
                "reconstruction_cosine": cosine,
                "reconstruction_tvd": tvd,
            }
        )
    aggregate = {
        "reconstruction_cosine": float(np.mean([record["reconstruction_cosine"] for record in sample_records])),
        "reconstruction_tvd": float(np.mean([record["reconstruction_tvd"] for record in sample_records])),
    }
    return aggregate, pd.DataFrame.from_records(sample_records)


def exposure_interval_metrics(
    truth_exposures: pd.DataFrame,
    run: ExpertRunResult,
    *,
    active_threshold: float = 0.0,
    prefix: str = "exposure_interval",
    source_filter: str | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    if not run.sample_results:
        return {
            f"{prefix}_coverage": math.nan,
            f"{prefix}_active_coverage": math.nan,
            f"{prefix}_mean_width": math.nan,
            f"{prefix}_sample_count": 0.0,
        }, pd.DataFrame()

    sample_ids = [sample_result.sample_id for sample_result in run.sample_results]
    truth = _ensure_dataframe(truth_exposures.reindex(index=run.signature_names, columns=sample_ids, fill_value=0.0))
    truth_norm = normalize_exposures(truth)
    sample_records: list[dict[str, Any]] = []

    for sample_result in run.sample_results:
        diagnostics = sample_result.diagnostics or {}
        interval_source = diagnostics.get("exposure_interval_source")
        if source_filter is not None and interval_source != source_filter:
            continue
        raw_intervals = diagnostics.get("exposure_intervals") or {}
        widths: list[float] = []
        covered: list[float] = []
        active_covered: list[float] = []
        for signature_name in run.signature_names:
            interval = raw_intervals.get(signature_name)
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                continue
            lower = float(interval[0])
            upper = float(interval[1])
            if not np.isfinite(lower) or not np.isfinite(upper):
                continue
            truth_value = float(truth_norm.loc[signature_name, sample_result.sample_id])
            hit = float(lower <= truth_value <= upper)
            widths.append(max(0.0, upper - lower))
            covered.append(hit)
            if truth_value > active_threshold:
                active_covered.append(hit)
        if not widths:
            continue
        sample_records.append(
            {
                "sample_id": sample_result.sample_id,
                f"{prefix}_coverage": float(np.mean(covered)),
                f"{prefix}_active_coverage": float(np.mean(active_covered)) if active_covered else math.nan,
                f"{prefix}_mean_width": float(np.mean(widths)),
                "exposure_interval_source": interval_source,
            }
        )

    if not sample_records:
        return {
            f"{prefix}_coverage": math.nan,
            f"{prefix}_active_coverage": math.nan,
            f"{prefix}_mean_width": math.nan,
            f"{prefix}_sample_count": 0.0,
        }, pd.DataFrame()

    sample_frame = pd.DataFrame.from_records(sample_records)
    aggregate = {
        f"{prefix}_coverage": float(sample_frame[f"{prefix}_coverage"].mean()),
        f"{prefix}_active_coverage": float(sample_frame[f"{prefix}_active_coverage"].mean(skipna=True)),
        f"{prefix}_mean_width": float(sample_frame[f"{prefix}_mean_width"].mean()),
        f"{prefix}_sample_count": float(len(sample_frame)),
    }
    return aggregate, sample_frame


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(y_true[mask]))
        confidence = float(np.mean(y_prob[mask]))
        ece += abs(accuracy - confidence) * (np.sum(mask) / len(y_true))
    return float(ece)


def binary_probability_metrics(
    labels: pd.Series | np.ndarray | list[int],
    scores: pd.Series | np.ndarray | list[float],
    *,
    prefix: str,
    n_bins: int = 10,
) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    valid_mask = np.isfinite(scores_array)
    labels_array = labels_array[valid_mask]
    scores_array = scores_array[valid_mask]
    if scores_array.size == 0:
        return {
            f"{prefix}_brier": math.nan,
            f"{prefix}_ece": math.nan,
            f"{prefix}_auroc": math.nan,
            f"{prefix}_auprc": math.nan,
        }
    brier = float(np.mean(np.square(scores_array - labels_array)))
    ece = expected_calibration_error(labels_array, scores_array, n_bins=n_bins)
    try:
        auroc = float(roc_auc_score(labels_array, scores_array))
    except ValueError:
        auroc = math.nan
    try:
        auprc = float(average_precision_score(labels_array, scores_array))
    except ValueError:
        auprc = math.nan
    return {
        f"{prefix}_brier": brier,
        f"{prefix}_ece": ece,
        f"{prefix}_auroc": auroc,
        f"{prefix}_auprc": auprc,
    }


def groupwise_probability_metrics(
    frame: pd.DataFrame,
    *,
    label_column: str,
    score_column: str,
    group_columns: list[str],
    prefix: str,
    n_bins: int = 10,
    min_group_size: int = 10,
) -> pd.DataFrame:
    if frame.empty or label_column not in frame.columns or score_column not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    valid = frame.dropna(subset=[label_column, score_column]).copy()
    for group_column in group_columns:
        if group_column not in valid.columns:
            continue
        for group_value, group_frame in valid.groupby(group_column, dropna=False):
            if len(group_frame) < min_group_size:
                continue
            metrics = binary_probability_metrics(
                group_frame[label_column],
                group_frame[score_column],
                prefix=prefix,
                n_bins=n_bins,
            )
            rows.append(
                {
                    "group_dimension": group_column,
                    "group_value": str(group_value),
                    "n_samples": int(len(group_frame)),
                    **metrics,
                }
            )
    return pd.DataFrame.from_records(rows)


def groupwise_interval_metrics(
    frame: pd.DataFrame,
    *,
    coverage_column: str,
    width_column: str,
    group_columns: list[str],
    prefix: str,
    target_coverage: float,
    active_coverage_column: str | None = None,
    min_group_size: int = 10,
) -> pd.DataFrame:
    if frame.empty or coverage_column not in frame.columns or width_column not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    valid = frame.dropna(subset=[coverage_column, width_column]).copy()
    for group_column in group_columns:
        if group_column not in valid.columns:
            continue
        for group_value, group_frame in valid.groupby(group_column, dropna=False):
            if len(group_frame) < min_group_size:
                continue
            mean_coverage = float(group_frame[coverage_column].mean())
            record = {
                "group_dimension": group_column,
                "group_value": str(group_value),
                "n_samples": int(len(group_frame)),
                f"{prefix}_coverage": mean_coverage,
                f"{prefix}_coverage_gap": float(mean_coverage - target_coverage),
                f"{prefix}_mean_width": float(group_frame[width_column].mean()),
            }
            if active_coverage_column is not None and active_coverage_column in group_frame.columns:
                record[f"{prefix}_active_coverage"] = float(group_frame[active_coverage_column].mean(skipna=True))
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def summarize_group_metric_frame(
    frame: pd.DataFrame,
    *,
    ece_column: str | None = None,
    coverage_gap_column: str | None = None,
    width_column: str | None = None,
    prefix: str,
) -> dict[str, float]:
    if frame.empty:
        summary = {}
        if ece_column is not None:
            summary[f"{prefix}_group_ece_max"] = math.nan
            summary[f"{prefix}_group_ece_mean"] = math.nan
        if coverage_gap_column is not None:
            summary[f"{prefix}_group_coverage_gap_max_abs"] = math.nan
            summary[f"{prefix}_group_coverage_gap_mean_abs"] = math.nan
        if width_column is not None:
            summary[f"{prefix}_group_mean_width_mean"] = math.nan
        return summary
    summary: dict[str, float] = {}
    if ece_column is not None and ece_column in frame.columns:
        ece_values = frame[ece_column].astype(float)
        summary[f"{prefix}_group_ece_max"] = float(ece_values.max())
        summary[f"{prefix}_group_ece_mean"] = float(ece_values.mean())
    if coverage_gap_column is not None and coverage_gap_column in frame.columns:
        gap_values = frame[coverage_gap_column].astype(float).abs()
        summary[f"{prefix}_group_coverage_gap_max_abs"] = float(gap_values.max())
        summary[f"{prefix}_group_coverage_gap_mean_abs"] = float(gap_values.mean())
    if width_column is not None and width_column in frame.columns:
        summary[f"{prefix}_group_mean_width_mean"] = float(frame[width_column].astype(float).mean())
    return summary


def calibration_metrics(
    truth_exposures: pd.DataFrame,
    score_frame: pd.DataFrame | None,
    *,
    threshold: float = 0.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    if score_frame is None or score_frame.empty:
        return {
            "calibration_brier": math.nan,
            "calibration_ece": math.nan,
            "calibration_auroc": math.nan,
            "calibration_auprc": math.nan,
        }, pd.DataFrame()

    truth = _ensure_dataframe(truth_exposures.reindex(columns=score_frame.index, fill_value=0.0))
    truth_mask = _active_mask(truth, threshold)
    aligned_scores = score_frame.reindex(index=truth.columns, columns=truth.index)

    flat_true = truth_mask.T.to_numpy(dtype=int).ravel()
    flat_scores = aligned_scores.to_numpy(dtype=float).ravel()
    valid_mask = np.isfinite(flat_scores)
    flat_true = flat_true[valid_mask]
    flat_scores = flat_scores[valid_mask]
    if len(flat_scores) == 0:
        return {
            "calibration_brier": math.nan,
            "calibration_ece": math.nan,
            "calibration_auroc": math.nan,
            "calibration_auprc": math.nan,
        }, pd.DataFrame()

    brier = float(np.mean(np.square(flat_scores - flat_true)))
    ece = expected_calibration_error(flat_true, flat_scores)
    try:
        auroc = float(roc_auc_score(flat_true, flat_scores))
    except ValueError:
        auroc = math.nan
    try:
        auprc = float(average_precision_score(flat_true, flat_scores))
    except ValueError:
        auprc = math.nan
    return {
        "calibration_brier": brier,
        "calibration_ece": ece,
        "calibration_auroc": auroc,
        "calibration_auprc": auprc,
    }, pd.DataFrame(
        {
            "truth_active": flat_true,
            "predicted_score": flat_scores,
        }
    )


def catalog_insufficiency_metrics(labels: pd.Series, scores: pd.Series) -> dict[str, float]:
    aligned = pd.concat([labels.rename("label"), scores.rename("score")], axis=1).dropna()
    if aligned.empty:
        return {
            "catalog_insufficiency_auroc": math.nan,
            "catalog_insufficiency_auprc": math.nan,
        }
    try:
        auroc = float(roc_auc_score(aligned["label"], aligned["score"]))
    except ValueError:
        auroc = math.nan
    try:
        auprc = float(average_precision_score(aligned["label"], aligned["score"]))
    except ValueError:
        auprc = math.nan
    return {
        "catalog_insufficiency_auroc": auroc,
        "catalog_insufficiency_auprc": auprc,
    }


def evaluate_expert_run(
    run: ExpertRunResult,
    *,
    sample_matrix: pd.DataFrame,
    signature_matrix: pd.DataFrame,
    truth_exposures: pd.DataFrame,
    active_threshold: float = 0.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    if run.status != "success" or not run.sample_results:
        return (
            {
                "expert_name": run.expert_name,
                "status": run.status,
                "sample_precision": math.nan,
                "sample_recall": math.nan,
                "sample_f1": math.nan,
                "sample_jaccard": math.nan,
                "signature_precision": math.nan,
                "signature_recall": math.nan,
                "signature_f1": math.nan,
                "exposure_tvd": math.nan,
                "exposure_mae": math.nan,
                "exposure_rmse": math.nan,
                "exposure_cosine": math.nan,
                "reconstruction_cosine": math.nan,
                "reconstruction_tvd": math.nan,
                "calibration_brier": math.nan,
                "calibration_ece": math.nan,
                "calibration_auroc": math.nan,
                "calibration_auprc": math.nan,
                "error": run.error,
            },
            pd.DataFrame(),
        )

    predicted_exposures = exposure_frame_from_run(run)
    score_frame = score_frame_from_run(run)

    active_aggregate, active_per_sample = active_set_metrics(
        truth_exposures,
        predicted_exposures,
        threshold=active_threshold,
    )
    exposure_aggregate, exposure_per_sample = exposure_error_metrics(
        truth_exposures,
        predicted_exposures,
    )
    reconstruction_aggregate, reconstruction_per_sample = reconstruction_metrics(
        sample_matrix,
        signature_matrix,
        predicted_exposures,
    )
    calibration_aggregate, _ = calibration_metrics(
        truth_exposures,
        score_frame,
        threshold=active_threshold,
    )

    sample_metrics = active_per_sample.merge(exposure_per_sample, on="sample_id", how="outer")
    sample_metrics = sample_metrics.merge(reconstruction_per_sample, on="sample_id", how="outer")

    aggregate = {
        "expert_name": run.expert_name,
        "status": run.status,
        **active_aggregate,
        **exposure_aggregate,
        **reconstruction_aggregate,
        **calibration_aggregate,
    }
    return aggregate, sample_metrics


__all__ = [
    "active_set_metrics",
    "align_exposure_frames",
    "binary_probability_metrics",
    "calibration_metrics",
    "catalog_insufficiency_metrics",
    "evaluate_expert_run",
    "expected_calibration_error",
    "exposure_error_metrics",
    "exposure_frame_from_run",
    "normalize_exposures",
    "reconstruction_metrics",
    "score_frame_from_run",
]
