from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import warnings
import math

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.decomposition import NMF

from ..experience.schema import ExperienceRecord
from ..experts.base import _cosine_similarity
from ..experts.schema import json_ready
from .recurrence import cosine_similarity, residual_vector_from_record


@dataclass(slots=True)
class LocalExtractionConfig:
    max_components: int = 2
    min_records: int = 2
    min_mean_residual_mass: float = 0.05
    min_error_gain: float = 0.08
    min_component_weight_fraction: float = 0.10
    bootstrap_repeats: int = 10
    max_iter: int = 800
    random_seed: int = 0


@dataclass(slots=True)
class ExtractedResidualComponent:
    component_id: str
    profile: list[float]
    channel_ids: list[str] = field(default_factory=list)
    recurrence_count: int = 0
    stability_score: float | None = None
    mean_residual_mass: float | None = None
    catalog_match_name: str | None = None
    catalog_match_cosine: float | None = None
    catalog_top_hits: list[dict[str, Any]] = field(default_factory=list)
    fit_improvement: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "component_id": self.component_id,
                "profile": self.profile,
                "channel_ids": self.channel_ids,
                "recurrence_count": self.recurrence_count,
                "stability_score": self.stability_score,
                "mean_residual_mass": self.mean_residual_mass,
                "catalog_match_name": self.catalog_match_name,
                "catalog_match_cosine": self.catalog_match_cosine,
                "catalog_top_hits": self.catalog_top_hits,
                "fit_improvement": self.fit_improvement,
                "metadata": self.metadata,
            }
        )


@dataclass(slots=True)
class LocalExtractionResult:
    components: list[ExtractedResidualComponent]
    fit_improvement_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "components": [component.to_dict() for component in self.components],
                "fit_improvement_summary": self.fit_improvement_summary,
                "metadata": self.metadata,
            }
        )


def _counts_from_record(record: ExperienceRecord) -> np.ndarray | None:
    sample_result = record.fused_sample_result or {}
    reconstructed = np.asarray(sample_result.get("reconstructed_counts") or [], dtype=float)
    residual = np.asarray(sample_result.get("residual_counts") or [], dtype=float)
    if reconstructed.size == 0 and residual.size == 0:
        return None
    if reconstructed.size == 0:
        return np.maximum(residual, 0.0)
    if residual.size == 0:
        return np.maximum(reconstructed, 0.0)
    if reconstructed.shape != residual.shape:
        return None
    return np.maximum(reconstructed + residual, 0.0)


def _residual_counts_from_record(record: ExperienceRecord) -> np.ndarray | None:
    residual = np.asarray((record.fused_sample_result or {}).get("residual_counts") or [], dtype=float)
    if residual.size == 0:
        return None
    residual = np.maximum(residual, 0.0)
    if float(np.sum(residual)) <= 0.0:
        return None
    return residual


def _catalog_matches(
    profile: np.ndarray,
    signature_matrix: pd.DataFrame | None,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if signature_matrix is None or signature_matrix.empty:
        return []
    if signature_matrix.shape[0] != profile.shape[0]:
        return []
    normalized_profile = profile / max(float(np.linalg.norm(profile)), 1e-12)
    hits: list[dict[str, Any]] = []
    for signature_name in signature_matrix.columns:
        signature = signature_matrix.loc[:, signature_name].to_numpy(dtype=float)
        denominator = float(np.linalg.norm(signature) * np.linalg.norm(normalized_profile))
        score = 0.0 if denominator <= 0.0 else float(np.dot(signature, normalized_profile) / denominator)
        hits.append({"signature_name": str(signature_name), "cosine": score})
    hits.sort(key=lambda item: item["cosine"], reverse=True)
    return hits[:top_k]


def _nmf_model(n_components: int, *, config: LocalExtractionConfig, seed: int) -> NMF:
    return NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=seed,
        max_iter=config.max_iter,
        solver="cd",
        beta_loss="frobenius",
    )


def _fit_nmf(
    residual_matrix: np.ndarray,
    *,
    n_components: int,
    config: LocalExtractionConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    model = _nmf_model(n_components, config=config, seed=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        channel_loadings = model.fit_transform(residual_matrix)
    sample_weights = model.components_
    return channel_loadings, sample_weights, float(model.reconstruction_err_)


def _normalized_profiles(channel_loadings: np.ndarray) -> np.ndarray:
    profiles = np.asarray(channel_loadings, dtype=float).copy()
    for index in range(profiles.shape[1]):
        column = profiles[:, index]
        total = float(np.sum(column))
        if total > 0.0:
            profiles[:, index] = column / total
    return profiles


def _choose_component_count(
    residual_matrix: np.ndarray,
    *,
    config: LocalExtractionConfig,
) -> tuple[int, dict[str, Any]]:
    max_possible = int(min(config.max_components, residual_matrix.shape[0], residual_matrix.shape[1]))
    max_possible = max(1, max_possible)
    fit_stats: dict[int, dict[str, Any]] = {}
    for n_components in range(1, max_possible + 1):
        channel_loadings, sample_weights, error = _fit_nmf(
            residual_matrix,
            n_components=n_components,
            config=config,
            seed=config.random_seed + n_components,
        )
        weight_sums = np.sum(sample_weights, axis=1)
        weight_fractions = weight_sums / max(float(np.sum(weight_sums)), 1e-12)
        fit_stats[n_components] = {
            "channel_loadings": channel_loadings,
            "sample_weights": sample_weights,
            "error": error,
            "weight_fractions": weight_fractions,
        }
    chosen = 1
    for n_components in range(2, max_possible + 1):
        prev_error = float(fit_stats[n_components - 1]["error"])
        current_error = float(fit_stats[n_components]["error"])
        gain = (prev_error - current_error) / max(prev_error, 1e-12)
        weight_fractions = fit_stats[n_components]["weight_fractions"]
        if gain >= config.min_error_gain and float(np.min(weight_fractions)) >= config.min_component_weight_fraction:
            chosen = n_components
        else:
            break
    chosen_stats = fit_stats[chosen]
    return chosen, {
        "chosen_components": chosen,
        "candidate_errors": {str(k): float(v["error"]) for k, v in fit_stats.items()},
        "chosen_weight_fractions": [float(value) for value in chosen_stats["weight_fractions"].tolist()],
        "channel_loadings": chosen_stats["channel_loadings"],
        "sample_weights": chosen_stats["sample_weights"],
    }


def _bootstrap_stability(
    residual_matrix: np.ndarray,
    base_profiles: np.ndarray,
    *,
    n_components: int,
    config: LocalExtractionConfig,
) -> list[float]:
    if config.bootstrap_repeats <= 0 or residual_matrix.shape[1] < 2:
        return [math.nan for _ in range(n_components)]
    rng = np.random.default_rng(config.random_seed)
    scores_by_component: list[list[float]] = [[] for _ in range(n_components)]
    for repeat in range(config.bootstrap_repeats):
        column_indices = rng.integers(0, residual_matrix.shape[1], size=residual_matrix.shape[1])
        bootstrap_matrix = residual_matrix[:, column_indices]
        try:
            channel_loadings, _, _ = _fit_nmf(
                bootstrap_matrix,
                n_components=n_components,
                config=config,
                seed=config.random_seed + 100 + repeat,
            )
        except Exception:
            continue
        bootstrap_profiles = _normalized_profiles(channel_loadings)
        for component_index in range(n_components):
            reference = base_profiles[:, component_index]
            similarities = [
                cosine_similarity(reference, bootstrap_profiles[:, candidate_index])
                for candidate_index in range(n_components)
            ]
            if similarities:
                scores_by_component[component_index].append(float(max(similarities)))
    return [
        float(np.mean(component_scores)) if component_scores else float("nan")
        for component_scores in scores_by_component
    ]


def _build_component_profiles(
    residual_matrix: np.ndarray,
    records: list[ExperienceRecord],
    *,
    config: LocalExtractionConfig,
    channel_ids: Sequence[str] | None,
    signature_matrix: pd.DataFrame | None,
) -> tuple[list[ExtractedResidualComponent], dict[str, Any]]:
    chosen_components, chosen_fit = _choose_component_count(residual_matrix, config=config)
    channel_loadings = np.asarray(chosen_fit["channel_loadings"], dtype=float)
    sample_weights = np.asarray(chosen_fit["sample_weights"], dtype=float)
    profiles = _normalized_profiles(channel_loadings)
    residual_masses = []
    for record in records:
        residual = _residual_counts_from_record(record)
        mutation_count = float((record.input_summary or {}).get("mutation_count", 0.0))
        if residual is None or mutation_count <= 0.0:
            continue
        residual_masses.append(float(np.sum(residual)) / mutation_count)
    mean_residual_mass = float(np.mean(residual_masses)) if residual_masses else None
    stability_scores = _bootstrap_stability(
        residual_matrix,
        profiles,
        n_components=chosen_components,
        config=config,
    )
    components: list[ExtractedResidualComponent] = []
    for component_index in range(chosen_components):
        profile = profiles[:, component_index]
        top_hits = _catalog_matches(profile, signature_matrix)
        best_hit = top_hits[0] if top_hits else {}
        activation_count = int(np.sum(sample_weights[component_index, :] > 0.0))
        components.append(
            ExtractedResidualComponent(
                component_id=f"local_component_{component_index + 1}",
                profile=[float(value) for value in profile.tolist()],
                channel_ids=[str(value) for value in channel_ids] if channel_ids is not None else [],
                recurrence_count=activation_count,
                stability_score=None if np.isnan(stability_scores[component_index]) else float(stability_scores[component_index]),
                mean_residual_mass=mean_residual_mass,
                catalog_match_name=best_hit.get("signature_name"),
                catalog_match_cosine=best_hit.get("cosine"),
                catalog_top_hits=top_hits,
                metadata={
                    "mean_weight_fraction": float(chosen_fit["chosen_weight_fractions"][component_index]),
                },
            )
        )
    metadata = {
        "status": "success",
        "n_records": len(records),
        "chosen_components": chosen_components,
        "candidate_errors": chosen_fit["candidate_errors"],
        "mean_residual_mass": mean_residual_mass,
        "component_weight_fractions": chosen_fit["chosen_weight_fractions"],
    }
    return components, metadata


def _metrics_from_reconstruction(sample_counts: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    residual = sample_counts - reconstructed
    rss = float(np.sum(np.square(residual)))
    l1_residual = float(np.linalg.norm(residual, ord=1))
    l2_residual = float(np.linalg.norm(residual, ord=2))
    total_mutations = float(np.sum(sample_counts))
    return {
        "reconstruction_cosine": _cosine_similarity(sample_counts, reconstructed),
        "rss": rss,
        "l1_residual": l1_residual,
        "l2_residual": l2_residual,
        "relative_l1_pct": float((l1_residual / total_mutations) * 100.0) if total_mutations > 0 else 0.0,
        "relative_l2_pct": float((l2_residual / (np.linalg.norm(sample_counts) + 1e-12)) * 100.0),
        "explained_fraction": float(1.0 - (rss / (float(np.sum(np.square(sample_counts))) + 1e-12))),
    }


def _known_signature_names(record: ExperienceRecord, signature_matrix: pd.DataFrame) -> list[str]:
    active_names = list((record.fused_sample_result or {}).get("active_signatures") or [])
    return [name for name in active_names if name in signature_matrix.columns]


def _fit_with_matrix(sample_counts: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.zeros(0, dtype=float), np.zeros_like(sample_counts)
    coefficients, _ = nnls(matrix, sample_counts)
    reconstructed = matrix @ coefficients
    return coefficients, reconstructed


def compute_fit_improvement_summary(
    records: list[ExperienceRecord],
    *,
    signature_matrix: pd.DataFrame | None,
    components: Sequence[ExtractedResidualComponent],
) -> dict[str, Any]:
    if signature_matrix is None or signature_matrix.empty or not components:
        return {
            "status": "unavailable",
            "reason": "signature_matrix_or_components_missing",
            "aggregate": {},
            "per_sample": [],
        }
    component_profiles = np.column_stack(
        [np.asarray(component.profile, dtype=float) for component in components]
    )
    per_sample: list[dict[str, Any]] = []
    for record in records:
        sample_counts = _counts_from_record(record)
        if sample_counts is None or sample_counts.shape[0] != signature_matrix.shape[0]:
            continue
        baseline_current_metrics = dict((record.fused_sample_result or {}).get("metrics") or {})
        known_names = _known_signature_names(record, signature_matrix)
        known_matrix = (
            signature_matrix.loc[:, known_names].to_numpy(dtype=float)
            if known_names
            else np.zeros((signature_matrix.shape[0], 0), dtype=float)
        )
        known_coefficients, known_reconstructed = _fit_with_matrix(sample_counts, known_matrix)
        known_metrics = _metrics_from_reconstruction(sample_counts, known_reconstructed)
        augmented_matrix = np.column_stack([known_matrix, component_profiles]) if known_matrix.size else component_profiles
        augmented_coefficients, augmented_reconstructed = _fit_with_matrix(sample_counts, augmented_matrix)
        augmented_metrics = _metrics_from_reconstruction(sample_counts, augmented_reconstructed)
        component_weights = (
            augmented_coefficients[len(known_names) :].tolist()
            if augmented_coefficients.size > len(known_names)
            else []
        )
        per_sample.append(
            {
                "record_id": record.record_id,
                "sample_id": record.sample_id,
                "known_signature_count": len(known_names),
                "baseline_current_reconstruction_cosine": float(baseline_current_metrics.get("reconstruction_cosine", 0.0)),
                "baseline_current_relative_l1_pct": float(baseline_current_metrics.get("relative_l1_pct", 0.0)),
                "known_only_reconstruction_cosine": known_metrics["reconstruction_cosine"],
                "known_only_relative_l1_pct": known_metrics["relative_l1_pct"],
                "augmented_reconstruction_cosine": augmented_metrics["reconstruction_cosine"],
                "augmented_relative_l1_pct": augmented_metrics["relative_l1_pct"],
                "delta_reconstruction_cosine_vs_current": (
                    augmented_metrics["reconstruction_cosine"] - float(baseline_current_metrics.get("reconstruction_cosine", 0.0))
                ),
                "delta_reconstruction_cosine_vs_known_only": (
                    augmented_metrics["reconstruction_cosine"] - known_metrics["reconstruction_cosine"]
                ),
                "delta_relative_l1_pct_vs_current": (
                    float(baseline_current_metrics.get("relative_l1_pct", 0.0)) - augmented_metrics["relative_l1_pct"]
                ),
                "delta_relative_l1_pct_vs_known_only": (
                    known_metrics["relative_l1_pct"] - augmented_metrics["relative_l1_pct"]
                ),
                "component_weights": [float(value) for value in component_weights],
            }
        )
    if not per_sample:
        return {
            "status": "unavailable",
            "reason": "no_aligned_samples",
            "aggregate": {},
            "per_sample": [],
        }
    frame = pd.DataFrame.from_records(per_sample)
    aggregate = {
        "mean_delta_reconstruction_cosine_vs_current": float(frame["delta_reconstruction_cosine_vs_current"].mean()),
        "mean_delta_reconstruction_cosine_vs_known_only": float(frame["delta_reconstruction_cosine_vs_known_only"].mean()),
        "mean_delta_relative_l1_pct_vs_current": float(frame["delta_relative_l1_pct_vs_current"].mean()),
        "mean_delta_relative_l1_pct_vs_known_only": float(frame["delta_relative_l1_pct_vs_known_only"].mean()),
        "improved_fraction_vs_current": float((frame["delta_reconstruction_cosine_vs_current"] > 0.0).mean()),
        "improved_fraction_vs_known_only": float((frame["delta_reconstruction_cosine_vs_known_only"] > 0.0).mean()),
        "n_samples": int(len(frame)),
    }
    return {
        "status": "success",
        "aggregate": aggregate,
        "per_sample": per_sample,
    }


def extract_local_residual_components(
    records: list[ExperienceRecord],
    *,
    config: LocalExtractionConfig | None = None,
    channel_ids: Sequence[str] | None = None,
    signature_matrix: pd.DataFrame | None = None,
) -> LocalExtractionResult:
    config = config or LocalExtractionConfig()
    residual_columns: list[np.ndarray] = []
    residual_masses: list[float] = []
    aligned_records: list[ExperienceRecord] = []
    for record in records:
        residual = _residual_counts_from_record(record)
        mutation_count = float((record.input_summary or {}).get("mutation_count", 0.0))
        if residual is None or mutation_count <= 0.0:
            continue
        residual_columns.append(residual)
        residual_masses.append(float(np.sum(residual)) / mutation_count)
        aligned_records.append(record)
    if len(residual_columns) < config.min_records:
        return LocalExtractionResult(
            components=[],
            metadata={"status": "insufficient_records", "n_records": len(residual_columns)},
        )
    mean_residual_mass = float(np.mean(residual_masses)) if residual_masses else None
    if mean_residual_mass is not None and mean_residual_mass < config.min_mean_residual_mass:
        return LocalExtractionResult(
            components=[],
            metadata={
                "status": "insufficient_residual_mass",
                "n_records": len(residual_columns),
                "mean_residual_mass": mean_residual_mass,
            },
        )
    residual_matrix = np.column_stack(residual_columns)
    components, metadata = _build_component_profiles(
        residual_matrix,
        aligned_records,
        config=config,
        channel_ids=channel_ids,
        signature_matrix=signature_matrix,
    )
    fit_improvement_summary = compute_fit_improvement_summary(
        aligned_records,
        signature_matrix=signature_matrix,
        components=components,
    )
    aggregate_fit = fit_improvement_summary.get("aggregate") or {}
    for component in components:
        component.fit_improvement = {
            "mean_delta_reconstruction_cosine_vs_current": float(
                aggregate_fit.get("mean_delta_reconstruction_cosine_vs_current", 0.0)
            ),
            "mean_delta_reconstruction_cosine_vs_known_only": float(
                aggregate_fit.get("mean_delta_reconstruction_cosine_vs_known_only", 0.0)
            ),
            "mean_delta_relative_l1_pct_vs_current": float(
                aggregate_fit.get("mean_delta_relative_l1_pct_vs_current", 0.0)
            ),
            "mean_delta_relative_l1_pct_vs_known_only": float(
                aggregate_fit.get("mean_delta_relative_l1_pct_vs_known_only", 0.0)
            ),
        }
    return LocalExtractionResult(
        components=components,
        fit_improvement_summary=fit_improvement_summary,
        metadata=metadata,
    )


__all__ = [
    "ExtractedResidualComponent",
    "LocalExtractionConfig",
    "LocalExtractionResult",
    "compute_fit_improvement_summary",
    "extract_local_residual_components",
]
