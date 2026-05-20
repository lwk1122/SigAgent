from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import normalize_exposures


def _numeric_matrix_from_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    numeric_columns = [
        column
        for column in frame.columns
        if pd.to_numeric(frame[column], errors="coerce").notna().all()
    ]
    if not numeric_columns:
        raise ValueError(f"No numeric signature columns found in {path}.")
    return frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise").astype(float)


def load_signature_matrix_for_design(path: str | Path) -> pd.DataFrame:
    matrix = _numeric_matrix_from_csv(path)
    matrix.index = [f"channel_{index:03d}" for index in range(matrix.shape[0])]
    return matrix


def _signature_entropy(values: np.ndarray) -> float:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    probabilities = values / total
    nonzero = probabilities[probabilities > 0.0]
    if nonzero.size <= 1:
        return 0.0
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return float(entropy / np.log(values.size))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def signature_property_frame(
    signature_matrix: pd.DataFrame,
    exposure_frame: pd.DataFrame,
    *,
    active_threshold: float = 0.0,
) -> pd.DataFrame:
    signature_names = [name for name in signature_matrix.columns if name in exposure_frame.index]
    if not signature_names:
        raise ValueError("Signature matrix and exposure frame do not share signature names.")
    signatures = signature_matrix.loc[:, signature_names].astype(float)
    exposures = exposure_frame.reindex(index=signature_names, fill_value=0.0).astype(float)
    normalized_exposures = normalize_exposures(exposures)

    rows: list[dict[str, Any]] = []
    signature_values = signatures.to_numpy(dtype=float)
    for index, signature_name in enumerate(signature_names):
        profile = signature_values[:, index]
        similarities = []
        for other_index, other_name in enumerate(signature_names):
            if other_index == index:
                continue
            similarities.append((other_name, _cosine(profile, signature_values[:, other_index])))
        similarities.sort(key=lambda item: item[1], reverse=True)
        nearest_name, nearest_cosine = similarities[0] if similarities else (None, 0.0)
        exposure_values = normalized_exposures.loc[signature_name]
        active_mask = exposure_values > active_threshold
        rows.append(
            {
                "signature_name": signature_name,
                "prevalence_count": int(active_mask.sum()),
                "prevalence_fraction": float(active_mask.mean()),
                "mean_exposure": float(exposure_values.mean()),
                "mean_active_exposure": float(exposure_values.loc[active_mask].mean()) if bool(active_mask.any()) else 0.0,
                "flatness_score": _signature_entropy(profile),
                "max_cosine_to_other": float(nearest_cosine),
                "nearest_signature": nearest_name,
                "benchmarkable_with_active_labels": bool(active_mask.any() and (~active_mask).any()),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    frame["prevalence_rank_desc"] = frame["prevalence_count"].rank(method="first", ascending=False).astype(int)
    frame["flatness_rank_desc"] = frame["flatness_score"].rank(method="first", ascending=False).astype(int)
    frame["similarity_rank_desc"] = frame["max_cosine_to_other"].rank(method="first", ascending=False).astype(int)
    return frame.sort_values(["prevalence_rank_desc", "signature_name"]).reset_index(drop=True)


@dataclass(slots=True)
class RemovalDesignConfig:
    n_per_group: int = 5
    active_threshold: float = 0.0
    include_unbenchmarkable_controls: bool = True


def _take_group(frame: pd.DataFrame, group_name: str, selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    output = selected.copy()
    output.insert(0, "selection_group", group_name)
    return output


def build_catalog_removal_design(
    signature_matrix: pd.DataFrame,
    exposure_frame: pd.DataFrame,
    *,
    mutation_type: str,
    config: RemovalDesignConfig | None = None,
) -> pd.DataFrame:
    config = config or RemovalDesignConfig()
    properties = signature_property_frame(
        signature_matrix,
        exposure_frame,
        active_threshold=config.active_threshold,
    )
    benchmarkable = properties.loc[properties["benchmarkable_with_active_labels"]].copy()
    active = benchmarkable.loc[benchmarkable["prevalence_count"] > 0].copy()
    groups = [
        _take_group(
            active,
            "high_prevalence_active",
            active.sort_values(["prevalence_count", "mean_active_exposure"], ascending=[False, False]).head(config.n_per_group),
        ),
        _take_group(
            active,
            "low_prevalence_active",
            active.sort_values(["prevalence_count", "mean_active_exposure"], ascending=[True, False]).head(config.n_per_group),
        ),
        _take_group(
            benchmarkable,
            "high_similarity",
            benchmarkable.sort_values("max_cosine_to_other", ascending=False).head(config.n_per_group),
        ),
        _take_group(
            benchmarkable,
            "flat_signature",
            benchmarkable.sort_values("flatness_score", ascending=False).head(config.n_per_group),
        ),
        _take_group(
            benchmarkable,
            "peaky_signature",
            benchmarkable.sort_values("flatness_score", ascending=True).head(config.n_per_group),
        ),
    ]
    if config.include_unbenchmarkable_controls:
        controls = properties.loc[~properties["benchmarkable_with_active_labels"]].copy()
        control_group_name = "inactive_or_unbenchmarkable_control"
        if controls.empty:
            controls = properties.sort_values(["prevalence_count", "mean_exposure"], ascending=[True, True]).head(config.n_per_group)
            control_group_name = "lowest_prevalence_control"
        groups.append(_take_group(controls, control_group_name, controls.head(config.n_per_group)))

    design = pd.concat([group for group in groups if not group.empty], ignore_index=True)
    if design.empty:
        return design
    design.insert(0, "mutation_type", mutation_type)
    design["active_threshold"] = float(config.active_threshold)
    design["n_per_group"] = int(config.n_per_group)
    return design


def build_catalog_removal_design_from_files(
    *,
    signature_source: str | Path,
    exposure_source: str | Path,
    mutation_type: str,
    config: RemovalDesignConfig | None = None,
) -> pd.DataFrame:
    signature_matrix = load_signature_matrix_for_design(signature_source)
    exposure_frame = pd.read_csv(exposure_source, index_col=0)
    return build_catalog_removal_design(
        signature_matrix,
        exposure_frame,
        mutation_type=mutation_type,
        config=config,
    )


def write_catalog_removal_design(design: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(path, sep="\t", index=False)
    return path


def load_catalog_removal_design(
    path: str | Path,
    *,
    benchmarkable_only: bool = False,
) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if benchmarkable_only and "benchmarkable_with_active_labels" in frame.columns:
        values = frame["benchmarkable_with_active_labels"]
        if values.dtype == bool:
            frame = frame.loc[values].copy()
        else:
            frame = frame.loc[values.astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    return frame


__all__ = [
    "RemovalDesignConfig",
    "build_catalog_removal_design",
    "build_catalog_removal_design_from_files",
    "load_catalog_removal_design",
    "load_signature_matrix_for_design",
    "signature_property_frame",
    "write_catalog_removal_design",
]
