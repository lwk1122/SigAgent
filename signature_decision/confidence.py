from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import train_test_split

from .conformal_groups import (
    AMUSA_GROUP_COLUMNS,
    ASSIGNMENT_GROUP_COLUMNS,
    EXPOSURE_GROUP_COLUMNS,
    annotate_group_columns,
    group_context_from_sample_counts,
)
from .experts.io import load_expert_request
from .experts.registry import ExpertRegistry, build_default_registry
from .group_calibration import (
    GroupedConformalMargin,
    GroupedProbabilityCalibrator,
    ProbabilityCalibrator,
    fit_grouped_conformal_margin,
    fit_grouped_probability_calibrator,
    fit_probability_calibrator,
)
from .metrics import active_set_metrics, normalize_exposures
from .simulation import scaled_truth_exposures, simulate_counts_from_truth, subset_samples

@dataclass(slots=True)
class ConfidenceArtifacts:
    amusa_probability_calibrator: ProbabilityCalibrator | None = None
    amusa_group_calibrator: GroupedProbabilityCalibrator | None = None
    final_assignment_calibrator: ProbabilityCalibrator | None = None
    final_assignment_group_calibrator: GroupedProbabilityCalibrator | None = None
    exposure_conformal_margin: float | None = None
    exposure_group_conformal: GroupedConformalMargin | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "amusa_probability_calibrator": None if self.amusa_probability_calibrator is None else self.amusa_probability_calibrator.to_dict(),
            "amusa_group_calibrator": None if self.amusa_group_calibrator is None else self.amusa_group_calibrator.to_dict(),
            "final_assignment_calibrator": None if self.final_assignment_calibrator is None else self.final_assignment_calibrator.to_dict(),
            "final_assignment_group_calibrator": (
                None if self.final_assignment_group_calibrator is None else self.final_assignment_group_calibrator.to_dict()
            ),
            "exposure_conformal_margin": self.exposure_conformal_margin,
            "exposure_group_conformal": None if self.exposure_group_conformal is None else self.exposure_group_conformal.to_dict(),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConfidenceArtifacts":
        return cls(
            amusa_probability_calibrator=(
                None
                if payload.get("amusa_probability_calibrator") is None
                else ProbabilityCalibrator.from_dict(payload["amusa_probability_calibrator"])
            ),
            amusa_group_calibrator=(
                None
                if payload.get("amusa_group_calibrator") is None
                else GroupedProbabilityCalibrator.from_dict(payload["amusa_group_calibrator"])
            ),
            final_assignment_calibrator=(
                None
                if payload.get("final_assignment_calibrator") is None
                else ProbabilityCalibrator.from_dict(payload["final_assignment_calibrator"])
            ),
            final_assignment_group_calibrator=(
                None
                if payload.get("final_assignment_group_calibrator") is None
                else GroupedProbabilityCalibrator.from_dict(payload["final_assignment_group_calibrator"])
            ),
            exposure_conformal_margin=payload.get("exposure_conformal_margin"),
            exposure_group_conformal=(
                None
                if payload.get("exposure_group_conformal") is None
                else GroupedConformalMargin.from_dict(payload["exposure_group_conformal"])
            ),
            metadata=payload.get("metadata") or {},
        )

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceArtifacts":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(slots=True)
class BootstrapConfig:
    n_replicates: int = 100
    alpha: float = 0.1
    random_seed: int = 0
    use_conformal: bool = True


def bootstrap_exposure_intervals(
    *,
    sample_counts: pd.Series,
    signature_matrix: pd.DataFrame,
    selected_signature_names: list[str],
    config: BootstrapConfig,
    conformal_margin: float | None = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    if config.n_replicates <= 0 or not selected_signature_names:
        return {}, {"interval_source": "none"}

    mutation_count = int(float(sample_counts.sum()))
    if mutation_count <= 0:
        return {signature_name: (0.0, 0.0) for signature_name in selected_signature_names}, {"interval_source": "bootstrap"}

    selected_matrix = signature_matrix.loc[:, selected_signature_names].to_numpy(dtype=float)
    profile = sample_counts.to_numpy(dtype=float)
    profile = profile / np.sum(profile)
    rng = np.random.default_rng(config.random_seed)
    bootstrap_exposures = np.zeros((config.n_replicates, len(selected_signature_names)), dtype=float)

    for replicate_index in range(config.n_replicates):
        counts = rng.multinomial(mutation_count, profile)
        coefficients, _ = nnls(selected_matrix, counts.astype(float))
        total = float(np.sum(coefficients))
        if total > 0.0:
            bootstrap_exposures[replicate_index, :] = coefficients / total

    lower_quantile = float(config.alpha / 2.0)
    upper_quantile = float(1.0 - config.alpha / 2.0)
    margin = float(conformal_margin or 0.0) if config.use_conformal else 0.0
    intervals = {}
    for signature_index, signature_name in enumerate(selected_signature_names):
        column = bootstrap_exposures[:, signature_index]
        lower = max(0.0, float(np.quantile(column, lower_quantile)) - margin)
        upper = min(1.0, float(np.quantile(column, upper_quantile)) + margin)
        intervals[str(signature_name)] = (lower, upper)
    return intervals, {
        "interval_source": "bootstrap_conformal" if margin > 0.0 else "bootstrap",
        "bootstrap_replicates": config.n_replicates,
        "bootstrap_alpha": config.alpha,
        "conformal_margin": margin,
    }


@dataclass(slots=True)
class KnownCatalogConfidenceTables:
    amusa_rows: pd.DataFrame
    assignment_rows: pd.DataFrame
    exposure_error_rows: pd.DataFrame


def _build_holdout_split(
    split_frame: pd.DataFrame,
    *,
    split_id_column: str = "split_id",
    label_column: str = "label",
    calibration_fraction: float = 0.25,
    random_seed: int = 0,
) -> tuple[set[str], set[str], dict[str, Any]]:
    if split_frame.empty:
        return set(), set(), {
            "strategy": "empty",
            "calibration_fraction": calibration_fraction,
            "train_count": 0,
            "calibration_count": 0,
        }

    split_manifest = (
        split_frame.loc[:, [split_id_column, label_column]]
        .dropna(subset=[split_id_column])
        .drop_duplicates(subset=[split_id_column])
        .copy()
    )
    split_manifest[split_id_column] = split_manifest[split_id_column].astype(str)
    split_manifest[label_column] = split_manifest[label_column].astype(int)
    if split_manifest.empty:
        return set(), set(), {
            "strategy": "empty",
            "calibration_fraction": calibration_fraction,
            "train_count": 0,
            "calibration_count": 0,
        }

    unique_ids = split_manifest[split_id_column].tolist()
    if calibration_fraction <= 0.0 or len(unique_ids) < 4:
        return set(unique_ids), set(), {
            "strategy": "disabled_or_too_small",
            "calibration_fraction": calibration_fraction,
            "train_count": len(unique_ids),
            "calibration_count": 0,
        }

    labels = split_manifest[label_column].to_numpy(dtype=int)
    class_counts = split_manifest[label_column].value_counts()
    can_stratify = len(class_counts) >= 2 and int(class_counts.min()) >= 2
    try:
        train_ids, calibration_ids = train_test_split(
            unique_ids,
            test_size=float(calibration_fraction),
            random_state=random_seed,
            stratify=labels if can_stratify else None,
        )
        strategy = "stratified_holdout" if can_stratify else "random_holdout"
    except ValueError:
        rng = np.random.default_rng(random_seed)
        shuffled = unique_ids[:]
        rng.shuffle(shuffled)
        calibration_count = min(max(1, int(round(len(shuffled) * calibration_fraction))), len(shuffled) - 1)
        calibration_ids = shuffled[:calibration_count]
        train_ids = shuffled[calibration_count:]
        strategy = "fallback_random_holdout"

    return set(train_ids), set(calibration_ids), {
        "strategy": strategy,
        "calibration_fraction": calibration_fraction,
        "train_count": len(train_ids),
        "calibration_count": len(calibration_ids),
    }


def collect_known_catalog_confidence_tables(
    *,
    sample_source: str | Path,
    signature_source: str | Path,
    exposure_source: str | Path,
    mutation_type: str,
    burdens: tuple[int, ...] = (100, 200, 500, 2000, 50000),
    max_samples_per_burden: int = 100,
    random_seed: int = 0,
    registry: ExpertRegistry | None = None,
    expert_names: list[str] | None = None,
    active_threshold: float = 0.0,
    assignment_f1_threshold: float = 0.8,
) -> KnownCatalogConfidenceTables:
    registry = registry or build_default_registry(".")
    from .benchmark import load_truth_exposures
    from .fusion import fuse_expert_runs

    base_request = load_expert_request(
        sample_source=sample_source,
        signature_source=signature_source,
        mutation_type=mutation_type,
    )
    truth = load_truth_exposures(
        exposure_source,
        mutation_type=base_request.mutation_type,
        sample_ids=base_request.sample_ids,
        signature_names=base_request.signature_names,
    )
    rng = np.random.default_rng(random_seed)
    amusa_records: list[dict[str, Any]] = []
    assignment_records: list[dict[str, Any]] = []
    exposure_error_records: list[dict[str, Any]] = []
    available_sample_ids = base_request.sample_ids

    for burden in burdens:
        chosen_sample_ids = available_sample_ids
        if max_samples_per_burden and len(available_sample_ids) > max_samples_per_burden:
            chosen_sample_ids = sorted(rng.choice(available_sample_ids, size=max_samples_per_burden, replace=False).tolist())
        truth_subset = subset_samples(truth.exposures, chosen_sample_ids)
        simulated_samples = simulate_counts_from_truth(
            base_request.signature_matrix,
            truth_subset,
            burden=burden,
            rng=rng,
        )
        scaled_truth = scaled_truth_exposures(truth_subset, burden)
        request = base_request.with_samples(chosen_sample_ids)
        request = type(request)(
            mutation_type=request.mutation_type,
            sample_matrix=simulated_samples,
            signature_matrix=request.signature_matrix.copy(),
            channel_metadata=request.channel_metadata,
            sample_source=request.sample_source,
            signature_source=request.signature_source,
            reference_name=request.reference_name,
            request_id=f"confidence_known_{mutation_type}_{burden}",
            alignment_strategy=request.alignment_strategy,
        )
        runs = registry.run_all(request, expert_names)
        fusion_output = fuse_expert_runs(runs, request)
        report_by_sample = {report.sample_id: report for report in fusion_output.reports}
        sample_group_contexts: dict[str, dict[str, Any]] = {}
        for sample_id in chosen_sample_ids:
            report = report_by_sample[sample_id]
            agreement_score = float(report.metadata.get("agreement_score", 0.0))
            sample_group_contexts[sample_id] = group_context_from_sample_counts(
                simulated_samples.loc[:, sample_id],
                mutation_type=request.mutation_type,
                disagreement_score=1.0 - agreement_score,
                risk_level=report.catalog_insufficiency_level,
            ).to_dict()

        amusa_run = next((run for run in runs if run.expert_name == "amusa"), None)
        if amusa_run is not None:
            truth_norm = normalize_exposures(scaled_truth)
            for sample_result in amusa_run.sample_results:
                sample_truth = truth_norm.loc[:, sample_result.sample_id]
                split_id = f"{sample_result.sample_id}::{burden}"
                sample_context = sample_group_contexts.get(sample_result.sample_id, {})
                for signature_name, raw_score in sample_result.signature_scores.items():
                    if raw_score is None:
                        continue
                    amusa_records.append(
                        {
                            "sample_id": sample_result.sample_id,
                            "split_id": split_id,
                            "signature_name": signature_name,
                            "burden": burden,
                            "raw_score": float(raw_score),
                            "label": int(float(sample_truth.get(signature_name, 0.0)) > active_threshold),
                            **sample_context,
                        }
                    )

        fused_exposures = pd.DataFrame(
            {
                sample_result.sample_id: sample_result.exposures
                for sample_result in fusion_output.fused_run.sample_results
            }
        ).reindex(index=request.signature_names, fill_value=0.0)
        active_aggregate, active_per_sample = active_set_metrics(scaled_truth, fused_exposures, threshold=active_threshold)
        _ = active_aggregate
        truth_norm = normalize_exposures(scaled_truth)
        pred_norm = normalize_exposures(fused_exposures)
        for _, row in active_per_sample.iterrows():
            sample_id = str(row["sample_id"])
            split_id = f"{sample_id}::{burden}"
            report = report_by_sample[sample_id]
            sample_context = sample_group_contexts.get(sample_id, {})
            assignment_records.append(
                {
                    "sample_id": sample_id,
                    "split_id": split_id,
                    "burden": burden,
                    "raw_score": float(report.assignment_confidence_raw_score or 0.0),
                    "label": int(float(row["active_set_f1"]) >= assignment_f1_threshold),
                    **sample_context,
                }
            )
            sample_abs_error = np.abs(
                truth_norm.loc[:, sample_id].to_numpy(dtype=float) - pred_norm.loc[:, sample_id].to_numpy(dtype=float)
            )
            for signature_name, abs_error in zip(truth_norm.index.tolist(), sample_abs_error.tolist()):
                exposure_error_records.append(
                    {
                        "sample_id": sample_id,
                        "split_id": split_id,
                        "burden": burden,
                        "signature_name": signature_name,
                        "abs_error": float(abs_error),
                        **sample_context,
                    }
                )
    amusa_rows = annotate_group_columns(pd.DataFrame.from_records(amusa_records), mutation_type=mutation_type)
    assignment_rows = annotate_group_columns(pd.DataFrame.from_records(assignment_records), mutation_type=mutation_type)
    exposure_error_rows = annotate_group_columns(pd.DataFrame.from_records(exposure_error_records), mutation_type=mutation_type)
    return KnownCatalogConfidenceTables(
        amusa_rows=amusa_rows,
        assignment_rows=assignment_rows,
        exposure_error_rows=exposure_error_rows,
    )


def fit_confidence_artifacts_from_known_catalog(
    *,
    sample_source: str | Path,
    signature_source: str | Path,
    exposure_source: str | Path,
    mutation_type: str,
    burdens: tuple[int, ...] = (100, 200, 500, 2000, 50000),
    max_samples_per_burden: int = 100,
    random_seed: int = 0,
    registry: ExpertRegistry | None = None,
    expert_names: list[str] | None = None,
    active_threshold: float = 0.0,
    assignment_f1_threshold: float = 0.8,
    amusa_method: str = "temperature",
    assignment_method: str = "isotonic",
    conformal_alpha: float = 0.1,
    calibration_fraction: float = 0.25,
    amusa_group_min_size: int = 200,
    assignment_group_min_size: int = 50,
    conformal_group_min_size: int = 100,
) -> ConfidenceArtifacts:
    tables = collect_known_catalog_confidence_tables(
        sample_source=sample_source,
        signature_source=signature_source,
        exposure_source=exposure_source,
        mutation_type=mutation_type,
        burdens=burdens,
        max_samples_per_burden=max_samples_per_burden,
        random_seed=random_seed,
        registry=registry,
        expert_names=expert_names,
        active_threshold=active_threshold,
        assignment_f1_threshold=assignment_f1_threshold,
    )
    split_source = tables.assignment_rows.loc[:, ["split_id", "label"]].copy() if not tables.assignment_rows.empty else pd.DataFrame()
    if split_source.empty and not tables.amusa_rows.empty:
        split_source = (
            tables.amusa_rows.groupby("split_id", as_index=False)["label"]
            .max()
            .rename(columns={"label": "label"})
        )
    train_split_ids, calibration_split_ids, split_metadata = _build_holdout_split(
        split_source,
        calibration_fraction=calibration_fraction,
        random_seed=random_seed,
    )
    amusa_fit_rows = tables.amusa_rows.loc[tables.amusa_rows["split_id"].isin(calibration_split_ids)].copy() if not tables.amusa_rows.empty else pd.DataFrame()
    assignment_fit_rows = tables.assignment_rows.loc[tables.assignment_rows["split_id"].isin(calibration_split_ids)].copy() if not tables.assignment_rows.empty else pd.DataFrame()
    conformal_rows = (
        tables.exposure_error_rows.loc[tables.exposure_error_rows["split_id"].isin(calibration_split_ids)].copy()
        if not tables.exposure_error_rows.empty
        else pd.DataFrame()
    )

    amusa_calibrator = None
    if not amusa_fit_rows.empty:
        amusa_calibrator = fit_probability_calibrator(
            amusa_fit_rows["raw_score"].to_numpy(dtype=float),
            amusa_fit_rows["label"].to_numpy(dtype=int),
            method=amusa_method,
        )
    amusa_group_calibrator = None
    if not amusa_fit_rows.empty:
        amusa_group_calibrator = fit_grouped_probability_calibrator(
            amusa_fit_rows,
            score_column="raw_score",
            label_column="label",
            group_columns=AMUSA_GROUP_COLUMNS,
            method=amusa_method,
            min_group_size=amusa_group_min_size,
        )
    assignment_calibrator = None
    if not assignment_fit_rows.empty:
        assignment_calibrator = fit_probability_calibrator(
            assignment_fit_rows["raw_score"].to_numpy(dtype=float),
            assignment_fit_rows["label"].to_numpy(dtype=int),
            method=assignment_method,
        )
    assignment_group_calibrator = None
    if not assignment_fit_rows.empty:
        assignment_group_calibrator = fit_grouped_probability_calibrator(
            assignment_fit_rows,
            score_column="raw_score",
            label_column="label",
            group_columns=ASSIGNMENT_GROUP_COLUMNS,
            method=assignment_method,
            min_group_size=assignment_group_min_size,
        )
    conformal_margin = None
    if not conformal_rows.empty:
        grouped_conformal = fit_grouped_conformal_margin(
            conformal_rows,
            error_column="abs_error",
            group_columns=EXPOSURE_GROUP_COLUMNS,
            alpha=conformal_alpha,
            min_group_size=conformal_group_min_size,
        )
        conformal_margin = grouped_conformal.global_margin
    else:
        grouped_conformal = None
    return ConfidenceArtifacts(
        amusa_probability_calibrator=amusa_calibrator,
        amusa_group_calibrator=amusa_group_calibrator,
        final_assignment_calibrator=assignment_calibrator,
        final_assignment_group_calibrator=assignment_group_calibrator,
        exposure_conformal_margin=conformal_margin,
        exposure_group_conformal=grouped_conformal,
        metadata={
            "mutation_type": mutation_type,
            "burdens": list(burdens),
            "max_samples_per_burden": max_samples_per_burden,
            "random_seed": random_seed,
            "active_threshold": active_threshold,
            "assignment_f1_threshold": assignment_f1_threshold,
            "amusa_calibration_method": amusa_method,
            "assignment_calibration_method": assignment_method,
            "conformal_alpha": conformal_alpha,
            "calibration_fraction": calibration_fraction,
            "split_strategy": split_metadata,
            "train_split_count": len(train_split_ids),
            "calibration_split_count": len(calibration_split_ids),
            "amusa_rows_total": int(len(tables.amusa_rows)),
            "amusa_rows_calibration": int(len(amusa_fit_rows)),
            "assignment_rows_total": int(len(tables.assignment_rows)),
            "assignment_rows_calibration": int(len(assignment_fit_rows)),
            "exposure_error_rows_total": int(len(tables.exposure_error_rows)),
            "exposure_error_rows_calibration": int(len(conformal_rows)),
            "amusa_group_schema": list(AMUSA_GROUP_COLUMNS),
            "assignment_group_schema": list(ASSIGNMENT_GROUP_COLUMNS),
            "exposure_group_schema": list(EXPOSURE_GROUP_COLUMNS),
            "amusa_group_min_size": amusa_group_min_size,
            "assignment_group_min_size": assignment_group_min_size,
            "conformal_group_min_size": conformal_group_min_size,
            "amusa_group_count": 0 if amusa_group_calibrator is None else len(amusa_group_calibrator.group_calibrators or {}),
            "assignment_group_count": (
                0
                if assignment_group_calibrator is None
                else len(assignment_group_calibrator.group_calibrators or {})
            ),
            "exposure_group_count": 0 if grouped_conformal is None else len(grouped_conformal.group_margins or {}),
            "conformal_margin_type": "split_conformal_global_absolute_error" if conformal_margin is not None else "unavailable",
            "conformal_margin_type_grouped": (
                "hierarchical_group_conditional_split_conformal" if grouped_conformal is not None else "unavailable"
            ),
        },
    )


__all__ = [
    "BootstrapConfig",
    "ConfidenceArtifacts",
    "GroupedConformalMargin",
    "GroupedProbabilityCalibrator",
    "KnownCatalogConfidenceTables",
    "ProbabilityCalibrator",
    "bootstrap_exposure_intervals",
    "collect_known_catalog_confidence_tables",
    "fit_confidence_artifacts_from_known_catalog",
    "fit_probability_calibrator",
]
