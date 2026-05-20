from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .conformal_groups import (
    CATALOG_ASSESSOR_GROUP_COLUMNS,
    annotate_group_columns,
    group_context_from_sample_counts,
)
from .experts.schema import ExpertSampleResult
from .group_calibration import (
    GroupedProbabilityCalibrator,
    ProbabilityCalibrator,
    fit_grouped_probability_calibrator,
    fit_probability_calibrator,
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def normalize_exposure_dict(exposures: dict[str, float]) -> dict[str, float]:
    positive = {str(name): float(value) for name, value in exposures.items() if float(value) > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in positive.items()}


def pairwise_jaccard(sets_by_name: dict[str, set[str]]) -> tuple[float, dict[str, float]]:
    names = list(sets_by_name)
    if len(names) < 2:
        return 1.0, {}
    pair_scores: dict[str, float] = {}
    values = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = sets_by_name[left_name]
            right = sets_by_name[right_name]
            union = len(left | right)
            score = 1.0 if union == 0 else len(left & right) / union
            pair_scores[f"{left_name}__{right_name}"] = float(score)
            values.append(score)
    return float(np.mean(values)) if values else 1.0, pair_scores


def residual_structure_score(sample_result: ExpertSampleResult) -> float:
    mutation_count = float(sample_result.metrics.get("mutation_count", 0.0))
    residual_values = np.maximum(np.asarray(sample_result.residual_counts, dtype=float), 0.0)
    residual_mass = float(np.sum(residual_values))
    if residual_mass <= 0.0 or mutation_count <= 0.0:
        return 0.0
    top_k = min(10, residual_values.shape[0])
    top_share = float(np.sum(np.sort(residual_values)[-top_k:]) / residual_mass)
    residual_fraction = residual_mass / mutation_count
    return _clamp(min(1.0, residual_fraction / 0.25) * min(1.0, top_share / 0.6))


def _pairwise_exposure_disagreement(
    normalized_exposures_by_expert: dict[str, dict[str, float]],
) -> tuple[float, dict[str, float]]:
    names = list(normalized_exposures_by_expert)
    if len(names) < 2:
        return 0.0, {}
    pair_scores: dict[str, float] = {}
    values = []
    signature_names = sorted(
        {
            signature_name
            for exposures in normalized_exposures_by_expert.values()
            for signature_name in exposures
        }
    )
    for index, left_name in enumerate(names):
        left = normalized_exposures_by_expert[left_name]
        for right_name in names[index + 1 :]:
            right = normalized_exposures_by_expert[right_name]
            left_vector = np.asarray([left.get(signature_name, 0.0) for signature_name in signature_names], dtype=float)
            right_vector = np.asarray([right.get(signature_name, 0.0) for signature_name in signature_names], dtype=float)
            tvd = 0.5 * float(np.sum(np.abs(left_vector - right_vector)))
            pair_scores[f"{left_name}__{right_name}"] = tvd
            values.append(tvd)
    return float(np.mean(values)) if values else 0.0, pair_scores


def catalog_insufficiency_level(probability: float) -> str:
    if probability >= 0.75:
        return "high"
    if probability >= 0.55:
        return "medium"
    return "low"


@dataclass(slots=True)
class CatalogInsufficiencyConfig:
    high_threshold: float = 0.75
    medium_threshold: float = 0.55
    low_reconstruction_threshold: float = 0.88
    high_disagreement_threshold: float = 0.45
    high_exposure_disagreement_threshold: float = 0.35
    structured_residual_threshold: float = 0.45
    missing_mass_threshold: float = 0.20
    entropy_threshold: float = 0.58


@dataclass(slots=True)
class CatalogInsufficiencyFeatures:
    mutation_count: float
    expert_count: int
    failed_expert_count: int
    failed_expert_fraction: float
    agreement_score: float
    disagreement_score: float
    exposure_disagreement_score: float
    mean_reconstruction_cosine: float
    best_reconstruction_cosine: float
    mean_relative_l1_pct: float
    mean_residual_structure_score: float
    max_residual_structure_score: float
    missing_catalog_probability_mass: float
    classifier_entropy: float
    pairwise_active_set_jaccard: dict[str, float]
    pairwise_exposure_disagreement: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CatalogInsufficiencyAssessment:
    raw_score: float
    probability: float
    level: str
    rationale: list[str]
    features: CatalogInsufficiencyFeatures
    component_scores: dict[str, float]

    @property
    def score(self) -> float:
        return self.raw_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_score": self.raw_score,
            "probability": self.probability,
            "level": self.level,
            "rationale": self.rationale,
            "component_scores": self.component_scores,
            "features": self.features.to_dict(),
        }


FEATURE_COLUMNS = [
    "mutation_count",
    "failed_expert_fraction",
    "disagreement_score",
    "exposure_disagreement_score",
    "mean_reconstruction_cosine",
    "best_reconstruction_cosine",
    "mean_relative_l1_pct",
    "mean_residual_structure_score",
    "max_residual_structure_score",
    "missing_catalog_probability_mass",
    "classifier_entropy",
]


def extract_catalog_insufficiency_features(
    sample_results_by_expert: dict[str, ExpertSampleResult],
    *,
    failed_expert_count: int = 0,
) -> CatalogInsufficiencyFeatures:
    if not sample_results_by_expert:
        return CatalogInsufficiencyFeatures(
            mutation_count=0.0,
            expert_count=0,
            failed_expert_count=failed_expert_count,
            failed_expert_fraction=1.0 if failed_expert_count > 0 else 0.0,
            agreement_score=0.0,
            disagreement_score=1.0,
            exposure_disagreement_score=0.0,
            mean_reconstruction_cosine=0.0,
            best_reconstruction_cosine=0.0,
            mean_relative_l1_pct=100.0,
            mean_residual_structure_score=0.0,
            max_residual_structure_score=0.0,
            missing_catalog_probability_mass=0.0,
            classifier_entropy=0.0,
            pairwise_active_set_jaccard={},
            pairwise_exposure_disagreement={},
        )

    active_sets = {
        expert_name: set(sample_result.active_signatures)
        for expert_name, sample_result in sample_results_by_expert.items()
    }
    agreement_score, pairwise_active = pairwise_jaccard(active_sets)
    normalized_exposures_by_expert = {
        expert_name: normalize_exposure_dict(sample_result.exposures)
        for expert_name, sample_result in sample_results_by_expert.items()
    }
    exposure_disagreement_score, pairwise_exposure = _pairwise_exposure_disagreement(normalized_exposures_by_expert)

    reconstruction_values = [
        float(sample_result.metrics.get("reconstruction_cosine", 0.0))
        for sample_result in sample_results_by_expert.values()
    ]
    relative_l1_values = [
        float(sample_result.metrics.get("relative_l1_pct", 0.0))
        for sample_result in sample_results_by_expert.values()
    ]
    residual_scores = [
        residual_structure_score(sample_result)
        for sample_result in sample_results_by_expert.values()
    ]
    missing_catalog_masses = [
        float(sample_result.diagnostics.get("missing_catalog_probability_mass", 0.0))
        for sample_result in sample_results_by_expert.values()
    ]
    classifier_entropies = [
        float(sample_result.diagnostics.get("classifier_entropy", 0.0))
        for sample_result in sample_results_by_expert.values()
        if "classifier_entropy" in sample_result.diagnostics
    ]
    mutation_counts = [
        float(sample_result.metrics.get("mutation_count", 0.0))
        for sample_result in sample_results_by_expert.values()
    ]

    expert_count = len(sample_results_by_expert)
    denominator = expert_count + failed_expert_count
    failed_fraction = float(failed_expert_count / denominator) if denominator > 0 else 0.0
    return CatalogInsufficiencyFeatures(
        mutation_count=float(np.mean(mutation_counts)) if mutation_counts else 0.0,
        expert_count=expert_count,
        failed_expert_count=failed_expert_count,
        failed_expert_fraction=failed_fraction,
        agreement_score=float(agreement_score),
        disagreement_score=float(1.0 - agreement_score),
        exposure_disagreement_score=float(exposure_disagreement_score),
        mean_reconstruction_cosine=float(np.mean(reconstruction_values)) if reconstruction_values else 0.0,
        best_reconstruction_cosine=float(np.max(reconstruction_values)) if reconstruction_values else 0.0,
        mean_relative_l1_pct=float(np.mean(relative_l1_values)) if relative_l1_values else 0.0,
        mean_residual_structure_score=float(np.mean(residual_scores)) if residual_scores else 0.0,
        max_residual_structure_score=float(np.max(residual_scores)) if residual_scores else 0.0,
        missing_catalog_probability_mass=float(np.mean(missing_catalog_masses)) if missing_catalog_masses else 0.0,
        classifier_entropy=float(np.mean(classifier_entropies)) if classifier_entropies else 0.0,
        pairwise_active_set_jaccard=pairwise_active,
        pairwise_exposure_disagreement=pairwise_exposure,
    )


def extract_catalog_insufficiency_features_from_sample_result(
    sample_result: ExpertSampleResult,
) -> CatalogInsufficiencyFeatures:
    return extract_catalog_insufficiency_features({"single_expert": sample_result}, failed_expert_count=0)


def assess_catalog_insufficiency(
    features: CatalogInsufficiencyFeatures,
    *,
    config: CatalogInsufficiencyConfig | None = None,
) -> CatalogInsufficiencyAssessment:
    config = config or CatalogInsufficiencyConfig()
    reconstruction_problem = max(
        _clamp((config.low_reconstruction_threshold - features.mean_reconstruction_cosine) / 0.18),
        _clamp((0.92 - features.best_reconstruction_cosine) / 0.18),
    )
    residual_component = max(features.mean_residual_structure_score, features.max_residual_structure_score * 0.85)
    disagreement_component = _clamp(features.disagreement_score / max(config.high_disagreement_threshold, 1e-6))
    exposure_disagreement_component = _clamp(
        features.exposure_disagreement_score / max(config.high_exposure_disagreement_threshold, 1e-6)
    )
    missing_mass_component = _clamp(
        features.missing_catalog_probability_mass / max(config.missing_mass_threshold, 1e-6)
    )
    entropy_component = _clamp(features.classifier_entropy / max(config.entropy_threshold, 1e-6))
    failure_component = _clamp(features.failed_expert_fraction / 0.5)

    probability = _clamp(
        0.05
        + 0.25 * reconstruction_problem
        + 0.24 * residual_component
        + 0.18 * disagreement_component
        + 0.10 * exposure_disagreement_component
        + 0.22 * missing_mass_component
        + 0.05 * entropy_component
        + 0.06 * failure_component
        - (
            0.12
            if (
                features.agreement_score >= 0.75
                and features.mean_reconstruction_cosine >= 0.92
                and features.mean_residual_structure_score < 0.2
            )
            else 0.0
        )
    )
    level = catalog_insufficiency_level(probability)

    rationale: list[str] = []
    if reconstruction_problem >= 0.45:
        rationale.append("Known signature catalog leaves weak reconstruction after refit.")
    if residual_component >= config.structured_residual_threshold:
        rationale.append("Structured residual signal remains after known-signature explanation.")
    if features.disagreement_score >= config.high_disagreement_threshold:
        rationale.append("Experts disagree on the active signature set.")
    if features.exposure_disagreement_score >= config.high_exposure_disagreement_threshold:
        rationale.append("Experts disagree on exposure allocation even when using the same catalog.")
    if features.missing_catalog_probability_mass >= config.missing_mass_threshold:
        rationale.append("Classifier support places non-trivial probability mass on signatures absent from the current catalog.")
    if features.classifier_entropy >= config.entropy_threshold:
        rationale.append("Support detector confidence is diffuse, which weakens catalog adequacy claims.")
    if features.failed_expert_fraction > 0.0:
        rationale.append("Some experts failed or were excluded, reducing confidence in catalog sufficiency.")
    if not rationale:
        rationale.append("Current catalog is broadly sufficient under the available expert evidence.")

    return CatalogInsufficiencyAssessment(
        raw_score=probability,
        probability=probability,
        level=level,
        rationale=rationale,
        features=features,
        component_scores={
            "reconstruction_problem": reconstruction_problem,
            "residual_component": residual_component,
            "disagreement_component": disagreement_component,
            "exposure_disagreement_component": exposure_disagreement_component,
            "missing_mass_component": missing_mass_component,
            "entropy_component": entropy_component,
            "failure_component": failure_component,
        },
    )


def assess_catalog_insufficiency_from_expert_results(
    sample_results_by_expert: dict[str, ExpertSampleResult],
    *,
    failed_expert_count: int = 0,
    config: CatalogInsufficiencyConfig | None = None,
) -> CatalogInsufficiencyAssessment:
    features = extract_catalog_insufficiency_features(
        sample_results_by_expert,
        failed_expert_count=failed_expert_count,
    )
    return assess_catalog_insufficiency(features, config=config)


def assess_catalog_insufficiency_from_sample_result(
    sample_result: ExpertSampleResult,
    *,
    config: CatalogInsufficiencyConfig | None = None,
) -> CatalogInsufficiencyAssessment:
    features = extract_catalog_insufficiency_features_from_sample_result(sample_result)
    return assess_catalog_insufficiency(features, config=config)


def catalog_insufficiency_score_from_sample_result(sample_result: ExpertSampleResult) -> float:
    diagnostics = sample_result.diagnostics or {}
    if "catalog_insufficiency_score" in diagnostics:
        return float(diagnostics["catalog_insufficiency_score"])
    if "catalog_insufficiency_proxy_score" in diagnostics:
        return float(diagnostics["catalog_insufficiency_proxy_score"])
    return assess_catalog_insufficiency_from_sample_result(sample_result).score


def catalog_insufficiency_level_from_sample_result(sample_result: ExpertSampleResult) -> str:
    diagnostics = sample_result.diagnostics or {}
    if "catalog_insufficiency_level" in diagnostics:
        return str(diagnostics["catalog_insufficiency_level"])
    return assess_catalog_insufficiency_from_sample_result(sample_result).level


def catalog_insufficiency_probability_from_sample_result(sample_result: ExpertSampleResult) -> float | None:
    diagnostics = sample_result.diagnostics or {}
    if "catalog_insufficiency_probability" in diagnostics and diagnostics["catalog_insufficiency_probability"] is not None:
        return float(diagnostics["catalog_insufficiency_probability"])
    return None


def catalog_feature_frame_from_records(records: list[CatalogInsufficiencyFeatures]) -> pd.DataFrame:
    rows = []
    for features in records:
        row = {
            "mutation_count": features.mutation_count,
            "failed_expert_fraction": features.failed_expert_fraction,
            "disagreement_score": features.disagreement_score,
            "exposure_disagreement_score": features.exposure_disagreement_score,
            "mean_reconstruction_cosine": features.mean_reconstruction_cosine,
            "best_reconstruction_cosine": features.best_reconstruction_cosine,
            "mean_relative_l1_pct": features.mean_relative_l1_pct,
            "mean_residual_structure_score": features.mean_residual_structure_score,
            "max_residual_structure_score": features.max_residual_structure_score,
            "missing_catalog_probability_mass": features.missing_catalog_probability_mass,
            "classifier_entropy": features.classifier_entropy,
        }
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=FEATURE_COLUMNS).astype(float)


@dataclass(slots=True)
class CatalogInsufficiencyModel:
    feature_names: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    coefficients: list[float]
    intercept: float
    probability_calibrator: ProbabilityCalibrator | None = None
    group_probability_calibrator: GroupedProbabilityCalibrator | None = None
    metadata: dict[str, Any] | None = None

    def raw_score_from_features(self, features: CatalogInsufficiencyFeatures) -> float:
        feature_frame = catalog_feature_frame_from_records([features])
        values = feature_frame.loc[0, self.feature_names].to_numpy(dtype=float)
        scaled = (values - np.asarray(self.scaler_mean, dtype=float)) / np.asarray(self.scaler_scale, dtype=float)
        logit_value = float(np.dot(scaled, np.asarray(self.coefficients, dtype=float)) + float(self.intercept))
        return float(1.0 / (1.0 + np.exp(-logit_value)))

    def assess(
        self,
        sample_results_by_expert: dict[str, ExpertSampleResult],
        *,
        failed_expert_count: int = 0,
        group_context: dict[str, Any] | None = None,
    ) -> CatalogInsufficiencyAssessment:
        features = extract_catalog_insufficiency_features(
            sample_results_by_expert,
            failed_expert_count=failed_expert_count,
        )
        raw_score = self.raw_score_from_features(features)
        probability = raw_score
        calibration_group = "none"
        if self.group_probability_calibrator is not None:
            probability, calibration_group = self.group_probability_calibrator.transform_one(
                raw_score,
                context=group_context,
                return_source=True,
            )
        elif self.probability_calibrator is not None:
            probability = float(self.probability_calibrator.transform([raw_score])[0])
            calibration_group = "global"
        level = catalog_insufficiency_level(probability)
        rationale = []
        if features.missing_catalog_probability_mass > 0.15:
            rationale.append("Classifier support highlights signatures absent from the reference catalog.")
        if features.mean_residual_structure_score > 0.45:
            rationale.append("Residual structure remains after refit.")
        if features.disagreement_score > 0.45:
            rationale.append("Experts disagree on the active signature set.")
        if features.exposure_disagreement_score > 0.35:
            rationale.append("Experts disagree on exposure allocation.")
        if not rationale:
            rationale.append("Trained assessor finds low catalog-insufficiency risk under current evidence.")
        return CatalogInsufficiencyAssessment(
            raw_score=raw_score,
            probability=probability,
            level=level,
            rationale=rationale,
            features=features,
            component_scores={
                "raw_model_score": raw_score,
                "calibration_group": calibration_group,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "probability_calibrator": None if self.probability_calibrator is None else self.probability_calibrator.to_dict(),
            "group_probability_calibrator": (
                None if self.group_probability_calibrator is None else self.group_probability_calibrator.to_dict()
            ),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CatalogInsufficiencyModel":
        probability_calibrator = payload.get("probability_calibrator")
        if probability_calibrator is not None:
            probability_calibrator = ProbabilityCalibrator.from_dict(probability_calibrator)
        group_probability_calibrator = payload.get("group_probability_calibrator")
        if group_probability_calibrator is not None:
            group_probability_calibrator = GroupedProbabilityCalibrator.from_dict(group_probability_calibrator)
        return cls(
            feature_names=[str(value) for value in payload["feature_names"]],
            scaler_mean=[float(value) for value in payload["scaler_mean"]],
            scaler_scale=[float(value) for value in payload["scaler_scale"]],
            coefficients=[float(value) for value in payload["coefficients"]],
            intercept=float(payload["intercept"]),
            probability_calibrator=probability_calibrator,
            group_probability_calibrator=group_probability_calibrator,
            metadata=payload.get("metadata") or {},
        )

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "CatalogInsufficiencyModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_catalog_insufficiency_model(
    feature_frame: pd.DataFrame,
    labels: pd.Series | np.ndarray | list[int],
    *,
    probability_calibrator: ProbabilityCalibrator | None = None,
    group_probability_calibrator: GroupedProbabilityCalibrator | None = None,
    metadata: dict[str, Any] | None = None,
) -> CatalogInsufficiencyModel:
    labels_array = np.asarray(labels, dtype=int)
    if feature_frame.empty or len(np.unique(labels_array)) < 2:
        raise ValueError("Catalog insufficiency model requires non-empty features and both label classes.")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feature_frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float))
    model = LogisticRegression(max_iter=2000)
    model.fit(x_scaled, labels_array)
    if probability_calibrator is None:
        probability_calibrator = None
    return CatalogInsufficiencyModel(
        feature_names=list(FEATURE_COLUMNS),
        scaler_mean=[float(value) for value in scaler.mean_.tolist()],
        scaler_scale=[float(value) for value in scaler.scale_.tolist()],
        coefficients=[float(value) for value in model.coef_[0].tolist()],
        intercept=float(model.intercept_[0]),
        probability_calibrator=probability_calibrator,
        group_probability_calibrator=group_probability_calibrator,
        metadata=metadata or {},
    )


def collect_catalog_insufficiency_training_data(
    *,
    sample_source: str | Path,
    signature_source: str | Path,
    exposure_source: str | Path,
    mutation_type: str,
    burdens: tuple[int, ...] = (200, 2000),
    removed_signatures: list[str] | None = None,
    max_positive_per_signature: int = 50,
    max_negative_per_signature: int = 50,
    active_threshold: float = 0.0,
    random_seed: int = 0,
    expert_names: list[str] | None = None,
    registry: Any | None = None,
) -> pd.DataFrame:
    from .benchmark import load_truth_exposures
    from .experts.io import load_expert_request
    from .experts.registry import build_default_registry
    from .metrics import normalize_exposures
    from .simulation import simulate_counts_from_truth, subset_samples

    registry = registry or build_default_registry(".")
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
    truth_norm = normalize_exposures(truth.exposures)
    rng = np.random.default_rng(random_seed)
    candidate_removed_signatures = list(removed_signatures) if removed_signatures is not None else base_request.signature_names
    rows: list[dict[str, Any]] = []

    for burden in burdens:
        for removed_signature in candidate_removed_signatures:
            if removed_signature not in truth_norm.index:
                continue
            positive_ids = truth_norm.columns[truth_norm.loc[removed_signature] > active_threshold].tolist()
            negative_ids = truth_norm.columns[truth_norm.loc[removed_signature] <= active_threshold].tolist()
            if not positive_ids or not negative_ids:
                continue
            if max_positive_per_signature and len(positive_ids) > max_positive_per_signature:
                positive_ids = sorted(rng.choice(positive_ids, size=max_positive_per_signature, replace=False).tolist())
            if max_negative_per_signature and len(negative_ids) > max_negative_per_signature:
                negative_ids = sorted(rng.choice(negative_ids, size=max_negative_per_signature, replace=False).tolist())
            chosen_sample_ids = positive_ids + negative_ids
            truth_subset = subset_samples(truth.exposures, chosen_sample_ids)
            simulated_samples = simulate_counts_from_truth(
                base_request.signature_matrix,
                truth_subset,
                burden=burden,
                rng=rng,
            )
            incomplete_catalog = base_request.signature_matrix.drop(columns=[removed_signature])
            request = type(base_request)(
                mutation_type=base_request.mutation_type,
                sample_matrix=simulated_samples,
                signature_matrix=incomplete_catalog,
                channel_metadata=base_request.channel_metadata,
                sample_source=base_request.sample_source,
                signature_source=base_request.signature_source,
                reference_name=base_request.reference_name,
                request_id=f"catalog_assessor_training_{mutation_type}_{removed_signature}_{burden}",
                alignment_strategy=base_request.alignment_strategy,
            )
            runs = registry.run_all(request, expert_names)
            successful_runs = [run for run in runs if run.status == "success" and run.sample_results]
            failed_runs = [run for run in runs if run.status != "success"]
            for sample_id in request.sample_ids:
                sample_results_by_expert = {}
                for run in successful_runs:
                    for sample_result in run.sample_results:
                        if sample_result.sample_id == sample_id:
                            sample_results_by_expert[run.expert_name] = sample_result
                            break
                if not sample_results_by_expert:
                    continue
                features = extract_catalog_insufficiency_features(
                    sample_results_by_expert,
                    failed_expert_count=len(failed_runs),
                )
                base_assessment = assess_catalog_insufficiency(features)
                sample_group_context = group_context_from_sample_counts(
                    simulated_samples.loc[:, sample_id],
                    mutation_type=request.mutation_type,
                    disagreement_score=features.disagreement_score,
                ).to_dict()
                row = {
                    "sample_id": sample_id,
                    "split_id": f"{sample_id}::{removed_signature}::{burden}",
                    "burden": burden,
                    "removed_signature": removed_signature,
                    "label": int(sample_id in positive_ids),
                    "rule_proxy_score": base_assessment.score,
                    "rule_level": base_assessment.level,
                    **sample_group_context,
                    **catalog_feature_frame_from_records([features]).iloc[0].to_dict(),
                }
                rows.append(row)
    return annotate_group_columns(pd.DataFrame.from_records(rows), mutation_type=mutation_type)


def _features_from_training_row(row: pd.Series) -> CatalogInsufficiencyFeatures:
    return CatalogInsufficiencyFeatures(
        mutation_count=float(row["mutation_count"]),
        expert_count=0,
        failed_expert_count=0,
        failed_expert_fraction=float(row["failed_expert_fraction"]),
        agreement_score=float(1.0 - row["disagreement_score"]),
        disagreement_score=float(row["disagreement_score"]),
        exposure_disagreement_score=float(row["exposure_disagreement_score"]),
        mean_reconstruction_cosine=float(row["mean_reconstruction_cosine"]),
        best_reconstruction_cosine=float(row["best_reconstruction_cosine"]),
        mean_relative_l1_pct=float(row["mean_relative_l1_pct"]),
        mean_residual_structure_score=float(row["mean_residual_structure_score"]),
        max_residual_structure_score=float(row["max_residual_structure_score"]),
        missing_catalog_probability_mass=float(row["missing_catalog_probability_mass"]),
        classifier_entropy=float(row["classifier_entropy"]),
        pairwise_active_set_jaccard={},
        pairwise_exposure_disagreement={},
    )


def fit_catalog_insufficiency_model_from_benchmark(
    *,
    sample_source: str | Path,
    signature_source: str | Path,
    exposure_source: str | Path,
    mutation_type: str,
    burdens: tuple[int, ...] = (200, 2000),
    removed_signatures: list[str] | None = None,
    max_positive_per_signature: int = 50,
    max_negative_per_signature: int = 50,
    active_threshold: float = 0.0,
    random_seed: int = 0,
    expert_names: list[str] | None = None,
    registry: Any | None = None,
    calibration_method: str = "isotonic",
    calibration_fraction: float = 0.25,
) -> tuple[CatalogInsufficiencyModel, pd.DataFrame]:
    from .confidence import _build_holdout_split

    training_frame = collect_catalog_insufficiency_training_data(
        sample_source=sample_source,
        signature_source=signature_source,
        exposure_source=exposure_source,
        mutation_type=mutation_type,
        burdens=burdens,
        removed_signatures=removed_signatures,
        max_positive_per_signature=max_positive_per_signature,
        max_negative_per_signature=max_negative_per_signature,
        active_threshold=active_threshold,
        random_seed=random_seed,
        expert_names=expert_names,
        registry=registry,
    )
    if training_frame.empty:
        raise ValueError("Catalog insufficiency training data is empty.")
    train_split_ids, calibration_split_ids, split_metadata = _build_holdout_split(
        training_frame.loc[:, ["split_id", "label"]],
        calibration_fraction=calibration_fraction,
        random_seed=random_seed,
    )
    training_frame = training_frame.copy()
    training_frame["split_partition"] = "unused"
    if train_split_ids:
        training_frame.loc[training_frame["split_id"].isin(train_split_ids), "split_partition"] = "train"
    if calibration_split_ids:
        training_frame.loc[training_frame["split_id"].isin(calibration_split_ids), "split_partition"] = "calibration"
    train_frame = (
        training_frame.loc[training_frame["split_id"].isin(train_split_ids)].copy()
        if train_split_ids
        else training_frame.copy()
    )
    calibration_frame = (
        training_frame.loc[training_frame["split_id"].isin(calibration_split_ids)].copy()
        if calibration_split_ids
        else pd.DataFrame(columns=training_frame.columns)
    )
    feature_frame = train_frame.loc[:, FEATURE_COLUMNS].astype(float)
    labels = train_frame["label"].astype(int)
    base_model = fit_catalog_insufficiency_model(
        feature_frame,
        labels,
        probability_calibrator=None,
        metadata={
            "mutation_type": mutation_type,
            "burdens": list(burdens),
            "random_seed": random_seed,
            "calibration_method": calibration_method,
            "calibration_fraction": calibration_fraction,
            "split_strategy": split_metadata,
        },
    )
    raw_scores = np.asarray(
        [
            base_model.raw_score_from_features(_features_from_training_row(row))
            for _, row in calibration_frame.iterrows()
        ],
        dtype=float,
    )
    calibrator = None
    if not calibration_frame.empty:
        calibrator = fit_probability_calibrator(
            raw_scores,
            calibration_frame["label"].to_numpy(dtype=int),
            method=calibration_method,
        )
    grouped_calibrator = None
    if not calibration_frame.empty:
        calibration_with_scores = calibration_frame.copy()
        calibration_with_scores["raw_score"] = raw_scores
        grouped_calibrator = fit_grouped_probability_calibrator(
            calibration_with_scores,
            score_column="raw_score",
            label_column="label",
            group_columns=CATALOG_ASSESSOR_GROUP_COLUMNS,
            method=calibration_method,
            min_group_size=25,
        )
    model = fit_catalog_insufficiency_model(
        feature_frame,
        labels,
        probability_calibrator=calibrator,
        group_probability_calibrator=grouped_calibrator,
        metadata={
            "mutation_type": mutation_type,
            "burdens": list(burdens),
            "random_seed": random_seed,
            "calibration_method": calibration_method,
            "calibration_fraction": calibration_fraction,
            "split_strategy": split_metadata,
            "train_rows": int(len(train_frame)),
            "calibration_rows": int(len(calibration_frame)),
            "group_schema": list(CATALOG_ASSESSOR_GROUP_COLUMNS),
            "group_calibration_count": 0 if grouped_calibrator is None else len(grouped_calibrator.group_calibrators or {}),
        },
    )
    return model, training_frame


__all__ = [
    "CatalogInsufficiencyAssessment",
    "CatalogInsufficiencyConfig",
    "CatalogInsufficiencyFeatures",
    "assess_catalog_insufficiency",
    "assess_catalog_insufficiency_from_expert_results",
    "assess_catalog_insufficiency_from_sample_result",
    "catalog_insufficiency_level",
    "catalog_insufficiency_level_from_sample_result",
    "catalog_insufficiency_probability_from_sample_result",
    "catalog_insufficiency_score_from_sample_result",
    "extract_catalog_insufficiency_features",
    "extract_catalog_insufficiency_features_from_sample_result",
    "normalize_exposure_dict",
    "pairwise_jaccard",
    "residual_structure_score",
]
