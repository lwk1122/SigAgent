from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return {
            "index": [str(v) for v in value.index.tolist()],
            "columns": [str(v) for v in value.columns.tolist()],
            "data": value.to_numpy().tolist(),
        }
    if isinstance(value, pd.Series):
        return {str(k): json_ready(v) for k, v in value.to_dict().items()}
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@dataclass(slots=True)
class ExpertRequest:
    mutation_type: str
    sample_matrix: pd.DataFrame
    signature_matrix: pd.DataFrame
    channel_metadata: pd.DataFrame | None = None
    sample_source: str | None = None
    signature_source: str | None = None
    reference_name: str | None = None
    request_id: str | None = None
    alignment_strategy: str | None = None

    def __post_init__(self) -> None:
        if self.sample_matrix.shape[0] != self.signature_matrix.shape[0]:
            raise ValueError("sample_matrix and signature_matrix must have the same number of channels.")
        if self.sample_matrix.index.tolist() != self.signature_matrix.index.tolist():
            raise ValueError("sample_matrix and signature_matrix must share the same channel index.")
        if self.request_id is None:
            self.request_id = "request"
        if self.reference_name is None and self.signature_source:
            self.reference_name = self.signature_source.rsplit("/", 1)[-1]

    @property
    def channel_ids(self) -> list[str]:
        return [str(v) for v in self.sample_matrix.index.tolist()]

    @property
    def sample_ids(self) -> list[str]:
        return [str(v) for v in self.sample_matrix.columns.tolist()]

    @property
    def signature_names(self) -> list[str]:
        return [str(v) for v in self.signature_matrix.columns.tolist()]

    def with_samples(self, sample_ids: list[str]) -> "ExpertRequest":
        missing = [sample_id for sample_id in sample_ids if sample_id not in self.sample_matrix.columns]
        if missing:
            raise KeyError(f"Unknown sample ids: {missing}")
        return ExpertRequest(
            mutation_type=self.mutation_type,
            sample_matrix=self.sample_matrix.loc[:, sample_ids].copy(),
            signature_matrix=self.signature_matrix.copy(),
            channel_metadata=None if self.channel_metadata is None else self.channel_metadata.copy(),
            sample_source=self.sample_source,
            signature_source=self.signature_source,
            reference_name=self.reference_name,
            request_id=self.request_id,
            alignment_strategy=self.alignment_strategy,
        )


@dataclass(slots=True)
class ExpertSampleResult:
    sample_id: str
    active_signatures: list[str]
    exposures: dict[str, float]
    signature_scores: dict[str, float | None] = field(default_factory=dict)
    signature_probabilities: dict[str, float | None] = field(default_factory=dict)
    reconstructed_counts: list[float] = field(default_factory=list)
    residual_counts: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "sample_id": self.sample_id,
                "active_signatures": self.active_signatures,
                "exposures": self.exposures,
                "signature_scores": self.signature_scores,
                "signature_probabilities": self.signature_probabilities,
                "reconstructed_counts": self.reconstructed_counts,
                "residual_counts": self.residual_counts,
                "metrics": self.metrics,
                "diagnostics": self.diagnostics,
                "warnings": self.warnings,
            }
        )


@dataclass(slots=True)
class ExpertRunResult:
    expert_name: str
    mutation_type: str
    request_id: str
    status: str
    signature_names: list[str]
    channel_ids: list[str]
    sample_results: list[ExpertSampleResult] = field(default_factory=list)
    runtime_seconds: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "expert_name": self.expert_name,
                "mutation_type": self.mutation_type,
                "request_id": self.request_id,
                "status": self.status,
                "signature_names": self.signature_names,
                "channel_ids": self.channel_ids,
                "sample_results": [sample.to_dict() for sample in self.sample_results],
                "runtime_seconds": self.runtime_seconds,
                "parameters": self.parameters,
                "artifacts": self.artifacts,
                "warnings": self.warnings,
                "error": self.error,
            }
        )

    @classmethod
    def failed(
        cls,
        *,
        expert_name: str,
        request: ExpertRequest,
        runtime_seconds: float,
        parameters: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        error: str,
    ) -> "ExpertRunResult":
        return cls(
            expert_name=expert_name,
            mutation_type=request.mutation_type,
            request_id=request.request_id or "request",
            status="failed",
            signature_names=request.signature_names,
            channel_ids=request.channel_ids,
            runtime_seconds=runtime_seconds,
            parameters=parameters or {},
            artifacts=artifacts or {},
            warnings=warnings or [],
            error=error,
        )
