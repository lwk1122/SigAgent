from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..experience.schema import ExperienceRecord
from ..experts.schema import json_ready


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        return values
    return values / norm


def residual_vector_from_record(record: ExperienceRecord) -> np.ndarray | None:
    raw_values = (record.fused_sample_result or {}).get("residual_counts") or []
    if not raw_values:
        return None
    values = np.asarray(raw_values, dtype=float)
    values = np.maximum(values, 0.0)
    if float(np.sum(values)) <= 0.0:
        return None
    return _normalize_vector(values)


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def record_signature_fingerprint(record: ExperienceRecord, *, top_k: int = 3) -> str:
    signatures = [
        str(item.get("name"))
        for item in (record.fusion_report.get("known_signatures") or [])[:top_k]
        if item.get("name")
    ]
    if not signatures:
        return "none"
    return "|".join(signatures)


@dataclass(slots=True)
class RecurrenceCluster:
    cluster_id: str
    mutation_type: str
    record_ids: list[str]
    sample_ids: list[str]
    fingerprint: str
    recurrence_count: int
    mean_pairwise_similarity: float
    mean_catalog_insufficiency_probability: float | None = None
    mean_residual_structure_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "cluster_id": self.cluster_id,
                "mutation_type": self.mutation_type,
                "record_ids": self.record_ids,
                "sample_ids": self.sample_ids,
                "fingerprint": self.fingerprint,
                "recurrence_count": self.recurrence_count,
                "mean_pairwise_similarity": self.mean_pairwise_similarity,
                "mean_catalog_insufficiency_probability": self.mean_catalog_insufficiency_probability,
                "mean_residual_structure_score": self.mean_residual_structure_score,
                "metadata": self.metadata,
            }
        )


def build_recurrence_clusters(
    records: list[ExperienceRecord],
    *,
    similarity_threshold: float = 0.90,
    min_cluster_size: int = 2,
) -> list[RecurrenceCluster]:
    if not records:
        return []
    vectors = {record.record_id: residual_vector_from_record(record) for record in records}
    by_mutation_type: dict[str, list[ExperienceRecord]] = {}
    for record in records:
        by_mutation_type.setdefault(record.mutation_type, []).append(record)

    clusters: list[RecurrenceCluster] = []
    fallback_similarity_threshold = max(0.60, similarity_threshold * 0.75)
    for mutation_type, group_records in by_mutation_type.items():
        adjacency: dict[str, set[str]] = {record.record_id: set() for record in group_records}
        for index, left_record in enumerate(group_records):
            left_vector = vectors[left_record.record_id]
            for right_record in group_records[index + 1 :]:
                right_vector = vectors[right_record.record_id]
                similarity = cosine_similarity(left_vector, right_vector)
                if similarity >= similarity_threshold:
                    adjacency[left_record.record_id].add(right_record.record_id)
                    adjacency[right_record.record_id].add(left_record.record_id)
                elif (
                    record_signature_fingerprint(left_record) == record_signature_fingerprint(right_record)
                    and record_signature_fingerprint(left_record) != "none"
                    and similarity >= fallback_similarity_threshold
                ):
                    adjacency[left_record.record_id].add(right_record.record_id)
                    adjacency[right_record.record_id].add(left_record.record_id)

        visited: set[str] = set()
        record_by_id = {record.record_id: record for record in group_records}
        for seed_record in group_records:
            if seed_record.record_id in visited:
                continue
            stack = [seed_record.record_id]
            component: list[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                stack.extend(sorted(adjacency[current] - visited))
            if len(component) < min_cluster_size:
                continue
            component_records = [record_by_id[record_id] for record_id in component]
            pair_scores: list[float] = []
            for index, left_id in enumerate(component):
                for right_id in component[index + 1 :]:
                    pair_scores.append(cosine_similarity(vectors[left_id], vectors[right_id]))
            mean_probability_values = [
                float(record.fusion_report.get("catalog_insufficiency_probability"))
                for record in component_records
                if record.fusion_report.get("catalog_insufficiency_probability") is not None
            ]
            residual_scores = [
                float((record.fused_sample_result.get("diagnostics") or {}).get("residual_structure_score", 0.0))
                for record in component_records
            ]
            fingerprint = record_signature_fingerprint(component_records[0])
            clusters.append(
                RecurrenceCluster(
                    cluster_id=f"{mutation_type}__cluster_{len(clusters) + 1}",
                    mutation_type=mutation_type,
                    record_ids=[record.record_id for record in component_records],
                    sample_ids=[record.sample_id for record in component_records],
                    fingerprint=fingerprint,
                    recurrence_count=len(component_records),
                    mean_pairwise_similarity=float(np.mean(pair_scores)) if pair_scores else 1.0,
                    mean_catalog_insufficiency_probability=(
                        float(np.mean(mean_probability_values)) if mean_probability_values else None
                    ),
                    mean_residual_structure_score=float(np.mean(residual_scores)) if residual_scores else None,
                    metadata={
                        "similarity_threshold": similarity_threshold,
                        "fallback_similarity_threshold": fallback_similarity_threshold,
                        "clustering_mode": "residual_cosine_with_fingerprint_assisted_fallback",
                    },
                )
            )
    return clusters


def cluster_map(clusters: list[RecurrenceCluster]) -> dict[str, RecurrenceCluster]:
    mapping: dict[str, RecurrenceCluster] = {}
    for cluster in clusters:
        for record_id in cluster.record_ids:
            mapping[record_id] = cluster
    return mapping


__all__ = [
    "RecurrenceCluster",
    "build_recurrence_clusters",
    "cluster_map",
    "cosine_similarity",
    "record_signature_fingerprint",
    "residual_vector_from_record",
]
