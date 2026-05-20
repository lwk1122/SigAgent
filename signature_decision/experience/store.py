from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from ..cohort import CohortAggregationOutput
from ..experts.schema import ExpertRequest, ExpertRunResult, ExpertSampleResult, json_ready
from ..fusion import RuleFusionOutput
from ..schemas import FinalSampleReport
from .schema import ExperienceRecord, ReviewDecision


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: Any) -> str:
    return _sha256_text(json.dumps(json_ready(payload), sort_keys=True, ensure_ascii=False))


def _safe_token(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return collapsed[:80] if collapsed else "sample"


def _find_sample_result(run: ExpertRunResult, sample_id: str) -> ExpertSampleResult | None:
    for sample_result in run.sample_results:
        if sample_result.sample_id == sample_id:
            return sample_result
    return None


def _sample_input_summary(request: ExpertRequest, sample_id: str) -> dict[str, Any]:
    sample_counts = request.sample_matrix.loc[:, sample_id].astype(float)
    mutation_count = float(sample_counts.sum())
    nonzero = sample_counts[sample_counts > 0.0]
    top_channels = [
        {"channel_id": str(channel_id), "count": float(count)}
        for channel_id, count in nonzero.sort_values(ascending=False).head(5).items()
    ]
    sample_hash = _payload_hash(
        {
            "sample_id": sample_id,
            "channels": request.channel_ids,
            "counts": [float(value) for value in sample_counts.tolist()],
        }
    )
    return {
        "mutation_count": mutation_count,
        "nonzero_channel_count": int((sample_counts > 0.0).sum()),
        "channel_sparsity": float(1.0 - ((sample_counts > 0.0).sum() / len(sample_counts))) if len(sample_counts) > 0 else 0.0,
        "top_channels": top_channels,
        "sample_hash": sample_hash,
    }


def _expert_outputs_for_sample(runs: list[ExpertRunResult], sample_id: str) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for run in runs:
        sample_result = _find_sample_result(run, sample_id)
        outputs[run.expert_name] = {
            "status": run.status,
            "runtime_seconds": run.runtime_seconds,
            "parameters": run.parameters,
            "warnings": run.warnings,
            "error": run.error,
            "sample_result": None if sample_result is None else sample_result.to_dict(),
        }
    return outputs


def _cohort_context_for_sample(
    cohort_output: CohortAggregationOutput,
    sample_id: str,
) -> dict[str, Any]:
    candidate_rows = cohort_output.candidates_frame.loc[
        cohort_output.candidates_frame["sample_id"] == sample_id
    ] if not cohort_output.candidates_frame.empty else pd.DataFrame(columns=["sample_id", "candidate_type", "reason"])
    queue_reasons = {
        str(row["candidate_type"]): str(row["reason"])
        for _, row in candidate_rows.iterrows()
    }
    queue_types = sorted(queue_reasons)
    return {
        "cohort_sample_count": cohort_output.report.n_samples,
        "recommendation_counts": cohort_output.report.recommendation_counts,
        "catalog_insufficiency_level_counts": cohort_output.report.catalog_insufficiency_level_counts,
        "queue_types": queue_types,
        "queue_reasons": queue_reasons,
    }


def build_decision_experience_records(
    *,
    request: ExpertRequest,
    runs: list[ExpertRunResult],
    fusion_output: RuleFusionOutput,
    cohort_output: CohortAggregationOutput,
    confidence_artifact_path: str | Path | None = None,
    catalog_assessor_artifact_path: str | Path | None = None,
    run_label: str | None = None,
    output_dir: str | Path | None = None,
) -> list[ExperienceRecord]:
    created_at = utc_now_iso()
    batch_token = created_at.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    confidence_sha = _sha256_file(confidence_artifact_path)
    assessor_sha = _sha256_file(catalog_assessor_artifact_path)
    fusion_config_hash = _payload_hash(fusion_output.fused_run.parameters.get("config") or {})
    records: list[ExperienceRecord] = []

    for report in fusion_output.reports:
        sample_id = report.sample_id
        input_summary = _sample_input_summary(request, sample_id)
        fused_sample_result = _find_sample_result(fusion_output.fused_run, sample_id)
        cohort_context = _cohort_context_for_sample(cohort_output, sample_id)
        queue_types = list(cohort_context.get("queue_types") or [])
        review_status = "pending_review" if queue_types else "archived"
        recommendation = {
            "primary_recommendation": report.primary_recommendation,
            "secondary_recommendations": report.secondary_recommendations,
            "recommendation_rationale": report.recommendation_rationale,
        }
        record_id = "__".join(
            [
                batch_token,
                request.mutation_type,
                _safe_token(sample_id),
                str(input_summary["sample_hash"])[:12],
            ]
        )
        records.append(
            ExperienceRecord(
                record_id=record_id,
                created_at=created_at,
                sample_id=sample_id,
                mutation_type=request.mutation_type,
                request_id=request.request_id,
                source_context={
                    "sample_source": request.sample_source,
                    "signature_source": request.signature_source,
                    "reference_name": request.reference_name,
                    "output_dir": None if output_dir is None else str(output_dir),
                },
                input_summary=input_summary,
                expert_outputs=_expert_outputs_for_sample(runs, sample_id),
                fusion_report=report.to_dict(),
                fused_sample_result={} if fused_sample_result is None else fused_sample_result.to_dict(),
                cohort_context=cohort_context,
                recommendation=recommendation,
                artifact_versions={
                    "confidence_artifact_path": None if confidence_artifact_path is None else str(confidence_artifact_path),
                    "confidence_artifact_sha256": confidence_sha,
                    "catalog_assessor_artifact_path": (
                        None if catalog_assessor_artifact_path is None else str(catalog_assessor_artifact_path)
                    ),
                    "catalog_assessor_artifact_sha256": assessor_sha,
                    "fusion_config_sha256": fusion_config_hash,
                    "expert_names": [run.expert_name for run in runs],
                },
                review_status=review_status,
                queue_types=queue_types,
                metadata={
                    "run_label": run_label or request.request_id or "decision",
                    "cohort_size": len(request.sample_ids),
                },
            )
        )
    return records


class ExperienceStore:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.records_dir = self.root_dir / "records"
        self.reviews_dir = self.root_dir / "reviews"
        self.discovery_packets_dir = self.root_dir / "discovery_packets"
        self.index_dir = self.root_dir / "index"
        self.datasets_dir = self.root_dir / "datasets"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.reviews_dir.mkdir(parents=True, exist_ok=True)
        self.discovery_packets_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def append_records(self, records: list[ExperienceRecord]) -> list[Path]:
        written_paths: list[Path] = []
        for record in records:
            path = self.records_dir / f"{record.record_id}.json"
            if path.exists():
                raise FileExistsError(f"Experience record already exists: {path.name}")
            path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
            written_paths.append(path)
        self.write_indexes()
        return written_paths

    def append_review_decisions(self, reviews: list[ReviewDecision]) -> list[Path]:
        written_paths: list[Path] = []
        for review in reviews:
            path = self.reviews_dir / f"{review.review_id}.json"
            if path.exists():
                raise FileExistsError(f"Review decision already exists: {path.name}")
            path.write_text(json.dumps(review.to_dict(), indent=2, ensure_ascii=False))
            written_paths.append(path)
        self.write_indexes()
        return written_paths

    def append_discovery_packets(self, packets: list[Any]) -> list[Path]:
        written_paths: list[Path] = []
        for packet in packets:
            path = self.discovery_packets_dir / f"{packet.packet_id}.json"
            if path.exists():
                raise FileExistsError(f"Discovery packet already exists: {path.name}")
            path.write_text(json.dumps(packet.to_dict(), indent=2, ensure_ascii=False))
            written_paths.append(path)
        self.write_indexes()
        return written_paths

    def load_records(self) -> list[ExperienceRecord]:
        return [
            ExperienceRecord.from_dict(json.loads(path.read_text()))
            for path in sorted(self.records_dir.glob("*.json"))
        ]

    def load_review_decisions(self) -> list[ReviewDecision]:
        return [
            ReviewDecision.from_dict(json.loads(path.read_text()))
            for path in sorted(self.reviews_dir.glob("*.json"))
        ]

    def load_discovery_packets(self) -> list[Any]:
        from ..discovery.packet import DiscoveryPacket

        return [
            DiscoveryPacket.from_dict(json.loads(path.read_text()))
            for path in sorted(self.discovery_packets_dir.glob("*.json"))
        ]

    def write_indexes(self) -> None:
        records = self.load_records()
        reviews = self.load_review_decisions()
        discovery_packets = self.load_discovery_packets()
        pd.DataFrame.from_records([record.to_index_row() for record in records]).to_csv(
            self.index_dir / "records.tsv",
            sep="\t",
            index=False,
        )
        pd.DataFrame.from_records([review.to_index_row() for review in reviews]).to_csv(
            self.index_dir / "reviews.tsv",
            sep="\t",
            index=False,
        )
        pd.DataFrame.from_records([packet.to_index_row() for packet in discovery_packets]).to_csv(
            self.index_dir / "discovery_packets.tsv",
            sep="\t",
            index=False,
        )


__all__ = [
    "ExperienceStore",
    "build_decision_experience_records",
    "utc_now_iso",
]
