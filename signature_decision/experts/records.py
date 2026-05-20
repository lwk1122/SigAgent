from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import ExpertRunResult


def runs_to_summary_frame(runs: list[ExpertRunResult]) -> pd.DataFrame:
    records: list[dict] = []
    for run in runs:
        if not run.sample_results:
            records.append(
                {
                    "expert_name": run.expert_name,
                    "status": run.status,
                    "sample_id": None,
                    "active_signature_count": None,
                    "top_signatures": None,
                    "reconstruction_cosine": None,
                    "rss": None,
                    "runtime_seconds": run.runtime_seconds,
                    "error": run.error,
                }
            )
            continue
        for sample in run.sample_results:
            records.append(
                {
                    "expert_name": run.expert_name,
                    "status": run.status,
                    "sample_id": sample.sample_id,
                    "active_signature_count": len(sample.active_signatures),
                    "top_signatures": ",".join(sample.active_signatures[:5]),
                    "reconstruction_cosine": sample.metrics.get("reconstruction_cosine"),
                    "rss": sample.metrics.get("rss"),
                    "runtime_seconds": run.runtime_seconds,
                    "error": run.error,
                }
            )
    return pd.DataFrame.from_records(records)


def runs_to_exposure_frame(runs: list[ExpertRunResult]) -> pd.DataFrame:
    records: list[dict] = []
    for run in runs:
        for sample in run.sample_results:
            for signature_name, exposure in sample.exposures.items():
                records.append(
                    {
                        "expert_name": run.expert_name,
                        "status": run.status,
                        "sample_id": sample.sample_id,
                        "signature_name": signature_name,
                        "exposure": exposure,
                        "score": sample.signature_scores.get(signature_name),
                        "probability": sample.signature_probabilities.get(signature_name),
                        "active": signature_name in sample.active_signatures,
                    }
                )
    return pd.DataFrame.from_records(records)


def write_runs(runs: list[ExpertRunResult], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for run in runs:
        run_path = output_path / f"{run.expert_name}.json"
        run_path.write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False))

    runs_to_summary_frame(runs).to_csv(output_path / "summary.tsv", sep="\t", index=False)
    runs_to_exposure_frame(runs).to_csv(output_path / "exposures.tsv", sep="\t", index=False)
    return output_path
