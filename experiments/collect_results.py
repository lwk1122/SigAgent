#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RESULT_FILES = {
    "aggregate_metrics": "aggregate_metrics.tsv",
    "per_sample_metrics": "per_sample_metrics.tsv",
    "assignment_group_metrics": "assignment_group_metrics.tsv",
    "catalog_probability_group_metrics": "catalog_probability_group_metrics.tsv",
    "interval_group_metrics": "interval_group_metrics.tsv",
    "fusion_evidence_features": "fusion_evidence_features.tsv",
    "discovery_trigger_candidates": "trigger_candidates.tsv",
    "discovery_recurrence_clusters": "recurrence_clusters.tsv",
    "discovery_packets": "packets.tsv",
    "discovery_components": "extracted_components.tsv",
    "discovery_fit_improvements": "fit_improvements.tsv",
}


def _read_result_frame(path: Path, *, table_name: str, step_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame.insert(0, "table_name", table_name)
    frame.insert(1, "step_name", step_name)
    frame.insert(2, "source_path", str(path))
    return frame


def collect_results(root: Path, output_dir: Path) -> None:
    raw_root = root / "raw"
    if not raw_root.exists():
        raise SystemExit(f"Missing raw results directory: {raw_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_by_table: dict[str, list[pd.DataFrame]] = {name: [] for name in RESULT_FILES}
    for step_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        for table_name, filename in RESULT_FILES.items():
            path = step_dir / filename
            if path.exists():
                frames_by_table[table_name].append(
                    _read_result_frame(path, table_name=table_name, step_name=step_dir.name)
                )

    for table_name, frames in frames_by_table.items():
        if not frames:
            continue
        pd.concat(frames, ignore_index=True).to_csv(output_dir / f"{table_name}.tsv", sep="\t", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect paper-suite benchmark TSV outputs.")
    parser.add_argument("root", help="Paper-suite output root.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/metrics.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "metrics"
    collect_results(root, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
