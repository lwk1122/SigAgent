#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_optional_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _step_dirs(root: Path) -> list[Path]:
    raw_root = root / "raw"
    if not raw_root.exists():
        return []
    return sorted(path for path in raw_root.iterdir() if path.is_dir())


def _with_step(frame: pd.DataFrame, *, step_dir: Path, table_name: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    working.insert(0, "table_name", table_name)
    working.insert(1, "step_name", step_dir.name)
    working.insert(2, "source_path", str(step_dir))
    return working


def _collect_step_tsv(root: Path, filename: str, table_name: str) -> pd.DataFrame:
    frames = []
    for step_dir in _step_dirs(root):
        frame = _read_optional_tsv(step_dir / filename)
        if not frame.empty:
            frames.append(_with_step(frame, step_dir=step_dir, table_name=table_name))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _flatten_catalog_hits(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step_dir in _step_dirs(root):
        packets_path = step_dir / "packets.json"
        if not packets_path.exists():
            continue
        packets = json.loads(packets_path.read_text())
        for packet in packets:
            packet_id = packet.get("packet_id")
            for match in packet.get("catalog_match_summary") or []:
                component_id = match.get("component_id")
                for rank, hit in enumerate(match.get("top_hits") or [], start=1):
                    rows.append(
                        {
                            "table_name": "discovery_catalog_hits",
                            "step_name": step_dir.name,
                            "source_path": str(packets_path),
                            "packet_id": packet_id,
                            "component_id": component_id,
                            "hit_rank": rank,
                            "signature_name": hit.get("signature_name"),
                            "cosine": hit.get("cosine"),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def _trigger_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    numeric_columns = [
        "priority_score",
        "catalog_insufficiency_probability",
        "residual_structure_score",
        "mutation_count",
        "recurrence_count",
    ]
    available_numeric = [column for column in numeric_columns if column in candidates.columns]
    for column in available_numeric:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    group_columns = [
        column
        for column in ["step_name", "mutation_type", "trigger_status", "review_gate_status"]
        if column in candidates.columns
    ]
    if not group_columns:
        return pd.DataFrame()
    summary = candidates.groupby(group_columns, dropna=False).size().rename("n_candidates").reset_index()
    if available_numeric:
        numeric_summary = (
            candidates.groupby(group_columns, dropna=False)[available_numeric]
            .mean(numeric_only=True)
            .reset_index()
        )
        summary = summary.merge(numeric_summary, on=group_columns, how="left")
    return summary


def make_discovery_tables(root: Path, output_dir: Path) -> None:
    candidates = _collect_step_tsv(root, "trigger_candidates.tsv", "discovery_trigger_candidates")
    clusters = _collect_step_tsv(root, "recurrence_clusters.tsv", "discovery_recurrence_clusters")
    packets = _collect_step_tsv(root, "packets.tsv", "discovery_packets")
    components = _collect_step_tsv(root, "extracted_components.tsv", "discovery_components")
    fit_improvements = _collect_step_tsv(root, "fit_improvements.tsv", "discovery_fit_improvements")
    catalog_hits = _flatten_catalog_hits(root)
    trigger_summary = _trigger_summary(candidates)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "discovery_trigger_candidates.tsv": candidates,
        "discovery_trigger_summary.tsv": trigger_summary,
        "discovery_recurrence_clusters.tsv": clusters,
        "discovery_packet_summary.tsv": packets,
        "discovery_component_summary.tsv": components,
        "discovery_fit_improvements.tsv": fit_improvements,
        "discovery_catalog_hits.tsv": catalog_hits,
    }
    for filename, frame in outputs.items():
        if not frame.empty:
            frame.to_csv(output_dir / filename, sep="\t", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paper-ready discovery packet tables.")
    parser.add_argument("root", help="Paper-suite output root.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/tables.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "tables"
    make_discovery_tables(root, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
