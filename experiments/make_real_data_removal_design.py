#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if pd.to_numeric(frame[column], errors="coerce").notna().all()
    ]


def _load_fused_exposure_rows(decision_dir: Path) -> pd.DataFrame:
    path = decision_dir / "fusion" / "fused_run.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing fused run JSON: {path}")
    data = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for sample_result in data.get("sample_results") or []:
        sample_id = sample_result.get("sample_id")
        for signature_name, exposure in (sample_result.get("exposures") or {}).items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "signature_name": signature_name,
                    "fused_exposure": float(exposure or 0.0),
                }
            )
    if not rows:
        raise ValueError(f"No fused exposure rows found in {path}.")
    return pd.DataFrame.from_records(rows)


def build_real_data_removal_design(
    decision_dir: Path,
    *,
    top_n: int = 4,
    exclude_signatures: set[str] | None = None,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    exclude_signatures = exclude_signatures or set()
    exposures = _load_fused_exposure_rows(decision_dir)
    exposures = exposures.loc[~exposures["signature_name"].isin(exclude_signatures)].copy()
    if exposures.empty:
        raise ValueError("No signatures remain after exclusions.")
    grouped = (
        exposures.groupby("signature_name", dropna=False)
        .agg(
            total_fused_exposure=("fused_exposure", "sum"),
            active_sample_count=("fused_exposure", lambda values: int((values > 0.0).sum())),
            max_sample_exposure=("fused_exposure", "max"),
            mean_sample_exposure=("fused_exposure", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["total_fused_exposure", "active_sample_count", "max_sample_exposure", "signature_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    grouped.insert(0, "removal_rank", range(1, len(grouped) + 1))
    grouped["selected_for_removal"] = grouped["removal_rank"] <= top_n
    grouped["selection_rule"] = f"top_{top_n}_total_fused_exposure"
    return grouped


def write_real_data_removal_design(
    *,
    decision_dir: Path,
    signature_source: Path,
    output_path: Path,
    manifest_path: Path,
    top_n: int = 4,
    exclude_signatures: set[str] | None = None,
) -> pd.DataFrame:
    design = build_real_data_removal_design(
        decision_dir,
        top_n=top_n,
        exclude_signatures=exclude_signatures,
    )
    selected = design.loc[design["selected_for_removal"], "signature_name"].tolist()
    signature_frame = pd.read_csv(signature_source)
    signature_columns = _numeric_columns(signature_frame)
    missing = sorted(set(selected) - set(signature_columns))
    if missing:
        raise ValueError(f"Selected signatures are missing from signature catalog: {missing}")
    variant = signature_frame.loc[:, [column for column in signature_frame.columns if column not in set(selected)]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variant.to_csv(output_path, index=False)

    manifest = design.copy()
    manifest.insert(0, "decision_dir", str(decision_dir))
    manifest.insert(1, "signature_source", str(signature_source))
    manifest.insert(2, "output_path", str(output_path))
    manifest["source_sha256"] = _sha256(signature_source)
    manifest["output_sha256"] = _sha256(output_path)
    manifest["top_n"] = int(top_n)
    manifest["removed_signatures"] = ",".join(selected)
    manifest["n_source_signatures"] = int(len(signature_columns))
    manifest["n_remaining_signatures"] = int(len(signature_columns) - len(selected))
    manifest["generated_at"] = _utc_now()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive a real-data active-signature removal catalog from full-catalog decision outputs.")
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--signature-source", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--exclude-signatures", default=None, help="Optional comma-separated signatures to exclude from data-driven removal.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    exclude_signatures = (
        {item.strip() for item in args.exclude_signatures.split(",") if item.strip()}
        if args.exclude_signatures
        else None
    )
    write_real_data_removal_design(
        decision_dir=Path(args.decision_dir),
        signature_source=Path(args.signature_source),
        output_path=Path(args.output_path),
        manifest_path=Path(args.manifest_path),
        top_n=args.top_n,
        exclude_signatures=exclude_signatures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
