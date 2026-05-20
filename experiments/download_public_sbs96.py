#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd


DEFAULT_URL = (
    "https://raw.githubusercontent.com/steverozen/PCAWG7.WGS.Spectra/"
    "2e8a1a16f7b3b761593002f3c72c29dd278a2da3/"
    "data-raw/spectra/WGS_PCAWG.96.csv"
)
DEFAULT_SOURCE_LABEL = (
    "PCAWG7.WGS.Spectra@2e8a1a16f7b3b761593002f3c72c29dd278a2da3:"
    "data-raw/spectra/WGS_PCAWG.96.csv"
)
DEFAULT_SOURCE_REFERENCE = (
    "PCAWG7 WGS SBS96 mutational spectra mirrored in steverozen/PCAWG7.WGS.Spectra; "
    "derived from the PCAWG mutational signature resource."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_or_download(source: str, destination: Path, *, timeout_seconds: int = 120, force: bool = False) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        with urlopen(source, timeout=timeout_seconds) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
        return
    local_path = Path(parsed.path if parsed.scheme == "file" else source)
    if not local_path.exists():
        raise FileNotFoundError(f"Raw SBS96 source not found: {source}")
    shutil.copy2(local_path, destination)


def _sample_columns(frame: pd.DataFrame) -> list[str]:
    metadata = {"Mutation.type", "Mutation type", "Trinucleotide"}
    columns = []
    for column in frame.columns:
        if column in metadata:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().all():
            columns.append(column)
    if not columns:
        raise ValueError("No numeric sample columns found in the SBS96 catalog.")
    return columns


def _tumor_type(sample_id: str) -> str:
    return str(sample_id).split("::", 1)[0] if "::" in str(sample_id) else "unknown"


def _select_samples(
    counts: pd.DataFrame,
    *,
    max_samples: int | None,
    selection_mode: str,
    tumor_types: set[str] | None = None,
) -> list[str]:
    sample_columns = list(counts.columns)
    if tumor_types:
        sample_columns = [sample for sample in sample_columns if _tumor_type(sample) in tumor_types]
    if not sample_columns:
        raise ValueError("No samples remain after applying tumor-type filters.")
    if max_samples is None or max_samples <= 0 or max_samples >= len(sample_columns):
        return sample_columns

    burdens = counts.loc[:, sample_columns].sum(axis=0).sort_values(ascending=False)
    if selection_mode == "first":
        return sample_columns[:max_samples]
    if selection_mode == "highest-burden":
        return burdens.head(max_samples).index.tolist()
    if selection_mode != "balanced-tumor-types":
        raise ValueError(f"Unsupported selection mode: {selection_mode}")

    by_type: dict[str, list[str]] = {}
    for sample_id in burdens.index:
        by_type.setdefault(_tumor_type(sample_id), []).append(sample_id)
    type_order = (
        pd.Series({tumor_type: float(burdens.loc[samples].max()) for tumor_type, samples in by_type.items()})
        .sort_values(ascending=False)
        .index.tolist()
    )
    selected: list[str] = []
    offset = 0
    while len(selected) < max_samples:
        added = False
        for tumor_type in type_order:
            samples = by_type[tumor_type]
            if offset < len(samples):
                selected.append(samples[offset])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        offset += 1
    return selected


def normalize_public_sbs96_catalog(
    raw_path: Path,
    output_dir: Path,
    *,
    source_url: str,
    source_label: str = DEFAULT_SOURCE_LABEL,
    source_reference: str = DEFAULT_SOURCE_REFERENCE,
    max_samples: int | None = None,
    selection_mode: str = "balanced-tumor-types",
    tumor_types: set[str] | None = None,
    expected_context_count: int = 96,
) -> dict[str, Any]:
    frame = pd.read_csv(raw_path)
    if "Mutation.type" not in frame.columns and "Mutation type" in frame.columns:
        frame = frame.rename(columns={"Mutation type": "Mutation.type"})
    required = {"Mutation.type", "Trinucleotide"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required SBS96 metadata columns: {missing}")
    if len(frame) != expected_context_count:
        raise ValueError(f"Expected {expected_context_count} SBS96 rows, found {len(frame)} in {raw_path}.")

    sample_columns = _sample_columns(frame)
    numeric = frame.loc[:, sample_columns].apply(pd.to_numeric, errors="raise")
    selected_samples = _select_samples(
        numeric,
        max_samples=max_samples,
        selection_mode=selection_mode,
        tumor_types=tumor_types,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "samples_sbs96.csv"
    normalized = pd.concat(
        [
            frame.loc[:, ["Mutation.type", "Trinucleotide"]].astype(str),
            numeric.loc[:, selected_samples].astype(int),
        ],
        axis=1,
    )
    normalized.to_csv(sample_path, index=False)

    burden = numeric.loc[:, selected_samples].sum(axis=0)
    sample_manifest = pd.DataFrame(
        [
            {
                "selected_rank": rank,
                "sample_id": sample_id,
                "tumor_type": _tumor_type(sample_id),
                "mutation_count": int(burden.loc[sample_id]),
            }
            for rank, sample_id in enumerate(selected_samples, start=1)
        ]
    )
    sample_manifest_path = output_dir / "sample_manifest.tsv"
    sample_manifest.to_csv(sample_manifest_path, sep="\t", index=False)

    public_manifest = pd.DataFrame(
        [
            {
                "source_label": source_label,
                "source_url": source_url,
                "source_reference": source_reference,
                "raw_path": str(raw_path),
                "sample_source": str(sample_path),
                "sample_manifest": str(sample_manifest_path),
                "raw_sha256": sha256(raw_path),
                "sample_source_sha256": sha256(sample_path),
                "n_contexts": int(len(normalized)),
                "n_selected_samples": int(len(selected_samples)),
                "selection_mode": selection_mode,
                "tumor_type_filter": ",".join(sorted(tumor_types)) if tumor_types else "",
                "total_selected_mutations": int(burden.sum()),
                "generated_at": _utc_now(),
            }
        ]
    )
    public_manifest_path = output_dir / "public_data_manifest.tsv"
    public_manifest.to_csv(public_manifest_path, sep="\t", index=False)

    source_manifest = {
        "source_label": source_label,
        "source_url": source_url,
        "source_reference": source_reference,
        "raw_path": str(raw_path),
        "sample_source": str(sample_path),
        "sample_manifest": str(sample_manifest_path),
        "public_data_manifest": str(public_manifest_path),
        "raw_sha256": sha256(raw_path),
        "sample_source_sha256": sha256(sample_path),
        "n_contexts": int(len(normalized)),
        "n_selected_samples": int(len(selected_samples)),
        "selection_mode": selection_mode,
        "tumor_type_filter": ",".join(sorted(tumor_types)) if tumor_types else "",
        "selected_samples": selected_samples,
        "generated_at": _utc_now(),
    }
    (output_dir / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False))
    return source_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and normalize a public SBS96 real-data catalog.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Public SBS96 CSV URL.")
    parser.add_argument("--raw-input", default=None, help="Optional local raw CSV to normalize instead of downloading.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-filename", default="WGS_PCAWG.96.csv")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument(
        "--selection-mode",
        default="balanced-tumor-types",
        choices=["balanced-tumor-types", "highest-burden", "first"],
    )
    parser.add_argument("--tumor-types", default=None, help="Optional comma-separated tumor-type prefixes.")
    parser.add_argument("--source-label", default=DEFAULT_SOURCE_LABEL)
    parser.add_argument("--source-reference", default=DEFAULT_SOURCE_REFERENCE)
    parser.add_argument("--force", action="store_true", help="Re-download or overwrite raw CSV.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_path = raw_dir / args.raw_filename
    source = args.raw_input or args.url
    _copy_or_download(source, raw_path, force=args.force)
    tumor_types = (
        {item.strip() for item in args.tumor_types.split(",") if item.strip()}
        if args.tumor_types
        else None
    )
    normalize_public_sbs96_catalog(
        raw_path,
        output_dir,
        source_url=args.url,
        source_label=args.source_label,
        source_reference=args.source_reference,
        max_samples=args.max_samples,
        selection_mode=args.selection_mode,
        tumor_types=tumor_types,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
