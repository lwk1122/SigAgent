#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

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


def make_signature_catalog_variant(
    signature_source: Path,
    output_path: Path,
    *,
    remove_signatures: list[str],
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(signature_source)
    signature_columns = _numeric_columns(frame)
    missing = sorted(set(remove_signatures) - set(signature_columns))
    if missing:
        raise ValueError(f"Cannot remove signatures not present in catalog: {missing}")
    kept_columns = [column for column in frame.columns if column not in set(remove_signatures)]
    variant = frame.loc[:, kept_columns].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variant.to_csv(output_path, index=False)

    manifest = pd.DataFrame(
        [
            {
                "signature_source": str(signature_source),
                "output_path": str(output_path),
                "source_sha256": _sha256(signature_source),
                "output_sha256": _sha256(output_path),
                "removed_signatures": ",".join(remove_signatures),
                "n_removed_signatures": int(len(remove_signatures)),
                "n_source_signatures": int(len(signature_columns)),
                "n_remaining_signatures": int(len(signature_columns) - len(remove_signatures)),
                "generated_at": _utc_now(),
            }
        ]
    )
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, sep="\t", index=False)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a reference signature catalog variant by removing signatures.")
    parser.add_argument("--signature-source", required=True)
    parser.add_argument("--remove-signatures", required=True, help="Comma-separated signature names.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--manifest-path", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    remove_signatures = [item.strip() for item in args.remove_signatures.split(",") if item.strip()]
    if not remove_signatures:
        raise SystemExit("--remove-signatures must name at least one signature.")
    make_signature_catalog_variant(
        Path(args.signature_source),
        Path(args.output_path),
        remove_signatures=remove_signatures,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
