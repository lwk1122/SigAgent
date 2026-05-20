#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signature_decision.removal_design import (  # noqa: E402
    RemovalDesignConfig,
    build_catalog_removal_design_from_files,
    write_catalog_removal_design,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper catalog-removal design manifests.")
    parser.add_argument("--signature-source", required=True)
    parser.add_argument("--exposure-source", required=True)
    parser.add_argument("--mutation-type", required=True, choices=["SBS96", "DBS78", "ID83"])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--n-per-group", type=int, default=5)
    parser.add_argument("--active-threshold", type=float, default=0.0)
    parser.add_argument("--exclude-unbenchmarkable-controls", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    design = build_catalog_removal_design_from_files(
        signature_source=args.signature_source,
        exposure_source=args.exposure_source,
        mutation_type=args.mutation_type,
        config=RemovalDesignConfig(
            n_per_group=args.n_per_group,
            active_threshold=args.active_threshold,
            include_unbenchmarkable_controls=not args.exclude_unbenchmarkable_controls,
        ),
    )
    write_catalog_removal_design(design, args.output_path)
    print(args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

