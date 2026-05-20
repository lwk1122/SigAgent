#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SBS_SUBSTITUTIONS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
BASES = ["A", "C", "G", "T"]
SIGNATURE_NAMES = ["toy_sbs_a", "toy_sbs_b", "toy_sbs_c", "toy_sbs_d"]
SAMPLE_IDS = [
    "toy_sample_01",
    "toy_sample_02",
    "toy_sample_03",
    "toy_sample_04",
    "toy_sample_05",
    "toy_sample_06",
]


def _sbs96_metadata() -> pd.DataFrame:
    rows = []
    for substitution in SBS_SUBSTITUTIONS:
        central_base = substitution[0]
        for left in BASES:
            for right in BASES:
                rows.append(
                    {
                        "Mutation.type": substitution,
                        "Trinucleotide": f"{left}{central_base}{right}",
                    }
                )
    return pd.DataFrame.from_records(rows)


def _normalized(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("Cannot normalize an empty profile.")
    return values / total


def _signature_matrix(metadata: pd.DataFrame) -> pd.DataFrame:
    n_channels = len(metadata)
    profiles: dict[str, np.ndarray] = {}
    for signature_name in SIGNATURE_NAMES:
        profiles[signature_name] = np.full(n_channels, 0.01, dtype=float)

    for idx, row in metadata.iterrows():
        substitution = str(row["Mutation.type"])
        trinucleotide = str(row["Trinucleotide"])
        left, _, right = trinucleotide
        if substitution == "C>T" and right == "G":
            profiles["toy_sbs_a"][idx] += 1.00
        if substitution == "C>A" and left == "C":
            profiles["toy_sbs_b"][idx] += 0.95
        if substitution == "T>C" and left in {"A", "T"}:
            profiles["toy_sbs_c"][idx] += 0.85
        if substitution in {"C>G", "T>A"}:
            profiles["toy_sbs_d"][idx] += 0.20

    signature_frame = metadata.copy()
    for signature_name, profile in profiles.items():
        signature_frame[signature_name] = _normalized(profile)
    return signature_frame


def _truth_exposures() -> pd.DataFrame:
    values = {
        "toy_sample_01": [0.80, 0.20, 0.00, 0.00],
        "toy_sample_02": [0.10, 0.75, 0.15, 0.00],
        "toy_sample_03": [0.00, 0.10, 0.80, 0.10],
        "toy_sample_04": [0.25, 0.25, 0.25, 0.25],
        "toy_sample_05": [0.00, 0.00, 0.10, 0.90],
        "toy_sample_06": [0.45, 0.00, 0.45, 0.10],
    }
    return pd.DataFrame(values, index=SIGNATURE_NAMES, dtype=float)


def _sample_matrix(
    signature_frame: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    burden: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    signature_values = signature_frame.loc[:, SIGNATURE_NAMES].to_numpy(dtype=float)
    exposure_values = exposures.loc[SIGNATURE_NAMES, SAMPLE_IDS].to_numpy(dtype=float)
    exposure_values = exposure_values / np.sum(exposure_values, axis=0, keepdims=True)
    profiles = signature_values @ exposure_values
    profiles = profiles / np.sum(profiles, axis=0, keepdims=True)

    sample_frame = signature_frame.loc[:, ["Mutation.type", "Trinucleotide"]].copy()
    for sample_index, sample_id in enumerate(SAMPLE_IDS):
        sample_frame[sample_id] = rng.multinomial(burden, profiles[:, sample_index])
    return sample_frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_release_smoke_fixtures(
    output_dir: Path,
    *,
    burden: int = 500,
    random_seed: int = 20260509,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _sbs96_metadata()
    signatures = _signature_matrix(metadata)
    exposures = _truth_exposures()
    samples = _sample_matrix(signatures, exposures, burden=burden, random_seed=random_seed)

    signature_path = output_dir / "toy_sbs96_signatures.csv"
    exposure_path = output_dir / "toy_sbs96_exposures.csv"
    sample_path = output_dir / "toy_sbs96_samples.csv"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    signatures.to_csv(signature_path, index=False)
    exposures.to_csv(exposure_path, index_label="signature")
    samples.to_csv(sample_path, index=False)

    manifest: dict[str, object] = {
        "fixture_name": "toy_sbs96_release_smoke",
        "mutation_type": "SBS96",
        "generator": "experiments/make_release_smoke_fixtures.py",
        "purpose": "Deterministic toy fixture for reviewer smoke tests; not biological evidence.",
        "random_seed": random_seed,
        "burden_per_sample": burden,
        "n_channels": int(len(metadata)),
        "n_signatures": int(len(SIGNATURE_NAMES)),
        "n_samples": int(len(SAMPLE_IDS)),
        "files": {
            "signature_source": {
                "path": signature_path.name,
                "sha256": _sha256(signature_path),
            },
            "exposure_source": {
                "path": exposure_path.name,
                "sha256": _sha256(exposure_path),
            },
            "sample_source": {
                "path": sample_path.name,
                "sha256": _sha256(sample_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    readme_path.write_text(
        "# Release Smoke Fixtures\n\n"
        "These deterministic SBS96 toy fixtures are generated by "
        "`experiments/make_release_smoke_fixtures.py`.\n\n"
        "They are intended only for installation checks, reviewer smoke tests, and "
        "pipeline reproducibility checks. They are not biological evidence and should "
        "not be used for manuscript performance claims.\n",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic toy SBS96 release-smoke fixtures.")
    parser.add_argument("--output-dir", default="results/paper/release_smoke/fixtures")
    parser.add_argument("--burden", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=20260509)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    make_release_smoke_fixtures(
        Path(args.output_dir),
        burden=args.burden,
        random_seed=args.random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
