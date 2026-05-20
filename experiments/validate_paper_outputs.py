#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EXPECTATIONS: list[dict[str, Any]] = [
    {
        "claim_id": "figure_2_catalog_insufficiency",
        "claim_label": "Catalog-insufficiency benchmark",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/catalog_insufficiency_by_group.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_2_catalog_insufficiency",
        "claim_label": "Catalog-insufficiency benchmark",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/catalog_insufficiency_overall_with_uncertainty.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_2_catalog_insufficiency",
        "claim_label": "Catalog-insufficiency benchmark",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/fusion_evidence_by_group.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_2_catalog_insufficiency",
        "claim_label": "Catalog-insufficiency benchmark",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/catalog_assessor_coefficients.tsv",
        "artifact_type": "interpretability_table",
    },
    {
        "claim_id": "figure_3_complete_catalog_support",
        "claim_label": "Complete-catalog support check",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/known_catalog_summary.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_3_complete_catalog_support",
        "claim_label": "Complete-catalog support check",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/known_catalog_overall_with_uncertainty.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_3_complete_catalog_support",
        "claim_label": "Complete-catalog support check",
        "suite": "paper_review_response_sbs96",
        "relative_path": "metrics/per_sample_metrics.tsv",
        "artifact_type": "metric_table",
    },
    {
        "claim_id": "figure_4_calibration",
        "claim_label": "Calibration diagnostics",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/reliability_bins.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_4_calibration",
        "claim_label": "Calibration diagnostics",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/reliability_summary.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_4_calibration",
        "claim_label": "Calibration diagnostics",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/calibration_overall_with_uncertainty.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_5_discovery_packet",
        "claim_label": "Discovery packet case study",
        "suite": "paper_discovery_smoke",
        "relative_path": "tables/discovery_trigger_summary.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_5_discovery_packet",
        "claim_label": "Discovery packet case study",
        "suite": "paper_discovery_smoke",
        "relative_path": "tables/discovery_packet_summary.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_5_discovery_packet",
        "claim_label": "Discovery packet case study",
        "suite": "paper_discovery_smoke",
        "relative_path": "tables/discovery_component_summary.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_5_discovery_packet",
        "claim_label": "Discovery packet case study",
        "suite": "paper_discovery_smoke",
        "relative_path": "tables/discovery_fit_improvements.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_5_discovery_packet",
        "claim_label": "Discovery packet case study",
        "suite": "paper_discovery_smoke",
        "relative_path": "tables/discovery_catalog_hits.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_6_real_data_stress",
        "claim_label": "Public real-data stress test",
        "suite": "paper_real_data_stress_smoke",
        "relative_path": "tables/real_data_sample_summary.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_6_real_data_stress",
        "claim_label": "Public real-data stress test",
        "suite": "paper_real_data_stress_smoke",
        "relative_path": "tables/real_data_recommendation_counts.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_6_real_data_stress",
        "claim_label": "Public real-data stress test",
        "suite": "paper_real_data_stress_smoke",
        "relative_path": "tables/real_data_expert_stability.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_6_real_data_stress",
        "claim_label": "Public real-data stress test",
        "suite": "paper_real_data_stress_smoke",
        "relative_path": "tables/real_data_catalog_stress_delta.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_6_real_data_stress",
        "claim_label": "Public real-data stress test",
        "suite": "paper_real_data_stress_smoke",
        "relative_path": "tables/real_data_stress_design_summary.tsv",
        "artifact_type": "paper_table",
    },
    {
        "claim_id": "figure_6_real_data_stress",
        "claim_label": "Public real-data stress test",
        "suite": "paper_real_data_stress_smoke",
        "relative_path": "tables/public_data_manifest.tsv",
        "artifact_type": "data_manifest",
    },
    {
        "claim_id": "figure_6_real_data_stress",
        "claim_label": "Public real-data stress test",
        "suite": "",
        "relative_path": "public_data/pcawg_sbs96_smoke/sample_manifest.tsv",
        "artifact_type": "data_manifest",
    },
    {
        "claim_id": "figure_6_real_data_stress",
        "claim_label": "Public real-data stress test",
        "suite": "",
        "relative_path": "public_data/pcawg_sbs96_smoke/signature_catalog_variant_manifest.tsv",
        "artifact_type": "data_manifest",
    },
    {
        "claim_id": "figure_6_real_data_stress",
        "claim_label": "Public real-data stress test",
        "suite": "",
        "relative_path": "public_data/pcawg_sbs96_smoke/active_signature_catalog_variant_manifest.tsv",
        "artifact_type": "data_manifest",
    },
    {
        "claim_id": "supplement_dbs78_id83_extension",
        "claim_label": "DBS78/ID83 cross-context extension",
        "suite": "paper_review_response_dbs78_id83",
        "relative_path": "tables/known_catalog_overall_with_uncertainty.tsv",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_dbs78_id83_extension",
        "claim_label": "DBS78/ID83 cross-context extension",
        "suite": "paper_review_response_dbs78_id83",
        "relative_path": "tables/catalog_insufficiency_overall_with_uncertainty.tsv",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_dbs78_id83_extension",
        "claim_label": "DBS78/ID83 cross-context extension",
        "suite": "paper_review_response_dbs78_id83",
        "relative_path": "tables/calibration_overall_with_uncertainty.tsv",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_dbs78_id83_extension",
        "claim_label": "DBS78/ID83 cross-context extension",
        "suite": "paper_review_response_dbs78_id83",
        "relative_path": "tables/headline_metrics.md",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_non_degradation_audit",
        "claim_label": "Integration-layer directional preservation audit",
        "suite": "paper_non_degradation_audit",
        "relative_path": "tables/non_degradation_summary.tsv",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_non_degradation_audit",
        "claim_label": "Integration-layer directional preservation audit",
        "suite": "paper_non_degradation_audit",
        "relative_path": "tables/non_degradation_compact.tsv",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_non_degradation_audit",
        "claim_label": "Integration-layer directional preservation audit",
        "suite": "paper_non_degradation_audit",
        "relative_path": "tables/non_degradation_headline.md",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_sensitivity_checks",
        "claim_label": "Sensitivity checks for operating labels and thresholds",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/sensitivity_checks.tsv",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_catalog_robustness",
        "claim_label": "Blocked catalog-insufficiency robustness checks",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/catalog_insufficiency_robustness.tsv",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_catalog_robustness",
        "claim_label": "Blocked catalog-insufficiency robustness checks",
        "suite": "paper_review_response_sbs96",
        "relative_path": "tables/catalog_insufficiency_robustness_summary.tsv",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_ablation",
        "claim_label": "Expert ablations",
        "suite": "paper_known_catalog_ablation_smoke",
        "relative_path": "tables/ablation_summary.tsv",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_ablation",
        "claim_label": "Expert ablations",
        "suite": "paper_catalog_insufficiency_ablation_smoke",
        "relative_path": "tables/ablation_summary.tsv",
        "artifact_type": "supplement_table",
    },
    {
        "claim_id": "supplement_removal_design",
        "claim_label": "Removal-design manifest diagnostics",
        "suite": "",
        "relative_path": "removal_manifests/sbs96.tsv",
        "artifact_type": "manifest",
    },
    {
        "claim_id": "supplement_removal_design",
        "claim_label": "Removal-design manifest diagnostics",
        "suite": "",
        "relative_path": "removal_manifests/dbs78.tsv",
        "artifact_type": "manifest",
    },
    {
        "claim_id": "supplement_removal_design",
        "claim_label": "Removal-design manifest diagnostics",
        "suite": "",
        "relative_path": "removal_manifests/id83.tsv",
        "artifact_type": "manifest",
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".tsv":
        return None
    try:
        return int(len(pd.read_csv(path, sep="\t")))
    except Exception:
        return None


def _artifact_path(root: Path, expectation: dict[str, Any]) -> Path:
    suite = str(expectation.get("suite") or "")
    relative_path = Path(str(expectation["relative_path"]))
    return root / suite / relative_path if suite else root / relative_path


def _artifact_rows(root: Path, expectations: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for expectation in expectations:
        path = _artifact_path(root, expectation)
        exists = path.exists() and path.is_file()
        row_count = _row_count(path)
        rows.append(
            {
                "claim_id": expectation["claim_id"],
                "claim_label": expectation["claim_label"],
                "suite": expectation.get("suite") or "",
                "relative_path": str(path.relative_to(root)) if path.exists() else str((Path(expectation.get("suite") or "") / expectation["relative_path"])),
                "artifact_type": expectation.get("artifact_type") or "artifact",
                "required": bool(expectation.get("required", True)),
                "exists": exists,
                "row_count": row_count,
                "nonempty": bool(exists and (row_count is None or row_count > 0)),
                "status": "ok" if exists and (row_count is None or row_count > 0) else "missing_or_empty",
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame.from_records(rows)


def _suite_rows(root: Path) -> pd.DataFrame:
    rows = []
    for manifest_path in sorted(root.glob("*/manifests/suite.manifest.json")):
        suite = manifest_path.parents[1].name
        try:
            manifests = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            rows.append(
                {
                    "suite": suite,
                    "step_name": None,
                    "command_name": None,
                    "status": "manifest_decode_error",
                    "returncode": None,
                    "manifest_path": str(manifest_path.relative_to(root)),
                }
            )
            continue
        for step in manifests:
            rows.append(
                {
                    "suite": suite,
                    "step_name": step.get("step_name"),
                    "command_name": step.get("command_name"),
                    "status": step.get("status"),
                    "returncode": step.get("returncode"),
                    "started_at": step.get("started_at"),
                    "completed_at": step.get("completed_at"),
                    "manifest_path": str(manifest_path.relative_to(root)),
                    "log_path": step.get("log_path"),
                }
            )
    return pd.DataFrame.from_records(rows)


def _claim_rows(artifact_inventory: pd.DataFrame) -> pd.DataFrame:
    if artifact_inventory.empty:
        return pd.DataFrame()
    required = artifact_inventory.loc[artifact_inventory["required"]].copy()
    if required.empty:
        return pd.DataFrame()
    summary = (
        required.groupby(["claim_id", "claim_label"], dropna=False)
        .agg(
            required_artifacts=("relative_path", "size"),
            ok_artifacts=("status", lambda values: int((values == "ok").sum())),
            missing_artifacts=("status", lambda values: int((values != "ok").sum())),
        )
        .reset_index()
    )
    summary["status"] = summary.apply(
        lambda row: "ok" if int(row["missing_artifacts"]) == 0 else "incomplete",
        axis=1,
    )
    return summary


def _write_report(
    *,
    output_dir: Path,
    root: Path,
    artifact_inventory: pd.DataFrame,
    suite_status: pd.DataFrame,
    claim_status: pd.DataFrame,
) -> None:
    missing = artifact_inventory.loc[artifact_inventory["status"] != "ok"].copy()
    failed_steps = (
        suite_status.loc[suite_status["status"] != "success"].copy()
        if not suite_status.empty and "status" in suite_status.columns
        else pd.DataFrame()
    )
    lines = [
        "# Paper Output Readiness Report",
        "",
        f"- Generated at: `{_utc_now()}`",
        f"- Results root: `{root}`",
        f"- Claims checked: `{len(claim_status)}`",
        f"- Required artifacts checked: `{len(artifact_inventory)}`",
        f"- Missing or empty artifacts: `{len(missing)}`",
        f"- Failed or incomplete suite steps: `{len(failed_steps)}`",
        "",
        "## Claim Status",
        "",
    ]
    if claim_status.empty:
        lines.append("No claims were checked.")
    else:
        for _, row in claim_status.iterrows():
            lines.append(
                f"- `{row['claim_id']}`: {row['status']} "
                f"({int(row['ok_artifacts'])}/{int(row['required_artifacts'])} artifacts)"
            )
    lines.extend(["", "## Missing Artifacts", ""])
    if missing.empty:
        lines.append("None.")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- `{row['claim_id']}` -> `{row['relative_path']}`")
    lines.extend(["", "## Suite Step Issues", ""])
    if failed_steps.empty:
        lines.append("None.")
    else:
        for _, row in failed_steps.iterrows():
            lines.append(
                f"- `{row.get('suite')}` / `{row.get('step_name')}`: "
                f"{row.get('status')} returncode={row.get('returncode')}"
            )
    (output_dir / "paper_readiness_report.md").write_text("\n".join(lines) + "\n")


def validate_paper_outputs(
    root: Path,
    output_dir: Path,
    *,
    expectations: list[dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    expectations = expectations or DEFAULT_EXPECTATIONS
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_inventory = _artifact_rows(root, expectations)
    suite_status = _suite_rows(root)
    claim_status = _claim_rows(artifact_inventory)

    artifact_inventory.to_csv(output_dir / "paper_output_inventory.tsv", sep="\t", index=False)
    suite_status.to_csv(output_dir / "paper_suite_status.tsv", sep="\t", index=False)
    claim_status.to_csv(output_dir / "paper_claim_status.tsv", sep="\t", index=False)
    _write_report(
        output_dir=output_dir,
        root=root,
        artifact_inventory=artifact_inventory,
        suite_status=suite_status,
        claim_status=claim_status,
    )
    return {
        "artifact_inventory": artifact_inventory,
        "suite_status": suite_status,
        "claim_status": claim_status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate paper-suite outputs against manuscript claim artifacts.")
    parser.add_argument("--root", default="results/paper", help="Paper results root.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/paper_readiness.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "paper_readiness"
    validate_paper_outputs(root, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
