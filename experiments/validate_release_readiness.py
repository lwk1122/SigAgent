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


def _status(blocked: bool = False, warning: bool = False) -> str:
    if blocked:
        return "blocker"
    if warning:
        return "warning"
    return "ok"


def _file_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_contains(path: Path, needle: str) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return needle in path.read_text(errors="replace")


def _tsv_nonempty(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return len(pd.read_csv(path, sep="\t")) > 0
    except Exception:
        return False


def _suite_manifest_success(path: Path) -> tuple[bool, str]:
    if not path.exists() or not path.is_file():
        return False, "missing suite manifest"
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False, "suite manifest is not valid JSON"
    if not isinstance(payload, list) or not payload:
        return False, "suite manifest has no steps"
    failures = [
        f"{step.get('step_name')}={step.get('status')}"
        for step in payload
        if step.get("status") != "success" or step.get("returncode") not in {0, "0"}
    ]
    if failures:
        return False, "; ".join(failures)
    return True, f"{len(payload)} successful steps"


def _release_fixture_manifest_ok(manifest_path: Path) -> tuple[bool, str]:
    if not manifest_path.exists() or not manifest_path.is_file():
        return False, "missing fixture manifest"
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False, "fixture manifest is not valid JSON"
    files = manifest.get("files") or {}
    if not isinstance(files, dict) or not files:
        return False, "fixture manifest has no file checksums"
    errors = []
    for label, record in files.items():
        relative = record.get("path") if isinstance(record, dict) else None
        expected = record.get("sha256") if isinstance(record, dict) else None
        if not relative or not expected:
            errors.append(f"{label}: missing path or sha256")
            continue
        path = manifest_path.parent / str(relative)
        observed = _sha256(path)
        if observed != expected:
            errors.append(f"{label}: checksum mismatch")
    if errors:
        return False, "; ".join(errors)
    return True, f"{len(files)} fixture files verified"


def _paper_readiness_ok(readiness_dir: Path) -> tuple[bool, str]:
    inventory_path = readiness_dir / "paper_output_inventory.tsv"
    suite_path = readiness_dir / "paper_suite_status.tsv"
    if not inventory_path.exists() or not suite_path.exists():
        return False, "paper readiness TSV files missing"
    try:
        inventory = pd.read_csv(inventory_path, sep="\t")
        suite = pd.read_csv(suite_path, sep="\t")
    except Exception as exc:
        return False, f"could not parse readiness TSV files: {exc}"
    missing = inventory.loc[inventory["status"].ne("ok")] if "status" in inventory.columns else inventory
    failed = suite.loc[suite["status"].ne("success")] if "status" in suite.columns else suite
    if len(missing) or len(failed):
        return False, f"{len(missing)} missing artifacts; {len(failed)} failed suite steps"
    return True, f"{len(inventory)} artifacts checked; {len(suite)} suite steps successful"


def _add_check(
    rows: list[dict[str, Any]],
    *,
    check_id: str,
    category: str,
    status: str,
    evidence: str,
    required_action: str = "",
    required_for_submission: bool = True,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "category": category,
            "status": status,
            "required_for_submission": required_for_submission,
            "evidence": evidence,
            "required_action": required_action,
        }
    )


def _check_required_files(repo_root: Path, rows: list[dict[str, Any]], *, release_stage: str) -> None:
    required_files = [
        ("readme", "README.md", "Keep top-level project orientation current."),
        ("reviewer_quickstart", "REVIEWER_QUICKSTART.md", "Keep reviewer quickstart aligned with release commands."),
        ("requirements", "requirements.txt", "Keep core dependency file installable."),
        ("requirements_optional", "requirements-optional.txt", "Keep optional wrapper dependencies documented."),
        ("third_party_notices", "THIRD_PARTY_NOTICES.md", "Finalize after license decisions."),
        ("availability_statement", "paper/data_code_availability_statement.md", "Replace placeholders before submission."),
        ("license_decision_note", "paper/license_decision_needed.md", "Replace with actual license decision before release."),
    ]
    for check_id, relative_path, action in required_files:
        path = repo_root / relative_path
        _add_check(
            rows,
            check_id=check_id,
            category="release_files",
            status=_status(blocked=not _file_nonempty(path)),
            evidence=f"{relative_path} {'exists' if _file_nonempty(path) else 'missing_or_empty'}",
            required_action=action if _file_nonempty(path) else f"Add {relative_path}.",
        )

    license_path = repo_root / "LICENSE"
    license_note = repo_root / "paper" / "license_decision_needed.md"
    license_blocked = not _file_nonempty(license_path) and release_stage == "public_submission"
    license_warning = not _file_nonempty(license_path) and release_stage != "public_submission"
    _add_check(
        rows,
        check_id="top_level_license",
        category="licensing",
        status=_status(blocked=license_blocked, warning=license_warning),
        evidence=(
            "LICENSE exists"
            if _file_nonempty(license_path)
            else f"No top-level LICENSE found; decision note exists={_file_nonempty(license_note)}."
        ),
        required_action=(
            "Add an author/institution-approved top-level LICENSE for original SigAgent code."
            if release_stage == "public_submission"
            else "Resolve before public archive; keep reviewer package marked as not publicly licensed."
        ),
    )


def _check_availability_placeholders(repo_root: Path, rows: list[dict[str, Any]], *, release_stage: str) -> None:
    availability_path = repo_root / "paper" / "data_code_availability_statement.md"
    has_placeholders = any(
        _text_contains(availability_path, needle)
        for needle in ["[GitHub repository URL]", "[Zenodo DOI]"]
    )
    _add_check(
        rows,
        check_id="archive_placeholders",
        category="archive",
        status=_status(blocked=has_placeholders and release_stage == "public_submission", warning=has_placeholders and release_stage != "public_submission"),
        evidence="GitHub/Zenodo placeholders remain." if has_placeholders else "No GitHub/Zenodo placeholders detected.",
        required_action=(
            "Create public repository release and archive DOI, then replace placeholders."
            if release_stage == "public_submission"
            else "Allowed for reviewer-package gate only; replace before public submission."
        ),
    )


def _check_release_smoke(repo_root: Path, results_root: Path, rows: list[dict[str, Any]]) -> None:
    script_path = repo_root / "experiments" / "make_release_smoke_fixtures.py"
    config_path = repo_root / "experiments" / "configs" / "paper_release_smoke.json"
    manifest_path = results_root / "paper_release_smoke" / "fixtures" / "manifest.json"
    suite_path = results_root / "paper_release_smoke" / "manifests" / "suite.manifest.json"
    metrics_path = results_root / "paper_release_smoke" / "raw" / "known_sbs96_toy_plain_nnls" / "aggregate_metrics.tsv"

    _add_check(
        rows,
        check_id="release_smoke_script",
        category="reviewer_reproduction",
        status=_status(blocked=not _file_nonempty(script_path)),
        evidence=str(script_path.relative_to(repo_root)) if _file_nonempty(script_path) else "missing generator script",
        required_action="Keep deterministic release-smoke fixture generator in the release package.",
    )
    _add_check(
        rows,
        check_id="release_smoke_config",
        category="reviewer_reproduction",
        status=_status(blocked=not _file_nonempty(config_path)),
        evidence=str(config_path.relative_to(repo_root)) if _file_nonempty(config_path) else "missing release-smoke suite config",
        required_action="Keep paper_release_smoke suite config in the release package.",
    )
    manifest_ok, manifest_evidence = _release_fixture_manifest_ok(manifest_path)
    _add_check(
        rows,
        check_id="release_smoke_fixture_checksums",
        category="reviewer_reproduction",
        status=_status(blocked=not manifest_ok),
        evidence=manifest_evidence,
        required_action="Run `python experiments/run_paper_suite.py experiments/configs/paper_release_smoke.json`.",
    )
    suite_ok, suite_evidence = _suite_manifest_success(suite_path)
    _add_check(
        rows,
        check_id="release_smoke_suite_success",
        category="reviewer_reproduction",
        status=_status(blocked=not suite_ok),
        evidence=suite_evidence,
        required_action="Re-run the release smoke suite until all steps succeed.",
    )
    _add_check(
        rows,
        check_id="release_smoke_metrics",
        category="reviewer_reproduction",
        status=_status(blocked=not _tsv_nonempty(metrics_path)),
        evidence=str(metrics_path.relative_to(repo_root)) if _tsv_nonempty(metrics_path) else "release-smoke metrics missing_or_empty",
        required_action="Ensure release smoke known-benchmark writes aggregate_metrics.tsv.",
    )


def _check_paper_readiness(repo_root: Path, results_root: Path, rows: list[dict[str, Any]]) -> None:
    ok, evidence = _paper_readiness_ok(results_root / "paper_readiness")
    _add_check(
        rows,
        check_id="paper_artifact_readiness",
        category="paper_artifacts",
        status=_status(blocked=not ok),
        evidence=evidence,
        required_action="Run `python experiments/validate_paper_outputs.py --root results/paper --output-dir results/paper/paper_readiness`.",
    )
    figure_manifest = repo_root / "paper" / "figures" / "figure_manifest.tsv"
    _add_check(
        rows,
        check_id="figure_manifest",
        category="paper_artifacts",
        status=_status(blocked=not _tsv_nonempty(figure_manifest)),
        evidence=str(figure_manifest.relative_to(repo_root)) if _tsv_nonempty(figure_manifest) else "figure manifest missing_or_empty",
        required_action="Run `python experiments/make_paper_figures.py --root results/paper --output-dir paper/figures`.",
    )


def _check_third_party(repo_root: Path, rows: list[dict[str, Any]]) -> None:
    amusa_license = any((repo_root / "AMuSa" / name).exists() for name in ["LICENSE", "LICENSE.txt", "COPYING"])
    amusa_readme = any((repo_root / "AMuSa" / name).exists() for name in ["README", "README.md"])
    amusa_citation = any((repo_root / "AMuSa" / name).exists() for name in ["CITATION", "CITATION.cff"])
    amusa_ok = amusa_license and amusa_readme and amusa_citation
    amusa_exclusion_note = repo_root / "paper" / "amusa_release_exclusion.md"
    amusa_excluded = _file_nonempty(amusa_exclusion_note)
    if amusa_ok:
        amusa_status = _status()
        amusa_evidence = f"license={amusa_license}; readme={amusa_readme}; citation={amusa_citation}"
        amusa_action = "Preserve AMuSA provenance files and citation in the release record."
    elif amusa_excluded:
        amusa_status = _status(warning=True)
        amusa_evidence = (
            f"license={amusa_license}; readme={amusa_readme}; citation={amusa_citation}; "
            "release exclusion note present"
        )
        amusa_action = "Exclude AMuSA assets from public archive and avoid AMuSA-dependent submission claims unless provenance is resolved."
    else:
        amusa_status = _status(blocked=True)
        amusa_evidence = f"license={amusa_license}; readme={amusa_readme}; citation={amusa_citation}"
        amusa_action = "Resolve AMuSa source, citation, license, and redistribution permission or exclude AMuSa assets/results from public submission claims."
    _add_check(
        rows,
        check_id="amusa_provenance",
        category="third_party",
        status=amusa_status,
        evidence=amusa_evidence,
        required_action=amusa_action,
    )

    musical_license = repo_root / "MuSiCal" / "LICENSE"
    _add_check(
        rows,
        check_id="musical_restricted_license",
        category="third_party",
        status=_status(warning=True),
        evidence="MuSiCal LICENSE present and restricts redistribution/use." if musical_license.exists() else "MuSiCal LICENSE missing.",
        required_action="Do not vendor MuSiCal in a public release without permission; keep as optional external integration.",
        required_for_submission=False,
    )

    sigprof_license = repo_root / "SigProfilerAssignment" / "LICENSE.txt"
    sigprof_readme = repo_root / "SigProfilerAssignment" / "README.md"
    gpl_wording = _text_contains(sigprof_readme, "GNU General Public License")
    _add_check(
        rows,
        check_id="sigprofiler_license_wording",
        category="third_party",
        status=_status(warning=gpl_wording, blocked=not sigprof_license.exists()),
        evidence=f"LICENSE.txt exists={sigprof_license.exists()}; README GPL wording={gpl_wording}",
        required_action="Verify upstream SigProfilerAssignment license at the release tag; document BSD/GPL wording discrepancy.",
    )


def _check_legacy_data(repo_root: Path, rows: list[dict[str, Any]]) -> None:
    data_dir = repo_root / "Data"
    provenance_files = [data_dir / "README.md", data_dir / "manifest.json", data_dir / "source_manifest.json"]
    has_data = data_dir.exists() and any(data_dir.glob("*.csv"))
    has_provenance = any(path.exists() for path in provenance_files)
    _add_check(
        rows,
        check_id="legacy_data_provenance",
        category="data",
        status=_status(blocked=has_data and not has_provenance),
        evidence=f"Data CSVs present={has_data}; provenance file present={has_provenance}",
        required_action="Add provenance/regeneration docs for legacy Data fixtures or exclude/replace them in the public release.",
    )


def _check_environment_lock(repo_root: Path, rows: list[dict[str, Any]]) -> None:
    lock_candidates = [
        "uv.lock",
        "requirements-lock.txt",
        "environment.yml",
        "environment.yaml",
        "Dockerfile",
        "pyproject.toml",
    ]
    found = [name for name in lock_candidates if (repo_root / name).exists()]
    _add_check(
        rows,
        check_id="environment_lock_or_container",
        category="environment",
        status=_status(warning=not found),
        evidence="found: " + ",".join(found) if found else "No lock file, pyproject, environment.yml, or Dockerfile found.",
        required_action="Add a lock file, pyproject, environment.yml, or container recipe for final archive.",
        required_for_submission=False,
    )


def _summary(inventory: pd.DataFrame) -> dict[str, int | str]:
    blockers = int((inventory["status"] == "blocker").sum())
    warnings = int((inventory["status"] == "warning").sum())
    ok = int((inventory["status"] == "ok").sum())
    return {
        "overall_status": "not_submission_ready" if blockers else "submission_gate_clear",
        "ok": ok,
        "warnings": warnings,
        "blockers": blockers,
        "checks": int(len(inventory)),
    }


def _write_report(output_dir: Path, inventory: pd.DataFrame, summary: dict[str, int | str]) -> None:
    lines = [
        "# Release Readiness Report",
        "",
        f"- Generated at: `{_utc_now()}`",
        f"- Overall status: `{summary['overall_status']}`",
        f"- Checks: `{summary['checks']}`",
        f"- OK: `{summary['ok']}`",
        f"- Warnings: `{summary['warnings']}`",
        f"- Blockers: `{summary['blockers']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = inventory.loc[inventory["status"].eq("blocker")]
    if blockers.empty:
        lines.append("None.")
    else:
        for _, row in blockers.iterrows():
            lines.append(f"- `{row['check_id']}`: {row['evidence']} Action: {row['required_action']}")

    lines.extend(["", "## Warnings", ""])
    warnings = inventory.loc[inventory["status"].eq("warning")]
    if warnings.empty:
        lines.append("None.")
    else:
        for _, row in warnings.iterrows():
            lines.append(f"- `{row['check_id']}`: {row['evidence']} Action: {row['required_action']}")

    lines.extend(["", "## OK Checks", ""])
    ok_rows = inventory.loc[inventory["status"].eq("ok")]
    if ok_rows.empty:
        lines.append("None.")
    else:
        for _, row in ok_rows.iterrows():
            lines.append(f"- `{row['check_id']}`: {row['evidence']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "release_readiness_report.md").write_text("\n".join(lines) + "\n")


def validate_release_readiness(
    repo_root: Path,
    results_root: Path,
    output_dir: Path,
    *,
    release_stage: str = "public_submission",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    _check_required_files(repo_root, rows, release_stage=release_stage)
    _check_availability_placeholders(repo_root, rows, release_stage=release_stage)
    _check_release_smoke(repo_root, results_root, rows)
    _check_paper_readiness(repo_root, results_root, rows)
    _check_third_party(repo_root, rows)
    _check_legacy_data(repo_root, rows)
    _check_environment_lock(repo_root, rows)

    inventory = pd.DataFrame.from_records(rows)
    summary = _summary(inventory)
    if release_stage == "review_package" and int(summary["blockers"]) == 0:
        summary["overall_status"] = "review_package_gate_clear"
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "release_readiness_inventory.tsv", sep="\t", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "release_readiness_summary.tsv", sep="\t", index=False)
    _write_report(output_dir, inventory, summary)
    report_path = output_dir / "release_readiness_report.md"
    report_text = report_path.read_text()
    report_path.write_text(report_text.replace("# Release Readiness Report", f"# Release Readiness Report\n\n- Release stage: `{release_stage}`", 1))
    return {
        "inventory": inventory,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate release and submission readiness gates.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-root", default="results/paper")
    parser.add_argument("--output-dir", default="results/paper/release_readiness")
    parser.add_argument("--release-stage", choices=["public_submission", "review_package"], default="public_submission")
    parser.add_argument("--fail-on-blocker", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = validate_release_readiness(
        Path(args.repo_root).resolve(),
        Path(args.results_root).resolve(),
        Path(args.output_dir).resolve(),
        release_stage=args.release_stage,
    )
    blockers = int(result["summary"]["blockers"])
    return 1 if args.fail_on_blocker and blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
