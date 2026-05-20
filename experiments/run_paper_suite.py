#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "skills" / "mutational-signature-decision" / "scripts" / "run_signature_decision.py"
SCRIPT_ENTRYPOINTS = {
    "download-public-sbs96": REPO_ROOT / "experiments" / "download_public_sbs96.py",
    "release-smoke-fixtures": REPO_ROOT / "experiments" / "make_release_smoke_fixtures.py",
    "real-data-removal-design": REPO_ROOT / "experiments" / "make_real_data_removal_design.py",
    "signature-catalog-variant": REPO_ROOT / "experiments" / "make_signature_catalog_variant.py",
}


SOURCE_KEYS = {
    "sample_source",
    "signature_source",
    "exposure_source",
    "confidence_artifact",
    "catalog_assessor_artifact",
    "removal_manifest",
}

OUTPUT_DIR_COMMANDS = {
    "decision",
    "known-benchmark",
    "catalog-insufficiency-benchmark",
    "discovery-run",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{path} is not valid JSON. Use JSON configs for paper suites to avoid YAML parser dependencies."
        ) from exc


def _format_value(value: Any, *, output_root: Path, repo_root: Path) -> Any:
    if isinstance(value, str):
        return value.format(output_root=str(output_root), repo_root=str(repo_root))
    if isinstance(value, list):
        return [_format_value(item, output_root=output_root, repo_root=repo_root) for item in value]
    if isinstance(value, dict):
        return {
            key: _format_value(item, output_root=output_root, repo_root=repo_root)
            for key, item in value.items()
        }
    return value


def _cli_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _build_command(command_name: str, args: dict[str, Any]) -> list[str]:
    script_entrypoint = SCRIPT_ENTRYPOINTS.get(command_name)
    if script_entrypoint is None:
        command = [sys.executable, str(ENTRYPOINT), command_name]
    else:
        command = [sys.executable, str(script_entrypoint)]
    for key, value in args.items():
        if value is None:
            continue
        flag = _cli_flag(key)
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        command.extend([flag, str(value)])
    return command


def _source_manifest(args: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for key in sorted(SOURCE_KEYS):
        value = args.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = REPO_ROOT / path
        manifest[key] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": _sha256(path),
        }
    return manifest


def _step_manifest(
    *,
    suite_name: str,
    step: dict[str, Any],
    command: list[str],
    args: dict[str, Any],
    started_at: str,
    completed_at: str | None,
    returncode: int | None,
    status: str,
) -> dict[str, Any]:
    return {
        "suite_name": suite_name,
        "step_name": step["name"],
        "command_name": step["command"],
        "argv": command,
        "status": status,
        "returncode": returncode,
        "started_at": started_at,
        "completed_at": completed_at,
        "repo_root": str(REPO_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "sources": _source_manifest(args),
    }


def run_suite(config_path: Path, *, dry_run: bool = False, continue_on_error: bool = False) -> int:
    config = _load_config(config_path)
    suite_name = str(config.get("suite_name") or config_path.stem)
    output_root = Path(str(config.get("output_root") or f"results/paper/{suite_name}"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    manifest_dir = output_root / "manifests"
    log_dir = output_root / "logs"
    if not dry_run:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    steps = list(config.get("steps") or [])
    if not steps:
        raise SystemExit(f"No steps configured in {config_path}.")

    suite_manifest: list[dict[str, Any]] = []
    final_code = 0
    for step in steps:
        if "name" not in step or "command" not in step:
            raise SystemExit(f"Each step needs name and command: {step}")
        raw_args = dict(step.get("args") or {})
        if "output_dir" not in raw_args and step["command"] in OUTPUT_DIR_COMMANDS:
            raw_args["output_dir"] = "{output_root}/raw/" + str(step["name"])
        formatted_args = _format_value(raw_args, output_root=output_root, repo_root=REPO_ROOT)
        command = _build_command(str(step["command"]), formatted_args)
        started_at = _utc_now()
        print(" ".join(command))
        if dry_run:
            manifest = _step_manifest(
                suite_name=suite_name,
                step=step,
                command=command,
                args=formatted_args,
                started_at=started_at,
                completed_at=None,
                returncode=None,
                status="dry_run",
            )
            suite_manifest.append(manifest)
            continue

        log_path = log_dir / f"{step['name']}.log"
        with log_path.open("w") as log_file:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        completed_at = _utc_now()
        status = "success" if process.returncode == 0 else "failed"
        manifest = _step_manifest(
            suite_name=suite_name,
            step=step,
            command=command,
            args=formatted_args,
            started_at=started_at,
            completed_at=completed_at,
            returncode=process.returncode,
            status=status,
        )
        manifest["log_path"] = str(log_path)
        manifest_path = manifest_dir / f"{step['name']}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        suite_manifest.append(manifest)
        if process.returncode != 0:
            final_code = process.returncode
            if not continue_on_error:
                break

    if not dry_run:
        (manifest_dir / "suite.manifest.json").write_text(
            json.dumps(suite_manifest, indent=2, ensure_ascii=False)
        )
    return final_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible paper experiment suites.")
    parser.add_argument("config", help="JSON suite config.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and manifests without running.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_suite(Path(args.config), dry_run=args.dry_run, continue_on_error=args.continue_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
