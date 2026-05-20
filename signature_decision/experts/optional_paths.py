from __future__ import annotations

import os
from pathlib import Path
import sys


def _candidate_paths(env_var: str, repo_root: Path, default_local_dir: str) -> list[Path]:
    candidates: list[Path] = []
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.extend(Path(item).expanduser() for item in env_value.split(os.pathsep) if item.strip())
    candidates.append(repo_root / default_local_dir)
    return [candidate.resolve() for candidate in candidates]


def _import_roots_for(candidate: Path, package_name: str) -> list[Path]:
    if candidate.name == package_name:
        return [candidate.parent]
    if (candidate / package_name).exists():
        return [candidate]
    return [candidate]


def add_optional_import_path(
    *,
    env_var: str,
    repo_root: Path,
    default_local_dir: str,
    package_name: str,
) -> list[Path]:
    """Prepend user-supplied optional-tool locations to sys.path.

    The helper supports either a path to the package directory itself
    (for example `/opt/MuSiCal/musical`) or a checkout root containing the
    importable package directory (for example `/opt/MuSiCal`).
    """
    added: list[Path] = []
    for candidate in _candidate_paths(env_var, repo_root, default_local_dir):
        for import_root in _import_roots_for(candidate, package_name):
            if import_root.exists() and str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
                added.append(import_root)
    return added


def resolve_optional_package_dir(
    *,
    env_var: str,
    repo_root: Path,
    default_local_dir: str,
    package_name: str,
) -> Path | None:
    """Resolve an optional package source directory when direct file loading is needed."""
    for candidate in _candidate_paths(env_var, repo_root, default_local_dir):
        if candidate.name == package_name and (candidate / "__init__.py").exists():
            return candidate
        nested = candidate / package_name
        if (nested / "__init__.py").exists():
            return nested
    return None
