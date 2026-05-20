from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .amusa import AMuSaExpert
from .amusa_support_only import AMuSaSupportOnlyExpert
from .base import BaseExpert
from .constrained_refit import ClassifierGuidedRefitExpert
from .musical import MuSiCalExpert
from .plain_nnls import PlainNNLSExpert
from .schema import ExpertRequest, ExpertRunResult
from .sigprofiler import SigProfilerAssignmentExpert


CORE_EXPERT_NAMES = ["plain_nnls"]
OPTIONAL_EXPERT_NAMES = [
    "amusa",
    "amusa_support_only",
    "classifier_guided_refit",
    "musical",
    "sigprofiler_assignment",
]


class ExpertRegistry:
    def __init__(
        self,
        experts: Iterable[BaseExpert] | None = None,
        *,
        default_expert_names: Iterable[str] | None = None,
    ) -> None:
        self._experts: dict[str, BaseExpert] = {}
        self._default_expert_names: list[str] = list(default_expert_names or [])
        for expert in experts or []:
            self.register(expert)

    def register(self, expert: BaseExpert, *, default: bool = True) -> None:
        self._experts[expert.expert_name] = expert
        if default and expert.expert_name not in self._default_expert_names:
            self._default_expert_names.append(expert.expert_name)
        if not default and expert.expert_name in self._default_expert_names:
            self._default_expert_names.remove(expert.expert_name)

    def names(self) -> list[str]:
        return sorted(self._experts.keys())

    def default_names(self) -> list[str]:
        return [expert_name for expert_name in self._default_expert_names if expert_name in self._experts]

    def get(self, expert_name: str) -> BaseExpert:
        return self._experts[expert_name]

    def run(self, expert_name: str, request: ExpertRequest) -> ExpertRunResult:
        return self.get(expert_name).run(request)

    def run_all(
        self,
        request: ExpertRequest,
        expert_names: list[str] | None = None,
    ) -> list[ExpertRunResult]:
        selected = expert_names or self.default_names()
        return [self.get(expert_name).run(request) for expert_name in selected]


def build_default_registry(
    repo_root: str | Path | None = None,
    *,
    confidence_artifacts: Any | None = None,
) -> ExpertRegistry:
    """Build the registry with a self-contained public core and optional adapters.

    The default expert set is intentionally limited to plain NNLS so released
    packages and reviewer smoke tests do not require vendored third-party tools.
    MuSiCal, SigProfilerAssignment, AMuSA, and AMuSA-derived adapters remain
    available when users install or provide those tools separately and request
    them explicitly via expert names.
    """
    root = Path(repo_root or Path.cwd()).resolve()
    amusa_calibrator = None
    if confidence_artifacts is not None:
        amusa_calibrator = (
            getattr(confidence_artifacts, "amusa_group_calibrator", None)
            or getattr(confidence_artifacts, "amusa_probability_calibrator", None)
        )
    registry = ExpertRegistry()
    registry.register(
        AMuSaExpert(
            repo_root=root,
            probability_calibrator=amusa_calibrator,
            prediction_profile=None,
            exposure_profile="sig_exposure_plus",
        ),
        default=False,
    )
    registry.register(
        ClassifierGuidedRefitExpert(
            repo_root=root,
            probability_calibrator=amusa_calibrator,
            prediction_profile=None,
        ),
        default=False,
    )
    registry.register(MuSiCalExpert(repo_root=root), default=False)
    registry.register(PlainNNLSExpert(repo_root=root), default=True)
    registry.register(SigProfilerAssignmentExpert(repo_root=root), default=False)
    registry.register(
        AMuSaSupportOnlyExpert(
            repo_root=root,
            probability_calibrator=amusa_calibrator,
        ),
        default=False,
    )
    return registry
