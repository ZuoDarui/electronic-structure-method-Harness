"""Serializable contracts used by the tutorial harness.

The objects in this module deliberately contain no agent-specific code. An
agent, a shell script, or a CI job can produce the same task and report files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _finite_number(value: Any) -> bool:
    """JSON booleans are not numerical measurements; NaN is never evidence."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class MethodSpec:
    """A bounded, executable request for one optimization experiment."""

    name: str
    description: str
    command: tuple[str, ...]
    cases: tuple[Mapping[str, Any], ...] = ()
    correctness_tolerances: Mapping[str, float] = field(default_factory=dict)
    performance: Mapping[str, float] = field(default_factory=dict)
    required_metrics: tuple[str, ...] = ()
    timeout_s: float = 300.0
    env: Mapping[str, str] = field(default_factory=dict)
    generation: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MethodSpec":
        command = raw.get("command")
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
            raise ValueError("command must be a sequence of executable arguments")
        if not command or any(not isinstance(item, str) for item in command):
            raise ValueError("command must contain at least one string argument")
        timeout_s = float(raw.get("timeout_s", 300.0))
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        generation = raw.get("generation", {})
        if not isinstance(generation, Mapping):
            raise ValueError("generation must be a mapping")
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            command=tuple(command),
            cases=tuple(raw.get("cases", ())),
            correctness_tolerances=dict(raw.get("correctness_tolerances", {})),
            performance=dict(raw.get("performance", {})),
            required_metrics=tuple(raw.get("required_metrics", ())),
            timeout_s=timeout_s,
            env=dict(raw.get("env", {})),
            generation=dict(generation),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["command"] = list(self.command)
        result["cases"] = list(self.cases)
        result["required_metrics"] = list(self.required_metrics)
        return result


@dataclass
class CandidateRun:
    """The complete, bounded result of running one candidate."""

    status: str
    command: list[str]
    returncode: int | None
    duration_s: float
    stdout: str
    stderr: str
    report_path: str | None = None
    failure_kind: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricReport:
    """Machine-readable metrics emitted by a candidate program."""

    metrics: dict[str, float | int | bool | str]
    details: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MetricReport":
        metrics = dict(raw.get("metrics", raw))
        details = dict(raw.get("details", {})) if "metrics" in raw else {}
        passed = bool(raw.get("passed", True)) if "metrics" in raw else True
        reasons = list(raw.get("reasons", ())) if "metrics" in raw else []
        return cls(metrics=metrics, details=details, passed=passed, reasons=reasons)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptanceGate:
    """Correctness-first decision policy for a candidate report.

    A metric named in ``correctness_tolerances`` is interpreted as an absolute
    error. ``max_slowdown`` compares ``speedup`` to a lower bound when present;
    otherwise it compares ``slowdown`` to an upper bound.
    """

    correctness_tolerances: Mapping[str, float] = field(default_factory=dict)
    required_metrics: tuple[str, ...] = ()
    min_speedup: float | None = None
    max_slowdown: float | None = None

    def evaluate(self, run: CandidateRun, report: MetricReport | None) -> MetricReport:
        reasons: list[str] = []
        metrics = {} if report is None else dict(report.metrics)
        if not run.ok:
            reasons.append(run.failure_kind or "candidate did not complete successfully")
        if report is None:
            reasons.append("candidate did not produce a JSON metric report")
        elif not report.passed:
            reasons.extend(report.reasons or ["candidate report marked itself as failed"])
        for key in self.required_metrics:
            if key not in metrics:
                reasons.append(f"missing required metric: {key}")
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not _finite_number(value):
                    reasons.append(f"non-finite metric: {key}={value!r}")
        for key, tolerance in self.correctness_tolerances.items():
            value = metrics.get(key)
            if value is None:
                reasons.append(f"missing correctness metric: {key}")
            elif not _finite_number(tolerance) or tolerance < 0:
                reasons.append(f"invalid correctness tolerance: {key}={tolerance!r}")
            elif not _finite_number(value) or abs(value) > tolerance:
                reasons.append(f"{key}={value!r} exceeds tolerance {tolerance:g}")
        finite = metrics.get("all_finite")
        if "all_finite" in metrics and finite is not True:
            reasons.append("all_finite must be the boolean true")
        converged = metrics.get("scf_converged")
        if "scf_converged" in metrics and converged is not True:
            reasons.append("SCF did not converge (scf_converged must be the boolean true)")
        speedup = metrics.get("speedup")
        slowdown = metrics.get("slowdown")
        if self.min_speedup is not None and speedup is None:
            reasons.append("missing performance metric: speedup")
        elif self.min_speedup is not None:
            if not _finite_number(self.min_speedup) or self.min_speedup < 0:
                reasons.append("invalid performance threshold: min_speedup")
            elif not _finite_number(speedup):
                reasons.append("speedup must be a finite number")
            elif speedup < self.min_speedup:
                reasons.append(f"speedup={speedup:g} is below {self.min_speedup:g}")
        if self.max_slowdown is not None and slowdown is None:
            reasons.append("missing performance metric: slowdown")
        elif self.max_slowdown is not None:
            if not _finite_number(self.max_slowdown) or self.max_slowdown < 0:
                reasons.append("invalid performance threshold: max_slowdown")
            elif not _finite_number(slowdown):
                reasons.append("slowdown must be a finite number")
            elif slowdown > self.max_slowdown:
                reasons.append(f"slowdown={slowdown:g} exceeds {self.max_slowdown:g}")
        details = {} if report is None else dict(report.details)
        return MetricReport(metrics=metrics, details=details, passed=not reasons, reasons=reasons)
