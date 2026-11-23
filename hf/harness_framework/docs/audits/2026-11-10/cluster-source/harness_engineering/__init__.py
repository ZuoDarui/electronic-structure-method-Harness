"""Small, framework-neutral harness primitives for electronic-structure work."""

from .models import AcceptanceGate, CandidateRun, MethodSpec, MetricReport
from .runner import evaluate_candidate, load_report, run_candidate

__all__ = [
    "AcceptanceGate",
    "CandidateRun",
    "MethodSpec",
    "MetricReport",
    "evaluate_candidate",
    "load_report",
    "run_candidate",
]
