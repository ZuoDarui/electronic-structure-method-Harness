"""Small reproducible timing helpers with no scientific dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Callable, Any


@dataclass(frozen=True)
class TimingSummary:
    samples_s: tuple[float, ...]
    median_s: float
    minimum_s: float
    p90_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples_s": list(self.samples_s),
            "median_s": self.median_s,
            "minimum_s": self.minimum_s,
            "p90_s": self.p90_s,
        }


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("values must be non-empty and fraction must be in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def benchmark(
    function: Callable[[], Any], *, warmups: int = 1, repeats: int = 5
) -> TimingSummary:
    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return TimingSummary(
        samples_s=tuple(samples),
        median_s=statistics.median(samples),
        minimum_s=min(samples),
        p90_s=percentile(samples, 0.9),
    )
