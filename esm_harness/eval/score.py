"""Accuracy and same-host speed scoring."""

from __future__ import annotations

import math


def geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("geomean of an empty series")
    if any(value <= 0 for value in values):
        raise ValueError("geomean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def fitness(
    max_abs_err: float | None,
    time_ratio: float | None,
    all_converged: bool,
    speed_weight: float,
    err_ceiling: float,
) -> float | None:
    """Return a score that prefers smaller energy error and a smaller time ratio.

    Programs that do not converge, or whose energy error is above ``err_ceiling``,
    are failures and receive no fitness.
    """
    if not all_converged or max_abs_err is None or time_ratio is None:
        return None
    if max_abs_err > err_ceiling or time_ratio <= 0:
        return None
    return -math.log10(max(max_abs_err, 1e-12)) - speed_weight * math.log(time_ratio)


def summarize_cases(cases: list[dict], speed_weight: float, err_ceiling: float) -> dict:
    if not cases or any(case.get("energy") is None for case in cases):
        return {
            "max_abs_err": None,
            "max_abs_err_vs_baseline": None,
            "geomean_time_ratio": None,
            "all_converged": False,
            "fitness": None,
        }
    all_converged = all(bool(case.get("converged")) for case in cases)
    max_abs_err = max(float(case["abs_err"]) for case in cases)
    baseline_errs = [case.get("abs_err_baseline") for case in cases]
    max_vs_baseline = None if None in baseline_errs else max(float(err) for err in baseline_errs)
    try:
        ratio = geomean([float(case["time_ratio"]) for case in cases])
    except (TypeError, ValueError):
        ratio = None
    return {
        "max_abs_err": max_abs_err,
        "max_abs_err_vs_baseline": max_vs_baseline,
        "geomean_time_ratio": ratio,
        "all_converged": all_converged,
        "fitness": fitness(max_abs_err, ratio, all_converged, speed_weight, err_ceiling),
    }


def meets_threshold(summary: dict, factors: dict) -> bool:
    err = summary.get("max_abs_err")
    ratio = summary.get("geomean_time_ratio")
    if not summary.get("all_converged") or err is None or ratio is None:
        return False
    return err <= float(factors["max_abs_err"]) and ratio <= float(factors["max_time_ratio"])
