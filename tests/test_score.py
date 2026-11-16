import math

from esm_harness.eval.score import fitness, geomean, meets_threshold, summarize_cases


def test_geomean_and_fitness_prefers_lower_error_and_time():
    assert geomean([1.0, 4.0]) == 2.0
    tight = fitness(1e-8, 1.0, True, 0.3, 1e-5)
    loose = fitness(1e-6, 2.0, True, 0.3, 1e-5)
    assert tight is not None and loose is not None
    assert tight > loose
    assert fitness(1e-4, 1.0, True, 0.3, 1e-5) is None
    assert fitness(1e-8, 1.0, False, 0.3, 1e-5) is None


def test_summarize_and_threshold():
    cases = [
        {"energy": -1.0, "abs_err": 1e-9, "converged": True, "time_ratio": 1.2},
        {"energy": -2.0, "abs_err": 1e-8, "converged": True, "time_ratio": 1.2},
    ]
    summary = summarize_cases(cases, speed_weight=0.0, err_ceiling=1e-5)
    assert summary["all_converged"] is True
    assert math.isclose(summary["max_abs_err"], 1e-8)
    factors = {"max_abs_err": 1e-6, "max_time_ratio": 1.5}
    assert meets_threshold(summary, factors)
    summary["geomean_time_ratio"] = 3.0
    assert meets_threshold(summary, factors) is False
