import math

import pytest

from esm_harness.config import load_yaml, REPO_ROOT, validate_factors
from esm_harness.eval.score import ACCEPTANCE_FLOOR, fitness, geomean, meets_threshold, summarize_cases
from esm_harness.trace import best_record


def test_geomean_and_fitness_prefers_lower_error_and_time():
    assert geomean([1.0, 4.0]) == 2.0
    tight = fitness(1e-9, 1.0, True, 0.3, 1e-8)
    slower = fitness(1e-9, 2.0, True, 0.3, 1e-8)
    looser = fitness(1e-8, 1.0, True, 0.3, 1e-8)
    assert tight is not None and slower is not None and looser is not None
    assert tight > slower and tight > looser
    assert fitness(1e-4, 1.0, True, 0.3, 1e-5) is None
    assert fitness(1e-9, 1.0, False, 0.3, 1e-5) is None


def test_floor_holds_whatever_the_ceiling():
    assert ACCEPTANCE_FLOOR == 1e-8
    assert fitness(2e-8, 0.1, True, 0.5, 1e-5) is None
    assert fitness(1e-8, 0.1, True, 0.5, 1e-5) is not None
    cases = [
        {"energy": -1.0, "abs_err": 1e-9, "converged": True, "time_ratio": 0.5},
        {"energy": -2.0, "abs_err": 3e-8, "converged": True, "time_ratio": 0.5},
    ]
    summary = summarize_cases(cases, speed_weight=0.5, err_ceiling=1e-5)
    assert summary["accepted"] is False and summary["fitness"] is None
    assert meets_threshold(summary, {"max_abs_err": 1e-6, "max_time_ratio": 1.0}) is False


def test_summarize_and_threshold():
    cases = [
        {"energy": -1.0, "abs_err": 1e-9, "converged": True, "time_ratio": 1.2},
        {"energy": -2.0, "abs_err": 1e-8, "converged": True, "time_ratio": 1.2},
    ]
    summary = summarize_cases(cases, speed_weight=0.0, err_ceiling=1e-8)
    assert summary["all_converged"] is True and summary["accepted"] is True
    assert math.isclose(summary["max_abs_err"], 1e-8)
    factors = {"max_abs_err": 1e-8, "max_time_ratio": 1.5}
    assert meets_threshold(summary, factors)
    summary["geomean_time_ratio"] = 3.0
    assert meets_threshold(summary, factors) is False
    assert summarize_cases([], 0.5, 1e-8)["accepted"] is False


def test_configs_cannot_loosen_the_floor():
    factors = load_yaml(REPO_ROOT / "configs" / "hf_rhf.yaml")["factors"]
    assert validate_factors(factors)["fitness_err_ceiling"] == 1e-8
    with pytest.raises(ValueError, match="acceptance floor"):
        validate_factors({**factors, "fitness_err_ceiling": 1e-6})
    with pytest.raises(ValueError, match="acceptance floor"):
        validate_factors({**factors, "max_abs_err": 1e-7})


def test_records_above_the_floor_never_become_best():
    def record(cid, fitness_value, err):
        return {"candidate_id": cid, "eval": {"fitness": fitness_value, "max_abs_err": err}}

    old = record("old", 20.0, 5e-7)
    good = record("good", 9.0, 1e-10)
    assert best_record([old, good])["candidate_id"] == "good"
    assert best_record([old]) is None
