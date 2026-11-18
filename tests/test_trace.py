from esm_harness.trace import append_record, read_records, write_ablation_ledger, write_ledger


def _record(generation, fitness, usage):
    return {
        "run_id": "r",
        "candidate_id": f"r-g{generation}",
        "generation": generation,
        "model": "claude-opus-5-5-high",
        "usage": usage,
        "eval": {
            "fitness": fitness,
            "max_abs_err": 1e-9,
            "max_abs_err_vs_baseline": 1e-9,
            "geomean_time_ratio": 0.8,
            "status": "ok",
            "cases": [],
        },
    }


def test_ledger_separates_metered_calls(tmp_path):
    path = tmp_path / "t.jsonl"
    append_record(path, _record(0, None, None))
    append_record(path, _record(1, 3.0, {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}))
    records = read_records(path)
    ledger = write_ledger(tmp_path / "ledger.json", records, {"model": "claude-opus-5.5"}, "done")
    assert ledger["calls"] == 2
    assert ledger["unmetered_calls"] == 1
    assert ledger["usage"]["total_tokens"] == 7
    assert ledger["models"] == {"claude-opus-5-5-high": 2}
    assert ledger["best"]["generation"] == 1
    summary = write_ablation_ledger(tmp_path / "ab.json", [ledger, ledger])
    assert summary["calls"] == 4 and summary["usage"]["total_tokens"] == 14
