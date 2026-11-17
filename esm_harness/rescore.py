"""Re-time recorded programs against another baseline without new model calls.

The original trajectory stays untouched: what the model saw and was scored
against is history. Results go to ``<session>/rescore/<baseline-stem>.jsonl``
next to the baseline measured for them.
"""

from __future__ import annotations

import json
from pathlib import Path

from esm_harness.config import resolve
from esm_harness.eval.sandbox import evaluate_program
from esm_harness.llm.prompts import read_baseline_source
from esm_harness.search.session import BASELINE_GATE, Session, _now, _sha, measure_baseline
from esm_harness.trace import read_records


def _case_rows(evaluation: dict) -> list[dict]:
    keys = ("id", "energy", "abs_err", "abs_err_baseline", "converged", "time_s", "baseline_time_s", "time_ratio")
    return [{key: case.get(key) for key in keys} for case in evaluation.get("cases") or []]


def rescore_session(session: Session, baseline: Path, submitter) -> dict:
    state = session.load()
    context = session._context(state)
    source = read_baseline_source(baseline)
    out_dir = session.root / "rescore"
    work_root = out_dir / "jobs"
    measured = measure_baseline(source, context, state["factors"], submitter, work_root)
    if measured["status"] != "ok" or not measured["all_converged"]:
        raise RuntimeError(f"{baseline} did not finish on the suite: {measured.get('stderr') or measured['status']}")
    for case in measured["cases"]:
        if case["abs_err"] is None or case["abs_err"] > BASELINE_GATE:
            raise RuntimeError(f"{baseline} {case['id']} abs_err {case['abs_err']} exceeds {BASELINE_GATE}")
    times = {case["id"]: float(case["time_s"]) for case in measured["cases"]}
    energies = {case["id"]: float(case["energy"]) for case in measured["cases"]}
    stem = baseline.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.baseline.json").write_text(
        json.dumps(
            {
                "source": str(baseline.relative_to(resolve("."))) if baseline.is_relative_to(resolve(".")) else str(baseline),
                "source_sha": _sha(source),
                "measured": _now(),
                "times": times,
                "energies": energies,
                "abs_err_vs_pyscf": {case["id"]: case["abs_err"] for case in measured["cases"]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    for record in read_records(session.trajectory):
        row = {
            "candidate_id": record["candidate_id"],
            "generation": record["generation"],
            "original_fitness": record["eval"].get("fitness"),
            "original_geomean_time_ratio": record["eval"].get("geomean_time_ratio"),
        }
        if record.get("code"):
            evaluation = evaluate_program(
                record["code"],
                cases=context["cases"],
                references=context["references"],
                baseline_times=times,
                baseline_energies=energies,
                submitter=submitter,
                factors=record["factors"],
                timeout_s=context["timeout_s"],
                memory_mb=context["memory_mb"],
                work_root=work_root,
                repeats=context["repeats"],
            )
        else:
            evaluation = {"status": record["eval"]["status"], "cases": [], "fitness": None,
                          "max_abs_err": None, "max_abs_err_vs_baseline": None, "geomean_time_ratio": None}
        row.update(
            {
                "status": evaluation["status"],
                "fitness": evaluation.get("fitness"),
                "max_abs_err": evaluation.get("max_abs_err"),
                "max_abs_err_vs_baseline": evaluation.get("max_abs_err_vs_baseline"),
                "geomean_time_ratio": evaluation.get("geomean_time_ratio"),
                "cases": _case_rows(evaluation),
            }
        )
        rows.append(row)
    (out_dir / f"{stem}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    scored = [row for row in rows if row["fitness"] is not None]
    best = max(scored, key=lambda row: row["fitness"]) if scored else None
    return {"run_id": state["run_id"], "factors": state["factors"], "baseline_times": times, "best": best, "rows": rows}


def _fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def write_rescore_report(root: Path, baseline: Path, results: list[dict]) -> Path:
    lines = [
        f"Recorded programs re-timed against `{baseline.name}`. Original scores are in REPORT.md.",
        "",
        "| run | prompt | search | speed_weight | effort | best gen | best fitness | max err vs PySCF | "
        "max err vs baseline | geomean time ratio | gen-0 time ratio |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        factors = result["factors"]
        best = result["best"] or {}
        gen0 = [row for row in result["rows"] if row["generation"] == 0 and row["geomean_time_ratio"] is not None]
        gen0_ratio = min(row["geomean_time_ratio"] for row in gen0) if gen0 else None
        lines.append(
            f"| {result['run_id']} | {factors['prompt_id']} | {factors['search']} | {factors['speed_weight']} | "
            f"{factors['reasoning_effort']} | {best.get('generation', '-')} | {_fmt(best.get('fitness'), '.3f')} | "
            f"{_fmt(best.get('max_abs_err'), '.1e')} | {_fmt(best.get('max_abs_err_vs_baseline'), '.1e')} | "
            f"{_fmt(best.get('geomean_time_ratio'), '.4f')} | {_fmt(gen0_ratio, '.4f')} |"
        )
    path = root / f"RESCORE_{baseline.stem}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
