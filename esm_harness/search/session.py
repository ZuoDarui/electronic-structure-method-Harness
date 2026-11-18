"""Resumable search session driven by prompt and response files.

A session directory holds everything a run produced::

    session.json          config, factors, bound model, status, generation
    baseline.json         baseline energies and times (measured on this host,
                          or recorded on the cluster when ``baseline: recorded``)
    prompts/<cid>.prompt.md
    responses/<cid>.response.md   written by the model
    meta/<cid>.meta.json          model id, agent id, reported token usage
    programs/<cid>.py
    trajectory.jsonl      one record per candidate
    ledger.json

``advance`` is the only state transition. It scores every answered candidate
of the pending generation, then writes the prompts for the next one. Calling it
again before the responses exist changes nothing, so a crashed driver resumes
where it stopped.

With an asynchronous submitter (the cluster), the first ``advance`` after the
responses exist submits each program's jobs and sets the status to
``evaluating``; later calls collect finished jobs, and the generation closes
only once every candidate has been scored.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import socket
import uuid
from pathlib import Path

from esm_harness.config import resolve, subagent_model, validate_factors
from esm_harness.eval.cases import load_cases
from esm_harness.eval.reference import read_reference_records
from esm_harness.eval.sandbox import collect_evaluation, evaluate_program, start_evaluation
from esm_harness.eval.score import meets_threshold
from esm_harness.llm.client import normalize_usage, read_meta, split_program
from esm_harness.llm.prompts import read_baseline_source, render_prompt, subagent_instructions
from esm_harness.search.strategies import strategy_for
from esm_harness.trace import append_record, best_record, read_records, write_ledger

BASELINE_GATE = 1e-8
RECORDED = "recorded"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def measure_baseline(source: str, context: dict, factors: dict, submitter, work_root: Path) -> dict:
    unit = {case["id"]: 1.0 for case in context["cases"]}
    return evaluate_program(
        source,
        cases=context["cases"],
        references=context["references"],
        baseline_times=unit,
        baseline_energies=unit,
        submitter=submitter,
        factors=factors,
        timeout_s=context["timeout_s"],
        memory_mb=context["memory_mb"],
        work_root=work_root,
        repeats=context["repeats"],
        trusted=True,
    )


def _unscored(status: str, stderr: str) -> dict:
    return {
        "status": status,
        "stderr": stderr,
        "cases": [],
        "submission": None,
        "max_abs_err": None,
        "max_abs_err_vs_baseline": None,
        "geomean_time_ratio": None,
        "all_converged": False,
        "accepted": False,
        "fitness": None,
    }


class Session:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.state_path = self.root / "session.json"
        self.trajectory = self.root / "trajectory.jsonl"
        self.ledger_path = self.root / "ledger.json"

    # ---- persistence -------------------------------------------------
    def load(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save(self, state: dict) -> None:
        state["updated"] = _now()
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def exists(self) -> bool:
        return self.state_path.is_file()

    def path(self, kind: str, cid: str) -> Path:
        suffix = {"prompts": ".prompt.md", "responses": ".response.md", "meta": ".meta.json", "programs": ".py"}[kind]
        return self.root / kind / f"{cid}{suffix}"

    # ---- setup -------------------------------------------------------
    def init(self, config: dict, submitter, run_id: str | None = None) -> dict:
        if self.exists():
            raise FileExistsError(f"{self.root} already holds a session; resume it or pick another out_dir")
        factors = validate_factors(config["factors"])
        expected_model = subagent_model(factors)
        self.root.mkdir(parents=True, exist_ok=True)
        for kind in ("prompts", "responses", "meta", "programs", "jobs"):
            (self.root / kind).mkdir(exist_ok=True)
        state = {
            "run_id": run_id or config.get("run_id") or f"hf-{uuid.uuid4().hex[:12]}",
            "created": _now(),
            "config": config,
            "factors": factors,
            "model": factors["model"],
            "subagent_model": expected_model,
            "status": "measuring_baseline",
            "generation": -1,
            "pending": [],
            "stale_generations": 0,
        }
        self.save(state)
        try:
            self._measure_baseline(state, submitter)
        except RuntimeError:
            state["status"] = "baseline_failed"
            self.save(state)
            raise
        self._open_generation(state, 0)
        return state

    def _context(self, state: dict) -> dict:
        config = state["config"]
        cases = load_cases(resolve(config["suite"]))
        records = read_reference_records(resolve(config["references"]), cases)
        return {
            "cases": cases,
            "references": {case_id: record["energy"] for case_id, record in records.items()},
            "reference_records": records,
            "timeout_s": float(config.get("timeout_s", 120)),
            "memory_mb": int(config.get("memory_mb", 8192)),
            "repeats": int(config.get("repeats", 3)),
        }

    def _timeouts(self, state: dict, context: dict, baseline: dict) -> dict[str, float]:
        """Per-case limit: ``timeout_s``, or ``timeout_factor`` times the baseline time if that is longer."""
        factor = state["config"].get("timeout_factor")
        out = {}
        for case in context["cases"]:
            limit = context["timeout_s"]
            if factor and baseline["times"].get(case["id"]):
                limit = max(limit, float(factor) * float(baseline["times"][case["id"]]))
            out[case["id"]] = limit
        return out

    def recorded_baseline(self, state: dict) -> bool:
        return state["config"].get("baseline") == RECORDED

    def _measure_baseline(self, state: dict, submitter) -> None:
        context = self._context(state)
        if self.recorded_baseline(state):
            records = context["reference_records"]
            missing = [case_id for case_id, record in records.items() if not record["time_s"]]
            if missing:
                raise RuntimeError(f"no recorded baseline time for {missing}")
            baseline = {
                "source": RECORDED,
                "references": state["config"]["references"],
                "measured": _now(),
                "host": "recorded",
                "times": {case_id: record["time_s"] for case_id, record in records.items()},
                "energies": {case_id: record["energy"] for case_id, record in records.items()},
                "time_sources": {case_id: record["source"] for case_id, record in records.items()},
            }
            (self.root / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
            return
        source = read_baseline_source(resolve(state["config"]["baseline"]))
        measured = measure_baseline(source, context, state["factors"], submitter, self.root / "jobs")
        if measured["status"] != "ok" or not measured["all_converged"]:
            raise RuntimeError("baseline did not finish on the suite: " + (measured.get("stderr") or measured["status"]))
        for case in measured["cases"]:
            if case["abs_err"] is None or case["abs_err"] > BASELINE_GATE:
                raise RuntimeError(f"baseline {case['id']} abs_err {case['abs_err']} exceeds {BASELINE_GATE}")
        baseline = {
            "source": state["config"]["baseline"],
            "source_sha": _sha(source),
            "measured": _now(),
            "host": socket.gethostname(),
            "times": {case["id"]: float(case["time_s"]) for case in measured["cases"]},
            "energies": {case["id"]: float(case["energy"]) for case in measured["cases"]},
            "abs_err_vs_pyscf": {case["id"]: case["abs_err"] for case in measured["cases"]},
            "submission": measured["submission"],
        }
        (self.root / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    def remeasure_baseline(self, submitter) -> dict:
        """Time the baseline again on this host; the previous measurement moves to baseline_history.jsonl.

        Rounds run on different machines, so each round's candidates are compared
        with a baseline timed on the machine that times them. A recorded baseline
        belongs to the cluster that runs the jobs and is left as it is.
        """
        if self.recorded_baseline(self.load()):
            return self.baseline()
        previous = self.baseline()
        self._measure_baseline(self.load(), submitter)
        append_record(self.root / "baseline_history.jsonl", previous)
        return self.baseline()

    def baseline(self) -> dict:
        return json.loads((self.root / "baseline.json").read_text(encoding="utf-8"))

    # ---- generations -------------------------------------------------
    def _feedback(self, records: list[dict]) -> dict | None:
        if not records:
            return None
        best = best_record(records)
        latest_generation = max(record["generation"] for record in records)
        latest = [record for record in records if record["generation"] == latest_generation]
        last = best_record(latest) or latest[-1]
        return {"best": best, "last": last}

    def _open_generation(self, state: dict, generation: int) -> None:
        context = self._context(state)
        factors = state["factors"]
        records = read_records(self.trajectory)
        shown = None
        if factors["prompt_id"] == "hf_show_baseline":
            shown = read_baseline_source(resolve(state["config"]["baseline"]))
        prompt = render_prompt(
            factors["prompt_id"],
            context["cases"],
            self._feedback(records),
            shown,
            allowed_imports=factors["allowed_imports"],
        )
        pending = []
        for sample in range(strategy_for(factors).samples(factors)):
            cid = f"{state['run_id']}-g{generation:02d}-s{sample}"
            self.path("prompts", cid).write_text(prompt, encoding="utf-8")
            pending.append(cid)
        state["generation"] = generation
        state["pending"] = pending
        state["status"] = "awaiting_responses"
        self.save(state)

    def call_failures(self, cid: str) -> int:
        return sum(
            1 for call in read_records(self.root / "calls.jsonl") if call["candidate_id"] == cid and call.get("error")
        )

    def _gave_up(self, state: dict, cid: str) -> bool:
        limit = int(state["config"].get("max_call_failures", 3))
        return not self.path("responses", cid).is_file() and self.call_failures(cid) >= limit

    def requests(self) -> list[dict]:
        """What the driver must answer before the session can advance."""
        state = self.load()
        return [
            {
                "candidate_id": cid,
                "model": state["subagent_model"],
                "prompt": str(self.path("prompts", cid)),
                "response": str(self.path("responses", cid)),
                "meta": str(self.path("meta", cid)),
                "instructions": subagent_instructions(
                    str(self.path("prompts", cid)), str(self.path("responses", cid))
                ),
            }
            for cid in state["pending"]
            if not self.path("responses", cid).is_file() and not self._gave_up(state, cid)
        ]

    def _answer(self, state: dict, cid: str) -> dict:
        failed = self._gave_up(state, cid)
        response = "" if failed else self.path("responses", cid).read_text(encoding="utf-8")
        meta = {"model": state["subagent_model"]} if failed else read_meta(self.path("meta", cid))
        model = meta.get("model")
        if model != state["subagent_model"] and not state["config"].get("allow_any_model"):
            raise ValueError(
                f"{cid}: response meta names model {model!r}; this session is bound to {state['subagent_model']!r}"
            )
        code, parse_error = split_program(response)
        if code:
            self.path("programs", cid).parent.mkdir(parents=True, exist_ok=True)
            self.path("programs", cid).write_text(code + "\n", encoding="utf-8")
        return {"failed": failed, "response": response, "meta": meta, "code": code, "parse_error": parse_error}

    def _unscored_evaluation(self, cid: str, answer: dict) -> dict:
        if answer["failed"]:
            return _unscored("call_failed", f"model call failed {self.call_failures(cid)} times; see calls.jsonl")
        return _unscored("no_program", answer["parse_error"])

    def _make_record(self, state: dict, cid: str, answer: dict, evaluation: dict) -> dict:
        factors = state["factors"]
        prompt = self.path("prompts", cid).read_text(encoding="utf-8")
        meta = answer["meta"]
        evaluation["meets_threshold"] = meets_threshold(evaluation, factors)
        generation = state["generation"]
        return {
            "run_id": state["run_id"],
            "candidate_id": cid,
            "generation": generation,
            "sample": int(cid.rsplit("-s", 1)[1]),
            "factors": factors,
            "model": meta.get("model"),
            "agent_id": meta.get("agent_id"),
            "call_failures": self.call_failures(cid),
            "usage": normalize_usage(meta.get("usage")),
            "usage_note": meta.get("usage_note"),
            "latency_s": meta.get("latency_s"),
            "from_scratch": generation == 0,
            "prompt_sha": _sha(prompt),
            "prompt": prompt,
            "response": answer["response"],
            "code": answer["code"],
            "eval": evaluation,
            "scored_at": _now(),
        }

    def _record(self, state: dict, cid: str, submitter, context: dict, baseline: dict) -> dict:
        answer = self._answer(state, cid)
        if answer["code"]:
            evaluation = evaluate_program(
                answer["code"],
                cases=context["cases"],
                references=context["references"],
                baseline_times=baseline["times"],
                baseline_energies=baseline["energies"],
                submitter=submitter,
                factors=state["factors"],
                timeout_s=context["timeout_s"],
                memory_mb=context["memory_mb"],
                work_root=self.root / "jobs",
                repeats=context["repeats"],
            )
        else:
            evaluation = self._unscored_evaluation(cid, answer)
        return self._make_record(state, cid, answer, evaluation)

    def _advance_async(self, state: dict, submitter, context: dict, baseline: dict) -> bool:
        """Submit and collect cluster jobs; True once every pending candidate is recorded."""
        evaluating = state.setdefault("evaluating", {})
        recorded = {record["candidate_id"] for record in read_records(self.trajectory)}
        timeouts = self._timeouts(state, context, baseline)
        for cid in state["pending"]:
            if cid in recorded or cid in evaluating:
                continue
            answer = self._answer(state, cid)
            if answer["code"]:
                started = start_evaluation(
                    answer["code"],
                    cases=context["cases"],
                    submitter=submitter,
                    factors=state["factors"],
                    timeouts=timeouts,
                    memory_mb=context["memory_mb"],
                    work_root=self.root / "jobs",
                    repeats=context["repeats"],
                    label=cid,
                )
            else:
                started = {"evaluation": self._unscored_evaluation(cid, answer)}
            evaluating[cid] = {**started, "started_at": _now()}
            self.save(state)
        for cid in list(evaluating):
            evaluation = collect_evaluation(
                evaluating[cid],
                submitter=submitter,
                cases=context["cases"],
                references=context["references"],
                baseline_times=baseline["times"],
                baseline_energies=baseline["energies"],
                factors=state["factors"],
            )
            if evaluation is None:
                continue
            evaluation["started_at"] = evaluating[cid].get("started_at")
            append_record(self.trajectory, self._make_record(state, cid, self._answer(state, cid), evaluation))
            del evaluating[cid]
            self.save(state)
        if evaluating:
            state["status"] = "evaluating"
            self.save(state)
            return False
        state.pop("evaluating", None)
        return True

    def advance(self, submitter) -> dict:
        state = self.load()
        if state["status"] == "done":
            return self.status()
        missing = self.requests()
        if missing:
            return self.status()
        context = self._context(state)
        baseline = self.baseline()
        if getattr(submitter, "asynchronous", False):
            if not self._advance_async(state, submitter, context, baseline):
                return self.status()
        else:
            for cid in state["pending"]:
                append_record(self.trajectory, self._record(state, cid, submitter, context, baseline))
        records = read_records(self.trajectory)
        before = best_record([record for record in records if record["generation"] < state["generation"]])
        after = best_record(records)
        improved = after is not None and (
            before is None or float(after["eval"]["fitness"]) > float(before["eval"]["fitness"])
        )
        state["stale_generations"] = 0 if improved else int(state.get("stale_generations", 0)) + 1
        config = state["config"]
        finished = state["generation"] + 1 >= int(config.get("max_generations", 8))
        patience = config.get("patience")
        if patience is not None and state["stale_generations"] >= int(patience):
            finished = True
        state["pending"] = []
        if finished:
            state["status"] = "done"
            self.save(state)
        else:
            self._open_generation(state, state["generation"] + 1)
            state = self.load()
        write_ledger(self.ledger_path, records, state["factors"], state["status"])
        return self.status()

    def status(self) -> dict:
        state = self.load()
        records = read_records(self.trajectory)
        best = best_record(records)
        return {
            "run_id": state["run_id"],
            "status": state["status"],
            "model": state["model"],
            "subagent_model": state["subagent_model"],
            "generation": state["generation"],
            "candidates_scored": len(records),
            "stale_generations": state.get("stale_generations", 0),
            "evaluating": sorted(state.get("evaluating") or {}),
            "best": None
            if best is None
            else {
                "candidate_id": best["candidate_id"],
                "fitness": best["eval"]["fitness"],
                "max_abs_err": best["eval"]["max_abs_err"],
                "geomean_time_ratio": best["eval"]["geomean_time_ratio"],
            },
            "requests": self.requests(),
        }


def answer_with_client(session: Session, client) -> None:
    """Fill pending responses from an in-process client (tests, or an API client)."""
    state = session.load()
    for request in session.requests():
        prompt = Path(request["prompt"]).read_text(encoding="utf-8")
        completion = client.complete(prompt=prompt, factors=state["factors"])
        Path(request["response"]).write_text(completion.text, encoding="utf-8")
        Path(request["meta"]).write_text(json.dumps(completion.meta(), indent=2) + "\n", encoding="utf-8")


def run_with_client(config: dict, client, submitter, out_dir: Path, run_id: str | None = None) -> dict:
    session = Session(out_dir)
    if not session.exists():
        session.init(config, submitter, run_id=run_id)
    while session.load()["status"] != "done":
        answer_with_client(session, client)
        session.advance(submitter)
    return json.loads(session.ledger_path.read_text(encoding="utf-8"))
