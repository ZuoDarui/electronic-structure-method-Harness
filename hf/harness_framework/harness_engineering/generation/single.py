"""Single-shot generation with a small-molecule correctness gate.

One run = one generated RHF program = one accuracy verdict.

* Exactly one candidate ``solution.py`` is produced per run (one model request;
  transport retries of that same request are not new programs). There is no
  repair loop: a failing candidate is reported, and a new run in a fresh output
  directory may supply the previous run's public diagnostics as feedback.
* Acceptance is accuracy only. Every case runs on CPU (NumPy backend) and the
  trusted verifier recomputes energy, J/K, density and SCF-consistency errors
  against independent PySCF references. There is no GPU stage, no external
  CUDA trace, no timing/speedup gate, no large-molecule run and no holdout.
* The candidate never sees references; the model sees only the public task,
  contract, and (optionally) the previous run's public diagnostics.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from .storage import atomic_json, digest, event, load_json, utcnow, write_manifest


SINGLE_SCHEMA = "rhf-single-shot-accuracy/1"
DEFAULT_TASK = Path(__file__).resolve().parents[2] / "harness/generation/rhf_small_accuracy.json"
DEFAULT_MAX_ATOMS = 8
DEFAULT_MAX_NAO = 40
_WORKER = Path(__file__).with_name("worker.py")

ACCURACY_CONTRACT = """Implement only solution.py with these real-valued float64 interfaces:
build_jk(problem, density, backend) -> (J, K)
solve(problem, options, backend) -> dict(energy, density, mo_coeff, mo_energy,
    occupations, converged, history).
This run is a CPU correctness gate: backend is always the string 'numpy'.
Speed, GPU use and large molecules are NOT assessed; prefer clear, numerically
robust NumPy/SciPy code over performance tricks.
problem is a dictionary: overlap=S[n,n], hcore=H[n,n], enuc=float,
nelectron=int, metadata=dict, plus eri[n,n,n,n] for exact mode or
df_factors[P,n,n] for DF mode. Arrays are read-only NumPy arrays. S is full
rank with eigenvalues >1e-10. metadata['integral_mode'] says exact or df.
ERI convention: eri[p,q,r,s]=(pq|rs); DF: (pq|rs)=sum_P B[P,p,q]B[P,r,s].
D=2*C_occ@C_occ.T, J[p,q]=sum_rs eri[p,q,r,s]D[r,s],
K[p,q]=sum_rs eri[p,r,q,s]D[r,s], F=H+J-K/2,
E=Tr[D@(H+F)]/2+enuc. J/K must work for any real symmetric D, including
indefinite matrices, and must be exactly linear in D.
solve options: conv_tol, residual_tol, max_cycle, diis_space,
df_block_size, memory_limit_mb. Use a robust SCF acceleration (e.g. Pulay
DIIS) with energy AND physical residual FDS-SDF convergence; max_cycle is a
hard cap. Nonconvergence returns converged=False, never a fake success.
Output orbitals are S-orthonormal and ordered by ascending energy, occupations
are 2 for the lowest nelectron//2 orbitals then 0. The returned density must
equal (C*occ)@C.T for the returned C and occupations. The returned energy and
orbital energies must correspond to the physical Fock built from the returned
density. Each history row has cycle (increasing positive int), energy and
residual_norm (finite). Return a nonempty history.
Only NumPy, SciPy and approved standard mathematical helpers may be imported.
No PySCF, other QC solver, filesystem, process or network access. Do not read
reference files or hard-code molecular answers.
Acceptance (all small molecules, exact and DF): |E-E_ref|<=1e-8 Eh, trial J/K
within 1e-9, density within 1e-6, FDS-SDF within 1e-7, S-orthonormality within
1e-10, and the other listed tolerances.
"""


class SingleShotError(RuntimeError):
    pass


class InvalidTask(SingleShotError):
    """The task itself violates the small-molecule accuracy protocol."""


def _count_atoms(atom) -> int:
    if isinstance(atom, list):
        return len(atom)
    parts = [p for p in re.split(r"[;\n]", str(atom)) if p.strip()]
    return len(parts)


def validate_accuracy_spec(raw: dict) -> dict:
    """Validate with the shared RHF protocol plus the small-molecule limits."""
    from .spec import validate_spec

    spec = validate_spec(raw)
    limits = spec.get("accuracy_only", {})
    if not isinstance(limits, dict):
        raise ValueError("accuracy_only must be an object")
    max_atoms = limits.get("max_atoms", DEFAULT_MAX_ATOMS)
    max_nao = limits.get("max_nao", DEFAULT_MAX_NAO)
    for name, value in (("max_atoms", max_atoms), ("max_nao", max_nao)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"accuracy_only.{name} must be a positive integer")
    if limits.get("backend", "numpy") != "numpy":
        raise ValueError("the single-shot accuracy gate runs the NumPy backend only")
    for case in spec["cases"]:
        count = _count_atoms(case["atom"])
        if count > max_atoms:
            raise ValueError(f"{case['id']}: {count} atoms exceeds the small-molecule limit {max_atoms}")
    spec["accuracy_only"] = {"max_atoms": max_atoms, "max_nao": max_nao, "backend": "numpy"}
    return spec


def _public_prompt(spec: dict) -> dict:
    return {key: copy.deepcopy(spec[key]) for key in (
        "schema_version", "name", "method", "description", "dependency_boundary", "integral_modes",
        "auxbasis", "cases", "tolerances", "solver", "memory_limit_mb", "accuracy_only",
    ) if key in spec}


def public_feedback(result: dict) -> dict:
    """Diagnostics a later run may show the model; never references or paths."""
    cases = []
    for record in result.get("cases", []):
        cases.append({key: record[key] for key in ("case_id", "status", "metrics", "reasons") if key in record})
    return {"status": result.get("status"), "reasons": list(result.get("reasons", []))[:50], "cases": cases}


class LocalProcessExecutor:
    """Run the worker in a separate host Python process on CPU.

    This is NOT a security sandbox: the generated code executes as the current
    user. It exists for small-molecule accuracy checks where no Docker or
    namespace runtime is available, and must be enabled explicitly with
    ``allow_unisolated_host_execution: true``. Defenses are limited to the AST
    import/operation guard, a scrubbed environment, a private working directory,
    a wall-clock timeout, CPU/file-size rlimits and (on Linux) an address-space limit.
    """

    def __init__(self, config: dict):
        if config.get("allow_unisolated_host_execution") is not True:
            raise ValueError("cpu_executor=subprocess requires allow_unisolated_host_execution=true "
                             "(the candidate runs as the current user without container isolation)")
        self.python = str(config.get("python") or sys.executable)
        self.timeout_s = float(config.get("timeout_s", 300))
        self.memory_mb = int(config.get("memory_mb", config.get("memory_limit_mb", 2048)))
        self.max_log_bytes = int(config.get("max_log_bytes", 1024 * 1024))
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0 or self.memory_mb <= 0:
            raise ValueError("timeout_s and memory_mb must be positive")

    def _env(self, home):
        env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "TMPDIR": str(home),
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            if key in os.environ:
                env[key] = os.environ[key]
        return env

    def _limits(self):
        import resource
        timeout = int(math.ceil(self.timeout_s)) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024 * 1024, 512 * 1024 * 1024))
        if sys.platform.startswith("linux"):
            limit = self.memory_mb * 1024 * 1024 + 1024 * 1024 * 1024  # interpreter + BLAS headroom
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    def _process(self, command, cwd, env):
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True, preexec_fn=self._limits)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return {"command": command, "returncode": process.returncode, "timed_out": timed_out,
                "elapsed_seconds": time.monotonic() - started,
                "stdout": stdout[-self.max_log_bytes:].decode("utf-8", "replace"),
                "stderr": stderr[-self.max_log_bytes:].decode("utf-8", "replace")}

    def preflight(self, backend):
        if backend != "numpy":
            return {"status": "invalid_runtime", "reasons": ["accuracy gate supports numpy only"]}
        with tempfile.TemporaryDirectory(prefix="hf-probe-") as home:
            probe = ("import json,sys,platform,numpy,scipy;print(json.dumps({'python':sys.version.split()[0],"
                     "'platform':platform.platform(),'numpy':numpy.__version__,'scipy':scipy.__version__}))")
            record = self._process([self.python, "-I", "-c", probe], home, self._env(home))
        if record["returncode"] != 0:
            return {"status": "invalid_runtime", "reasons": ["python/numpy/scipy probe failed: " + record["stderr"][-2000:]]}
        return {"status": "passed", "image": None, "executor": "subprocess", "isolated": False,
                "probe": json.loads(record["stdout"].strip().splitlines()[-1]), "reasons": [],
                "require_external_trace": False}

    def run(self, candidate_dir, problem_path, output_dir, backend="numpy", options=None):
        candidate_dir, problem_path, output_dir = Path(candidate_dir), Path(problem_path), Path(output_dir)
        candidate = candidate_dir / "solution.py"
        result = {"status": "invalid_runtime", "returncode": None, "output_path": None,
                  "runtime": {}, "hashes": {}, "reasons": [], "executor": "subprocess", "isolated": False}
        if backend != "numpy":
            result["reasons"].append("accuracy gate supports numpy only")
            return result
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("output directory must be fresh and empty")
        runtime = self.preflight(backend)
        result["runtime"] = runtime
        if runtime["status"] != "passed":
            result["reasons"] = runtime["reasons"]
            return result
        result["candidate_sha256"] = digest(candidate)
        result["input_sha256"] = digest(problem_path)
        result["hashes"] = {"candidate_sha256": result["candidate_sha256"], "input_sha256": result["input_sha256"],
                            "worker_sha256": digest(_WORKER)}
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hf-candidate-") as temporary:
            stage = Path(temporary)
            for name in ("candidate", "input", "runner", "output", "home"):
                (stage / name).mkdir()
            for source, dest in ((candidate, "candidate/solution.py"), (problem_path, "input/problem.npz"),
                                 (_WORKER, "runner/worker.py")):
                shutil.copyfile(source, stage / dest)
                (stage / dest).chmod(0o444)
            (stage / "input/options.json").write_text(json.dumps(dict(options or {}), allow_nan=False))
            command = [self.python, "-I", str(stage / "runner/worker.py"),
                       "--candidate", str(stage / "candidate/solution.py"),
                       "--problem", str(stage / "input/problem.npz"),
                       "--output", str(stage / "output/result.npz"),
                       "--backend", "numpy", "--options", str(stage / "input/options.json")]
            process = self._process(command, stage / "home", self._env(stage / "home"))
            process["command"] = [part.replace(str(stage), "<stage>") for part in command]
            for name in ("result.npz", "worker.json"):
                if (stage / "output" / name).is_file() and not (stage / "output" / name).is_symlink():
                    shutil.copyfile(stage / "output" / name, output_dir / name)
        result.update(returncode=process["returncode"], processes=[process],
                      stdout=process["stdout"], stderr=process["stderr"])
        worker = load_json(output_dir / "worker.json") if (output_dir / "worker.json").is_file() else {}
        result["worker"] = worker
        result_file = output_dir / "result.npz"
        reasons = []
        if process["timed_out"]:
            reasons.append(f"candidate exceeded the {self.timeout_s:g} s wall-clock limit")
        if process["returncode"] != 0:
            reasons.append(f"worker exited with {process['returncode']}: {worker.get('error_type', '')} {worker.get('error', '')}".strip())
        if worker.get("candidate_sha256") != result["candidate_sha256"] or worker.get("input_sha256") != result["input_sha256"]:
            reasons.append("worker hashes differ from the staged candidate/input")
        if not result_file.is_file() or worker.get("status") != "passed":
            reasons.append("worker did not produce a complete result")
        elif worker.get("output_sha256") != digest(result_file):
            reasons.append("result differs from the worker output receipt")
        else:
            result["output_path"] = str(result_file.resolve())
        if digest(candidate) != result["candidate_sha256"] or digest(problem_path) != result["input_sha256"]:
            reasons.append("controller inputs changed during execution")
        result["status"] = "failed" if reasons else "passed"
        result["reasons"] = reasons
        result["hashes"]["artifacts"] = {p.name: digest(p) for p in output_dir.iterdir() if p.is_file()}
        atomic_json(output_dir / "execution.json", result)
        return result


def make_executor(config: dict):
    """CPU-only executors: isolated docker/namespace (existing) or explicit subprocess."""
    kind = config.get("cpu_executor", "docker")
    if kind == "subprocess":
        return LocalProcessExecutor(config)
    if kind not in {"docker", "namespace"}:
        raise ValueError("cpu_executor must be docker, namespace or subprocess")
    from .campaign import _configured_executor
    return _configured_executor(config)


class SingleShotRun:
    """Prepare references -> generate one program -> CPU accuracy verdict."""

    def __init__(self, root, *, generator_factory=None, executor_factory=None, science_module=None):
        from . import science
        self.root = Path(root).resolve()
        self.generator_factory = generator_factory
        self.executor_factory = executor_factory or make_executor
        self.science = science_module or science

    # -- helpers -----------------------------------------------------------
    def _status(self, **fields):
        state_path = self.root / "state.json"
        state = load_json(state_path) if state_path.exists() else {"schema": SINGLE_SCHEMA, "created_at": utcnow()}
        state.update(fields, updated_at=utcnow())
        atomic_json(state_path, state)
        return state

    def _finish(self, result: dict) -> dict:
        result.setdefault("schema", SINGLE_SCHEMA)
        result["finished_at"] = utcnow()
        atomic_json(self.root / "result.json", result)
        self._status(status=result["status"], phase="finished")
        (self.root / "report.md").write_text(render_report(result), encoding="utf-8")
        event(self.root, "single_shot_finished", status=result["status"], reasons=result.get("reasons", [])[:5])
        write_manifest(self.root)
        return result

    # -- stages ------------------------------------------------------------
    def _prepare(self, spec):
        from .spec import expand_cases
        prepared = []
        limit = spec["accuracy_only"]["max_nao"]
        for index, case in enumerate(expand_cases(spec)):
            event(self.root, "preparing_pyscf_reference", case_id=case["id"])
            record = self.science.prepare_case(
                case, case["mode"], self.root / "inputs", self.root / "private/references",
                seed=spec["seed"] + index, memory_limit_mb=spec["memory_limit_mb"])
            nao = record.get("identity", {}).get("nao")
            if isinstance(nao, int) and nao > limit:
                raise InvalidTask(f"{case['id']}: nao={nao} exceeds the small-molecule limit {limit}")
            prepared.append(record)
        atomic_json(self.root / "prepared.json", prepared)
        return prepared

    def _generate(self, spec, generation_config, feedback, previous_source):
        from .generator import GenerationError, ResponsesGenerator, apply_bundle, initial_source

        factory = self.generator_factory or ResponsesGenerator
        context = {"public_prompt": _public_prompt(spec), "contract": ACCURACY_CONTRACT,
                   "current_source": previous_source or initial_source(), "round": 1}
        if feedback:
            context["dev_feedback"] = feedback
        event(self.root, "requesting_single_candidate", model=generation_config.get("model"),
              with_feedback=bool(feedback))
        try:
            bundle = factory(generation_config).generate(context, self.root / "generation")
            atomic_json(self.root / "bundle.json", bundle)
            candidate_dir = self.root / "candidate"
            # The AST/import guard rejects the program; it is not an environment fault.
            applied = apply_bundle(bundle, candidate_dir)
        except GenerationError as exc:
            return None, {"status": exc.status, "reason": exc.reason}
        atomic_json(self.root / "applied.json", applied)
        event(self.root, "candidate_generated", sha256=digest(candidate_dir / "solution.py"),
              rationale=bundle.get("rationale"))
        return candidate_dir, None

    def _use_supplied(self, source_path):
        from .generator import GenerationError, apply_bundle
        source = Path(source_path).read_text(encoding="utf-8")
        bundle = {"files": [{"path": "solution.py", "content": source}],
                  "rationale": f"Supplied candidate {Path(source_path).name}; no model request.",
                  "expected_effect": "Accuracy assessment of an existing program."}
        try:
            applied = apply_bundle(bundle, self.root / "candidate")  # same AST guard as generated code
        except GenerationError as exc:
            return None, {"status": exc.status, "reason": exc.reason}
        atomic_json(self.root / "applied.json", applied)
        event(self.root, "candidate_supplied", sha256=digest(self.root / "candidate/solution.py"))
        return self.root / "candidate", None

    def _evaluate(self, spec, prepared, executor, candidate_dir):
        options = {**spec["solver"], "memory_limit_mb": spec["memory_limit_mb"]}
        options.setdefault("conv_tol", 1e-10)
        options.setdefault("residual_tol", 1e-8)
        records = []
        for case in prepared:
            case_root = self.root / "cases" / case["id"]
            case_root.mkdir(parents=True, exist_ok=False)
            event(self.root, "executing_case", case_id=case["id"], nao=case.get("identity", {}).get("nao"))
            started = time.monotonic()
            execution = executor.run(candidate_dir, Path(case["problem_path"]), case_root / "output", "numpy", options)
            elapsed = time.monotonic() - started
            atomic_json(case_root / "execution.json", execution)
            if execution.get("status") == "invalid_runtime":
                raise SingleShotError("execution runtime unavailable: " + "; ".join(map(str, execution.get("reasons", []))))
            output = Path(execution.get("output_path") or case_root / "output/result.npz")
            record = self.science.verify_case(case, output, spec["tolerances"], execution)
            if record["status"] == "invalid_runtime" and execution.get("status") != "passed":
                # A crashing or timed-out candidate is a scientific failure of
                # this program, not an unavailable environment.
                record["status"] = "failed"
                record["reasons"] = list(execution.get("reasons", [])) or record["reasons"]
            atomic_json(case_root / "verification.json", record)
            details = record.get("details", {})
            records.append({"case_id": case["id"], "status": record["status"], "metrics": record.get("metrics", {}),
                            "reasons": record.get("reasons", []), "nao": case.get("identity", {}).get("nao"),
                            "candidate_energy": details.get("candidate_energy"),
                            "reference_energy": details.get("reference_energy"),
                            "scf_cycles": len(details.get("history", [])) or None,
                            "wall_seconds_informational": round(elapsed, 3)})
            event(self.root, "case_verified", case_id=case["id"], status=record["status"],
                  energy_error=record.get("metrics", {}).get("energy_max_abs"))
        suite = self.science.verify_suite([c["id"] for c in prepared],
                                          [{"case_id": r["case_id"], "status": r["status"], "metrics": r["metrics"],
                                            "reasons": r["reasons"]} for r in records])
        return suite, records

    # -- entry point -------------------------------------------------------
    def run(self, task_path, *, execution_config, generation_config=None, candidate_source=None,
            previous_run=None):
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError("A single-shot run requires a new, empty output directory")
        self.root.mkdir(parents=True, exist_ok=True)
        spec = validate_accuracy_spec(load_json(task_path))
        execution = dict(execution_config or {})
        execution.setdefault("timeout_s", spec.get("execution", {}).get("timeout_s", 300))
        execution.setdefault("memory_mb", spec["memory_limit_mb"])
        execution.update(stop_before_gpu=True, require_external_trace=False, defer_gpu_preflight=True)
        generation = dict(generation_config or {})
        for key in generation:
            if key.lower() in {"api_key", "authorization", "token", "secret"}:
                raise ValueError("API credentials must be provided through an environment variable")
        if candidate_source is None:
            from .generator import validate_budget
            if not generation.get("model"):
                raise ValueError("Specify --model or OPENAI_MODEL, or pass --candidate to assess an existing program")
            validate_budget(generation)
        feedback, previous_source = None, None
        if previous_run is not None:
            previous = Path(previous_run).resolve()
            previous_result = load_json(previous / "result.json")
            feedback = public_feedback(previous_result)
            if (previous / "candidate/solution.py").is_file():
                previous_source = (previous / "candidate/solution.py").read_text(encoding="utf-8")
        (self.root / "private").mkdir(mode=0o700)
        atomic_json(self.root / "spec.json", spec)
        atomic_json(self.root / "execution_config.json", execution)
        atomic_json(self.root / "generation_config.json",
                    {k: v for k, v in generation.items() if k not in {"transport", "sleep"}})
        self._status(status="active", phase="preflight", previous_run=str(previous_run) if previous_run else None)
        base = {"task": spec["name"], "model": generation.get("model"),
                "reasoning_effort": generation.get("reasoning_effort"),
                "mode": "supplied_candidate" if candidate_source else "generated",
                "previous_run": str(previous_run) if previous_run else None,
                "scope": ("Accuracy only: one program, CPU/NumPy, small molecules, PySCF references. "
                          "No speed, GPU or large-system assessment.")}
        try:
            executor = self.executor_factory(execution)
            runtime = executor.preflight("numpy")
            atomic_json(self.root / "preflight.json", runtime)
            if runtime.get("status") != "passed":
                return self._finish({**base, "status": "environment_unavailable",
                                     "reasons": list(map(str, runtime.get("reasons", [runtime])))})
            self._status(phase="preparing_references")
            prepared = self._prepare(spec)
            self._status(phase="generating")
            if candidate_source is not None:
                candidate_dir, failure = self._use_supplied(candidate_source)
            else:
                candidate_dir, failure = self._generate(spec, generation, feedback, previous_source)
            if failure:
                return self._finish({**base, "status": "generation_failed", "reasons": [failure["reason"]],
                                     "generation_error": failure, "api_usage": _usage(self.root)})
            self._status(phase="accuracy_assessment")
            suite, records = self._evaluate(spec, prepared, executor, candidate_dir)
        except InvalidTask as exc:
            return self._finish({**base, "status": "invalid_task", "reasons": [str(exc)]})
        except (SingleShotError, RuntimeError, OSError, ValueError) as exc:
            return self._finish({**base, "status": "environment_unavailable", "reasons": [f"{type(exc).__name__}: {exc}"]})
        accepted = bool(suite["passed"])
        if accepted:
            (self.root / "accepted").mkdir()
            shutil.copyfile(candidate_dir / "solution.py", self.root / "accepted/solution.py")
        return self._finish({**base, "status": "accepted" if accepted else "rejected",
                             "candidate_sha256": digest(candidate_dir / "solution.py"),
                             "case_count": len(records), "passed_count": sum(r["status"] == "passed" for r in records),
                             "max_metrics": suite["metrics"], "tolerances": spec["tolerances"],
                             "reasons": suite["reasons"], "cases": records,
                             "runtime": {"executor": execution.get("cpu_executor", "docker"),
                                         "probe": runtime.get("probe"), "isolated": runtime.get("isolated", True)},
                             "api_usage": _usage(self.root)})


def _usage(root: Path) -> dict:
    totals = {}
    for path in sorted(Path(root).glob("generation/attempt-*/response.json")):
        usage = load_json(path).get("usage", {})
        for key, value in (usage.items() if isinstance(usage, dict) else ()):
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    return totals


def _fmt(value, spec=".2e"):
    return "-" if value is None else format(value, spec)


def render_report(result: dict) -> str:
    lines = ["# RHF single-shot accuracy gate", "",
             f"Status: **{result['status']}**  ", f"Task: `{result.get('task')}`  ",
             f"Candidate: {result.get('mode')}; model `{result.get('model')}`, reasoning `{result.get('reasoning_effort')}`  ",
             f"Scope: {result.get('scope')}", ""]
    if result.get("cases"):
        lines += [f"Cases passed: {result['passed_count']} / {result['case_count']}", "",
                  "| case | nao | E candidate (Eh) | E PySCF (Eh) | abs dE | max dD | max dJ/K | FDS-SDF | cycles | result |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in result["cases"]:
            m = r.get("metrics", {})
            lines.append(f"| {r['case_id']} | {r.get('nao', '-')} | {_fmt(r.get('candidate_energy'), '.10f')} | "
                         f"{_fmt(r.get('reference_energy'), '.10f')} | {_fmt(m.get('energy_max_abs'))} | "
                         f"{_fmt(m.get('density_max_abs'))} | {_fmt(m.get('jk_max_abs'))} | "
                         f"{_fmt(m.get('residual_max_abs'))} | {r.get('scf_cycles') or '-'} | {r['status']} |")
        lines += ["", "Worst-case metrics vs tolerance:", ""]
        for key, value in result.get("max_metrics", {}).items():
            tol = result.get("tolerances", {}).get(key)
            flag = "ok" if tol is not None and value <= tol else "FAIL"
            lines.append(f"- `{key}`: {value:.3e} (tol {tol:.0e}) {flag}")
    if result.get("reasons"):
        lines += ["", "Reasons:", ""] + [f"- {reason}" for reason in result["reasons"][:40]]
    if result.get("status") == "accepted":
        lines += ["", "Accepted source: `accepted/solution.py`."]
    lines += ["", "Correctness is limited to the listed small molecules, exact/DF modes and CPU float64. "
              "References are PySCF RHF/DF-RHF from the same integrals; no speed claim is made.", ""]
    return "\n".join(lines)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="generate single",
        description="Generate ONE RHF program and accept it only on small-molecule accuracy vs PySCF (CPU, no speed gate)")
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-config", type=Path, default=None,
                        help="JSON executor config (docker/namespace/subprocess). Required unless --local-subprocess")
    parser.add_argument("--local-subprocess", action="store_true",
                        help="Run candidates in an unisolated host Python subprocess (explicit opt-in)")
    parser.add_argument("--python", default=None, help="Interpreter for --local-subprocess (needs numpy/scipy)")
    parser.add_argument("--candidate", type=Path, default=None,
                        help="Assess an existing solution.py instead of calling the model")
    parser.add_argument("--previous-run", type=Path, default=None,
                        help="Show the previous run's public diagnostics and source to the model")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-timeout", type=float, default=600)
    parser.add_argument("--budget-profile", choices=("standard", "extended"), default="standard")
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args(argv)
    if args.executor_config is None and not args.local_subprocess:
        parser.error("give --executor-config or explicitly choose --local-subprocess")
    execution = load_json(args.executor_config) if args.executor_config else {}
    if args.local_subprocess:
        execution.update(cpu_executor="subprocess", allow_unisolated_host_execution=True)
        if args.python:
            execution["python"] = args.python
    generation = None
    if args.candidate is None:
        generation = {"model": args.model, "reasoning_effort": args.reasoning_effort, "base_url": args.api_base_url,
                      "api_key_env": args.api_key_env, "timeout_s": args.api_timeout,
                      "budget_profile": args.budget_profile, "max_output_tokens": args.max_output_tokens,
                      "max_retries": 2, "streaming": args.stream}
    try:
        result = SingleShotRun(args.output).run(args.task, execution_config=execution, generation_config=generation,
                                                candidate_source=args.candidate, previous_run=args.previous_run)
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    summary = {key: result.get(key) for key in ("status", "case_count", "passed_count", "reasons")}
    summary["reasons"] = (summary["reasons"] or [])[:5]
    summary["report"] = str(Path(args.output).resolve() / "report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "accepted" else 1
