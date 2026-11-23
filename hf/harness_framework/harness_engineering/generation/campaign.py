"""Persistent generation, repair, freeze, blind assessment and promotion.

The controller owns references and scores. The model receives only public
development inputs and diagnostics, and the candidate receives only integrals.
Runtime isolation must be observed, never inferred from submitted Docker flags.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.parse
import uuid

from . import SCHEMA_VERSION
from .storage import (atomic_json, campaign_lock, digest, event, inventory,
                      load_json, utcnow, verify_inventory, write_manifest)


CONTRACT = """Implement only solution.py with these real-valued float64 interfaces:
build_jk(problem, density, backend) -> (J, K)
solve(problem, options, backend) -> dict(energy, density, mo_coeff, mo_energy,
    occupations, converged, history).
backend is the string 'numpy' or 'cupy'. For cupy, keep the main matrices,
integrals, orbitals, diagonalization and contractions on the CUDA device;
return device matrices, with scalar diagnostics allowed on the host.
problem is a dictionary: overlap=S[n,n], hcore=H[n,n], enuc=float,
nelectron=int, metadata=dict, plus eri[n,n,n,n] for exact mode or
df_factors[P,n,n] for DF mode. Arrays are NumPy arrays on entry. S is full
rank with eigenvalues >1e-10. metadata['integral_mode'] says exact or df.
ERI convention: eri[p,q,r,s]=(pq|rs); DF: (pq|rs)=sum_P B[P,p,q]B[P,r,s].
D=2*C_occ@C_occ.T, J[p,q]=sum_rs eri[p,q,r,s]D[r,s],
K[p,q]=sum_rs eri[p,r,q,s]D[r,s], F=H+J-K/2,
E=Tr[D@(H+F)]/2+enuc. J/K must work for any real symmetric D, including
indefinite matrices. Do not use occupations to change the definition of K.
solve options: conv_tol, residual_tol, max_cycle, diis_space,
df_block_size, memory_limit_mb. Use a robust SCF acceleration (e.g. Pulay
DIIS) with energy AND physical F(D) residual convergence; max_cycle is a
hard cap. Nonconvergence returns converged=False, never a fake success.
Output orbitals are S-orthonormal and ordered by their energies, occupations
are 2 for the lowest nelectron//2 orbitals then 0. Density reconstructs from
the returned coefficients and occupations. Energy and orbital energies must
correspond to the physical Fock at the final returned density. Each history
row has cycle (increasing positive int), energy and residual_norm (finite);
record other finite diagnostics if helpful. Return a nonempty history.
Only NumPy, SciPy, CuPy and approved standard mathematical helpers may be
imported. No PySCF, other QC solver, production HF module, filesystem or
network access. Do not read reference files or hard-code molecular answers.
Do not change the mathematical method to gain convergence or performance.
Exact stores n^4 integrals; DF exchanges should block along P to control
workspace. GPU stages are observed by a trusted external CUDA profiler.
For development GPU diagnostics, each build_jk invocation must use an
observable CuPy contraction (e.g. cupy.einsum or cupy.dot) and solve must use
cupy.linalg.eigh. Plain ndarray @ alone is not intercepted by Python hooks.
Never satisfy this with dummy operations: the observed work must compute the
requested physical J/K and SCF. Final acceptance also requires external tracing.
memory_limit_mb is in decimal MB (1e6 bytes). Estimate resident integral and
matrix storage plus bounded workspace before allocating large backend arrays;
if they cannot fit, raise MemoryError rather than clamping workspace to one
slice and ignoring the exceeded budget. This estimate is not an OS RSS limit.
"""

TERMINAL = {"accepted", "rejected", "exhausted"}
STAGES = ("dev_cpu", "dev_properties", "dev_gpu", "holdout_cpu", "holdout_gpu")


def _implementation_identity():
    return inventory(sorted(Path(__file__).parent.glob("*.py")))


def _public_prompt(spec):
    # Whitelist, rather than trying to remove private keys from a large state.
    return {key: copy.deepcopy(spec[key]) for key in (
        "schema_version", "name", "method", "dependency_boundary", "integral_modes",
        "auxbasis", "cases", "tolerances", "solver", "memory_limit_mb",
    ) if key in spec}


def _public_runtime(identities):
    return {backend: {key: value for key, value in identity.get("probe", {}).items()
                      if key in {"python", "numpy", "scipy", "cupy", "cuda_runtime", "cuda_driver"}}
            for backend, identity in identities.items()}


def _configured_executor(config):
    from .executor import DockerExecutor
    kind = config.get("cpu_executor", "docker")
    if kind == "docker":
        return DockerExecutor(config)
    if kind != "namespace":
        raise ValueError("cpu_executor must be docker or namespace")
    from .namespace_executor import NamespaceExecutor
    cpu = NamespaceExecutor(config)
    class BackendExecutor:
        def preflight(self, backend):
            return cpu.preflight(backend) if backend == "numpy" else DockerExecutor(config).preflight(backend)
        def run(self, candidate_dir, problem_path, output_dir, backend, options):
            executor = cpu if backend == "numpy" else DockerExecutor(config)
            return executor.run(candidate_dir, problem_path, output_dir, backend, options)
    return BackendExecutor()


def _feedback(report):
    return {"status": report["status"], "metrics": report.get("metrics", {}),
            "reasons": report.get("reasons", []),
            "cases": [{key: record[key] for key in ("case_id", "kind", "status", "metrics", "reasons")
                       if key in record} for record in report.get("records", [])]}


def _approximation_diagnostics(records):
    by_id = {r["case_id"]: r for r in records}
    diagnostics = []
    for case_id, exact in by_id.items():
        if not case_id.endswith("-exact"):
            continue
        fitted = by_id.get(case_id[:-6] + "-df")
        if not fitted or exact["status"] != "passed" or fitted["status"] != "passed":
            continue
        values = {}
        for label, key in (("candidate", "candidate_energy"), ("reference", "reference_energy")):
            if key in exact.get("details", {}) and key in fitted.get("details", {}):
                values[label + "_df_vs_exact_energy_abs"] = abs(exact["details"][key] - fitted["details"][key])
        if values:
            diagnostics.append({"base_case_id": case_id[:-6], **values,
                                "scope": "Approximation difference, separate from matching-method implementation error"})
    return diagnostics


class Campaign:
    """Single-writer state machine. Dependencies can be injected by tests only."""

    def __init__(self, root, *, generator_factory=None, executor_factory=None, science_module=None):
        from . import science
        from .generator import ResponsesGenerator

        self.root = Path(root).resolve()
        self.generator_factory = generator_factory or ResponsesGenerator
        self.executor_factory = executor_factory or _configured_executor
        self.science = science_module or science
        self.state = None

    def _save(self):
        self.state["updated_at"] = utcnow()
        atomic_json(self.root / "state.json", self.state)
        self._write_report()

    def _write_report(self):
        state = self.state
        lines = ["# Electronic-structure generation campaign", "",
                 f"Status: **{state['status']}**", f"Phase: `{state['phase']}`", "",
                 f"Model: `{state['model']}`; reasoning: `{state.get('reasoning_effort', 'default')}`.",
                 f"Generation rounds used: {state['round']} / {state['max_rounds']}.",
                 "", "This report summarizes recorded actions and measurements. It does not expose hidden model reasoning.", ""]
        if state.get("blocked_reason"):
            lines += ["Execution requires attention: " + state["blocked_reason"], ""]
        for stage, entry in state.get("stage_reports", {}).items():
            lines += [f"- {stage}: {entry['status']}; evidence: `{entry['path']}`"]
        if state.get("accepted_candidate"):
            lines += ["", "Accepted source: `accepted/solution.py`.",
                      "Correctness is limited to the recorded cases, modes, precision and devices. No speedup claim."]
        lines += ["", "Evidence: events.jsonl, immutable round snapshots, requests/responses, independent references, raw arrays and manifest.json.",
                  "References share the trusted PySCF integral engine; integral-engine independence is not claimed.", ""]
        (self.root / "report.md").write_text("\n".join(lines), encoding="utf-8")

    def _initialize(self, spec_path, holdout_path, model, reasoning_effort, execution_config, generation_config):
        from .generator import initial_source, validate_budget
        from .spec import load_spec

        if (self.root / "state.json").exists():
            raise ValueError("Campaign already exists; use resume")
        if any(path.name != ".controller.lock" for path in self.root.iterdir()):
            raise ValueError("A new campaign requires an empty output directory")
        model = model or os.environ.get("OPENAI_MODEL")
        if not model or not isinstance(model, str):
            raise ValueError("Specify --model or OPENAI_MODEL; no model is substituted")
        spec, holdout = load_spec(spec_path), load_spec(holdout_path)
        if spec["method"] != holdout["method"] or spec["integral_modes"] != holdout["integral_modes"]:
            raise ValueError("Development and holdout must assess the same method and modes")
        if spec["tolerances"] != holdout["tolerances"]:
            raise ValueError("Development and holdout correctness thresholds must match")
        if {c["id"] for c in spec["cases"]} & {c["id"] for c in holdout["cases"]}:
            raise ValueError("Development and holdout case identities must be disjoint")
        config = dict(generation_config or {})
        if set(config) - {"base_url", "api_key_env", "timeout_s", "max_output_tokens", "max_retries", "streaming", "budget_profile"}:
            raise ValueError("Unsupported generation configuration (credentials are never stored here)")
        validate_budget(config)
        for key, value in config.items():
            if key.lower() in {"api_key", "authorization", "token", "secret"}:
                raise ValueError("API credentials must be provided through an environment variable")
        endpoint = urllib.parse.urlsplit(config.get("base_url", "https://api.openai.com/v1"))
        if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ValueError("Responses URL must use HTTPS without embedded credentials, query or fragment")
        config.update({"model": model, "reasoning_effort": reasoning_effort})
        if "streaming" in config and type(config["streaming"]) is not bool:
            raise ValueError("streaming must be a boolean")
        execution = dict(execution_config or {})
        execution.setdefault("timeout_s", spec.get("execution", {}).get("timeout_s", 300))
        execution.setdefault("memory_mb", spec["memory_limit_mb"])
        execution.setdefault("require_external_trace", True)
        # Freeze specifications before any model call; private material is not
        # sent in prompts or mounted into a candidate container.
        (self.root / "private").mkdir(mode=0o700)
        (self.root / "active").mkdir()
        (self.root / "initial").mkdir()
        atomic_json(self.root / "spec.json", spec)
        atomic_json(self.root / "private/holdout.json", holdout)
        atomic_json(self.root / "execution_config.json", execution)
        atomic_json(self.root / "generation_config.json", config)
        (self.root / "initial/solution.py").write_text(initial_source())
        shutil.copyfile(self.root / "initial/solution.py", self.root / "active/solution.py")
        frozen = _implementation_identity()
        frozen.update(inventory([self.root / "spec.json", self.root / "private/holdout.json",
                                 self.root / "execution_config.json", self.root / "generation_config.json",
                                 self.root / "initial/solution.py"]))
        self.state = {
            "schema_version": SCHEMA_VERSION, "status": "active", "phase": "preflight",
            "created_at": utcnow(), "model": model, "reasoning_effort": reasoning_effort,
            "round": 0, "max_rounds": spec.get("generation", {}).get("max_rounds", 8),
            "trusted_identity": frozen, "prepared": {"dev": [], "properties": [], "holdout": []},
            "stage_reports": {}, "feedback": {}, "last_cpu_correct": None,
            "holdout_started": False, "rounds": [], "api_usage": {},
            "evidence_identity": {},
        }
        event(self.root, "initialized", model=model, reasoning_effort=reasoning_effort,
              empty_source_sha256=digest(self.root / "initial/solution.py"))
        self._save()

    def run(self, spec_path, holdout_path, *, model=None, reasoning_effort=None,
            execution_config=None, generation_config=None):
        with campaign_lock(self.root):
            self._initialize(spec_path, holdout_path, model, reasoning_effort,
                             execution_config, generation_config)
            return self._advance()

    def resume(self, *, enable_gpu=False):
        with campaign_lock(self.root):
            self.state = load_json(self.root / "state.json")
            self._check_identity()
            if self.state["status"] in TERMINAL:
                return copy.deepcopy(self.state)
            if enable_gpu:
                self.state["gpu_authorized"] = True
                event(self.root, "gpu_work_explicitly_enabled")
            self.state["status"] = "active"
            self.state.pop("blocked_reason", None)
            event(self.root, "resumed", phase=self.state["phase"], round=self.state["round"])
            return self._advance()

    def _check_identity(self):
        if self.state.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Unsupported campaign state version")
        verify_inventory(self.state["trusted_identity"])
        verify_inventory(self.state.get("evidence_identity", {}))
        if self.state.get("candidate_path"):
            verify_inventory({self.state["candidate_path"]: self.state["candidate_sha256"]})
        for dataset in self.state["prepared"].values():
            for case in dataset:
                verify_inventory({case["problem_path"]: case["input_sha256"],
                                  case["reference_path"]: case["reference_sha256"]})
        if self.state.get("frozen_candidate"):
            verify_inventory({self.state["frozen_candidate"]: self.state["frozen_sha256"]})
        if self.state.get("accepted_candidate"):
            verify_inventory({self.state["accepted_candidate"]: self.state["frozen_sha256"]})
        for saved in self.state.get("stage_reports", {}).values():
            verify_inventory({saved["path"]: saved["sha256"]})

    def _seal(self, paths):
        """Completed evidence is append-only across checkpoints, not just at exit."""
        expected = self.state.setdefault("evidence_identity", {})
        for path in paths:
            path = Path(path)
            if path.is_file():
                name = str(path.resolve())
                if name in expected:
                    verify_inventory({name: expected[name]})
                else:
                    expected[name] = digest(path)

    def _block(self, reason):
        self.state["status"] = "waiting_for_environment"
        self.state["blocked_reason"] = str(reason)
        event(self.root, "environment_unavailable", phase=self.state["phase"], reason=str(reason))
        self._save()
        write_manifest(self.root)
        return copy.deepcopy(self.state)

    def _preflight(self):
        config = load_json(self.root / "execution_config.json")
        executor = self.executor_factory(config)
        records = {}
        deferred = config.get("defer_gpu_preflight", False)
        if type(deferred) is not bool:
            raise ValueError("defer_gpu_preflight must be a boolean")
        cpu_phase = self.state["phase"] in {"preflight", "preparing", "generate", "generating", "dev_cpu", "dev_properties"}
        stop_gpu = config.get("stop_before_gpu", False)
        if type(stop_gpu) is not bool:
            raise ValueError("stop_before_gpu must be a boolean")
        stop_gpu = stop_gpu and not self.state.get("gpu_authorized", False)
        backends = ("numpy",) if stop_gpu or (deferred and cpu_phase) else ("numpy", "cupy")
        for backend in backends:
            records[backend] = executor.preflight(backend)
        atomic_json(self.root / "preflight.json", records)
        failures = [f"{key}: {value.get('reason', value.get('reasons', value))}"
                    for key, value in records.items() if value.get("status") != "passed"]
        if failures:
            return "; ".join(failures)
        identities = {key: {"image": value.get("image"), "probe": value.get("probe")}
                      for key, value in records.items()}
        previous = self.state.setdefault("runtime_identity", {})
        if any(key in previous and value != previous[key] for key, value in identities.items()):
            return "Actual container/dependency/device identity differs from this campaign's frozen runtime"
        previous.update(identities)
        self._save()
        return None

    def _prepare(self):
        from .spec import expand_cases

        for dataset, spec_path in (("dev", self.root / "spec.json"),
                                   ("holdout", self.root / "private/holdout.json")):
            spec = load_json(spec_path)
            completed = {case["id"] for case in self.state["prepared"][dataset]}
            for index, case in enumerate(expand_cases(spec)):
                if case["id"] in completed:
                    continue
                event(self.root, "preparing_integrals_and_reference", dataset=dataset, case_id=case["id"])
                prepared = self.science.prepare_case(
                    case, case["mode"], self.root / "inputs" / dataset,
                    self.root / "private/references" / dataset,
                    seed=spec["seed"] + index, memory_limit_mb=spec["memory_limit_mb"],
                )
                self.state["prepared"][dataset].append(prepared)
                self._save()
        development = {p["id"]: p for p in self.state["prepared"]["dev"]}
        dev_identities = {p.get("identity", {}).get("physical_input_sha256") for p in development.values()}
        holdout_identities = {p.get("identity", {}).get("physical_input_sha256")
                              for p in self.state["prepared"]["holdout"]}
        if (dev_identities & holdout_identities) - {None}:
            raise ValueError("Holdout duplicates a resolved physical development input, even if its id differs")
        spec = load_json(self.root / "spec.json")
        complete = {p["id"] for p in self.state["prepared"]["properties"]}
        challenges = self.science.make_challenge_cases(spec)
        for index, case in enumerate(challenges):
            if case["id"] in complete:
                continue
            prepared = self.science.prepare_case(
                case, case["mode"], self.root / "inputs/properties", self.root / "private/references/properties",
                seed=spec["seed"] + 100 + index, memory_limit_mb=spec["memory_limit_mb"],
            )
            prepared["challenge"] = case["challenge"]
            self.state["prepared"]["properties"].append(prepared)
            self._save()
        # Reuse identical public integrals and private references to test the
        # block algorithm and the one-cycle termination contract independently.
        if challenges:
            base_ids = [c["challenge"]["base_case_id"] for c in challenges]
            exact = next((development[name] for name in base_ids if name.endswith("-exact")), None)
            fitted = next((development[name] for name in base_ids if name.endswith("-df")), None)
            variants = []
            if fitted:
                for block in (1, 7):
                    variants.append({**copy.deepcopy(fitted), "id": fitted["id"] + f"-block{block}",
                                     "options_override": {"df_block_size": block},
                                     "challenge": {"kind": "df_block", "base_case_id": fitted["id"]}})
            if exact:
                variants.append({**copy.deepcopy(exact), "id": exact["id"] + "-one-cycle",
                                 "options_override": {"max_cycle": 1},
                                 "challenge": {"kind": "nonconvergence", "base_case_id": exact["id"]}})
            known = {p["id"] for p in self.state["prepared"]["properties"]}
            self.state["prepared"]["properties"].extend(p for p in variants if p["id"] not in known)
            self._save()

    def _generate(self):
        from .generator import GenerationError, apply_bundle

        if self.state["holdout_started"]:
            raise RuntimeError("Generation is forbidden after blind assessment has begun")
        if self.state["phase"] == "generating":
            # A completed bundle is a durable boundary; a lost/incomplete API
            # request consumes its round and is never silently replayed.
            round_root = self.root / "rounds" / f"{self.state['round']:04d}"
            if (round_root / "bundle.json").is_file():
                bundle = load_json(round_root / "bundle.json")
            elif (round_root / "generation/bundle.json").is_file():
                from .generator import validate_bundle
                bundle = validate_bundle(load_json(round_root / "generation/bundle.json"))
                atomic_json(round_root / "bundle.json", bundle)
                event(self.root, "recovered_model_response", round=self.state["round"])
            else:
                event(self.root, "interrupted_generation_recorded", round=self.state["round"])
                self.state["phase"] = "generate"
                self._save()
                return
        else:
            if self.state["round"] >= self.state["max_rounds"]:
                self.state.update(status="exhausted", phase="finished")
                event(self.root, "generation_budget_exhausted", rounds=self.state["round"])
                self._save()
                return
            self.state["round"] += 1
            round_root = self.root / "rounds" / f"{self.state['round']:04d}"
            round_root.mkdir(parents=True, exist_ok=False)
            self.state["phase"] = "generating"
            self.state["rounds"].append({"round": self.state["round"], "status": "generating"})
            self._save()
            source = Path(self.state.get("repair_source", self.root / "active/solution.py")).read_text()
            public_prompt = _public_prompt(load_json(self.root / "spec.json"))
            public_prompt["target_runtime"] = _public_runtime(self.state.get("runtime_identity", {}))
            context = {"public_prompt": public_prompt,
                       "contract": CONTRACT, "current_source": source,
                       "dev_feedback": self.state.get("feedback", {}), "round": self.state["round"]}
            event(self.root, "requesting_candidate", round=self.state["round"], model=self.state["model"])
            try:
                bundle = self.generator_factory(load_json(self.root / "generation_config.json")).generate(
                    context, round_root / "generation")
                atomic_json(round_root / "bundle.json", bundle)
            except GenerationError as exc:
                self._seal(round_root.rglob("*"))
                self.state["rounds"][-1].update(status=exc.status, reason=exc.reason)
                event(self.root, "generation_failed", round=self.state["round"], status=exc.status, reason=exc.reason)
                if exc.status in {"config_error", "configuration_error", "authentication_error", "model_unavailable",
                                   "api_error", "http_error", "network_error", "output_budget_exceeded", "invalid_usage",
                                   "invalid_stream", "stream_limit"}:
                    self.state["phase"] = "generate"
                    self._save()
                    self._block(exc.reason)
                    return
                self.state["feedback"] = {"status": exc.status, "reasons": [exc.reason]}
                self.state["phase"] = "generate"
                self._save()
                return
        candidate_dir = round_root / "candidate"
        candidate_dir.mkdir(exist_ok=True)
        # Parent and diff describe the actual parent, including a failed source
        # supplied for repair. Active always has a recoverable known baseline.
        parent = Path(self.state.get("repair_source", self.root / "active/solution.py"))
        if not (candidate_dir / "solution.py").exists():
            shutil.copyfile(parent, candidate_dir / "solution.py")
        applied = apply_bundle(bundle, candidate_dir)
        atomic_json(round_root / "applied.json", applied)
        path = candidate_dir / "solution.py"
        shutil.copyfile(path, self.root / "active/solution.py")
        self.state.update(candidate_path=str(path), candidate_sha256=digest(path), phase="dev_cpu")
        self.state["rounds"][-1].update(status="evaluating", candidate_sha256=digest(path),
                                        rationale=bundle["rationale"], expected_effect=bundle["expected_effect"])
        self._seal(round_root.rglob("*"))
        event(self.root, "candidate_applied", round=self.state["round"], sha256=digest(path),
              rationale=bundle["rationale"], expected_effect=bundle["expected_effect"])
        self._record_usage()
        self._save()

    def _record_usage(self):
        totals = {}
        for path in sorted((self.root / "rounds").glob("*/generation/attempt-*/response.json")):
            response = load_json(path)
            usage = response.get("usage", {})
            for key, value in (usage.items() if isinstance(usage, dict) else ()):
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[key] = totals.get(key, 0) + value
        self.state["api_usage"] = totals

    def _evaluate_stage(self, stage):
        dataset = "holdout" if stage.startswith("holdout") else "properties" if stage == "dev_properties" else "dev"
        backend = "cupy" if stage.endswith("gpu") else "numpy"
        spec = load_json(self.root / ("private/holdout.json" if dataset == "holdout" else "spec.json"))
        config = load_json(self.root / "execution_config.json")
        # Development diagnostics are allowed before final freezing. Final GPU
        # always requires externally observed CUDA kernels, never just wrapping.
        config["require_external_trace"] = stage == "holdout_gpu"
        executor = self.executor_factory(config)
        stage_root = self.root / "rounds" / f"{self.state['round']:04d}" / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        records = []
        for prepared in self.state["prepared"][dataset]:
            case_root = stage_root / prepared["id"]
            case_root.mkdir(exist_ok=True)
            execution_path = case_root / "execution.json"
            if execution_path.exists():
                execution = load_json(execution_path)
            else:
                self._check_identity()
                output_dir = case_root / "output"
                collected = output_dir / "execution.json"
                if collected.is_file():
                    # Executor collection completed before the controller's
                    # checkpoint. Its receipt and every artifact are checked below.
                    execution = load_json(collected)
                    atomic_json(execution_path, execution)
                    event(self.root, "recovered_execution_receipt", stage=stage, case_id=prepared["id"])
                else:
                    if output_dir.exists() and any(output_dir.iterdir()):
                        orphan = case_root / ("interrupted-output-" + uuid.uuid4().hex)
                        output_dir.rename(orphan)
                        event(self.root, "preserved_interrupted_output", stage=stage,
                              case_id=prepared["id"], evidence=str(orphan))
                    event(self.root, "executing_case", stage=stage, case_id=prepared["id"], backend=backend)
                    execution = executor.run(
                        Path(self.state["candidate_path"]).parent, Path(prepared["problem_path"]),
                        output_dir, backend,
                        {**spec.get("solver", {}), "memory_limit_mb": spec["memory_limit_mb"],
                         **prepared.get("options_override", {})},
                    )
                    atomic_json(execution_path, execution)
            # Run-time absence is resumable; do not cache it as a scientific result.
            if execution.get("status") == "invalid_runtime":
                return self._unavailable_case(case_root, execution, records,
                                              execution.get("reasons") or [execution.get("reason", "Runtime unavailable")])
            self._check_identity()
            if execution.get("candidate_sha256") != self.state["candidate_sha256"]:
                return self._unavailable_case(case_root, execution, records,
                                              ["Executed candidate hash does not match the frozen snapshot"])
            observed_runtime = execution.get("runtime", {})
            observed_identity = {key: observed_runtime.get(key) for key in ("image", "probe")}
            if observed_identity != self.state["runtime_identity"][backend]:
                return self._unavailable_case(case_root, execution, records,
                                              ["Container or loaded runtime changed between preflight and execution"])
            output_path = Path(execution.get("output_path") or case_root / "output/result.npz")
            if execution.get("status") == "passed":
                artifacts = execution.get("hashes", {}).get("artifacts", {})
                for name, sha in artifacts.items():
                    if Path(name).name != name:
                        raise RuntimeError("Invalid artifact receipt path")
                    verify_inventory({output_path.parent / name: sha})
                observed_hash = artifacts.get("result.npz")
                if not output_path.is_file() or observed_hash != digest(output_path):
                    return self._unavailable_case(case_root, execution, records,
                                                  ["Raw result is missing or changed after trusted collection"])
                if stage == "holdout_gpu":
                    if not execution.get("trace", {}).get("external", {}).get("valid"):
                        return self._unavailable_case(case_root, execution, records,
                                                      ["Final GPU gate lacks a valid external CUDA trace"])
                    if not {"cuda_trace.sqlite", "cuda_trace.nsys-rep"} <= set(artifacts):
                        return self._unavailable_case(case_root, execution, records,
                                                      ["External GPU profiling artifacts are missing from the receipt"])
            if prepared.get("challenge", {}).get("kind") == "nonconvergence":
                record = self.science.verify_nonconverged_case(prepared, output_path, spec["tolerances"], execution, max_cycle=1)
            else:
                record = self.science.verify_case(prepared, output_path, spec["tolerances"], execution)
            atomic_json(case_root / "verification.json", record)
            records.append(record)
            self._seal([execution_path, case_root / "verification.json", *output_path.parent.glob("*")])
            self._save()
            event(self.root, "case_verified", stage=stage, case_id=prepared["id"],
                  status=record["status"], metrics=record.get("metrics", {}), reasons=record.get("reasons", []))
        suite = self.science.verify_suite([case["id"] for case in self.state["prepared"][dataset]], records)
        suite["records"] = records
        suite["backend"] = backend
        suite["candidate_sha256"] = self.state["candidate_sha256"]
        suite["approximation_diagnostics"] = _approximation_diagnostics(records)
        if stage == "dev_properties":
            self._property_comparisons(suite, stage_root.parent / "dev_cpu/suite.json", spec)
        path = stage_root / "suite.json"
        atomic_json(path, suite)
        self._seal([path])
        self.state["stage_reports"][stage] = {"status": suite["status"], "path": str(path), "sha256": digest(path)}
        self._save()
        return suite

    def _unavailable_case(self, case_root, execution, records, reasons):
        """A temporary collection/runtime failure cannot poison the next resume."""
        archive = case_root / ("unavailable-" + uuid.uuid4().hex)
        archive.mkdir()
        output = case_root / "output"
        if output.exists():
            output.rename(archive / "output")
        receipt = case_root / "execution.json"
        if receipt.exists():
            receipt.rename(archive / "execution.json")
        atomic_json(archive / "unavailable.json", {"execution": execution, "reasons": reasons})
        self._seal(archive.rglob("*"))
        self._save()
        return {"status": "invalid_runtime", "passed": False, "records": records,
                "metrics": {}, "reasons": list(map(str, reasons))}

    def _property_comparisons(self, suite, base_path, spec):
        """Use the matching round's CPU baseline in both live and offline gates."""
        base_records = {r["case_id"]: r for r in load_json(base_path)["records"]}
        current = {r["case_id"]: r for r in suite.get("records", [])}
        comparisons = []
        for prepared in self.state["prepared"]["properties"]:
            challenge = prepared.get("challenge", {})
            if challenge.get("kind") == "rigid_motion" and prepared["id"] in current:
                comparisons.append(self.science.verify_rigid_motion(
                    base_records[challenge["base_case_id"]], current[prepared["id"]],
                    spec["tolerances"]["energy_max_abs"]))
        suite["metamorphic_comparisons"] = comparisons
        for comparison in comparisons:
            if not comparison["passed"]:
                suite.update(status="failed", passed=False)
                suite["reasons"].extend(comparison["reasons"])

    def _repair(self, report):
        self.state["rounds"][-1].update(status="rejected_development", failure_stage=self.state["phase"])
        self.state["feedback"] = _feedback(report)
        self.state["repair_source"] = self.state["candidate_path"]
        baseline = Path(self.state["last_cpu_correct"] or self.root / "initial/solution.py")
        shutil.copyfile(baseline, self.root / "active/solution.py")
        event(self.root, "rolled_back", rejected_sha256=self.state["candidate_sha256"], restored_sha256=digest(baseline))
        self.state["phase"] = "generate"
        self._save()

    def _freeze(self):
        # Claim the resolved holdout, not merely its file name or version label.
        # This prevents accidental adaptive reuse across campaigns under the
        # same evidence parent. Deliberate copying to another registry is outside
        # this local policy boundary and must not be called a fresh blind test.
        identities = sorted(p.get("identity", {}).get("physical_input_sha256", p["input_sha256"])
                            for p in self.state["prepared"]["holdout"])
        key = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
        registry = self.root.parent / ".holdout-assessments"
        registry.mkdir(exist_ok=True, mode=0o700)
        claim_path = registry / (key + ".json")
        claim = {"schema_version": 1, "campaign": str(self.root), "claimed_at": utcnow(),
                 "holdout_identity_sha256": key, "candidate_sha256": self.state["candidate_sha256"]}
        try:
            fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump(claim, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            existing = load_json(claim_path)
            if existing.get("campaign") != str(self.root) or existing.get("candidate_sha256") != self.state["candidate_sha256"]:
                self.state.update(status="rejected", phase="finished")
                event(self.root, "holdout_reuse_rejected", policy="This resolved holdout has already been consumed by another candidate")
                self._save()
                return
        self.state["trusted_identity"].update(inventory([claim_path]))
        frozen = self.root / "frozen"
        frozen.mkdir(exist_ok=True)
        shutil.copyfile(self.state["candidate_path"], frozen / "solution.py")
        self.state.update(frozen_candidate=str(frozen / "solution.py"), frozen_sha256=digest(frozen / "solution.py"),
                          holdout_started=True, phase="holdout_cpu")
        atomic_json(frozen / "identity.json", {"candidate_sha256": self.state["frozen_sha256"],
                                              "trusted_identity": self.state["trusted_identity"]})
        self._seal(frozen.rglob("*"))
        event(self.root, "candidate_frozen", sha256=self.state["frozen_sha256"],
              policy="No more generation or holdout feedback after this boundary")
        self._save()

    def _advance(self):
        self._check_identity()
        reason = self._preflight()
        if reason:
            return self._block(reason)
        while self.state["status"] == "active":
            self._check_identity()
            phase = self.state["phase"]
            if phase == "preflight":
                self.state["phase"] = "preparing"
                self._save()
            elif phase == "preparing":
                try:
                    self._prepare()
                except (RuntimeError, OSError) as exc:
                    return self._block(str(exc))
                self.state["phase"] = "generate"
                self._save()
            elif phase in {"generate", "generating"}:
                self._generate()
            elif phase in STAGES:
                if (phase.endswith("gpu") and load_json(self.root / "execution_config.json").get("stop_before_gpu", False)
                        and not self.state.get("gpu_authorized", False)):
                    self.state.update(status="waiting_for_gpu_authorization",
                                      blocked_reason="GPU work was explicitly deferred; CPU evidence is retained. Resume with explicit GPU authorization to continue.")
                    event(self.root, "stopped_before_gpu", phase=phase)
                    self._save()
                    write_manifest(self.root)
                    return copy.deepcopy(self.state)
                if (phase.endswith("gpu") and "cupy" not in self.state.get("runtime_identity", {})):
                    reason = self._preflight()
                    if reason:
                        return self._block(reason)
                report = self._evaluate_stage(phase)
                if report["status"] == "invalid_runtime":
                    return self._block("; ".join(map(str, report["reasons"])))
                if not report["passed"]:
                    if phase.startswith("holdout"):
                        self.state.update(status="rejected", phase="finished")
                        event(self.root, "blind_assessment_rejected", failed_stage=phase,
                              policy="No feedback to generator; a new holdout version is required for another campaign")
                        self._save()
                    else:
                        self._repair(report)
                    continue
                if phase == "dev_cpu":
                    self.state["last_cpu_correct"] = self.state["candidate_path"]
                    self.state["phase"] = "dev_properties"
                elif phase == "dev_properties":
                    self.state["phase"] = "dev_gpu"
                elif phase == "dev_gpu":
                    self._freeze()
                elif phase == "holdout_cpu":
                    self.state["phase"] = "holdout_gpu"
                else:
                    accepted = self.root / "accepted"
                    accepted.mkdir(exist_ok=True)
                    shutil.copyfile(self.state["frozen_candidate"], accepted / "solution.py")
                    self.state.update(status="accepted", phase="finished", accepted_candidate=str(accepted / "solution.py"))
                    self.state["rounds"][-1]["status"] = "accepted"
                    event(self.root, "candidate_promoted", sha256=digest(accepted / "solution.py"))
                self._save()
            else:
                raise RuntimeError(f"Unknown campaign phase: {phase}")
        self._record_usage()
        self._save()
        write_manifest(self.root)
        return copy.deepcopy(self.state)

    def evaluate(self):
        """Recompute decisions from saved raw tensors, without a model or rerun.

        This cannot expose a holdout to a repair loop or spend a second blind
        assessment on an altered source. Stored manifests are checked first.
        """
        with campaign_lock(self.root):
            self.state = load_json(self.root / "state.json")
            self._check_identity()
            manifest_path = self.root / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError("Campaign has no completed evidence manifest")
            for name, entry in load_json(manifest_path)["files"].items():
                verify_inventory({self.root / name: entry["sha256"]})
            reports = {}
            for stage, saved in self.state["stage_reports"].items():
                path = Path(saved["path"])
                verify_inventory({path: saved["sha256"]})
                original = load_json(path)
                dataset = "holdout" if stage.startswith("holdout") else "properties" if stage == "dev_properties" else "dev"
                spec = load_json(self.root / ("private/holdout.json" if dataset == "holdout" else "spec.json"))
                by_id = {p["id"]: p for p in self.state["prepared"][dataset]}
                records = []
                for case_id, prepared in by_id.items():
                    execution_path = path.parent / case_id / "execution.json"
                    if not execution_path.exists():
                        continue
                    execution = load_json(execution_path)
                    if prepared.get("challenge", {}).get("kind") == "nonconvergence":
                        record = self.science.verify_nonconverged_case(prepared, execution["output_path"], spec["tolerances"], execution, max_cycle=1)
                    else:
                        record = self.science.verify_case(prepared, execution["output_path"], spec["tolerances"], execution)
                    records.append(record)
                result = self.science.verify_suite(list(by_id), records)
                result["records"] = records
                result["approximation_diagnostics"] = _approximation_diagnostics(records)
                if stage == "dev_properties":
                    self._property_comparisons(result, path.parent.parent / "dev_cpu/suite.json", spec)
                if result["passed"] != original["passed"] or result.get("metrics") != original.get("metrics"):
                    raise RuntimeError(f"Recomputed measurements disagree with recorded decision: {stage}")
                reports[stage] = result
            final = all(stage in reports and reports[stage]["passed"] for stage in STAGES)
            return {"status": self.state["status"], "accepted": self.state["status"] == "accepted" and final,
                    "identity_verified": True, "recomputed_stages": reports,
                    "reason": None if final else "All required development, property and CPU/GPU blind gates have not passed"}
