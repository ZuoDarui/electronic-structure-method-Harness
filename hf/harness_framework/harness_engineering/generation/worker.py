"""Container entry point for untrusted, integral-only RHF candidates.

This module is an execution adapter, not a numerical verifier. It never imports
PySCF or reads a reference result. Docker isolation is supplied by executor.py;
running this file directly does not provide isolation. In-process instrumentation
is useful evidence for cooperative candidates, not an adversarial attestation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np


STAGES = ("hf.jk0", "hf.jk1", "hf.jksum", "hf.solve")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_problem(path):
    """Load only the explicitly defined candidate-facing arrays, without pickle."""
    allowed = {"overlap", "hcore", "enuc", "nelectron", "eri", "df_factors",
               "trial_densities", "metadata_json"}
    with np.load(path, allow_pickle=False) as archive:
        if not set(archive.files) <= allowed:
            raise ValueError("problem contains fields outside the candidate contract")
        problem = {key: archive[key].copy() for key in archive.files}
    for key in ("overlap", "hcore", "enuc", "nelectron", "trial_densities"):
        if key not in problem:
            raise ValueError(f"problem is missing {key}")
    if ("eri" in problem) == ("df_factors" in problem):
        raise ValueError("problem must contain exactly one of eri and df_factors")
    metadata = problem.pop("metadata_json", np.asarray("{}"))
    if metadata.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string")
    problem["metadata"] = json.loads(str(metadata.item()))
    problem["enuc"] = float(problem["enuc"].item())
    problem["nelectron"] = int(problem["nelectron"].item())
    nao = problem["overlap"].shape[0]
    if problem["overlap"].shape != (nao, nao) or problem["hcore"].shape != (nao, nao):
        raise ValueError("overlap and hcore must be square AO matrices")
    if problem["trial_densities"].shape != (2, nao, nao):
        raise ValueError("trial_densities must have shape (2, nao, nao)")
    for value in problem.values():
        if isinstance(value, np.ndarray):
            if value.dtype != np.dtype("float64") or not np.isfinite(value).all():
                raise ValueError("problem arrays must be finite float64")
            value.flags.writeable = False
    return problem


class GPUObserver:
    """Capture NVTX stages and selected device operations; do not certify intent."""

    def __init__(self, backend):
        self.backend, self.cp, self.stage_name = backend, None, None
        self.operations, self.stages, self.originals = [], [], []
        self.metadata = {"backend": backend, "gpu_executed": False,
                         "instrumentation_scope": "cooperative Python hooks; candidate can bypass or tamper with these hooks"}
        if backend == "cupy":
            import cupy as cp
            self.cp = cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            # These preflight operations are deliberately outside candidate stages.
            probe = cp.asarray([1.0, 2.0], dtype=cp.float64)
            if cp.sum(probe).item() != 3.0:
                raise RuntimeError("CUDA float64 preflight failed")
            cp.cuda.get_current_stream().synchronize()
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            name = props["name"]
            self.metadata.update({"cupy_version": cp.__version__,
                                  "device_name": name.decode() if isinstance(name, bytes) else str(name),
                                  "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                                  "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion())})
            self._asnumpy, self._array_type = cp.asnumpy, cp.ndarray

    def install(self):
        if self.cp is None:
            return
        for module, attr, kind in ((self.cp, "einsum", "contraction"),
                                   (self.cp, "matmul", "contraction"),
                                   (self.cp, "dot", "contraction"),
                                   (self.cp, "tensordot", "contraction"),
                                   (self.cp.linalg, "eigh", "eigensolver")):
            original = getattr(module, attr)
            self.originals.append((module, attr, original))

            def observed(*args, _original=original, _name=attr, _kind=kind, **kwargs):
                if self.stage_name is None:
                    return _original(*args, **kwargs)
                device_operands = [v for v in args if isinstance(v, self._array_type)]
                start, stop = self.cp.cuda.Event(), self.cp.cuda.Event()
                start.record()
                result = _original(*args, **kwargs)
                stop.record()
                stop.synchronize()
                self.operations.append({"stage": self.stage_name, "operation": _name,
                                        "kind": _kind, "device_operands": len(device_operands),
                                        "operand_shapes": [list(v.shape) for v in device_operands],
                                        "cuda_elapsed_ms": float(self.cp.cuda.get_elapsed_time(start, stop))})
                return result
            setattr(module, attr, observed)

    def restore(self):
        for module, attr, original in self.originals:
            setattr(module, attr, original)

    @contextmanager
    def stage(self, name):
        self.stage_name = name
        started = time.perf_counter()
        if self.cp is not None:
            self.cp.cuda.get_current_stream().synchronize()
            self.cp.cuda.nvtx.RangePush(name)
        try:
            yield
        finally:
            if self.cp is not None:
                self.cp.cuda.get_current_stream().synchronize()
                self.cp.cuda.nvtx.RangePop()
            self.stages.append({"name": name, "elapsed_seconds": time.perf_counter() - started})
            self.stage_name = None

    def host_array(self, value):
        if self.cp is not None and isinstance(value, self._array_type):
            return self._asnumpy(value)
        return np.asarray(value)

    def evidence(self):
        coverage = {}
        for stage in STAGES:
            operations = [op for op in self.operations if op["stage"] == stage and op["device_operands"] > 0]
            coverage[stage] = {"device_operation_count": len(operations),
                               "eigensolver_count": sum(op["kind"] == "eigensolver" for op in operations),
                               "contraction_count": sum(op["kind"] == "contraction" for op in operations)}
        # @ is a C-level ndarray operator and is not reliably interceptable by
        # replacing cupy.matmul. External CUPTI kernels are the final GPU gate.
        executed = all(c["device_operation_count"] > 0 for c in coverage.values())
        executed = executed and coverage["hf.solve"]["eigensolver_count"] > 0
        return {**self.metadata, "gpu_executed": bool(executed),
                "stages": self.stages, "operations": self.operations,
                "stage_coverage": coverage}


def _matrix(value, name, observer):
    result = observer.host_array(value)
    if result.dtype != np.dtype("float64") or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite float64")
    return result


def run(candidate, problem_path, output, backend, options):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("worker refuses an existing result file")
    record_path = output.parent / "worker.json"
    record = {"schema": "hf.candidate-worker/1", "status": "failed", "backend": backend,
              "python": sys.version, "platform": platform.platform(), "numpy_version": np.__version__,
              "candidate_sha256": _digest(candidate), "input_sha256": _digest(problem_path),
              "pid": os.getpid(), "reference_available": False}
    observer = None
    try:
        problem = load_problem(problem_path)
        observer = GPUObserver(backend)
        observer.install()
        # Observe candidate imports too, but never credit them to a JK/SCF stage.
        spec = importlib.util.spec_from_file_location("hf_generated_solution", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "build_jk", None)) or not callable(getattr(module, "solve", None)):
            raise TypeError("candidate must define build_jk and solve")
        densities = problem["trial_densities"]
        arrays = {}
        for stage, density, suffix in (("hf.jk0", densities[0], "0"),
                                        ("hf.jk1", densities[1], "1"),
                                        ("hf.jksum", densities[0] + densities[1], "sum")):
            with observer.stage(stage):
                j, k = module.build_jk(problem, density, backend)
            arrays[f"j{suffix}"] = _matrix(j, f"j{suffix}", observer)
            arrays[f"k{suffix}"] = _matrix(k, f"k{suffix}", observer)
        with observer.stage("hf.solve"):
            result = module.solve(problem, options, backend)
        if not isinstance(result, dict):
            raise TypeError("solve must return a dict")
        for name in ("density", "mo_coeff", "mo_energy", "occupations"):
            arrays[name] = _matrix(result[name], name, observer)
        energy = observer.host_array(result["energy"])
        converged = observer.host_array(result["converged"])
        if energy.shape != () or energy.dtype.kind not in "fi" or not np.isfinite(energy):
            raise ValueError("energy must be a finite real scalar")
        if converged.shape != () or converged.dtype.kind != "b":
            raise ValueError("converged must be a boolean scalar")
        history = result["history"]
        if not isinstance(history, list) or not history:
            raise ValueError("history must be a nonempty list")
        previous = 0
        for row in history:
            if not isinstance(row, dict):
                raise ValueError("history rows must be mappings")
            cycle = row.get("cycle")
            if type(cycle) is not int or cycle <= previous:
                raise ValueError("history cycles must be positive and increasing")
            previous = cycle
            for key in ("energy", "residual_norm"):
                if isinstance(row.get(key), bool) or not isinstance(row.get(key), (int, float)) or not np.isfinite(row[key]):
                    raise ValueError(f"history {key} must be finite")
        history_json = json.dumps(history, allow_nan=False, separators=(",", ":"))
        arrays.update(energy=np.asarray(float(energy)), converged=np.asarray(bool(converged)),
                      history_json=np.asarray(history_json))
        # Only transfer the public candidate output; verifier derives all errors.
        np.savez_compressed(output, **arrays)
        record.update(status="passed", output_sha256=_digest(output),
                      input_sha256_after=_digest(problem_path), candidate_sha256_after=_digest(candidate))
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if output.exists():
            output.unlink()
    finally:
        if observer is not None:
            record["gpu_trace"] = observer.evidence()
            observer.restore()
        record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    parser.add_argument("--options", type=Path, required=True)
    args = parser.parse_args(argv)
    options = json.loads(args.options.read_text())
    if not isinstance(options, dict):
        parser.error("options JSON must be an object")
    record = run(args.candidate, args.problem, args.output, args.backend, options)
    print(json.dumps({"status": record["status"], "error": record.get("error")}), flush=True)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
