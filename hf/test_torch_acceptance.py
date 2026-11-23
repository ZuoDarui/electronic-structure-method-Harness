"""Accuracy and interleaved timing gate for the PyTorch HF candidate.

Run in the cluster's PySCF environment, for example::

    python test_torch_acceptance.py --devices cpu cuda --repeat 3 \
        --output outputs/torch_acceptance/report.json

The original and candidate are loaded under different module names. Neither
source is modified. ``--quick`` tests small direct-HF molecules and dense
algebra only; it is deliberately not a performance/promotion acceptance.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback


ETHANE = """
C  0.0000000000  0.0000000000  0.0000000000
C  1.2703003855  0.8455843722  0.0000000000
H -0.8722270682  0.6536971329  0.0000000000
H -0.0182096609 -0.6292719916 -0.8898231110
H -0.0182096609 -0.6292719916  0.8898231110
H  1.0038375773  1.9025127387  0.0000000000
H  1.8578549845  0.6195436141 -0.8898231110
H  1.8578549845  0.6195436141  0.8898231110
"""

DIRECT_CASES = [
    ("H2", "H 0 0 0; H 0 0 0.74", "6-31g"),
    ("H2O", "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "sto-3g"),
    ("CH4", "C 0 0 0; H .629 .629 .629; H -.629 -.629 .629; "
     "H -.629 .629 -.629; H .629 -.629 -.629", "sto-3g"),
]

RI_OPTIONS = dict(
    jk_mode="rijcosx", auxbasis="weigend", tol=1e-8,
    grid_rad=30, grid_ang=110, final_grid_rad=35, final_grid_ang=302,
    batch_size=512, max_iteration=200, incremental_k=True,
    kfull_every=5, grid_switch_energy_tol=1e-5,
)


def load_source(name, path):
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, time.perf_counter() - started


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def maximum_error(left, right):
    import numpy as np
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def check_density(mol, overlap, energy, density, reference):
    import numpy as np
    error = abs(float(energy) - reference[0])
    dm_error = maximum_error(density, reference[1])
    electron_error = abs(float(np.einsum("ij,ji", density, overlap)) - mol.nelectron)
    finite = bool(np.isfinite(energy) and np.isfinite(density).all())
    metrics = dict(
        energy_error_hartree=error, density_max_abs_error=dm_error,
        electron_count_error=electron_error, finite=finite,
        density_dtype=str(density.dtype),
        density_symmetry_error=maximum_error(density, density.T),
        density_idempotency_error=maximum_error(density @ overlap @ density, 2 * density),
    )
    metrics["passed"] = bool(
        finite and error < 1e-8 and dm_error < 1e-6 and electron_error < 1e-8
        and density.dtype == np.float64
        and metrics["density_symmetry_error"] < 1e-10
        and metrics["density_idempotency_error"] < 1e-8
    )
    return metrics


def make_profiles(candidate, baseline, devices, defaults=True):
    profiles = [{"name": "baseline", "module": baseline,
                 "kwargs": {"dense_backend": "cpu"}, "device": "cpu"}]
    for device in devices:
        profiles.append({"name": "torch_" + device, "module": candidate,
                         "kwargs": {"dense_backend": "torch", "torch_device": device},
                         "device": device})
        if defaults:
            profiles.append({"name": "default_" + device, "module": candidate,
                             "kwargs": {"torch_device": device}, "device": device})
    return profiles


def synchronize(torch, device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def solve_once(profile, mol, hcore, overlap, options, torch, log_path,
               check_calls=0):
    """Time construction plus the complete SCF, including final J/K rebuild."""
    kwargs = dict(profile["kwargs"], dense_check_calls=check_calls,
                  max_memory=4096, k_max_memory_mb=2048,
                  memory_ceiling_mb=12288, jk_engine="pyscf_reference")
    with log_path.open("a") as logfile, contextlib.redirect_stdout(logfile):
        synchronize(torch, profile["device"])
        started = time.perf_counter()
        solver = profile["module"].Hartree_Fock(mol, **kwargs)
        energy, density = solver.solve_rhf(
            mol.nelec, hcore, overlap, mol, mol.energy_nuc(), **options)
        synchronize(torch, profile["device"])
        elapsed = time.perf_counter() - started
    details = solver.last_scf_result
    # Python pools otherwise survive until cyclic GC, distorting later trials.
    pool = getattr(solver, "_pool", None)
    if pool is not None:
        pool.shutdown(wait=True)
        solver._pool = None
    return float(energy), density, elapsed, details


def dense_accuracy(profile, mol, overlap, fock):
    """Check eigenvectors by the equation, not by their arbitrary signs."""
    import numpy as np
    kwargs = dict(profile["kwargs"])
    if "dense_backend" in kwargs:
        kwargs["backend"] = kwargs.pop("dense_backend")
    else:
        kwargs["backend"] = inspect.signature(
            profile["module"].Hartree_Fock).parameters["dense_backend"].default
    dense = profile["module"].DenseLinearAlgebra(overlap, check_calls=2, **kwargs)
    values, coeff, density = dense.solve(fock, mol.nelectron // 2)
    residual = fock @ coeff - (overlap @ coeff) * values[None, :]
    scale = max(1.0, float(np.linalg.norm(fock)) * float(np.linalg.norm(coeff)))
    metrics = {
        "backend": dense.backend, "device": dense.device,
        "eigen_residual_max_abs": float(np.max(np.abs(residual))),
        "eigen_residual_relative": float(np.linalg.norm(residual) / scale),
        "orthogonality_max_abs_error": maximum_error(coeff.T @ overlap @ coeff,
                                                       np.eye(mol.nao_nr())),
        "electron_count_error": abs(float(np.einsum("ij,ji", density, overlap))
                                    - mol.nelectron),
        "check_max_abs_diff": dict(dense.check_max),
        "float64_outputs": all(a.dtype == np.float64 for a in (values, coeff, density)),
    }
    metrics["passed"] = bool(
        metrics["eigen_residual_max_abs"] < 1e-9
        and metrics["orthogonality_max_abs_error"] < 1e-9
        and metrics["electron_count_error"] < 1e-8 and metrics["float64_outputs"]
    )
    dense.clear()
    return metrics


def direct_accuracy(report, profiles, torch, directory, save):
    from pyscf import gto, scf
    for name, atom, basis in DIRECT_CASES:
        mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
        hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        overlap = mol.intor("int1e_ovlp")
        reference = scf.RHF(mol)
        reference.conv_tol = 1e-13
        reference.conv_tol_grad = 1e-10
        reference.max_cycle = 200
        reference.kernel()
        if not reference.converged:
            raise RuntimeError(name + " independent PySCF reference did not converge")
        reference_pair = (float(reference.e_tot), reference.make_rdm1())
        fock = reference.get_fock(dm=reference_pair[1])
        case = {"name": name, "basis": basis, "nao": mol.nao_nr(),
                "pyscf_energy_hartree": reference_pair[0], "profiles": []}
        report["direct_accuracy"].append(case)
        baseline_pair = None
        for profile in profiles:
            name_tag = profile["name"]
            energy, density, elapsed, details = solve_once(
                profile, mol, hcore, overlap,
                dict(jk_mode="direct", tol=1e-10, max_iteration=200), torch,
                directory / (name + "_" + name_tag + ".log"), check_calls=2)
            if baseline_pair is None:
                baseline_pair = (energy, density)
            entry = {"profile": name_tag, "energy_hartree": energy,
                     "first_scf_seconds": elapsed,
                     "vs_pyscf": check_density(mol, overlap, energy, density, reference_pair),
                     "vs_original": check_density(mol, overlap, energy, density, baseline_pair),
                     "dense": dense_accuracy(profile, mol, overlap, fock)}
            entry["passed"] = all(entry[key]["passed"]
                                   for key in ("vs_pyscf", "vs_original", "dense"))
            case["profiles"].append(entry)
            print(json.dumps({"case": name, "profile": name_tag, "passed": entry["passed"],
                              "seconds": elapsed, "energy_hartree": energy}), flush=True)
            save()


def rijcosx_benchmark(report, profiles, torch, directory, repeat, warmup,
                     slowdown_limit, save):
    import numpy as np
    from pyscf import gto
    mol = gto.M(atom=ETHANE, basis="cc-pvtz", unit="Angstrom", verbose=0)
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    benchmark = {
        "name": "C2H6", "basis": "cc-pvtz", "nao": mol.nao_nr(),
        "options": RI_OPTIONS, "warmup_rounds": warmup, "timed_rounds": repeat,
        "max_candidate_over_baseline": slowdown_limit,
        "timing_scope": "constructor and complete SCF, including final full J/K rebuild; "
                        "molecule/one-electron preparation excluded equally",
        "rounds": [], "summaries": {},
    }
    report["rijcosx_benchmark"] = benchmark
    reference_pair = None
    for round_index in range(warmup + repeat):
        # Rotate and reverse alternate rounds to reduce order/temperature bias.
        offset = round_index % len(profiles)
        ordered = profiles[offset:] + profiles[:offset]
        if round_index % 2:
            ordered = ordered[::-1]
        rows = []
        benchmark["rounds"].append({"round": round_index,
                                    "warmup": round_index < warmup, "results": rows})
        for profile in ordered:
            energy, density, elapsed, details = solve_once(
                profile, mol, hcore, overlap, RI_OPTIONS, torch,
                directory / ("C2H6_" + profile["name"] + ".log"))
            if reference_pair is None:
                if profile["name"] != "baseline":
                    raise RuntimeError("first benchmark must be the original baseline")
                reference_pair = (energy, density)
            entry = {"profile": profile["name"], "seconds": elapsed,
                     "energy_hartree": energy,
                     "accuracy": check_density(mol, overlap, energy, density, reference_pair),
                     "scf": details}
            if details is None or not details.get("converged"):
                entry["accuracy"]["passed"] = False
            elif [stage["grid"] for stage in details["grid_stages"]] != [[30, 110], [35, 302]]:
                entry["accuracy"]["passed"] = False
            dense_metadata = (details or {}).get("dense", {})
            expected_backend = profile["kwargs"].get("dense_backend", "torch")
            entry["routing_passed"] = (
                dense_metadata.get("backend") == expected_backend
                and profile["device"] in dense_metadata.get("device", "")
            )
            entry["accuracy"]["passed"] = (
                entry["accuracy"]["passed"] and entry["routing_passed"])
            rows.append(entry)
            print(json.dumps({"case": "C2H6", "round": round_index,
                              "warmup": round_index < warmup, "profile": profile["name"],
                              "seconds": elapsed, "passed": entry["accuracy"]["passed"]}),
                  flush=True)
            save()
    for profile in profiles:
        first = next(row for trial in benchmark["rounds"] for row in trial["results"]
                     if row["profile"] == profile["name"])
        rows = [row for trial in benchmark["rounds"] if not trial["warmup"]
                for row in trial["results"] if row["profile"] == profile["name"]]
        seconds = [row["seconds"] for row in rows]
        benchmark["summaries"][profile["name"]] = {
            "first_ri_scf_seconds": first["seconds"], "timed_seconds": seconds,
            "median_seconds": statistics.median(seconds),
            "min_seconds": min(seconds), "max_seconds": max(seconds),
            "accuracy_passed": all(row["accuracy"]["passed"] for row in rows)
                               and first["accuracy"]["passed"],
        }
    baseline_median = benchmark["summaries"]["baseline"]["median_seconds"]
    for summary in benchmark["summaries"].values():
        ratio = summary["median_seconds"] / baseline_median
        summary["candidate_over_baseline"] = ratio
        summary["speed_passed"] = ratio <= slowdown_limit
    benchmark["passed"] = all(
        row["accuracy"]["passed"]
        for trial in benchmark["rounds"] for row in trial["results"]
    ) and all(summary["speed_passed"] for summary in benchmark["summaries"].values())
    save()


def cli_benchmark(report, profiles, directory, baseline, candidate, threads,
                  repeat, slowdown_limit, save):
    """Run the actual command-line entry points in fresh Python processes.

    This includes interpreter startup, imports, CUDA initialization and the
    complete SCF. OS file caches are shared, but no Python/CUDA process state is.
    The baseline and candidate use the same geometry, resource and SCF settings.
    """
    directory = directory / "fresh_process_cli"
    directory.mkdir(exist_ok=True)
    xyz = directory / "c2h6.xyz"
    xyz.write_text("8\nC2H6 acceptance\n" + ETHANE.strip() + "\n")
    benchmark = {"name": "C2H6", "basis": "cc-pvtz", "repeat": repeat,
                 "timing_scope": "fresh interpreter: real source.py --config invocation, "
                                 "imports, CUDA initialization, molecule/integrals, SCF, logs and exit",
                 "max_candidate_over_baseline": slowdown_limit,
                 "rounds": [], "summaries": {}}
    report["fresh_process_cli_benchmark"] = benchmark
    for round_index in range(repeat):
        offset = round_index % len(profiles)
        ordered = profiles[offset:] + profiles[:offset]
        rows = []
        benchmark["rounds"].append({"round": round_index, "results": rows})
        for profile in ordered:
            tag = profile["name"] + "_" + str(round_index)
            source = baseline if profile["name"] == "baseline" else candidate
            config = {
                "xyz": str(xyz), "basis": "cc-pvtz", "unit": "Angstrom",
                "charge": 0, "spin": 0, "output_dir": str(directory),
                "output_prefix": tag,
                "resources": {"threads": threads, "cpu_cores": threads,
                              "memory_gib": 16, "k_memory_gib": 2,
                              "memory_ceiling_fraction": 0.75,
                              "parallel_backend": "serial", "parallel_workers": 1,
                              "dense_backend": profile["kwargs"].get("dense_backend", "torch"),
                              "torch_device": profile["device"], "dense_check_iterations": 0},
                "solver_options": dict(RI_OPTIONS),
            }
            config_path = directory / (tag + ".config.json")
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            log_path = directory / (tag + ".stdout.log")
            command = [sys.executable, str(source.resolve()), "--config", str(config_path)]
            entry = {"profile": profile["name"], "command": command,
                     "stdout_log": str(log_path), "config": str(config_path),
                     "passed": False}
            process_env = dict(os.environ, PYTHONUNBUFFERED="1")
            started = time.perf_counter()
            with log_path.open("w") as log:
                try:
                    process = subprocess.run(command, cwd=source.resolve().parent,
                                             env=process_env, stdout=log,
                                             stderr=subprocess.STDOUT, timeout=1800, check=False)
                    entry["returncode"] = process.returncode
                except subprocess.TimeoutExpired:
                    entry["error"] = "fresh-process CLI exceeded 1800 seconds"
            entry["seconds"] = time.perf_counter() - started
            content = log_path.read_text(errors="replace")
            diagnostics = []
            for line in content.splitlines():
                if "SCF final diagnostics " in line:
                    diagnostics.append(json.loads(line.split("SCF final diagnostics ", 1)[1]))
            if diagnostics:
                final = diagnostics[-1]
                entry["energy_hartree"] = final["energy_hartree"]
                entry["electron_count_error"] = abs(final["electron_count"] - 18)
                entry["final_grid"] = final["grid"]
                expected_backend = profile["kwargs"].get("dense_backend", "torch")
                backend_lines = re.findall(r"SCF dense linear algebra backend=([^\n]+)", content)
                entry["dense_logs"] = backend_lines
                entry["passed"] = bool(
                    entry.get("returncode") == 0 and entry["electron_count_error"] < 1e-8
                    and final["grid"] == [35, 302] and final.get("physical_guard") == "passed"
                    and backend_lines and all(line.startswith(expected_backend + " ")
                                              and profile["device"] in line for line in backend_lines)
                )
            else:
                entry["error"] = entry.get("error", "CLI did not log final physical diagnostics")
            rows.append(entry)
            print(json.dumps({"case": "C2H6_fresh_process_cli", "round": round_index,
                              "profile": profile["name"], "seconds": entry["seconds"],
                              "passed": entry["passed"]}), flush=True)
            save()
    baseline_rows = [row for trial in benchmark["rounds"] for row in trial["results"]
                     if row["profile"] == "baseline"]
    reference_energies = [row["energy_hartree"] for row in baseline_rows if row.get("passed")]
    reference_energy = statistics.median(reference_energies) if reference_energies else None
    for trial in benchmark["rounds"]:
        for row in trial["results"]:
            if reference_energy is not None and "energy_hartree" in row:
                row["energy_error_hartree"] = abs(row["energy_hartree"] - reference_energy)
                row["passed"] = row["passed"] and row["energy_error_hartree"] < 1e-8
            else:
                row["passed"] = False
    baseline_median = statistics.median(row["seconds"] for row in baseline_rows)
    for profile in profiles:
        rows = [row for trial in benchmark["rounds"] for row in trial["results"]
                if row["profile"] == profile["name"]]
        seconds = [row["seconds"] for row in rows]
        ratio = statistics.median(seconds) / baseline_median
        benchmark["summaries"][profile["name"]] = {
            "seconds": seconds, "median_seconds": statistics.median(seconds),
            "candidate_over_baseline": ratio,
            "accuracy_passed": all(row["passed"] for row in rows),
            "speed_passed": ratio <= slowdown_limit,
        }
    benchmark["passed"] = all(summary["accuracy_passed"] and summary["speed_passed"]
                              for summary in benchmark["summaries"].values())
    save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--baseline", type=Path, default=root / "hf.py")
    parser.add_argument("--candidate", type=Path, default=root / "hf_torch.py")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cli-repeat", type=int, default=3,
                        help="fresh-process complete CLI trials per backend (at least 3)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--slowdown-limit", type=float, default=1.10)
    parser.add_argument("--output", type=Path, default=root / "outputs/torch_acceptance/report.json")
    parser.add_argument("--quick", action="store_true", help="small-molecule correctness only")
    args = parser.parse_args(argv)
    if (args.repeat < 3 or args.cli_repeat < 3 or args.warmup < 1
            or args.threads < 1 or args.slowdown_limit < 1):
        parser.error("repeat and cli-repeat must be >=3, warmup >=1, threads >=1, slowdown-limit >=1")
    if args.baseline.resolve() == args.candidate.resolve():
        parser.error("baseline and candidate must be separate source files")
    source_sha256 = {"baseline": digest(args.baseline), "candidate": digest(args.candidate)}
    if source_sha256["baseline"] == source_sha256["candidate"]:
        parser.error("baseline and candidate have identical contents; use the original HF backup")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs = args.output.parent / (args.output.stem + "_logs")
    logs.mkdir(exist_ok=True)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    report = {"status": "running", "quick": args.quick, "promotion_eligible": False,
              "devices_requested": args.devices, "direct_accuracy": [],
              "source_sha256": source_sha256,
              "source_paths": {"baseline": str(args.baseline.resolve()),
                               "candidate": str(args.candidate.resolve())},
              "thresholds": {"energy_hartree": 1e-8, "density_max_abs": 1e-6,
                             "electron_count": 1e-8, "eigen_residual_max_abs": 1e-9},
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "threads": args.threads}}
    save = lambda: save_report(args.output, report)
    save()
    started = time.perf_counter()
    try:
        baseline, baseline_load = load_source("hf_acceptance_baseline", args.baseline.resolve())
        candidate, candidate_load = load_source("hf_acceptance_candidate", args.candidate.resolve())
        from pyscf import lib
        import numpy as np
        import pyscf
        import scipy
        lib.num_threads(args.threads)
        torch_started = time.perf_counter()
        torch = candidate.import_torch()
        torch_load = time.perf_counter() - torch_started
        torch.set_num_threads(args.threads)
        cuda_started = time.perf_counter()
        if "cuda" in args.devices:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available; no silent skip")
            torch.cuda.init()
            torch.cuda.synchronize()
        cuda_init = time.perf_counter() - cuda_started
        report["startup_seconds"] = {
            "baseline_import_including_scientific_dependencies": baseline_load,
            "candidate_import_after_dependencies": candidate_load,
            "torch_import": torch_load, "cuda_initialize": cuda_init,
            "note": "Separate measured startup costs; first-use SCF timings are also retained. "
                    "These are not separate clean-process startup comparisons.",
        }
        report["environment"].update({"numpy": np.__version__, "scipy": scipy.__version__,
                                      "pyscf": pyscf.__version__, "torch": torch.__version__,
                                      "torch_cuda": torch.version.cuda,
                                      "torch_threads": torch.get_num_threads(),
                                      "pyscf_threads": lib.num_threads()})
        if "cuda" in args.devices:
            props = torch.cuda.get_device_properties(0)
            report["environment"]["gpu"] = {"name": props.name,
                                               "memory_bytes": props.total_memory}
        profiles = make_profiles(candidate, baseline, args.devices)
        report["candidate_default_dense_backend"] = inspect.signature(
            candidate.Hartree_Fock).parameters["dense_backend"].default
        save()
        direct_accuracy(report, profiles, torch, logs, save)
        if not args.quick:
            timing_profiles = [profile for profile in profiles
                               if not profile["name"].startswith("default_")]
            rijcosx_benchmark(report, timing_profiles, torch, logs, args.repeat,
                             args.warmup, args.slowdown_limit, save)
            cli_benchmark(report, timing_profiles, logs, args.baseline, args.candidate,
                          args.threads, args.cli_repeat, args.slowdown_limit, save)
        passed = all(entry["passed"] for case in report["direct_accuracy"]
                     for entry in case["profiles"])
        if not args.quick:
            passed = (passed and report["rijcosx_benchmark"]["passed"]
                      and report["fresh_process_cli_benchmark"]["passed"])
        report["status"] = "passed_quick" if passed and args.quick else "passed" if passed else "failed"
        report["promotion_eligible"] = bool(passed and not args.quick and set(args.devices) == {"cpu", "cuda"})
        if report["source_sha256"] != {"baseline": digest(args.baseline), "candidate": digest(args.candidate)}:
            raise RuntimeError("source changed during acceptance; rerun against immutable sources")
    except Exception as exc:
        report["status"] = "failed"
        report["promotion_eligible"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        save()
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "promotion_eligible": report["promotion_eligible"]}), flush=True)
    return 0 if report["status"] in ("passed", "passed_quick") else 1


if __name__ == "__main__":
    raise SystemExit(main())
