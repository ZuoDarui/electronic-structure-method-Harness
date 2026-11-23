"""Time one training RIJCOSX iteration and project it to 50000 basis functions.

The iteration is ``Hartree_Fock.iteration_rijcosx`` in hf.py, called through
harness_engineering.training_hf. It does not run solve_rhf. J, K, and the
dense diagonalization are timed separately. J and K exponents are gated; the
50000-basis projection is recorded either way.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from pyscf import gto, lib, scf

ROOT = Path(__file__).resolve().parents[2]
# A direct script must select this framework before importing its package.
for _module_name, _filename in (("harness_engineering", "__init__.py"),
                                ("harness_engineering.paths", "paths.py")):
    _cached = sys.modules.get(_module_name)
    if _cached is not None:
        _origin = getattr(_cached, "__file__", None)
        if _origin is None or Path(_origin).resolve() != ROOT / "harness_engineering" / _filename:
            raise ImportError(f"{_module_name} is already loaded outside this harness framework")
sys.path[:] = [str(ROOT), *(value for value in sys.path if value != str(ROOT))]
from harness_engineering.paths import HF_ROOT, configure_import_paths  # noqa: E402

configure_import_paths()
from hf import estimate_dense_solver_memory_gib  # noqa: E402
from harness_engineering.examples.linear_get_jk_candidate import (  # noqa: E402
    AO_TOL,
    AUXBASIS,
    BASIS,
    BATCH_SIZE,
    BENZENE,
    BENZENE_SPACING,
    COSX_TOL,
    DF_TOL,
    GRID_ANG,
    GRID_RAD,
    TASK_TIMEOUT_S,
    WATER_MONOMER,
    WATER_SPACING,
    WORKERS,
    alkane_atom_string,
    estimate_linear_seconds,
    make_solver,
    monomer_cosx,
    scaling_exponent,
    should_skip_linear,
    translated,
    water_cluster_reference,
)
from harness_engineering.training_hf import (  # noqa: E402
    TARGET_BASIS_FUNCTIONS,
    TARGET_SCF_ITERATIONS,
    TARGET_WALL_HOURS,
    iteration_rijcosx,
    project_iteration_hours,
)


def _remaining(started):
    return TASK_TIMEOUT_S - (time.perf_counter() - started)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _drop_matrices(result):
    kept = dict(result)
    for key in ("j_matrix", "k_matrix", "density"):
        kept.pop(key, None)
    return kept


def run_iteration(solver, mol, dm):
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    n_alpha, n_beta = mol.nelec
    if n_alpha != n_beta:
        raise ValueError("the training iteration supports closed shells only")
    return iteration_rijcosx(
        solver,
        dm,
        hcore,
        overlap,
        n_alpha,
        mol.nelectron,
        AUXBASIS,
        DF_TOL,
        GRID_RAD,
        GRID_ANG,
        BATCH_SIZE,
        COSX_TOL,
        AO_TOL,
        True,
    )


def monomer_repeat():
    """Run the same monomer density twice. The difference is diagnostic."""

    mol = gto.M(
        atom=translated(WATER_MONOMER, 1, 0, WATER_SPACING),
        basis=BASIS,
        unit="Angstrom",
        verbose=0,
    )
    dm = np.asarray(scf.hf.init_guess_by_minao(mol))
    solver = make_solver(mol, 1)
    first = run_iteration(solver, mol, dm)
    second = run_iteration(solver, mol, dm)
    repeat = float(np.max(np.abs(first["density"] - second["density"])))
    summary = _drop_matrices(first)
    summary["nao"] = int(mol.nao_nr())
    summary["repeat_density_max_abs"] = repeat
    summary["repeat_all_finite"] = bool(first["all_finite"] and second["all_finite"])
    return summary


def run_series(name, sizes, atom_string, started):
    records = []
    previous = None
    for size in sizes:
        mol = gto.M(atom=atom_string(size), basis=BASIS, unit="Angstrom", verbose=0)
        nao = int(mol.nao_nr())
        remaining = _remaining(started)
        estimate = None
        if previous is not None:
            estimate = estimate_linear_seconds(
                previous["seconds"], previous["nao"], nao)
        if previous is not None and should_skip_linear(estimate, remaining):
            reason = (
                f"estimated iteration time {estimate:.3f} s exceeds "
                f"the remaining {remaining:.3f} s"
            )
            print(f"{name} size={size} {reason}", file=sys.stderr, flush=True)
            records.append({
                "size": size,
                "nao": nao,
                "skipped": True,
                "estimated_seconds": estimate,
                "skip_reason": reason,
            })
            continue
        dm = np.asarray(scf.hf.init_guess_by_minao(mol))
        solver = make_solver(mol, WORKERS)
        result = run_iteration(solver, mol, dm)
        total = result["j_seconds"] + result["k_seconds"] + result["eigh_seconds"]
        print(
            f"{name} size={size} nao={nao} j={result['j_seconds']:.3f} "
            f"k={result['k_seconds']:.3f} eigh={result['eigh_seconds']:.3f} "
            f"electrons={result['density_electron_abs']:.3e}",
            file=sys.stderr,
            flush=True,
        )
        record = _drop_matrices(result)
        record.update({
            "size": size,
            "nao": nao,
            "seconds": total,
            "skipped": False,
        })
        records.append(record)
        previous = {"seconds": total, "nao": nao}
    return records


def _component_exponent(records, key):
    completed = [record for record in records if not record["skipped"] and key in record]
    if len(completed) < 2:
        return None
    first, second = completed[-2], completed[-1]
    return scaling_exponent(
        first[key], first["nao"], second[key], second["nao"])


def _project_component(records, key, exponent):
    completed = [record for record in records if not record["skipped"] and key in record]
    if not completed or exponent is None:
        return None
    last = completed[-1]
    return project_iteration_hours(last[key], last["nao"], exponent)


def _dominant(hours):
    finite = {name: value for name, value in hours.items() if value is not None}
    if not finite:
        return None
    return max(finite, key=finite.get)


def main():
    lib.num_threads(1)
    started = time.perf_counter()
    monomer = monomer_cosx()
    print(
        f"monomer cosx={monomer['cosx_max_abs']:.6e}",
        file=sys.stderr,
        flush=True,
    )
    repeat = monomer_repeat()
    print(
        f"monomer iteration electrons={repeat['density_electron_abs']:.3e} "
        f"repeat={repeat['repeat_density_max_abs']:.3e}",
        file=sys.stderr,
        flush=True,
    )
    cluster = water_cluster_reference()
    print(
        f"water16 ri_j={cluster['ri_j_max_abs']:.6e} "
        f"parallel_j={cluster['j_parallel_max_abs']:.6e} "
        f"seconds={cluster['parallel_seconds']:.3f}",
        file=sys.stderr,
        flush=True,
    )
    builders = {
        "water": (
            (8, 16, 32, 64),
            lambda count: translated(WATER_MONOMER, count, 2, WATER_SPACING),
        ),
        "benzene": (
            (2, 4, 8, 12),
            lambda count: translated(BENZENE, count, 0, BENZENE_SPACING),
        ),
        "alkane": ((8, 16, 32, 48), alkane_atom_string),
    }
    series = {}
    for name, (sizes, builder) in builders.items():
        series[name] = run_series(name, sizes, builder, started)

    finite = bool(
        monomer["all_finite"] and cluster["all_finite"] and repeat["repeat_all_finite"]
    )
    asymmetry = max(cluster["j_asymmetry"], cluster["k_asymmetry"])
    density_asymmetry = repeat["density_asymmetry"]
    density_electrons_abs = repeat["density_electron_abs"]
    projections = {}
    exponents = {}
    for name, records in series.items():
        exponents[name] = {
            "j": _component_exponent(records, "j_seconds"),
            "k": _component_exponent(records, "k_seconds"),
            "eigh": _component_exponent(records, "eigh_seconds"),
        }
        projections[name] = {
            component: _project_component(records, f"{component}_seconds", exponent)
            for component, exponent in exponents[name].items()
        }
        for record in records:
            if record.get("all_finite") is False:
                finite = False
            if "j_asymmetry" in record:
                asymmetry = max(asymmetry, record["j_asymmetry"], record["k_asymmetry"])
            if "density_asymmetry" in record:
                density_asymmetry = max(density_asymmetry, record["density_asymmetry"])
                density_electrons_abs = max(
                    density_electrons_abs, record["density_electron_abs"])

    memory = estimate_dense_solver_memory_gib(TARGET_BASIS_FUNCTIONS)
    metrics = {
        "monomer_nao": monomer["nao"],
        "nao_16": cluster["nao"],
        "all_finite": bool(finite),
        "cosx_monomer_max_abs": monomer["cosx_max_abs"],
        "cosx_full_tensor_calls": monomer["cosx_full_tensor_calls"],
        "ri_j_16_max_abs": cluster["ri_j_max_abs"],
        "j_parallel_16_max_abs": cluster["j_parallel_max_abs"],
        "k_parallel_16_max_abs": cluster["k_parallel_max_abs"],
        "j_asymmetry_16": cluster["j_asymmetry"],
        "k_asymmetry_16": cluster["k_asymmetry"],
        "jk_asymmetry_max": asymmetry,
        "density_asymmetry_max": density_asymmetry,
        "density_electron_abs_max": density_electrons_abs,
        "repeat_density_max_abs": repeat["repeat_density_max_abs"],
        "one_dense_matrix_gib_50000": memory["one_dense_matrix_gib"],
        "dense_working_set_gib_50000": memory["estimated_dense_working_set_gib"],
    }
    dominant = {}
    for name, parts in exponents.items():
        for component, exponent in parts.items():
            if exponent is not None:
                metrics[f"exponent_{component}_{name}"] = exponent
        for component, hours in projections[name].items():
            if hours is not None:
                metrics[f"projected_{component}_hours_{name}"] = hours
        family_hours = {
            component: projections[name][component] for component in ("j", "k", "eigh")
        }
        if all(value is not None for value in family_hours.values()):
            total_hours = sum(family_hours.values())
            metrics[f"projected_iteration_hours_{name}"] = total_hours
            family_hours = dict(family_hours)
        dominant[name] = _dominant(family_hours)
    for name, records in series.items():
        for record in records:
            size = record["size"]
            metrics[f"skipped_{name}_{size}"] = bool(record["skipped"])
            metrics[f"nao_{name}_{size}"] = record["nao"]
            for component in ("j", "k", "eigh"):
                key = f"{component}_seconds"
                if key in record:
                    metrics[f"{component}_{name}_{size}_s"] = record[key]
            if "estimated_seconds" in record:
                metrics[f"estimated_seconds_{name}_{size}"] = record["estimated_seconds"]
    overall = _dominant({
        f"{component}_{name}": projections[name][component]
        for name in projections
        for component in ("j", "k", "eigh")
    })
    report = {
        "metrics": metrics,
        "details": {
            "basis": BASIS,
            "auxbasis": AUXBASIS,
            "df_tol": DF_TOL,
            "grid_rad": GRID_RAD,
            "grid_ang": GRID_ANG,
            "production_scf_executed": False,
            "reference_module": "hf.py",
            "training_module": "harness_engineering/training_hf.py",
            "generated_module": "harness_engineering/generated/occupy_density.py",
            "long_term_target": {
                "basis_functions": TARGET_BASIS_FUNCTIONS,
                "scf_iterations": TARGET_SCF_ITERATIONS,
                "wall_time_hours": TARGET_WALL_HOURS,
                "hours_per_iteration": TARGET_WALL_HOURS / TARGET_SCF_ITERATIONS,
            },
            "dominant_projection": overall,
            "dominant_by_family": dominant,
            "monomer_cosx": monomer,
            "monomer_iteration": repeat,
            "water16": cluster,
            "series": series,
            "projections_hours": projections,
        },
    }
    payload = json.dumps(_json_ready(report), indent=2, sort_keys=True)
    report_path = os.environ.get("HARNESS_REPORT_PATH")
    if report_path:
        Path(report_path).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
