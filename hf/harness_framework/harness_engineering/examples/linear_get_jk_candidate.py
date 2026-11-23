"""Time one RIJCOSX get_jk and measure its size exponent.

Water, benzene, and alkanes are extended in one dimension. A 16-water
cluster remains the density-fitting regression. This candidate does not run
the SCF loop.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from pyscf import df, gto, lib, scf

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
from hf import Hartree_Fock  # noqa: E402
from harness_engineering.water_clusters import cluster_atom_string  # noqa: E402
from make_alkane_xyz import alkane_coordinates  # noqa: E402


BASIS = "cc-pvtz"
AUXBASIS = "weigend"
DF_TOL = 1e-9
GRID_RAD = 30
GRID_ANG = 110
MONOMER_GRID_RAD = 40
MONOMER_GRID_ANG = 146
BATCH_SIZE = 256
COSX_TOL = 1e-10
AO_TOL = 1e-10
WORKERS = 4
CHUNK_SIZE = 16
MIN_SHELLS = 12
TASK_TIMEOUT_S = 3600.0
WATER_SPACING = 4.0
BENZENE_SPACING = 8.0
WATER_MONOMER = (
    ("O", 0.0, 0.0, 0.0),
    ("H", 0.0, -0.757, 0.587),
    ("H", 0.0, 0.757, 0.587),
)
BENZENE = (
    ("C", 0.000000, 1.396792, 0.000000),
    ("C", -1.209657, 0.698396, 0.000000),
    ("C", -1.209657, -0.698396, 0.000000),
    ("C", 0.000000, -1.396792, 0.000000),
    ("C", 1.209657, -0.698396, 0.000000),
    ("C", 1.209657, 0.698396, 0.000000),
    ("H", 0.000000, 2.481000, 0.000000),
    ("H", -2.147000, 1.240500, 0.000000),
    ("H", -2.147000, -1.240500, 0.000000),
    ("H", 0.000000, -2.481000, 0.000000),
    ("H", 2.147000, -1.240500, 0.000000),
    ("H", 2.147000, 1.240500, 0.000000),
)


def estimate_linear_seconds(previous_seconds, previous_nao, next_nao):
    """Scale a finished get_jk by the ratio of basis-function counts."""

    if (previous_seconds is None or previous_seconds <= 0
            or previous_nao <= 0 or next_nao <= 0):
        return None
    return float(previous_seconds) * (float(next_nao) / float(previous_nao))


def should_skip_linear(estimated_seconds, remaining_seconds):
    return estimated_seconds is not None and estimated_seconds > remaining_seconds


def scaling_exponent(previous_seconds, previous_nao, next_seconds, next_nao):
    """log(t2/t1) / log(n2/n1). Linear scaling is 1."""

    if (previous_seconds is None or next_seconds is None
            or previous_seconds <= 0 or next_seconds <= 0
            or previous_nao <= 0 or next_nao <= previous_nao):
        return None
    return math.log(float(next_seconds) / float(previous_seconds)) / math.log(
        float(next_nao) / float(previous_nao))


def _atom_string(rows):
    return "; ".join(
        f"{symbol} {x:.6f} {y:.6f} {z:.6f}" for symbol, x, y, z in rows)


def translated(monomer, count, axis, spacing):
    rows = []
    for index in range(count):
        shift = [0.0, 0.0, 0.0]
        shift[axis] = index * spacing
        for symbol, x, y, z in monomer:
            rows.append((symbol, x + shift[0], y + shift[1], z + shift[2]))
    return _atom_string(rows)


def alkane_atom_string(carbon_count):
    return _atom_string(
        (element, xyz[0], xyz[1], xyz[2])
        for element, xyz in alkane_coordinates(carbon_count)
    )


def _remaining(started):
    return TASK_TIMEOUT_S - (time.perf_counter() - started)


def _asymmetry(matrix):
    return float(np.max(np.abs(matrix - matrix.T)))


def make_solver(mol, workers):
    if workers == 1:
        return Hartree_Fock(mol, parallel_backend="serial")
    return Hartree_Fock(
        mol,
        parallel_backend="cpu",
        parallel_workers=workers,
        parallel_chunk_size=CHUNK_SIZE,
        parallel_min_shells=MIN_SHELLS,
    )


def run_get_jk(solver, dm):
    started = time.perf_counter()
    j_matrix, k_matrix = solver.get_jk(
        dm,
        jk_mode="rijcosx",
        auxbasis=AUXBASIS,
        df_tol=DF_TOL,
        grid_rad=GRID_RAD,
        grid_ang=GRID_ANG,
        batch_size=BATCH_SIZE,
        cosx_tol=COSX_TOL,
        ao_tol=AO_TOL,
        overlap_fit=True,
    )
    return j_matrix, k_matrix, time.perf_counter() - started


def monomer_cosx():
    mol = gto.M(atom=_atom_string(WATER_MONOMER), basis=BASIS, unit="Angstrom", verbose=0)
    dm = np.asarray(scf.hf.init_guess_by_minao(mol))
    solver = make_solver(mol, 1)
    k_matrix = solver.build_cosx_k(
        dm,
        grid_rad=MONOMER_GRID_RAD,
        grid_ang=MONOMER_GRID_ANG,
        batch_size=BATCH_SIZE,
        cosx_tol=COSX_TOL,
        ao_tol=AO_TOL,
        overlap_fit=True,
    )
    _, reference = scf.RHF(mol).get_jk(mol, dm, with_j=False, with_k=True)
    return {
        "nao": int(mol.nao_nr()),
        "cosx_max_abs": float(np.max(np.abs(k_matrix - reference))),
        "cosx_full_tensor_calls": int(solver.cosx_full_tensor_calls),
        "all_finite": bool(np.isfinite(k_matrix).all() and np.isfinite(reference).all()),
    }


def water_cluster_reference():
    mol = gto.M(atom=cluster_atom_string(16), basis=BASIS, unit="Angstrom", verbose=0)
    dm = np.asarray(scf.hf.init_guess_by_minao(mol))
    serial = make_solver(mol, 1)
    j_serial, k_serial, serial_seconds = run_get_jk(serial, dm)
    reference, _ = df.DF(mol, auxbasis=AUXBASIS).get_jk(dm, with_j=True, with_k=False)
    parallel = make_solver(mol, WORKERS)
    j_matrix, k_matrix, parallel_seconds = run_get_jk(parallel, dm)
    return {
        "nao": int(mol.nao_nr()),
        "serial_seconds": serial_seconds,
        "parallel_seconds": parallel_seconds,
        "ri_j_max_abs": float(np.max(np.abs(j_serial - reference))),
        "j_parallel_max_abs": float(np.max(np.abs(j_matrix - j_serial))),
        "k_parallel_max_abs": float(np.max(np.abs(k_matrix - k_serial))),
        "j_asymmetry": _asymmetry(j_matrix),
        "k_asymmetry": _asymmetry(k_matrix),
        "all_finite": bool(
            np.isfinite(j_serial).all() and np.isfinite(k_serial).all()
            and np.isfinite(j_matrix).all() and np.isfinite(k_matrix).all()
            and np.isfinite(reference).all()
        ),
    }


def run_series(name, sizes, atom_string, started):
    records = []
    previous = None
    for size in sizes:
        mol = gto.M(atom=atom_string(size), basis=BASIS, unit="Angstrom", verbose=0)
        nao = int(mol.nao_nr())
        remaining = _remaining(started)
        estimate = None
        if previous is not None:
            estimate = estimate_linear_seconds(previous["seconds"], previous["nao"], nao)
        if previous is not None and should_skip_linear(estimate, remaining):
            reason = (
                f"estimated get_jk time {estimate:.3f} s exceeds "
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
        j_matrix, k_matrix, seconds = run_get_jk(solver, dm)
        print(
            f"{name} size={size} nao={nao} seconds={seconds:.3f} "
            f"pairs={solver.last_shell_pairs} screened={solver.last_ri_screened_pairs}",
            file=sys.stderr,
            flush=True,
        )
        records.append({
            "size": size,
            "nao": nao,
            "seconds": seconds,
            "skipped": False,
            "coverage_complete": bool(solver.coverage_complete),
            "shell_pairs": int(solver.last_shell_pairs),
            "screened_pairs": int(solver.last_ri_screened_pairs),
            "j_asymmetry": _asymmetry(j_matrix),
            "k_asymmetry": _asymmetry(k_matrix),
            "all_finite": bool(np.isfinite(j_matrix).all() and np.isfinite(k_matrix).all()),
        })
        previous = {"seconds": seconds, "nao": nao}
    completed = [record for record in records if not record["skipped"]]
    exponent = None
    if len(completed) >= 2:
        first, second = completed[-2], completed[-1]
        exponent = scaling_exponent(
            first["seconds"], first["nao"], second["seconds"], second["nao"])
    return records, exponent


def main():
    lib.num_threads(1)
    started = time.perf_counter()
    monomer = monomer_cosx()
    print(
        f"monomer cosx={monomer['cosx_max_abs']:.6e}",
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
    series = {}
    exponents = {}
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
    finite = bool(monomer["all_finite"] and cluster["all_finite"])
    asymmetry = max(cluster["j_asymmetry"], cluster["k_asymmetry"])
    for name, (sizes, builder) in builders.items():
        records, exponent = run_series(name, sizes, builder, started)
        series[name] = records
        exponents[name] = exponent
        for record in records:
            if record.get("all_finite") is False:
                finite = False
            if "j_asymmetry" in record:
                asymmetry = max(asymmetry, record["j_asymmetry"], record["k_asymmetry"])
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
        "jk_16_s": cluster["parallel_seconds"],
    }
    for name, exponent in exponents.items():
        if exponent is not None:
            metrics[f"exponent_{name}"] = exponent
    for name, records in series.items():
        for record in records:
            size = record["size"]
            metrics[f"skipped_{name}_{size}"] = bool(record["skipped"])
            metrics[f"nao_{name}_{size}"] = record["nao"]
            if "seconds" in record:
                metrics[f"jk_{name}_{size}_s"] = record["seconds"]
            if "estimated_seconds" in record:
                metrics[f"estimated_seconds_{name}_{size}"] = record["estimated_seconds"]
    report = {
        "metrics": metrics,
        "details": {
            "basis": BASIS,
            "auxbasis": AUXBASIS,
            "df_tol": DF_TOL,
            "grid_rad": GRID_RAD,
            "grid_ang": GRID_ANG,
            "jk_scf_executed": False,
            "monomer": monomer,
            "water16": cluster,
            "series": series,
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    report_path = os.environ.get("HARNESS_REPORT_PATH")
    if report_path:
        Path(report_path).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
