"""Trusted integral provider, independent reference worker and RHF verifier.

Candidate code is never imported here. It receives public integral NPZ files
and returns raw tensors; this module computes the acceptance measurements.
The same PySCF integral engine underlies both paths, so this checks the method
implementation and does not independently certify Gaussian integral routines.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence
import zipfile

import numpy as np

from .spec import CASE_ID, DEFAULT_TOLERANCES, validate_spec


PROBLEM_SCHEMA_VERSION = 1
OUTPUT_FIELDS = {
    "energy", "density", "mo_coeff", "mo_energy", "occupations", "j0", "k0",
    "j1", "k1", "jsum", "ksum", "history_json", "converged",
}
MAX_HISTORY_CHARS = 262144
MAX_NPY_HEADER_BYTES = 4096


def _read_bounded_npy_header(stream) -> tuple[tuple, np.dtype, int]:
    """Read NPY metadata without letting an attacker choose an allocation."""
    prefix = stream.read(8)
    if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
        raise ValueError("output member is not an NPY array")
    version = tuple(prefix[6:])
    if version not in {(1, 0), (2, 0), (3, 0)}:
        raise ValueError("unsupported NPY format version")
    width = 2 if version == (1, 0) else 4
    length_bytes = stream.read(width)
    if len(length_bytes) != width:
        raise ValueError("truncated NPY header length")
    length = int.from_bytes(length_bytes, "little")
    # NumPy's decoder may read the stated header before applying its own limit.
    # Check it first, then parse only a small literal containing simple types.
    if not 0 < length <= MAX_NPY_HEADER_BYTES:
        raise ValueError("NPY header exceeds bounded metadata budget")
    encoded = stream.read(length)
    if len(encoded) != length:
        raise ValueError("truncated NPY header")
    try:
        header = ast.literal_eval(encoded.decode("utf-8" if version == (3, 0) else "latin1"))
    except (ValueError, SyntaxError, UnicodeError, RecursionError) as exc:
        raise ValueError("malformed NPY header") from exc
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError("invalid NPY header keys")
    shape = header["shape"]
    if (not isinstance(shape, tuple) or len(shape) > 2
            or any(type(v) is not int or v < 0 for v in shape)):
        raise ValueError("invalid or excessive NPY dimensions")
    if type(header["fortran_order"]) is not bool or not isinstance(header["descr"], str):
        raise ValueError("structured or object NPY metadata is unsupported")
    if len(header["descr"]) > 64:
        raise ValueError("excessive NPY dtype descriptor")
    try:
        dtype = np.dtype(header["descr"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid NPY dtype") from exc
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError("object or structured NPY arrays are forbidden")
    return shape, dtype, 8 + width + length


def load_candidate_output(path: str | Path, nao: int) -> dict:
    """Bound archive, headers, shapes and decompressed bytes before NumPy load.

    In particular a tiny compressed zip cannot request an enormous density or
    Unicode scalar. The complete bounded snapshot is read once, so preflight
    and decoding cannot observe different files after a concurrent rename.
    """
    if type(nao) is not int or nao < 1:
        raise ValueError("trusted AO dimension must be a positive integer")
    matrix_fields = {"density", "mo_coeff", "j0", "k0", "j1", "k1", "jsum", "ksum"}
    shapes = {key: (nao, nao) for key in matrix_fields}
    shapes.update({"energy": (), "converged": (), "history_json": (),
                   "mo_energy": (nao,), "occupations": (nao,)})
    payload_bound = 8 * (len(matrix_fields) * nao**2 + 2 * nao + 1) + 1 + 4 * MAX_HISTORY_CHARS
    unpacked_bound = payload_bound + len(OUTPUT_FIELDS) * (MAX_NPY_HEADER_BYTES + 12)
    archive_bound = unpacked_bound + 65536
    with Path(path).open("rb") as stream:
        if os.fstat(stream.fileno()).st_size > archive_bound:
            raise ValueError("compressed output exceeds bounded archive budget")
        snapshot = stream.read(archive_bound + 1)
    if len(snapshot) > archive_bound:
        raise ValueError("compressed output exceeds bounded archive budget")
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            members = archive.infolist()
            expected = {name + ".npy" for name in OUTPUT_FIELDS}
            names = [member.filename for member in members]
            if len(names) != len(expected) or set(names) != expected:
                raise ValueError("output archive schema mismatch or duplicate members")
            if sum(member.file_size for member in members) > unpacked_bound:
                raise ValueError("output exceeds bounded decompression budget")
            for member in members:
                key = member.filename[:-4]
                if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED} or member.flag_bits & 1:
                    raise ValueError("unsupported output compression or encryption")
                member_payload_bound = (4 * MAX_HISTORY_CHARS if key == "history_json" else
                                        1 if key == "converged" else 8 * math.prod(shapes[key]))
                if member.file_size > member_payload_bound + MAX_NPY_HEADER_BYTES + 12:
                    raise ValueError(f"{key} exceeds bounded member decompression budget")
                with archive.open(member) as stream:
                    shape, dtype, header_bytes = _read_bounded_npy_header(stream)
                if shape != shapes[key]:
                    raise ValueError(f"{key} NPY shape does not match trusted AO dimension")
                if key == "history_json":
                    if dtype.kind != "U" or not 0 < dtype.itemsize <= 4 * MAX_HISTORY_CHARS:
                        raise ValueError("history_json must be a bounded Unicode scalar")
                elif key == "converged":
                    if dtype != np.dtype("bool"):
                        raise ValueError("converged must be a boolean scalar")
                elif dtype != np.dtype("float64"):
                    raise ValueError(f"{key} must be float64")
                if member.file_size != header_bytes + math.prod(shape) * dtype.itemsize:
                    raise ValueError(f"{key} has inconsistent NPY payload length")
        with np.load(io.BytesIO(snapshot), allow_pickle=False, max_header_size=MAX_NPY_HEADER_BYTES) as archive:
            return {name: np.array(archive[name], copy=True) for name in OUTPUT_FIELDS}
    except (zipfile.BadZipFile, EOFError, NotImplementedError, RuntimeError) as exc:
        raise ValueError("invalid candidate output archive") from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_identity(arrays: Mapping[str, np.ndarray], metadata: dict) -> str:
    digest = hashlib.sha256(_json(metadata).encode())
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(_json([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_case(case: dict, mode: str, input_dir: Path, reference_dir: Path,
                 seed: int = 20261111, memory_limit_mb: float = 2048) -> dict:
    """Prepare a public problem and private oracle in a fresh trusted process.

    ``case['id']`` is the exact suite identity, normally from ``expand_cases``.
    References are never placed in ``input_dir`` or returned in public inputs.
    The caller must keep ``reference_dir`` inaccessible to candidate workers.
    """
    case_id = case.get("id", case.get("name"))
    if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id):
        raise ValueError("case id must be a safe identifier")
    if mode not in {"exact", "df"}:
        raise ValueError("mode must be exact or df")
    input_dir, reference_dir = Path(input_dir).resolve(), Path(reference_dir).resolve()
    if input_dir == reference_dir or input_dir in reference_dir.parents or reference_dir in input_dir.parents:
        raise ValueError("public inputs and private references must be disjoint directories")
    input_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    request_path = reference_dir / f"{case_id}.request.json"
    manifest_path = reference_dir / f"{case_id}.manifest.json"
    request_path.write_text(_json({"case": case, "mode": mode, "seed": seed,
                                  "memory_limit_mb": memory_limit_mb,
                                  "input_dir": str(input_dir), "reference_dir": str(reference_dir)}) + "\n")
    command = [sys.executable, "-m", "harness_engineering.generation.science",
               "--prepare-request", str(request_path), "--manifest", str(manifest_path)]
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "TMPDIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
           if key in os.environ}
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    started = time.perf_counter()
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=env,
                            capture_output=True, text=True, timeout=300, check=False)
    log_path = reference_dir / f"{case_id}.reference.log"
    log_path.write_text("STDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr)
    if result.returncode != 0 or not manifest_path.is_file():
        raise RuntimeError(f"trusted reference preparation failed for {case_id}: {result.stderr[-2000:]}")
    prepared = json.loads(manifest_path.read_text())
    prepared["reference_execution"] = {"command": command, "returncode": result.returncode,
                                        "duration_s": time.perf_counter() - started,
                                        "log_path": str(log_path), "independent_process": True}
    manifest_path.write_text(_json(prepared) + "\n")
    return prepared


def _prepare_worker(request: dict) -> dict:
    # PySCF is imported only in the separate trusted preparation worker.
    from pyscf import df, gto, lib, scf
    from pyscf.gto import moleintor
    import pyscf
    import scipy
    lib.num_threads(1)
    case, mode = request["case"], request["mode"]
    case_id = case.get("id", case.get("name"))
    memory = float(request["memory_limit_mb"])
    if not math.isfinite(memory) or memory <= 0:
        raise ValueError("memory_limit_mb must be positive and finite")
    seed = request["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    mol = gto.M(atom=case["atom"], basis=case["basis"], unit=case.get("unit", "Angstrom"),
                charge=case.get("charge", 0), spin=case.get("spin", 0), verbose=0,
                max_memory=memory)
    if mol.spin != 0 or mol.nelectron <= 0 or mol.nelectron % 2 or mol._ecp or mol._pseudo:
        raise ValueError("only closed-shell, even-electron, all-electron RHF is supported")
    n = mol.nao_nr()
    auxbasis = case.get("auxbasis", "weigend")
    auxmol = df.addons.make_auxmol(mol, auxbasis) if mode == "df" else None
    estimate = (2 * 8 * n**4 if mode == "exact" else
                4 * 8 * auxmol.nao_nr() * n**2 + 3 * 8 * auxmol.nao_nr()**2) + 8 * 64 * n**2
    if estimate > memory * 1e6:
        raise MemoryError(f"integral/reference working set estimate {estimate / 1e6:g} MB exceeds {memory:g} MB")
    overlap = mol.intor_symmetric("int1e_ovlp")
    eigenvalues = np.linalg.eigvalsh(overlap)
    rank = int(np.count_nonzero(eigenvalues > 1e-10))
    if rank != n:
        raise ValueError("v1 reference protocol requires full-rank overlap above 1e-10")
    arrays = {"overlap": overlap, "hcore": mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc"),
              "enuc": np.asarray(mol.energy_nuc(), dtype=np.float64),
              "nelectron": np.asarray(mol.nelectron, dtype=np.int64)}
    if mode == "exact":
        arrays["eri"] = mol.intor("int2e", aosym="s1").reshape((n,) * 4)
    else:
        packed = df.incore.cholesky_eri(mol, auxmol=auxmol, max_memory=memory, verbose=0)
        arrays["df_factors"] = lib.unpack_tril(packed).reshape(-1, n, n)
    rng = np.random.default_rng(seed)
    trial = rng.normal(size=(2, n, n)) * 0.2
    arrays["trial_densities"] = (trial + trial.transpose(0, 2, 1)) * 0.5
    metadata = {
        "schema_version": PROBLEM_SCHEMA_VERSION, "id": case_id, "method": "RHF",
        "integral_mode": mode, "units": {"energy": "hartree", "geometry": "bohr"},
        "geometry_bohr": [[mol.atom_symbol(i), *map(float, mol.atom_coord(i))] for i in range(mol.natm)],
        "basis": case["basis"], "resolved_basis": mol._basis,
        "shells": [{"atom": int(mol.bas_atom(i)), "angular": int(mol.bas_angular(i)),
                     "exponents": mol.bas_exp(i).tolist(), "contractions": mol.bas_ctr_coeff(i).tolist()}
                    for i in range(mol.nbas)],
        "ao_labels": mol.ao_labels(), "ao_representation": "real spherical Gaussian AO, PySCF ordering",
        "nao": n, "overlap_rank": rank, "lindep_tol": 1e-10,
        "nelectron": int(mol.nelectron), "nalpha": int(mol.nelec[0]), "nbeta": int(mol.nelec[1]),
        "charge": int(mol.charge), "spin": int(mol.spin), "dtype": "float64",
        "auxbasis": auxbasis if mode == "df" else None, "random_seed": seed,
        "conventions": {"eri": "chemist: eri[p,q,r,s]=(pq|rs)",
                        "df_factors": "real Coulomb-metric-whitened B[P,p,q]; (pq|rs)~=sum_P B[P,p,q] B[P,r,s]",
                        "density": "spin-summed D=2*C_occ@C_occ.T", "fock": "F=H+J(D)-K(D)/2",
                        "energy": "E=Tr[D@(H+F)]/2+Enuc", "occupations": "ascending MO energies; 2 for nocc lowest then 0"},
        "provider": {"pyscf_version": pyscf.__version__, "numpy_version": np.__version__},
        "reference_boundary": "Independent SCF algorithm; shared PySCF Gaussian integrals and basis data.",
        "estimated_working_set_mb": estimate / 1e6,
    }
    def physical_atoms(molecule):
        atoms = []
        for atom in range(molecule.natm):
            shells = [{"angular": int(molecule.bas_angular(shell)),
                       "exponents": np.round(molecule.bas_exp(shell), 14).tolist(),
                       "contractions": np.round(molecule.bas_ctr_coeff(shell), 14).tolist()}
                      for shell in range(molecule.nbas) if molecule.bas_atom(shell) == atom]
            atoms.append({"nuclear_charge": int(molecule.atom_charge(atom)),
                          "geometry_bohr": np.round(molecule.atom_coord(atom), 10).tolist(),
                          "basis_shells": shells})
        # Normalize coordinate units, atom labels/order and basis-name aliases.
        return sorted(atoms, key=_json)
    physical_definition = {
        "schema": "rhf.physical-input/1", "method": "RHF", "integral_mode": mode,
        "atoms": physical_atoms(mol), "auxiliary_atoms": physical_atoms(auxmol) if auxmol is not None else None,
        "charge": int(mol.charge), "spin": int(mol.spin), "nelectron": int(mol.nelectron),
        "conventions": metadata["conventions"],
        "normalization": "Bohr coordinates rounded to 1e-10; resolved shell parameters rounded to 1e-14; atom-order independent; Cartesian frame retained",
    }
    physical_hash = hashlib.sha256(_json(physical_definition).encode()).hexdigest()
    metadata["physical_input_sha256"] = physical_hash
    identity_hash = _array_identity(arrays, metadata)
    metadata["identity_sha256"] = identity_hash
    identity = {"case_id": case_id, "schema_version": PROBLEM_SCHEMA_VERSION,
                "method": "RHF", "mode": mode, "nao": n, "nelectron": int(mol.nelectron),
                "scientific_input_sha256": identity_hash, "physical_input_sha256": physical_hash}
    problem_path = Path(request["input_dir"]) / f"{case_id}.npz"
    np.savez_compressed(problem_path, **arrays, metadata_json=np.asarray(_json(metadata)))
    reference = scf.RHF(mol)
    if mode == "df":
        reference = reference.density_fit(auxbasis=auxbasis)
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-12, 1e-9, 150
    energy = reference.kernel()
    if not reference.converged or not np.isfinite(energy):
        raise RuntimeError("trusted PySCF reference SCF did not converge")
    dm = reference.make_rdm1()
    j0, k0 = reference.get_jk(mol, arrays["trial_densities"][0])
    j1, k1 = reference.get_jk(mol, arrays["trial_densities"][1])
    jsum, ksum = reference.get_jk(mol, arrays["trial_densities"].sum(axis=0))
    final_j, final_k = reference.get_jk(mol, dm)
    def loaded_file(path):
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("loaded scientific runtime provenance is not a regular file")
        return {"path": str(resolved), "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size}
    actual_loaded = {
        name: loaded_file(importlib.import_module(name).__file__)
        for name in ("numpy", "numpy._core._multiarray_umath", "scipy",
                     "scipy.linalg._flapack", "pyscf", "pyscf.gto.moleintor")
    }
    # CDLL._name identifies the actual native integral wrapper loaded by PySCF.
    # This records selected used components, not every transitive dependency.
    actual_loaded["pyscf.gto.moleintor.libcgto"] = loaded_file(moleintor.libcgto._name)
    actual_loaded["python_executable"] = loaded_file(sys.executable)
    reference_data = {"energy": np.asarray(energy), "density": dm,
                      "mo_coeff": reference.mo_coeff, "mo_energy": reference.mo_energy,
                      "occupations": reference.mo_occ, "overlap": arrays["overlap"], "hcore": arrays["hcore"],
                      "j0": j0, "k0": k0, "j1": j1, "k1": k1, "jsum": jsum, "ksum": ksum,
                      "final_j": final_j, "final_k": final_k,
                      "converged": np.asarray(True), "metadata_json": np.asarray(_json({
                          "identity": identity, "reference": "PySCF RHF" if mode == "exact" else "PySCF DF-RHF",
                          "auxbasis": auxbasis if mode == "df" else None,
                          "provenance": {"process_id": os.getpid(), "python": sys.version,
                                         "python_executable": sys.executable, "pyscf_version": pyscf.__version__,
                                         "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                                         "actual_loaded": actual_loaded,
                                         "reference_algorithm": type(reference).__name__,
                                         "conv_tol": reference.conv_tol, "conv_tol_grad": reference.conv_tol_grad,
                                         "max_cycle": reference.max_cycle},
                          "anchor_energy": case.get("anchor_energy") if mode == "exact" else None}))}
    if not all(np.isfinite(array).all() for key, array in reference_data.items() if key != "metadata_json"):
        raise FloatingPointError("non-finite trusted reference")
    reference_path = Path(request["reference_dir"]) / f"{case_id}.reference.npz"
    np.savez_compressed(reference_path, **reference_data)
    return {"id": case_id, "problem_path": str(problem_path.resolve()),
            "input_sha256": sha256_file(problem_path), "reference_path": str(reference_path.resolve()),
            "reference_sha256": sha256_file(reference_path), "identity": identity}


def load_problem(path: str | Path) -> dict:
    """Load only numerical public arrays, never unpickle candidate data."""
    with np.load(path, allow_pickle=False) as archive:
        problem = {name: np.array(archive[name], copy=True) for name in archive.files if name != "metadata_json"}
        problem["metadata"] = json.loads(str(archive["metadata_json"].item()))
    return problem


def make_challenge_cases(spec: dict) -> list[dict]:
    """Return a rigidly moved development molecule in every required mode.

    These are additional development inputs, not holdout data. They use the
    same public integral schema and ordinary trusted reference preparation.
    Choose a polyatomic case when available, so angular basis functions and
    molecular orientation are exercised as well as origin invariance.
    """
    spec = validate_spec(spec)

    def atoms(case):
        value = case["atom"]
        if isinstance(value, str):
            rows = [line.split() for line in value.replace(";", "\n").splitlines() if line.strip()]
            if any(len(row) != 4 for row in rows):
                raise ValueError("rigid challenge needs explicit Cartesian atom coordinates")
            return [(row[0], list(map(float, row[1:]))) for row in rows]
        result = []
        for row in value:
            if len(row) == 2 and isinstance(row[1], (list, tuple)):
                result.append((str(row[0]), list(map(float, row[1]))))
            elif len(row) == 4:
                result.append((str(row[0]), list(map(float, row[1:]))))
            else:
                raise ValueError("rigid challenge needs explicit Cartesian atom coordinates")
        return result

    parsed = [(case, atoms(case)) for case in spec["cases"]]
    case, positions = next(((case, positions) for case, positions in parsed if len(positions) >= 3), parsed[0])
    rng = np.random.default_rng(spec["seed"] + 913)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = np.asarray([0.41, -0.73, 1.17])
    moved = [[symbol, (rotation @ np.asarray(coords) + translation).tolist()] for symbol, coords in positions]
    result = []
    for mode in spec["integral_modes"]:
        record = dict(case)
        record.update({"id": case["id"] + "-rigid-" + mode, "base_id": case["id"], "mode": mode,
                       "atom": moved, "auxbasis": spec.get("auxbasis", "weigend"),
                       "challenge": {"kind": "rigid_motion", "base_case_id": case["id"] + "-" + mode,
                                     "rotation": rotation.tolist(), "translation_input_units": translation.tolist()}})
        if not CASE_ID.fullmatch(record["id"]):
            raise ValueError("rigid challenge id exceeds identifier length limit")
        result.append(record)
    return result


def verify_rigid_motion(base_record: dict, moved_record: dict,
                        energy_tolerance: float = 1e-8) -> dict:
    """Require two independently verified results and energy invariance."""
    reasons, metrics = [], {}
    if (isinstance(energy_tolerance, bool) or not isinstance(energy_tolerance, (int, float))
            or not math.isfinite(energy_tolerance) or energy_tolerance < 0):
        raise ValueError("energy_tolerance must be finite and nonnegative")
    for name, record in (("base", base_record), ("moved", moved_record)):
        if record.get("status") != "passed":
            reasons.append(f"{name} calculation did not independently pass scientific validation")
    try:
        if (base_record["details"]["identity"]["mode"] != moved_record["details"]["identity"]["mode"]
                or base_record["details"]["identity"]["nelectron"] != moved_record["details"]["identity"]["nelectron"]):
            reasons.append("rigid-motion pair must have matching methods and electron counts")
        for label, key in (("candidate_energy_invariance_abs", "candidate_energy"),
                           ("reference_energy_invariance_abs", "reference_energy")):
            value = abs(float(base_record["details"][key]) - float(moved_record["details"][key]))
            if not math.isfinite(value):
                reasons.append(f"{label} is non-finite")
            else:
                metrics[label] = value
                if value > energy_tolerance:
                    reasons.append(f"{label}={value:.12g} exceeds {energy_tolerance:.12g}")
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append(f"incomplete rigid-motion evidence: {exc}")
    return {"status": "failed" if reasons else "passed", "passed": not reasons,
            "metrics": metrics, "reasons": reasons,
            "details": {"base_case_id": base_record.get("case_id"), "moved_case_id": moved_record.get("case_id")}}


def _error(value) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _reconstruct_jk(problem: dict, density: np.ndarray):
    if "eri" in problem:
        eri = problem["eri"]
        return (np.einsum("pqrs,rs->pq", eri, density, optimize=True),
                np.einsum("prqs,rs->pq", eri, density, optimize=True))
    factors = problem["df_factors"]
    return (np.einsum("Ppq,Prs,rs->pq", factors, factors, density, optimize=True),
            np.einsum("Ppr,Pqs,rs->pq", factors, factors, density, optimize=True))


def _invalid(case_id: str, reasons: list[str], details: dict | None = None) -> dict:
    return {"case_id": case_id, "status": "invalid_runtime", "metrics": {}, "reasons": reasons,
            "details": details or {}}


def verify_case(prepared: dict, output_path: Path, tolerances: dict,
                execution_record: dict) -> dict:
    """Validate raw candidate tensors and recompute all scientific metrics.

    Runtime metadata is trusted executor evidence, never a candidate field.
    A result being internally consistent is not sufficient: independent
    reference density/energy and trial J/K are also mandatory checks.
    """
    case_id = prepared["id"]
    details = {"identity": prepared.get("identity"), "execution": execution_record,
               "output_path": str(Path(output_path).resolve())}
    if execution_record.get("returncode") != 0 or execution_record.get("status") != "passed":
        return _invalid(case_id, ["candidate execution did not complete successfully"], details)
    if execution_record.get("input_sha256") != prepared.get("input_sha256"):
        return _invalid(case_id, ["execution input hash does not match prepared input"], details)
    try:
        if set(tolerances) != set(DEFAULT_TOLERANCES):
            raise ValueError("incomplete or unknown scientific tolerance protocol")
        for key, value in tolerances.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid tolerance: {key}")
        if sha256_file(prepared["problem_path"]) != prepared["input_sha256"]:
            raise ValueError("public input hash changed after preparation")
        if sha256_file(prepared["reference_path"]) != prepared["reference_sha256"]:
            raise ValueError("private reference hash changed after preparation")
        problem = load_problem(prepared["problem_path"])
        if problem["metadata"].get("identity_sha256") != prepared["identity"]["scientific_input_sha256"]:
            raise ValueError("input identity mismatch")
        with np.load(prepared["reference_path"], allow_pickle=False) as archive:
            reference = {name: np.array(archive[name]) for name in archive.files if name != "metadata_json"}
            reference_meta = json.loads(str(archive["metadata_json"].item()))
        if reference_meta["identity"] != prepared["identity"]:
            raise ValueError("reference identity mismatch")
        n = problem["overlap"].shape[0]
        output = load_candidate_output(output_path, n)
        for key in OUTPUT_FIELDS - {"history_json", "converged"}:
            value = output[key]
            if value.dtype != np.dtype("float64") or not np.isfinite(value).all():
                raise ValueError(f"{key} must contain only finite float64 values")
        if output["energy"].shape != ():
            raise ValueError("energy must be scalar")
        if output["converged"].shape != () or output["converged"].dtype.kind != "b":
            raise ValueError("converged must be a scalar boolean")
        for key in ("density", "j0", "k0", "j1", "k1", "jsum", "ksum"):
            if output[key].shape != (n, n):
                raise ValueError(f"{key} shape must be {(n, n)}")
        if output["mo_coeff"].shape != (n, n) or output["mo_energy"].shape != (n,) or output["occupations"].shape != (n,):
            raise ValueError("MO coefficient/energy/occupation shapes do not match full overlap rank")
        history = json.loads(str(output["history_json"].item()))
        if not isinstance(history, list) or not history:
            raise ValueError("history_json must be a nonempty list")
        previous_cycle = 0
        for row in history:
            if not isinstance(row, dict) or not {"cycle", "energy", "residual_norm"} <= set(row):
                raise ValueError("history row must include cycle, energy and residual_norm")
            cycle = row["cycle"]
            if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle <= previous_cycle:
                raise ValueError("history cycles must be increasing positive integers")
            previous_cycle = cycle
            for key, value in row.items():
                if not isinstance(value, (int, float, str, bool, type(None))):
                    raise ValueError(f"history values must be JSON scalars: {key}")
                if isinstance(value, (int, float)) and (isinstance(value, bool) or not math.isfinite(value)):
                    raise ValueError(f"non-finite or boolean history value: {key}")
            for key in ("energy", "residual_norm"):
                if isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
                    raise ValueError(f"history {key} must be finite and numerical")
        dm, coeff, eps, occ = (output[key] for key in ("density", "mo_coeff", "mo_energy", "occupations"))
        overlap, hcore = problem["overlap"], problem["hcore"]
        j, k = _reconstruct_jk(problem, dm)
        fock = hcore + j - 0.5 * k
        reconstructed_energy = float(0.5 * np.einsum("pq,qp", dm, hcore + fock) + problem["enuc"])
        nocc = int(problem["nelectron"]) // 2
        expected_occ = np.zeros(n)
        expected_occ[:nocc] = 2.0
        metrics = {
            "energy_max_abs": abs(float(output["energy"]) - float(reference["energy"])),
            "jk_max_abs": max(_error(output[key] - reference[key]) for key in ("j0", "k0", "j1", "k1", "jsum", "ksum")),
            "jk_symmetry_max_abs": max(_error(output[key] - output[key].T) for key in ("j0", "k0", "j1", "k1", "jsum", "ksum")),
            "jk_linearity_max_abs": max(_error(output["jsum"] - output["j0"] - output["j1"]), _error(output["ksum"] - output["k0"] - output["k1"])),
            "density_max_abs": _error(dm - reference["density"]),
            "density_symmetry_max_abs": _error(dm - dm.T),
            "electron_count_abs": abs(float(np.trace(dm @ overlap)) - int(problem["nelectron"])),
            "idempotency_max_abs": _error(dm @ overlap @ dm - 2 * dm),
            "residual_max_abs": _error(fock @ dm @ overlap - overlap @ dm @ fock),
            "orthogonality_max_abs": _error(coeff.T @ overlap @ coeff - np.eye(n)),
            "orbital_equation_max_abs": _error(fock @ coeff - (overlap @ coeff) * eps),
            "density_reconstruction_max_abs": _error(dm - (coeff * occ) @ coeff.T),
            "energy_reconstruction_abs": abs(float(output["energy"]) - reconstructed_energy),
            "occupation_max_abs": _error(occ - expected_occ),
            "orbital_order_error": float(max(0.0, -float(np.min(np.diff(eps), initial=0.0)))),
            "anchor_energy_abs": abs(float(output["energy"]) - reference_meta["anchor_energy"])
                                  if reference_meta.get("anchor_energy") is not None else 0.0,
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("candidate tensors produced non-finite derived scientific measurements")
        reasons = [f"{key}={value:.12g} exceeds {tolerances[key]:.12g}"
                   for key, value in metrics.items() if not math.isfinite(value) or value > tolerances[key]]
        if not bool(output["converged"].item()):
            reasons.append("candidate did not converge")
        details.update({"candidate_energy": float(output["energy"]), "reference_energy": float(reference["energy"]),
                        "recomputed_energy": reconstructed_energy, "history": history,
                        "candidate_converged": bool(output["converged"].item()), "reference_converged": True,
                        "output_sha256": sha256_file(output_path), "input_sha256": prepared["input_sha256"],
                        "reference_sha256": prepared["reference_sha256"],
                        "reference_provenance": reference_meta["provenance"],
                        "reference_boundary": problem["metadata"]["reference_boundary"]})
        return {"case_id": case_id, "status": "failed" if reasons else "passed",
                "metrics": metrics, "reasons": reasons, "details": details}
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, FloatingPointError) as exc:
        return _invalid(case_id, [f"{type(exc).__name__}: {exc}"], details)


def verify_nonconverged_case(prepared: dict, output_path: Path, tolerances: dict,
                            execution_record: dict, max_cycle: int = 1) -> dict:
    """Validate the truthful max-cycle contract, not a converged RHF solution.

    Schema, runtime, identity, trial J/K, electron count, occupations,
    idempotency, orthogonality and reconstruction retain their normal gates.
    Final-reference agreement and stationarity are measured but deliberately
    excluded from this separate contract. Ordinary ``verify_case`` is unchanged.
    """
    if isinstance(max_cycle, bool) or not isinstance(max_cycle, int) or max_cycle < 1:
        raise ValueError("max_cycle contract must be a positive integer")
    if (set(tolerances) != set(DEFAULT_TOLERANCES)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in tolerances.values())):
        raise ValueError("nonconvergence contract requires the complete finite tolerance protocol")
    excluded = {"energy_max_abs", "density_max_abs", "residual_max_abs",
                "orbital_equation_max_abs", "orbital_order_error", "anchor_energy_abs"}
    relaxed = dict(tolerances)
    for key in excluded:
        relaxed[key] = 1e300
    record = verify_case(prepared, output_path, relaxed, execution_record)
    record["kind"] = "nonconvergence_contract"
    record["details"].update({
        "validation_scope": "A max-cycle exhaustion contract; passing does not certify a converged scientific solution, final-Fock Aufbau occupation or global Fock spectral ordering.",
        "required_max_cycle": max_cycle,
        "excluded_acceptance_metrics": sorted(excluded),
        "enforced_acceptance_metrics": sorted(set(DEFAULT_TOLERANCES) - excluded),
    })
    if record["status"] == "invalid_runtime":
        return record
    reasons = [reason for reason in record["reasons"] if reason != "candidate did not converge"]
    if record["details"].get("candidate_converged") is not False:
        reasons.append("max-cycle challenge requires converged=false")
    history = record["details"].get("history", [])
    if not history or len(history) > max_cycle or any(row["cycle"] > max_cycle for row in history):
        reasons.append(f"iteration history exceeds max_cycle={max_cycle}")
    record.update(status="failed" if reasons else "passed", reasons=reasons)
    return record


def verify_suite(expected_ids: Sequence[str], records: Sequence[dict]) -> dict:
    """Reject missing, duplicate, unknown or empty evidence before aggregation."""
    expected, actual = list(expected_ids), [record.get("case_id") for record in records]
    reasons = []
    if not expected or not records:
        reasons.append("suite must have nonempty expected cases and result records")
    if len(set(expected)) != len(expected):
        reasons.append("duplicate expected case ids")
    if len(set(actual)) != len(actual):
        reasons.append("duplicate result case ids")
    if set(expected) != set(actual):
        reasons.append(f"case coverage mismatch: missing={sorted(set(expected)-set(actual))}, unknown={sorted(set(actual)-set(expected), key=str)}")
    for record in records:
        if record.get("status") != "passed":
            reasons.append(f"{record.get('case_id')}: {record.get('status')}: {'; '.join(record.get('reasons', []))}")
        if set(record.get("metrics", {})) != set(DEFAULT_TOLERANCES):
            reasons.append(f"{record.get('case_id')}: missing or unknown scientific metrics")
        elif not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                     for v in record["metrics"].values()):
            reasons.append(f"{record.get('case_id')}: invalid numerical measurements")
    metrics = {}
    for key in DEFAULT_TOLERANCES:
        values = [record.get("metrics", {}).get(key) for record in records]
        finite = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        metrics[key] = max(finite, default=0.0)
    return {"status": "failed" if reasons else "passed", "passed": not reasons,
            "metrics": metrics, "reasons": reasons,
            "details": {"expected_ids": expected, "actual_ids": actual, "case_count": len(records), "cases": list(records)}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-request", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = _prepare_worker(json.loads(args.prepare_request.read_text()))
    args.manifest.write_text(_json(manifest) + "\n")
    print(_json({"prepared": manifest["id"], "input_sha256": manifest["input_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
