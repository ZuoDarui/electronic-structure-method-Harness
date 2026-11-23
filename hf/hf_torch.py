import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import gc
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.linalg import eigh
from pyscf import gto, scf, df, dft, lib
from pyscf.dft import gen_grid, radi, numint
from pyscf.gto import moleintor


def import_cupy():
    """Import CuPy from this environment, or from the GPU image.

    The ``pyscf`` conda environment does not install CuPy. The notebook
    image already has a CUDA build in its system ``dist-packages``. That
    directory is appended, so packages already provided by conda stay first.
    """

    try:
        import cupy
        return cupy
    except ImportError:
        pass
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidate = Path(f"/usr/local/lib/python{version}/dist-packages")
    if not (candidate / "cupy").is_dir():
        raise ImportError("No module named 'cupy'")
    location = str(candidate)
    if location not in sys.path:
        sys.path.append(location)
    import cupy
    return cupy


def import_torch():
    """Import PyTorch from this environment, or from the GPU image.

    Like ``import_cupy``: the ``pyscf`` conda environment does not install
    Torch, the notebook image has a CUDA build in its system ``dist-packages``.
    """

    threads = lib.num_threads()
    try:
        import torch
    except ImportError:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        candidate = Path(f"/usr/local/lib/python{version}/dist-packages")
        if not (candidate / "torch").is_dir():
            raise ImportError("No module named 'torch'")
        location = str(candidate)
        if location not in sys.path:
            sys.path.append(location)
        import torch
    # The two libraries can share OpenMP; preserve the caller's budget even
    # on the first import, before a Hartree_Fock instance is constructed.
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    if lib.num_threads() != threads:
        lib.num_threads(threads)
    return torch


DENSE_BACKENDS = ("cpu", "cupy", "torch")


def configure_torch_cpu():
    """Hide CUDA and make an optional Torch runtime CPU-only."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FORCE_CPU"] = "1"
    try:
        import torch
    except ImportError:
        return {"torch_available": False, "torch_cuda_available": False}
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        raise RuntimeError("Torch CUDA is available after CPU-only configuration")
    return {"torch_available": True, "torch_cuda_available": False}


def orbital_gradient_rms(gradient, nocc, nvir):
    """RMS of the occupied-virtual orbital gradient.

    ``get_grad`` returns one entry per occupied-virtual pair. The raw
    Frobenius norm grows with the pair count, so a fixed cutoff stops
    accepting long chains after the energy has already settled.
    """

    count = max(1, int(nocc) * int(nvir))
    return float(np.linalg.norm(np.ravel(gradient)) / np.sqrt(count))


def scf_energy_terms(density, hcore, j_matrix, k_matrix, overlap, ene_nuc):
    """Energy components of one density with the J and K built from it."""

    terms = {
        "one_electron_hartree": float(np.einsum("ij,ji", density, hcore)),
        "coulomb_hartree": float(0.5 * np.einsum("ij,ji", density, j_matrix)),
        "exchange_hartree": float(-0.25 * np.einsum("ij,ji", density, k_matrix)),
        "nuclear_hartree": float(ene_nuc),
        "electron_count": float(np.einsum("ij,ji", density, overlap)),
    }
    terms["energy_hartree"] = (
        terms["nuclear_hartree"] + terms["one_electron_hartree"]
        + terms["coulomb_hartree"] + terms["exchange_hartree"]
    )
    return terms


def check_physical_energy(terms, stage, nelectron=None, electron_tol=1e-6):
    """Raise when ``scf_energy_terms`` describe an impossible state.

    For any nonzero real density the Coulomb self-repulsion is positive and
    the exchange energy is negative, so a sign error means the J or K build
    is wrong. A bound molecule also has a negative total energy. Pass
    ``nelectron=None`` for a guess density that is not normalized.
    """

    failures = []
    if not terms["coulomb_hartree"] > 0.0:
        failures.append(
            f"Coulomb energy E_J={terms['coulomb_hartree']:.12g} Ha is not positive")
    if not terms["exchange_hartree"] < 0.0:
        failures.append(
            f"exchange energy E_K={terms['exchange_hartree']:.12g} Ha is not negative")
    if not terms["energy_hartree"] < 0.0:
        failures.append(
            f"total energy E={terms['energy_hartree']:.12g} Ha is not negative")
    if nelectron is not None:
        error = abs(terms["electron_count"] - float(nelectron))
        if not error <= electron_tol:
            failures.append(
                f"electron count Tr(DS)={terms['electron_count']:.12g} differs "
                f"from {int(nelectron)} by {error:.3e} > {electron_tol:.0e}")
    if failures:
        raise RuntimeError(
            f"Physical guard failed at {stage}: " + "; ".join(failures)
            + " terms=" + json.dumps(terms, sort_keys=True))


def matching_cosx_grids(mol, grid_rad, grid_ang):
    """Atomic grid shared with the PySCF SGX reference.

    The radial map is Treutler-Ahlrichs and the angular grid is pruned with
    the NWChem rule. The mask is built once and reused for every K build on
    this grid.
    """

    grids = gen_grid.Grids(mol)
    symbols = {mol.atom_pure_symbol(i) for i in range(mol.natm)}
    grids.atom_grid = {
        symbol: (int(grid_rad), int(grid_ang)) for symbol in symbols
    }
    grids.radi_method = radi.treutler_ahlrichs
    grids.prune = gen_grid.nwchem_prune
    grids.build(with_non0tab=True)
    grids.non0tab = grids.make_mask(mol, grids.coords)
    grids.screen_index = grids.non0tab
    return grids


def _check_thread_budget(parallel_workers, threads, cpu_cores):
    """Keep Python workers and BLAS threads from oversubscribing the node."""

    product = int(parallel_workers) * int(threads)
    if product > int(cpu_cores):
        raise ValueError(
            "parallel_workers * threads exceeds cpu_cores: "
            f"{int(parallel_workers)} * {int(threads)} > {int(cpu_cores)}"
        )
    return product


# PySCF sizes SGX blocks from max_memory minus the resident set. Its default
# of 2000 MB clamps the block to the library minimum once RSS passes that.
_DEFAULT_MAX_MEMORY_MB = 8000


def job_max_memory_mb(resources):
    """Return the SGX and DF block budget in MB.

    A quarter of the job memory leaves a large block after a few GiB of
    density and DIIS matrices. Jobs without ``memory_gib`` use a budget
    above the PySCF default.
    """

    memory_gib = None if resources is None else resources.get("memory_gib")
    if memory_gib is None:
        return _DEFAULT_MAX_MEMORY_MB
    memory_gib = float(memory_gib)
    if memory_gib <= 0:
        raise ValueError("resources.memory_gib must be positive")
    return memory_gib * 1024.0 * 0.25


def job_k_memory_mb(resources):
    """Return the fixed SGX block budget and the process ceiling in MB.

    ``resources.k_memory_gib`` sizes every SGX block from that budget alone,
    so the DIIS history and J's buffers no longer shrink it. The ceiling is
    ``memory_ceiling_fraction`` (default 0.75) of ``memory_gib``. Without
    ``k_memory_gib`` both are None and blocks follow max_memory minus RSS.
    """

    resources = resources or {}
    k_memory_gib = resources.get("k_memory_gib")
    if k_memory_gib is None:
        return None, None
    k_memory_gib = float(k_memory_gib)
    if k_memory_gib <= 0:
        raise ValueError("resources.k_memory_gib must be positive")
    memory_gib = resources.get("memory_gib")
    if memory_gib is None:
        return k_memory_gib * 1024.0, None
    fraction = float(resources.get("memory_ceiling_fraction", 0.75))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("resources.memory_ceiling_fraction must lie in (0, 1]")
    ceiling = float(memory_gib) * 1024.0 * fraction
    if k_memory_gib * 1024.0 >= ceiling:
        raise ValueError("resources.k_memory_gib must stay below the memory ceiling")
    return k_memory_gib * 1024.0, ceiling


def display_basis_name(basis):
    """Return the canonical display spelling for common basis names."""
    if str(basis).lower().replace("_", "-") == "cc-pvtz":
        return "cc-pVTZ"
    return str(basis)


def read_xyz(path):
    """Read an XYZ file and return a PySCF atom string and header metadata."""
    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file is incomplete: {path}")
    try:
        natm = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First XYZ line must be an atom count: {path}") from exc

    coordinate_lines = [line for line in lines[2:] if line.strip()]
    if len(coordinate_lines) != natm:
        raise ValueError(
            f"XYZ atom count mismatch for {path}: header={natm}, "
            f"coordinates={len(coordinate_lines)}"
        )

    atoms = []
    for line_number, line in enumerate(coordinate_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ coordinate at line {line_number}: {line}")
        atoms.append(" ".join(fields[:4]))

    comment = lines[1].strip()
    expected_basis_functions = None
    match = re.search(r"nao[_-]?ccpvtz\s*=\s*(\d+)", comment, re.IGNORECASE)
    if match:
        expected_basis_functions = int(match.group(1))

    return ";".join(atoms), {
        "xyz_path": str(path),
        "xyz_atom_count": natm,
        "xyz_comment": comment,
        "xyz_expected_basis_functions": expected_basis_functions,
    }


def build_molecule_from_xyz(path, basis="cc-pvtz", unit="Angstrom",
                            charge=0, spin=0, verbose=0):
    """Build a PySCF molecule from XYZ and return it with input metadata."""
    atom, xyz_metadata = read_xyz(path)
    mol = gto.M(
        atom=atom,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        verbose=verbose,
    )
    return mol, xyz_metadata


def molecule_metadata(mol, basis, source_path=None, xyz_metadata=None):
    """Return reproducible basis and system metadata for logs and summaries."""
    alpha, beta = mol.nelec
    metadata = {
        "source_path": str(source_path) if source_path is not None else None,
        "basis": display_basis_name(basis),
        "basis_input": str(basis),
        "basis_functions": int(mol.nao_nr()),
        "basis_shells": int(mol.nbas),
        "atom_count": int(mol.natm),
        "nelectron": int(mol.nelectron),
        "nalpha": int(alpha),
        "nbeta": int(beta),
        "nocc": int(alpha) if alpha == beta else None,
        "charge": int(mol.charge),
        "spin": int(mol.spin),
    }
    if xyz_metadata:
        metadata.update(xyz_metadata)
    return metadata


def validate_basis_function_count(mol_metadata, *, abort_on_mismatch=True):
    """Validate the alk375 header AO count when one is present."""
    expected = mol_metadata.get("xyz_expected_basis_functions")
    actual = mol_metadata["basis_functions"]
    if expected is None:
        return None
    if expected == actual:
        return None
    message = (
        "Basis-function count mismatch: "
        f"XYZ header expects {expected}, PySCF built {actual} "
        f"for {mol_metadata['basis']}"
    )
    if abort_on_mismatch:
        print(f"WARNING: {message}", file=sys.stderr)
        raise ValueError(message)
    return message


def _emit_progress(message, log_file=None, molecule_name=None):
    """Write one SCF progress line to the active molecule log and flush it."""
    if molecule_name:
        message = f"{molecule_name}: {message}"
    print(message, flush=True)
    if log_file is None:
        return
    streams = getattr(sys.stdout, "streams", None)
    if streams is not None and log_file in streams:
        return
    log_file.write(message + "\n")
    log_file.flush()


def _output_prefix_path(config, config_path, source_path):
    """Return the output path stem for the molecule being calculated."""
    output_dir = Path(config.get("output_dir", "."))
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configured = config.get("output_prefix")
    if configured:
        output_prefix = Path(str(configured))
    elif source_path is not None:
        output_prefix = Path(Path(source_path).stem)
    else:
        output_prefix = Path(config_path.stem)
    if output_prefix.is_absolute():
        return output_prefix
    return output_dir / output_prefix


def _available_log_path(prefix):
    """Return a log path that leaves an existing result in place.

    The first run writes ``<prefix>.log``. A repeated submission writes
    ``<prefix>_2.log``, then ``<prefix>_3.log``, and so on.
    """

    path = prefix.with_suffix(".log")
    if not path.exists():
        return path
    number = 2
    while True:
        candidate = prefix.parent / f"{prefix.name}_{number}.log"
        if not candidate.exists():
            return candidate
        number += 1


def estimate_dense_solver_memory_gib(basis_functions, matrix_count=8):
    """Estimate dense-array storage; this is a preflight estimate, not a limit."""
    matrix_bytes = int(basis_functions) ** 2 * np.dtype(np.float64).itemsize
    return {
        "one_dense_matrix_gib": matrix_bytes / (1024 ** 3),
        "estimated_dense_working_set_gib": matrix_count * matrix_bytes / (1024 ** 3),
        "matrix_count_assumption": int(matrix_count),
    }

_SCHWARZ_EXACT_MAX_SHELLS = 128
_SCHWARZ_SHELL_BLOCK = 4
_AUX_SHELL_BLOCK = 32
_COSX_SHELL_BLOCK = 4
_COSX_LOCAL_RUN = 12
_SHELL_RADIUS_PRECISION = 1e-20
_SHELL_RADIUS_PAD = 10.0
# Extended molecules leave the +10 Bohr coverage sphere. Their neighborhoods
# use the bare Gaussian tail so the work per shell stays bounded.
_LOCAL_RADIUS_PRECISION = 1e-6
_LOCAL_DOMAIN_BOHR = 24.0
_LOCAL_HALO_BOHR = 4.0
_DENSE_CHECK_TOL = 1e-10


def _cupy_device_description(cupy):
    device = cupy.cuda.Device()
    props = cupy.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return (f"{name} (device {device.id}, {props['totalGlobalMem'] / 1024 ** 3:.1f} GiB, "
            f"cupy {cupy.__version__}, cuda runtime "
            f"{cupy.cuda.runtime.runtimeGetVersion()})")


def resolve_torch_device(torch, requested="auto"):
    """``'auto'`` picks CUDA when Torch sees a device, otherwise the CPU."""

    requested = str(requested or "auto").lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("torch_device must be 'cpu', 'cuda', 'cuda:N' or 'auto'")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch_device='cuda' requires a visible CUDA device")
    return device


class _CupyOps:
    name = "CuPy"

    def __init__(self):
        self.cupy = import_cupy()
        from cupyx.scipy.linalg import solve_triangular

        self._solve_triangular = solve_triangular
        self.device = _cupy_device_description(self.cupy)

    def asarray(self, matrix):
        return self.cupy.asarray(matrix)

    def host(self, matrix):
        return self.cupy.asnumpy(matrix)

    def copy(self, matrix):
        return self.cupy.array(matrix, copy=True)

    def cholesky_inverse(self, matrix):
        """Inverse of the lower Cholesky factor of ``matrix``."""
        cupy = self.cupy
        lower = cupy.linalg.cholesky(matrix)
        inverse = self._solve_triangular(lower, cupy.eye(matrix.shape[0]), lower=True)
        # cuSOLVER reports a failed factorization as NaN, not as an exception.
        if not self.all_finite(inverse):
            raise RuntimeError("CuPy Cholesky factorization of the overlap failed")
        return inverse

    def eigh(self, matrix):
        return self.cupy.linalg.eigh(matrix)

    def vdot(self, left, right):
        return float(self.cupy.vdot(left, right).real)

    def norm(self, matrix):
        return float(self.cupy.linalg.norm(matrix))

    def all_finite(self, matrix):
        return bool(self.cupy.isfinite(matrix).all())

    def pool_gib(self):
        return self.cupy.get_default_memory_pool().total_bytes() / 1024 ** 3

    def release(self):
        self.cupy.get_default_memory_pool().free_all_blocks()


class _TorchOps:
    name = "Torch"

    def __init__(self, device="auto"):
        threads = lib.num_threads()
        torch = import_torch()
        self.torch = torch
        self.dev = resolve_torch_device(torch, device)
        # Importing Torch can change the shared OpenMP runtime. Apply the job's
        # thread budget on CUDA jobs too, since their integrals still run on CPU.
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        if self.dev.type == "cuda":
            index = self.dev.index if self.dev.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            self.device = (f"{props.name} (torch {torch.__version__}, cuda:{index}, "
                           f"{props.total_memory / 1024 ** 3:.1f} GiB, "
                           f"cuda runtime {torch.version.cuda})")
        else:
            self.device = f"cpu (torch {torch.__version__}, {torch.get_num_threads()} threads)"

    def asarray(self, matrix):
        torch = self.torch
        if isinstance(matrix, torch.Tensor):
            return matrix.detach().to(device=self.dev, dtype=torch.float64)
        return torch.as_tensor(np.asarray(matrix, dtype=np.float64), device=self.dev)

    def host(self, matrix):
        return matrix.detach().cpu().numpy()

    def copy(self, matrix):
        return matrix.clone()

    def cholesky_inverse(self, matrix):
        """Inverse of the lower Cholesky factor of ``matrix``."""
        torch = self.torch
        lower, info = torch.linalg.cholesky_ex(matrix)
        if int(info) != 0:
            raise RuntimeError("Torch Cholesky factorization of the overlap failed")
        identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        inverse = torch.linalg.solve_triangular(lower, identity, upper=False)
        if not self.all_finite(inverse):
            raise RuntimeError("Torch Cholesky factorization of the overlap failed")
        return inverse

    def eigh(self, matrix):
        return self.torch.linalg.eigh(matrix)

    def vdot(self, left, right):
        return float(self.torch.vdot(left.reshape(-1), right.reshape(-1)))

    def norm(self, matrix):
        return float(self.torch.linalg.norm(matrix))

    def all_finite(self, matrix):
        return bool(self.torch.isfinite(matrix).all())

    def pool_gib(self):
        if self.dev.type != "cuda":
            return 0.0
        return self.torch.cuda.memory_reserved(self.dev) / 1024 ** 3

    def release(self):
        if self.dev.type == "cuda":
            self.torch.cuda.empty_cache()


def _diis_coefficients(gram):
    """Pulay coefficients from the error Gram matrix, or None (Phase 1 rules)."""

    space = gram.shape[0]
    B = np.zeros((space + 1, space + 1))
    B[-1, :-1] = B[:-1, -1] = 1
    B[-1, -1] = 0
    B[:-1, :-1] = gram
    g = np.zeros(space + 1)
    g[-1] = 1
    if not np.isfinite(np.linalg.cond(B)) or np.linalg.cond(B) > 1e12:
        return None
    try:
        return scipy.linalg.solve(B, g, assume_a='sym')[:-1]
    except scipy.linalg.LinAlgError:
        return None


class DenseLinearAlgebra:
    """Generalized eigensolve, density and DIIS for one SCF grid stage.

    ``backend='cpu'`` uses SciPy ``eigh(F, S)`` and BLAS products: the
    density is one GEMM and the commutator is F D S minus its transpose.
    ``backend='cupy'`` and ``backend='torch'`` keep S, the inverse of its
    Cholesky factor and the DIIS history on their device (``torch_device``
    selects the Torch CPU or a CUDA device). Torch CPU uses a zero-copy
    SciPy/LAPACK generalized eigensolve, avoiding the extra dense transforms
    required by torch.linalg.eigh, which has no generalized-eigenproblem API.
    CPU matrix products and DIIS still use FP64 Torch. J and K never move. The orbital
    energies, coefficients and density handed back to the SCF are NumPy arrays.

    Each new error vector is dotted only with the stored ones; the Gram
    entries keep the operand order of the full Phase 1 recomputation.
    ``check_calls`` repeats the first eigensolves, commutators and
    extrapolations with NumPy/SciPy and raises when a result differs by more
    than ``check_tol``.
    """

    def __init__(self, overlap, backend="cpu", max_diis=8, check_calls=0,
                 check_tol=_DENSE_CHECK_TOL, torch_device="auto"):
        if backend not in DENSE_BACKENDS:
            raise ValueError(f"dense_backend must be one of {DENSE_BACKENDS}")
        self.backend = backend
        self.max_diis = int(max_diis)
        self.overlap = np.asarray(overlap, dtype=np.float64)
        self.errors = []
        self.focks = []
        self._gram = np.zeros((0, 0))
        self.check_calls = int(check_calls)
        self.check_tol = float(check_tol)
        self.check_max = {}
        self._checks_done = 0
        self._reference_error = None
        self._shadow_errors = []
        self._shadow_focks = []
        if backend == "cpu":
            from pyscf import lib
            self.device = f"cpu (NumPy/SciPy, {lib.num_threads()} threads)"
            self.eigensolver = "scipy.linalg.eigh"
            return
        self._ops = _CupyOps() if backend == "cupy" else _TorchOps(torch_device)
        self.device = self._ops.device
        self._s = self._ops.asarray(self.overlap)
        self._torch_cpu = backend == "torch" and self._ops.dev.type == "cpu"
        if self._torch_cpu:
            # Validate the metric once without forming an unnecessary inverse.
            try:
                scipy.linalg.cholesky(self.overlap, lower=True)
            except scipy.linalg.LinAlgError as exc:
                raise RuntimeError("Torch CPU overlap is not positive definite") from exc
            self._linv = None
            self.eigensolver = "scipy.linalg.eigh (zero-copy Torch CPU bridge)"
        else:
            self._linv = self._ops.cholesky_inverse(self._s)
            self.eigensolver = ("torch.linalg.eigh" if backend == "torch"
                                else "cupy.linalg.eigh")

    def _checking(self):
        return self.backend != "cpu" and self._checks_done < self.check_calls

    def _check(self, name, value, reference):
        difference = float(np.max(np.abs(np.asarray(value) - np.asarray(reference))))
        self.check_max[name] = max(self.check_max.get(name, 0.0), difference)
        if not difference <= self.check_tol:
            raise RuntimeError(
                f"{self._ops.name} dense {name} differs from the CPU result by "
                f"{difference:.3e} > {self.check_tol:.0e}")

    def _host(self, matrix):
        if isinstance(matrix, np.ndarray):
            return matrix
        return self._ops.host(matrix)

    def pool_gib(self):
        """Device memory held by the CuPy pool or the Torch caching allocator, in GiB."""
        if self.backend == "cpu":
            return 0.0
        return self._ops.pool_gib()

    def _cpu_commutator(self, fock, dm):
        # (F D S)^T = S D F for symmetric F, D and S.
        half = fock @ dm @ self.overlap
        return half - half.T

    @staticmethod
    def _cpu_density(mo_coeff, nocc):
        occupied = mo_coeff[:, :nocc]
        return 2.0 * (occupied @ occupied.T)

    def commutator(self, fock, dm):
        """DIIS error F D S - S D F."""
        if self.backend == "cpu":
            return self._cpu_commutator(fock, dm)
        ops = self._ops
        half = ops.asarray(fock) @ ops.asarray(dm) @ self._s
        error = half - half.T
        if self._checking():
            self._reference_error = self._cpu_commutator(fock, dm)
            self._check("diis_error", ops.host(error), self._reference_error)
        return error

    def norm(self, error):
        if self.backend == "cpu":
            return float(np.linalg.norm(error))
        return self._ops.norm(error)

    def push(self, error, fock):
        """Store one error/Fock pair and keep at most ``max_diis`` of them."""
        if self.backend == "cpu":
            stored = np.array(fock, copy=True)
            dots = [np.vdot(error, old).real for old in self.errors]
            self_dot = np.vdot(error, error).real
        else:
            ops = self._ops
            stored = ops.copy(ops.asarray(fock))
            dots = [ops.vdot(error, old) for old in self.errors]
            self_dot = ops.vdot(error, error)
            if self._checking():
                self._shadow_errors.append(self._reference_error)
                self._shadow_focks.append(np.array(self._host(fock), copy=True))
        count = len(self.errors)
        gram = np.zeros((count + 1, count + 1))
        gram[:count, :count] = self._gram
        gram[count, :count] = gram[:count, count] = dots
        gram[count, count] = self_dot
        self.errors.append(error)
        self.focks.append(stored)
        self._gram = gram
        if len(self.errors) > self.max_diis:
            self.errors.pop(0)
            self.focks.pop(0)
            self._gram = self._gram[1:, 1:].copy()
            if self._shadow_errors:
                self._shadow_errors.pop(0)
                self._shadow_focks.pop(0)

    def extrapolate(self):
        """DIIS Fock from the stored history, or None when it is not usable."""
        if len(self.errors) < 2:
            return None
        coefficients = _diis_coefficients(self._gram)
        checking = self._checking() and len(self._shadow_errors) == len(self.errors)
        if checking:
            errors = self._shadow_errors
            gram = np.array([[np.vdot(errors[max(i, j)], errors[min(i, j)]).real
                              for j in range(len(errors))] for i in range(len(errors))])
            reference = _diis_coefficients(gram)
            if (coefficients is None) != (reference is None):
                raise RuntimeError(
                    f"{self._ops.name} and CPU DIIS disagree on whether to extrapolate")
        if coefficients is None:
            return None
        if self.backend == "cpu":
            return np.einsum("i,ijk->jk", coefficients, self.focks)
        fock = float(coefficients[0]) * self.focks[0]
        for coefficient, stored in zip(coefficients[1:], self.focks[1:]):
            if self.backend == "torch":
                fock.add_(stored, alpha=float(coefficient))
            else:
                fock += float(coefficient) * stored
        if checking:
            self._check("diis_coefficients", coefficients, reference)
            self._check("diis_fock", self._ops.host(fock),
                        np.einsum("i,ijk->jk", reference, self._shadow_focks))
        return fock

    def solve(self, fock, nocc):
        """Orbital energies, coefficients and density 2 C_occ C_occ^T."""
        if self.backend == "cpu":
            mo_energy, mo_coeff = eigh(fock, self.overlap)
            return mo_energy, mo_coeff, self._cpu_density(mo_coeff, nocc)
        ops = self._ops
        if self._torch_cpu:
            mo_energy, mo_coeff = eigh(self._host(fock), self.overlap)
            values, coeff = ops.asarray(mo_energy), ops.asarray(mo_coeff)
        else:
            reduced = self._linv @ ops.asarray(fock) @ self._linv.T
            reduced = 0.5 * (reduced + reduced.T)
            values, vectors = ops.eigh(reduced)
            del reduced
            coeff = self._linv.T @ vectors
            del vectors
        occupied = coeff[:, :nocc]
        density_device = 2.0 * (occupied @ occupied.T)
        mo_energy = ops.host(values)
        mo_coeff = ops.host(coeff)
        density = ops.host(density_device)
        del coeff, occupied, density_device, values
        if not (np.isfinite(mo_energy).all() and np.isfinite(density).all()):
            raise RuntimeError(f"{ops.name} eigensolve returned non-finite values")
        if self._checking():
            host_fock = self._host(fock)
            ref_energy, ref_coeff = eigh(host_fock, self.overlap)
            ref_density = self._cpu_density(ref_coeff, nocc)
            self._check("density", density, ref_density)
            self._check("occupied_orbital_energy", mo_energy[:nocc], ref_energy[:nocc])
            self._check("band_energy", np.einsum("ij,ji", host_fock, density),
                        np.einsum("ij,ji", host_fock, ref_density))
            self._checks_done += 1
            if not self._checking():
                self._shadow_errors.clear()
                self._shadow_focks.clear()
        return mo_energy, mo_coeff, density

    def clear(self):
        """Drop the DIIS history (host or device memory)."""
        self.errors.clear()
        self.focks.clear()
        self._gram = np.zeros((0, 0))
        self._shadow_errors.clear()
        self._shadow_focks.clear()
        self._reference_error = None
        if self.backend != "cpu":
            self._ops.release()


def _pin_sgx_blocks(engine):
    """Let ``engine._hf_blksize`` override the block of ``get_k_only``.

    PySCF sizes the SGX block inside ``get_k_only`` from max_memory minus
    the resident set at that moment, which includes the DIIS history and,
    while the two run together, J's buffers. The screening data is rebuilt
    whenever the K tolerance changes, so every rebuilt object gets the
    override. ``None`` keeps PySCF's own block.
    """

    build = engine._build_pjs

    def build_pjs(direct_scf_tol):
        build(direct_scf_tol)
        data = engine._pjs_data
        loop_data = data.get_loop_data

        def get_loop_data(*args, **kwargs):
            values = loop_data(*args, **kwargs)
            pinned = engine._hf_blksize
            if pinned is None:
                return values
            return tuple(values[:-1]) + (min(int(pinned), int(data.grids.weights.size)),)

        data.get_loop_data = get_loop_data

    engine._build_pjs = build_pjs
    engine._hf_blksize = None


class Hartree_Fock:
    """Custom RHF solver with optional task parallelism.

    PySCF's ``lib.num_threads`` controls threads inside compiled kernels, but
    the shell-pair and grid-batch loops in this module are Python loops.  The
    ``parallel_backend='cpu'`` mode uses a thread pool around those independent
    tasks and reduces private matrices in the caller, avoiding races on the
    Fock matrix.  ``'gpu'`` is reserved for an optional CuPy contraction path;
    it fails explicitly when CuPy is unavailable instead of silently claiming
    to use a GPU.
    """

    def __init__(self, mol, max_memory=4000, parallel_backend="serial",
                 parallel_workers=1, parallel_chunk_size=1,
                 parallel_min_shells=12, jk_engine="pyscf_reference",
                 hierarchical_options=None, k_max_memory_mb=None,
                 memory_ceiling_mb=None, dense_backend="torch",
                 dense_check_calls=0, torch_device="auto"):
        self.mol = mol
        self.max_memory = max_memory
        if k_max_memory_mb is not None and not float(k_max_memory_mb) > 0:
            raise ValueError("k_max_memory_mb must be positive")
        if memory_ceiling_mb is not None and not float(memory_ceiling_mb) > 0:
            raise ValueError("memory_ceiling_mb must be positive")
        self.k_max_memory_mb = None if k_max_memory_mb is None else float(k_max_memory_mb)
        self.memory_ceiling_mb = None if memory_ceiling_mb is None else float(memory_ceiling_mb)
        self._last_k_budget_mb = None
        if dense_backend not in DENSE_BACKENDS:
            raise ValueError(f"dense_backend must be one of {DENSE_BACKENDS}")
        if dense_backend == "torch":
            try:
                resolve_torch_device(import_torch(), torch_device)
            except Exception as exc:
                raise RuntimeError(
                    f"dense_backend='torch' with torch_device={torch_device!r} "
                    "needs PyTorch and, for CUDA, a visible device"
                ) from exc
        if dense_backend == "cupy":
            try:
                cupy = import_cupy()
                if cupy.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("no CUDA device is visible")
            except Exception as exc:
                raise RuntimeError(
                    "dense_backend='cupy' requires CuPy and a visible CUDA device"
                ) from exc
        if (isinstance(dense_check_calls, bool)
                or not isinstance(dense_check_calls, (int, np.integer))
                or dense_check_calls < 0):
            raise ValueError("dense_check_calls must be a non-negative integer")
        self.dense_backend = dense_backend
        self.dense_check_calls = int(dense_check_calls)
        self.torch_device = str(torch_device)
        self._last_k_grid = None
        self.nao = mol.nao_nr()
        self.nshells = mol.nbas
        if jk_engine not in ("hierarchical_rijcosx", "pyscf_reference"):
            raise ValueError("jk_engine must be hierarchical_rijcosx or pyscf_reference")
        self.jk_engine = jk_engine
        self.hierarchical_options = dict(hierarchical_options or {})
        self._hierarchical_j_engines = {}
        self._local_k_engines = {}
        self.last_j_stats = {}
        self.last_k_stats = {}
        self.last_scf_result = None

        backend = str(parallel_backend).lower().replace("_", "-")
        if backend == "auto":
            backend = "cpu" if int(parallel_workers) > 1 else "serial"
        if backend not in ("serial", "cpu", "gpu"):
            raise ValueError(
                "parallel_backend must be 'serial', 'cpu', or 'gpu'"
            )
        if (isinstance(parallel_workers, bool)
                or not isinstance(parallel_workers, (int, np.integer))
                or parallel_workers <= 0):
            raise ValueError("parallel_workers must be a positive integer")
        if (isinstance(parallel_chunk_size, bool)
                or not isinstance(parallel_chunk_size, (int, np.integer))
                or parallel_chunk_size <= 0):
            raise ValueError("parallel_chunk_size must be a positive integer")
        if (isinstance(parallel_min_shells, bool)
                or not isinstance(parallel_min_shells, (int, np.integer))
                or parallel_min_shells <= 0):
            raise ValueError("parallel_min_shells must be a positive integer")
        if backend == "gpu":
            try:
                import_cupy()
            except ImportError as exc:
                raise RuntimeError(
                    "parallel_backend='gpu' requires CuPy; install a CuPy "
                    "build matching the CUDA runtime"
                ) from exc
        self.parallel_backend = backend
        self.parallel_workers = int(parallel_workers)
        self.parallel_chunk_size = int(parallel_chunk_size)
        self.parallel_min_shells = int(parallel_min_shells)

        self.ao_loc = mol.ao_loc_nr()

        self.schwarz_thresh = 1e-12
        self.density_thresh = 1e-12
        self._ri_cache = None
        self._df_engines = {}
        self._sgx_engines = {}
        self._pool = None
        self._stats_lock = threading.Lock()
        self.schwarz_pairs_evaluated = 0
        self.significant_shell_pairs = 0
        self.significant_pair_fraction = 0.0
        self.last_ri_screened_pairs = 0
        self.last_shell_pairs = 0
        self.cosx_full_tensor_calls = 0

        self.parallel_active = (
            backend == "gpu"
            or (backend == "cpu" and self.parallel_workers > 1
                and self.nshells >= self.parallel_min_shells)
        )

        self._prepare_shell_neighbors()
        self._build_schwarz_ints()

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass
            self._pool = None

    def _executor(self):
        """Reuse one thread pool so each J/K build does not start new workers."""
        pool = self._pool
        if pool is None or pool._shutdown:
            self._pool = ThreadPoolExecutor(max_workers=self.parallel_workers)
        return self._pool

    def _parallel_map(self, function, tasks):
        """Map independent integral tasks with deterministic reduction order."""
        if not self.parallel_active or self.parallel_workers == 1:
            return (function(task) for task in tasks)
        # The integral calls and NumPy contractions release the GIL.  Keeping
        # one process avoids copying a large Mole object and its integral data.
        def results():
            pool = self._executor()
            pending = deque()
            task_iter = iter(tasks)
            chunk_size = self.parallel_chunk_size

            def chunks():
                while True:
                    chunk = []
                    for _ in range(chunk_size):
                        try:
                            chunk.append(next(task_iter))
                        except StopIteration:
                            break
                    if not chunk:
                        return
                    yield tuple(chunk)

            chunk_iter = iter(chunks())

            def run_chunk(chunk):
                return tuple(function(task) for task in chunk)

            for _ in range(self.parallel_workers * 2):
                try:
                    pending.append(pool.submit(run_chunk, next(chunk_iter)))
                except StopIteration:
                    break
            try:
                while pending:
                    future = pending.popleft()
                    for result in future.result():
                        yield result
                    try:
                        pending.append(pool.submit(run_chunk, next(chunk_iter)))
                    except StopIteration:
                        pass
            finally:
                for future in pending:
                    if future.cancel():
                        continue
                    try:
                        future.result()
                    except Exception:
                        pass

        return results()

    @staticmethod
    def _gpu_einsum(subscripts, *operands):
        """Evaluate a small contraction on CuPy and return a NumPy array."""
        cupy = import_cupy()
        cupy_operands = [cupy.asarray(value) for value in operands]
        return cupy.asnumpy(cupy.einsum(subscripts, *cupy_operands))

    @staticmethod
    def _gaussian_radii(mol, precision, pad):
        """Gaussian shell radii in Bohr."""

        radii = np.empty(mol.nbas, dtype=float)
        for ish in range(mol.nbas):
            exps = np.asarray(mol.bas_exp(ish), dtype=float)
            coeffs = np.asarray(mol.bas_ctr_coeff(ish), dtype=float)
            alpha = float(np.min(exps)) if exps.size else 0.0
            cmax = float(np.max(np.abs(coeffs))) if coeffs.size else 0.0
            if alpha <= 0.0 or cmax <= 0.0:
                radii[ish] = float(pad)
                continue
            angular = float(mol.bas_angular(ish))
            radicand = (np.log(cmax / precision) + 0.5 * angular) / alpha
            radii[ish] = float(np.sqrt(max(radicand, 0.0)) + pad)
        return radii

    @staticmethod
    def _max_distance(coords):
        worst = 0.0
        for i in range(len(coords)):
            delta = coords[i + 1:] - coords[i]
            if delta.size == 0:
                continue
            worst = max(worst, float(np.max(np.linalg.norm(delta, axis=1))))
        return worst

    @staticmethod
    def _pairs_within_cutoff(coords, radii):
        """Upper-triangle pairs, stored as (i, j) with i >= j, from a cell list."""

        count = len(coords)
        if count == 0:
            return []
        reach = float(np.max(radii) + np.max(radii))
        if reach <= 0.0:
            reach = 1e-8
        buckets = defaultdict(list)
        for index, point in enumerate(coords):
            key = tuple(np.floor(point / reach).astype(int))
            buckets[key].append(index)
        pairs = []
        for i in range(count):
            base = np.floor(coords[i] / reach).astype(int)
            ri = float(radii[i])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (int(base[0]) + dx, int(base[1]) + dy, int(base[2]) + dz)
                        for j in buckets.get(key, ()):
                            if j > i:
                                continue
                            dist = float(np.linalg.norm(coords[i] - coords[j]))
                            if dist <= ri + float(radii[j]) + 1e-8:
                                pairs.append((i, j))
        return pairs

    def _shell_coords_of(self, mol):
        atoms = np.asarray(mol._bas[:, gto.ATOM_OF], dtype=int)
        return np.asarray(mol.atom_coords(), dtype=float)[atoms]

    def _prepare_shell_neighbors(self):
        """Neighbor shells inside the coverage radius, or a bounded local radius.

        A molecule that fits inside the coverage radii keeps every shell pair,
        so small systems and a 16-water cluster stay on the existing integrals.
        Longer molecules use the bare Gaussian radius and a cell list.
        """

        mol = self.mol
        nbas = self.nshells
        total_pairs = nbas * (nbas + 1) // 2
        self.shell_coords = self._shell_coords_of(mol)
        coverage = self._gaussian_radii(
            mol, _SHELL_RADIUS_PRECISION, _SHELL_RADIUS_PAD)
        local = self._gaussian_radii(mol, _LOCAL_RADIUS_PRECISION, 0.0)
        distance = self._max_distance(np.asarray(mol.atom_coords(), dtype=float))
        rmin = float(np.min(coverage)) if nbas else 0.0
        rmax = float(np.max(coverage)) if nbas else 0.0
        if nbas == 0 or distance <= rmin + rmin:
            pairs = [(i, j) for i in range(nbas) for j in range(i + 1)]
            complete = True
            radii = coverage
        elif distance > rmax + rmax:
            pairs = self._pairs_within_cutoff(self.shell_coords, local)
            complete = False
            radii = local
        else:
            covered = self._pairs_within_cutoff(self.shell_coords, coverage)
            complete = len(covered) == total_pairs
            if complete:
                pairs = covered
                radii = coverage
            else:
                pairs = self._pairs_within_cutoff(self.shell_coords, local)
                radii = local
        self.shell_radii = radii
        self.neighbor_pairs = pairs
        self.coverage_complete = bool(complete and len(pairs) == total_pairs)
        self.neighbors_complete = self.coverage_complete
        buckets = [[] for _ in range(nbas)]
        for ish, jsh in pairs:
            buckets[ish].append(jsh)
            if ish != jsh:
                buckets[jsh].append(ish)
        self.shell_neighbors = [np.asarray(item, dtype=np.int32) for item in buckets]
        reach = float(np.max(radii)) if nbas else 1e-8
        if reach <= 0.0:
            reach = 1e-8
        query_buckets = defaultdict(list)
        for index, point in enumerate(self.shell_coords):
            key = tuple(np.floor(point / reach).astype(int))
            query_buckets[key].append(index)
        self._query_reach = reach
        self._query_buckets = query_buckets

    @staticmethod
    def _shell_runs(shells, limit):
        """Contiguous shell ranges. ``limit`` caps the number of shells."""

        runs = []
        if len(shells) == 0:
            return runs
        start = int(shells[0])
        previous = start
        for shell in shells[1:]:
            shell = int(shell)
            if shell == previous + 1 and (limit <= 0 or shell - start < limit):
                previous = shell
                continue
            runs.append((start, previous + 1))
            start = previous = shell
        runs.append((start, previous + 1))
        return runs

    @staticmethod
    def _spatial_grid_batches(coords, batch_size, bin_size):
        """Grid-point batches that stay inside one spatial bin."""

        count = len(coords)
        if count == 0:
            return []
        keys = np.floor(np.asarray(coords, dtype=float) / bin_size).astype(int)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        batches = []
        start = 0
        for index in range(1, count + 1):
            if index < count and np.array_equal(keys[order[index]], keys[order[start]]):
                continue
            group = order[start:index]
            for offset in range(0, len(group), batch_size):
                batches.append(np.asarray(group[offset:offset + batch_size], dtype=int))
            start = index
        return batches

    def _shells_near_points(self, points):
        """Shells that can be nonzero anywhere in a grid batch."""

        if len(points) == 0 or self.nshells == 0:
            return np.zeros(0, dtype=int)
        center = np.mean(points, axis=0)
        extent = float(np.max(np.linalg.norm(points - center, axis=1)))
        reach = self._query_reach
        base = np.floor(center / reach).astype(int)
        span = int(np.ceil((extent + reach) / reach)) + 1
        found = []
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for dz in range(-span, span + 1):
                    key = (int(base[0]) + dx, int(base[1]) + dy, int(base[2]) + dz)
                    for shell in self._query_buckets.get(key, ()):
                        dist = float(np.linalg.norm(self.shell_coords[shell] - center))
                        if dist <= extent + float(self.shell_radii[shell]) + 1e-8:
                            found.append(shell)
        if not found:
            return np.zeros(0, dtype=int)
        return np.asarray(sorted(set(found)), dtype=int)

    def _absorb_selected_schwarz(self, eri, i0, j0, pairs):
        ao_loc = self.ao_loc
        base_i = int(ao_loc[i0])
        base_j = int(ao_loc[j0])
        for ish, jsh in pairs:
            i_a = int(ao_loc[ish]) - base_i
            i_b = int(ao_loc[ish + 1]) - base_i
            j_a = int(ao_loc[jsh]) - base_j
            j_b = int(ao_loc[jsh + 1]) - base_j
            block = eri[i_a:i_b, j_a:j_b, i_a:i_b, j_a:j_b]
            self.schwarz_pairs_evaluated += 1
            if block.size == 0:
                continue
            value = float(np.sqrt(np.max(np.abs(block))))
            self.schwarz_ints[ish, jsh] = value
            self.schwarz_ints[jsh, ish] = value

    def _build_schwarz_ints(self):
        """Diagonal shell-pair bounds from blocked (ij|ij) integrals.

        Only neighbor pairs are visited. A molecule inside the coverage radius
        still evaluates every pair.
        """

        nbas = self.nshells
        self.schwarz_ints = np.zeros((nbas, nbas))
        self.schwarz_pairs_evaluated = 0
        total_pairs = nbas * (nbas + 1) // 2
        if total_pairs == 0 or not self.neighbor_pairs:
            self.significant_shell_pairs = 0
            self.significant_pair_fraction = 0.0
            return
        intor = self.mol._add_suffix('int2e')
        atm = np.asarray(self.mol._atm, dtype=np.int32, order='C')
        bas = np.asarray(self.mol._bas, dtype=np.int32, order='C')
        env = np.asarray(self.mol._env, dtype=np.float64, order='C')
        self._schwarz_cintopt = moleintor.make_cintopt(atm, bas, env, intor)
        block = _SCHWARZ_SHELL_BLOCK
        groups = {}
        for ish, jsh in self.neighbor_pairs:
            groups.setdefault((ish // block, jsh // block), []).append((ish, jsh))
        for (ib, jb), pairs in groups.items():
            i0 = ib * block
            j0 = jb * block
            i1 = min(nbas, i0 + block)
            j1 = min(nbas, j0 + block)
            eri = moleintor.getints(
                intor, atm, bas, env,
                (i0, i1, j0, j1, i0, i1, j0, j1),
                aosym='s1', cintopt=self._schwarz_cintopt,
            )
            self._absorb_selected_schwarz(eri, i0, j0, pairs)
        positive = int(np.count_nonzero(np.triu(self.schwarz_ints)))
        self.significant_shell_pairs = positive
        self.significant_pair_fraction = positive / total_pairs
    
    def _schwarz_screen(self, ish, jsh, ksh, lsh):
        schwarz_est = (self.schwarz_ints[ish, jsh] * self.schwarz_ints[ksh, lsh])
        return schwarz_est > self.schwarz_thresh

    def build_jk(self, dm):
        dm = np.asarray(dm)
        if dm.shape != (self.nao, self.nao):
            raise ValueError("dm must have shape (nao, nao)")
        def iter_tasks():
            # Visit one representative of each eight-fold ERI symmetry orbit,
            # then contract each distinct shell-index permutation exactly once.
            for ish in range(self.nshells):
                for jsh in range(ish + 1):
                    pair_ij = ish * (ish + 1) // 2 + jsh
                    for ksh in range(self.nshells):
                        for lsh in range(ksh + 1):
                            pair_kl = ksh * (ksh + 1) // 2 + lsh
                            if pair_ij < pair_kl:
                                continue

                            if not self._schwarz_screen(ish, jsh, ksh, lsh):
                                continue

                            orbit = (
                                ((ish, jsh, ksh, lsh), (0, 1, 2, 3)),
                                ((jsh, ish, ksh, lsh), (1, 0, 2, 3)),
                                ((ish, jsh, lsh, ksh), (0, 1, 3, 2)),
                                ((jsh, ish, lsh, ksh), (1, 0, 3, 2)),
                                ((ksh, lsh, ish, jsh), (2, 3, 0, 1)),
                                ((lsh, ksh, ish, jsh), (3, 2, 0, 1)),
                                ((ksh, lsh, jsh, ish), (2, 3, 1, 0)),
                                ((lsh, ksh, jsh, ish), (3, 2, 1, 0)),
                            )
                            distinct = {}
                            for shell_tuple, axes in orbit:
                                distinct.setdefault(shell_tuple, axes)
                            if not any(
                                max(np.max(np.abs(dm[self.ao_loc[b]:self.ao_loc[b + 1],
                                                      self.ao_loc[d]:self.ao_loc[d + 1]])),
                                    np.max(np.abs(dm[self.ao_loc[c]:self.ao_loc[c + 1],
                                                      self.ao_loc[d]:self.ao_loc[d + 1]])))
                                > self.density_thresh
                                for a, b, c, d in distinct
                            ):
                                continue

                            yield (ish, jsh, ksh, lsh, tuple(distinct.items()))

        def contract(task):
            ish, jsh, ksh, lsh, distinct_items = task
            i0, i1 = self.ao_loc[ish:ish + 2]
            j0, j1 = self.ao_loc[jsh:jsh + 2]
            k0, k1 = self.ao_loc[ksh:ksh + 2]
            l0, l1 = self.ao_loc[lsh:lsh + 2]
            eri = self.mol.intor(
                'int2e_sph',
                shls_slice=(ish, ish + 1, jsh, jsh + 1,
                            ksh, ksh + 1, lsh, lsh + 1),
            ).reshape(i1-i0, j1-j0, k1-k0, l1-l0)
            contributions = []
            for (a, b, c, d), axes in distinct_items:
                ia, ja = self.ao_loc[a:a + 2]
                ib, jb = self.ao_loc[b:b + 2]
                ic, jc = self.ao_loc[c:c + 2]
                id_, jd = self.ao_loc[d:d + 2]
                block = eri.transpose(axes)
                dm_j = dm[ic:jc, id_:jd]
                dm_k = dm[ib:jb, id_:jd]
                if self.parallel_backend == "gpu":
                    j_block = self._gpu_einsum('pqrs,rs->pq', block, dm_j)
                    k_block = self._gpu_einsum('pqrs,qs->pr', block, dm_k)
                else:
                    j_block = np.einsum('pqrs,rs->pq', block, dm_j)
                    k_block = np.einsum('pqrs,qs->pr', block, dm_k)
                contributions.append((ia, ja, ib, jb, ic, jc, j_block, k_block))
            return contributions

        vj = np.zeros((self.nao, self.nao))
        vk = np.zeros((self.nao, self.nao))
        for contributions in self._parallel_map(contract, iter_tasks()):
            for ia, ja, ib, jb, ic, jc, j_block, k_block in contributions:
                vj[ia:ja, ib:jb] += j_block
                vk[ia:ja, ic:jc] += k_block

        vj = (vj + vj.T) / 2
        vk = (vk + vk.T) / 2

        return vj, vk

    def build_fock_incremental(self, fock_prev, delta_dm):
        """Update a physical Fock matrix using the density difference."""
        fock_prev = np.asarray(fock_prev)
        delta_dm = np.asarray(delta_dm)
        expected_shape = (self.nao, self.nao)
        if fock_prev.shape != expected_shape:
            raise ValueError("fock_prev must have shape (nao, nao)")
        if delta_dm.shape != expected_shape:
            raise ValueError("delta_dm must have shape (nao, nao)")

        delta_j, delta_k = self.build_jk(delta_dm)
        return fock_prev + delta_j - 0.5 * delta_k






    def build_ri_j(self, dm, auxbasis, df_tol=1e-8):
        """Build globally consistent RI-J without changing the engine by size."""

        dm = np.asarray(dm)
        if dm.shape != (self.nao, self.nao):
            raise ValueError("dm must have shape (nao, nao)")
        if auxbasis is None:
            raise ValueError("RI-J requires an explicit auxbasis")
        if df_tol <= 0:
            raise ValueError("df_tol must be positive")
        if self.jk_engine == "pyscf_reference":
            result = self.build_df_j(dm, auxbasis)
            self.last_j_stats = {"engine": "pyscf_reference", "device": "cpu"}
            return result
        engine = self._hierarchical_j_engine(auxbasis)
        result = engine.build(dm)
        self.last_j_stats = dict(engine.stats)
        self.last_j_stats["engine"] = self.jk_engine
        return result

    def _hierarchical_j_engine(self, auxbasis):
        from hierarchical_ri import HierarchicalRIJ

        key = str(auxbasis)
        if key not in self._hierarchical_j_engines:
            self._hierarchical_j_engines[key] = HierarchicalRIJ(
                self.mol, auxbasis=auxbasis, backend=self.parallel_backend,
                max_memory_mb=self.max_memory, options=self.hierarchical_options,
            )
        return self._hierarchical_j_engines[key]








    def _cosx_grid(self, grid_rad, grid_ang):
        grids = gen_grid.Grids(self.mol)
        grids.atom_grid = {self.mol.atom_pure_symbol(i): (grid_rad, grid_ang)
                           for i in range(self.mol.natm)}
        grids.radi_method = radi.treutler_ahlrichs
        grids.prune = gen_grid.nwchem_prune
        grids.build()
        return grids.coords, grids.weights

    def build_cosx_k(self, dm, grid_rad=30, grid_ang=110, batch_size=512,
                     cosx_tol=1e-10, ao_tol=1e-10, overlap_fit=True,
                     screen_dm=None, auxbasis="weigend"):
        """COSX with a consistent cross-atom overlap projection."""
        if self.jk_engine == "pyscf_reference":
            result = self.build_sgx_k(
                dm, auxbasis, grid_rad, grid_ang, overlap_fit,
                direct_scf_tol=cosx_tol, screen_dm=screen_dm,
            )
            self.last_k_stats = {"engine": "pyscf_reference", "device": "cpu"}
            return result
        engine = self._local_cosx_engine(grid_rad, grid_ang, batch_size, overlap_fit)
        result = engine.build(dm, screen_dm=screen_dm,
                              direct_scf_tol=cosx_tol, ao_tol=ao_tol)
        self.last_k_stats = dict(engine.stats)
        self.last_k_stats["engine"] = self.jk_engine
        self._last_k_grid = (engine.ngrids, engine.blksize)
        return result

    def _local_cosx_engine(self, grid_rad, grid_ang, batch_size, overlap_fit):
        from local_cosx import LocalCOSX

        key = (int(grid_rad), int(grid_ang), int(batch_size), bool(overlap_fit))
        if key not in self._local_k_engines:
            self._local_k_engines[key] = LocalCOSX(
                self.mol, grid_rad=grid_rad, grid_ang=grid_ang,
                batch_size=batch_size, backend=self.parallel_backend,
                overlap_fit=overlap_fit, max_memory_mb=self.max_memory,
            )
        return self._local_k_engines[key]




    def _df_engine(self, auxbasis):
        """One global density-fitting object per auxiliary basis.

        Coulomb screening state stays on this object for the whole SCF.
        The COSX grid is not part of the key: J does not use it.
        """

        key = str(auxbasis)
        engine = self._df_engines.get(key)
        if engine is None:
            engine = df.DF(self.mol, auxbasis=auxbasis)
            engine.verbose = 0
            engine.max_memory = float(self.max_memory)
            self._df_engines[key] = engine
        return engine

    def _sgx_engine(self, auxbasis, grid_rad, grid_ang, overlap_fit):
        """One SGX object per auxiliary basis, grid, and overlap fit.

        The grid, overlap-fit matrix, and P-junction bounds are built on the
        first K evaluation and then reused.
        """

        key = (str(auxbasis), int(grid_rad), int(grid_ang), bool(overlap_fit))
        engine = self._sgx_engines.get(key)
        if engine is None:
            from pyscf.sgx import sgx as sgx_mod

            engine = sgx_mod.SGX(self.mol, auxbasis=auxbasis)
            engine.verbose = 0
            engine.max_memory = float(self.max_memory)
            engine.dfj = True
            engine.optk = True
            engine.fit_ovlp = bool(overlap_fit)
            engine.grids_level_i = 1
            engine.grids_level_f = 1
            engine.grids = matching_cosx_grids(self.mol, grid_rad, grid_ang)
            engine._opt = sgx_mod._make_opt(self.mol)
            engine._pjs_data = None
            _pin_sgx_blocks(engine)
            self._sgx_engines[key] = engine
        return engine

    def release_grid(self, auxbasis, grid_rad, grid_ang, overlap_fit):
        """Free the K engines, grids and screening data of one grid."""

        self._sgx_engines.pop(
            (str(auxbasis), int(grid_rad), int(grid_ang), bool(overlap_fit)), None)
        for key in [key for key in self._local_k_engines
                    if key[:2] == (int(grid_rad), int(grid_ang))]:
            self._local_k_engines.pop(key)
        # The block-size override closes over its engine, so the engine is
        # only reclaimed by the cycle collector.
        gc.collect()

    def _j_reserve_mb(self):
        """Memory DF-J may hold while K runs: one AO matrix per thread, plus one."""

        from pyscf import lib

        return (lib.num_threads() + 1) * self.nao * self.nao * 8 / 1e6

    def _sgx_block_size(self, engine):
        """Grid count and block size that ``sgx_jk.get_k_only`` will use.

        Without ``k_max_memory_mb`` this copies ``SGXData.get_loop_data`` for
        one density matrix: the block is max_memory minus the resident set
        the kernel will see. With ``k_max_memory_mb`` the block comes from
        that budget alone, capped so RSS + J reserve + K block stays under
        ``memory_ceiling_mb``, and is pinned for the next ``get_k_only``.
        """

        from pyscf import lib
        from pyscf.sgx import sgx_jk

        ngrids = int(engine.grids.coords.shape[0])
        rss_mb = float(lib.current_memory()[0])
        if self.k_max_memory_mb is None:
            available = float(engine.max_memory) - rss_mb
            self._last_k_budget_mb = None
        else:
            available = self.k_max_memory_mb
            if self.memory_ceiling_mb is not None:
                available = min(
                    available, self.memory_ceiling_mb - rss_mb - self._j_reserve_mb())
            self._last_k_budget_mb = available
            # A screening-data rebuild inside this call sizes its own AO
            # loops from max_memory minus RSS.
            engine.max_memory = rss_mb + max(available, 0.0)
        data_dim = 4
        raw = int(available * 1e6 / 8 / (data_dim * self.nao))
        blksize = max(int(sgx_jk.SGX_BLKSIZE), raw)
        aligned = max(1, blksize // int(sgx_jk.SGX_BLKSIZE)) * int(sgx_jk.SGX_BLKSIZE)
        blksize = min(ngrids, aligned)
        engine._hf_blksize = None if self.k_max_memory_mb is None else blksize
        return ngrids, blksize

    def _k_grid_note(self):
        ngrids, blksize = self._last_k_grid
        note = f"ngrids={ngrids} blksize={blksize}"
        if self._last_k_budget_mb is not None:
            note += f" k_budget_mb={self._last_k_budget_mb:.0f}"
        return note

    def _memory_note(self):
        if self.k_max_memory_mb is None:
            return "k_memory_mb=max_memory-minus-rss"
        ceiling = ("none" if self.memory_ceiling_mb is None
                   else f"{self.memory_ceiling_mb:.0f}")
        return f"k_memory_mb={self.k_max_memory_mb:.0f} memory_ceiling_mb={ceiling}"

    def build_df_j(self, dm, auxbasis, direct_scf_tol=1e-13):
        """Global RI-J through PySCF's compiled density-fitting kernel."""

        from pyscf.df import df_jk

        dm = np.asarray(dm, dtype=np.float64, order="C")
        if dm.shape != (self.nao, self.nao):
            raise ValueError("dm must have shape (nao, nao)")
        if auxbasis is None:
            raise ValueError("RI-J requires an explicit auxbasis")
        vj = df_jk.get_j(
            self._df_engine(auxbasis), dm, hermi=1,
            direct_scf_tol=direct_scf_tol,
        )
        return np.asarray(vj, dtype=np.float64)

    def build_sgx_k(self, dm, auxbasis, grid_rad, grid_ang, overlap_fit=True,
                    direct_scf_tol=1e-13, screen_dm=None):
        """Exchange through PySCF's compiled SGX kernel.

        ``screen_dm`` is the full density used only to decide which shell
        pairs survive. Pass it with a density difference so an incremental K
        uses the same pairs as a full K.
        """

        from pyscf.sgx import sgx_jk

        dm = np.asarray(dm, dtype=np.float64, order="C")
        if dm.shape != (self.nao, self.nao):
            raise ValueError("dm must have shape (nao, nao)")
        if auxbasis is None:
            raise ValueError("SGX exchange requires an explicit auxbasis")
        if grid_rad <= 0 or grid_ang <= 0:
            raise ValueError("grid sizes must be positive")
        engine = self._sgx_engine(auxbasis, grid_rad, grid_ang, overlap_fit)
        self._last_k_grid = self._sgx_block_size(engine)
        if screen_dm is None:
            engine._full_dm = None
        else:
            screen_dm = np.asarray(screen_dm, dtype=np.float64, order="C")
            if screen_dm.shape != dm.shape:
                raise ValueError("screen_dm must have the same shape as dm")
            engine._full_dm = screen_dm
        try:
            vk = sgx_jk.get_k_only(
                engine, dm, hermi=1, direct_scf_tol=direct_scf_tol,
            )
        finally:
            engine._full_dm = None
        return np.asarray(vk, dtype=np.float64)

    def get_jk(
        self,
        dm,
        jk_mode="rijcosx",
        auxbasis="weigend",
        df_tol=1e-9,
        grid_rad=30,
        grid_ang=110,
        batch_size=256,
        cosx_tol=1e-10,
        ao_tol=1e-10,
        overlap_fit=True,
        timings=None,
    ):
        """Return Coulomb and exchange matrices for one density.

        ``direct`` is the analytic small-system path. ``rijcosx`` is the
        large-system path and does not run the SCF loop. When ``timings``
        is a dict and ``jk_mode`` is ``rijcosx``, that dict receives
        ``j_seconds`` and ``k_seconds`` for the two kernels.
        """

        if jk_mode == "direct":
            return self.build_jk(dm)
        if jk_mode != "rijcosx":
            raise ValueError("jk_mode must be 'direct' or 'rijcosx'")
        if auxbasis is None:
            raise ValueError("rijcosx mode requires an explicit auxbasis")
        started = time.perf_counter()
        vj = self.build_ri_j(dm, auxbasis, df_tol)
        j_seconds = time.perf_counter() - started
        started = time.perf_counter()
        vk = self.build_cosx_k(
            dm, grid_rad, grid_ang, batch_size, cosx_tol, ao_tol, overlap_fit,
            auxbasis=auxbasis)
        k_seconds = time.perf_counter() - started
        if timings is not None:
            timings["j_seconds"] = j_seconds
            timings["k_seconds"] = k_seconds
        return vj, vk

    def iteration_rijcosx(
        self,
        dm,
        hcore,
        overlap,
        nocc,
        nelectron,
        auxbasis="weigend",
        df_tol=1e-9,
        grid_rad=30,
        grid_ang=110,
        batch_size=256,
        cosx_tol=1e-10,
        ao_tol=1e-10,
        overlap_fit=True,
    ):
        """Build one RIJCOSX Fock, diagonalize it, and rebuild the density.

        J and K come from ``get_jk`` and are timed separately. The
        diagonalization uses the configured dense backend, as in
        ``solve_rhf``, timed by itself. This method does not enter that
        loop. The electronic energy belongs to the input density and this
        Fock. The returned density belongs to the occupied eigenvectors.
        """

        density_in = np.array(dm, dtype=np.float64, copy=True)
        if density_in.shape != hcore.shape or hcore.shape != overlap.shape:
            raise ValueError("dm, hcore, and overlap must share one square shape")
        if density_in.ndim != 2 or density_in.shape[0] != density_in.shape[1]:
            raise ValueError("dm must be a square matrix")
        nocc = int(nocc)
        nao = density_in.shape[0]
        if nocc < 0 or nocc > nao:
            raise ValueError("nocc must lie between 0 and nao")

        timings = {}
        j_matrix, k_matrix = self.get_jk(
            density_in,
            jk_mode="rijcosx",
            auxbasis=auxbasis,
            df_tol=df_tol,
            grid_rad=grid_rad,
            grid_ang=grid_ang,
            batch_size=batch_size,
            cosx_tol=cosx_tol,
            ao_tol=ao_tol,
            overlap_fit=overlap_fit,
            timings=timings,
        )
        j_matrix = np.asarray(j_matrix, dtype=np.float64)
        k_matrix = np.asarray(k_matrix, dtype=np.float64)
        fock = hcore + j_matrix - 0.5 * k_matrix
        dense = DenseLinearAlgebra(
            overlap, backend=self.dense_backend, torch_device=self.torch_device,
            check_calls=self.dense_check_calls)
        started = time.perf_counter()
        orbital_energy, coeff, density_out = dense.solve(fock, nocc)
        eigh_seconds = time.perf_counter() - started
        electronic = 0.5 * np.einsum("mn,mn->", hcore + fock, density_in)
        electrons = float(np.einsum(
            "ij,ji->", np.asarray(overlap), np.asarray(density_out)))
        finite = bool(
            np.isfinite(j_matrix).all()
            and np.isfinite(k_matrix).all()
            and np.isfinite(fock).all()
            and np.isfinite(orbital_energy).all()
            and np.isfinite(density_out).all()
            and np.isfinite(electronic)
        )

        def asymmetry(matrix):
            matrix = np.asarray(matrix)
            return float(np.max(np.abs(matrix - matrix.T)))

        return {
            "j_matrix": j_matrix,
            "k_matrix": k_matrix,
            "density": density_out,
            "j_seconds": float(timings["j_seconds"]),
            "k_seconds": float(timings["k_seconds"]),
            "eigh_seconds": float(eigh_seconds),
            "dense_backend": dense.backend,
            "dense_device": dense.device,
            "eigensolver": dense.eigensolver,
            "j_asymmetry": asymmetry(j_matrix),
            "k_asymmetry": asymmetry(k_matrix),
            "density_asymmetry": asymmetry(density_out),
            "density_electron_abs": abs(electrons - float(nelectron)),
            "electronic_energy": float(electronic),
            "all_finite": finite,
            "shell_pairs": int(self.last_shell_pairs),
            "screened_pairs": int(self.last_ri_screened_pairs),
            "cosx_full_tensor_calls": int(self.cosx_full_tensor_calls),
            "coverage_complete": bool(self.coverage_complete),
        }

    def _gpu_grid_batch(self, requested):
        """Keep one COSX batch inside half of a 72 GB GPU.

        Four ``(batch, nao)`` arrays are live during the contraction. A
        chain of 10000 functions must not upload the whole grid at once.
        """

        requested = int(requested)
        if self.parallel_backend != "gpu":
            return requested
        budget = int(72 * (1024 ** 3) * 0.5)
        per_grid = 4 * max(1, int(self.nao)) * 8
        return max(1, min(requested, budget // per_grid))

    def _dense_product(self, left, right):
        """Matrix product. The GPU backend uses one CuPy GEMM."""

        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if (self.parallel_backend != "gpu" or left.size == 0 or right.size == 0):
            return left @ right
        cupy = import_cupy()
        return cupy.asnumpy(cupy.asarray(left) @ cupy.asarray(right))

    def _prepare_exchange_builders(self, auxbasis, grid_rad, grid_ang, overlap_fit):
        """Build J and K caches on the calling thread before overlapping them."""

        if self.jk_engine == "hierarchical_rijcosx":
            self._hierarchical_j_engine(auxbasis).prepare()
            return
        self._df_engine(auxbasis)
        self._sgx_engine(auxbasis, grid_rad, grid_ang, overlap_fit)

    def _coulomb_matrix(self, dm, auxbasis, df_tol):
        """Use the selected mathematical engine on every SCF entry point."""
        return self.build_ri_j(dm, auxbasis, df_tol)

    def _exchange_matrix(self, dm, auxbasis, grid_rad, grid_ang, batch_size,
                         overlap_fit, ao_tol, direct_scf_tol, screen_dm=None):
        """Use the same projection and screening in full/incremental builds."""
        return self.build_cosx_k(
            dm, grid_rad, grid_ang, batch_size,
            cosx_tol=direct_scf_tol, ao_tol=ao_tol, overlap_fit=overlap_fit,
            screen_dm=screen_dm, auxbasis=auxbasis,
        )

    def solve_rhf(
        self,
        nelecs,
        hcore: np.ndarray,
        overlap: np.ndarray,
        mol,
        ene_nuc: float = 0.0,
        max_iteration: int = 200,
        tol: float = 1e-8,
        dm_init = None,
        incremental_rebuild_interval: int = 10,
        jk_mode="direct",
        auxbasis=None,
        df_tol=1e-8,
        cosx_tol=1e-10,
        ao_tol=1e-10,
        grid_rad=30,
        grid_ang=110,
        final_grid_rad=35,
        final_grid_ang=302,
        batch_size=512,
        overlap_fit=True,
        incremental_k=False,
        kfull_every=5,
        progress_log=None,
        molecule_name=None,
        jk_engine=None,
        grid_switch_energy_tol=None,
    ):
        """Closed-shell SCF.

        In ``rijcosx`` mode the SCF converges on ``grid`` and then
        re-converges on ``final_grid`` with the same thresholds, followed by
        a full J/K rebuild. With ``grid_switch_energy_tol`` the coarse grid
        stops as soon as its energy change falls below that value instead of
        converging to ``tol`` first.
        """
        import time
        self.last_scf_result = None

        if jk_engine is not None:
            if jk_engine not in ("hierarchical_rijcosx", "pyscf_reference"):
                raise ValueError("unknown jk_engine")
            self.jk_engine = jk_engine

        if jk_mode not in ("direct", "rijcosx"):
            raise ValueError("jk_mode must be 'direct' or 'rijcosx'")
        if jk_mode == "rijcosx" and auxbasis is None:
            raise ValueError("rijcosx mode requires an explicit auxbasis")
        if kfull_every <= 0:
            raise ValueError("kfull_every must be positive")
        if jk_mode == "rijcosx" and min(
                grid_rad, grid_ang, final_grid_rad, final_grid_ang, batch_size) <= 0:
            raise ValueError("grid sizes and batch_size must be positive")
        if jk_mode == "rijcosx" and min(df_tol, cosx_tol, ao_tol) <= 0:
            raise ValueError("df_tol, cosx_tol, and ao_tol must be positive")
        if grid_switch_energy_tol is not None:
            grid_switch_energy_tol = float(grid_switch_energy_tol)
            if not grid_switch_energy_tol > 0:
                raise ValueError("grid_switch_energy_tol must be positive")
        early_switch = (jk_mode == "rijcosx" and grid_switch_energy_tol is not None
                        and (grid_rad, grid_ang) != (final_grid_rad, final_grid_ang))
        if jk_mode == "rijcosx":
            _emit_progress(
                f"RIJCOSX: jk_engine={self.jk_engine}, "
                f"requested_device={self.parallel_backend}, auxbasis={auxbasis}, "
                f"direct_scf_tol=1e-13, "
                f"grid={grid_rad}x{grid_ang}, final_grid="
                f"{final_grid_rad}x{final_grid_ang}, incremental_k={incremental_k}, "
                f"kfull_every={kfull_every}, max_memory_mb={float(self.max_memory):.0f}, "
                f"{self._memory_note()}, dense_backend={self.dense_backend}, "
                f"grid_switch_energy_tol={grid_switch_energy_tol}",
                progress_log,
                molecule_name,
            )
        if (isinstance(incremental_rebuild_interval, bool)
                or not isinstance(incremental_rebuild_interval, (int, np.integer))
                or incremental_rebuild_interval <= 0):
            raise ValueError("incremental_rebuild_interval must be a positive integer")

        nelecs_alph, nelecs_beta = nelecs
        if nelecs_alph != nelecs_beta:
            raise ValueError("solve_rhf supports closed-shell systems only")
        nelectron = nelecs_alph + nelecs_beta
        nao = hcore.shape[0]

        if hcore.shape != (nao, nao) or overlap.shape != (nao, nao):
            raise ValueError("hcore and overlap must be square AO matrices of equal shape")
        if nelecs_alph < 0 or nelecs_alph > nao:
            raise ValueError("the alpha electron count must be between 0 and nao")

        if dm_init is None:
            dm_old = scf.hf.init_guess_by_minao(mol)
        else:
            dm_old = np.asarray(dm_init)
            if dm_old.shape != (nao, nao):
                raise ValueError("dm_init must have shape (nao, nao)")
            dm_old = dm_old.copy()

        iteration_scf = 0
        is_converged = False
        is_max_iteration = False

        energy_err = 1.0
        dm_err = 1.0
        

        ene_rhf = None
        ene_old = None
        ene_cur = None
        dm_cur = None
        switch_ready = False

        dense = DenseLinearAlgebra(
            overlap, backend=self.dense_backend, max_diis=8,
            check_calls=self.dense_check_calls, torch_device=self.torch_device)
        _emit_progress(
            f"SCF dense linear algebra backend={dense.backend} device={dense.device}",
            progress_log, molecule_name)

        def parallel_coulomb_exchange(density, k_density, screen_tol, screen_dm, k_base):
            """Run J and K together. DIIS uses the Fock only after both return."""

            def coulomb():
                started_j = time.perf_counter()
                rss_j = _rss_gib()
                matrix = self._coulomb_matrix(density, auxbasis, df_tol)
                return matrix, _stage_record("j", started_j, rss_j)

            def exchange():
                started_k = time.perf_counter()
                rss_k = _rss_gib()
                matrix = self._exchange_matrix(
                    k_density, auxbasis, grid_rad, grid_ang, batch_size,
                    overlap_fit, ao_tol, screen_tol, screen_dm,
                )
                if k_base is not None:
                    matrix = k_base + matrix
                return matrix, _stage_record("k", started_k, rss_k)

            with ThreadPoolExecutor(max_workers=2) as pool:
                coulomb_future = pool.submit(coulomb)
                exchange_future = pool.submit(exchange)
                j_matrix, j_record = coulomb_future.result()
                k_matrix, k_record = exchange_future.result()
            return j_matrix, k_matrix, j_record, k_record

        def gradient_rms(fock_ao):
            return orbital_gradient_rms(
                scf.hf.get_grad(C_occ, mo_occ, fock_ao),
                nelecs_alph,
                nao - nelecs_alph,
            )

        _emit_progress("SCF initial Fock", progress_log, molecule_name)
        if jk_mode == "direct":
            started = time.perf_counter()
            rss_before = _rss_gib()
            vj_old, vk_old = self.build_jk(dm_old)
            jk_record = _stage_record("jk", started, rss_before)
            fock_physical = hcore + vj_old - 0.5 * vk_old
            k_physical = None
            _emit_progress(
                f"SCF initial Fock {jk_record}",
                progress_log,
                molecule_name,
            )
        else:
            self._prepare_exchange_builders(
                auxbasis, grid_rad, grid_ang, overlap_fit)
            j_physical, k_physical, j_record, k_record = parallel_coulomb_exchange(
                dm_old, dm_old, 1e-13, None, None)
            fock_physical = hcore + j_physical - 0.5 * k_physical
            _emit_progress(
                f"SCF initial Fock {j_record} {k_record} {self._k_grid_note()}",
                progress_log,
                molecule_name,
            )
            # A MINAO guess is not normalized, so its electron count is not checked.
            terms = scf_energy_terms(
                dm_old, hcore, j_physical, k_physical, overlap, ene_nuc)
            check_physical_energy(terms, "initial Fock")
            _emit_progress(
                "SCF initial diagnostics " + json.dumps({
                    "iteration": 0,
                    "grid": [grid_rad, grid_ang],
                    **terms,
                    "physical_guard": "passed",
                    "j": self.last_j_stats,
                    "k": self.last_k_stats,
                }, sort_keys=True, default=lambda value: value.item()),
                progress_log, molecule_name,
            )
        incremental_updates = 0
        # The commutator of (fock_physical, dm_old) formed at the end of an
        # iteration is the DIIS error of the next one.
        pending_error = None

        while not is_converged and not is_max_iteration:
            iteration_started = time.perf_counter()
            iteration_rss = _rss_gib()
            _emit_progress(
                f"SCF iteration {iteration_scf + 1}/{max_iteration}",
                progress_log,
                molecule_name,
            )
            # Keep the physical Fock (which matches dm_old) separate from
            # the DIIS extrapolation used only for diagonalization.
            started = time.perf_counter()
            rss_before = _rss_gib()
            if pending_error is None:
                err = dense.commutator(fock_physical, dm_old)
            else:
                err = pending_error
            pending_error = None
            err_norm = dense.norm(err)
            if err_norm > 1e-12:
                dense.push(err, fock_physical)
            fock_for_diagonalization = dense.extrapolate()
            if fock_for_diagonalization is None:
                fock_for_diagonalization = fock_physical
            diis_record = _stage_record("diis", started, rss_before)

            started = time.perf_counter()
            rss_before = _rss_gib()
            ene_mo, C_occ, dm_cur = dense.solve(fock_for_diagonalization, nelecs_alph)
            fock_for_diagonalization = None
            delta_dm = dm_cur - dm_old
            eigh_record = _stage_record("eigh", started, rss_before)
            if jk_mode == "direct":
                started = time.perf_counter()
                rss_before = _rss_gib()
                fock_cur = self.build_fock_incremental(fock_physical, delta_dm)
                fock_record = _stage_record("fock", started, rss_before)
                incremental_updates += 1
                jk_record = "jk_s=0.000 jk_rss_gib=0.000 jk_rss_increase_gib=0.000"
                k_build = "direct"
                if incremental_updates % incremental_rebuild_interval == 0:
                    started = time.perf_counter()
                    rss_before = _rss_gib()
                    vj_cur, vk_cur = self.build_jk(dm_cur)
                    fock_cur = hcore + vj_cur - 0.5 * vk_cur
                    jk_record = _stage_record("jk", started, rss_before)
                part_times = f"{fock_record} {jk_record}"
            else:
                kfull = (not incremental_k or iteration_scf == 0
                         or (iteration_scf + 1) % kfull_every == 0
                         or iteration_scf + 1 >= max_iteration)
                # A full K keeps the 1e-13 screen. An incremental K can drop
                # pairs below 1e-10; the density difference is already small.
                screen_tol = 1e-13 if kfull else 1e-10
                if kfull:
                    j_cur, k_cur, j_record, k_record = parallel_coulomb_exchange(
                        dm_cur, dm_cur, screen_tol, None, None)
                    k_build = "full"
                else:
                    j_cur, k_cur, j_record, k_record = parallel_coulomb_exchange(
                        dm_cur, delta_dm, screen_tol, dm_cur, k_physical)
                    k_build = "incremental"
                fock_cur = hcore + j_cur - 0.5 * k_cur
                j_physical, k_physical = j_cur, k_cur
                part_times = (
                    f"{j_record} {k_record} {self._k_grid_note()} k_build={k_build}"
                )

            ene_mo = ene_cur = 0.5 * np.einsum("mn,mn->", hcore + fock_cur, dm_cur)
            ene_rhf = ene_nuc + ene_mo
            if not (np.isfinite(ene_rhf) and np.isfinite(dm_cur).all()
                    and np.isfinite(fock_cur).all()):
                raise RuntimeError("SCF produced a non-finite density, Fock or energy")
            ene_cur = ene_rhf
            if ene_old is not None:
                energy_err = abs(ene_cur - ene_old)
            if jk_mode == "rijcosx":
                stage = f"SCF iteration {iteration_scf + 1}"
                terms = scf_energy_terms(
                    dm_cur, hcore, j_physical, k_physical, overlap, ene_nuc)
                check_physical_energy(terms, stage, nelectron)
                mo_occ = np.zeros(nao)
                mo_occ[:nelecs_alph] = 2.0
                grad_norm = gradient_rms(fock_cur)
                # The fine grid restarts from this density with a full J/K,
                # so a coarse stage that stops early needs no full K here.
                switch_ready = (early_switch and ene_old is not None
                                and energy_err < grid_switch_energy_tol)
                # An incremental K can look converged because the density
                # difference is small. Rebuild the full exchange before stopping.
                if (k_build == "incremental" and ene_old is not None
                        and energy_err < tol and grad_norm < tol
                        and not switch_ready):
                    started = time.perf_counter()
                    rss_before = _rss_gib()
                    k_cur = self._exchange_matrix(
                        dm_cur, auxbasis, grid_rad, grid_ang, batch_size,
                        overlap_fit, ao_tol, 1e-13,
                    )
                    k_record = _stage_record("k", started, rss_before)
                    k_build = "full"
                    k_physical = k_cur
                    fock_cur = hcore + j_cur - 0.5 * k_cur
                    ene_mo = ene_cur = 0.5 * np.einsum(
                        "mn,mn->", hcore + fock_cur, dm_cur)
                    ene_rhf = ene_nuc + ene_mo
                    ene_cur = ene_rhf
                    energy_err = abs(ene_cur - ene_old)
                    terms = scf_energy_terms(
                        dm_cur, hcore, j_physical, k_physical, overlap, ene_nuc)
                    check_physical_energy(terms, f"{stage} full K rebuild", nelectron)
                    grad_norm = gradient_rms(fock_cur)
                    part_times = (
                        f"{j_record} {k_record} {self._k_grid_note()} "
                        f"k_build={k_build}"
                    )

            if ene_old is not None:
                dm_err = np.linalg.norm(dm_cur - dm_old)
                energy_err = abs(ene_cur - ene_old)
                delta_energy = f"{energy_err:.3e}"
                delta_dm = f"{dm_err:.3e}"
            else:
                delta_energy = "n/a"
                delta_dm = "n/a"
            dm_old = dm_cur
            ene_old = ene_cur
            started = time.perf_counter()
            err = dense.commutator(fock_cur, dm_cur)
            err_norm = dense.norm(err)
            commutator_seconds = time.perf_counter() - started
            pending_error = err
            fock_physical = fock_cur
            iteration_scf += 1
            is_max_iteration = iteration_scf >= max_iteration
            if jk_mode == "rijcosx":
                is_converged = (energy_err < tol and grad_norm < tol) or switch_ready
                grad_field = f" grad={grad_norm:.3e}"
            else:
                is_converged = energy_err < tol and dm_err < tol and err_norm < tol
                grad_field = ""
            iteration_record = _stage_record(
                "iteration", iteration_started, iteration_rss,
            )
            gpu_field = (f" gpu_pool_gib={dense.pool_gib():.3f}"
                         if dense.backend != "cpu" else "")
            _emit_progress(
                f"SCF iteration {iteration_scf}/{max_iteration} "
                f"energy={ene_rhf:.12f} Ha "
                f"delta_energy={delta_energy} "
                f"delta_dm={delta_dm} "
                f"diis_err={err_norm:.3e}"
                f"{grad_field} "
                f"{diis_record} {eigh_record} {part_times} "
                f"commutator_s={commutator_seconds:.3f} {iteration_record} "
                f"peak_rss_gib={_peak_rss_gib():.3f}{gpu_field}",
                progress_log,
                molecule_name,
            )
            if jk_mode == "rijcosx":
                _emit_progress(
                    "SCF diagnostics " + json.dumps({
                        "iteration": iteration_scf,
                        "grid": [grid_rad, grid_ang],
                        **terms,
                        "physical_guard": "passed",
                        "j": self.last_j_stats,
                        "k": self.last_k_stats,
                    }, sort_keys=True, default=lambda value: value.item()),
                    progress_log, molecule_name,
                )

        if is_converged:
            if switch_ready:
                _emit_progress(
                    f"SCF coarse grid reached grid_switch_energy_tol at iteration "
                    f"{iteration_scf}: delta_energy={energy_err:.3e} < "
                    f"{grid_switch_energy_tol:.0e}",
                    progress_log, molecule_name,
                )
            else:
                _emit_progress(
                    f"SCF converged at iteration {iteration_scf}",
                    progress_log,
                    molecule_name,
                )
            if jk_mode == "rijcosx":
                if (grid_rad, grid_ang) != (final_grid_rad, final_grid_ang):
                    remaining = max_iteration - iteration_scf
                    if remaining < 2:
                        raise RuntimeError("SCF iteration budget exhausted before fine-grid convergence")
                    if dense.check_max:
                        _emit_progress(
                            f"SCF dense check grid={grid_rad}x{grid_ang} "
                            + json.dumps(dense.check_max, sort_keys=True),
                            progress_log, molecule_name)
                    # This frame stays alive during the fine-grid stage, so
                    # drop its DIIS history, J/K/Fock matrices and coarse grid.
                    rss_held = _rss_gib()
                    dense.clear()
                    dense = err = pending_error = None
                    fock_physical = fock_cur = j_physical = k_physical = None
                    j_cur = k_cur = C_occ = ene_mo = None
                    self.release_grid(auxbasis, grid_rad, grid_ang, overlap_fit)
                    _emit_progress(
                        "SCF switching to final grid; resetting DIIS and full J/K; "
                        f"released coarse-grid state rss_gib={rss_held:.3f}->{_rss_gib():.3f}",
                        progress_log, molecule_name,
                    )
                    result = self.solve_rhf(
                        nelecs, hcore, overlap, mol, ene_nuc,
                        max_iteration=remaining, tol=tol, dm_init=dm_old,
                        incremental_rebuild_interval=incremental_rebuild_interval,
                        jk_mode=jk_mode, auxbasis=auxbasis, df_tol=df_tol,
                        cosx_tol=cosx_tol, ao_tol=ao_tol,
                        grid_rad=final_grid_rad, grid_ang=final_grid_ang,
                        final_grid_rad=final_grid_rad, final_grid_ang=final_grid_ang,
                        batch_size=batch_size, overlap_fit=overlap_fit,
                        incremental_k=incremental_k, kfull_every=kfull_every,
                        progress_log=progress_log, molecule_name=molecule_name,
                        jk_engine=self.jk_engine,
                        grid_switch_energy_tol=grid_switch_energy_tol,
                    )
                    self.last_scf_result["iterations"] += iteration_scf
                    self.last_scf_result["grid_stages"].insert(0, {
                        "grid": [grid_rad, grid_ang], "iterations": iteration_scf,
                        "exit": "energy_switch" if switch_ready else "converged",
                    })
                    return result
                _emit_progress("SCF final energy", progress_log, molecule_name)
                started = time.perf_counter()
                rss_before = _rss_gib()
                j_final = self._coulomb_matrix(dm_old, auxbasis, df_tol)
                j_record = _stage_record("j", started, rss_before)
                started = time.perf_counter()
                rss_before = _rss_gib()
                k_final = self._exchange_matrix(
                    dm_old, auxbasis, final_grid_rad, final_grid_ang, batch_size,
                    overlap_fit, ao_tol, 1e-13, None)
                k_record = _stage_record("k", started, rss_before)
                fock_final = hcore + j_final - 0.5 * k_final
                ene_rhf = ene_nuc + 0.5 * np.einsum(
                    "mn,mn->", hcore + fock_final, dm_old)
                final_gradient = gradient_rms(fock_final)
                if (not np.isfinite(ene_rhf) or not np.isfinite(final_gradient)
                        or final_gradient >= tol
                        or abs(ene_rhf - ene_old) >= tol):
                    raise RuntimeError("Final full J/K rebuild failed finite-value, energy or gradient acceptance")
                terms = scf_energy_terms(
                    dm_old, hcore, j_final, k_final, overlap, ene_nuc)
                check_physical_energy(terms, "final full J/K rebuild", nelectron)
                self.last_scf_result = {
                    "converged": True, "grid": [final_grid_rad, final_grid_ang],
                    "energy_hartree": float(ene_rhf),
                    "gradient_rms": float(final_gradient),
                    "electron_count": float(np.einsum("ij,ji", dm_old, overlap)),
                    "jk_engine": self.jk_engine,
                    "iterations": iteration_scf,
                    "grid_stages": [{"grid": [final_grid_rad, final_grid_ang],
                                     "iterations": iteration_scf, "exit": "converged"}],
                    "j": dict(self.last_j_stats), "k": dict(self.last_k_stats),
                    "dense": {"backend": dense.backend, "device": dense.device,
                              "eigensolver": dense.eigensolver,
                              "check_calls": dense.check_calls,
                              "check_max_abs_diff": dict(dense.check_max)},
                    "grid_switch_energy_tol": grid_switch_energy_tol,
                    "k_memory": self._memory_note(),
                }
                if dense.check_max:
                    _emit_progress(
                        f"SCF dense check grid={final_grid_rad}x{final_grid_ang} "
                        + json.dumps(dense.check_max, sort_keys=True),
                        progress_log, molecule_name)
                dense.clear()
                _emit_progress(
                    f"SCF final energy={ene_rhf:.12f} Ha {j_record} {k_record} "
                    f"{self._k_grid_note()} peak_rss_gib={_peak_rss_gib():.3f}",
                    progress_log,
                    molecule_name,
                )
                _emit_progress(
                    "SCF final diagnostics " + json.dumps({
                        "grid": [final_grid_rad, final_grid_ang],
                        **terms,
                        "physical_guard": "passed",
                    }, sort_keys=True),
                    progress_log, molecule_name,
                )
        else:
            if ene_rhf is not None:
                raise RuntimeError(
                    f"SCF did not converge in {max_iteration} iterations; "
                    f"last energy was {ene_rhf:.12f} Ha"
                )
            else:
                raise RuntimeError("SCF is not running, fill in the code in the main loop.")

    

        return ene_rhf, dm_old

class _Tee:
    """Write solver output to both the console and a text log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _write_metadata_text(stream, metadata, estimate):
    ordered_keys = (
        "source_path", "basis", "basis_functions", "basis_shells",
        "atom_count", "nelectron", "nocc", "charge", "spin",
        "allocated_cpu_cores", "threads", "parallel_backend",
        "parallel_workers", "max_memory_mb", "k_memory_mb", "memory_ceiling_mb",
        "dense_backend", "torch_device", "grid_switch_energy_tol", "jk_engine",
    )
    for key in ordered_keys:
        print(f"{key}: {metadata.get(key)}", file=stream)
    print(
        "one_dense_matrix_gib: "
        f"{estimate['one_dense_matrix_gib']:.6f}",
        file=stream,
    )
    print(
        "estimated_dense_working_set_gib: "
        f"{estimate['estimated_dense_working_set_gib']:.6f}",
        file=stream,
    )


def _peak_rss_gib():
    """Process high-water mark in GiB. Linux reports VmHWM in KiB."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / (1024 * 1024)
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 ** 3)
    return rss / (1024 ** 2)


def _rss_gib():
    """Current resident set size in GiB. Linux reads VmRSS."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024 * 1024)
    return _peak_rss_gib()


def _stage_record(name, started, rss_before):
    """Text fields for one stage: seconds, RSS, and RSS growth."""
    import time
    rss = _rss_gib()
    seconds = time.perf_counter() - started
    increase = max(0.0, rss - rss_before)
    return (
        f"{name}_s={seconds:.3f} "
        f"{name}_rss_gib={rss:.3f} "
        f"{name}_rss_increase_gib={increase:.3f}"
    )


def _gib(array):
    return float(array.nbytes) / (1024 ** 3)


def benchmark_one_iteration(config_path, config):
    """Time one pass: integrals, guess, J, K, Fock, and diagonalization."""
    import time

    source_path = Path(config["xyz"])
    if not source_path.is_absolute():
        source_path = config_path.parent / source_path
    basis = config.get("basis", "cc-pvtz")
    resources = dict(config.get("resources", {}))
    threads = int(resources.get("threads", 1))
    allocated_cpu_cores = int(resources.get("cpu_cores", 1))
    parallel_backend = str(resources.get("parallel_backend", "serial"))
    parallel_workers = int(resources.get("parallel_workers", 1))
    parallel_chunk_size = int(resources.get("parallel_chunk_size", 1))
    parallel_min_shells = int(resources.get("parallel_min_shells", 12))
    if parallel_backend.lower() == "auto":
        parallel_backend = "cpu" if parallel_workers > 1 else "serial"
    dense_backend = str(resources.get("dense_backend", "torch")).lower()
    torch_device = str(resources.get("torch_device", "auto")).lower()
    lib.num_threads(threads)

    prefix = _output_prefix_path(config, config_path, source_path)
    summary_path = prefix.with_suffix(".json")

    solver_options = dict(config.get("solver_options", {}))
    stages = []
    previous_peak = _peak_rss_gib()

    def stage(name, function, measure):
        nonlocal previous_peak
        started = time.perf_counter()
        value = function()
        seconds = time.perf_counter() - started
        output_gib = float(measure(value))
        peak = _peak_rss_gib()
        record = {
            "stage": name,
            "seconds": seconds,
            "peak_rss_gib": peak,
            "rss_increase_gib": max(0.0, peak - previous_peak),
            "output_gib": output_gib,
        }
        previous_peak = peak
        stages.append(record)
        print(
            f"stage={name} seconds={seconds:.6f} "
            f"peak_rss_gib={peak:.6f} "
            f"rss_increase_gib={record['rss_increase_gib']:.6f} "
            f"output_gib={output_gib:.6f}",
            flush=True,
        )
        return value

    mol, xyz_metadata = stage(
        "molecule",
        lambda: build_molecule_from_xyz(
            source_path, basis=basis, unit=config.get("unit", "Angstrom"),
            charge=int(config.get("charge", 0)), spin=int(config.get("spin", 0)),
            verbose=int(config.get("pyscf_verbose", 0)),
        ),
        lambda _value: 0.0,
    )
    metadata = molecule_metadata(
        mol, basis, source_path=source_path, xyz_metadata=xyz_metadata,
    )
    expected = config.get("expected_basis_functions")
    if expected is not None and int(expected) != metadata["basis_functions"]:
        raise ValueError(
            "Configured basis-function count mismatch: "
            f"expected {int(expected)}, built {metadata['basis_functions']}"
        )
    estimate = estimate_dense_solver_memory_gib(metadata["basis_functions"])

    def one_electron():
        kinetic = mol.intor("int1e_kin")
        nuclear = mol.intor("int1e_nuc")
        overlap = mol.intor("int1e_ovlp")
        return kinetic + nuclear, overlap

    hcore, overlap = stage(
        "int1e",
        one_electron,
        lambda value: _gib(value[0]) + _gib(value[1]),
    )

    dm = stage(
        "init_guess",
        lambda: scf.hf.init_guess_by_minao(mol),
        _gib,
    )

    max_memory_mb = job_max_memory_mb(resources)
    mol.max_memory = max_memory_mb
    k_max_memory_mb, memory_ceiling_mb = job_k_memory_mb(resources)
    solver = Hartree_Fock(
        mol,
        max_memory=max_memory_mb,
        parallel_backend=parallel_backend,
        parallel_workers=parallel_workers,
        parallel_chunk_size=parallel_chunk_size,
        parallel_min_shells=parallel_min_shells,
        jk_engine=solver_options.get("jk_engine", "pyscf_reference"),
        k_max_memory_mb=k_max_memory_mb,
        memory_ceiling_mb=memory_ceiling_mb,
        dense_backend=dense_backend,
        torch_device=torch_device,
        dense_check_calls=int(resources.get("dense_check_iterations", 0)),
    )
    auxbasis = solver_options.get("auxbasis", "weigend")
    df_tol = float(solver_options.get("df_tol", 1e-8))
    grid_rad = int(solver_options.get("grid_rad", 30))
    grid_ang = int(solver_options.get("grid_ang", 110))
    batch_size = int(solver_options.get("batch_size", 512))
    cosx_tol = float(solver_options.get("cosx_tol", 1e-10))
    ao_tol = float(solver_options.get("ao_tol", 1e-10))
    overlap_fit = bool(solver_options.get("overlap_fit", True))

    coulomb = stage(
        "ri_j",
        lambda: solver.build_ri_j(dm, auxbasis, df_tol),
        _gib,
    )
    exchange = stage(
        "cosx_k",
        lambda: solver.build_cosx_k(
            dm, grid_rad, grid_ang, batch_size, cosx_tol, ao_tol, overlap_fit,
        ),
        _gib,
    )
    fock = stage(
        "fock",
        lambda: hcore + coulomb - 0.5 * exchange,
        _gib,
    )
    dense = stage(
        "dense_init",
        lambda: DenseLinearAlgebra(overlap, backend=dense_backend,
                                  torch_device=torch_device),
        lambda _value: 0.0,
    )
    evals, evecs, density_out = stage(
        "eigh",
        lambda: dense.solve(fock, mol.nelectron // 2),
        lambda value: sum(_gib(array) for array in value),
    )

    summary = {
        "status": "success",
        "mode": "benchmark_one_iteration",
        "passes": 1,
        "basis_functions": metadata["basis_functions"],
        "stages": stages,
        **metadata,
        **estimate,
        "allocated_cpu_cores": allocated_cpu_cores,
        "threads": threads,
        "parallel_backend": parallel_backend,
        "parallel_workers": parallel_workers,
        "auxbasis": auxbasis,
        "grid_rad": grid_rad,
        "grid_ang": grid_ang,
        "dense_backend": dense.backend,
        "dense_device": dense.device,
        "eigensolver": dense.eigensolver,
        "torch_device": torch_device if dense_backend == "torch" else None,
        "config_path": str(config_path),
        "orbital_energy_0": float(evals[0]),
        "density_shape": list(dm.shape),
        "fock_finite": bool(np.isfinite(fock).all()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"benchmark_one_iteration done basis_functions={metadata['basis_functions']}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    return summary


def run_from_config(config_path):
    """Run the configured HF calculation and write the text log."""
    import contextlib
    import time

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("benchmark_one_iteration"):
        return benchmark_one_iteration(config_path, config)

    resources = dict(config.get("resources") or {})
    backend = str(resources.get("parallel_backend", "serial")).lower().replace("_", "-")
    dense_backend = str(resources.get("dense_backend", "torch")).lower()
    torch_device = str(resources.get("torch_device", "auto")).lower()
    # Each backend selects its device explicitly. Do not hide CUDA globally:
    # callers may run CPU and GPU calculations consecutively in one process.
    source_path = Path(config["xyz"])
    if not source_path.is_absolute():
        source_path = config_path.parent / source_path

    basis = config.get("basis", "cc-pvtz")
    unit = config.get("unit", "Angstrom")
    charge = int(config.get("charge", 0))
    spin = int(config.get("spin", 0))
    mol, xyz_metadata = build_molecule_from_xyz(
        source_path, basis=basis, unit=unit, charge=charge, spin=spin,
        verbose=int(config.get("pyscf_verbose", 0)),
    )
    metadata = molecule_metadata(
        mol, basis, source_path=source_path, xyz_metadata=xyz_metadata,
    )
    mismatch = validate_basis_function_count(metadata)
    expected = config.get("expected_basis_functions")
    if expected is not None and int(expected) != metadata["basis_functions"]:
        raise ValueError(
            "Configured basis-function count mismatch: "
            f"expected {int(expected)}, built {metadata['basis_functions']}"
        )
    estimate = estimate_dense_solver_memory_gib(metadata["basis_functions"])
    max_memory_gib = config.get("max_memory_gib")
    if max_memory_gib is not None and estimate["estimated_dense_working_set_gib"] > float(max_memory_gib):
        raise MemoryError(
            "Estimated dense working set exceeds max_memory_gib: "
            f"{estimate['estimated_dense_working_set_gib']:.3f} > "
            f"{float(max_memory_gib):.3f}"
        )

    resources = dict(config.get("resources", {}))
    allocated_cpu_cores = int(resources.get("cpu_cores", 1))
    threads = int(resources.get("threads", 1))
    if allocated_cpu_cores <= 0 or threads <= 0:
        raise ValueError("resources.cpu_cores and resources.threads must be positive")
    if threads > allocated_cpu_cores:
        raise ValueError("resources.threads cannot exceed resources.cpu_cores")
    _check_thread_budget(resources.get("parallel_workers", 1), threads, allocated_cpu_cores)
    parallel_backend = str(resources.get("parallel_backend", "serial"))
    parallel_workers = int(resources.get("parallel_workers", 1))
    parallel_chunk_size = int(resources.get("parallel_chunk_size", 1))
    parallel_min_shells = int(resources.get("parallel_min_shells", 12))
    if parallel_chunk_size <= 0:
        raise ValueError("resources.parallel_chunk_size must be positive")
    if parallel_min_shells <= 0:
        raise ValueError("resources.parallel_min_shells must be positive")
    if parallel_backend.lower() == "auto":
        parallel_backend = "cpu" if parallel_workers > 1 else "serial"
    if parallel_workers > allocated_cpu_cores:
        raise ValueError(
            "resources.parallel_workers cannot exceed resources.cpu_cores"
        )
    lib.num_threads(threads)
    max_memory_mb = job_max_memory_mb(resources)
    mol.max_memory = max_memory_mb
    k_max_memory_mb, memory_ceiling_mb = job_k_memory_mb(resources)
    dense_check_calls = int(resources.get("dense_check_iterations", 0))

    prefix = _output_prefix_path(config, config_path, source_path)
    log_path = _available_log_path(prefix)

    solver_options = dict(config.get("solver_options", {}))
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    start = time.perf_counter()
    molecule_name = source_path.stem

    with log_path.open("w") as log_file:
        text_metadata = {
            **metadata,
            "allocated_cpu_cores": allocated_cpu_cores,
            "threads": threads,
            "parallel_backend": parallel_backend,
            "parallel_workers": parallel_workers,
            "parallel_chunk_size": parallel_chunk_size,
            "parallel_min_shells": parallel_min_shells,
            "max_memory_mb": max_memory_mb,
            "k_memory_mb": k_max_memory_mb,
            "memory_ceiling_mb": memory_ceiling_mb,
            "dense_backend": dense_backend,
            "torch_device": torch_device if dense_backend == "torch" else None,
            "grid_switch_energy_tol": solver_options.get("grid_switch_energy_tol"),
            "jk_engine": solver_options.get("jk_engine", "pyscf_reference"),
        }
        _write_metadata_text(log_file, text_metadata, estimate)
        _write_metadata_text(__import__("sys").stdout, text_metadata, estimate)

        if mismatch:
            print(f"WARNING: {mismatch}", file=log_file)
            print(f"WARNING: {mismatch}")

        tee = _Tee(__import__("sys").stdout, log_file)
        try:
            with contextlib.redirect_stdout(tee):
                solver = Hartree_Fock(
                    mol,
                    max_memory=max_memory_mb,
                    parallel_backend=parallel_backend,
                    parallel_workers=parallel_workers,
                    parallel_chunk_size=parallel_chunk_size,
                    parallel_min_shells=parallel_min_shells,
                    jk_engine=solver_options.get("jk_engine", "pyscf_reference"),
                    k_max_memory_mb=k_max_memory_mb,
                    memory_ceiling_mb=memory_ceiling_mb,
                    dense_backend=dense_backend,
                    dense_check_calls=dense_check_calls,
                    torch_device=torch_device,
                )
                energy, density = solver.solve_rhf(
                    mol.nelec,
                    hcore,
                    overlap,
                    mol,
                    mol.energy_nuc(),
                    progress_log=log_file,
                    molecule_name=molecule_name,
                    **solver_options,
                )
            result = {
                "status": "success",
                "energy_hartree": float(energy),
                "density_finite": bool(np.isfinite(density).all()),
                "elapsed_seconds": time.perf_counter() - start,
                "scf": solver.last_scf_result,
            }
        except Exception as exc:
            _emit_progress(
                f"SCF failed: {type(exc).__name__}: {exc}",
                log_file,
                molecule_name,
            )
            raise
        finally:
            _emit_progress(
                f"total_elapsed_s={time.perf_counter() - start:.3f}",
                log_file,
                molecule_name,
            )

    print(f"text_log: {log_path}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the custom RHF solver from JSON.")
    parser.add_argument("--config", required=True, help="JSON input profile")
    args = parser.parse_args(argv)
    run_from_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
