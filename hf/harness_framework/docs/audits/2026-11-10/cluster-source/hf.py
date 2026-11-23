import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import json
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.linalg import eigh
from pyscf import gto, scf, df, dft, lib
from pyscf.dft import gen_grid, radi, numint
from pyscf.gto import moleintor


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
                 parallel_min_shells=12):
        self.mol = mol
        self.max_memory = max_memory
        self._last_k_grid = None
        self.nao = mol.nao_nr()
        self.nshells = mol.nbas

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
                import cupy  # noqa: F401
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
        import cupy
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

    def _prepare_ri_j(self, auxbasis):
        if self._ri_cache is not None and self._ri_cache[0] == auxbasis:
            return self._ri_cache[1:]
        auxmol = df.addons.make_auxmol(self.mol, auxbasis)
        if not self.mol.cart and auxmol.cart:
            raise NotImplementedError('Interface for int3c2e_ssc')
        if self.mol.cart and not auxmol.cart:
            raise RuntimeError(
                'Cartesian orbitals for mol and spherical orbitals for auxmol not supported')
        metric = auxmol.intor('int2c2e')
        chol = scipy.linalg.cholesky(metric, lower=True)
        aux_loc = auxmol.ao_loc_nr()
        aux_q = np.zeros(auxmol.nbas)
        for psh in range(auxmol.nbas):
            diag = auxmol.intor('int2c2e', shls_slice=(psh, psh+1, psh, psh+1))
            aux_q[psh] = np.sqrt(np.max(np.abs(diag)))
        aux_order = np.argsort(aux_q)[::-1]
        ri_atm, ri_bas, ri_env = gto.mole.conc_env(
            self.mol._atm, self.mol._bas, self.mol._env,
            auxmol._atm, auxmol._bas, auxmol._env)
        intor = self.mol._add_suffix('int3c2e')
        # The optimizer is read by the integral driver. Reusing it removes a
        # per-pair setup that otherwise holds the GIL and hides thread speedup.
        cintopt = moleintor.make_cintopt(ri_atm, ri_bas, ri_env, intor)
        ao_loc = moleintor.make_loc(ri_bas, intor)
        self._ri_cache = (
            auxbasis, auxmol, chol, aux_loc, aux_q, aux_order,
            ri_atm, ri_bas, ri_env, intor, cintopt, ao_loc,
        )
        return self._ri_cache[1:]

    def _ri_pair_tensor(self, ish, jsh, ri_atm, ri_bas, ri_env, intor,
                        cintopt, ao_loc, aux_range):
        """(ij|P) for one contiguous block of auxiliary shells."""
        nbas = self.nshells
        p0, p1 = aux_range
        return moleintor.getints(
            intor, ri_atm, ri_bas, ri_env,
            (ish, ish + 1, jsh, jsh + 1, nbas + p0, nbas + p1),
            aosym='s1', ao_loc=ao_loc, cintopt=cintopt)

    def _foreach_aux_block(self, idx, aux_loc, aux_nbas, contract):
        """Visit auxiliary-shell blocks that contain screened AO indices."""
        if idx.size == 0:
            return
        for p0 in range(0, aux_nbas, _AUX_SHELL_BLOCK):
            p1 = min(aux_nbas, p0 + _AUX_SHELL_BLOCK)
            ao0 = int(aux_loc[p0])
            ao1 = int(aux_loc[p1])
            chosen = idx[(idx >= ao0) & (idx < ao1)]
            if chosen.size == 0:
                continue
            contract(p0, p1, chosen, chosen - ao0)

    @staticmethod
    def _aux_order_layout(aux_loc, order):
        """AO indices and prefix lengths for an auxiliary shell order."""
        ranges = [np.arange(aux_loc[psh], aux_loc[psh + 1]) for psh in order]
        if not ranges:
            empty = np.zeros(0, dtype=np.intp)
            return empty, empty
        counts = np.cumsum([len(chunk) for chunk in ranges])
        return np.concatenate(ranges), counts

    @staticmethod
    def _screened_aux_index(index, counts, weights, threshold):
        """Prefix of a descending weight order whose weights stay >= threshold."""
        if counts.size == 0:
            return index[:0]
        nkeep = counts.size - int(np.searchsorted(weights[::-1], threshold, side='left'))
        if nkeep <= 0:
            return index[:0]
        return index[:int(counts[nkeep - 1])]

    def build_ri_j(self, dm, auxbasis, df_tol=1e-8):
        """Build RI-J from screened AO pairs and blocked auxiliary shells."""

        dm = np.asarray(dm)
        if dm.shape != (self.nao, self.nao):
            raise ValueError("dm must have shape (nao, nao)")
        if auxbasis is None:
            raise ValueError("RI-J requires an explicit auxbasis")
        if df_tol <= 0:
            raise ValueError("df_tol must be positive")
        if self.coverage_complete:
            return self._build_ri_j_global(dm, auxbasis, df_tol)
        return self._build_ri_j_local(dm, auxbasis, df_tol)

    def _build_ri_j_global(self, dm, auxbasis, df_tol):
        """Global auxiliary fit. Used when every shell pair is in range."""

        (auxmol, chol, aux_loc, aux_q, aux_order,
         ri_atm, ri_bas, ri_env, intor, cintopt, ri_ao_loc) = self._prepare_ri_j(auxbasis)
        aux_nbas = auxmol.nbas
        gamma_index, gamma_counts = self._aux_order_layout(aux_loc, aux_order)
        gamma_weights = aux_q[aux_order]
        # df_tol targets the assembled J. Each shell pair is tested more tightly
        # so the auxiliary terms omitted from many pairs stay inside that target.
        pair_tol = df_tol / self.nao
        pair_data = []
        for ish, jsh in self.neighbor_pairs:
            i0, i1 = self.ao_loc[ish:ish+2]
            j0, j1 = self.ao_loc[jsh:jsh+2]
            qij = self.schwarz_ints[ish, jsh]
            dmax = np.max(np.abs(dm[i0:i1, j0:j1]))
            pair_data.append((ish, jsh, i0, i1, j0, j1, qij, dmax))
        self.last_shell_pairs = len(pair_data)
        gamma_pairs = []
        for pair in pair_data:
            qij = pair[6]
            dmax = pair[7]
            idx = self._screened_aux_index(
                gamma_index, gamma_counts, gamma_weights * (qij * dmax), pair_tol)
            if idx.size:
                gamma_pairs.append((*pair, idx))
        self.last_ri_screened_pairs = len(gamma_pairs)

        def gamma_task(pair):
            ish, jsh, i0, i1, j0, j1, qij, dmax, idx = pair
            local = np.zeros(auxmol.nao_nr())
            dm_block = dm[i0:i1, j0:j1]
            fac = 1.0 if ish == jsh else 2.0

            def contract(p0, p1, chosen, columns):
                eri = self._ri_pair_tensor(
                    ish, jsh, ri_atm, ri_bas, ri_env, intor, cintopt,
                    ri_ao_loc, (p0, p1))
                block = eri[:, :, columns]
                if self.parallel_backend == "gpu":
                    value = self._gpu_einsum('ijp,ij->p', block, dm_block)
                else:
                    value = np.einsum('ijp,ij->p', block, dm_block)
                local[chosen] = fac * value

            self._foreach_aux_block(idx, aux_loc, aux_nbas, contract)
            return local

        gamma = np.zeros(auxmol.nao_nr())
        for local in self._parallel_map(gamma_task, gamma_pairs):
            gamma += local
        coeff = scipy.linalg.cho_solve((chol, True), gamma)
        coeff_q = np.array([
            aux_q[psh] * np.max(np.abs(coeff[aux_loc[psh]:aux_loc[psh+1]]))
            for psh in range(auxmol.nbas)
        ])
        coeff_order = np.argsort(coeff_q)[::-1]
        j_index, j_counts = self._aux_order_layout(aux_loc, coeff_order)
        j_weights = coeff_q[coeff_order]

        j_pairs = []
        for pair in pair_data:
            qij = pair[6]
            idx = self._screened_aux_index(j_index, j_counts, j_weights * qij, pair_tol)
            if idx.size:
                j_pairs.append((*pair, idx))

        def j_task(pair):
            ish, jsh, i0, i1, j0, j1, qij, _, idx = pair
            block = np.zeros((i1 - i0, j1 - j0))

            def contract(p0, p1, chosen, columns):
                eri = self._ri_pair_tensor(
                    ish, jsh, ri_atm, ri_bas, ri_env, intor, cintopt,
                    ri_ao_loc, (p0, p1))
                eri_block = eri[:, :, columns]
                coeff_block = coeff[chosen]
                if self.parallel_backend == "gpu":
                    block[...] += self._gpu_einsum('ijp,p->ij', eri_block, coeff_block)
                else:
                    block[...] += np.einsum('ijp,p->ij', eri_block, coeff_block)

            self._foreach_aux_block(idx, aux_loc, aux_nbas, contract)
            return ish, jsh, i0, i1, j0, j1, block

        j = np.zeros_like(dm)
        for _, _, i0, i1, j0, j1, block in self._parallel_map(j_task, j_pairs):
            j[i0:i1, j0:j1] = block
            j[j0:j1, i0:i1] = block.T
        return (j + j.T) * 0.5

    @staticmethod
    def _rectangular_neighbors(coords, radii, other_coords, other_radii):
        """Auxiliary shells inside the cutoff of each AO shell."""

        if len(coords) == 0 or len(other_coords) == 0:
            return [np.zeros(0, dtype=np.int32) for _ in coords]
        reach = float(np.max(radii) + np.max(other_radii))
        if reach <= 0.0:
            reach = 1e-8
        buckets = defaultdict(list)
        for index, point in enumerate(other_coords):
            key = tuple(np.floor(point / reach).astype(int))
            buckets[key].append(index)
        neighbors = []
        for point, radius in zip(coords, radii):
            base = np.floor(point / reach).astype(int)
            found = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (
                            int(base[0]) + dx,
                            int(base[1]) + dy,
                            int(base[2]) + dz,
                        )
                        for other in buckets.get(key, ()):
                            dist = float(np.linalg.norm(point - other_coords[other]))
                            if dist <= float(radius) + float(other_radii[other]) + 1e-8:
                                found.append(other)
            neighbors.append(np.asarray(found, dtype=np.int32))
        return neighbors

    def _aux_metric(self, auxmol, shells):
        """Two-center metric of an arbitrary auxiliary-shell list."""

        shells = np.asarray(shells, dtype=int)
        loc = auxmol.ao_loc_nr()
        sizes = loc[shells + 1] - loc[shells]
        nao = int(np.sum(sizes))
        metric = np.zeros((nao, nao))
        runs = []
        start = 0
        while start < len(shells):
            stop = start + 1
            while stop < len(shells) and shells[stop] == shells[stop - 1] + 1:
                stop += 1
            runs.append((start, stop))
            start = stop
        run_ao = []
        cursor = 0
        for begin, end in runs:
            run_ao.append(cursor)
            cursor += int(np.sum(sizes[begin:end]))
        for left, (begin, end) in enumerate(runs):
            for right, (other_begin, other_end) in enumerate(runs):
                block = auxmol.intor(
                    'int2c2e',
                    shls_slice=(
                        int(shells[begin]), int(shells[end - 1]) + 1,
                        int(shells[other_begin]), int(shells[other_end - 1]) + 1,
                    ),
                )
                row = run_ao[left]
                col = run_ao[right]
                row_n = int(np.sum(sizes[begin:end]))
                col_n = int(np.sum(sizes[other_begin:other_end]))
                metric[row:row + row_n, col:col + col_n] = block
        return metric

    def _local_aux_layout(self, aux_loc, shells):
        """Map each auxiliary shell to a slice of the local metric."""

        layout = {}
        cursor = 0
        for shell in shells:
            start = int(aux_loc[shell])
            stop = int(aux_loc[shell + 1])
            layout[int(shell)] = (cursor, cursor + (stop - start), start, stop)
            cursor += stop - start
        return layout, cursor

    def _contract_local_aux(self, ish, jsh, shells, layout, ri_atm, ri_bas,
                            ri_env, intor, cintopt, ri_ao_loc, contract):
        """Integrate one AO pair against contiguous runs of auxiliary shells."""

        if len(shells) == 0:
            return
        start = 0
        shells = [int(shell) for shell in shells]
        while start < len(shells):
            stop = start + 1
            while stop < len(shells) and shells[stop] == shells[start] + (stop - start):
                stop += 1
            run = shells[start:stop]
            p0 = run[0]
            origin = layout[p0][2]
            eri = self._ri_pair_tensor(
                ish, jsh, ri_atm, ri_bas, ri_env, intor, cintopt,
                ri_ao_loc, (p0, run[-1] + 1))
            columns = []
            local_cols = []
            for shell in run:
                local0, local1, absolute0, absolute1 = layout[shell]
                col0 = absolute0 - origin
                columns.extend(range(col0, col0 + (absolute1 - absolute0)))
                local_cols.extend(range(local0, local1))
            contract(eri[:, :, columns], np.asarray(local_cols, dtype=int))
            start = stop

    def _prepare_ri_local(self, auxbasis):
        cache = getattr(self, "_ri_local_cache", None)
        if cache is not None and cache[0] == auxbasis:
            return cache[1]
        auxmol = df.addons.make_auxmol(self.mol, auxbasis)
        if not self.mol.cart and auxmol.cart:
            raise NotImplementedError('Interface for int3c2e_ssc')
        if self.mol.cart and not auxmol.cart:
            raise RuntimeError(
                'Cartesian orbitals for mol and spherical orbitals for auxmol not supported')
        aux_loc = auxmol.ao_loc_nr()
        aux_q = np.zeros(auxmol.nbas)
        for shell in range(auxmol.nbas):
            diag = auxmol.intor('int2c2e', shls_slice=(shell, shell + 1, shell, shell + 1))
            aux_q[shell] = np.sqrt(np.max(np.abs(diag))) if diag.size else 0.0
        aux_coords = self._shell_coords_of(auxmol)
        aux_radii = self._gaussian_radii(auxmol, _LOCAL_RADIUS_PRECISION, 0.0)
        ao_aux = self._rectangular_neighbors(
            self.shell_coords, self.shell_radii, aux_coords, aux_radii)
        domain_of = [
            tuple(int(value) for value in np.floor(point / _LOCAL_DOMAIN_BOHR))
            for point in aux_coords
        ]
        domain_shells = defaultdict(list)
        for shell, key in enumerate(domain_of):
            domain_shells[key].append(shell)
        fit_shells = {}
        for key, owned in domain_shells.items():
            owned_coords = aux_coords[np.asarray(owned, dtype=int)]
            chosen = []
            for shell, point in enumerate(aux_coords):
                distance = np.linalg.norm(owned_coords - point, axis=1)
                if float(np.min(distance)) <= _LOCAL_HALO_BOHR + 1e-8:
                    chosen.append(shell)
            fit_shells[key] = chosen
        ri_atm, ri_bas, ri_env = gto.mole.conc_env(
            self.mol._atm, self.mol._bas, self.mol._env,
            auxmol._atm, auxmol._bas, auxmol._env)
        intor = self.mol._add_suffix('int3c2e')
        cintopt = moleintor.make_cintopt(ri_atm, ri_bas, ri_env, intor)
        ri_ao_loc = moleintor.make_loc(ri_bas, intor)
        payload = {
            "auxmol": auxmol,
            "aux_loc": aux_loc,
            "aux_q": aux_q,
            "ao_aux": ao_aux,
            "domain_of": domain_of,
            "domain_shells": domain_shells,
            "fit_shells": fit_shells,
            "ri_atm": ri_atm,
            "ri_bas": ri_bas,
            "ri_env": ri_env,
            "intor": intor,
            "cintopt": cintopt,
            "ri_ao_loc": ri_ao_loc,
            "metrics": {},
        }
        self._ri_local_cache = (auxbasis, payload)
        return payload

    def _build_ri_j_local(self, dm, auxbasis, df_tol):
        """Near-field RI-J. Each domain fits a halo but emits only its own functions."""

        prepared = self._prepare_ri_local(auxbasis)
        auxmol = prepared["auxmol"]
        aux_loc = prepared["aux_loc"]
        aux_q = prepared["aux_q"]
        ao_aux = prepared["ao_aux"]
        domain_shells = prepared["domain_shells"]
        fit_shells = prepared["fit_shells"]
        pair_tol = df_tol
        shell_domains = defaultdict(list)
        for key, shells in fit_shells.items():
            for shell in shells:
                shell_domains[shell].append(key)
        domain_pairs = defaultdict(list)
        self.last_shell_pairs = len(self.neighbor_pairs)
        screened = 0
        for ish, jsh in self.neighbor_pairs:
            left = ao_aux[ish]
            right = ao_aux[jsh]
            if left.size == 0 and right.size == 0:
                continue
            if left.size == 0:
                aux_shells = right
            elif right.size == 0:
                aux_shells = left
            else:
                aux_shells = np.unique(np.concatenate((left, right)))
            i0, i1 = (int(bound) for bound in self.ao_loc[ish:ish + 2])
            j0, j1 = (int(bound) for bound in self.ao_loc[jsh:jsh + 2])
            qij = float(self.schwarz_ints[ish, jsh])
            dmax = float(np.max(np.abs(dm[i0:i1, j0:j1])))
            by_domain = defaultdict(list)
            for shell in aux_shells:
                shell = int(shell)
                if aux_q[shell] * qij * dmax < pair_tol:
                    continue
                for key in shell_domains[shell]:
                    by_domain[key].append(shell)
            if not by_domain:
                continue
            screened += 1
            for key, kept in by_domain.items():
                domain_pairs[key].append(
                    (ish, jsh, i0, i1, j0, j1, qij, dmax, kept))
        self.last_ri_screened_pairs = screened
        j_matrix = np.zeros_like(dm)
        for key, pairs in domain_pairs.items():
            shells = np.asarray(fit_shells[key], dtype=int)
            if shells.size == 0:
                continue
            cached = prepared["metrics"].get(key)
            if cached is None:
                metric = self._aux_metric(auxmol, shells)
                try:
                    chol = scipy.linalg.cholesky(metric, lower=True)
                except scipy.linalg.LinAlgError:
                    metric = metric + (1e-12 * np.eye(metric.shape[0]))
                    chol = scipy.linalg.cholesky(metric, lower=True)
                layout, width = self._local_aux_layout(aux_loc, shells)
                prepared["metrics"][key] = (chol, layout, width, shells)
            else:
                chol, layout, width, shells = cached
            ri_args = (
                prepared["ri_atm"], prepared["ri_bas"], prepared["ri_env"],
                prepared["intor"], prepared["cintopt"], prepared["ri_ao_loc"],
            )
            owned = {int(shell) for shell in domain_shells[key]}
            shell_of_local = np.empty(width, dtype=int)
            for shell, (local0, local1, _start, _stop) in layout.items():
                shell_of_local[local0:local1] = int(shell)

            def gamma_task(pair, _shells=shells, _layout=layout, _width=width,
                           _owned=owned, _shell_of_local=shell_of_local):
                ish, jsh, i0, i1, j0, j1, qij, dmax, kept = pair
                local = np.zeros(_width)
                kept_set = set(kept)
                run = [shell for shell in _shells if shell in kept_set]
                dm_block = dm[i0:i1, j0:j1]
                factor = 1.0 if ish == jsh else 2.0
                pieces = []

                def contract(eri_block, local_cols):
                    value = np.einsum('ijp,ij->p', eri_block, dm_block)
                    local[local_cols] = factor * value
                    cursor = 0
                    start = 0
                    while start < len(local_cols):
                        shell = int(_shell_of_local[int(local_cols[start])])
                        local0, local1, _absolute0, _absolute1 = _layout[shell]
                        span = local1 - local0
                        if shell in _owned:
                            pieces.append((
                                shell, np.array(eri_block[:, :, cursor:cursor + span], copy=True)))
                        cursor += span
                        start += span

                self._contract_local_aux(
                    ish, jsh, run, _layout, *ri_args, contract)
                return local, pieces

            gamma = np.zeros(width)
            stored = []
            for local, pieces in self._parallel_map(gamma_task, pairs):
                gamma += local
                stored.append(pieces)
            coeff = scipy.linalg.cho_solve((chol, True), gamma)
            for pair, pieces in zip(pairs, stored):
                ish, jsh, i0, i1, j0, j1, qij, dmax, kept = pair
                block = np.zeros((i1 - i0, j1 - j0))
                for shell, eri_block in pieces:
                    local0, local1, _absolute0, _absolute1 = layout[shell]
                    weight = aux_q[shell] * float(np.max(np.abs(coeff[local0:local1])))
                    if weight * qij < pair_tol:
                        continue
                    block += np.einsum('ijp,p->ij', eri_block, coeff[local0:local1])
                j_matrix[i0:i1, j0:j1] += block
                if ish != jsh:
                    j_matrix[j0:j1, i0:i1] += block.T
        return (j_matrix + j_matrix.T) * 0.5

    def _cosx_grid(self, grid_rad, grid_ang):
        grids = gen_grid.Grids(self.mol)
        grids.atom_grid = {self.mol.atom_pure_symbol(i): (grid_rad, grid_ang)
                           for i in range(self.mol.natm)}
        grids.radi_method = radi.treutler_ahlrichs
        grids.prune = gen_grid.nwchem_prune
        grids.build()
        return grids.coords, grids.weights

    def build_cosx_k(self, dm, grid_rad=30, grid_ang=110, batch_size=512,
                     cosx_tol=1e-10, ao_tol=1e-10, overlap_fit=True):
        """Build semi-numerical COSX exchange in grid batches.

        The pair cutoff follows Q_ij * max|F_i,F_j| >= cosx_tol; it is an
        empirical screening rule, not a certified error bound.
        """
        dm = np.asarray(dm)
        if dm.shape != (self.nao, self.nao):
            raise ValueError("dm must have shape (nao, nao)")
        if batch_size <= 0 or grid_rad <= 0 or grid_ang <= 0:
            raise ValueError("grid sizes and batch_size must be positive")
        if not self.coverage_complete:
            return self._build_cosx_k_local(
                dm, grid_rad, grid_ang, batch_size, cosx_tol, ao_tol, overlap_fit)
        coords, weights = self._cosx_grid(grid_rad, grid_ang)
        ao_loc = self.ao_loc
        batches = [(g0, min(g0 + batch_size, len(weights)))
                   for g0 in range(0, len(weights), batch_size)]
        intor = self.mol._add_suffix('int1e_grids')
        atm = self.mol._atm
        bas = self.mol._bas
        base_env = np.asarray(self.mol._env, dtype=np.float64)
        self.cosx_full_tensor_calls = 0
        shell_block = 1 if self.nshells <= 1 else min(
            _COSX_SHELL_BLOCK, self.nshells - 1)

        def grid_task(batch):
            g0, g1 = batch
            xyz = coords[g0:g1]
            w = weights[g0:g1]
            x = numint.eval_ao(self.mol, xyz)
            x[np.abs(x) < ao_tol] = 0.0
            f = x @ dm
            s_batch = x.T @ (w[:, None] * x)
            g = np.zeros_like(x)
            flat_grids = np.asarray(xyz, dtype=np.float64).ravel()
            env = np.empty(base_env.size + flat_grids.size, dtype=np.float64)
            env[:base_env.size] = base_env
            env[base_env.size:] = flat_grids
            env[moleintor.NGRIDS] = xyz.shape[0]
            env[moleintor.PTR_GRIDS] = base_env.size
            cintopt = moleintor.make_cintopt(atm, bas, env, intor)
            fmax = np.array([np.max(np.abs(f[:, ao_loc[i]:ao_loc[i + 1]]))
                             for i in range(self.nshells)])
            for i0 in range(0, self.nshells, shell_block):
                i1 = min(self.nshells, i0 + shell_block)
                for j0 in range(0, i1, shell_block):
                    j1 = min(self.nshells, j0 + shell_block)
                    selected = []
                    for ish in range(i0, i1):
                        j_hi = min(j1, ish + 1)
                        for jsh in range(j0, j_hi):
                            if self.schwarz_ints[ish, jsh] * max(fmax[ish], fmax[jsh]) < cosx_tol:
                                continue
                            selected.append((ish, jsh))
                    if not selected:
                        continue
                    if (i1 - i0) == self.nshells and (j1 - j0) == self.nshells:
                        with self._stats_lock:
                            self.cosx_full_tensor_calls += 1
                    a = moleintor.getints(
                        intor, atm, bas, env, (i0, i1, j0, j1), cintopt=cintopt)
                    base_i = int(ao_loc[i0])
                    base_j = int(ao_loc[j0])
                    for ish, jsh in selected:
                        ia = int(ao_loc[ish]) - base_i
                        ib = int(ao_loc[ish + 1]) - base_i
                        ja = int(ao_loc[jsh]) - base_j
                        jb = int(ao_loc[jsh + 1]) - base_j
                        a_block = a[:, ia:ib, ja:jb]
                        i_ao0, i_ao1 = int(ao_loc[ish]), int(ao_loc[ish + 1])
                        j_ao0, j_ao1 = int(ao_loc[jsh]), int(ao_loc[jsh + 1])
                        g[:, i_ao0:i_ao1] += np.einsum(
                            'gij,gj->gi', a_block, f[:, j_ao0:j_ao1])
                        if ish != jsh:
                            g[:, j_ao0:j_ao1] += np.einsum(
                                'gij,gi->gj', a_block, f[:, i_ao0:i_ao1])
            k_batch = g.T @ (w[:, None] * x)
            return s_batch, k_batch

        k = np.zeros((self.nao, self.nao))
        s_num = np.zeros_like(k)
        for s_batch, k_batch in self._parallel_map(grid_task, batches):
            s_num += s_batch
            k += k_batch
        if overlap_fit:
            overlap = self.mol.intor('int1e_ovlp')
            try:
                s_fit = np.linalg.solve(s_num, overlap)
                k = (s_fit.T @ k + k @ s_fit) * 0.5
            except np.linalg.LinAlgError:
                pass
        return (k + k.T) * 0.5

    def _local_overlap_fit(self, k_matrix, s_num, overlap):
        """Fit each atom block. A single covering block is the dense solve."""

        atoms = np.asarray(self.mol._bas[:, gto.ATOM_OF], dtype=int)
        fitted = k_matrix.copy()
        for atom in range(self.mol.natm):
            shells = np.flatnonzero(atoms == atom)
            if shells.size == 0:
                continue
            ao0 = int(self.ao_loc[int(shells[0])])
            ao1 = int(self.ao_loc[int(shells[-1]) + 1])
            block_s = s_num[ao0:ao1, ao0:ao1]
            block_k = k_matrix[ao0:ao1, ao0:ao1]
            try:
                s_fit = np.linalg.solve(block_s, overlap[ao0:ao1, ao0:ao1])
            except np.linalg.LinAlgError:
                continue
            fitted[ao0:ao1, ao0:ao1] = (s_fit.T @ block_k + block_k @ s_fit) * 0.5
        return fitted

    def _build_cosx_k_local(self, dm, grid_rad, grid_ang, batch_size,
                            cosx_tol, ao_tol, overlap_fit):
        """COSX using only basis functions near each grid batch."""

        coords, weights = self._cosx_grid(grid_rad, grid_ang)
        ao_loc = self.ao_loc
        batches = self._spatial_grid_batches(coords, batch_size, 6.0)
        intor = self.mol._add_suffix('int1e_grids')
        atm = self.mol._atm
        bas = self.mol._bas
        base_env = np.asarray(self.mol._env, dtype=np.float64)
        self.cosx_full_tensor_calls = 0
        schwarz = self.schwarz_ints
        if self.nshells <= 1:
            run_limit = 1
        else:
            run_limit = min(_COSX_LOCAL_RUN, self.nshells - 1)

        def grid_task(index):
            xyz = coords[index]
            w = weights[index]
            active = self._shells_near_points(xyz)
            x = np.zeros((xyz.shape[0], self.nao))
            eval_runs = self._shell_runs(active, 0)
            for begin, end in eval_runs:
                values = numint.eval_ao(self.mol, xyz, shls_slice=(begin, end))
                x[:, int(ao_loc[begin]):int(ao_loc[end])] = values
            x[np.abs(x) < ao_tol] = 0.0
            columns = np.concatenate([
                np.arange(int(ao_loc[begin]), int(ao_loc[end]))
                for begin, end in eval_runs
            ]) if eval_runs else np.zeros(0, dtype=int)
            f = np.zeros_like(x)
            g = np.zeros_like(x)
            weighted = None
            if columns.size:
                f[:, columns] = x[:, columns] @ dm[np.ix_(columns, columns)]
                weighted = w[:, None] * x[:, columns]
            flat_grids = np.asarray(xyz, dtype=np.float64).ravel()
            env = np.empty(base_env.size + flat_grids.size, dtype=np.float64)
            env[:base_env.size] = base_env
            env[base_env.size:] = flat_grids
            env[moleintor.NGRIDS] = xyz.shape[0]
            env[moleintor.PTR_GRIDS] = base_env.size
            cintopt = moleintor.make_cintopt(atm, bas, env, intor)
            fmax = np.zeros(self.nshells)
            for begin, end in eval_runs:
                fmax[begin:end] = [
                    np.max(np.abs(f[:, int(ao_loc[shell]):int(ao_loc[shell + 1])]))
                    for shell in range(begin, end)
                ]
            integral_runs = self._shell_runs(active, run_limit)
            for left, (i0, i1) in enumerate(integral_runs):
                left_f = float(np.max(fmax[i0:i1])) if i1 > i0 else 0.0
                for j0, j1 in integral_runs[:left + 1]:
                    if i1 <= i0 or j1 <= j0:
                        continue
                    bound = float(np.max(schwarz[i0:i1, j0:j1]))
                    right_f = float(np.max(fmax[j0:j1]))
                    if bound * max(left_f, right_f) < cosx_tol:
                        continue
                    if (i1 - i0) == self.nshells and (j1 - j0) == self.nshells:
                        with self._stats_lock:
                            self.cosx_full_tensor_calls += 1
                    values = moleintor.getints(
                        intor, atm, bas, env, (i0, i1, j0, j1), cintopt=cintopt)
                    i_ao0, i_ao1 = int(ao_loc[i0]), int(ao_loc[i1])
                    j_ao0, j_ao1 = int(ao_loc[j0]), int(ao_loc[j1])
                    g[:, i_ao0:i_ao1] += np.einsum(
                        'gij,gj->gi', values, f[:, j_ao0:j_ao1])
                    if (i0, i1) != (j0, j1):
                        g[:, j_ao0:j_ao1] += np.einsum(
                            'gij,gi->gj', values, f[:, i_ao0:i_ao1])
            if not columns.size:
                empty = np.zeros((0, 0))
                return columns, empty, empty
            s_block = x[:, columns].T @ weighted
            k_block = g[:, columns].T @ weighted
            return columns, s_block, k_block

        k_matrix = np.zeros((self.nao, self.nao))
        s_num = np.zeros_like(k_matrix)
        for columns, s_block, k_block in self._parallel_map(grid_task, batches):
            if columns.size == 0:
                continue
            positions = np.ix_(columns, columns)
            s_num[positions] += s_block
            k_matrix[positions] += k_block
        if overlap_fit:
            overlap = self.mol.intor('int1e_ovlp')
            k_matrix = self._local_overlap_fit(k_matrix, s_num, overlap)
        return (k_matrix + k_matrix.T) * 0.5

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
            self._sgx_engines[key] = engine
        return engine

    def _sgx_block_size(self, engine):
        """Grid count and block size used by ``sgx_jk.get_k_only``.

        This copies ``SGXData.get_loop_data`` for one density matrix. The
        value is recorded before the kernel runs, while the resident set is
        the one that kernel will see.
        """

        from pyscf import lib
        from pyscf.sgx import sgx_jk

        ngrids = int(engine.grids.coords.shape[0])
        available = float(engine.max_memory) - float(lib.current_memory()[0])
        data_dim = 4
        raw = int(available * 1e6 / 8 / (data_dim * self.nao))
        blksize = max(int(sgx_jk.SGX_BLKSIZE), raw)
        aligned = max(1, blksize // int(sgx_jk.SGX_BLKSIZE)) * int(sgx_jk.SGX_BLKSIZE)
        return ngrids, min(ngrids, aligned)

    def _k_grid_note(self):
        ngrids, blksize = self._last_k_grid
        return f"ngrids={ngrids} blksize={blksize}"

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
    ):
        """Return Coulomb and exchange matrices for one density.

        ``direct`` is the analytic small-system path. ``rijcosx`` is the
        large-system path and does not run the SCF loop.
        """

        if jk_mode == "direct":
            return self.build_jk(dm)
        if jk_mode != "rijcosx":
            raise ValueError("jk_mode must be 'direct' or 'rijcosx'")
        if auxbasis is None:
            raise ValueError("rijcosx mode requires an explicit auxbasis")
        vj = self.build_ri_j(dm, auxbasis, df_tol)
        vk = self.build_cosx_k(
            dm, grid_rad, grid_ang, batch_size, cosx_tol, ao_tol, overlap_fit)
        return vj, vk

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
    ):
        import time

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
        if jk_mode == "rijcosx":
            _emit_progress(
                f"RIJCOSX: backend=pyscf-sgx, auxbasis={auxbasis}, "
                f"direct_scf_tol=1e-13, "
                f"grid={grid_rad}x{grid_ang}, final_grid="
                f"{final_grid_rad}x{final_grid_ang}, incremental_k={incremental_k}, "
                f"kfull_every={kfull_every}, max_memory_mb={float(self.max_memory):.0f}",
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
    

        error_list = []
        fock_list = []
        max_diis = 8
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
            started = time.perf_counter()
            rss_before = _rss_gib()
            j_physical = self.build_df_j(dm_old, auxbasis)
            j_record = _stage_record("j", started, rss_before)
            started = time.perf_counter()
            rss_before = _rss_gib()
            k_physical = self.build_sgx_k(
                dm_old, auxbasis, grid_rad, grid_ang, overlap_fit)
            k_record = _stage_record("k", started, rss_before)
            fock_physical = hcore + j_physical - 0.5 * k_physical
            _emit_progress(
                f"SCF initial Fock {j_record} {k_record} {self._k_grid_note()}",
                progress_log,
                molecule_name,
            )
        incremental_updates = 0

        def diis_error(fock, ovlp, dm):
            return fock @ dm @ ovlp - ovlp @ dm @ fock
            

        def diis_extrapolate(errors, focks):
                space = len(errors)
                if space < 2:
                    return None
                B = np.zeros((space+1, space+1))
                B[-1, :-1] = B[:-1, -1] = 1
                B[-1, -1] = 0
                g = np.zeros(space+1)
                g[-1] = 1
                for i in range(space):
                    for j in range(space):
                        B[i, j] = B[j, i] = np.vdot(errors[i], errors[j]).real
                if not np.isfinite(np.linalg.cond(B)) or np.linalg.cond(B) > 1e12:
                    return None
                try:
                    c = scipy.linalg.solve(B, g, assume_a='sym')[:-1]
                except scipy.linalg.LinAlgError:
                    return None

                fock_new = np.einsum("i,ijk->jk", c, focks)
                return fock_new

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
            err = diis_error(fock_physical, overlap, dm_old)
            err_norm = np.linalg.norm(err)
            if err_norm > 1e-12:
                error_list.append(err)
                fock_list.append(fock_physical.copy())
            if len(error_list) > max_diis:
                error_list.pop(0)
                fock_list.pop(0)

            fock_for_diagonalization = fock_physical
            if len(error_list) >= 2:
                fock_new = diis_extrapolate(error_list, fock_list)
                if fock_new is not None:
                    fock_for_diagonalization = fock_new
            diis_record = _stage_record("diis", started, rss_before)

            started = time.perf_counter()
            rss_before = _rss_gib()
            ene_mo, C_occ = eigh(fock_for_diagonalization, overlap)
            C_occ_cur = C_occ[:, :nelecs_alph]
            dm_cur = 2.0 * np.einsum("ma,na->mn", C_occ_cur, C_occ_cur.conj())
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
                started = time.perf_counter()
                rss_before = _rss_gib()
                j_cur = self.build_df_j(dm_cur, auxbasis)
                j_record = _stage_record("j", started, rss_before)
                kfull = (not incremental_k or iteration_scf == 0
                         or (iteration_scf + 1) % kfull_every == 0
                         or iteration_scf + 1 >= max_iteration)
                started = time.perf_counter()
                rss_before = _rss_gib()
                if kfull:
                    k_cur = self.build_sgx_k(
                        dm_cur, auxbasis, grid_rad, grid_ang, overlap_fit)
                    k_build = "full"
                else:
                    k_cur = k_physical + self.build_sgx_k(
                        delta_dm, auxbasis, grid_rad, grid_ang, overlap_fit,
                        screen_dm=dm_cur)
                    k_build = "incremental"
                k_record = _stage_record("k", started, rss_before)
                fock_cur = hcore + j_cur - 0.5 * k_cur
                j_physical, k_physical = j_cur, k_cur
                part_times = (
                    f"{j_record} {k_record} {self._k_grid_note()} k_build={k_build}"
                )

            ene_mo = ene_cur = 0.5 * np.einsum("mn,mn->", hcore + fock_cur, dm_cur)
            ene_rhf = ene_nuc + ene_mo
            ene_cur = ene_rhf
            if ene_old is not None:
                energy_err = abs(ene_cur - ene_old)
            if jk_mode == "rijcosx":
                mo_occ = np.zeros(nao)
                mo_occ[:nelecs_alph] = 2.0
                grad_norm = float(np.linalg.norm(
                    scf.hf.get_grad(C_occ, mo_occ, fock_cur)))
                # An incremental K can look converged because the density
                # difference is small. Rebuild the full exchange before stopping.
                if (k_build == "incremental" and ene_old is not None
                        and energy_err < tol and grad_norm < tol):
                    started = time.perf_counter()
                    rss_before = _rss_gib()
                    k_cur = self.build_sgx_k(
                        dm_cur, auxbasis, grid_rad, grid_ang, overlap_fit)
                    k_record = _stage_record("k", started, rss_before)
                    k_build = "full"
                    k_physical = k_cur
                    fock_cur = hcore + j_cur - 0.5 * k_cur
                    ene_mo = ene_cur = 0.5 * np.einsum(
                        "mn,mn->", hcore + fock_cur, dm_cur)
                    ene_rhf = ene_nuc + ene_mo
                    ene_cur = ene_rhf
                    energy_err = abs(ene_cur - ene_old)
                    grad_norm = float(np.linalg.norm(
                        scf.hf.get_grad(C_occ, mo_occ, fock_cur)))
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
            err = diis_error(fock_cur, overlap, dm_cur)
            err_norm = scipy.linalg.norm(err)
            fock_physical = fock_cur
            iteration_scf += 1
            is_max_iteration = iteration_scf >= max_iteration
            if jk_mode == "rijcosx":
                is_converged = energy_err < tol and grad_norm < tol
                grad_field = f" grad={grad_norm:.3e}"
            else:
                is_converged = energy_err < tol and dm_err < tol and err_norm < tol
                grad_field = ""
            iteration_record = _stage_record(
                "iteration", iteration_started, iteration_rss,
            )
            _emit_progress(
                f"SCF iteration {iteration_scf}/{max_iteration} "
                f"energy={ene_rhf:.12f} Ha "
                f"delta_energy={delta_energy} "
                f"delta_dm={delta_dm} "
                f"diis_err={err_norm:.3e}"
                f"{grad_field} "
                f"{diis_record} {eigh_record} {part_times} {iteration_record}",
                progress_log,
                molecule_name,
            )

        if is_converged:
            _emit_progress(
                f"SCF converged at iteration {iteration_scf}",
                progress_log,
                molecule_name,
            )
            if jk_mode == "rijcosx":
                _emit_progress("SCF final energy", progress_log, molecule_name)
                started = time.perf_counter()
                rss_before = _rss_gib()
                j_final = self.build_df_j(dm_old, auxbasis)
                j_record = _stage_record("j", started, rss_before)
                started = time.perf_counter()
                rss_before = _rss_gib()
                k_final = self.build_sgx_k(
                    dm_old, auxbasis, final_grid_rad, final_grid_ang, overlap_fit)
                k_record = _stage_record("k", started, rss_before)
                fock_final = hcore + j_final - 0.5 * k_final
                ene_rhf = ene_nuc + 0.5 * np.einsum(
                    "mn,mn->", hcore + fock_final, dm_old)
                _emit_progress(
                    f"SCF final energy={ene_rhf:.12f} Ha {j_record} {k_record} "
                    f"{self._k_grid_note()}",
                    progress_log,
                    molecule_name,
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
        "parallel_workers", "max_memory_mb",
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

    torch_runtime = configure_torch_cpu()
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
    solver = Hartree_Fock(
        mol,
        max_memory=max_memory_mb,
        parallel_backend=parallel_backend,
        parallel_workers=parallel_workers,
        parallel_chunk_size=parallel_chunk_size,
        parallel_min_shells=parallel_min_shells,
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
    evals, evecs = stage(
        "eigh",
        lambda: eigh(fock, overlap),
        lambda value: _gib(value[0]) + _gib(value[1]),
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
        **torch_runtime,
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

    configure_torch_cpu()
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
