"""Auditable, float64 restricted Hartree--Fock with NumPy/CuPy backends.

PySCF supplies Gaussian AO integrals only. SCF, DIIS, occupation, J/K
contractions and diagonalization are implemented here. CUDA calculations
keep integrals, density, Fock, orbitals and DIIS vectors on the device;
only scalar diagnostics and the final result return to the host.

Exact mode stores O(nao**4) ERIs and targets small reference calculations.
DF mode stores O(naux * nao**2) factors and blocks exchange intermediates.
Neither mode is an out-of-core or linear-scaling implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import Callable

import numpy as np
from pyscf import df, gto, lib


class BackendUnavailable(RuntimeError):
    """The requested CUDA backend cannot execute a float64 calculation."""


def _backend(name: str):
    if name not in {"numpy", "cupy", "auto"}:
        raise ValueError("backend must be numpy, cupy or auto")
    if name == "numpy":
        return np, {"requested_backend": name, "backend": "numpy", "gpu_executed": False}
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CUDA device count is zero")
        # Importability is insufficient: force a real allocation and kernel.
        probe = cp.asarray([1.0, 2.0], dtype=cp.float64)
        if float(cp.sum(probe).item()) != 3.0:
            raise RuntimeError("CUDA float64 probe failed")
        cp.cuda.get_current_stream().synchronize()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        device_name = props["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        return cp, {
            "requested_backend": name, "backend": "cupy", "gpu_executed": True,
            "cupy_version": cp.__version__, "device_name": str(device_name),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        }
    except Exception as exc:
        message = f"CuPy/CUDA backend unavailable: {type(exc).__name__}: {exc}"
        if name == "cupy":
            raise BackendUnavailable(message) from exc
        return np, {
            "requested_backend": name, "backend": "numpy", "gpu_executed": False,
            "fallback_reason": message,
        }


@dataclass
class RHFResult:
    energy: float
    density: np.ndarray
    mo_coeff: np.ndarray
    mo_energy: np.ndarray
    orbital_occupations: np.ndarray
    converged: bool
    history: list[dict]
    iterations: int
    residual_norm: float
    backend: str
    integral_mode: str
    metadata: dict

    def summary(self) -> dict:
        """JSON-safe scalar evidence; matrices can be saved separately as NPZ."""
        return {
            "energy_hartree": self.energy, "converged": self.converged,
            "iterations": self.iterations, "residual_norm": self.residual_norm,
            "backend": self.backend, "integral_mode": self.integral_mode,
            "metadata": self.metadata, "history": self.history,
        }


class _DIIS:
    """Pulay extrapolation in an orthonormal AO basis, stored on the backend."""

    def __init__(self, xp, space: int):
        self.xp, self.space = xp, space
        self.focks, self.errors = [], []

    def update(self, fock, error):
        if self.space < 2:
            return fock
        xp = self.xp
        self.focks.append(fock.copy())
        self.errors.append(error.ravel().copy())
        self.focks = self.focks[-self.space:]
        self.errors = self.errors[-self.space:]
        size = len(self.focks)
        if size < 2:
            return fock
        errors = xp.stack(self.errors)
        gram = errors @ errors.T
        scale = float(xp.max(xp.abs(gram)).item())
        if scale < 1e-28:
            return fock
        system = xp.empty((size + 1, size + 1), dtype=xp.float64)
        system[:size, :size] = gram / scale
        system[-1, :size] = system[:size, -1] = -1.0
        system[-1, -1] = 0.0
        rhs = xp.zeros(size + 1, dtype=xp.float64)
        rhs[-1] = -1.0
        # lstsq handles nearly redundant SCF errors without a singular solve.
        coeff = xp.linalg.lstsq(system, rhs, rcond=1e-12)[0][:size]
        if not bool(xp.isfinite(coeff).all().item()) or float(xp.max(xp.abs(coeff)).item()) > 1e4:
            self.focks, self.errors = self.focks[-1:], self.errors[-1:]
            return fock
        return xp.einsum("i,ipq->pq", coeff, xp.stack(self.focks), optimize=True)


class GPUFriendlyRHF:
    """Closed-shell molecular RHF; ``D = 2 C_occ C_occ.T`` throughout.

    ``F = H + J(D) - K(D)/2`` and
    ``E = Tr[D (H + F)]/2 + E_nuc``.

    Explicit ``backend='cupy'`` fails when CUDA is missing. Only ``auto``
    may fall back, with the reason retained in result metadata. Inputs use
    PySCF's Mole conventions (including charge and Angstrom/Bohr units).
    """

    def __init__(
        self, mol, backend="numpy", integral_mode="exact", auxbasis="weigend",
        conv_tol=1e-10, residual_tol=1e-8, max_cycle=100, diis_space=8,
        df_block_size=32, memory_limit_mb=2048, lindep_tol=1e-10,
        callback: Callable[[dict], None] | None = None,
    ):
        if not getattr(mol, "_built", False):
            raise ValueError("mol must be a built PySCF Mole")
        if mol.spin != 0 or mol.nelectron <= 0 or mol.nelectron % 2:
            raise ValueError("RHF requires a positive, even electron count and spin=0")
        if getattr(mol, "_ecp", {}) or getattr(mol, "_pseudo", {}):
            raise ValueError("ECP/pseudopotentials are not supported by this integral setup")
        if integral_mode not in {"exact", "df"}:
            raise ValueError("integral_mode must be exact or df")
        for name, value in (("conv_tol", conv_tol), ("residual_tol", residual_tol),
                            ("memory_limit_mb", memory_limit_mb), ("lindep_tol", lindep_tol)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value, minimum in (("max_cycle", max_cycle, 1), ("diis_space", diis_space, 0),
                                     ("df_block_size", df_block_size, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.mol, self.integral_mode, self.auxbasis = mol, integral_mode, auxbasis
        self.conv_tol, self.residual_tol = float(conv_tol), float(residual_tol)
        self.max_cycle, self.diis_space = max_cycle, diis_space
        self.df_block_size, self.memory_limit_mb = df_block_size, float(memory_limit_mb)
        self.callback = callback
        self.nocc, self.nao = mol.nelectron // 2, mol.nao_nr()
        self.xp, self.metadata = _backend(backend)
        xp = self.xp
        self.backend = self.metadata["backend"]
        self.metadata.update({
            "precision": "float64", "integrals_backend": "pyscf-cpu",
            "nao": self.nao, "nelectron": mol.nelectron, "charge": mol.charge,
            "spin": mol.spin, "basis": mol.basis, "auxbasis": auxbasis if integral_mode == "df" else None,
            "geometry_bohr": [
                [mol.atom_symbol(i), *map(float, mol.atom_coord(i))] for i in range(mol.natm)
            ],
            "memory_limit_mb": self.memory_limit_mb,
            "memory_policy": "conservative estimated working set; not a process RSS hard limit",
            "df_block_size": df_block_size, "diis_space": diis_space,
            "conv_tol": self.conv_tol, "residual_tol": self.residual_tol,
            "max_cycle": max_cycle,
        })
        # Guard the large integral allocation before calling the integral engine.
        matrix_bytes = 8 * self.nao**2
        fixed_bytes = matrix_bytes * (24 + 3 * diis_space)
        auxmol = None
        if integral_mode == "exact":
            estimate = 2 * 8 * self.nao**4 + fixed_bytes
        else:
            auxmol = df.addons.make_auxmol(mol, auxbasis)
            naux = auxmol.nao_nr()
            # Packed factors + full factors + copies, metric and block workspace.
            estimate = 4 * 8 * naux * self.nao**2 + 3 * 8 * naux**2 + fixed_bytes
            estimate += 3 * matrix_bytes * min(df_block_size, naux)
        self.metadata["estimated_working_set_mb"] = estimate / 1e6
        if estimate > self.memory_limit_mb * 1e6:
            raise MemoryError(
                f"Estimated working set {estimate / 1e6:.1f} MB exceeds memory_limit_mb="
                f"{memory_limit_mb:g}; use df mode, a smaller basis or a larger explicit budget"
            )
        if self.backend == "cupy":
            free_bytes, _ = xp.cuda.runtime.memGetInfo()
            if estimate > 0.8 * free_bytes:
                raise MemoryError("Estimated working set exceeds 80% of currently free CUDA memory")
        self.overlap = xp.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=xp.float64)
        self.hcore = xp.asarray(
            mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc"),
            dtype=xp.float64,
        )
        self.enuc = float(mol.energy_nuc())
        if not bool(xp.isfinite(self.hcore).all().item()) or not math.isfinite(self.enuc):
            raise ValueError("Non-finite one-electron integrals or nuclear energy")
        eigenvalues, eigenvectors = xp.linalg.eigh(self.overlap)
        if not bool(xp.isfinite(eigenvalues).all().item()) or float(eigenvalues[0].item()) < -lindep_tol:
            raise ValueError("Overlap matrix is not positive semidefinite")
        keep = eigenvalues > lindep_tol
        self.rank = int(xp.count_nonzero(keep).item())
        if self.rank < self.nocc:
            raise ValueError("Overlap rank is smaller than occupied orbital count")
        self.orthogonalizer = eigenvectors[:, keep] / xp.sqrt(eigenvalues[keep])
        self.metadata.update({"overlap_rank": self.rank, "lindep_tol": lindep_tol})
        self.eri = self.df_factors = None
        if integral_mode == "exact":
            self.eri = xp.asarray(mol.intor("int2e", aosym="s1").reshape((self.nao,) * 4))
        else:
            packed = df.incore.cholesky_eri(
                mol, auxmol=auxmol, max_memory=self.memory_limit_mb, verbose=0,
            )
            factors = lib.unpack_tril(packed).reshape(-1, self.nao, self.nao)
            self.df_factors = xp.asarray(factors, dtype=xp.float64)
            self.metadata["naux"] = int(self.df_factors.shape[0])
        self.synchronize()

    def synchronize(self):
        if self.backend == "cupy":
            self.xp.cuda.get_current_stream().synchronize()

    def to_numpy(self, array) -> np.ndarray:
        return np.asarray(array) if self.backend == "numpy" else self.xp.asnumpy(array)

    def _validated_density(self, density):
        xp = self.xp
        raw = xp.asarray(density)
        if xp.iscomplexobj(raw):
            raise ValueError("Only real densities are supported")
        density = raw.astype(xp.float64, copy=False)
        if density.shape != (self.nao, self.nao):
            raise ValueError(f"Density shape must be {(self.nao, self.nao)}")
        if not bool(xp.isfinite(density).all().item()):
            raise ValueError("Density must be finite")
        if float(xp.max(xp.abs(density - density.T)).item()) > 1e-10:
            raise ValueError("Density must be symmetric")
        return density

    def _build_jk(self, density):
        xp = self.xp
        if self.integral_mode == "exact":
            coulomb = xp.einsum("pqrs,rs->pq", self.eri, density, optimize=True)
            exchange = xp.einsum("prqs,rs->pq", self.eri, density, optimize=True)
        else:
            coulomb = xp.zeros_like(density)
            exchange = xp.zeros_like(density)
            for start in range(0, self.df_factors.shape[0], self.df_block_size):
                factors = self.df_factors[start:start + self.df_block_size]
                flat = factors.reshape(len(factors), -1)
                charges = flat @ density.ravel()
                coulomb += (charges @ flat).reshape(self.nao, self.nao)
                intermediate = factors @ density
                exchange += xp.einsum("Pps,Pqs->pq", intermediate, factors, optimize=True)
        return (coulomb + coulomb.T) * 0.5, (exchange + exchange.T) * 0.5

    def build_jk(self, density):
        """Return J and K for an arbitrary real symmetric spin-summed density."""
        return self._build_jk(self._validated_density(density))

    def _diagonalize(self, fock):
        xp, x = self.xp, self.orthogonalizer
        transformed = x.T @ fock @ x
        energies, vectors = xp.linalg.eigh((transformed + transformed.T) * 0.5)
        return energies, x @ vectors

    def _density(self, coefficients):
        occupied = coefficients[:, :self.nocc]
        return 2.0 * (occupied @ occupied.T)

    def _fock(self, density):
        coulomb, exchange = self._build_jk(density)
        return self.hcore + coulomb - 0.5 * exchange

    def _error(self, fock, density):
        x, s = self.orthogonalizer, self.overlap
        commutator = fock @ density @ s - s @ density @ fock
        return x.T @ commutator @ x

    def kernel(self, dm0=None) -> RHFResult:
        """Solve SCF; convergence requires both energy change and residual.

        Every recorded energy uses a freshly rebuilt physical Fock at that
        density, never the DIIS-extrapolated Fock. A final canonicalization
        must also satisfy the residual; max-cycle exhaustion stays failure.
        """
        xp = self.xp
        started = time.perf_counter()
        energies, coefficients = self._diagonalize(self.hcore)
        density = self._density(coefficients) if dm0 is None else self._validated_density(dm0).copy()
        if dm0 is not None:
            electron_error = abs(float(xp.sum(density * self.overlap).item()) - self.mol.nelectron)
            if electron_error > 1e-7:
                raise ValueError("Initial density has incorrect electron count Tr(D S)")
        diis = _DIIS(xp, self.diis_space)
        fock = self._fock(density)
        old_energy = float((0.5 * xp.sum(density * (self.hcore + fock))).item()) + self.enuc
        history, converged = [], False
        for cycle in range(1, self.max_cycle + 1):
            extrapolated = diis.update(fock, self._error(fock, density))
            energies, coefficients = self._diagonalize(extrapolated)
            next_density = self._density(coefficients)
            next_fock = self._fock(next_density)
            energy = float((0.5 * xp.sum(next_density * (self.hcore + next_fock))).item()) + self.enuc
            residual = float(xp.linalg.norm(self._error(next_fock, next_density)).item())
            density_change = float(xp.linalg.norm(next_density - density).item())
            delta = energy - old_energy
            if not all(math.isfinite(v) for v in (energy, residual, density_change, delta)):
                raise FloatingPointError("SCF generated non-finite energy, density or residual")
            record = {
                "cycle": cycle, "energy": energy, "energy_change": delta,
                "density_change": density_change, "residual_norm": residual,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)
            if self.callback is not None:
                self.callback(dict(record))
            density, fock, old_energy = next_density, next_fock, energy
            if abs(delta) < self.conv_tol and residual < self.residual_tol:
                # Canonical orbitals must describe the *same* final density.
                canonical_energies, canonical_coefficients = self._diagonalize(fock)
                canonical_density = self._density(canonical_coefficients)
                canonical_fock = self._fock(canonical_density)
                canonical_energy = float((0.5 * xp.sum(
                    canonical_density * (self.hcore + canonical_fock)
                )).item()) + self.enuc
                canonical_residual = float(xp.linalg.norm(
                    self._error(canonical_fock, canonical_density)
                ).item())
                if (math.isfinite(canonical_energy) and math.isfinite(canonical_residual)
                        and abs(canonical_energy - energy) < self.conv_tol
                        and canonical_residual < self.residual_tol):
                    density, fock, energy, residual = (
                        canonical_density, canonical_fock, canonical_energy, canonical_residual
                    )
                    energies, coefficients = canonical_energies, canonical_coefficients
                    converged = True
                    break
        # Returned orbital energies are Rayleigh quotients of final physical F.
        energies = xp.diag(coefficients.T @ fock @ coefficients)
        occupations = np.zeros(self.rank)
        occupations[:self.nocc] = 2.0
        self.synchronize()
        metadata = dict(self.metadata)
        metadata.update({
            "elapsed_seconds": time.perf_counter() - started,
            "electron_count": float(xp.sum(density * self.overlap).item()),
            "final_energy_change_from_last_history": energy - history[-1]["energy"],
        })
        return RHFResult(
            energy=energy, density=self.to_numpy(density), mo_coeff=self.to_numpy(coefficients),
            mo_energy=self.to_numpy(energies), orbital_occupations=occupations,
            converged=converged, history=history, iterations=len(history), residual_norm=residual,
            backend=self.backend, integral_mode=self.integral_mode, metadata=metadata,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--atom", default=None, help="PySCF geometry string, in Angstrom")
    geometry.add_argument("--xyz", type=Path, help="XYZ file, in Angstrom")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--backend", choices=("numpy", "cupy", "auto"), default="numpy")
    parser.add_argument("--integral-mode", choices=("exact", "df"), default="exact")
    parser.add_argument("--auxbasis", default="weigend")
    parser.add_argument("--conv-tol", type=float, default=1e-10)
    parser.add_argument("--residual-tol", type=float, default=1e-8)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--df-block-size", type=int, default=32)
    parser.add_argument("--memory-limit-mb", type=float, default=2048)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("gpu_rhf_result.json"))
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    lib.num_threads(args.threads)
    if args.xyz:
        # gto.fromfile validates the XYZ format; integral engine remains CPU.
        atom = gto.fromfile(str(args.xyz), format="xyz")
    else:
        atom = args.atom or "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
    mol = gto.M(atom=atom, basis=args.basis, charge=args.charge, spin=0, verbose=0)
    def progress(row):
        print(f"SCF {row['cycle']:3d} E={row['energy']:.12f} Eh "
              f"dE={row['energy_change']:+.3e} residual={row['residual_norm']:.3e}", flush=True)
    solver = GPUFriendlyRHF(
        mol, backend=args.backend, integral_mode=args.integral_mode, auxbasis=args.auxbasis,
        conv_tol=args.conv_tol, residual_tol=args.residual_tol, max_cycle=args.max_cycle,
        df_block_size=args.df_block_size, memory_limit_mb=args.memory_limit_mb, callback=progress,
    )
    print(json.dumps(solver.metadata, ensure_ascii=False), flush=True)
    result = solver.kernel()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.summary(), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"), density=result.density, mo_coeff=result.mo_coeff,
        mo_energy=result.mo_energy, orbital_occupations=result.orbital_occupations,
    )
    print(f"converged={result.converged} E={result.energy:.12f} Eh; evidence={args.output}")
    return 0 if result.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
