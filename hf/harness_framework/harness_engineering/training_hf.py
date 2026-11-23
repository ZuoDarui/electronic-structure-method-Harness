"""Training RIJCOSX iteration outside the production solver.

``hf.py`` stays the reference for full molecular runs. This module calls its
public ``get_jk`` and replaces the density with ``occupy_density``. The dense
implementation below is the oracle. Generated candidates replace
``harness_engineering.generated.occupy_density`` and do not enter ``solve_rhf``.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.linalg import eigh


TARGET_BASIS_FUNCTIONS = 50000
TARGET_SCF_ITERATIONS = 20
TARGET_WALL_HOURS = 48.0


def project_iteration_hours(
    seconds,
    nao,
    exponent,
    target_nao=TARGET_BASIS_FUNCTIONS,
    iterations=TARGET_SCF_ITERATIONS,
):
    """Project one measured stage to ``iterations`` at ``target_nao``.

    The scale factor is ``(target_nao / nao) ** exponent``, using the exponent
    measured between the last two completed sizes. The result is hours.
    """

    if (seconds is None or exponent is None or seconds <= 0 or nao <= 0
            or target_nao <= 0 or iterations <= 0):
        return None
    try:
        scaled = float(seconds) * (float(target_nao) / float(nao)) ** float(exponent)
        hours = float(iterations) * scaled / 3600.0
    except OverflowError:
        return None
    if not math.isfinite(hours):
        return None
    return hours


def density_electrons(overlap, density):
    """Trace of density times overlap, equal to the electron count."""

    return float(np.einsum("ij,ji->", np.asarray(overlap), np.asarray(density)))


def density_asymmetry(density):
    matrix = np.asarray(density)
    return float(np.max(np.abs(matrix - matrix.T)))


def occupy_density_reference(fock, overlap, nocc):
    """Occupied density from one dense generalized diagonalization.

    This is the same ``eigh(fock, overlap)`` kernel ``solve_rhf`` uses. The
    returned density is ``2 C_occ C_occ^T`` for the lowest ``nocc`` orbitals.
    """

    nocc = int(nocc)
    orbital_energy, coeff = eigh(
        np.asarray(fock, dtype=np.float64),
        np.asarray(overlap, dtype=np.float64),
    )
    if nocc < 0 or nocc > orbital_energy.shape[0]:
        raise ValueError("nocc must lie between 0 and the matrix dimension")
    occupied = np.asarray(coeff)[:, :nocc]
    density = 2.0 * np.einsum("ma,na->mn", occupied, occupied.conj())
    return np.asarray(np.real(density), dtype=np.float64)


def iteration_rijcosx(
    solver,
    dm,
    hcore,
    overlap,
    nocc,
    nelectron,
    auxbasis,
    df_tol,
    grid_rad,
    grid_ang,
    batch_size,
    cosx_tol,
    ao_tol,
    overlap_fit=True,
):
    """Build one Fock with the reference kernels and update the density.

    J and K come from ``Hartree_Fock.get_jk`` and are timed separately. The
    density update is ``generated.occupy_density``, timed as ``eigh_seconds``.
    The electronic energy belongs to the input density and this Fock.
    """

    from harness_engineering.generated.occupy_density import occupy_density

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
    j_matrix, k_matrix = solver.get_jk(
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
    started = time.perf_counter()
    density_out = np.asarray(
        occupy_density(fock, overlap, nocc), dtype=np.float64)
    eigh_seconds = time.perf_counter() - started
    electronic = 0.5 * np.einsum("mn,mn->", hcore + fock, density_in)
    electrons = density_electrons(overlap, density_out)
    finite = bool(
        np.isfinite(j_matrix).all()
        and np.isfinite(k_matrix).all()
        and np.isfinite(fock).all()
        and np.isfinite(density_out).all()
        and np.isfinite(electronic)
    )
    return {
        "j_matrix": j_matrix,
        "k_matrix": k_matrix,
        "density": density_out,
        "j_seconds": float(timings["j_seconds"]),
        "k_seconds": float(timings["k_seconds"]),
        "eigh_seconds": float(eigh_seconds),
        "j_asymmetry": density_asymmetry(j_matrix),
        "k_asymmetry": density_asymmetry(k_matrix),
        "density_asymmetry": density_asymmetry(density_out),
        "density_electron_abs": abs(float(electrons) - float(nelectron)),
        "electronic_energy": float(electronic),
        "all_finite": finite,
        "shell_pairs": int(getattr(solver, "last_shell_pairs", 0)),
        "screened_pairs": int(getattr(solver, "last_ri_screened_pairs", 0)),
        "cosx_full_tensor_calls": int(getattr(solver, "cosx_full_tensor_calls", 0)),
        "coverage_complete": bool(getattr(solver, "coverage_complete", False)),
    }
