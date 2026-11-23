"""Restricted Hartree--Fock for the supplied NumPy-only accuracy contract."""

import numpy as np


def _require_numpy(backend):
    if backend != 'numpy':
        raise ValueError("The supplied CPU-only contract requires backend='numpy'.")


def _sym(a):
    return 0.5 * (a + a.T)


def _integrals(problem):
    mode = problem['metadata']['integral_mode']
    if mode == 'exact':
        return mode, np.asarray(problem['eri'], dtype=np.float64)
    if mode == 'df':
        return mode, np.asarray(problem['df_factors'], dtype=np.float64)
    raise ValueError('integral_mode must be exact or df')


def _jk(mode, integrals, density, block_size):
    d = np.asarray(density, dtype=np.float64)
    if mode == 'exact':
        j = np.einsum('pqrs,rs->pq', integrals, d, optimize=True)
        k = np.einsum('prqs,rs->pq', integrals, d, optimize=True)
    else:
        j = np.zeros_like(d)
        k = np.zeros_like(d)
        for start in range(0, integrals.shape[0], block_size):
            b = integrals[start:start + block_size]
            charge = np.einsum('Ppq,pq->P', b, d, optimize=True)
            j += np.einsum('P,Ppq->pq', charge, b, optimize=True)
            bd = b @ d
            k += np.einsum('Pps,Pqs->pq', bd, b, optimize=True)
    return _sym(j), _sym(k)


def build_jk(problem, density, backend):
    """Contract exact or density-fitted integrals with any symmetric density."""
    _require_numpy(backend)
    mode, integrals = _integrals(problem)
    return _jk(mode, integrals, density, 32)


def _pulay(focks, errors):
    m = len(focks)
    if m < 2:
        return focks[-1]
    vectors = np.stack(errors)
    gram = vectors @ vectors.T
    scale = float(np.max(np.diag(gram)))
    if scale <= 0.0:
        return focks[-1]

    system = np.empty((m + 1, m + 1), dtype=np.float64)
    system[:m, :m] = gram / scale
    system[:m, m] = -1.0
    system[m, :m] = -1.0
    system[m, m] = 0.0
    rhs = np.zeros(m + 1, dtype=np.float64)
    rhs[m] = -1.0
    try:
        # Truncation handles nearly dependent commutator histories.
        weights = np.linalg.lstsq(system, rhs, rcond=1e-12)[0][:m]
    except np.linalg.LinAlgError:
        return focks[-1]
    total = float(np.sum(weights))
    if not np.all(np.isfinite(weights)) or abs(total) < 1e-12:
        return focks[-1]
    weights = weights / total
    if np.max(np.abs(weights)) > 1e4:
        return focks[-1]
    result = np.zeros_like(focks[-1])
    for weight, fock in zip(weights, focks):
        result += weight * fock
    return _sym(result)


def _canonicalize(coeff, fock, nocc):
    """Diagonalize occupied and virtual blocks without changing occupations."""
    levels = []
    columns = []
    for lo, hi in ((0, nocc), (nocc, coeff.shape[1])):
        if lo == hi:
            continue
        subspace = coeff[:, lo:hi]
        energy, rotation = np.linalg.eigh(_sym(subspace.T @ fock @ subspace))
        levels.append(energy)
        columns.append(subspace @ rotation)
    return np.concatenate(levels), np.concatenate(columns, axis=1)


def solve(problem, options, backend):
    """Solve closed-shell RHF with orthogonalization and Pulay DIIS."""
    _require_numpy(backend)
    conv_tol = float(options.get('conv_tol', 1e-11))
    residual_tol = float(options.get('residual_tol', 1e-9))
    max_cycle = int(options.get('max_cycle', 150))
    diis_space = int(options.get('diis_space', 8))
    block_size = int(options.get('df_block_size', 32))
    memory_mb = float(options.get('memory_limit_mb', 2048))
    if max_cycle < 1 or diis_space < 0 or block_size < 1 or memory_mb <= 0:
        raise ValueError('Invalid iteration, DIIS, block-size, or memory option')
    if conv_tol < 0 or residual_tol < 0:
        raise ValueError('Convergence tolerances must be nonnegative')

    s = _sym(np.asarray(problem['overlap'], dtype=np.float64))
    h = _sym(np.asarray(problem['hcore'], dtype=np.float64))
    enuc = float(problem['enuc'])
    nelectron = int(problem['nelectron'])
    n = h.shape[0]
    if nelectron <= 0 or nelectron % 2 or nelectron > 2 * n:
        raise ValueError('Invalid closed-shell electron count')
    nocc = nelectron // 2
    occupations = np.zeros(n, dtype=np.float64)
    occupations[:nocc] = 2.0

    overlap_values, overlap_vectors = np.linalg.eigh(s)
    if np.any(overlap_values <= 0.0):
        raise ValueError('Overlap must be positive definite')
    x = overlap_vectors / np.sqrt(overlap_values)[None, :]
    mode, integrals = _integrals(problem)
    if mode == 'df':
        # Reserve space for resident integrals, dense matrices and DIIS;
        # reduce auxiliary blocking to limit additional temporary storage.
        budget = int(memory_mb * 1024 * 1024)
        resident = integrals.nbytes + (16 + 2 * min(diis_space, max_cycle)) * 8 * n * n
        available = max(0, budget - resident)
        block_size = min(block_size, max(1, available // (16 * n * n + 8)))

    def evaluate(density):
        j, k = _jk(mode, integrals, density, block_size)
        fock = _sym(h + j - 0.5 * k)
        energy = float(0.5 * np.einsum('pq,qp->', density, h + fock) + enuc)
        residual = fock @ density @ s - s @ density @ fock
        residual_norm = float(np.linalg.norm(residual))
        fock_orth = _sym(x.T @ fock @ x)
        error_orth = x.T @ residual @ x
        return fock, energy, residual_norm, fock_orth, error_orth

    _, vectors = np.linalg.eigh(_sym(x.T @ h @ x))
    coeff = x @ vectors
    density = (coeff * occupations) @ coeff.T
    fock, energy, residual_norm, fock_orth, error_orth = evaluate(density)
    focks = []
    errors = []
    history = []
    converged = False

    for cycle in range(1, max_cycle + 1):
        old_energy = energy
        if diis_space >= 2:
            focks.append(fock_orth.copy())
            errors.append(error_orth.ravel().copy())
            if len(focks) > diis_space:
                focks.pop(0)
                errors.pop(0)
            effective_fock = _pulay(focks, errors)
        else:
            effective_fock = fock_orth

        _, vectors = np.linalg.eigh(effective_fock)
        coeff = x @ vectors
        density = (coeff * occupations) @ coeff.T
        fock, energy, residual_norm, fock_orth, error_orth = evaluate(density)
        history.append({
            'cycle': cycle,
            'energy': energy,
            'residual_norm': residual_norm,
        })

        if (abs(energy - old_energy) <= conv_tol
                and residual_norm <= residual_tol
                and float(np.linalg.norm(error_orth)) <= residual_tol):
            levels, canonical_coeff = _canonicalize(coeff, fock, nocc)
            # A stationary density must also occupy the lowest physical levels.
            if nocc < n and levels[nocc - 1] > levels[nocc]:
                focks.clear()
                errors.clear()
                continue
            coeff = canonical_coeff
            converged = True
            break

    if not converged:
        _, coeff = _canonicalize(coeff, fock, nocc)

    # These subspace rotations preserve the density, rather than taking an
    # uncounted SCF step. Rebuild to make all returned quantities consistent.
    density = (coeff * occupations) @ coeff.T
    fock, energy, residual_norm, fock_orth, error_orth = evaluate(density)
    mo_energy = np.linalg.eigvalsh(fock_orth)
    history[-1]['energy'] = energy
    history[-1]['residual_norm'] = residual_norm
    converged = bool(
        converged
        and abs(energy - old_energy) <= conv_tol
        and residual_norm <= residual_tol
        and float(np.linalg.norm(error_orth)) <= residual_tol
    )
    return {
        'energy': energy,
        'density': density,
        'mo_coeff': coeff,
        'mo_energy': mo_energy,
        'occupations': occupations,
        'converged': converged,
        'history': history,
    }
