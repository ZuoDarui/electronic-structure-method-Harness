"""Restricted Hartree--Fock for the supplied CPU/NumPy contract."""

import numpy as np


def _check_backend(backend):
    if backend != 'numpy':
        raise ValueError('The supplied CPU-only contract requires the NumPy backend.')


def _sym(a):
    return 0.5 * (a + a.T)


def _jk(problem, density, block_size):
    d = np.asarray(density, dtype=np.float64)
    mode = problem['metadata']['integral_mode']
    if mode == 'exact':
        eri = np.asarray(problem['eri'], dtype=np.float64)
        # Direct contractions avoid a transposed four-index temporary.
        j = np.einsum('pqrs,rs->pq', eri, d, optimize=False)
        k = np.einsum('prqs,rs->pq', eri, d, optimize=False)
    elif mode == 'df':
        factors = np.asarray(problem['df_factors'], dtype=np.float64)
        j = np.zeros_like(d)
        k = np.zeros_like(d)
        for start in range(0, factors.shape[0], block_size):
            b = factors[start:start + block_size]
            charge = np.einsum('Prs,rs->P', b, d, optimize=False)
            j += np.einsum('P,Ppq->pq', charge, b, optimize=False)
            bd = b @ d
            k += np.einsum('Pps,Pqs->pq', bd, b, optimize=False)
    else:
        raise ValueError('Unknown integral mode: ' + str(mode))
    return _sym(j), _sym(k)


def build_jk(problem, density, backend):
    """Build linear Coulomb and exchange matrices for a spin-summed density."""
    _check_backend(backend)
    return _jk(problem, density, 32)


def _pulay(focks, errors):
    # Minimize ||sum_i w_i error_i|| with sum_i w_i = 1.
    # Eliminating the last weight permits a rank-revealing least-squares
    # solve without squaring the condition number in a Gram matrix.
    while len(focks) > 1:
        last = errors[-1].ravel()
        differences = np.column_stack([
            error.ravel() - last for error in errors[:-1]
        ])
        try:
            coefficients = np.linalg.lstsq(
                differences, -last, rcond=1e-12
            )[0]
            weights = np.append(coefficients, 1.0 - coefficients.sum())
            if (np.all(np.isfinite(weights))
                    and np.max(np.abs(weights)) <= 1e4):
                result = focks[-1].copy()
                for weight, fock in zip(coefficients, focks[:-1]):
                    result += weight * (fock - focks[-1])
                return _sym(result)
        except np.linalg.LinAlgError:
            pass
        # Discard old, nearly dependent information rather than using an
        # unstable extrapolation.
        del focks[0]
        del errors[0]
    return focks[-1]


def _state(problem, h, s, d, block_size):
    j, k = _jk(problem, d, block_size)
    f = _sym(h + j - 0.5 * k)
    energy = float(
        0.5 * np.einsum('ij,ji->', d, h + f) + float(problem['enuc'])
    )
    residual = f @ d @ s - s @ d @ f
    return f, energy, residual, float(np.linalg.norm(residual))


def _canonicalize(c, f, nocc):
    # Separate occupied/virtual rotations preserve the last SCF density,
    # including when the iteration limit is reached.
    result = c.copy()
    for lo, hi in ((0, nocc), (nocc, c.shape[1])):
        if hi > lo:
            block = c[:, lo:hi]
            _, rotation = np.linalg.eigh(_sym(block.T @ f @ block))
            result[:, lo:hi] = block @ rotation
    return result


def solve(problem, options, backend):
    """Solve closed-shell RHF with overlap orthogonalization and Pulay DIIS."""
    _check_backend(backend)
    s = np.asarray(problem['overlap'], dtype=np.float64)
    h = np.asarray(problem['hcore'], dtype=np.float64)
    n = h.shape[0]
    nelectron = int(problem['nelectron'])
    if nelectron <= 0 or nelectron % 2 or nelectron > 2 * n:
        raise ValueError('Invalid closed-shell electron count.')
    nocc = nelectron // 2

    conv_tol = float(options.get('conv_tol', 1e-11))
    residual_tol = float(options.get('residual_tol', 1e-9))
    max_cycle = int(options.get('max_cycle', 150))
    diis_space = max(0, int(options.get('diis_space', 8)))
    block_size = max(1, int(options.get('df_block_size', 32)))
    memory_mb = float(options.get('memory_limit_mb', 2048.0))
    if (max_cycle < 1 or conv_tol <= 0 or residual_tol <= 0
            or memory_mb <= 0
            or not np.all(np.isfinite([conv_tol, residual_tol, memory_mb]))):
        raise ValueError('Iteration limit, tolerances and memory must be positive.')

    # Conservatively bound DF scratch storage; input integrals are not copied.
    workspace_bytes = int(0.25 * memory_mb * 1024 * 1024)
    block_cap = max(1, workspace_bytes // (8 * (3 * n * n + 1)))
    block_size = min(block_size, block_cap)

    overlap_values, overlap_vectors = np.linalg.eigh(_sym(s))
    if overlap_values[0] <= 0:
        raise ValueError('Overlap must be positive definite.')
    x = overlap_vectors / np.sqrt(overlap_values)
    _, initial_vectors = np.linalg.eigh(_sym(x.T @ h @ x))
    c = x @ initial_vectors
    occupations = np.zeros(n, dtype=np.float64)
    occupations[:nocc] = 2.0

    history = []
    focks = []
    errors = []
    previous_energy = None
    converged = False

    for cycle in range(1, max_cycle + 1):
        density = (c * occupations) @ c.T
        f, energy, residual, residual_norm = _state(
            problem, h, s, density, block_size
        )
        f_orth = _sym(x.T @ f @ x)
        error = x.T @ residual @ x
        orthogonal_residual_norm = float(np.linalg.norm(error))
        history.append({
            'cycle': cycle,
            'energy': energy,
            'residual_norm': residual_norm,
        })

        energy_ok = (previous_energy is not None
                     and abs(energy - previous_energy) <= conv_tol)
        residual_ok = (residual_norm <= residual_tol
                       and orthogonal_residual_norm <= residual_tol)
        if energy_ok and residual_ok:
            # A stationary non-Aufbau occupation is not an accepted solution.
            occupied = c[:, :nocc]
            occupied_max = np.linalg.eigvalsh(
                _sym(occupied.T @ f @ occupied)
            )[-1]
            virtual_min = np.inf
            if nocc < n:
                virtual = c[:, nocc:]
                virtual_min = np.linalg.eigvalsh(
                    _sym(virtual.T @ f @ virtual)
                )[0]
            if occupied_max <= virtual_min + 1e-12:
                converged = True
                break
            focks.clear()
            errors.clear()

        if cycle == max_cycle:
            break
        previous_energy = energy

        if diis_space > 1:
            focks.append(f_orth)
            errors.append(error)
            if len(focks) > diis_space:
                del focks[0]
                del errors[0]
            effective_fock = _pulay(focks, errors)
        else:
            effective_fock = f_orth
        _, vectors = np.linalg.eigh(_sym(effective_fock))
        c = x @ vectors

    # Canonicalize without taking an additional SCF density-update step.
    # Rebuild the physical quantities to include final rotation roundoff.
    c = _canonicalize(c, f, nocc)
    density = (c * occupations) @ c.T
    f, energy, residual, residual_norm = _state(
        problem, h, s, density, block_size
    )
    mo_energy = np.linalg.eigvalsh(_sym(x.T @ f @ x))
    orthogonal_residual_norm = float(np.linalg.norm(x.T @ residual @ x))
    converged = bool(
        converged
        and previous_energy is not None
        and abs(energy - previous_energy) <= conv_tol
        and residual_norm <= residual_tol
        and orthogonal_residual_norm <= residual_tol
    )
    history[-1]['energy'] = energy
    history[-1]['residual_norm'] = residual_norm

    return {
        'energy': energy,
        'density': density,
        'mo_coeff': c,
        'mo_energy': mo_energy,
        'occupations': occupations,
        'converged': converged,
        'history': history,
    }
