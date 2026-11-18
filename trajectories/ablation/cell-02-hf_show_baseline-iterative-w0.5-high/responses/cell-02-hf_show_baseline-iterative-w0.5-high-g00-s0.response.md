```python
import numpy as np
from scipy.linalg import eigh


def _full_eri_matrix(mol, n):
    """Return (ij|kl) as an (n*n, n*n) matrix, computed with 8-fold symmetry."""
    s8 = np.asarray(mol.intor("int2e", aosym="s8")).ravel()
    npair = n * (n + 1) // 2
    rows, cols = np.tril_indices(npair)
    m = np.empty((npair, npair))
    m[rows, cols] = s8
    m[cols, rows] = s8
    i, j = np.indices((n, n))
    hi = np.maximum(i, j)
    lo = np.minimum(i, j)
    pidx = (hi * (hi + 1) // 2 + lo).ravel()
    return m[np.ix_(pidx, pidx)]


def solve(mol, max_iteration=200, tol=1e-10, max_diis=8):
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    s = mol.intor("int1e_ovlp")
    n = h.shape[0]
    nn = n * n
    nocc = mol.nelectron // 2
    enuc = float(mol.energy_nuc())

    jmat = _full_eri_matrix(mol, n)
    kmat = jmat.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(nn, nn)
    g = jmat - 0.5 * kmat
    del jmat, kmat

    w, v = np.linalg.eigh(s)
    keep = w > 1e-10
    x = v[:, keep] / np.sqrt(w[keep])

    def density_from_fock(f):
        fo = x.T @ f @ x
        _, c = np.linalg.eigh(fo)
        co = x @ c[:, :nocc]
        return 2.0 * co @ co.T

    d = density_from_fock(h)

    errs = []
    focks = []
    e_prev = None
    energy = 0.0
    converged = False

    for _ in range(max_iteration):
        f = h + (g @ d.ravel()).reshape(n, n)
        energy = enuc + 0.5 * float(np.vdot(d, h + f))
        fds = f @ d @ s
        comm = fds - fds.T
        cnorm = float(np.linalg.norm(comm))
        if e_prev is not None and abs(energy - e_prev) < tol and cnorm < tol:
            converged = True
            break
        e_prev = energy

        errs.append(x.T @ comm @ x)
        focks.append(f)
        if len(errs) > max_diis:
            errs.pop(0)
            focks.pop(0)
        k = len(errs)
        if k >= 2:
            e_arr = np.array([e.ravel() for e in errs])
            b = np.empty((k + 1, k + 1))
            b[:k, :k] = e_arr @ e_arr.T
            b[k, :k] = 1.0
            b[:k, k] = 1.0
            b[k, k] = 0.0
            rhs = np.zeros(k + 1)
            rhs[k] = 1.0
            try:
                coef = np.linalg.solve(b, rhs)[:k]
                if not np.all(np.isfinite(coef)):
                    raise np.linalg.LinAlgError
                f_use = np.tensordot(coef, np.array(focks), axes=1)
            except np.linalg.LinAlgError:
                errs = errs[-1:]
                focks = focks[-1:]
                f_use = f
        else:
            f_use = f
        d = density_from_fock(f_use)

    if not converged:
        _, c = eigh(f, s)
        co = c[:, :nocc]
        d = 2.0 * co @ co.T
        for _ in range(50):
            f = h + (g @ d.ravel()).reshape(n, n)
            energy = enuc + 0.5 * float(np.vdot(d, h + f))
            fds = f @ d @ s
            cnorm = float(np.linalg.norm(fds - fds.T))
            if e_prev is not None and abs(energy - e_prev) < tol and cnorm < tol:
                converged = True
                break
            e_prev = energy
            d = 0.5 * d + 0.5 * density_from_fock(f)

    return {"e_tot": float(energy), "converged": bool(converged)}
```
