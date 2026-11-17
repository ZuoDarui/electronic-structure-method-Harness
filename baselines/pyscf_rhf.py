"""Speed and accuracy baseline: PySCF's own closed-shell RHF driver.

Default PySCF settings (MINAO initial guess, DIIS, conv_tol 1e-9) on the
molecule the harness builds. Any file that exposes
``solve(mol) -> {"e_tot": float, "converged": bool}`` can replace this one via
the ``baseline`` key of a run config.
"""

from __future__ import annotations

from pyscf import scf


def solve(mol):
    mf = scf.RHF(mol)
    mf.verbose = 0
    e_tot = mf.kernel()
    return {"e_tot": float(e_tot), "converged": bool(mf.converged)}
