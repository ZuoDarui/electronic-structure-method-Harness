"""PySCF RIJCOSX with the settings of the cluster reference runs.

Same calculation as ``fieldquantum/hf/pyscf_water32.py`` on the TI-ONE cluster:
``scf.RHF(mol).COSX(auxbasis)`` with density-fitted J, SGX exchange on the
Treutler-Ahlrichs / NWChem-pruned grid of ``hf.matching_cosx_grids``, MINAO
guess, DIIS space 8, and ``conv_tol = conv_tol_grad = method["conv_tol"]``.
SCF converges on ``method["grid"]``. The final energy then depends on
``method["final_energy"]``:

- ``single_point``: J and K are rebuilt once on ``method["final_grid"]`` from
  the converged density (``pyscf_water32.py``).
- ``reconverge``: SCF continues on ``method["final_grid"]`` from the converged
  density until it converges again (``hf.py`` since 2026-11-16, used for C172).
"""

from __future__ import annotations

import numpy as np
from pyscf import scf, sgx
from pyscf.df import df_jk
from pyscf.dft import gen_grid, radi
from pyscf.sgx import sgx_jk


def matching_cosx_grids(mol, grid_rad, grid_ang):
    grids = gen_grid.Grids(mol)
    symbols = {mol.atom_pure_symbol(i) for i in range(mol.natm)}
    grids.atom_grid = {symbol: (int(grid_rad), int(grid_ang)) for symbol in symbols}
    grids.radi_method = radi.treutler_ahlrichs
    grids.prune = gen_grid.nwchem_prune
    grids.build(with_non0tab=True)
    grids.non0tab = grids.make_mask(mol, grids.coords)
    grids.screen_index = grids.non0tab
    return grids


def _rijcosx(mol, method, grid):
    mf = scf.RHF(mol).COSX(auxbasis=method["auxbasis"])
    mf.verbose = 0
    mf.chkfile = None
    mf.direct_scf = False
    mf.conv_tol = float(method["conv_tol"])
    mf.conv_tol_grad = float(method["conv_tol"])
    mf.max_cycle = int(method.get("max_cycle", 200))
    mf.init_guess = "minao"
    mf.diis_space = 8
    mf.with_df.dfj = True
    mf.with_df.optk = True
    mf.with_df.grids_level_i = 1
    mf.with_df.grids_level_f = 1
    original_build = mf.with_df.build

    def build_on_grid(level=None):
        original_build(level)
        mf.with_df.grids = matching_cosx_grids(mol, *grid)

    mf.with_df.build = build_on_grid
    return mf


def solve(mol, method):
    grid = tuple(method["grid"])
    final_grid = tuple(method["final_grid"])
    mf = _rijcosx(mol, method, grid)
    mf.kernel()
    if not mf.converged:
        return {"e_tot": float(mf.e_tot), "converged": False}
    if method["final_energy"] == "reconverge":
        fine = _rijcosx(mol, method, final_grid)
        fine.kernel(dm0=mf.make_rdm1())
        return {"e_tot": float(fine.e_tot), "converged": bool(fine.converged)}
    if method["final_energy"] != "single_point":
        raise ValueError(f"unknown final_energy {method['final_energy']!r}")
    mf.with_df.grids = matching_cosx_grids(mol, *final_grid)
    dm = mf.make_rdm1()
    vj = df_jk.get_j(mf.with_df, dm, hermi=1)
    vk = sgx_jk.get_k_only(mf.with_df, dm, hermi=1)
    hcore = mf.get_hcore()
    fock = hcore + vj - 0.5 * vk
    energy = mol.energy_nuc() + 0.5 * float(np.einsum("ij,ij->", hcore + fock, dm))
    return {"e_tot": float(energy), "converged": True}
