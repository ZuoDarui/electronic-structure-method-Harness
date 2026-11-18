import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf.dft import gen_grid, radi

# Memory budgets (bytes). The node has ~256 GB; keep comfortable margins.
_INT3C_STORE_THRESH = 60.0e9   # store full (mn|P) if it fits, else recompute blocked
_AO_STORE_THRESH = 8.0e9       # store full grid AO values if they fit
_A_BATCH_CAP = 4.0e9           # per-batch memory for int1e_grids (g, nao, nao)
_AO_BATCH_CAP = 1.0e9          # per-batch memory for AO evaluation when not stored

_AUXBASIS_ALIASES = {
    "weigend": "def2-universal-jfit",
}


# --------------------------------------------------------------------------- #
# Auxiliary (fitting) molecule for RI-J
# --------------------------------------------------------------------------- #
def _build_auxmol(mol, auxbasis):
    name = _AUXBASIS_ALIASES.get(str(auxbasis).lower(), str(auxbasis))
    candidates = [name, "def2-universal-jfit", "def2universaljfit", str(auxbasis)]
    last_err = None
    for cand in candidates:
        try:
            auxmol = gto.M(
                atom=mol._atom,
                basis=cand,
                unit="Bohr",
                charge=0,
                spin=0,
                cart=mol.cart,
                verbose=0,
            )
            return auxmol
        except Exception as err:  # unknown basis name -> try the next alias
            last_err = err
            continue
    raise last_err


def _aux_shell_blocks(auxmol, nao, cap):
    """Group auxiliary shells into contiguous blocks whose 3c tensor fits `cap`."""
    ao_loc = auxmol.ao_loc_nr()
    per = max(1, int(cap / (nao * nao * 8)))
    blocks = []
    sh = 0
    nbas = auxmol.nbas
    while sh < nbas:
        start = sh
        a0 = ao_loc[sh]
        while sh < nbas and (ao_loc[sh + 1] - a0) <= per:
            sh += 1
        if sh == start:  # a single shell already exceeds the cap; take it alone
            sh = start + 1
        a1 = ao_loc[sh]
        blocks.append((start, sh, int(a0), int(a1)))
    return blocks


class JBuilder:
    """Density-fitted Coulomb: J_mn = sum_PQ (mn|P) [(P|Q)^-1] (Q|ls) D_ls."""

    def __init__(self, mol, auxmol, int2c):
        self.pmol = mol + auxmol  # concatenated molecule for int3c2e
        self.nbas = mol.nbas
        self.aux_nbas = auxmol.nbas
        self.nao = mol.nao_nr()
        self.naux = auxmol.nao_nr()
        self.cho = scipy.linalg.cho_factor(int2c)

        size = self.nao * self.nao * self.naux * 8.0
        self.stored = size < _INT3C_STORE_THRESH
        if self.stored:
            int3c = self.pmol.intor(
                "int3c2e",
                shls_slice=(0, self.nbas, 0, self.nbas,
                            self.nbas, self.nbas + self.aux_nbas),
                aosym="s1",
            )
            self.M2 = int3c.reshape(self.nao * self.nao, self.naux)
        else:
            self.M2 = None
            self.blocks = _aux_shell_blocks(auxmol, self.nao, _INT3C_STORE_THRESH)

    def _int3c_block(self, p0, p1):
        return self.pmol.intor(
            "int3c2e",
            shls_slice=(0, self.nbas, 0, self.nbas,
                        self.nbas + p0, self.nbas + p1),
            aosym="s1",
        )

    def get_j(self, D):
        Dv = D.reshape(-1)
        if self.stored:
            gamma = self.M2.T @ Dv
            x = scipy.linalg.cho_solve(self.cho, gamma)
            return (self.M2 @ x).reshape(self.nao, self.nao)

        gamma = np.zeros(self.naux)
        for (p0, p1, a0, a1) in self.blocks:
            blk = self._int3c_block(p0, p1).reshape(self.nao * self.nao, -1)
            gamma[a0:a1] = blk.T @ Dv
        x = scipy.linalg.cho_solve(self.cho, gamma)
        J = np.zeros((self.nao, self.nao))
        for (p0, p1, a0, a1) in self.blocks:
            blk = self._int3c_block(p0, p1).reshape(self.nao * self.nao, -1)
            J += (blk @ x[a0:a1]).reshape(self.nao, self.nao)
        return J


# --------------------------------------------------------------------------- #
# Seminumerical exchange (COSX) with overlap fitting
# --------------------------------------------------------------------------- #
def _build_grid(mol, rad_ang):
    rad, ang = int(rad_ang[0]), int(rad_ang[1])
    grids = gen_grid.Grids(mol)
    grids.atom_grid = {mol.atom_pure_symbol(i): (rad, ang)
                       for i in range(mol.natm)}
    grids.radi_method = radi.treutler_ahlrichs
    grids.prune = gen_grid.nwchem_prune
    grids.build(with_non0tab=True)
    return grids


class KBuilder:
    """K via overlap-fitted seminumerical exchange on a quadrature grid."""

    def __init__(self, mol, grids, S):
        self.mol = mol
        self.coords = np.asarray(grids.coords)
        self.weights = np.asarray(grids.weights)
        self.nao = mol.nao_nr()
        self.ngrid = self.coords.shape[0]

        self.a_batch = max(1, int(_A_BATCH_CAP / (self.nao * self.nao * 8)))
        self.store_ao = (self.ngrid * self.nao * 8.0) < _AO_STORE_THRESH
        self.ao_batch = max(1, int(_AO_BATCH_CAP / (self.nao * 8)))

        # Numerical overlap  S_num = ao^T diag(w) ao, then Q = S_num^{-1} S.
        Snum = np.zeros((self.nao, self.nao))
        if self.store_ao:
            self.ao = np.asarray(mol.eval_gto("GTOval", self.coords))
            Snum = (self.ao * self.weights[:, None]).T @ self.ao
        else:
            self.ao = None
            for g0 in range(0, self.ngrid, self.ao_batch):
                g1 = min(self.ngrid, g0 + self.ao_batch)
                aob = mol.eval_gto("GTOval", self.coords[g0:g1])
                Snum += (aob * self.weights[g0:g1, None]).T @ aob
        self.Q = scipy.linalg.solve(Snum, S)

    def _ao_slice(self, g0, g1):
        if self.store_ao:
            return self.ao[g0:g1]
        return self.mol.eval_gto("GTOval", self.coords[g0:g1])

    def get_k(self, D):
        QD = self.Q @ D
        K = np.zeros((self.nao, self.nao))
        for g0 in range(0, self.ngrid, self.a_batch):
            g1 = min(self.ngrid, g0 + self.a_batch)
            aob = self._ao_slice(g0, g1)
            wb = self.weights[g0:g1]
            Fb = aob @ QD                                   # F[g, s]
            Ab = self.mol.intor("int1e_grids",
                                grids=self.coords[g0:g1])   # (g, n, s)
            gv = np.matmul(Ab, Fb[:, :, None])[:, :, 0]     # sum_s A[g,n,s] F[g,s]
            K += (aob * wb[:, None]).T @ gv                 # sum_g w ao[g,m] gv[g,n]
        return 0.5 * (K + K.T)


# --------------------------------------------------------------------------- #
# DIIS (commutator / Pulay, error in an orthonormal basis)
# --------------------------------------------------------------------------- #
class DIIS:
    def __init__(self, space, X):
        self.space = space
        self.X = X
        self.F = []
        self.E = []

    def update(self, S, D, F):
        err = self.X.T @ (F @ D @ S - S @ D @ F) @ self.X
        self.F.append(F.copy())
        self.E.append(err.ravel())
        if len(self.F) > self.space:
            self.F.pop(0)
            self.E.pop(0)
        n = len(self.F)
        if n == 1:
            return F
        B = np.empty((n + 1, n + 1))
        B[-1, :] = -1.0
        B[:, -1] = -1.0
        B[-1, -1] = 0.0
        for i in range(n):
            for j in range(i, n):
                B[i, j] = B[j, i] = self.E[i] @ self.E[j]
        rhs = np.zeros(n + 1)
        rhs[-1] = -1.0
        try:
            c = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            c, *_ = np.linalg.lstsq(B, rhs, rcond=None)
        Fnew = np.zeros_like(F)
        for i in range(n):
            Fnew += c[i] * self.F[i]
        return Fnew


# --------------------------------------------------------------------------- #
# SCF driver
# --------------------------------------------------------------------------- #
def _run_scf(D, hcore, S, enuc, nocc, get_j, get_k, X,
             conv_tol, max_cycle=200, diis_space=8):
    diis = DIIS(diis_space, X)
    e_old = 0.0
    converged = False
    D_cur = D
    E = 0.0
    for cyc in range(max_cycle):
        J = get_j(D_cur)
        K = get_k(D_cur)
        F = hcore + J - 0.5 * K
        E = enuc + 0.5 * np.einsum("ij,ij->", D_cur, hcore + F)
        comm = F @ D_cur @ S - S @ D_cur @ F
        gnorm = np.linalg.norm(X.T @ comm @ X)
        if cyc > 0 and abs(E - e_old) < conv_tol and gnorm < conv_tol:
            converged = True
            e_old = E
            break
        e_old = E
        Fd = diis.update(S, D_cur, F)
        _, C = scipy.linalg.eigh(Fd, S)
        Cocc = C[:, :nocc]
        D_cur = 2.0 * (Cocc @ Cocc.T)
    return E, D_cur, converged


def solve(mol, method):
    S = mol.intor_symmetric("int1e_ovlp")
    hcore = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    enuc = mol.energy_nuc()
    nocc = mol.nelectron // 2
    conv_tol = float(method.get("conv_tol", 1e-8))

    # Symmetric orthonormalizer X = S^{-1/2} (used for DIIS error metric).
    sval, svec = np.linalg.eigh(S)
    X = (svec * (1.0 / np.sqrt(sval))) @ svec.T

    # RI-J machinery.
    auxmol = _build_auxmol(mol, method["auxbasis"])
    int2c = auxmol.intor("int2c2e")
    jb = JBuilder(mol, auxmol, int2c)

    # COSX on the SCF grid.
    scf_grid = _build_grid(mol, method["grid"])
    kb = KBuilder(mol, scf_grid, S)

    # Core-Hamiltonian initial guess (converges to the unique closed-shell state).
    _, C0 = scipy.linalg.eigh(hcore, S)
    Cocc = C0[:, :nocc]
    D = 2.0 * (Cocc @ Cocc.T)

    E, D, converged = _run_scf(D, hcore, S, enuc, nocc,
                               jb.get_j, kb.get_k, X, conv_tol)

    final_mode = method.get("final_energy", "single_point")
    if final_mode == "reconverge":
        final_grid = _build_grid(mol, method["final_grid"])
        kb2 = KBuilder(mol, final_grid, S)
        E, D, converged = _run_scf(D, hcore, S, enuc, nocc,
                                   jb.get_j, kb2.get_k, X, conv_tol)
    else:  # "single_point": rebuild J and K once on the fine grid, no reconvergence
        final_grid = _build_grid(mol, method["final_grid"])
        kb2 = KBuilder(mol, final_grid, S)
        Jf = jb.get_j(D)
        Kf = kb2.get_k(D)
        Ff = hcore + Jf - 0.5 * Kf
        E = enuc + 0.5 * np.einsum("ij,ij->", D, hcore + Ff)

    return {"e_tot": float(E), "converged": bool(converged)}
