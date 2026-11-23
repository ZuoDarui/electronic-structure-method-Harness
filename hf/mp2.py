"""Primary (canonical, in-core) closed-shell MP2 on top of the custom RHF solver.

The AO two-electron integrals come from PySCF: pyscf.ao2mo transforms them to
the (ia|jb) MO block.  RHF orbitals are rebuilt from the converged density that
Hartree_Fock.solve_rhf returns (it returns only the energy and the density).

E2 = sum_{ijab} (ia|jb) [2 (ia|jb) - (ib|ja)] / (e_i + e_j - e_a - e_b)
"""

import argparse
import json

import numpy as np
import scipy.linalg
from pyscf import ao2mo

from hf import Hartree_Fock, build_molecule_from_xyz


def canonical_orbitals(solver, hcore, overlap, dm):
    """Rebuild the Fock matrix from a converged density and diagonalize it."""
    vj, vk = solver.build_jk(dm)
    fock = hcore + vj - 0.5 * vk
    eps, coeff = scipy.linalg.eigh(fock, overlap)
    return eps, coeff


def estimate_mp2_memory_gib(nocc_active, nvir, copies=3):
    """Memory for the dense (ov|ov) block plus the working copies."""
    return copies * 8.0 * (nocc_active * nvir) ** 2 / 1024 ** 3


def mp2_correlation(mol, coeff, eps, nocc, frozen=0, max_memory_gib=None):
    """Closed-shell MP2 correlation energy from MO coefficients and energies."""
    nmo = coeff.shape[1]
    if not 0 <= frozen < nocc:
        raise ValueError('frozen must satisfy 0 <= frozen < nocc')
    if nocc >= nmo:
        raise ValueError('no virtual orbitals: MP2 correlation energy is undefined')
    o = nocc - frozen
    v = nmo - nocc
    estimate = estimate_mp2_memory_gib(o, v)
    if max_memory_gib is not None and estimate > float(max_memory_gib):
        raise MemoryError(
            f'Estimated MP2 working set exceeds max_memory_gib: '
            f'{estimate:.3f} > {float(max_memory_gib):.3f}'
        )

    c_occ = coeff[:, frozen:nocc]
    c_vir = coeff[:, nocc:]
    # (ia|jb) in chemists' notation, shape (o, v, o, v).
    eri_ovov = ao2mo.general(mol, (c_occ, c_vir, c_occ, c_vir), compact=False)
    eri_ovov = np.asarray(eri_ovov).reshape(o, v, o, v)

    e_occ = eps[frozen:nocc]
    e_vir = eps[nocc:]
    denom = (e_occ[:, None, None, None] - e_vir[None, :, None, None]
             + e_occ[None, None, :, None] - e_vir[None, None, None, :])

    exchange = eri_ovov.transpose(0, 3, 2, 1)  # (ib|ja) laid out as [i, a, j, b]
    e_os = np.einsum('iajb,iajb->', eri_ovov, eri_ovov / denom)
    e_ss = np.einsum('iajb,iajb->', eri_ovov, (eri_ovov - exchange) / denom)
    return {
        'e_corr': float(e_os + e_ss),
        'e_os': float(e_os),
        'e_ss': float(e_ss),
        'nocc_active': int(o),
        'nvir': int(v),
        'estimated_memory_gib': float(estimate),
    }


def run_mp2(mol, frozen=0, tol=1e-10, max_memory_gib=None, solver=None,
            **solver_options):
    """Run the custom RHF solver, then canonical MP2 on its orbitals."""
    if solver_options.get('jk_mode', 'direct') != 'direct':
        raise ValueError(
            "MP2 rebuilds the Fock matrix with the exact direct J/K, so "
            "jk_mode must be 'direct'"
        )
    nelec_alpha, nelec_beta = mol.nelec
    if nelec_alpha != nelec_beta:
        raise ValueError('MP2 here supports closed-shell systems only')
    nocc = nelec_alpha

    hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    overlap = mol.intor('int1e_ovlp')
    solver = solver if solver is not None else Hartree_Fock(mol)
    e_hf, dm = solver.solve_rhf(
        mol.nelec, hcore, overlap, mol, mol.energy_nuc(), tol=tol,
        **solver_options,
    )
    eps, coeff = canonical_orbitals(solver, hcore, overlap, dm)
    result = mp2_correlation(
        mol, coeff, eps, nocc, frozen=frozen, max_memory_gib=max_memory_gib,
    )
    result.update({
        'e_hf': float(e_hf),
        'e_tot': float(e_hf + result['e_corr']),
        'nocc': int(nocc),
        'frozen': int(frozen),
        'nao': int(mol.nao_nr()),
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Primary closed-shell MP2.')
    parser.add_argument('--xyz', required=True, help='XYZ geometry file')
    parser.add_argument('--basis', default='cc-pvtz')
    parser.add_argument('--unit', default='Angstrom')
    parser.add_argument('--charge', type=int, default=0)
    parser.add_argument('--frozen', type=int, default=0, help='frozen core orbitals')
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--max-memory-gib', type=float, default=None)
    args = parser.parse_args(argv)

    mol, _ = build_molecule_from_xyz(
        args.xyz, basis=args.basis, unit=args.unit, charge=args.charge, spin=0,
        verbose=0,
    )
    result = run_mp2(
        mol, frozen=args.frozen, tol=args.tol, max_memory_gib=args.max_memory_gib,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
