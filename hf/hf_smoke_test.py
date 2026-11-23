#!/usr/bin/env python
"""Small cc-pVTZ smoke test for the custom RHF implementation."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from pyscf import gto, scf

from hf import Hartree_Fock, molecule_metadata


def main():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="cc-pvtz",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    metadata = molecule_metadata(mol, "cc-pvtz", source_path="H2 smoke test")
    assert metadata["basis"] == "cc-pVTZ"
    assert metadata["basis_functions"] == mol.nao_nr()
    assert metadata["basis_shells"] == mol.nbas
    assert metadata["atom_count"] == 2
    assert metadata["nelectron"] == 2
    assert metadata["nocc"] == 1

    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    overlap = mol.intor("int1e_ovlp")
    custom_energy, density = Hartree_Fock(mol).solve_rhf(
        mol.nelec,
        hcore,
        overlap,
        mol,
        mol.energy_nuc(),
        tol=1e-10,
    )

    reference = scf.RHF(mol)
    reference.conv_tol = 1e-12
    reference.kernel()

    assert np.isfinite(custom_energy)
    assert np.isfinite(density).all()
    assert density.shape == (mol.nao_nr(), mol.nao_nr())
    assert abs(custom_energy - reference.e_tot) < 1e-8

    print(json.dumps({
        "status": "success",
        "basis": metadata["basis"],
        "basis_functions": metadata["basis_functions"],
        "basis_shells": metadata["basis_shells"],
        "atom_count": metadata["atom_count"],
        "nelectron": metadata["nelectron"],
        "nocc": metadata["nocc"],
        "custom_energy_hartree": float(custom_energy),
        "pyscf_energy_hartree": float(reference.e_tot),
        "energy_difference_hartree": float(custom_energy - reference.e_tot),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
