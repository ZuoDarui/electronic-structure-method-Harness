Write one self-contained Python program for closed-shell restricted Hartree–Fock (RHF)
with the RIJCOSX approximation. Define solve(mol, method) and return
{"e_tot": float, "converged": bool}. mol is a built pyscf.gto.Mole (cc-pVTZ basis,
even electron count, spin 0). method is a dict such as
{"jk": "rijcosx", "auxbasis": "weigend", "grid": [30, 110], "final_grid": [35, 302],
 "conv_tol": 1e-08, "final_energy": "single_point"}.
You may import numpy, scipy, pyscf.gto, pyscf.dft.gen_grid, pyscf.dft.radi, and the pure standard-library modules math, cmath,
itertools, functools, collections, typing, and dataclasses; any other import is
rejected. Use mol.intor(...) for integrals, mol.eval_gto(...) for orbital values
on grid points, and mol.energy_nuc() for the nuclear repulsion.
Do not import pyscf.scf, pyscf.df, or pyscf.sgx, do not use pyscf.dft drivers, and
do not call kernel(). File I/O, exec, and eval are rejected. Each molecule is
solved once, in a fresh process, as its own job on a cluster node (48 cores,
256 GB, run with OMP_NUM_THREADS=8), so nothing can be cached between molecules.
Reply with the complete program in one fenced python block.


The reference energies were computed on the same cluster with PySCF 2.14:
    mf = scf.RHF(mol).COSX(auxbasis=method["auxbasis"])
    mf.with_df.dfj = True; mf.with_df.optk = True
    mf.with_df.grids = <the grid below, at method["grid"]>
    mf.init_guess = "minao"; mf.diis_space = 8; mf.max_cycle = 200
    mf.conv_tol = mf.conv_tol_grad = method["conv_tol"]
Reproduce that approximate energy, not the exact RHF energy (they differ by about
1e-4 Hartree):
- J by density fitting with the auxiliary basis method["auxbasis"] ("weigend" is
  PySCF's name for def2-universal-JFIT): J_mn = sum_PQ (mn|P) [(P|Q)^-1]_PQ (Q|ls) D_ls.
  Build the auxiliary molecule with pyscf.gto (same atoms, that basis) and use
  int3c2e and int2c2e.
- K by seminumerical exchange with overlap fitting, on the grid
      grids = gen_grid.Grids(mol)
      grids.atom_grid = {mol.atom_pure_symbol(i): (rad, ang) for i in range(mol.natm)}
      grids.radi_method = radi.treutler_ahlrichs
      grids.prune = gen_grid.nwchem_prune
      grids.build(with_non0tab=True)
  With ao = mol.eval_gto("GTOval", grids.coords), w = grids.weights,
  S_num = ao^T diag(w) ao, Q = solve(S_num, S), F = ao (Q D), and
  A_g[n, s] = mol.intor("int1e_grids", grids=coords)[g, n, s] = (n s | 1/|r - r_g|):
  K_mn = sum_g w_g ao[g, m] sum_s A_g[n, s] F[g, s], then K = (K + K^T) / 2.
  That is exactly what PySCF computes with optk = False. The reference used
  optk = True, whose integral screening in pyscf.sgx.sgx_jk.get_k_only (P-junction
  screening, bound_algo "sample_pos", direct_scf_tol 1e-13, sgx_tol_energy and
  sgx_tol_potential "auto") moves the final energy by 5.2e-8 Hartree on water8
  and more on larger molecules, so it has to be reproduced to stay within 1e-8.
- F = h + J - 0.5 K and E = mol.energy_nuc() + 0.5 sum_mn D_mn (h_mn + F_mn).
  Converge the SCF on method["grid"] until the energy change is below
  method["conv_tol"] and the orbital-gradient norm is below method["conv_tol"].
- method["final_energy"] == "single_point": rebuild J and K once on
  method["final_grid"] from the converged density and return that energy.
  method["final_energy"] == "reconverge": continue the SCF on method["final_grid"]
  from the converged density until it converges again, and return that energy.


Acceptance: every molecule must converge, and its energy must be within 1e-8
Hartree of the reference (PySCF where PySCF converged, otherwise the cluster's
hf.py). A program that misses this on any molecule is rejected: it gets no score
and is never used as the best program. Accepted programs are ranked by
solve(mol, method) wall time relative to the reference program's recorded time
on each molecule (same node type, 8 threads); lower is better.


Molecules (water clusters on a 4 Å cubic lattice and all-trans straight-chain alkanes):
- water8: H16O8, 24 atoms, basis cc-pvtz, 464 basis functions, final_energy single_point
- water16: H32O16, 48 atoms, basis cc-pvtz, 928 basis functions, final_energy single_point
- water32: H64O32, 96 atoms, basis cc-pvtz, 1856 basis functions, final_energy single_point
- water64: H128O64, 192 atoms, basis cc-pvtz, 3712 basis functions, final_energy single_point
- water128: H256O128, 384 atoms, basis cc-pvtz, 7424 basis functions, final_energy single_point
- c21h44: C21H44, 65 atoms, basis cc-pvtz, 1246 basis functions, final_energy single_point
- c43h88: C43H88, 131 atoms, basis cc-pvtz, 2522 basis functions, final_energy single_point
- c86h174: C86H174, 260 atoms, basis cc-pvtz, 5016 basis functions, final_energy single_point
- c172h346: C172H346, 518 atoms, basis cc-pvtz, 10004 basis functions, final_energy reconverge
