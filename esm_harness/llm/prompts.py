"""Prompt templates. Each ``prompt_id`` is an ablation level."""

from __future__ import annotations

from pathlib import Path

PROMPT_IDS = ("hf_minimal", "hf_spec", "hf_show_baseline", "hf_rijcosx")

RIJCOSX_INTERFACE = """\
Write one self-contained Python program for closed-shell restricted Hartree–Fock (RHF)
with the RIJCOSX approximation. Define solve(mol, method) and return
{{"e_tot": float, "converged": bool}}. mol is a built pyscf.gto.Mole (cc-pVTZ basis,
even electron count, spin 0). method is a dict such as
{{"jk": "rijcosx", "auxbasis": "weigend", "grid": [30, 110], "final_grid": [35, 302],
 "conv_tol": 1e-08, "final_energy": "single_point"}}.
You may import {imports}, and the pure standard-library modules math, cmath,
itertools, functools, collections, typing, and dataclasses; any other import is
rejected. Use mol.intor(...) for integrals, mol.eval_gto(...) for orbital values
on grid points, and mol.energy_nuc() for the nuclear repulsion.
Do not import pyscf.scf, pyscf.df, or pyscf.sgx, do not use pyscf.dft drivers, and
do not call kernel(). File I/O, exec, and eval are rejected. Each molecule is
solved once, in a fresh process, as its own job on a cluster node (48 cores,
256 GB, run with OMP_NUM_THREADS=8), so nothing can be cached between molecules.
Reply with the complete program in one fenced python block.
"""

RIJCOSX_METHOD = """\
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
"""

RIJCOSX_SCORING = """\
Acceptance: every molecule must converge, and its energy must be within 1e-8
Hartree of the reference (PySCF where PySCF converged, otherwise the cluster's
hf.py). A program that misses this on any molecule is rejected: it gets no score
and is never used as the best program. Accepted programs are ranked by
solve(mol, method) wall time relative to the reference program's recorded time
on each molecule (same node type, 8 threads); lower is better.
"""

INTERFACE = """\
Write one self-contained Python program for closed-shell restricted Hartree–Fock (RHF).
Define solve(mol) and return {"e_tot": float, "converged": bool}.
mol is a built pyscf.gto.Mole with an even electron count and spin 0.
You may import numpy, scipy, pyscf.gto, and the pure standard-library modules
math, cmath, itertools, functools, collections, typing, and dataclasses; any
other import is rejected. Use mol.intor(...) for integrals and mol.energy_nuc()
for the nuclear repulsion.
Do not import pyscf.scf or pyscf.df, and do not call kernel(). File I/O, exec,
and eval are rejected. Each timed call runs in a freshly loaded module with a
fresh Mole, so nothing can be cached between calls.
Reply with the complete program in one fenced python block.
"""

EQUATIONS = """\
Use chemist integrals eri[i,j,k,l] = (ij|kl) from mol.intor("int2e").
h = mol.intor("int1e_kin") + mol.intor("int1e_nuc"), S = mol.intor("int1e_ovlp").
nocc = mol.nelectron // 2. Build D from the occupied eigenvectors of the
generalized problem F C = S C epsilon. With D the total closed-shell density,
J_ij = sum_kl (ij|kl) D_kl and K_ij = sum_kl (ik|jl) D_kl, F = h + J - 0.5 K,
E = mol.energy_nuc() + 0.5 * sum_ij D_ij (h_ij + F_ij).
Accelerate with DIIS on the commutator F D S - S D F.
Stop when the energy change and the commutator norm are both under 1e-10.
"""

SCORING = """\
Scoring: every molecule below must converge, and the energy is compared with an
independent RHF reference (target error under 1e-8 Hartree). solve(mol) wall
time is compared with a reference program on the same host; lower is better.
Integral evaluation through mol.intor counts toward your time.
"""


def _case_lines(cases: list[dict]) -> list[str]:
    if any("xyz" in case for case in cases):
        return _summary_lines(cases)
    lines = ["Molecules, with coordinates in the stated unit:"]
    for case in cases:
        lines.append(
            f"- {case['id']}: basis {case['basis']}, charge {case['charge']}, "
            f"spin {case['spin']}, unit {case.get('unit', 'Angstrom')}, atom {case['atom']}"
        )
    return lines


def _summary_lines(cases: list[dict]) -> list[str]:
    """Large molecules are summarized; the program receives each one as a built Mole."""
    lines = ["Molecules (water clusters on a 4 Å cubic lattice and all-trans straight-chain alkanes):"]
    for case in cases:
        atoms = [entry.split()[0] for entry in case["atom"].split(";") if entry.strip()]
        formula = "".join(
            f"{symbol}{atoms.count(symbol) if atoms.count(symbol) > 1 else ''}" for symbol in sorted(set(atoms))
        )
        method = case.get("method") or {}
        lines.append(
            f"- {case['id']}: {formula}, {len(atoms)} atoms, basis {case['basis']}"
            + (f", {case['nao']} basis functions" if case.get("nao") else "")
            + (f", final_energy {method['final_energy']}" if method.get("final_energy") else "")
        )
    return lines


def _result_lines(evaluation: dict) -> list[str]:
    lines = [f"status: {evaluation['status']}"]
    if "accepted" in evaluation:
        lines.append(f"accepted (every molecule converged and within 1e-8 Hartree): {bool(evaluation['accepted'])}")
    if evaluation.get("stderr"):
        lines.append("error output (tail):")
        lines.append(evaluation["stderr"][-3000:])
    for case in evaluation.get("cases") or []:
        energy = case.get("energy")
        reference = case.get("ref_energy")
        signed = None if energy is None or reference is None else float(energy) - float(reference)
        ratio = case.get("time_ratio")
        ratio_text = "n/a" if ratio is None else f"{ratio:.3f}"
        lines.append(
            f"- {case['id']}: converged={case['converged']} signed_err={signed} "
            f"time_ratio={ratio_text}"
        )
    if evaluation.get("geomean_time_ratio") is not None:
        lines.append(
            f"max_abs_err={evaluation['max_abs_err']:.3e} "
            f"geomean_time_ratio={evaluation['geomean_time_ratio']:.3f}"
        )
    return lines


def render_prompt(
    prompt_id: str,
    cases: list[dict],
    feedback: dict | None,
    baseline_source: str | None,
    allowed_imports: list[str] | None = None,
) -> str:
    if prompt_id not in PROMPT_IDS:
        raise ValueError(f"unknown prompt_id {prompt_id}")
    if prompt_id == "hf_rijcosx":
        imports = ", ".join(allowed_imports or ["numpy", "scipy", "pyscf.gto"])
        parts = [RIJCOSX_INTERFACE.format(imports=imports), RIJCOSX_METHOD, RIJCOSX_SCORING]
    else:
        parts = [INTERFACE]
        if prompt_id in ("hf_spec", "hf_show_baseline"):
            parts.append(EQUATIONS)
        parts.append(SCORING)
    parts.append("\n".join(_case_lines(cases)))
    if prompt_id == "hf_show_baseline":
        parts.append(
            "Reference program whose time you are compared against (shown for this "
            "ablation only; it may use modules your program is not allowed to import):\n"
            f"```python\n{baseline_source or ''}\n```"
        )
    if feedback:
        best = feedback.get("best")
        last = feedback.get("last")
        if best:
            parts.append(
                "Best program so far and its scores. time_ratio is your time divided by "
                "the reference program's time; under 1 means faster."
            )
            parts.append("\n".join(_result_lines(best["eval"])))
            parts.append(f"```python\n{best['code']}\n```")
        if last and (not best or last["candidate_id"] != best["candidate_id"]):
            parts.append("Most recent attempt, which did not beat the best program:")
            parts.append("\n".join(_result_lines(last["eval"])))
            if last.get("code"):
                parts.append(f"```python\n{last['code']}\n```")
        parts.append(
            "Write an improved program: keep every molecule converged with energy error "
            "under 1e-8 Hartree and reduce solve(mol) time. Reply with the full program."
        )
    return "\n\n".join(parts) + "\n"


def read_baseline_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


DRIVER_WRAPPER = Path(__file__).with_name("driver_wrapper.md")


def subagent_instructions(prompt_path: str, response_path: str) -> str:
    """The exact text a driver sends to a fresh subagent; `audit` compares launches against it."""
    template = DRIVER_WRAPPER.read_text(encoding="utf-8")
    return template.replace("{prompt}", prompt_path).replace("{response}", response_path)
