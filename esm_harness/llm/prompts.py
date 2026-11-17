"""Prompt templates. Each ``prompt_id`` is an ablation level."""

from __future__ import annotations

from pathlib import Path

PROMPT_IDS = ("hf_minimal", "hf_spec", "hf_show_baseline")

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
    lines = ["Molecules, with coordinates in the stated unit:"]
    for case in cases:
        lines.append(
            f"- {case['id']}: basis {case['basis']}, charge {case['charge']}, "
            f"spin {case['spin']}, unit {case.get('unit', 'Angstrom')}, atom {case['atom']}"
        )
    return lines


def _result_lines(evaluation: dict) -> list[str]:
    lines = [f"status: {evaluation['status']}"]
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
) -> str:
    if prompt_id not in PROMPT_IDS:
        raise ValueError(f"unknown prompt_id {prompt_id}")
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
