"""PySCF RIJCOSX reference for the same JSON input used by hf.py.

The text log records each SCF iteration, the time of each part, and the
resident memory. Its last line is that molecule's total elapsed time.
It does not write a JSON summary.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf, sgx
from pyscf.df import df_jk
from pyscf.sgx import sgx_jk

import hf


def rss_gib():
    """Current resident set size in GiB. Linux reads VmRSS."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024 * 1024)
    return hf._peak_rss_gib()


def stage_record(name, started, rss_before):
    rss = rss_gib()
    seconds = time.perf_counter() - started
    increase = max(0.0, rss - rss_before)
    return (
        f"{name}_s={seconds:.3f} "
        f"{name}_rss_gib={rss:.3f} "
        f"{name}_rss_increase_gib={increase:.3f}"
    )


def matching_grids(mol, radial, angular):
    """Same atomic grid construction as hf.matching_cosx_grids."""
    return hf.matching_cosx_grids(mol, radial, angular)


def log_line(log_file, message):
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run PySCF RIJCOSX from the same JSON profile as hf.py.",
    )
    parser.add_argument("--config", required=True, help="JSON input profile")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    source_path = Path(config["xyz"])
    if not source_path.is_absolute():
        source_path = config_path.parent / source_path
    resources = dict(config.get("resources", {}))
    threads = int(resources.get("threads", 1))
    options = dict(config.get("solver_options", {}))
    auxbasis = options.get("auxbasis", "weigend")
    tol = float(options.get("tol", 1e-8))
    max_iteration = int(options.get("max_iteration", 200))
    grid_rad = int(options.get("grid_rad", 30))
    grid_ang = int(options.get("grid_ang", 110))
    final_grid_rad = int(options.get("final_grid_rad", 35))
    final_grid_ang = int(options.get("final_grid_ang", 302))
    run_started = time.perf_counter()

    lib.num_threads(threads)
    atom, metadata = hf.read_xyz(source_path)
    mol = gto.M(
        atom=atom,
        basis=config.get("basis", "cc-pvtz"),
        unit=config.get("unit", "Angstrom"),
        charge=int(config.get("charge", 0)),
        spin=int(config.get("spin", 0)),
        verbose=0,
    )
    expected = config.get("expected_basis_functions")
    if expected is not None and int(expected) != mol.nao_nr():
        raise ValueError(
            "Configured basis-function count mismatch: "
            f"expected {int(expected)}, built {mol.nao_nr()}"
        )

    prefix = hf._output_prefix_path(config, config_path, source_path)
    log_path = prefix.parent / f"{prefix.name}_pyscf.log"
    molecule_name = source_path.stem

    mf = scf.RHF(mol).COSX(auxbasis=auxbasis)
    mf.verbose = 0
    mf.chkfile = None
    mf.direct_scf = False
    mf.conv_tol = tol
    mf.conv_tol_grad = tol
    mf.max_cycle = max_iteration
    mf.init_guess = "minao"
    mf.diis_space = 8
    mf.with_df.dfj = True
    mf.with_df.optk = True
    mf.with_df.grids_level_i = 1
    mf.with_df.grids_level_f = 1
    original_build = mf.with_df.build

    def build_scf_grids(level=None):
        original_build(level)
        mf.with_df.grids = matching_grids(mol, grid_rad, grid_ang)

    mf.with_df.build = build_scf_grids

    stages = {}
    cycle_mark = {"started": None, "rss": None}

    def timed_get_jk(dm, hermi=1, vhfopt=None, with_j=True, with_k=True,
                     direct_scf_tol=1e-13, omega=None):
        if omega is not None or not (with_j and with_k and mf.with_df.dfj):
            return sgx.SGX.get_jk(
                mf.with_df, dm, hermi, vhfopt, with_j, with_k,
                direct_scf_tol, omega,
            )
        started = time.perf_counter()
        rss_before = rss_gib()
        vj = df_jk.get_j(mf.with_df, dm, hermi, direct_scf_tol)
        stages["j"] = stage_record("j", started, rss_before)
        started = time.perf_counter()
        rss_before = rss_gib()
        vk = sgx_jk.get_k_only(mf.with_df, dm, hermi, direct_scf_tol)
        stages["k"] = stage_record("k", started, rss_before)
        return vj, vk

    mf.with_df.get_jk = timed_get_jk
    real_get_fock = mf.get_fock
    real_eig = mf.eig

    def timed_get_fock(*args, **kwargs):
        diis = kwargs.get("diis")
        if diis is None and len(args) >= 6:
            diis = args[5]
        started = time.perf_counter()
        rss_before = rss_gib()
        if diis is not None:
            cycle_mark["started"] = started
            cycle_mark["rss"] = rss_before
        fock = real_get_fock(*args, **kwargs)
        if diis is not None:
            stages["diis"] = stage_record("diis", started, rss_before)
        return fock

    def timed_eig(*args, **kwargs):
        started = time.perf_counter()
        rss_before = rss_gib()
        result = real_eig(*args, **kwargs)
        stages["eigh"] = stage_record("eigh", started, rss_before)
        return result

    mf.get_fock = timed_get_fock
    mf.eig = timed_eig
    initial_logged = {"done": False}

    real_get_jk_after = timed_get_jk

    def timed_get_jk_with_initial(*args, **kwargs):
        result = real_get_jk_after(*args, **kwargs)
        if cycle_mark["started"] is None and not initial_logged["done"]:
            log_line(
                log_file,
                f"{molecule_name}: SCF initial Fock {stages['j']} {stages['k']}",
            )
            initial_logged["done"] = True
        return result

    mf.with_df.get_jk = timed_get_jk_with_initial

    with log_path.open("w") as log_file:
        log_line(log_file, f"program: pyscf {getattr(scf, '__version__', '2')}")
        log_line(log_file, f"source_path: {source_path}")
        log_line(log_file, f"basis: {hf.display_basis_name(config.get('basis', 'cc-pvtz'))}")
        log_line(log_file, f"basis_functions: {mol.nao_nr()}")
        log_line(log_file, f"atom_count: {metadata['xyz_atom_count']}")
        log_line(log_file, f"nelectron: {mol.nelectron}")
        log_line(log_file, f"charge: {mol.charge}")
        log_line(log_file, f"spin: {mol.spin}")
        log_line(log_file, f"threads: {threads}")
        log_line(log_file, f"auxbasis: {auxbasis}")
        log_line(log_file, f"tol: {tol:.1e}")
        log_line(log_file, f"grid: {grid_rad}x{grid_ang}")
        log_line(log_file, f"final_grid: {final_grid_rad}x{final_grid_ang}")
        log_line(log_file, f"max_iteration: {max_iteration}")
        log_line(
            log_file,
            f"{molecule_name}: RIJCOSX: auxbasis={auxbasis}, tol={tol:.1e}, "
            f"grid={grid_rad}x{grid_ang}, final_grid={final_grid_rad}x{final_grid_ang}",
        )
        log_line(log_file, f"{molecule_name}: SCF initial Fock")

        def callback(envs):
            cycle = int(envs["cycle"]) + 1
            energy = float(envs["e_tot"])
            previous = envs["last_hf_e"]
            if cycle == 1 or previous is None:
                delta_energy = "n/a"
                delta_dm = "n/a"
            else:
                delta_energy = f"{abs(energy - float(previous)):.3e}"
                delta_dm = f"{float(np.linalg.norm(envs['dm'] - envs['dm_last'])):.3e}"
            iteration = stage_record(
                "iteration", cycle_mark["started"], cycle_mark["rss"],
            )
            log_line(
                log_file,
                f"{molecule_name}: SCF iteration {cycle}/{max_iteration} "
                f"energy={energy:.12f} Ha "
                f"delta_energy={delta_energy} "
                f"delta_dm={delta_dm} "
                f"{stages['diis']} {stages['eigh']} {stages['j']} {stages['k']} "
                f"{iteration}",
            )

        mf.callback = callback
        try:
            mf.kernel()
            if mf.converged:
                log_line(
                    log_file,
                    f"{molecule_name}: SCF converged at iteration {mf.cycles}",
                )
                log_line(log_file, f"{molecule_name}: SCF final energy")
                mf.with_df.grids = matching_grids(mol, final_grid_rad, final_grid_ang)
                dm = mf.make_rdm1()
                started = time.perf_counter()
                rss_before = rss_gib()
                vj = df_jk.get_j(mf.with_df, dm, hermi=1)
                j_record = stage_record("j", started, rss_before)
                started = time.perf_counter()
                rss_before = rss_gib()
                vk = sgx_jk.get_k_only(mf.with_df, dm, hermi=1)
                k_record = stage_record("k", started, rss_before)
                hcore = mf.get_hcore()
                fock = hcore + vj - 0.5 * vk
                energy = mol.energy_nuc() + 0.5 * float(
                    np.einsum("ij,ij->", hcore + fock, dm)
                )
                log_line(
                    log_file,
                    f"{molecule_name}: SCF final energy={energy:.12f} Ha "
                    f"{j_record} {k_record}",
                )
            else:
                log_line(
                    log_file,
                    f"{molecule_name}: SCF did not converge in {max_iteration} iterations",
                )
                raise SystemExit(1)
            log_line(log_file, f"text_log: {log_path}")
        finally:
            log_line(
                log_file,
                f"{molecule_name}: total_elapsed_s={time.perf_counter() - run_started:.3f}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
