"""Measurements that decide the C172 speed items; hf.py is not changed.

* ``k-profile``: time every step inside PySCF ``sgx_jk.get_k_only`` on one
  alkane: AO values, the two sparse AO matrix products, the ESP integrals,
  the screening work and the screening-data build that runs when the K
  tolerance changes. A full K and an incremental K on the coarse grid and a
  full K on the fine grid are profiled.
* ``blksize``: the same full K with pinned SGX block sizes.
* ``memory``: process high-water mark of one DF-J build and of each K build.
* ``dense``: CPU (SciPy, BLAS threads) against CuPy for the generalized
  eigensolve, the density, the DIIS commutator and the DIIS bookkeeping of
  the Phase 1 driver, with the differences between the two.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import scipy.linalg
from pyscf import df, gto, lib, scf
from pyscf.df import df_jk
from pyscf.sgx import sgx as sgx_mod
from pyscf.sgx import sgx_jk

from hf import import_cupy, matching_cosx_grids, read_xyz


def status_gib(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return float(line.split()[1]) / (1024 * 1024)
    return None


def reset_peak():
    """Reset VmHWM to the current RSS (Linux clear_refs code 5)."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
        return True
    except OSError:
        return False


class Profiler:
    """Exclusive wall time per call path of wrapped functions (one thread)."""

    def __init__(self):
        self.exclusive = defaultdict(float)
        self.inclusive = defaultdict(float)
        self.calls = defaultdict(int)
        self.stack = []

    def wrap(self, name, function):
        def wrapper(*args, **kwargs):
            path = "/".join([frame[0] for frame in self.stack] + [name])
            started = time.perf_counter()
            self.stack.append([name, 0.0])
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                _, child = self.stack.pop()
                self.exclusive[path] += elapsed - child
                self.inclusive[path] += elapsed
                self.calls[path] += 1
                if self.stack:
                    self.stack[-1][1] += elapsed
        return wrapper

    def report(self, total):
        rows = {path: {"exclusive_s": self.exclusive[path],
                       "inclusive_s": self.inclusive[path],
                       "calls": self.calls[path],
                       "exclusive_share": self.exclusive[path] / total if total else None}
                for path in sorted(self.exclusive)}
        timed = sum(self.exclusive.values())
        return {"total_s": total, "untimed_s": total - timed, "steps": rows}


def make_sgx(mol, radial, angular, memory_mb):
    """The SGX object hf.Hartree_Fock builds for the pyscf_reference engine."""
    engine = sgx_mod.SGX(mol, auxbasis="weigend")
    engine.verbose = 0
    engine.max_memory = float(memory_mb)
    engine.dfj = True
    engine.optk = True
    engine.fit_ovlp = True
    engine.grids_level_i = 1
    engine.grids_level_f = 1
    engine.grids = matching_cosx_grids(mol, radial, angular)
    engine._opt = sgx_mod._make_opt(mol)
    engine._pjs_data = None
    return engine


def profiled_k(mol, engine, dm, tol, screen_dm=None, memory_mb=None):
    """One ``get_k_only`` call with each inner step timed.

    ``memory_mb`` sets the SGX block budget: ``max_memory = RSS + memory_mb``
    just before the call, so the block does not depend on the resident set.
    """
    profiler = Profiler()
    patches = []

    def patch(owner, name, label):
        original = getattr(owner, name)
        patches.append((owner, name, original))
        setattr(owner, name, profiler.wrap(label, original))

    patch(sgx_jk, "_sgxdot_ao_dm_sparse", "matmul_fg_ao_dm_sparse")
    patch(sgx_jk, "_sgxdot_ao_gv_sparse", "matmul_vk_ao_gv_sparse")
    patch(sgx_jk.SGXData, "get_g_threshold", "screen_gv_threshold")
    patch(sgx_jk.SGXData, "reduce_ao_to_shl_mat", "screen_ao_shell_norms")
    patch(sgx_jk.SGXData, "reduce_mask_ni2sgx", "screen_block_reduce")
    patch(sgx_jk.SGXData, "get_dm_threshold_matrix", "screen_dm_pairs")
    patch(sgx_jk.SGXData, "build", "screening_data_build")
    patch(lib, "einsum", "lib_einsum")
    patch(lib, "dot", "lib_dot")
    original_gen = sgx_jk._gen_k_direct
    patches.append((sgx_jk, "_gen_k_direct", original_gen))

    def gen_k_direct(*args, **kwargs):
        return profiler.wrap("esp_integrals_k_drv", original_gen(*args, **kwargs))

    sgx_jk._gen_k_direct = gen_k_direct
    seen_blocks = []
    original_loop = sgx_jk.SGXData.get_loop_data

    def loop_data(self, *args, **kwargs):
        values = original_loop(self, *args, **kwargs)
        seen_blocks.append(int(values[-1]))
        return values

    patches.append((sgx_jk.SGXData, "get_loop_data", original_loop))
    sgx_jk.SGXData.get_loop_data = loop_data
    mol.eval_gto = profiler.wrap("ao_values_eval_gto", gto.Mole.eval_gto.__get__(mol))
    rebuild = engine._pjs_data is None or engine._pjs_data._itol != tol
    if memory_mb is not None:
        engine.max_memory = lib.current_memory()[0] + float(memory_mb)
        if engine._pjs_data is not None:
            engine._pjs_data.max_memory = engine.max_memory
    engine._full_dm = None if screen_dm is None else np.asarray(screen_dm, order="C")
    rss_before = status_gib("VmRSS")
    reset_peak()
    started = time.perf_counter()
    try:
        vk = sgx_jk.get_k_only(engine, np.asarray(dm, order="C"), hermi=1,
                               direct_scf_tol=tol)
    finally:
        total = time.perf_counter() - started
        engine._full_dm = None
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
        del mol.eval_gto
    report = profiler.report(total)
    steps = report["steps"]
    top = {path: row for path, row in steps.items() if not path.startswith("screening_data_build")}
    matmul = sum(row["exclusive_s"] for path, row in top.items()
                 if path.split("/")[-1].startswith("matmul_"))
    dense_einsum = sum(row["exclusive_s"] for path, row in top.items() if path == "lib_einsum")
    build = steps.get("screening_data_build", {}).get("inclusive_s", 0.0)
    report.update(
        tolerance=tol, incremental=screen_dm is not None, screening_rebuilt=rebuild,
        screening_build_s=build, k_loop_s=total - build,
        ngrids=int(engine.grids.weights.size),
        loop_blksize=seen_blocks[-1] if seen_blocks else None,
        blocks=int(-(-engine.grids.weights.size // seen_blocks[-1])) if seen_blocks else None,
        sparse_matmul_s=matmul, sparse_matmul_share=matmul / total,
        dense_proj_einsum_s=dense_einsum,
        esp_integral_s=sum(row["exclusive_s"] for path, row in top.items()
                           if path.endswith("esp_integrals_k_drv")),
        ao_values_s=sum(row["exclusive_s"] for path, row in top.items()
                        if path == "ao_values_eval_gto"),
        rss_before_gib=rss_before, peak_rss_gib=status_gib("VmHWM"),
        peak_increase_gib=status_gib("VmHWM") - rss_before,
    )
    return vk, report


def dense_benchmark(mol, threads):
    """SciPy eigh(F, S) against a CuPy Cholesky reduction on the same input."""
    lib.num_threads(threads)
    overlap = mol.intor_symmetric("int1e_ovlp")
    hcore = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    nocc = mol.nelectron // 2
    nao = overlap.shape[0]
    result = {"nao": nao, "nocc": nocc}
    started = time.perf_counter()
    energies, coeff = scipy.linalg.eigh(hcore, overlap)
    result["cpu_eigh_s"] = time.perf_counter() - started
    started = time.perf_counter()
    occupied = coeff[:, :nocc]
    density = 2.0 * np.einsum("ma,na->mn", occupied, occupied.conj())
    result["cpu_density_einsum_s"] = time.perf_counter() - started
    started = time.perf_counter()
    density_blas = 2.0 * (occupied @ occupied.T)
    result["cpu_density_gemm_s"] = time.perf_counter() - started
    result["cpu_density_einsum_vs_gemm"] = float(np.max(np.abs(density - density_blas)))
    started = time.perf_counter()
    commutator = hcore @ density @ overlap - overlap @ density @ hcore
    result["cpu_diis_error_4gemm_s"] = time.perf_counter() - started
    started = time.perf_counter()
    half = hcore @ density @ overlap
    commutator2 = half - half.T
    result["cpu_diis_error_2gemm_s"] = time.perf_counter() - started
    result["cpu_diis_error_2gemm_vs_4gemm"] = float(np.max(np.abs(commutator - commutator2)))
    errors = [commutator + 1e-3 * k * overlap for k in range(8)]
    focks = [hcore + 1e-3 * k * overlap for k in range(8)]
    started = time.perf_counter()
    gram = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            gram[i, j] = gram[j, i] = np.vdot(errors[i], errors[j]).real
    result["cpu_diis_gram_64_vdots_s"] = time.perf_counter() - started
    coefficients = np.full(8, 1.0 / 8)
    started = time.perf_counter()
    np.einsum("i,ijk->jk", coefficients, focks)
    result["cpu_diis_einsum_extrapolate_s"] = time.perf_counter() - started
    started = time.perf_counter()
    errors[0].copy()
    result["cpu_copy_one_matrix_s"] = time.perf_counter() - started
    del errors, focks
    if nao <= 6000:
        result["overlap_condition"] = float(np.linalg.cond(overlap))
    try:
        cupy = import_cupy()
        from cupyx.scipy.linalg import solve_triangular
    except Exception as exc:
        result["gpu_error"] = f"{type(exc).__name__}: {exc}"
        return result
    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
    name = props["name"]
    result["gpu"] = {"name": name.decode() if isinstance(name, bytes) else str(name),
                     "total_memory_gib": props["totalGlobalMem"] / 1024 ** 3,
                     "cupy": cupy.__version__,
                     "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion()}

    def sync_time(function):
        cupy.cuda.Device().synchronize()
        started = time.perf_counter()
        value = function()
        cupy.cuda.Device().synchronize()
        return value, time.perf_counter() - started

    n = 8192
    a = cupy.random.random((n, n))
    b = cupy.random.random((n, n))
    sync_time(lambda: a @ b)
    _, seconds = sync_time(lambda: a @ b)
    result["gpu_dgemm_tflops"] = 2.0 * n ** 3 / seconds / 1e12
    del a, b
    host_a = np.random.random((4096, 4096))
    host_a @ host_a
    started = time.perf_counter()
    host_a @ host_a
    result["cpu_dgemm_tflops_4096"] = 2.0 * 4096 ** 3 / (time.perf_counter() - started) / 1e12
    del host_a

    s_gpu, upload_s = sync_time(lambda: cupy.asarray(overlap))
    f_gpu = cupy.asarray(hcore)
    result["gpu_upload_one_matrix_s"] = upload_s
    lower, chol_s = sync_time(lambda: cupy.linalg.cholesky(s_gpu))
    inverse, inv_s = sync_time(lambda: solve_triangular(lower, cupy.eye(nao), lower=True))
    result["gpu_cholesky_s"] = chol_s
    result["gpu_triangular_inverse_s"] = inv_s

    def solve():
        reduced = inverse @ f_gpu @ inverse.T
        reduced = 0.5 * (reduced + reduced.T)
        values, vectors = cupy.linalg.eigh(reduced)
        mo = inverse.T @ vectors
        occ = mo[:, :nocc]
        return values, mo, 2.0 * (occ @ occ.T)

    solve()
    (values, mo, density_gpu), solve_s = sync_time(solve)
    result["gpu_eigh_reduce_density_s"] = solve_s
    density_host, download_s = sync_time(lambda: cupy.asnumpy(density_gpu))
    result["gpu_download_density_s"] = download_s
    result["eigh_density_max_abs_diff"] = float(np.max(np.abs(density_host - density)))
    result["eigh_energy_trace_diff"] = abs(float(np.einsum("ij,ji", density_host - density, hcore)))
    result["orbital_energy_max_abs_diff"] = float(np.max(np.abs(cupy.asnumpy(values) - energies)))
    d_gpu = cupy.asarray(density)

    def commutator_gpu():
        half = f_gpu @ d_gpu @ s_gpu
        return half - half.T

    commutator_gpu()
    error_gpu, error_s = sync_time(commutator_gpu)
    result["gpu_diis_error_s"] = error_s
    result["diis_error_max_abs_diff"] = float(np.max(np.abs(cupy.asnumpy(error_gpu) - commutator)))
    history = [error_gpu + 1e-3 * k * s_gpu for k in range(8)]
    _, dots_s = sync_time(lambda: [float(cupy.vdot(history[-1], h)) for h in history])
    result["gpu_diis_8_dots_s"] = dots_s
    del history
    cupy.get_default_memory_pool().free_all_blocks()
    return result


def run(args, result):
    lib.num_threads(args.threads)
    if "dense" in args.steps:
        for carbons in args.dense_carbons:
            path = Path(__file__).with_name(f"c{carbons}h{2 * carbons + 2}.xyz")
            dense_mol = gto.M(atom=read_xyz(path)[0], basis="cc-pvtz", unit="Angstrom",
                              verbose=0)
            report = dense_benchmark(dense_mol, args.threads)
            result.setdefault("dense", {})[f"c{carbons}"] = report
            print("DENSE", carbons, json.dumps(report), flush=True)
            dump(args, result)
    if not set(args.steps) & {"k-profile", "blksize", "memory"}:
        return
    xyz = Path(__file__).with_name(f"c{args.carbons}h{2 * args.carbons + 2}.xyz")
    mol = gto.M(atom=read_xyz(xyz)[0], basis="cc-pvtz", unit="Angstrom", verbose=0,
                max_memory=args.memory_mb)
    result.update(carbons=args.carbons, nao=mol.nao_nr(), nbas=mol.nbas,
                  threads=args.threads, sgx_blksize_unit=int(sgx_jk.SGX_BLKSIZE))
    overlap = mol.intor_symmetric("int1e_ovlp")
    hcore = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    density0 = scf.hf.init_guess_by_minao(mol)
    dfobj = df.DF(mol, auxbasis="weigend")
    dfobj.verbose = 0
    dfobj.max_memory = args.memory_mb
    for label in ("dfj_first", "dfj_repeat"):
        rss_before = status_gib("VmRSS")
        reset_peak()
        started = time.perf_counter()
        vj = df_jk.get_j(dfobj, density0, hermi=1, direct_scf_tol=1e-13)
        result[label + "_s"] = time.perf_counter() - started
        result[label + "_peak_increase_gib"] = status_gib("VmHWM") - rss_before
    print("DFJ", json.dumps({k: v for k, v in result.items() if k.startswith("dfj")}), flush=True)
    dump(args, result)

    coarse = make_sgx(mol, 30, 110, args.memory_mb)
    vk0, report = profiled_k(mol, coarse, density0, 1e-13, memory_mb=args.k_memory_mb)
    result.setdefault("k_profile", {})["coarse_full_minao"] = report
    print("KPROFILE coarse_full_minao", json.dumps(report), flush=True)
    dump(args, result)
    _, coeff = scipy.linalg.eigh(hcore + vj - 0.5 * vk0, overlap)
    occupied = coeff[:, :mol.nelectron // 2]
    density1 = 2.0 * occupied @ occupied.T
    if "blksize" in args.steps:
        nao = mol.nao_nr()
        for blocks in args.block_counts:
            points = -(-coarse.grids.weights.size // blocks)
            points = -(-points // sgx_jk.SGX_BLKSIZE) * sgx_jk.SGX_BLKSIZE
            memory_mb = points * 4 * nao * 8 / 1e6 + 64.0
            _, report = profiled_k(mol, coarse, density1, 1e-13, memory_mb=memory_mb)
            report["requested_blocks"] = blocks
            result.setdefault("blksize_scan", []).append(report)
            print("BLKSIZE", blocks, json.dumps({k: report[k] for k in (
                "total_s", "loop_blksize", "blocks", "screening_rebuilt",
                "sparse_matmul_share", "peak_increase_gib")}), flush=True)
            dump(args, result)
    _, report = profiled_k(mol, coarse, density1 - density0, 1e-10, screen_dm=density1,
                           memory_mb=args.k_memory_mb)
    result["k_profile"]["coarse_incremental"] = report
    print("KPROFILE coarse_incremental", json.dumps(report), flush=True)
    dump(args, result)
    if args.fine:
        del coarse
        fine = make_sgx(mol, 35, 302, args.memory_mb)
        _, report = profiled_k(mol, fine, density1, 1e-13, memory_mb=args.k_memory_mb)
        result["k_profile"]["fine_full"] = report
        print("KPROFILE fine_full", json.dumps(report), flush=True)
        dump(args, result)


def dump(args, result):
    args.output.write_text(json.dumps(result, indent=1, default=float))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbons", type=int, default=86)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-mb", type=float, default=65536)
    parser.add_argument("--k-memory-mb", type=float, default=None)
    parser.add_argument("--steps", nargs="+", default=["dense", "k-profile", "blksize", "memory"])
    parser.add_argument("--dense-carbons", type=int, nargs="+", default=[86, 172])
    parser.add_argument("--block-counts", type=int, nargs="+", default=[40, 8, 2])
    parser.add_argument("--fine", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"accepted": False, "argv": {**vars(args), "output": str(args.output)}}
    started = time.perf_counter()
    try:
        run(args, result)
        result["accepted"] = True
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        dump(args, result)


if __name__ == "__main__":
    main()
