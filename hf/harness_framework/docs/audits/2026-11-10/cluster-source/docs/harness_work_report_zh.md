# 电子结构程序 Harness Engineering 工作报告

日期：2026-11-04

环境：Python 3.12.16，NumPy 2.5.4，SciPy 1.18.1，PySCF 2.14.0。测量在单机 CPU 上完成，BLAS 线程数为 1。本报告只整理已经落地的验收文件和教程中仍然保留的记录，没有重跑苯基准，也没有启动生产 SCF。

## 1. 已完成的工作

两周教程的主线已经闭合。被测对象是 `hf.py` 中的闭壳层 Hartree–Fock，覆盖直接 J/K、Schwarz 筛选、增量 Fock、RI-J、COSX、DIIS 和 RHF 主循环。

Harness 提供框架无关的 `MethodSpec`、`CandidateRun`、`MetricReport` 和 `AcceptanceGate`。候选程序通过 `HARNESS_REPORT_PATH` 写入指标，CLI 负责超时、失败分类和接受或拒绝。正确性失败时，性能收益不算通过。

已经可重复运行的任务有三件：

- `rhf_direct_baseline`：直接 RHF 与 PySCF 对照。
- `rhf_jk_parallel`：直接 J/K 的 `serial`、`cpu/2`、`cpu/4`。
- `rhf_rijcosx_parallel`：RI-J、COSX 和 RIJCOSX-SCF 的分模块与全链路基准。

指定 `--report-dir` 时，最新的 `<任务名>.acceptance.json` 仍会被覆盖。历史副本写到 `runs/<candidate_id>.acceptance.json`，决策摘要追加到 `experiments.jsonl`。当前归档的第一行是已接受的 RIJCOSX 并行候选 `rhf-rijcosx-parallel-20261104T041258076806Z`。

## 2. 直接 J/K

教程第 6.1 节记录，分块调度之前的综合 speedup 为 0.684，低于 1.05。加入 `parallel_chunk_size=16` 和小体系串行回退之后，苯/6-31G 的 CPU/2 从约 30.6 秒降到约 26.1 秒。

当前 `outputs/benchmark/rhf-jk-parallel.acceptance.json` 的结果是：

- 对 PySCF 的 J/K 最大绝对误差 `jk_max_abs` 为 6.01e-12。
- 串行与并行的矩阵差为 0，矩阵有限。
- 四个案例的串行中位数合计为 30.67 秒，4 线程合计为 27.87 秒。
- 综合 speedup 为 1.101，任务接受。案例中位数 speedup 为 1.022，只作诊断。苯案例为 1 次正式重复，其余案例为 1 次预热加 5 次重复。

## 3. RI-J/COSX 调度前

这一轮的验收文件后来被覆盖，没有进入 `experiments.jsonl`。下面的数字来自覆盖前的验收输出，报告中单独标明。

当时综合 speedup 为 0.713，由三个模块 speedup 的最小值决定：RI-J 0.882，COSX 0.713，RIJCOSX-SCF 0.995。数值本身已经通过：RI-J 对 PySCF DF 的最大误差约 1.32e-8，COSX 对解析 K 的最大误差约 1.07e-4，串行与并行的矩阵差为 0。SCF 只统计了水和水二聚体，这两个体系的壳层数低于 `parallel_min_shells=12`，实际没有并行，所以全链路 speedup 停在约 1.0。

变慢集中在唯一真正并行的苯/6-31G。RI-J 从 8.60 秒增到 9.77 秒，COSX 从 4.72 秒增到 6.70 秒。原因是每个辅助壳层、每个壳层对都单独调用 PySCF 积分，并反复构造 `cintopt` 和环境数组。这些准备工作占着 GIL，libcint 本体太短，线程池只增加开销。

## 4. RI-J/COSX 调度后

`outputs/benchmark/rhf-rijcosx-parallel.acceptance.json` 已接受。固定设置仍是 `threads=1`、`parallel_chunk_size=16`、`parallel_min_shells=12`、辅助基组 weigend、COSX 网格 40×146、最终网格 12×74、批大小 256。苯的 SCF 纳入全链路计时，任务超时为 900 秒。

正确性：

- RI-J 对 PySCF density fitting 的最大绝对误差为 1.32e-8，门槛为 2e-8。串行与并行差为 0。
- COSX 对 PySCF 解析交换矩阵的最大绝对误差为 1.07e-4，门槛为 2e-4。串行与并行差为 0。
- 水、水二聚体和苯/6-31G 的 RIJCOSX-SCF 都收敛。串行与并行总能量差为 0，密度有限且形状正确。
- `scf_pyscf_energy_max_abs` 为 0.063 Ha。它说明 RI-J 加 COSX 与全解析 RHF 不是同一个数值方法，不参与接受或拒绝。

性能，定义为各案例串行中位数之和除以 4 线程中位数之和：

- RI-J speedup 为 1.613。苯从 0.085 秒降到 0.051 秒。串行合计 0.0893 秒，4 线程合计 0.0554 秒。
- COSX speedup 为 2.449。苯从 3.095 秒降到 1.151 秒。串行合计 3.273 秒，4 线程合计 1.337 秒。
- RIJCOSX-SCF speedup 为 2.501。串行合计 52.90 秒，4 线程合计 21.15 秒。
- 综合 speedup 为 1.613，高于 1.05。

归档标识为 `rhf-rijcosx-parallel-20261104T041258076806Z`，`passed` 为 true。

## 5. 优化方案

直接 J/K 把多个壳层任务合成一块再提交，并用 `parallel_min_shells=12` 让小体系回到串行。壳层数低于这个值时，线程管理成本高于计算成本。

RI-J 在准备阶段缓存拼接后的积分环境和优化数据。每个 AO 壳层对只计算一次覆盖全部辅助壳层的三中心积分，再按原来的 `aux_q` 降序和 `df_tol` 前缀收缩。被筛选掉的辅助壳层不进入 J 矩阵。

COSX 的每个网格批次只计算一次全壳层 `int1e_grids`，再按原来的 `Q_ij * max|F|` 规则缩并。批次之间并行，主线程按提交顺序归并。

`ThreadPoolExecutor` 在同一次求解器的多次 J/K 构建之间复用，归并顺序保持先提交先归并。直接 J/K 的任务划分没有改成全壳层积分。

这三项改动带来的收益来自更长的 libcint 调用，而不是增加 worker 数。4 线程相对串行的加速仍低于 4 倍：RI-J 为 1.613，COSX 为 2.449。水和水二聚体继续因壳层数不足而走串行路径。

## 6. 下一步 Harness 计划

综合 speedup 已经超过 1.05。下一步是 `alk375` / cc-pVTZ 的可行性预检，而不是直接运行 `alk375_ccpvtz.json` 的生产 SCF。该配置预期 21778 个基函数，4 个 CPU worker，辅助基组 weigend，SCF 网格 30×110，最终网格 35×302。

预检任务只做三件事：

1. 构建分子，记录基函数数、壳层数，以及 `estimate_dense_solver_memory_gib` 给出的内存估计。
2. 内存超过配置上限时停止，并把原因写入 JSONL。
3. 内存允许时，只用较小网格各测一次 RI-J 和 COSX 的串行与 4 线程。结果沿用现有 `experiments.jsonl` 归档。

通过预检之后，才考虑一次生产规模 SCF。预检要补的指标是内存峰值、积分调用次数和屏蔽比例。UHF、GPU 和后 HF 方法排在这次预检之后。
