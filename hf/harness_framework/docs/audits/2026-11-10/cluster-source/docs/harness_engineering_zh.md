# 电子结构程序的 Harness Engineering 入门

这份教程把当前仓库的 RHF 实现变成一个可以持续改进的实验系统。目标不是让代理“凭感觉写更快的代码”，而是让每次代码生成都经过同一套规格、参考对照、性质测试、性能测量和接受门槛。

第一阶段只覆盖单机 CPU 上的 Hartree–Fock 性能。完成教程后，你应能把一个优化任务写成机器可执行的规格，让任意代理或脚本生成候选实现，再自动回答三个问题：结果对不对、程序是否收敛、速度是否真的改善。

## 1. 先运行现有程序

当前仓库有三个关键文件：

- `hf.py`：`Hartree_Fock` 类，包含直接 J/K、Schwarz 筛选、增量 Fock、RI-J、COSX、DIIS 和 RHF SCF 主循环。
- `test_direct_hf.py`：以 PySCF 为参考的数值测试，覆盖 H₂、水、RI-J、COSX、增量重建和输入校验。
- `alk375.xyz`：可用于以后扩展较大体系的结构输入。

先准备一个包含 NumPy、SciPy、PySCF 的 Python 环境，然后运行：

```bash
python3 -m pip install -r requirements.txt
```

如果使用 Conda，建议在 Python 3.11–3.13 环境中安装 PySCF，再运行：

```bash
source /Users/darui/workspace/fieldquantum/.miniforge3/bin/activate \
  /Users/darui/workspace/fieldquantum/.pyscf-env
python3 -m unittest test_direct_hf.py -v
```

如果只想验证本教程的标准库部分，不需要科学计算依赖：

```bash
python3 -m unittest test_harness_engineering.py -v
```

### 1.1 CPU 并行配置

`lib.num_threads(threads)` 只控制 PySCF 和 BLAS 的底层线程；本仓库的壳层积分和 COSX 网格批次还可以通过任务级线程并行。配置文件的 `resources` 可以这样写：

```json
{
  "cpu_cores": 48,
  "threads": 8,
  "parallel_backend": "cpu",
  "parallel_workers": 4,
  "parallel_chunk_size": 16,
  "parallel_min_shells": 12
}
```

`parallel_backend` 支持 `serial`、`cpu` 和可选的 `gpu`。CPU 模式并行化直接 J/K、RI-J 和 COSX 的独立任务，并在主线程归并局部矩阵；`parallel_chunk_size` 将多个壳层任务合并后再提交，减少线程池调度开销；`parallel_min_shells` 让小体系自动回退串行，避免并行管理成本超过计算成本。`parallel_workers` 建议小于 `threads`，避免 BLAS 线程过量。GPU 模式需要安装与 CUDA 匹配的 CuPy，未安装时程序会明确报错。`alk375_ccpvtz.json` 已启用四个 CPU worker，可直接用 `run_alk375.sh` 运行。

## 2. 什么是这里的 Harness

Harness 是围绕被测算法的一层可重复控制程序。它把“请优化 J/K”变成一个有限、可判定的实验：

```text
MethodSpec
    ↓
运行候选程序（有超时、独立输出和报告路径）
    ↓
读取 CandidateRun + MetricReport
    ↓
参考结果、性质测试、性能统计
    ↓
AcceptanceGate：接受、拒绝或分类失败
```

这里的候选程序可以由人写、由 Codex 生成、由其他代理生成，或者只是一个参数扫描脚本。Harness 不依赖具体代理框架，因此代理更换不会改变科学验收标准。

三个原则决定了系统是否可靠：

1. **规格先于代码**：先写输入范围、允许改动的接口、误差门槛和性能目标。
2. **正确性先于速度**：数值结果、矩阵性质和 SCF 收敛任一失败，性能收益都不算有效。
3. **证据和决策分离**：候选程序只产生原始指标；Harness 根据统一规则做接受/拒绝。

## 3. 仓库中的最小接口

`harness_engineering/models.py` 提供四个纯数据对象。

### 3.1 `MethodSpec`

一个任务规格至少要说明候选命令、测试分子、正确性阈值、性能阈值和超时时间：

```json
{
  "name": "rhf-direct-baseline",
  "description": "Validate direct RHF against PySCF",
  "command": ["python3", "harness_engineering/examples/rhf_direct_candidate.py"],
  "cases": [
    {"name": "h2-sto3g", "atom": "H 0 0 0; H 0 0 1", "basis": "sto-3g"}
  ],
  "correctness_tolerances": {
    "energy_max_abs": 1e-7,
    "jk_max_abs": 1e-7
  },
  "required_metrics": [
    "energy_max_abs", "jk_max_abs", "all_finite", "scf_converged"
  ],
  "performance": {"max_slowdown": 1.25},
  "timeout_s": 300
}
```

对应的 Python 对象是不可变的 `MethodSpec`。这样代理不能在运行中偷偷改变验收条件。

### 3.2 `CandidateRun`

`run_candidate` 使用子进程执行候选程序，并记录：

- `status`：`completed`、`timeout` 或 `launch_error`；
- 返回码、耗时、stdout、stderr；
- `failure_kind`：语法、依赖、数值、SCF、超时或一般运行时失败；
- 候选写入的 JSON 报告路径。

候选通过环境变量 `HARNESS_REPORT_PATH` 获得报告路径。它不需要知道 Harness 的内部实现，只需写入 JSON。

### 3.3 `MetricReport`

候选报告使用稳定的机器格式：

```json
{
  "metrics": {
    "energy_max_abs": 2.1e-9,
    "jk_max_abs": 4.8e-9,
    "all_finite": true,
    "scf_converged": true,
    "median_seconds": 0.42
  },
  "cases": ["h2-sto3g"]
}
```

原始证据可以继续写入 stdout 或额外文件；`metrics` 只放用于决策的标量和布尔值。

### 3.4 `AcceptanceGate`

验收逻辑会检查：

- 候选是否正常退出并生成报告；
- 所有必需指标是否存在；
- 绝对误差是否低于阈值；
- 所有值是否有限、SCF 是否收敛；
- `speedup` 是否达到下限，或 `slowdown` 是否低于上限。

失败原因会作为列表返回，方便代理下一轮只修复已知问题，而不是盲目重写整个实现。

## 4. 运行第一个端到端任务

任务文件位于 `harness/tasks/rhf_direct_baseline.json`，候选程序位于 `harness_engineering/examples/rhf_direct_candidate.py`。依赖准备好后运行：

```bash
python3 -m harness_engineering.cli run \
  harness/tasks/rhf_direct_baseline.json \
  --workdir . \
  --report-dir outputs/benchmark
```

`--workdir` 是候选进程的工作目录；任务中的相对命令和导入路径都相对于它解析。生产环境可以把候选命令改成绝对路径，或在一个专用 checkout 中运行。

指定 `--report-dir` 时，候选原始指标保存为 `<任务名>.report.json`，Harness 的最终接受/拒绝结果保存为 `<任务名>.acceptance.json`。

命令会打印两部分 JSON：

- `run`：进程、耗时、输出和失败分类；
- `decision`：最终指标、`passed` 和拒绝原因。

这个示例候选做四件事：对 H₂ 和水构造 PySCF 分子；调用仓库中的 `build_jk`；与 PySCF 的 J/K 对照；运行 RHF 并比较总能量。它还记录每个案例的运行时间。参考实现只用于验收，不替代被测代码。

## 5. 把电子结构正确性写成性质

单个能量对照不够。对于 HF，至少要覆盖下列不变量：

| 性质 | 检查方式 | 失败意味着什么 |
| --- | --- | --- |
| J/K 矩阵形状 | `(nao, nao)` | 壳层切片或 AO 映射错误 |
| J/K 对称性 | `max(abs(A-A.T))` | 对称轨道积分或转置处理错误 |
| 有限值 | `isfinite` | 溢出、坏阈值或奇异线性代数 |
| 参考误差 | 与 PySCF 比较最大绝对误差 | 缩并、交换项或索引错误 |
| 密度电子数 | `trace(D S)` | 占据轨道或重叠矩阵处理错误 |
| 增量一致性 | `F(D+δD)` 与 `F(D)+ΔF(δD)` | 增量更新不满足线性关系 |
| SCF 残差 | `FDS-SDF` 范数 | 收敛判据或 DIIS 使用错误 |

对于 `hf.py`，`test_direct_hf.py` 已经给出了很好的第一批回归测试。新增优化时，先把性质测试加入测试文件，再让代理修改算法；这样代理无法通过删测试来“提高”分数。

## 6. 性能测量的基本规则

`harness_engineering/benchmark.py` 提供 `benchmark`，先预热，再重复运行，并返回中位数、最小值和 P90。建议遵循以下规则：

1. 预热次数与正式重复次数写入任务规格；
2. 比较同一进程、同一分子集合和同一收敛阈值；
3. 把正确性和速度分成独立指标；
4. 报告中位数，而不是一次偶然的最快时间；
5. 至少用 H₂、水和一个更大体系检查优化是否只对单个输入有效；
6. 记录 Python、NumPy、SciPy、PySCF 版本和 CPU 型号，便于解释回归。

性能门槛可以这样定义：候选必须先满足 `energy_max_abs <= 1e-7`、`jk_max_abs <= 1e-7`、`all_finite=true` 和 `scf_converged=true`，再检查 `speedup >= 1.05`。如果没有可重复收益，就保留正确候选但拒绝合入性能分支。

### 6.1 直接 J/K 并行基准

运行四个固定案例的串并行任务：

```bash
python3 -m harness_engineering.cli run \
  harness/tasks/rhf_jk_parallel.json \
  --workdir . \
  --report-dir outputs/benchmark
```

该任务比较 `serial`、`cpu/2` 和 `cpu/4`，并将 H₂、水、水二聚体和苯/6-31G 的逐案例指标写入报告。苯案例单次测量以避免当前纯 Python 壳层积分循环让整轮实验超过超时；报告中的 `case_repeats` 会明确记录这一点。

总体 `speedup` 定义为所有案例的逐案例中位数耗时之和的比值，避免一个最重案例被简单的案例中位数统计忽略；报告同时保留 `case_median_speedup` 供诊断。分块调度前的基准 speedup 为 `0.684`，低于 1.05；加入分块和小体系串行回退后，苯/6-31G 的 CPU/2 已从约 30.6 秒降至约 26.1 秒，说明任务粒度优化有效。最终验收以重新运行后的报告为准。

### 6.2 RI-J、COSX 与 RIJCOSX-SCF 基准

运行分模块和全链路任务：

```bash
python3 -m harness_engineering.cli run \
  harness/tasks/rhf_rijcosx_parallel.json \
  --workdir . \
  --report-dir outputs/benchmark
```

该任务使用水、水二聚体和苯/6-31G，分别比较 `serial`、`cpu/2` 和 `cpu/4`。RI-J 以 PySCF density fitting 的 J 矩阵为参考，COSX 以 PySCF 解析交换矩阵为参考，RIJCOSX-SCF 则检查串行与并行能量一致性、密度有限性和收敛状态。苯的 SCF 也计入全链路耗时；任务超时为 900 秒。

当前机器的结果是：RI-J 最大误差 `1.32e-8`，COSX 最大误差 `1.07e-4`，串行与并行的 RI-J/COSX 矩阵差异为 `0`，RIJCOSX-SCF 的串行与并行能量差为 `0`；三个案例的 SCF 都收敛，矩阵有限且对称。每个 AO 壳层对只计算一次三中心积分，每个 COSX 网格批次只计算一次全壳层 `int1e_grids`，线程池在多次构建之间复用。模块 speedup 为 RI-J `1.613`、COSX `2.449`、RIJCOSX-SCF `2.501`，综合 speedup 为 `1.613`，高于 `1.05`，任务接受。报告中的 `scf_pyscf_energy_max_abs` 为 `0.063`，仅作诊断，因为当前实现是 RI-J 加 COSX 近似，而 PySCF RHF 是全解析 J/K，两者不是同一个数值方法，不能把这项差异直接当作并行正确性失败。

## 7. 面向代理的任务模板

每个自动生成任务都应包含以下文本和机器字段：

```text
目标：降低 build_jk 在指定案例集合上的中位数耗时。
允许修改：Hartree_Fock.build_jk 及其私有辅助函数。
禁止修改：公开参数、电子数定义、测试中的参考路径。
正确性：J/K 最大绝对误差 <= 1e-7；矩阵对称；所有值有限。
性能：三个案例的中位数至少提升 5%，否则拒绝。
验证：先运行 test_direct_hf.py，再运行本任务。
失败策略：语法/接口错误修复一次；数值错误回到最近通过候选；连续两次无收益则停止。
```

代理的输出应是候选补丁和报告，而不是“看起来合理”的解释。Harness 只接受可运行、可复现、通过门槛的结果。

## 8. 两周冲刺安排

### 第 1–2 天：基线

- 阅读 `hf.py` 的输入、输出和主要数据流；
- 运行现有测试；
- 为 H₂、水记录直接 J/K、RHF 能量和耗时；
- 保存依赖版本和硬件信息。

### 第 3–4 天：最小 Harness

- 阅读 `models.py` 和 `runner.py`；
- 新建一个只写固定 JSON 的 toy candidate；
- 验证超时、返回码、缺失报告和失败分类；
- 把一次人工实验改成 `MethodSpec`。

### 第 5–6 天：参考与性质测试

- 加入 PySCF J/K 对照；
- 增加对称性、有限值、电子数和增量一致性；
- 将每个失败原因变成可搜索的稳定字符串。

### 第 7–8 天：性能与参数

- 使用 `benchmark` 统计中位数和 P90；
- 扫描 Schwarz 阈值、密度阈值、批大小和增量重建周期；
- 画出误差—耗时数据，先排除不满足正确性的参数。

### 第 9–10 天：自动代码任务

- 从直接 J/K 开始，让代理只修改一个函数；
- 每轮都运行原有回归和任务规格；
- 保存补丁、报告、stdout、stderr 和候选标识。

### 第 11–12 天：RI-J 与 COSX

- 把 `build_ri_j` 和 `build_cosx_k` 拆成独立任务；
- 为辅助基组、网格、截断阈值和最终高精度网格建立规格；
- 将近似误差和运行时间同时记录。

### 第 13–14 天：端到端闭环

- 选择一个优化目标；
- 让代理生成候选；
- 自动测试、对照、基准、接受或拒绝；
- 归档最终候选和失败候选，写一页实验结论。

这一轮的结论和归档格式见第 11 节。

## 9. 失败处理与重试策略

| 分类 | 例子 | 处理 |
| --- | --- | --- |
| `syntax_error` | 代码无法解析 | 允许一次局部修复 |
| `dependency_error` | 缺少包或错误导入 | 修正环境/导入，不改算法 |
| `numerical_failure` | J/K 或能量超阈值 | 回退最近通过版本，检查索引和缩并 |
| `scf_failure` | 达到最大迭代仍未收敛 | 检查 DIIS、初猜和 Fock 更新 |
| `timeout` | 超过任务时间 | 先缩小案例确认逻辑，再优化复杂度 |
| `runtime_error` | 未分类异常 | 保留完整 stderr，禁止静默吞错 |
| 性能退化 | 正确但变慢 | 保留为正确候选，拒绝性能目标 |

重试必须有上限。建议同一候选最多两次局部修复；连续两次没有改善就停止，避免代理在不变的错误上循环。

## 10. 后续扩展顺序

完成本教程后，可以按以下顺序扩展：

1. 用 SQLite 保存多轮实验的版本关系；JSONL 追加记录已经由第 11 节的归档完成；
2. 将 `MetricReport` 增加内存峰值、积分调用次数和屏蔽比例；
3. 添加 UHF/开放壳层，但重新定义参考和收敛性质；
4. 加入并行候选和固定 CPU 亲和性；
5. 最后再移植到 GPU 或加入后 HF 方法。

不要在验收门槛尚不稳定时同时引入新的电子结构方法、分布式运行和复杂代理编排。先保证每一个候选都有可解释的科学证据。

## 11. 实验结论：RI-J/COSX 调度

优化目标是让 RI-J、COSX 和 RIJCOSX-SCF 在 `serial`、`cpu/2`、`cpu/4` 上保持数值结果，并把综合 speedup 提高到至少 `1.05`。允许改动的是积分任务粒度和线程池复用。网格、辅助基组、筛选阈值和 `parallel_min_shells=12` 保持不变。

任务 [`harness/tasks/rhf_rijcosx_parallel.json`](../harness/tasks/rhf_rijcosx_parallel.json) 已接受。证据来自 [`outputs/benchmark/rhf-rijcosx-parallel.acceptance.json`](../outputs/benchmark/rhf-rijcosx-parallel.acceptance.json)：

- RI-J 对 PySCF density fitting 的最大绝对误差是 `1.32e-8`，串行与并行差是 `0`。
- COSX 对 PySCF 解析交换矩阵的最大绝对误差是 `1.07e-4`，串行与并行差是 `0`。
- 水、水二聚体和苯/6-31G 的 RIJCOSX-SCF 都收敛，串行与并行能量差是 `0`，密度有限。
- 模块 speedup 是 RI-J `1.613`、COSX `2.449`、RIJCOSX-SCF `2.501`。综合 speedup 是 `1.613`。
- `scf_pyscf_energy_max_abs` 是 `0.063`。它只说明 RI-J 加 COSX 与全解析 RHF 不是同一个数值方法，不参与接受或拒绝。

指定 `--report-dir` 时，Harness 除了覆盖最新的 `<任务名>.acceptance.json`，还会做两件不会互相覆盖的事：

- 把完整验收 JSON 复制到 `<report-dir>/runs/<candidate_id>.acceptance.json`；
- 向 `<report-dir>/experiments.jsonl` 追加一行。

`candidate_id` 是任务名加 UTC 时间戳。每一行包含 `candidate_id`、`task`、`command`、`status`、`failure_kind`、`duration_s`、`passed`、`reasons`、标量 `metrics` 和 `acceptance_path`。缺报告、数值失败、SCF 失败和性能拒绝都会留下 `passed: false` 与原因。stdout 和 stderr 留在 acceptance 副本里。

当前第一行记录是 `rhf-rijcosx-parallel-20261104T041258076806Z`，`passed` 为 true。更早一次被覆盖掉的性能拒绝没有补造。

这一页没有开始 `alk375` 生产规模预检，也没有加入 GPU 或后 HF 方法。

## 12. 水团簇尺度阶梯

长期目标是约 `50000` 个基函数、20 轮 SCF、收敛阈值 `1e-8`、48 小时内完成。`hf.py` 仍是参照实现。当前阶段先把可控的尺度阶梯放进 Harness，不生成新的 HF 算法。

任务 [`harness/tasks/water_cluster_ccpvtz_scale.json`](../harness/tasks/water_cluster_ccpvtz_scale.json) 使用 16、32、64、128 个水分子和 cc-pVTZ。一个水分子是 58 个基函数，四个团簇分别是 928、1856、3712、7424 个基函数。几何由 [`water_clusters.py`](../harness_engineering/water_clusters.py) 生成：氧原子位于间距 4.0 Å 的晶格上，16 为 2×2×4，32 为 4×2×4，64 为 4×4×4，128 为 4×4×8。

每个团簇构建参照用的 `Hartree_Fock`，记录原子数、基函数数、壳层数、电子数和八个稠密矩阵的内存估计，并计时核心哈密顿、重叠矩阵和一次广义对角化。16 与 32 使用 1 次预热加 3 次重复，64 与 128 只测 1 次。基函数数必须等于单个水的基函数数乘以分子数。尺度阶梯本身不调用 `build_jk`、`build_ri_j`、`build_cosx_k` 或 `solve_rhf`，也没有 speedup 门槛。

当前阶段优化这四个团簇上的一次 `get_jk`。任务 [`harness/tasks/water_cluster_ccpvtz_jk.json`](../harness/tasks/water_cluster_ccpvtz_jk.json) 使用 MINAO 密度、weigend、`df_tol=1e-9` 和 `30×110` 的 COSX 网格，只调用 `get_jk(jk_mode="rijcosx")`，不跑 SCF。Schwarz 改为分块 `(ij|ij)`；壳层数达到 128 后，距离超过壳层半径之和的对不再计算。RI-J 先丢掉壳层对，再按辅助壳层分块。辅助函数的筛选阈值是 `df_tol / nao`，避免很多小贡献加起来超过 `2e-8`。COSX 只对筛过的壳层对计算网格积分，不再分配 `(ngrids, nao, nao)`。16 个水必须完成：RI-J 对照 PySCF density fitting 的门槛是 `2e-8`，串行与 4 线程的 J、K 差分别不超过 `2e-8` 和 `2e-10`。单个水的解析 K 对照使用既有 COSX 门槛对应的 `40×146` 网格；团簇的 `get_jk` 计时仍用 `30×110`。32、64、128 走同一条路径；若按显著壳层对估计的耗时会超过任务剩余时间，就记下 `skipped_reason` 和估计值，不启动该档的 `get_jk`。这一阶段没有 speedup 门槛。

`alk375` / cc-pVTZ 仍是通往约 50000 基函数的后续大体系，不在这一阶段运行。

这一阶段完成后，下一阶段仍是密度矩阵初猜和对角化的时间与内存，并用这四个水团簇作为回归尺度。

完整数据和此前的优化过程见 [`harness_work_report_zh.md`](harness_work_report_zh.md)。


