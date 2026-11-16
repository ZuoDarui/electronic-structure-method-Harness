# electronic-structure-method-Harness

一个与具体方法分开的“生成—评测—反馈”闭环。第一项任务是闭壳层 Hartree–Fock：harness 绑定 **Claude Opus 5.5**，第 0 代只给接口和分子列表、不给任何程序，由模型从无到有写出 `solve(mol)`；之后每一代把当前最好程序及其逐分子误差、耗时比交回模型，让它继续改。精度真值是 PySCF `scf.RHF`，速度对照同一台机器上的基线 `baselines/hf.py`（集群上的 `hf.py`）。每一次模型调用的提示、回复、程序、评分都写进轨迹，供以后消融。

## 布局

- `esm_harness/search/session.py`：可续跑的会话。每一代写出提示文件，等模型写回回复文件，再打分、写下一代提示。
- `esm_harness/llm/prompts.py`：三档提示（消融因子 `prompt_id`）。`driver_wrapper.md`：发给模型的固定外层指令。
- `esm_harness/eval/`：静态策略、沙箱子进程、本机/Slurm 提交、打分。
- `baselines/hf.py`：速度与精度基线。换成集群上的程序时保持 `solve(mol) -> {"e_tot", "converged"}`。
- `benchmarks/hf_rhf.yaml`：固定的 7 个分子/基组（STO-3G 到 cc-pVDZ，最大 34 个基函数）；`benchmarks/reference_energies.json`：PySCF 参考能量。
- `configs/hf_rhf.yaml`：主搜索；`ablations/hf_rhf.yaml`：消融格子。
- `trajectories/`：真实运行产生的全部轨迹（见下文）。

## 模型绑定

配置里 `factors.model` 只能是 `claude-opus-5.5`，推理档位 `reasoning_effort` 取 `low|medium|high|xhigh|max`，对应调用的模型 `claude-opus-5-5-<effort>`。每个回复都要附一份 `meta/<cid>.meta.json`，写明回答它的模型和调用 id；`step` 发现模型名与会话绑定的不一致就报错，不打分。

模型调用通过 Cursor 子代理完成（计入 Cursor 账户用量池，不需要另开 API 密钥）。每个候选由一个全新的 Opus 5.5 子代理生成：它只被允许读取这一份提示文件、写入这一份回复文件，不能看评测代码、其他候选或执行任何命令（见 `esm_harness/llm/driver_wrapper.md`）。子代理调用不返回 token 计数，所以账本把这些调用记为 `unmetered_calls`，`usage` 为 `null`，不做估算。若改用会返回 usage 的接口，`meta --usage-json` 写进去的数字会原样进入账本。

## 运行

```bash
python -m esm_harness check-baseline            # 基线与 PySCF 误差需 ≤ 1e-8，否则不开始
python -m esm_harness init --config configs/hf_rhf.yaml
python -m esm_harness status --session trajectories/hf-opus55
# 对 status 列出的每个提示，用 claude-opus-5-5-<effort> 生成回复写到指定路径，然后：
python -m esm_harness meta --cid <cid> --model claude-opus-5-5-high --agent-id <id>
python -m esm_harness step --session trajectories/hf-opus55   # 打分并写下一代提示
```

`init` 先在本机（或 Slurm）上把基线跑一遍，记下逐分子能量与耗时（`baseline.json`），之后所有耗时比都相对这次测量。会话在 `max_generations` 代后结束，或连续 `patience` 代没有提高最佳适应度时提前结束。调用失败用 `meta --error` 记下（写入 `calls.jsonl`），然后重发同一个提示。

消融：

```bash
python -m esm_harness ablate-init --grid ablations/hf_rhf.yaml
python -m esm_harness ablate-step --grid ablations/hf_rhf.yaml   # 每答完一代调一次
python -m esm_harness report                                     # 写 trajectories/REPORT.md 和 summary.json
```

## 评分

适应度 `fitness = -log10(max(err, 1e-12)) - speed_weight * ln(geomean_time_ratio)`，`err` 是 7 个分子中相对 PySCF 的最大绝对误差，`geomean_time_ratio` 是候选耗时除以基线耗时的几何平均。任一分子未收敛或误差超过 `fitness_err_ceiling`（1e-6）时没有适应度。`meets_threshold` 另外标出误差 ≤ `max_abs_err`（1e-8）且耗时比 ≤ `max_time_ratio`（1.0）的候选。

沙箱规则（提示里也写明）：只能导入 numpy、scipy、pyscf.gto 和少数纯标准库模块；不能导入 `pyscf.scf`/`pyscf.df`，不能调用 `kernel()`，不能做文件读写、`exec`/`eval`。每次计时调用都重新加载程序模块并新建 `Mole`，模块全局变量里的缓存带不到下一次调用。计时取 3 次中位数，前面有一次不计时的预热。子进程环境去掉所有名字含 `API_KEY`/`TOKEN`/`SECRET`/`PASSWORD` 的变量。

## 轨迹

每个会话目录（`trajectories/hf-opus55/`、`trajectories/ablation/cell-*/`）包含：

- `session.json`：配置、因子、绑定模型、状态
- `baseline.json`：本次测得的基线能量、耗时及其相对 PySCF 的误差
- `prompts/`、`responses/`、`meta/`、`programs/`：每个候选的原始输入、原始回复、调用信息、抽出的程序
- `trajectory.jsonl`：每个候选一行，字段有 `factors`、`model`、`agent_id`、`from_scratch`、`prompt`、`response`、`code`、`usage`，以及 `eval`（状态、错误输出、逐分子 `energy`/`abs_err`/`abs_err_baseline`/`time_s`/`baseline_time_s`/`time_ratio`、`fitness`）
- `calls.jsonl`：每一次子代理调用，包括超时等失败
- `ledger.json`：按代汇总，列出最佳候选与调用计数

`trajectories/REPORT.md` 是所有会话最佳结果的一张表，`trajectories/ablation/ablation-ledger.json` 按格子汇总。

## 当前结果

主搜索 `hf-opus55`（`hf_minimal`、`iterative`、`speed_weight=0.5`、`high`，8 代）：第 0 代从零写出的程序在 7 个分子上全部收敛，最大误差 1.1e-13 Hartree，几何平均耗时为基线的 0.100；最佳为第 3 代，误差 4.4e-13，耗时比 0.060。它自己改用 8 重对称的双电子积分和打包的对指标超矩阵，用一次矩阵—向量乘积得到 J 和 K。第 4–7 代把耗时比压到 0.050，但误差升到 1.3e-11，按适应度不如第 3 代。各消融格子的数字见 `trajectories/REPORT.md`。

耗时比是相对仓库里这份 `baselines/hf.py` 在同一台机器上的测量；它在 `solve` 里做 MINAO 初猜并用 einsum 构造 J/K，所以生成程序能快一个数量级。换成集群上的 `hf.py` 或在集群上用 `submit: slurm` 运行时，`init` 会重新测基线，比值改以那份程序和那台机器为准。

## 测试

```bash
python -m pytest
```

测试里的 `StubClient` 只用于检查循环和账本格式，产生的不是模型数据。
