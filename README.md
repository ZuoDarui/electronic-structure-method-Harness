# electronic-structure-method-Harness

一个与具体方法分开的“生成—评测—反馈”闭环。第一项任务是闭壳层 Hartree–Fock：harness 绑定 **Claude Opus 5.5**，第 0 代只给接口和分子列表、不给任何程序，由模型从无到有写出 `solve(mol)`；之后每一代把当前最好程序及其逐分子误差、耗时比交回模型，让它继续改。精度真值与速度基线都是 PySCF `scf.RHF`（`baselines/pyscf_rhf.py`，默认设置），在同一台机器上计时。每一次模型调用的提示、回复、程序、评分都写进轨迹，供以后消融。

## 布局

- `esm_harness/search/session.py`：可续跑的会话。每一代写出提示文件，等模型写回回复文件，再打分、写下一代提示。
- `esm_harness/llm/prompts.py`：三档提示（消融因子 `prompt_id`）。`driver_wrapper.md`：发给模型的固定外层指令。
- `esm_harness/eval/`：静态策略、沙箱子进程、本机/Slurm 提交、打分。
- `baselines/pyscf_rhf.py`：速度与精度基线，PySCF `scf.RHF` 默认设置（MINAO 初猜、DIIS、`conv_tol=1e-9`）。基线属于受信代码，不受候选程序的导入限制；换别的基线时用配置项 `baseline` 指向一个 `solve(mol) -> {"e_tot", "converged"}`。
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
# 对 status 列出的每个提示，启动一个全新的 claude-opus-5-5-<effort> 子代理，
# 把该请求的 instructions 字段原样发给它；它写完回复后：
python -m esm_harness meta --cid <cid> --model claude-opus-5-5-high --agent-id <id>
python -m esm_harness step --session trajectories/hf-opus55   # 打分并写下一代提示
```

`init` 先在本机（或 Slurm）上把基线跑一遍，记下逐分子能量与耗时（`baseline.json`），之后所有耗时比都相对这次测量。会话在 `max_generations` 代后结束，或连续 `patience` 代没有提高最佳适应度时提前结束。调用失败用 `meta --error` 记下（写入 `calls.jsonl`），然后重发同一个提示；同一提示失败达到 `max_call_failures` 次（默认 3）后，该候选以 `call_failed` 记入轨迹（无适应度），会话继续下一代。

消融：

```bash
python -m esm_harness ablate-init --grid ablations/hf_rhf.yaml
python -m esm_harness ablate-step --grid ablations/hf_rhf.yaml   # 每答完一代调一次
python -m esm_harness report                                     # 写 trajectories/REPORT.md 和 summary.json
```

重新计时：

```bash
python -m esm_harness rescore --baseline baselines/pyscf_rhf.py  # 所有会话的已记录程序对新基线重测
```

`rescore` 不改原轨迹，只在每个会话下写 `rescore/<基线名>.baseline.json`、`rescore/<基线名>.jsonl`（每个候选一行，含原适应度和新适应度），并写汇总表 `trajectories/RESCORE_<基线名>.md`。

## 按轮运行（主代理 + 子代理 + 监控）

分工：

- **主代理**：每一轮开一个新的 Cursor 云端代理，它只活一轮，按 `LOOP.md` 执行：`round begin` 加锁并打开 `loop.yaml` 里的会话 → 每个待答提示各启动一个全新的 Opus 5.5 子代理（并行）→ 逐个 `meta` 记录 → `round end` 打分、写下一代提示、写轮报告 `trajectories/rounds/round-NNN.json`、解锁 → 推送后结束。一轮让每个会话前进一代。
- **子代理**：只读自己的提示、只写自己的回复（同前）。
- **监控代理**：在轮次进行时只读，用 `monitor` 查看锁、各候选的回复/meta/调用记录，可附上平台对话看每个子代理的模型和是否已写出回复；轮次之间（`loop-state` 返回 0）才修改 harness，轮后用 `audit` 核查。

主代理必须是顶层云端代理：Cursor 的子代理没有再启动子代理的工具，所以“主代理再分出子代理”这一层只能由顶层代理承担。

每轮的主代理在不同的虚拟机上运行，所以 `round end` 打分前会在本机重测一次基线（本轮 `begin` 时刚测过就不重测），旧的测量移到 `baseline_history.jsonl`，轮报告里记下这次基线的测量时间和主机。每条轨迹记录里本来就存有逐分子的 `baseline_time_s`，比值所用的基线可以追溯。

锁 `LOOP_LOCK.json` 记录启动该轮的代理 id 和 harness 文件（`esm_harness/`、`configs/`、`ablations/`、`baselines/`、`benchmarks/`、`tests/`、`pyproject.toml`、`loop.yaml`、`LOOP.md`）的指纹。锁存在时：别的代理不能开始新的一轮；指纹一变，`round end` 和 `step` 都拒绝打分。

开一轮：在 Cursor 里对本仓库新建一个云端代理，发送

```text
按仓库 cursor/hf-harness-66dd 分支上的 LOOP.md 执行一轮 electronic structure harness 循环，只执行一轮，完成后报告并结束。
```

监控（只读）：

```bash
python -m esm_harness loop-state          # 0 = 空闲，可以改 harness；3 = 有一轮在进行
python -m esm_harness monitor             # 锁、指纹是否变化、每个待答候选的状态、历次轮报告
python -m esm_harness monitor --driver-transcript <主代理 transcript.json> --transcripts <子代理对话目录>
```

暂停：把 `loop.yaml` 里的 `paused` 改成 `true`，之后定时触发的主代理执行 `round begin` 会被拒绝，直接退出，不提交任何东西；已经开始的那一轮照常完成。改回 `false` 即恢复。

`loop.yaml` 目前只包含以 PySCF 为基线、从第 0 代重新开始的主搜索 `hf-opus55-pyscf`（8 代，即最多 8 轮）。

## 审计

`meta` 和 `calls.jsonl` 由驱动者（在 Cursor 里运行 harness 的那个模型）填写，harness 本身只能看到文件。`audit` 拿平台记录的对话来核对：驱动者的对话里有每次启动子代理时请求的模型、发出的指令和返回的子代理 id；每个子代理的对话里有它收到的指令、每次工具调用和它创建的文件内容。

```bash
python -m esm_harness audit-ids --driver-transcript <驱动者 transcript.json>   # 列出要取对话的子代理 id，每行 50 个
# 用 Cursor 的 batch-fetch-details（includeTranscripts）取回这些对话，然后：
python -m esm_harness audit --driver-transcript <驱动者 transcript.json> --transcripts <取回目录> --archive
python -m esm_harness audit --strict                                            # 之后只用仓库里的存档复核
```

逐个候选检查，任何一条不符都记为 violation：

- 每个被打分的候选都对应驱动者对话里的一次启动，请求的模型等于会话绑定的模型，指令与 `driver_wrapper.md` 渲染结果逐字相同，子代理 id 只用于这一个候选；
- 子代理只读了自己的提示文件、只写了自己的回复文件，没有用其他工具；
- 打分用的回复与子代理写出的内容 SHA-256 相同（从子代理创建文件的 diff 还原），即驱动者没有改过回复；
- 每个候选的启动次数等于 `calls.jsonl` 的记录数（没有未记录的调用），记为失败的那些启动确实没有写出回复（没有挑结果）。

结果写入每个会话的 `audit.json` 和总表 `trajectories/AUDIT.md`。`--archive` 把用到的启动记录（`launches.jsonl`）和子代理对话（`agent_transcripts/`）存进会话目录，其中也有模型的思考过程，可用于消融分析。驱动者对话本身不存档，因为它包含整段会话。

审计的前提是对话由平台写入、驱动者改不了。存档进仓库以后，复核只能说明存档内部一致；要排除存档被改过，就用 `batch-fetch-details` 重新取一次对话，再用 `--driver-transcript`/`--transcripts` 审计。

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
- `audit.json`、`launches.jsonl`、`agent_transcripts/`：审计结果、驱动者的启动记录、每个子代理的完整对话（见“审计”）

`trajectories/REPORT.md` 是所有会话最佳结果的一张表，`trajectories/ablation/ablation-ledger.json` 按格子汇总。

## 当前结果

下面的搜索是在把基线换成 PySCF 之前做的：当时每一代反馈给模型的耗时比都相对旧的占位基线 `baselines/hf.py`（已从仓库删除，仍留在 git 历史里），所以 `trajectory.jsonl`、`REPORT.md` 里的耗时比和适应度都是相对那份程序。用 `rescore` 把全部 48 个已记录程序对 PySCF `scf.RHF` 重新计时后（`trajectories/RESCORE_pyscf_rhf.md`），结论如下。PySCF 在这 7 个分子上每个耗时 12–33 ms，主要是建对象、MINAO 初猜（逐原子 SCF）和各项检查的固定开销；分子这么小时，生成程序省掉这些开销，就快了一个数量级以上。分子变大后积分与 J/K 构造占主导，这个比值会改变。

主搜索 `hf-opus55`（`hf_minimal`、`iterative`、`speed_weight=0.5`、`high`，8 代）：第 0 代从零写出的程序在 7 个分子上全部收敛，最大误差 1.1e-13 Hartree，几何平均耗时为 PySCF 的 0.112；最佳为第 3 代，误差 4.4e-13，耗时比 0.074。它自己改用 8 重对称的双电子积分和打包的对指标超矩阵，用一次矩阵—向量乘积得到 J 和 K。第 4–7 代把耗时比压到 0.057，但误差升到 1.3e-11，按适应度不如第 3 代。

消融（每格 4 代，单因子相对主配置变化，全部从零开始）：所有格子第 0 代都写出了误差 ≤ 2e-13 的正确程序。按适应度选出的最佳程序相对 PySCF 的耗时比：`max` 档 0.059（第 0 代）、显示基线源码 0.062、best-of-3 0.064、`speed_weight=2` 0.064、给出方程 0.071、默认 0.083、`medium` 档 0.098、`speed_weight=0` 0.125（只奖励精度时误差已触到 1e-12 上限，适应度不再变化，速度也就没有改进）。`max` 档的调用常超出子代理时限：第 1 代 3 次调用都超时，记为 `call_failed`；第 3 代前两次分别超时、被 Anthropic 使用策略拦截，第三次才返回。这些失败都写在该格的 `calls.jsonl` 里。注意“显示基线源码”这一格当时给模型看的是旧 `hf.py` 的源码。

审计（`trajectories/AUDIT.md`）：54 次子代理启动全部在平台记录里，请求的模型都等于各会话绑定的模型（`high`/`medium`/`max`），每个子代理只用于一个候选；47 个被打分的回复与子代理写出的内容逐字节一致，7 次失败启动都没有写出回复，启动次数与 `calls.jsonl` 一一对应，违规 0 条。备注 5 条：两个子代理第一次写文件失败后重写；其中一个为了满足写前须读的工具规则，读了一次自己的回复文件；另有一次启动发生在主搜索重新开始之前，属于被丢弃的旧会话，不计入。

以后用新配置 `init` 的会话会直接对 PySCF 计时，反馈给模型的耗时比也以 PySCF 为准。在集群上用 `submit: slurm` 运行时，基线在集群节点上重新测量。

## 测试

```bash
python -m pytest
```

测试里的 `StubClient` 只用于检查循环和账本格式，产生的不是模型数据。
