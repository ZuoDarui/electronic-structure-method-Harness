# 从空接口生成电子结构方法

新旧 harness 已统一整理到 `hf/harness_framework/`。以下模块命令和相对路径均从该目录执行；也可在外层通过 `python harness_framework/run.py generate ...` 或 `python -m harness_framework generate ...` 使用统一入口。生产 `hf.py`、`gpu_hf.py` 位于上一级，详细目录与旧优化入口见本目录 `README.md`。历史实验按原字节迁入 `outputs/`，原绝对路径和身份不改写，不应直接恢复旧实验。

这是独立于原有 HF 程序的新生成框架。它由模型生成 `solution.py`，执行器运行候选，可信验证器重新计算所有科学误差。模型和候选均不能修改参考或验收门槛。已有 `hf.py`、`gpu_hf.py` 和旧 harness 命令保持兼容。

首版覆盖闭壳层、全电子、实数双精度分子 RHF，支持 exact 和 DF，以及 NumPy/CuPy。PySCF 只在可信进程提供积分和独立 SCF 参考；生成的候选容器不安装 PySCF。本框架没有承诺一次生成必然成功：失败、预算耗尽及缺少运行环境都是明确结果。

## 安装与执行环境

控制器需要 Python 3.11–3.13 和仓库 `requirements.txt`。Responses API 客户端用标准库实现，无需另装 OpenAI SDK。当前本机环境可用：

```bash
PYTHON=/Users/darui/workspace/fieldquantum/.pyscf-env/bin/python
"$PYTHON" -m harness_engineering.generation --help
```

执行候选需要 Linux Docker 和已构建的候选镜像。可在本机 Docker Linux VM 或配置好的 SSH 服务器上执行；本机缺少 Docker 时不会退回宿主执行。GPU 端还需要 NVIDIA 容器运行时和 Nsight Systems（`nsys`）。镜像构建时可联网，候选运行时断网。

在 Linux 执行机上显式构建镜像：

```bash
docker build -f containers/Dockerfile.cpu -t hf-harness-cpu:local .
docker build -f containers/Dockerfile.gpu -t hf-harness-gpu:local .
```

GPU Dockerfile 默认没有安装 Nsight Systems。严格验收需要选择已有 `nsys` 的可信基础镜像，或把执行机上的既有 Nsight 安装目录只读挂到容器中，例如在执行配置中加入：

```json
{
  "nsys_path": "/opt/nsys/bin/nsys",
  "nsys_mount": {"source": "/opt/nvidia/nsight-systems", "target": "/opt/nsys"}
}
```

路径必须是执行机实际安装路径。挂载的安装应与镜像 ABI 和驱动兼容。预检会检查 Linux、非 root 容器、数值库、CUDA 双精度求解、镜像身份和 `nsys` 是否存在；真正的阶段跟踪仍需候选运行成功才能确认。镜像不会自动拉取，执行时绑定实际 image content ID。

候选容器使用无网络、只读根目录、去除 capabilities、非 root、内存/CPU/进程数/文件大小限制；仅候选、输入和可信 worker 只读挂载，输出目录可写。远端执行仅发送这三部分及求解选项，不发送参考、验证器或 API 密钥。

执行器会核对 Docker 实际创建的隔离策略，再决定是否启动；提交了 `--network=none` 等参数不构成这些限制已经生效的证据。某些托管 Docker 会静默改写参数，这类运行时必须拒绝。

CPU 也支持显式选择非特权 Linux 命名空间容器：执行配置设置 `cpu_executor: "namespace"`，并提供可信 CPU 镜像导出的根文件系统及其冻结清单。候选仍执行镜像内 Python，具有独立 user/mount/network/PID/IPC/UTS 空间、只读根目录和输入、私有临时输出；它不是直接运行宿主 Python。当前开发机禁止新挂载 `/proc`，采用显式 `cpu-namespace-noproc/1` 协议：候选看不到任何 `/proc` 内容，启动器和镜像内探针用实际命名空间观测及系统调用核对隔离和权限。只有完整探针及数值计算实测通过才合格。该后端的内存限制是每进程地址空间上限，进程数限制按真实用户计数，CPU 由亲和性限制；不宣称等同于 Docker cgroup 的进程树总内存硬上限。

## 单次生成 + 小分子准确性验收（`generate single`）

这是当前推荐的生成步骤：**每次运行只生成一个 HF 程序**，只在小分子测试集上验收数值准确性。这一步不跑大分子，不做速度或加速比验收，不进入 GPU 和 Nsight 阶段，也不消耗保留集。

流程：冻结任务 → 可信进程用 PySCF 为每个小分子准备积分和参考解（exact 与 DF）→ 模型请求 **一次**，得到一个 `solution.py`（同一请求的网络重试不算新程序）→ 逐个用例在 CPU/NumPy 上运行 → 可信验证器重新计算全部误差 → 给出 `accepted`、`rejected`、`generation_failed`、`invalid_task` 或 `environment_unavailable` 之一。没有自动修复轮次。想再试一次时，新开一个输出目录，并用 `--previous-run` 把上一次的公开诊断（指标与原因，不含参考值）和源码交给模型。

测试集在 `harness/generation/rhf_small_accuracy.json`：15 个闭壳层小分子（H2、拉伸 H2、HeH+、He、Be、LiH、HF、H2O（6-31G* 和 cc-pVDZ）、NH3、CH4、N2、CO、C2H4、H2CO），每个分子跑 exact 和 DF 两种模式，共 30 个用例，nao 不超过 28。任务里的 `accuracy_only.max_atoms`（默认 8）和 `max_nao`（默认 40）是硬上限，超出的任务直接判为 `invalid_task`。验收门槛沿用完整的 RHF 指标协议：能量 1e-8 Eh、试探 J/K 1e-9、密度 1e-6、FDS-SDF 1e-7、正交性 1e-10 等。

```bash
cd hf/harness_framework
PYTHON=/Users/darui/workspace/fieldquantum/.pyscf-env/bin/python   # 集群上用 miniforge3/envs/pyscf/bin/python

# 生成一个程序并验收（需要 OPENAI_API_KEY）
"$PYTHON" run.py generate single --output outputs/single_01 \
  --model gpt-6.1-sol --reasoning-effort max \
  --executor-config /absolute/path/executor.json      # docker 或 namespace，隔离执行

# 基于上一次失败的诊断，再生成一个新程序
"$PYTHON" run.py generate single --output outputs/single_02 --previous-run outputs/single_01 \
  --model gpt-6.1-sol --executor-config /absolute/path/executor.json

# 不调用模型，只验收已有的 solution.py
"$PYTHON" run.py generate single --output outputs/check_ref \
  --candidate tests/fixtures/rhf_numpy_solution.py --local-subprocess --python "$PYTHON"
```

执行器：`--executor-config` 可以沿用原来的 docker 或 namespace 配置（只用 CPU 部分，GPU 预检被强制关闭）。`--local-subprocess` 会在宿主机上用独立的 Python 子进程运行候选，**不是安全沙箱**：候选以当前用户身份运行，防护只有 AST 导入守卫、清空的环境变量、私有工作目录、超时，以及 CPU 时间、文件大小和地址空间（Linux）限制。它需要显式开启，适合在没有 Docker 或 namespace 运行时的机器上做开发验收。

每次运行的输出包括：`report.md`（逐个用例的能量对比、误差表和最坏指标）、`result.json`、`cases/<id>/`（原始输出和验证记录）、`private/references/`（PySCF 参考，不会交给模型或候选）、`generation/`（请求与响应证据），以及 `manifest.json`。通过时，源码复制到 `accepted/solution.py`。`wall_seconds_informational` 只作记录，不参与判定。

用 `tests/fixtures/rhf_numpy_solution.py`（参考实现）在本机验证：30/30 通过，最大能量误差 1.6e-12 Eh，全程约 19 秒。原有的 `generate run/resume/evaluate` 多阶段流程保持不变。

### 生成模型：Claude（默认）

`generate single` 默认使用 Claude（官方 `anthropic` Python SDK，Messages API）：模型 `claude-opus-5-5`，自适应思考，`effort` 默认 `xhigh`，流式输出，`max_tokens` 64000。结构化输出用 JSON schema 约束，固定返回 `{solution_py, rationale, expected_effect}` 三个字段，再经过与原来相同的 `validate_bundle` 导入和 AST 检查。请求默认带上 `fallbacks: "default"`（beta `server-side-fallback-2026-07-01`），被安全策略拒绝时由服务端自动换用推荐模型重试；如果走不支持这个 beta 的代理，用 `--no-refusal-fallback` 关闭。密钥只从环境变量读取（默认 `ANTHROPIC_API_KEY`），不会写入任何证据文件。

```bash
ANTHROPIC_API_KEY="$(cat ~/secrets/anthropic_api_key)" "$PYTHON" run.py generate single \
  --output outputs/single_claude_01 --local-subprocess --python "$PYTHON"
# 可选：--model claude-fable-5-1  --reasoning-effort max  --api-base-url https://<代理>/  --api-key-env NAME
```

`--provider openai` 保留原来的 Responses API 路径（需要 `--model` 和 `OPENAI_API_KEY`）。

## 启动、恢复与复核

从 `harness/generation/executor.local.example.json` 或 `executor.ssh.example.json` 复制执行配置，填入实际镜像、GPU 和 SSH 信息。SSH 使用批处理密钥登录及严格主机密钥检查；可通过 `ssh_options` 指定私钥和单独的 known-hosts 文件，不能在配置里放密码、API 密钥或私钥内容。

在控制器的环境设置 `OPENAI_API_KEY`。模型显式指定，不自动换模型或降推理档位。本次用户确认的设置为 `gpt-6.1-sol` / `max`；此前服务对 `ultra` 的明确拒绝另有原始记录。

```bash
"$PYTHON" -m harness_engineering.generation run \
  --task harness/generation/rhf.json \
  --holdout harness/generation/holdout_rhf_v1.json \
  --output outputs/rhf_campaign_01 \
  --model gpt-6.1-sol --reasoning-effort max \
  --executor-config /absolute/path/executor.json
```

使用已授权的其他 Responses 服务时，显式添加 `--api-base-url https://your-service.example/v1`；密钥仍通过环境变量传入。服务必须支持指定模型、Responses 和严格结构化输出。拒绝、截断、HTTP 错误及未完成响应均保存证据，不会作为代码应用。完成响应还必须给出合法 token 用量；报告输出超过请求上限时判为 `output_budget_exceeded`，控制器停止等待处理，不重试或自动生成下一轮。请求参数不能保证供应商正确执行计费上限，框架只能记录并拒绝不合格响应。

服务支持流式 Responses 时可添加 `--stream`。传输在独立子进程中接收有界 SSE，只接受完整 `response.completed` 中的响应及用量，再执行同一预算和代码检查；文本增量、文本结束或 `[DONE]` 均不能替代完成。600 秒是覆盖连接、响应头和内容的绝对期限，超时或取消会终止传输进程。进度只记录事件类型、序号和耗时，不记录私有推理文本。流式方式不保证第三方服务能及时生成完整结果。

默认的非流式 JSON 请求也使用同一受控子进程、绝对截止时间和取消清理。持续缓慢返回数据不能把请求延长到总时限之外；JSON 请求不会被加上 `stream=true`。完整 JSON 仍需通过原有状态、用量、拒绝及代码检查。无效或过大的 JSON 保留有上限的失败前缀，HTTP 错误继续保留状态和有上限的脱敏错误正文，重试规则保持原样。

传输诊断通过容量上限为 2 MiB 的匿名通道及时写出，超时或取消后仍可收集：连接阶段（DNS/TCP/TLS 合并观测）、请求发送、HTTP 状态和内容类型类别、首字节、首末事件及字节/事件计数。未知事件类型统一折叠；正文、推理文本、密钥和任意响应头不进入诊断。`response.json` 中的 `received=false` 表示未取得完整 Response；是否已收到部分字节或事件要查看 `transport_metadata`。取消时若归档磁盘不可写，仍保留原中断，不自动重试。

```bash
"$PYTHON" -m harness_engineering.generation resume \
  --campaign outputs/rhf_campaign_01

"$PYTHON" -m harness_engineering.generation evaluate \
  --campaign outputs/rhf_campaign_01
```

`resume` 从持久状态继续，重用已完成且身份完整的计算；不会偷偷重发失去结果的 API 请求，该请求占用一轮。`evaluate` 在不调用模型、不重新运行候选的情况下，核对完整清单并从归档原始矩阵重新计算判决。它不是重新暴露保留集给模型的调参入口。

环境不合格返回退出码 2 和 `waiting_for_environment`；科学拒绝或预算耗尽返回 1；只有所有必需关卡均通过并晋升为 `accepted` 时返回 0。新实验必须使用空输出目录。恢复不允许修改已冻结的任务、执行配置、模型配置、验证器、已完成证据或候选快照；环境定义确需改变时新建实验。

需要先做 CPU 部分时，可在执行配置显式设置 `defer_gpu_preflight: true`，先完成合格的 CPU 开发与属性关卡，再在 GPU 关卡检查环境。若用户要求暂不进行任何 GPU 工作，另设置 `stop_before_gpu: true`；此时连 GPU 预检都不会调用，CPU 完成后状态为 `waiting_for_gpu_authorization`。普通 `resume` 保留该停止边界，只有未来明确允许后的 `resume --enable-gpu` 才继续 GPU；最终科学验收门槛保持不变。

## 自动生成闭环

1. 冻结公开规格、保留集、控制器/验证器源码和配置，创建只有两个 `NotImplementedError` 接口的初始源码。
2. 预检 CPU 和严格 GPU 执行环境。没有合格容器时不消耗模型生成预算。
3. 在独立可信子进程准备公开积分、解析后的物理输入身份和私有参考。
4. 请求模型返回严格 `PatchBundle`，只允许替换一个 `solution.py`。记录修改依据、预期效果、原始响应、用量、补丁和前后哈希；原子应用到新轮次快照。
5. 依次运行 CPU 开发集、CPU 属性挑战和 GPU 开发集。候选失败时将开发反馈发给模型，保存失败候选并实际恢复最后 CPU 正确版本；下一轮可以针对失败源码修复。
6. 开发全部通过后冻结候选，只进行一次 CPU/GPU 保留集验收。失败后结束实验，不把保留集结果用于修复；另一次研究必须使用新版本保留集。实验父目录的 `.holdout-assessments/` 按解析后的物理输入登记使用情况，改文件名或版本标签不能重复使用。迁移实验应保留此登记；有意换目录绕过登记不构成新的盲测。
7. 所有关卡通过后复制冻结源码到 `accepted/solution.py`，生成证据清单和可审阅报告。

默认最多 8 轮，每次输出上限 16,384 tokens，单次请求超时 600 秒（可用 `--api-timeout` 缩短）；429、5xx 和网络临时错误最多重试两次，代码或数值失败不会伪装成网络重试。没有收到用量的请求不能推断为零费用，记录的用量仅为服务实际返回部分。状态文件采用原子替换及单写锁。每个已完成轮次和计算的证据立即固定哈希，恢复前检查；执行器已收集的回执可以恢复，未完整收集的输出会保留为 `interrupted-output-*` 再使用新目录执行。

2026-11-12 用户明确授权本轮使用扩展预算。新实验可显式添加 `--budget-profile extended --max-output-tokens 32768 --api-timeout 1200`，或在控制器配置中记录同名字段。只有 `extended` 允许这些较高上限；省略配置时仍执行原 `standard` 上限。选择扩展配置本身不会自动提高请求值，实际请求和用量判决继续绑定冻结配置。旧实验及旧超预算响应不因本轮授权而改变身份或判决；科学门槛、最多 8 轮和 GPU 停止边界也保持原样。

## 候选接口与科学验收

候选只实现：

```python
build_jk(problem, density, backend)  # -> J, K
solve(problem, options, backend)    # -> dict of raw results
```

`backend` 是 `numpy` 或 `cupy` 字符串。`problem` 包含 `overlap`、`hcore`、`enuc`、`nelectron`、`metadata`，及 `eri` 或 `df_factors`；积分使用 chemist 索引与自旋求和密度。完整公式和输出契约在生成请求中固定。源码 AST 检查只作前置约束，执行安全边界来自容器。

开发集为六个体系 × exact/DF 共 12 组，保留集为四组冻结的扰动结构 × exact/DF 共 8 组。解析后的基组、几何、辅助基组、电子数、方法和数值约定被记录并绑定输入；改名或更改坐标单位不能把相同开发输入冒充为保留集。

附加 CPU 属性关卡有五次执行：旋转/平移后的多原子分子在 exact/DF 各一次、同一 DF 输入按辅助指标分块 1 和 7 各一次、代表体系限制为单次迭代一次。常规案例要求最终收敛；单次迭代记录明确只检验终止及物理状态一致性，不要求接近收敛参考的能量、密度或驻点残差。

可信验证器重算能量、J/K、电子数、幂等性、密度重构、正交性、轨道方程和 SCF 残差。主要门槛为能量 `1e-8 Eh`、J/K `1e-9`、密度 `1e-6`；其余阈值固定在任务文件。exact 和相同辅助基组的 DF 分别与相匹配参考比较。共享 PySCF 积分引擎，不能据此声称独立认证了积分算法。

这里的 `exact` 指使用完整四中心积分的 RHF。它的参考能量来自可信子进程独立运行的 PySCF RHF；DF 参考来自相同辅助基组的 PySCF DF-RHF。这些都是指定有限基组中的 Hartree–Fock 参考能量，不代表包含全部电子相关的精确多体能量。

空报告、漏案例、重复案例、未知状态、非有限值、错误类型及未收敛都不能获得常规验收通过。对候选 NPZ 先检查压缩包、NPY 头部、形状和解压字节上限，再读取数组，避免候选输出拖垮验证进程。

开发 GPU 使用阶段观测辅助反馈；最终 GPU 强制归档 Nsight CUDA/NVTX 追踪，四个 J/K/SCF 阶段均需有同线程关联的实际 CUDA kernel，SCF 还需观测设备本征求解。该证据证明阶段内设备计算，不能证明恶意候选没有穿插无用 GPU 工作；独立数值验收和受保护评测共同构成可信边界。当前不作 GPU 加速结论。

## 如何审阅

每次实验包含 `report.md`、`state.json`、`events.jsonl`、`manifest.json`；`rounds/` 保存 API 请求响应、源码、差异、回滚、执行回执、逐轮历史、原始矩阵、逐项指标和 GPU 追踪。`private/` 保存参考及保留集，永不挂入候选容器或加入模型上下文。不要把带本地运维信息的完整实验目录自动发布。

方法成熟度有三个不同结论：框架测试通过、模型成功生成代码、生成代码通过独立科学及 GPU 验收。只有最后一项可以产生已接受的方法实现。2026-11-11 的阶段记录见 `docs/generation_framework_implementation_zh.md`，后续开发机及 CPU 记录见 `docs/generation_cpu_validation_20261112_zh.md`。此前旧 `gpu_hf.py` 的验证不代替新框架的生成验收。

2026-11-12 授权扩展预算后的独立实验见 `docs/generation_cpu_extended_20261112_zh.md`：235 项回归及 CPU 环境复核通过，但 SSE 协议序号失败、普通 JSON 网络错误均未产生合法完整候选。两段实验未进入候选科学验收，GPU 继续暂停。该记录保留实际请求、失败诊断、未知用量、离线身份复核和证据清单。
