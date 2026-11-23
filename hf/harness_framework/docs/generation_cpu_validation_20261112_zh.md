# 2026-11-12：生成框架的开发机与 CPU 验证记录

当前结论：**CPU 隔离执行环境已实测合格，补齐诊断及默认 JSON 总时限后框架共 230 项回归通过；生成服务连续超时，尚无本次新候选进入数值验收。** 按用户要求，GPU 预检、GPU 计算和最终 GPU 验收均暂停。

## 真实运行时检查与修正

SSH 密码仅用于建立本次连接，没有写入源码、执行配置或验证报告。

开发机 Docker 的实际行为与请求参数不同。最初的非 root、写入失败、网络超时探针不足以证明完整隔离；后续 `docker inspect` 发现网络被改为宿主网络，只读根目录、权限限制和部分资源参数也被静默改写。因此撤销原 CPU Docker 合格结论，并强化为先核对实际容器策略、合格后才启动。相应失败证据保留在 `outputs/generation_remote_20261112/`。

发现此问题时，第一轮实验仍处于模型生成阶段，已中断控制器，候选执行次数为 **0**。原始记录保留在 `outputs/generation_campaign_20261112/`，不作为任何科学验收证据。已发送但未收到完整结果的请求可能产生费用，不能记为零用量。

CPU 改用显式 `cpu-namespace-noproc/1` 容器协议。它执行可信 CPU 镜像导出的 Python，与直接运行宿主 Python 不同。真实检查已确认：

- user、mount、network、PID、IPC、UTS 六类命名空间均与宿主不同，可信启动器在新 PID 空间中为 PID 1。
- 镜像内进程为非 root，五类 capability 集全部清零，`NoNewPrivs=1`，额外继承文件描述符为空。
- 根目录及候选、输入、worker 只读；临时目录和输出各使用有容量上限的私有 tmpfs。
- 内核禁止新挂载 `/proc`，因此候选的 `/proc` 完全为空且只读；没有挂入宿主进程目录、共享 home、Docker socket、参考、验证器或密钥。
- 网络空间中只有 loopback，外网连接失败；NumPy/SciPy 的实际导入路径、原生扩展哈希与双精度本征求解均经过实测。
- 内存限制为每进程地址空间上限，进程数按真实 UID 限制，CPU 使用亲和性；这不等同于 cgroup 的进程树总内存硬限制。

可信两轨道测试样例也通过了输入传输、隔离执行、worker 输出、产物收集和哈希检查。该样例仅证明执行基础设施，不是新生成 RHF 的科学验收。

## 新的 CPU 专用实验

实验目录：`outputs/generation_cpu_campaign_20261112/`。模型为用户确认的 `gpt-6.1-sol / max`，从两个 `NotImplementedError` 接口开始，输出 token 上限仍为 16,384，超上限响应继续拒绝。旧的超预算候选没有被复用。

执行配置使用已实测的命名空间容器，并设置 `stop_before_gpu=true`。预期依次完成 12 组 CPU 开发计算和 5 项 CPU 属性挑战；完成后停在 `waiting_for_gpu_authorization`。普通恢复也不会发起 GPU 预检，保留集尚不会用于调参或最终验收。

生成前明确了真实目标运行库版本、总输出预算与大数组分配前的内存估算要求。另在任何候选执行前，预先修正了单轮未收敛契约：非驻点的占据空间不一定具有最终 Fock 的全局 Aufbau 能量顺序，因此该专用契约不认证最终谱排序；电子数、幂等性、正交性、密度/能量重构与 J/K 仍严格检查。普通收敛验收的所有门槛保持不变。

本次框架新旧回归为 **181 项通过**，包含 Docker 策略被改写的拒绝测试、命名空间协议、CPU 停止边界和科学验证器测试。日志为 `outputs/generation_remote_20261112/regression_tests.log`。

## 生成服务的实际阻断

合格 CPU 环境上的非流式实验连续两次在 600 秒后超时，第三次等待被中断，未获得候选。原始文件均保留。模型列表接口可达并列出指定模型，但这不能证明完整生成请求能及时完成。

依据官方 Responses 流式协议，随后增加显式 `stream=true`：独立传输子进程接收有界 SSE，仅将 `response.completed` 中的完整响应交给原预算、结构和代码验收。文本增量或文本结束事件不作候选；超时和取消均终止传输进程；模型、`max` 档位和 16,384 总输出上限保持不变。新旧回归增加为 **203 项通过**，日志为 `regression_streaming_tests.log`。

新的流式 CPU 实验位于 `outputs/generation_cpu_sse_campaign_20261112/`，本轮关闭网络自动重试。该真实请求也在绝对 600 秒期限内未获得完整可验收响应，状态为 `waiting_for_environment`，原因属于生成传输超时，阶段停在 `generate`。没有生成或应用候选代码，CPU 科学执行 **0 次**，GPU 执行 **0 次**，保留集没有进入候选验收。离线复核确认所有已归档身份完整，同时明确 `accepted=false`。

本日三段记录共发送 6 次生成请求：Docker 判断撤销前 2 次、合格 CPU 的非流式实验 3 次、流式实验 1 次。4 次返回超时判定，另 2 次被中断；均没有获得服务完整用量。**不能据此推断零费用，也不能把未知用量写成零。** 此前 2026-11-11 服务曾报告 23,663 个输出 tokens，超出原请求的 16,384 上限，这一历史拒绝记录仍保留，没有在本次偷偷复用该候选或放宽预算。

当前需要解决生成服务长请求与预算的兼容性，或由用户明确批准调整预算和等待时间；在取得该决定前保持原约束，不追加模型请求。CPU 验收底座已经可用，之后仍需真实生成代码、完成 12 组 CPU 开发计算及 5 项属性检查，才能宣称 CPU 开发门槛通过。GPU 工作继续保持暂停。

## 超时后的本地诊断修订

旧流式实验的证据仅证明约 600 秒时触发本地绝对期限，未取得完整 Response。旧版诊断在传输子进程正常结束时才交回，超时杀死子进程后可能丢失中途记录，因此 `received=false` 不能解释成“一个字节也没收到”。现有证据无法区分服务排队、模型推理、代理缓冲或连接停滞，也没有足够证据认定流式协议不兼容；未发现可以确定修正的请求构造错误。增加 token 或等待时间能否解决问题仍然未知。

在不新增真实请求的前提下，已增加容量上限为 2 MiB 的匿名诊断通道，及时保存连接（DNS/TCP/TLS 合并阶段）、请求发送、HTTP 状态与内容类型类别、首字节、首末事件及字节/事件计数。超时或取消后会收集完整诊断行，未知事件类型折叠，正文、推理文本、密钥和任意头部不进入诊断。取消时写盘失败不会覆盖原中断；对外错误堆栈隐藏未经清理的供应商异常内容。

新旧回归共 **214 项通过**，包含真实本地子进程写诊断后停滞的超时模拟、取消与通道清理、截断及大小上限，以及重试和错误归档。仅使用模拟服务，没有新增真实模型请求、开发机连接或真实 GPU 计算；模型、`max` 档位、16,384 tokens 和 600 秒上限保持原设置。测试日志见 `regression_diagnostics_tests.log`，版本身份及测试期间源码未变化的检查见 `diagnostics_revision_validation.json`，新源码副本见 `implementation_source_diagnostics/`。

本次真实请求仍对应原来的 `implementation_source_sse/` 冻结版。新诊断没有追写到旧实验，也不能补造旧请求的中途信息。

## 默认 JSON 请求的总时限修复

后续完成性检查发现，原默认非流式路径仅给网络操作设置 timeout；持续缓慢返回数据时，整体请求仍可能远超 600 秒。此前 SSE 的绝对截止时间测试不能证明该默认路径也有同样保证。

默认 JSON 请求现已共用 SSE 的受控子进程、父进程绝对截止时间、进程组结束与匿名安全诊断。请求内容保持原样，缺省的 `stream` 继续缺省；HTTP 状态、429/5xx/网络错误的重试规则、脱敏错误正文及无效/过大 JSON 的有限失败前缀均保留。模型、`max`、16,384 tokens、600 秒及 2 MB 原始响应上限没有提高。只有子进程内部回执上限调整为 16 MB，以容纳错误前缀的 JSON 转义；这不增加可接受响应或模型输出预算。

新增 14 项 JSON 传输测试及 2 项控制器接线测试，新旧回归共 **230 项通过**。慢速滴流与等待响应头的测试使用真实本地子进程和内存模拟服务，确认总时限到达后子进程已退出；取消测试也检查进程组清理与匿名描述符关闭。测试期间 15 个控制器及测试文件身份未变化，证据见 `json_deadline_revision_validation.json` 与 `regression_json_deadline_tests.log`，源码保存于 `implementation_source_json_deadline/`。

本次修复没有发起真实模型请求、连接开发机或执行真实 GPU 任务，也没有生成候选或消费保留集。旧请求的超时原因依然未知，不能从这个默认 JSON 缺陷推断旧 SSE 请求为何超时。

## 会话收尾与恢复边界

本次专用 SSH 主控连接已经关闭，远端 CPU 镜像、rootfs 和准备材料保留。执行配置中的控制 socket 需要在后续实际运行前重新建立；密码没有写入任何文件。

当前可复核的实验是 `outputs/generation_cpu_sse_campaign_20261112/`。此前两段实验保留了被中断时的历史状态，不表示控制器仍在运行，也不应作为合格实验直接恢复。流式实验曾用原冻结版完成离线 `evaluate`，身份核验通过、`accepted=false`；原结果保留在 `cpu_sse_offline_evaluation.json`。后补诊断及 JSON 总时限修订改变了控制器源码身份，当前工作区直接对旧实验 `resume/evaluate` 会先拒绝身份不匹配。保留各次冻结源码供审阅；后续使用新版应新建实验，并等待用户对预算及等待时限的决定，不能修改旧实验记录或将旧超预算响应追认为合格。

收尾核验逐项检查了原证据清单的 491 个文件，哈希及大小全部匹配。连接关闭记录见 `session_cleanup.json`，更新后的完整性清单继续覆盖本报告及新增记录。

## 证据入口

- `namespace_preflight.json`：真实 CPU 隔离及运行库身份。
- `namespace_worker_probe_summary.json`：可信执行样例和完整回执。
- `cpu_preflight_hardened.json`、`resource_policy_probe.json`：Docker 实际策略不合格的证据。
- `invalidated_campaign.json`：此前生成实验中断及零候选执行确认。
- `namespace_executor.json`：当前 CPU 容器配置，包含明确的 GPU 停止边界。
- 新实验的 `events.jsonl`、`rounds/`、`state.json`：模型输入、源码、修改、原始矩阵及逐项判决。
- `regression_streaming_tests.log`：真实请求所用版本的 203 项回归结果。
- `regression_diagnostics_tests.log`、`diagnostics_revision_validation.json`：后补诊断版本的 214 项回归与身份记录。
- `regression_json_deadline_tests.log`、`json_deadline_revision_validation.json`：默认 JSON 总时限修复后的 230 项回归与身份记录。
- `nonstream_transport_interruption.json`：非流式重试与中断记录。
- `cpu_sse_offline_evaluation.json`：流式实验的完整性复核与未接受结论。
- `implementation_source_sse/`：本轮冻结的控制器、验证器和执行器源码副本，可独立于工作区后续修改审阅。
- `implementation_source_diagnostics/`：超时后新增诊断的控制器版本；尚未发起真实生成实验。
- `implementation_source_json_deadline/`：当前控制器版本，JSON 与 SSE 均有总时限；尚未发起真实生成实验。
- `setup/README.md`：CPU 镜像、来源文件和 rootfs 的可复现准备材料。

这些原始结果与失败记录用于审阅；框架测试、隔离合格、生成代码和科学计算通过是不同结论。CPU 开发门槛通过也不等于完整 CPU/GPU 保留集验收通过。
