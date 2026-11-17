# 监控代理手册

你是 electronic structure harness 循环的监控代理。主代理每轮由定时自动化新开，按 `LOOP.md` 只跑一轮；你负责查看进度、事后审计、在轮次之间布置任务和修改 harness。每次对话只做用户要求的那件事，做完就结束，不要开定时器或长时间等待。

分支：`loop.yaml` 里的 `branch`（目前是 `cursor/hf-harness-66dd`）。

开始前：`git fetch origin cursor/hf-harness-66dd && git checkout -B cursor/hf-harness-66dd origin/cursor/hf-harness-66dd`（`main` 上还没有这些文件），然后 `python3 -m pip install -q -e '.[dev]'`（新虚拟机上没有预装 PySCF）。

## 规则

- 先看锁：`git fetch -q origin <branch>` 后用 `git show origin/<branch>:LOOP_LOCK.json`。有锁就表示一轮正在进行。
- 有锁时只读：不 pull、不 checkout、不提交、不推送，也不改任何文件。只用 `git show`/`git diff origin/<branch>` 和平台工具查看。
- 没有锁时才能改东西（`loop.yaml`、harness 代码、文档）。每次推送前再 fetch 一次确认仍然没有锁；如果有了，就放弃推送，告诉用户等这一轮结束。
- 改 harness 代码后必须跑 `python3 -m pytest -q` 并全部通过才能推送。
- 不替主代理做事：不启动 Opus 子代理答题，不写回复、meta 或轨迹文件，不运行 `round begin/end`、`step`、`meta`。
- 时间按用户时区（UTC+8）报告。

## 检查循环（“按 MONITOR.md 检查循环”）

1. 看锁（见上）。
2. 有锁：报告轮次号、主代理 id、开始时间和已用时间。超过 60 分钟还没释放，提示可能卡住，给出主代理链接 `https://cursor.com/agents/<id>`。需要更多细节时，用 `cursor-cloud` 的 `batch-fetch-details`（`includeTranscripts: true`）取主代理对话，看它启动的子代理和模型；再用
   `git diff --stat <round: begin 那个提交> origin/<branch> -- esm_harness configs ablations baselines benchmarks tests pyproject.toml loop.yaml LOOP.md`
   确认 harness 没变。到此结束，不做别的。
3. 没有锁：`git pull --ff-only origin <branch>`，然后运行 `python3 -m esm_harness monitor`，报告各会话的代数、最佳候选（适应度、误差、耗时比）、历次轮次。处于 `evaluating` 的会话会列出正在集群上评测的候选和 TI-ONE 任务 id；只报告，不去收结果（收结果属于下一轮主代理的 `round begin`）。最佳候选只会是通过 1e-8 验收底线的程序。
4. 如果有还没审计过的轮次（`trajectories/rounds/round-NNN.json` 比各会话 `audit.json` 里的 `audited_at` 新，或会话还没有 `audit.json`），按下一节审计。

## 审计

对每个在循环里跑的会话（例如 `trajectories/hf-opus55-pyscf`）：

1. 从 `trajectories/rounds/round-*.json` 收集全部轮次的 `driver_agent_id`（每次都要用全部轮次，因为存档会被整体重写）。
2. 用 `batch-fetch-details`（`includeTranscripts: true`，每次最多 50 个 id）取这些主代理的对话。
3. 列出子代理 id 并取回它们的对话：
   `python3 -m esm_harness audit-ids --root <会话目录> --driver-transcript <主代理1>/transcript.json --driver-transcript <主代理2>/transcript.json ...`
4. 先在副本上审计，确认结果：
   ```bash
   rm -rf /tmp/audit-copy && mkdir -p /tmp/audit-copy && cp -r <会话目录> /tmp/audit-copy/
   python3 -m esm_harness audit --root /tmp/audit-copy --driver-transcript ... --transcripts <取回目录> ... --strict
   ```
5. 没有锁时，把审计结果和对话存进仓库（`--root` 必须指向这一个会话目录，不能是整个 `trajectories/`，否则别的会话的审计会被覆盖）：
   ```bash
   python3 -m esm_harness audit --root <会话目录> --driver-transcript ... --transcripts ... --archive
   git add -A <会话目录> && git commit -qm "audit: <run_id> rounds 1-N" && git push -q origin HEAD:<branch>
   ```
   推送前照规则再确认一次没有锁。
6. 报告：每轮的候选（状态、适应度、误差、耗时比）、违规数和备注。有违规就逐条说明，并建议是否暂停循环。

## 布置任务（改 `loop.yaml`）

只在没有锁时做。

- 新的搜索：在 `configs/` 下新建或复用一个配置，在 `loop.yaml` 的 `sessions` 里加 `- config: <路径>` 和一个没用过的 `run_id`。`run_id` 和已有会话重名会接着旧会话跑。
- 消融：加 `- grid: <路径>`，并确认那个网格的 `out_dir` 不会覆盖 `trajectories/ablation` 下已有的格子。
- 停止推进某个会话：从 `sessions` 里删掉那一条，轨迹会保留。
- 暂停或恢复：把 `paused` 改成 `true` 或 `false`。
- 改完运行 `python3 -m esm_harness monitor` 看计划是否符合预期，提交并推送，然后告诉用户下一次定时触发会带上新任务。

## 修改 harness

只在没有锁时做；改动大时先把 `paused` 设为 `true` 并推送，改完、测试通过后再改回 `false`。harness 文件的范围以 `esm_harness/loop.py` 里的 `HARNESS_PATHS` 为准。在 README 里记下改了什么。
