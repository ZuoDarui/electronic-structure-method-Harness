# HF 实现与验证记录（2026-11-10）

## 目标与实现范围

交付一个 Python 编写、适合 GPU 矩阵运算的 Hartree–Fock 程序，并用可重复的 harness 验证。实现入口为 `gpu_hf.py`；原有 `hf.py` 保留。本次明确选择闭壳层 RHF，实数双精度，不包含开壳层、ECP 或周期体系。

本文记录可审阅的设计依据、采取的行动、原始证据位置和判断。完整逐轮数值轨迹在 harness JSON 和日志中。

## 行动与依据

1. **检查现状。** 当前目录已经有 CPU HF、PySCF 环境和 harness，但没有 Git 元数据。本机为 macOS arm64，没有 CUDA/CuPy。决定新增独立入口，以便直接审阅新实现，也避免把已有程序的测试结果误当成新程序的证据。
2. **先固定验收范围。** 写入 6 个分子/原子体系、exact/DF 两种模式、13 项误差门槛及收敛要求；CPU 与必须实际执行 GPU 的任务分别保存。测试输入直接来自任务文件，而非在候选程序里另写一套。
3. **自行实现求解过程。** PySCF 仅提供 AO 积分；新程序实现正交化、密度、J/K、Fock、DIIS、SCF 和收敛判断。exact 便于查清索引错误，DF 降低积分存储量，二者与对应方法的参考分别对照。
4. **安排独立审查。** 实现、验证 harness、代码审查和 GPU 环境探测并行进行。审查发现最终 canonicalization 被拒绝时可能混用新轨道与旧密度；修复后新增专门回归测试，确保所有返回量属于同一状态。
5. **先修验收漏洞。** 原有 gate 中 `abs(NaN) > tolerance` 为假，可能误放行 NaN；现改为显式检查有限数值，并拒绝布尔冒充数值、无效性能指标、伪装的有限性/收敛标记。负面测试覆盖这些情况。
6. **运行独立对照与性质检查。** 候选与 PySCF 分别求解，另外对随机对称密度检查 J/K，重建最终物理 Fock，检查电子数、幂等性、正交性、轨道方程和能量。
7. **验证 harness 会拒绝错误。** 仅在独立子进程把公共 `build_jk` 返回的 K 乘以 1.001，保持原始源文件不变。harness 检出 J/K 误差并拒绝；该注入不修改 SCF 内部私有路径，因此能量仍准确，说明检查矩阵有必要。
8. **完成本地回归。** 新 RHF 11 项、harness 25 项和原有 direct HF 36 项，共 72 项全部通过。
9. **检查 GPU 实机资源。** 本地 GPU-required 任务如预期拒绝，明确报告 CuPy 缺失，没有以 CPU 回退冒充 GPU。另探测现有腾讯开发机的空闲 GPU，使用临时目录和专用 kernel，避免修改已有项目或运行中的任务。实机结果见下文和原始归档。

## 公式与数值约定

密度采用自旋求和约定：`D = 2 C_occ C_occ.T`。

- `J_pq = Σ_rs (pq|rs) D_rs`
- `K_pq = Σ_rs (pr|qs) D_rs`
- `F = H + J - K/2`
- `E_total = Tr[D(H + F)]/2 + E_nuc`

对重叠矩阵做 canonical orthogonalization；低于 `lindep_tol` 的方向剔除，剩余维数必须足以容纳占据轨道。DIIS 使用正交基中的交换子误差。收敛同时要求能量变化和物理 Fock 的残差达标，随后再确认最终轨道、密度和能量的一致性。达到最大迭代次数会返回 `converged=false`，CLI 返回非零退出码。

GPU 模式在设备上保存积分/DF 因子、密度、Fock、轨道和 DIIS 历史。DF 交换按辅助基分块，避免建立全辅助维度的额外交换中间量。积分生成仍在 CPU；本版本不是全 GPU 积分引擎，也不是线性标度或外存算法。

## CPU 验收结果

任务：`harness/tasks/gpu_rhf_correctness.json`，**12/12 通过**。6 个体系为 H₂/STO-3G、拉伸 H₂/6-31G、水/6-31G、LiH/6-31G、NH₃/STO-3G 和 He/cc-pVDZ；每个分别运行 exact 和 DF，所有候选及参考均收敛。

- 最大总能量误差：`7.1054e-14 Eh`，门槛 `1e-8 Eh`。
- 最大 J/K 绝对误差：`1.5543e-15`，门槛 `1e-9`。
- 最大密度绝对误差：`6.5808e-10`，门槛 `1e-6`。
- 最大电子数误差：`8.8818e-15`，门槛 `1e-9`。
- 最大物理 SCF 交换子绝对残差：`3.6384e-11`，门槛 `1e-7`。
- 最大轨道方程绝对残差：`3.1909e-11`，门槛 `1e-7`。

例如平衡 H₂/STO-3G 总能量为 `-1.1167593073964253 Eh`，同时通过固定能量锚点。水/6-31G 的 exact 总能量为 `-75.9839484981057 Eh`。

权威证据为 `outputs/gpu_hf_validation/gpu-rhf-correctness.acceptance.json`，其中包含运行规格、stdout/stderr、决策和逐案例证据；同名 `.report.json` 包含版本、源码 SHA256、迭代历史和矩阵路径。`gpu-rhf-correctness.matrices/` 保存候选及参考的原始矩阵。

故障注入结果为 **拒绝**：最大 J/K 误差 `0.0013752563`，超过固定 `1e-9` 门槛；原始记录在 `gpu-rhf-exchange-mutation.acceptance.json`。本地没有 GPU 时，`local_no_gpu/gpu-rhf-required.acceptance.json` 也为拒绝，`gpu_executed=false`。

## GPU 实机验收与 CPU/GPU 直接对照

任务：`harness/tasks/gpu_rhf_required.json`，**12/12 通过**，每例都确认 `backend=cupy`、产生 CuPy 设备数组，并完成候选及参考的 SCF。GPU 为现有开发机的空闲 GPU0，型号 **NVIDIA RTX PRO 5000 72GB Blackwell**；运行环境为 Python 3.12.16、NumPy 2.5.4、SciPy 1.18.1、PySCF 2.14.0、CuPy 13.6.0、CUDA runtime 12.9。

- 最大总能量误差：`8.5265e-14 Eh`，门槛 `1e-8 Eh`。
- 最大 J/K 绝对误差：`1.1102e-15`，门槛 `1e-9`。
- 最大密度绝对误差：`6.5759e-10`，门槛 `1e-6`。
- 最大物理 SCF 交换子绝对残差：`3.6385e-11`，门槛 `1e-7`。
- 所有候选及参考收敛，所有数值有限，`gpu_executed=true`。

原始证据在 `outputs/gpu_hf_validation/remote_gpu/`：包含验收 JSON、完整指标报告、12 组矩阵和日志。远端源文件 SHA256 与本地最终版本一致；取回的矩阵也逐个校验哈希。报告内的原始远端矩阵路径属于已清理的临时目录，对应文件已原样取回同目录名下的本地 `remote_gpu/gpu-rhf-required.matrices/`。

另从两端归档的矩阵直接比较相同输入和模式，最大 CPU/GPU 总能量差 `8.5265e-14 Eh`，最大密度差 `2.3648e-14`，最大 J/K 差 `1.1102e-15`。结果在 `cpu_gpu_comparison.json`；轨道向量可能存在任意符号，故使用最终密度比较占据空间。

实机第一次因两个 Python 环境的 OpenSSL/h5py 依赖冲突而未能启动，失败证据保留在 `remote_gpu/remote_execution_attempt1.json`。后改用已有 PySCF 解释器，并追加已有 CuPy 包路径，无需安装或修改既有环境，完整验收通过。运行使用远端临时目录和专用 kernel；结束后临时目录删除，GPU0 恢复为 `0 MiB / 0%`。没有创建云训练任务。硬件、环境、执行过程与清理证据见 `gpu_environment.json` 和 `remote_gpu/remote_execution.json`。

## 密度拟合偏差和验证边界

DF 的实现误差以相同 `weigend` 辅助基组的 PySCF DF-RHF 为参照；DF 相对 exact 的变化另列诊断。例如：

- 水/6-31G：DF 与 exact 能量差约 `0.0007376112 Eh`。
- NH₃/STO-3G：DF 与 exact 能量差约 `0.0016465353 Eh`。

因此，匹配 DF 参考到浮点误差，并不说明拟合误差可以忽略。需要更高精度时应选择合适辅助基组并对照 exact。

参考与候选共享 PySCF 积分引擎和基组数据；本次独立验证覆盖 SCF 与矩阵运算，没有独立认证积分引擎。RHF 自洽收敛也不保证全局最低能量或波函数稳定性。当前测试支持这些已运行体系与模式的正确性，不是任意分子的普遍证明。

计时仅供诊断，候选和参考计时范围不同，不作加速倍数判断。GPU 代码结构、GPU 正确性和 GPU 性能分别需要对应证据。

## 复现入口

使用说明与 CPU/GPU 命令见仓库根目录 `README_GPU_HF_zh.md`。

本地完整回归命令：

```bash
/Users/darui/workspace/fieldquantum/.pyscf-env/bin/python \
  -m unittest test_gpu_hf test_harness_engineering test_direct_hf -v
```

完整输出在 `outputs/gpu_hf_validation/regression_tests.log`，实际为 `Ran 72 tests ... OK`。源码、任务、报告与矩阵的完整性以最终 `evidence_manifest.json` 为准。

## 最终判断

**接受这个实现作为已验证的 GPU 友好闭壳层 RHF 基线。** 依据是 CPU 12 组及实际 GPU 12 组均通过固定门槛，72 项回归通过，注入错误会被 harness 拒绝，原始矩阵与源码哈希可供复查。精确 HF 与指定 DF 近似应分别解读，GPU 运算正确性通过；更大体系的可靠性、内存峰值与加速效果需要单独验证。
