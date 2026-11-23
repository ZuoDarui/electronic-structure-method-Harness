# 可验证的 Python Hartree–Fock 程序

新增入口是 `gpu_hf.py`，现有 `hf.py` 保留。程序实现实数、非相对论、闭壳层限制性 Hartree–Fock（RHF），使用双精度，支持普通分子高斯基组、分子电荷和两种积分模式：

- `exact`：完整四中心电子排斥积分，适合小体系的精确方法对照；存储随 AO 数的四次方增长。
- `df`：使用相同辅助基组对 J 和 K 做密度拟合，完整保存三指标因子，按辅助指标分块计算交换。存储约为 `naux × nao²`，仍有规模限制。

`numpy` 和 `cupy` 共用同一套公式。选择 CuPy 后，SCF 中的密度、Fock、积分/拟合因子、DIIS 历史、矩阵乘法和本征求解留在 GPU；迭代向 CPU 读取少量诊断标量，结束后返回结果矩阵。PySCF 在 CPU 上提供积分、基组和核排斥能；本程序自行实现 J/K 缩并、正交化、占据、DIIS 和 SCF，没有调用 PySCF 的 SCF 求解器。

本次已在 CPU 和 NVIDIA RTX PRO 5000 72GB Blackwell 上分别通过 12 组验收，另有 72 项回归测试通过。GPU 最大能量误差 `8.53e-14 Eh`，J/K 最大误差 `1.11e-15`。故意注入交换矩阵错误时，harness 正确拒绝。

## 运行

需要 Python 3.11–3.13、NumPy、SciPy 和 PySCF。当前可复现环境是：

```bash
PYTHON=/Users/darui/workspace/fieldquantum/.pyscf-env/bin/python
"$PYTHON" gpu_hf.py --backend numpy --integral-mode exact \
  --basis sto-3g --output outputs/gpu_hf_validation/water_exact.json
```

不指定结构时默认计算水分子。程序逐轮打印能量、能量变化和 SCF 残差，并保存 JSON 迭代记录及 NPZ 矩阵。

其他结构可用 `--atom 'H 0 0 0; H 0 0 0.74'` 或 `--xyz water8.xyz`，坐标单位为 Angstrom。例如：

```bash
"$PYTHON" gpu_hf.py --atom 'Li 0 0 0; H 0 0 1.6' \
  --basis 6-31g --integral-mode df --auxbasis weigend \
  --backend numpy --output outputs/gpu_hf_validation/lih_df.json
```

在 Linux/NVIDIA CUDA 环境中安装与驱动、CUDA 匹配的 CuPy。CUDA 12 环境通常使用 `cupy-cuda12x`，CUDA 13 使用 `cupy-cuda13x`，同一环境只装一种。然后运行：

```bash
python gpu_hf.py --backend cupy --integral-mode df --basis 6-31g \
  --df-block-size 32 --memory-limit-mb 2048 \
  --output outputs/gpu_hf_validation/water_gpu_df.json
```

显式 `--backend cupy` 在没有可用 CUDA 时抛出错误；`--backend auto` 才允许回退，并在结果中记录原因。`--memory-limit-mb` 在分配大积分前检查保守工作集估计，是规划门槛，不是进程内存的硬限制。

## 固定 harness 验收

生成和优化框架已统一迁入 `harness_framework/`。以下命令从外层 `hf/` 执行，统一入口会把任务及输出的相对路径解析到框架目录。

运行 CPU 数值验收：

```bash
"$PYTHON" harness_framework/run.py optimize run \
  harness/tasks/gpu_rhf_correctness.json --workdir . \
  --report-dir outputs/gpu_hf_validation
```

运行 GPU 必须实际执行的验收：

```bash
python harness_framework/run.py optimize run \
  harness/tasks/gpu_rhf_required.json --workdir . \
  --report-dir outputs/gpu_hf_validation
```

两项任务使用固定的 6 个体系：平衡 H₂/STO-3G、拉伸 H₂/6-31G、水/6-31G、LiH/6-31G、NH₃/STO-3G 和 He/cc-pVDZ。每个体系分别运行 exact 和 df，共 12 组。GPU 任务还要求每一组实际产生 CuPy 设备数组并完成 SCF；CPU 通过不能替代这个条件。

验收独立调用 PySCF 的 RHF 或相同辅助基组的 DF-RHF，检查总能量、随机对称密度的 J/K、J/K 对称性与线性、最终密度、电子数、密度幂等性、SCF 残差、轨道正交性、轨道方程、轨道对密度的重构，以及从最终密度重新计算的能量。H₂ 另有固定的数值能量锚点。所有候选和参考都必须收敛，数值必须有限。

门槛预先写在任务 JSON 中，例如总能量误差 `≤ 1e-8 Eh`、J/K 最大绝对误差 `≤ 1e-9`、密度最大绝对误差 `≤ 1e-6`。候选输出原始指标，外层 harness 再作接受或拒绝，拒绝时返回非零退出码。每轮接受结果还有带时间戳的副本，不会只保留最后一次绿灯。

## 审阅证据

`harness_framework/outputs/gpu_hf_validation/` 保存本次验收证据：

- `gpu-rhf-correctness.acceptance.json`：CPU 验收决策、固定规格、运行输出和逐案例证据。
- `gpu-rhf-correctness.report.json`：完整指标、源码哈希、版本信息和迭代历史。
- `gpu-rhf-correctness.matrices/`：逐案例密度、轨道、重叠、Fock、J/K 和独立参考矩阵。
- `gpu-rhf-exchange-mutation.acceptance.json`：故意把 K 乘以 `1.001` 后被拒绝的记录，验证 harness 能抓住交换矩阵错误。
- `unit_tests.log`：数值边界和 harness 负面测试。
- `gpu_environment.json`：硬件和 GPU 可用性探测，不包含凭证。
- `remote_gpu/gpu-rhf-required.acceptance.json`：实际 CUDA GPU 的 12 组验收和完整迭代记录。
- `remote_gpu/remote_execution.json`：实际环境、源码哈希、硬件选择和临时目录清理记录。
- `cpu_gpu_comparison.json`：同一输入在 CPU 和 GPU 上的直接对照，密度最大差异 `2.36e-14`。
- `evidence_manifest.json`：交付源码、规格、文档和证据文件的完整性清单。

最终结果、行动记录和适用边界见 `harness_framework/docs/gpu_hf_validation_zh.md`。核对源文件可使用报告中的 SHA256；报告还记录对应矩阵文件的 SHA256。

## 适用边界

该版本没有实现开壳层 UHF/ROHF、ECP、周期体系、梯度或相对论修正。RHF 自洽收敛证明驻点的一致性，不保证是全局最低能态，也不构成波函数稳定性证明。密度拟合相对 exact 的偏差单独记录，不能因为它与 DF 参考一致就声称没有拟合误差。

候选和参考共享 PySCF 的积分引擎，因此验证独立覆盖的是 SCF 算法与矩阵运算，不能据此独立认证积分引擎。当前任务的计时范围不同，均作为诊断，不能用于 GPU 加速倍数声明；性能评估需要另设同工作量、预热、重复、GPU 同步和内存测量的任务。
