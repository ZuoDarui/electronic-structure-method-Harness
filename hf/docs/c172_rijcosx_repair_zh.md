# C172 RIJCOSX 修改流程与交接

按用户最后确认的范围，本次交付为代码修改与公式推导；后续测试由用户执行。没有提交集群训练任务。此前已完成的局部检查保留为阶段性记录，不宣称 C172、GPU 或近线性标度已经验收。

公式工程：[Overleaf：C172 J-K 优化公式推导与修复验收](https://www.overleaf.com/project/6af6bb114ed66a0cad9ae782)。本地源文件位于 `jk_optimization_derivation/`，`C172-JK-derivations.zip` 可导入 Overleaf，编译器选择 XeLaTeX。

## 1. 固化原始问题

原始 `hf.py`、测试和 C172 配置已保存到 `outputs/c172_repair_20261113/baseline/`，附 SHA-256。最新原任务日志与物理输入检查在 `outputs/c172_energy_diagnosis_20261113/`。

原局域 J 路径同时存在两类问题：按当前 D_ij 筛选后把同一筛选复用到输出 J_ij；截断远场并把独立局域求逆结果按所属函数拼装。它们不能保证完整 RI 的一致性。C21 正常初猜库仑能从参考约 1839.35 Ha 被算成约 230710.83 Ha；错误初始 Fock 随后产生异常密度。

新增零密度块和远离电子的回归案例，旧代码两项均失败。原记录为 `baseline_regressions.log`。修复目标是恢复正确的积分映射，不以总能量变负作为验收标准。

## 2. J 的重构

新增 `hierarchical_ri.py`，实现共享三中心算子的 B、Bᵀ 作用、全局一致的 Mc=γ 求解以及 J=Bc。原始 AO 积分打包配合 wD，非对角 AO 对的权重为 2；输出空间不随 D 的零元素改变。

- 近场为有界缓存／流式精确积分，远场用 ACA 低秩因子；不保存完整三中心张量。
- 度量的对称块共用因子；局域区域用于对称加性 Schwarz 预条件，加入有上限的聚合粗空间。
- 每次 PCG 从零初值开始，检查真实残差和正曲率；精度失败按 1e-12、1e-13、1e-14 重建算子，不改变物理模型或静默添加对角偏移。
- ACA 拟合使用独立保留行列检查，避免在参与拟合的探针上假收敛。另有精确积分抽样审计，抽样仍不能代替完整参考。
- 算子、缓存和临时工作区具有内存预算检查；它们是该引擎的预算估计，并非整个 SCF 进程的硬内存限制。

修复过程中又发现：每 25 步无条件重启 PCG 会破坏收敛效率；逐点重叠权重会恶化辅助基近相关方向的预条件效果。现改为依据递推／真实残差偏离重启，并采用两侧一致的 W_s=I。

## 3. K 的重构

新增 `local_cosx.py`，保留局域网格／密度支持，按完整重叠方程残差扩大求解区域，保留跨原子耦合。

实际 PySCF SGX 默认启用对称拟合。因此实施时将初始计划的单边 QD 纠正为：Q=S_num⁻¹S，P=(Q+I)/2，K=sym(Pᵀ R(PD))。通过输入和输出两次稀疏方程求解应用投影，避免显式形成完整投影矩阵；原始矩阵方向固定为 R=XᵀWG。

Gaussian 乘积和库仑势上界用于贡献筛选，增量 K 结合完整密度的支持。C 层融合原语 `SGXnr_direct_k_drv` 计算 G=VF，避免 Python 形成网格×AO×AO 张量；投影、支持选择、原始矩阵积累和最终校正仍在新模块中实现。生产路径没有调用高层 `get_k_only` 或参考 SCF。

共享的 C 积分原语在 CPU 上执行；GPU 只用于实际 CuPy 运算。日志明确记录混合执行，不能把它描述成整个内核都驻留 GPU。显式张量模式可通过 `LocalCOSX(..., fused_integrals=False)` 复核。

## 4. 统一入口、SCF 和失败处理

`hf.py` 的直接 RI-J/COSX 调用、`get_jk`、`iteration_rijcosx`、SCF 迭代及最终完整重建采用同一引擎。旧局域 J/K 实现已从生产类移除，原代码只保留在快照中。

- 新选择项：`jk_engine="hierarchical_rijcosx"` 为默认，`"pyscf_reference"` 为显式参考。C172 JSON 配置已指定前者。
- `parallel_backend` 控制设备；返回的 J/K、能量及密度类型保持兼容。
- 构造器可通过 `hierarchical_options` 调整层次算子参数。旧 `df_tol` 形参保留兼容和合法性检查；层次内核精度由层次选项控制，`df_tol` 不是总能量误差保证。
- 粗网格收敛后，清空 DIIS 及网格相关状态，在细网格继续 SCF；两阶段共享总迭代预算。
- 最终完整重建同时检查能量有限、梯度有限、能量跳变和梯度门槛。NaN 梯度和“梯度小但能量跳变”均须拒绝。
- 增加能量分项、实际引擎／设备、压缩秩、残差、PCG 次数、稀疏支持及阶段计时日志。

## 5. 已有记录与用户后续测试入口

需求调整之前的记录包括：66 项本地回归通过；乙烷/STO-3G 完整候选和参考 SCF 的总能量差约 3.72e-11 Ha；C21 初猜 J 最大误差约 7.49e-12 Ha、K 最大误差约 3.86e-12 Ha。各记录含环境或源码身份；它们不替代完整修复版本的用户验收。

后续由用户手动执行，当前没有自动测试或训练任务在运行。例如：

```bash
PY=/Users/darui/workspace/fieldquantum/.pyscf-env/bin/python
"$PY" -m unittest test_hierarchical_ri test_local_cosx test_direct_hf test_rijcosx_repair -v
"$PY" validate_rijcosx_repair.py --carbons 21 --backend cpu --threads 8 --scf --output outputs/manual_c21.json
```

`validate_rijcosx_repair.py` 提供固定密度、首次更新密度、独立 SCF 与计时记录；`--benchmark` 预热一次、重复三次，并单列数值内核和稠密接口成本。`run_c172_repair_validation.sh` 是可供用户手动使用的集群顺序验收脚本，任何门槛失败就退出，不会自行提交任务。

冻结代码副本已上传至集群独立目录 `/home/tione/notebook/home/zuodarui/fieldquantum/hf-c172-repair-20261113/v1`，15 个文件哈希核对一致。随后只进行了未提交的任务配置预检查；用户收窄范围后已中断该检查。原项目目录和原作业未改动。

## 6. 尚需验收的边界

必须分别验证 C172 收敛总能量误差 ≤1e-6 Ha，以及指定烷烃序列上 J/K 最大两档时间指数 ≤1.5。当前没有这两项完成证据，也没有最终版本的 GPU 数值验收。

C21 的局域重叠求解区域覆盖了全部 1246 个 AO，说明近线性依赖于更大尺度上支持范围、远场秩和 PCG 次数的实际增长；不能凭代码中有局域树就宣称成立。ACA 抽样检查、Gaussian 尾部控制和原始 K 筛选界也不是最终能量的严格误差证明。

稠密输入扫描、J/K 展开、DIIS 和本征求解仍分别含 N²/N³ 成本，完整 SCF 不宣称近线性。所有验证门槛保持原计划，不通过时应保留失败记录。

完整代码差异在 `outputs/c172_repair_20261113/changes.patch`；交付文件哈希在同目录 `final_manifest.json`。
