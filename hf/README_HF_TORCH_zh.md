# PyTorch HF

`hf_torch.py` 是从本地 `hf.py` 迁移的候选实现。CPU 与 CUDA 均使用
FP64，保持原来的积分、筛选阈值、RIJCOSX 网格切换、DIIS 收敛条件和
最终完整 J/K 重建验证。验收通过前保留原 `hf.py`。

## 运行

在安装了 PySCF、NumPy、SciPy 和 PyTorch 的 Python 环境中运行：

```bash
python hf_torch.py --config examples/torch_cpu.json
python hf_torch.py --config examples/torch_cuda.json
```

已有 JSON 输入只需设置 `resources.dense_backend` 为 `"torch"`，
`resources.torch_device` 为 `"cpu"`、`"cuda"`、`"cuda:0"` 或 `"auto"`。
省略后端时默认使用 PyTorch；`auto` 在 CUDA 可用时选择 CUDA，否则选择 CPU。
明确指定 CUDA 而没有可用设备时直接报错。`cuda:N` 是当前
`CUDA_VISIBLE_DEVICES` 中的逻辑编号。CPU 计算不修改 CUDA 可见性，
同一个进程可以先运行 CPU、再运行 GPU。

旧配置显式指定 `dense_backend="cpu"` 时仍使用原 NumPy/SciPy 路径；
`"cupy"` 仍保留兼容。旧启动脚本若设置了 `CUDA_VISIBLE_DEVICES=""`，
使用 GPU 时需要移除该设置。`resources.threads` 控制 CPU 线程预算。

## 实现范围与性能

PyTorch 负责密度重建、DIIS 误差与历史、Fock 外推以及 CUDA 上的
Cholesky 变换和对角化。CPU 上的广义本征问题使用 SciPy/LAPACK 与
Torch 张量的零拷贝接口：PyTorch 尚无广义 `eigh(F, S)`，直接用普通
`torch.linalg.eigh` 加稠密变换会增加开销。结果元数据 `eigensolver`
记录实际调用的求解器。

PySCF/libcint 仍负责量化积分及现有 J/K 引擎。这个版本是双精度
PyTorch 与成熟积分内核的混合实现，不是全程 GPU 积分实现。
`parallel_backend` 控制原有 J/K 任务路径；它与 `dense_backend`
相互独立，旧 `parallel_backend="gpu"` 路径仍依赖 CuPy。
使用 PyTorch CUDA 的常规配置可保持 `parallel_backend="serial"`。

`dense_check_iterations` 可开启前几次迭代与 SciPy 的双算校验；
正式性能测试设为 0，避免把校验成本计入计算。

## 验收与回退

```bash
HF_TEST_TORCH_DEVICE=cpu python -m unittest test_torch_dense -v
HF_TEST_TORCH_DEVICE=cuda python -m unittest test_torch_dense -v
python test_torch_acceptance.py --devices cpu cuda --repeat 3 --cli-repeat 3 \
  --output outputs/torch_acceptance/report.json
```

验收使用独立原版文件，比较 H₂/H₂O/CH₄ 与 PySCF、以及 C₂H₆/cc-pVTZ
与原版的能量、密度、电子数、密度对称性与幂等性；前三种分子还检查
轨道正交性及广义本征残差。C₂H₆ 完整
SCF 分别测量预热后的三轮交错计时和三个独立 Python 进程的 CLI
耗时，后者包括导入、CUDA 初始化和退出。

能量差要求小于 `1e-8 Hartree`，密度最大差小于 `1e-6`，电子数误差
小于 `1e-8`。速度门限为原版中位耗时的 110%，用于容纳运行噪声；
报告保留全部样本和实际比值。源文件 SHA256 必须在整个验收期间一致。
原版与候选版的路径及 SHA256 必须不同，防止覆盖后误把新版本与自身比较。
`--quick` 仅检查小分子精度，不允许据此覆盖原版。小分子验收不能
代替大型分子的性能与内存测试。

原文件备份位于 `outputs/torch_20261120/validation_v2/baseline/hf.py`。
覆盖后复测时用 `--baseline` 指向该备份，避免把新版本与自身比较。
