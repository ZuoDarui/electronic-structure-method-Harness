# 2026-11-13 集群同步说明

本次把本地 HF 项目代码、生成框架、测试、任务、报告、图表与原始实验记录更新到集群目录：

`/home/tione/notebook/home/zuodarui/fieldquantum/hf`

更新会先核对文件并备份远端同名旧文件，再按文件原子替换；远端独有文件保留。同步不包含 Python 缓存、系统预览缓存、控制锁或凭据。源代码、脚本执行权限和历史实验原始字节均保留，文件清单、备份位置和远端验证记录另存于 `outputs/cluster_sync_20261113/` 及集群项目旁的部署记录目录。

## 集群使用入口

先使用集群上实际存在、安装了 NumPy、SciPy、PySCF 的 Python 环境，再在项目根目录查看帮助：

```bash
cd /home/tione/notebook/home/zuodarui/fieldquantum/hf
python -m harness_engineering.generation --help
```

完整参数和生成流程见 `README_GENERATION_zh.md`。实际 Python 环境和本次 CPU 检查结果以同步回执为准；Mac 文档中的 `/Users/darui/...` 解释器路径不是集群启动路径。

## 历史实验与新实验

历史实验的哈希、绝对路径、参考、私有保留集、请求响应和失败判决原样保留。它们绑定了原运行机器的文件身份，因此复制到集群不等于可以直接 `resume` 或 `evaluate`。不要批量改写历史 JSON 路径，也不要用新路径覆盖旧清单来冒充原实验身份。

新计算应使用新的实验目录、集群实际路径和执行配置，并重新检查运行环境。保留 `.holdout-assessments` 登记；不能改目录或文件名重复使用已消费的保留集。现有 CPU rootfs 位于项目之外，同步项目本身不证明这些外部资产存在或合格。原配置中的 Mac SSH socket 和临时 known-hosts 路径仅属于旧会话，不能直接用于集群本地执行。

## 验证与当前结论

不要使用整仓 `unittest discover` 或自动执行所有 `test_*.py`：旧 `test_usa0.py` 在导入时会调用外部服务。CPU 回归需要显式列出允许的模块，不启动生成请求、提交脚本或生产 SCF。GPU 继续按用户要求暂停，CPU-only 配置必须保持 `stop_before_gpu=true`。

同步不会改变科学结论：早期 `gpu_hf.py` 基线有独立 CPU/GPU 验证记录；新自动生成框架最近 235 项回归通过，但真实生成因服务协议及网络故障受阻，尚无合法新候选完成科学验收。同步后的检查也不能代替候选的开发集与保留集验收。
