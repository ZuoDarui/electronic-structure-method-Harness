# HF Harness：生成、优化与独立验证

新程序生成框架和原有优化 harness 统一放在本目录。生产算法仍在上一级 `hf.py`、`gpu_hf.py`，框架通过明确的外层源码路径导入它们，不复制或修改算法。

```text
hf/
├── hf.py                       原生产 HF
├── gpu_hf.py                   已验证的 GPU 友好 RHF 基线
├── make_alkane_xyz.py          分子构造工具
├── 分子数据、生产配置和提交脚本
└── harness_framework/
    ├── run.py                  统一入口
    ├── harness_engineering/    共用运行器、验收、归档及优化工具
    │   ├── generation/        Responses 自动生成、修复和独立科学验收
    │   ├── examples/          原有基线、性能和性质任务
    │   └── generated/         原优化流程的可替换算法槽
    ├── harness/
    │   ├── tasks/             原有优化及正确性任务
    │   └── generation/        新生成任务、保留集与执行配置
    ├── containers/            候选运行环境配方
    ├── tests/                 两套 harness 的测试
    ├── docs/                  设计、验证、审计与历史报告
    └── outputs/               原始证据及后续运行结果
```

## 集群入口

```bash
cd /home/tione/notebook/home/zuodarui/fieldquantum/hf/harness_framework
PYTHON=/home/tione/notebook/home/zuodarui/miniforge3/envs/pyscf/bin/python
"$PYTHON" run.py --help
"$PYTHON" run.py generate --help
"$PYTHON" run.py optimize --help
"$PYTHON" run.py loop --help
"$PYTHON" run.py test
```

也可在外层 `hf/` 使用 `python -m harness_framework`。统一入口把任务相对路径的基准固定在 `harness_framework/`，因此工作目录、候选脚本、可编辑路径和默认输出指向同一框架。显式传入的绝对路径继续有效。

原优化任务的 CPU 小体系验证：

```bash
"$PYTHON" run.py optimize run harness/tasks/rhf_direct_baseline.json \
  --workdir . --report-dir outputs/smoke-direct
```

**单次生成 + 小分子准确性验收**：`run.py generate single`。每次只生成一个 HF 程序，在 CPU 上对 15 个小分子（exact/DF，共 30 个用例）与 PySCF 参考比对准确性，不跑大分子，也不做速度或 GPU 验收，详见 `README_GENERATION_zh.md`。

完整多阶段流程仍通过 `run.py generate run`、`resume`、`evaluate` 操作。模型、API 端点、执行环境和预算必须在新实验中明确配置，详见 `README_GENERATION_zh.md`。默认预算仍为 16,384 输出 tokens、最长 600 秒，授权扩展需显式选择 `extended`。GPU 暂停期间执行配置保持 `stop_before_gpu=true`，不会因本次目录合并而恢复 GPU 工作。

`run.py loop` 保留旧版单轮 oracle/budget 决策语义；新的完整 RHF 自动生成与真实修复由 `generate` 管理。合并的是工程入口、代码组织和验证工具，没有把旧单轮决策流程描述成已经完成的自主优化系统。

## 测试与历史证据

`run.py test` 仅加载明确列出的框架测试和现有 CPU HF 基线，排除需要 GPU 可用性检查的测试，也不加载根目录会发送真实请求的 `test_usa0.py`。测试结果写入新的 `outputs/tests/<时间戳>/`。这证明框架回归与 CPU 基线，不替代新生成方法的开发集及保留集验收。

旧 `outputs/` 与历史报告按原始字节搬迁。它们记录的绝对 Mac/集群路径、源码身份和清单没有批量改写，所以迁移记录不代表旧实验可以直接恢复。请保留保留集消费登记，后续用新目录发起新实验并重新预检。历史报告中的旧路径描述当时布局；本 README 和 `outputs/reorganization_20261113/` 的映射说明当前位置。

本次目录整理不会改变科学结论：早期 RHF 基线有 CPU/GPU 验证证据；新自动生成实验仍因服务协议与网络错误没有合法候选获得科学接受。
