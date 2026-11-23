# J/K 优化公式推导（Overleaf 工程）

可直接导入 Overleaf 的中文 LaTeX 工程，解释 RHF、RI-J、AO 对打包、
层次低秩远场、PCG、两层预条件、COSX、跨原子双边重叠校正、
增量 K、误差预算和近线性条件。

## 在 Overleaf 中使用

1. 将 main.tex、status.tex、repair_speedup.tex、README.md 打包为 ZIP。
2. 在 Overleaf 选择 **New Project → Upload Project**，上传 ZIP。
3. 主文档选择 **main.tex**。
4. 在项目设置中将 **Compiler** 设为 **XeLaTeX**。
5. 编译；如目录或引用未刷新，再编译一次。

工程使用 ctexart 与 TeX Live 自带 Fandol 中文字体，不依赖本机字体、
外部图片或 shell escape。建议选择 Overleaf 可用的较新 TeX Live。

## 文件

- main.tex：完整推导和原实现诊断。
- status.tex：实施和验收状态页，避免把数学目标误写为已通过测试。
- repair_speedup.tex：第 15 节，全局 DF-J + SGX 修复与加速的公式推导和验收数据。
- overleaf-main.tex：把 status.tex 与 repair_speedup.tex 内联后的单文件版本，即 Overleaf 上的 main.tex。
- README.md：编译说明。

## 本地编译

安装 TeX Live / MacTeX 后，在本目录执行两次：

    xelatex -interaction=nonstopmode -halt-on-error main.tex

## 重要说明

原实现数据来自已保存诊断。新实现结果只在真实测试后更新。
C172 能量与近线性是两个独立验收条件。

PySCF SGX 实际默认对称拟合为 P=(Q+I)/2、
K=sym(P.T R(PD))，正文同时解释非对称选项。
J 正文使用平方根权重推导，并明确映射到实现的原始积分加二重密度权重约定。

工程不包含口令、云端凭据或认证信息。
