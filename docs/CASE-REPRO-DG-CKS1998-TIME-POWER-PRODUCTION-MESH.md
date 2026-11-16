# CASE · REPRO-DG-CKS1998 — 时间误差归因仪器：生产网格（n=32）支路补测与双网格判决

> 锚点：`REPRO-DG-CKS1998`（论文复现·CKS1998, *JCP* 141:199–224 / *SIAM Review* 43(1)），验收原文「quantitative_targets 逐项容差」。
> 前置：`CASE-REPRO-DG-CKS1998-TIME-POWER-CLOSURE`（m22930）把 m22636 的三条未闭合项用测量补上，但**自认只完成一半**：生产网格（p1 n=32）的实测未取到（集群作业 `cj4480` 全程排队未出队），`GATE_VACUOUS_CERTIFIED` 只是**本地 n=10 网格上的认证结论**，文档明示「不能外推成『生产网格上也如此』」，并预登记了补齐方式：`--meshes 10 32`，`mesh_independent` 直接比较两网格判决。
> 单号 m23825（自找活·paper 席）。增量类型：**验证欠账闭合**——执行预登记的补齐命令，把「生产网格上同样空判据」从外推变成测量。本单**不改任何求解器内核、不改任何预登记常数**（`ETA_MAX=0.05`、`ADMISSIBLE_CFL_MAX=0.5`、`BETA=4`、`ORDER_TOL=0.10` 全部沿用 m22930 的预登记值）。

## ① 案例介绍（物理背景 / 几何 / 参数）

二维等熵涡（CKS1998 §5.3）：周期盒 `[-5,5]²`，γ=1.4，强度 5，背景速度 (1,1)，T=1.0，
解析解 `rho = [1 - (γ-1)β²/(8γπ²)·exp(1-r²)]^{1/(γ-1)}` 随涡心平移。它是该文里唯一有
**解析解**的二维算例，也是「阶次偏差归因」的试验台：认证研究（m22219）用两条支路把
p1/p2/p3 的阶次偏差判为空间主导（`SPATIAL_ISOLATED`）——

```
eta   = |e(CFL) - e(CFL/beta)| / e(CFL)     时间占比，判据 eta <= 5 %
drift = |p(CFL) - p(CFL/beta)| / p(CFL)     阶漂移，判据 drift <= 1 %
```

本单把 `beta = 4`、生产 CFL = 0.2、求解器硬窗口 `0 < CFL <= 0.5` 全部当作**给定条件**，
只补 m22930 缺的那一档测量：

| 参数 | 值 | 来源 |
|---|---|---|
| 网格 | n = 10（local）与 n = 32（production，`TIME_PAIR[1]` 细网格） | m22930 预登记 |
| CFL 梯次 | 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5（伙伴完备） | m22930 预登记 |
| 拟合 CFL | 0.05, 0.1, 0.2, 0.4（窗口边 0.5 只测不拟合） | m22930 预登记 |
| β / β² | 4 / 16 | m22930 预登记 |
| 阶一致性带 | `|s - q_obs|/q_obs <= 10 %` | m22930 预登记 |
| 判决 | jackknife（全样本 + 全部 LOO 子集） | m22930 预登记 |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **执行预登记命令**（m22930 §5 原文：「补齐方式：`--meshes 10 32`」）：

   ```bash
   OMP_NUM_THREADS=1 python -m benchmarks.cks1998_vortex_time_power_closure \
       --degree 1 --meshes 10 32
   ```

   集群作业 `cj4804`（stage_dir 直发本工作树，m22930 模块未合入 main），单进程，
   工件 `validation/cks1998_vortex_2d/time_power_closure.json` + `.md` 由模块
   `assemble()` 程序化生成，零手抄。
2. **双网格判决比较**：`assemble()` 对每个网格独立做 jackknife 判决，
   `mesh_independent = (两网格判决集合大小为 1)`——这是 m22930 代码里已实现、
   但从未被真实数据触发过的字段（此前工件 `meshes=[10]`，单网格时该字段恒真、
   不承载信息）。本单让它第一次由**两个真实网格**喂数。
3. **回归门** `tests/test_cks1998_vortex_time_power_production_mesh.py`：
   把「双网格已测、判决一致、空判据认证在生产网格上成立」写成断言，
   工件回退到单网格或判决被手改即红。

## ③ 计算结果（程序化注入：`validation/cks1998_vortex_2d/time_power_closure.json`）

（数字由 `gen_report_m23825_cks1998_production_mesh.py` 从工件程序化注入，零手抄。）

## ④ 结果图

（matplotlib 生成，内嵌报告：生产网格密度场云图 + 双网格 η(CFL) 梯次比较 + 剖面 vs 解析解。）

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

（待工件回传后填写。）

## ⑥ 误差判定

（待工件回传后填写：关键量 vs 基准逐项，显式回答「是否 < 3 %」。）
