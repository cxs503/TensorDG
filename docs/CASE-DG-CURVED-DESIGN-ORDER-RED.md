# CASE-DG-CURVED · 曲线网格 Euler DG 设计阶未达标（HEAD 红测）+ 稳定性归因纠正

> 锚点：`REPRO-DG-CKS1998`（Cockburn & Shu 1998, JCP 141:199-224, RKDG V: multidimensional systems）
> 分支：`agent/m21491`　日期：2026-11-15　类型：**缺陷定位（honest FAIL，未修复）**
> 范围：`tensordg/euler_curved.py`（曲边等参元 RKDG）；只读诊断，未改动求解器。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

CKS1998 的多维 RKDG 必须在**曲边/扭曲网格**上保持设计阶 `p+1`，否则"流场绕圆柱"
（§4 的 curved-wall 算例）无意义。仓内 `tensordg/euler_curved.py` 是该能力的实现，
`tests/test_euler_curved.py` 是它的验收测试（F1–F6 门）。

本单复核该模块在 **HEAD（b4a3f3e）** 上的两条验收线：

| 项 | 设置 |
|---|---|
| 几何 | `warped_box_mesh(cells, cells, degree=3, length=10, amplitude=0.06)`（波长拟合的扭曲周期盒） |
| 状态 | 等熵涡 `euler_2d.vortex_primitive`，strength=5, background=(1,1), 振幅 1 |
| 网格 | 8×8 与 16×16 扭曲单元，P1，CFL=0.2 |
| 时间 | t=0 → 0.1，SSP-RK3 |
| 判据（论文级 `order ≥ p + 0.5`） | P1 要求 `order ≥ 1.5` |

## ② 怎么做的（方法 / 流程 / 工具链）

1. 跑仓内验收测试：`PYTHONPATH=. python -m pytest tests/test_euler_curved.py -x -q`。
2. 复现失败项并打印数字：`pytest tests/test_euler_curved.py::test_design_order_on_the_warped_mesh -q`。
3. **归因诊断**（自写脚本，不 import 求解器内部私有量，只用公开 API）：
   `examples/curved_euler_operator_diagnostics.py`，四组实验：
   - **E1 边界无关性**：同一均匀流基态 + 1e-6 随机模态扰动，在 (a) 全周期、(b) 周期 x+壁面 y、
     (c) 出流 x+壁面 y、(d) 周期 y 四种边界组合下测每步增长比；
   - **E2 通量无关性**：把 `euler_curved.hllc_normal` 猴子补丁换成 Rusanov / 中心通量，重跑同一 marched；
   - **E3 阶数**：面元阶 p=0,1,2 的同类 marched；
   - **E4 光滑 vs 随机扰动**：1e-6 幅度**静止熵波**（ρ 扰动、p 与 u 不变，解析上严格驻定）。
4. 记录 dt 依赖性（dt, dt/2, dt/4, dt/10）以区分"时间步/CFL"与"半离散算子"。

## ③ 计算结果（数字全部来自实跑，可复算）

### 3.1 HEAD 红测（验收费**

```
tests/test_euler_curved.py::test_design_order_on_the_warped_mesh FAILED
AssertionError: (1.153263323359299, [0.05437350350748853, 0.024446712438790184])
assert 1.153263323359299 >= 1.5
```

| 网格 | worst 相对 L2 误差 | 设计阶 | 观测阶 | 判定 |
|---|---|---|---|---|
| 8×8 扭曲 P1 | **5.437e-2** | 2 | — | — |
| 16×16 扭曲 P1 | **2.445e-2** | 2 | **1.153** | **FAIL**（门 p+0.5=1.5） |
| 同文件其余 7 项 | F1/F1n/F2/F4/F5/F6 | — | — | PASS（7 passed 后在此项失败） |

复算命令：`PYTHONPATH=. python -m pytest tests/test_euler_curved.py::test_design_order_on_the_warped_mesh -q`

### 3.2 归因诊断（E1–E4，全周期均匀 8×6 网格、仿射单元、p=0/1）

| 实验 | 配置 | 结果 | 含义 |
|---|---|---|---|
| E1 | 全周期（无壁面、无边界） | 失稳，dt 归一化后增长率 ~40 量级 | **不是壁面 BC 的问题** |
| E1 | 周期 x + 壁面 y | 每步增长比与全周期**逐位同量**（1.0998 @dt/2） | 壁面不改变增长率 |
| E1 | 出流 x + 壁面 y | 同上 | 边界组合无关 |
| E2 | 通量换成 Rusanov | 同时步发散（p=0: step 177 vs 175） | 该模态不由 Riemann 通量耗散控制 |
| E2 | 通量换成中心通量 | 同时步发散（p=1: step 167） | 同上 |
| E3 | p=0（K=1，退化为 FV） | 同样发散（step 175） | 与高阶模态无关 |
| E4 | 1e-6 **静止熵波**（严格驻定解） | 前 t≈1.6 **比值恒 1.000**，随后 25 步内冲到 1e37 | 延迟爆裂，非指数线性不稳定 |
| E4 | 同一试验 dt/4 | t=1.19 内比值恒 1.000 | 触发时刻随 dt 变化 |

注：`rhs(均匀基态)=6.7e-15`（F1 自由流保持成立），质量/动量/能量残差在光滑扰动下为 1e-16 量级
（守恒成立）——**即：守恒与自由流保持都通过，但扰动仍会延迟爆裂**。

## ④ 结果图

本单为**缺陷定位**类，产出的是"红测 + 抵消实验"的数字判据，没有可信的物理场云图可画
（求解器在诊断配置下最终发散，场量无意义）。若需图示：失败项的两条误差（5.44%/2.44%）在
③.1 表中已给全，作图只是重绘。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **模块 docstring 的归因不成立（本单的主要纠正）**：`euler_curved.py` 把爆裂归因为
   "reflective boundary is linearly unstable"，但 E1 证明**完全周期、没有任何壁面**的均匀网格
   上同样爆裂。壁面通量（`wall_normal_flux`）至多是同源症状的放大器，不是根因。
   → 建议把 docstring 的因果句改写为"内部算子存在未定位的延迟爆裂；壁面算例只是最先撞上它"。
2. **不是 CFL/dt**：dt→dt/10 时增长率（每单位时间）不随 dt 下降，且 E2 显示换耗散更强的
   Rusanov 也无效。
3. **不是 Riemann 通量耦合**：三种通量（HLLC/Rusanov/中心）发散步号仅差 2 步。
4. **未定位的部分（诚实声明）**：根因仍未找到。E4 的"前 1.6 时间单位完美保持、随后突变"形态
   排除纯线性不稳定，指向某个**状态相关的分支/钳位**（`pres = max(·, 1e-12)`、HLLC 波号分支、
   或 `apply_mass_inverse` 的逐单元逆矩阵），本单预算内未取证。
5. **设计阶失败很可能同源**：扭曲网格 P1 只有 1.15 阶（误差 5.4%@8×8）与"细尺度被污染"一致，
   但也可能独立（几何/度量一致性）。两者未能分离，不得互相代替证据。
6. 不得借此单的结果声称圆柱绕流可算——现状仍是 **NOT_SUPPORTED**。

## ⑥ 误差判定（逐项 vs 基准）

| 关键量 | 基准 | 实测 | 误差 | <3%? |
|---|---|---|---|---|
| 扭曲网格 P1 收敛阶 | CKS1998 设计阶 p+1=2（门 p+0.5=1.5） | 1.153 | **−42.4%**（相对 2） | **否** |
| 8×8 worst 相对 L2 误差 | 0（同分辨率直边网格 F4 对标 ~1e-2 量级） | 5.44e-2 | 结构性偏差 | **否** |
| 均匀基态残差（自由流保持） | 0 | 6.7e-15 | — | 是 |
| 光滑扰动守恒残差 | 0 | ~1e-16 | — | 是 |

**honest FAIL**：锚点 `REPRO-DG-CKS1998` 的曲边实现腿**未达标**（设计阶红测挂在 HEAD）。

## ⑦ 复现命令

```bash
PYTHONPATH=. python -m pytest tests/test_euler_curved.py::test_design_order_on_the_warped_mesh -q   # 红
PYTHONPATH=. python -m pytest tests/test_euler_curved_design_order_gap.py -q                          # 本单的钉子测试（绿）
PYTHONPATH=. python examples/curved_euler_operator_diagnostics.py                                    # 归因诊断（~60 s）
```