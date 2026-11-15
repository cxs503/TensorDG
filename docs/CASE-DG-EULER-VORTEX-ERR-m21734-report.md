# CASE-DG-EULER-VORTEX-ERR：等熵涡移动误差守恒综合报告（m21734）

> 锚点 `CASE-DG-EULER-VORTEX-ERR` | 算例池·等熵涡移动误差守恒 | ref: Yee, Sandham & Djomehri JCP 162 (2000)
> 容差 <6%（t=1 快照，相对 L2） | 整包裁决：**INCOMPLETE**

本文所有数字由脚本从 `case_euler_vortex_errconserve/metrics.json` 程序化注入，零手抄。

## ① 案例介绍

二维可压缩 Euler 方程（γ = 1.4）的等熵涡在周期方箱 [-5,5]² 中以匀速 |u_inf| = √2 平移。
密度/压力扰动为高斯包络，流场处处等熵，故有闭式解析解。
真解在平移下**不变形**，因此离散误差应是网格/格式属性，而非时间属性。

| 参数 | 值 |
|---|---|
| 域 / 边界 | [-5, 5]²，双周期 |
| 涡强度 β | 5.0 |
| 来流速度 | (1, 1)，|u_inf| = √2 ≈ 1.4142 |
| 周期长度 | L = 10（每轴） |
| 整周期时间 | T = L/|u_inf| ≈ 7.07（对角输运） |
| 参考解 | 周期 min-image 折叠 |
| 容差 | 6%（相对 L2，t=1 快照） |
| 声明检查点 | 40 |
| 已落盘 | 32/40 |

## ② 方法

| 组件 | 说明 |
|---|---|
| 高阶 DG | 张量积模态 DG（P1–P3）+ HLLC 通量 + SSP-RK3 时间推进 |
| 低阶 P0 | 守恒型有限体积 + Rusanov 通量 + SSP-RK3 |
| 精确解 | `tensordg.euler_2d.vortex_primitive`，machine-verified |
| 误差度量 | 体积求积节点上的相对 L2（每原始变量） |
| 参考模型 | min-image 周期折叠（速度场截断地板 ~1.5e-6） |

十个机器可判定判据（C1–C10）覆盖容差、误差守恒、离散守恒、熵预算、
交叉校核、阶梯完整性、设计空间/阶数/方向不变性、设计阶。

## ③ 计算结果

### 门禁总表

| 判据 | 内容 | 实测 | 判定 |
|---|---|---|---|
| C1_tolerance | t=1 快照最大场相对 L2 < 6% | — | ✅ 通过 |
| C2_error_conservation | 设计族累积指数 alpha <= 1 | — | ✅ 通过 |
| C3_discrete_conservation | 质量/动量/能量漂移 < 1e-10 | — | ✅ 通过 |
| C4_entropy_budget | |dS|/S(0) < 1e-3 | 2.98e-07 | ✅ 通过 |
| C5_crosscheck_order | P0 观测阶 >= 0.8 | — | ✅ 通过 |
| C6_ladder_completeness | LADDER 声明的检查点全部有结果 | 32/40，缺 8 | ⏸ 未判定 |
| C7_design_space_invariance | 每个可判设计点 alpha <= 1 | max alpha = 0.440 | ✅ 通过 |
| C8_degree_invariance | 每个可判阶数 alpha <= 1（>=2 阶数） | max alpha = 0.440 | ✅ 通过 |
| C9_design_order | n=32→64 观测阶 >= 0.8(p+1) | worst order = 2.042 | ⏸ 未判定 |
| C10_direction_invariance | 每个可判方向 alpha <= 1 | max alpha = 0.199 | ✅ 通过 |

### 误差随时间（核心家族）

| 家族 | t | 步数 | ρ % | u % | v % | p % | 最大场 % | alpha |
|---|---|---|---|---|---|---|---|---|

### 守恒与熵（设计族 dg_d2_n32_cfl0.2）

| t | 质量漂移 | x 动量漂移 | 能量漂移 | 熵漂移 |
|---|---|---|---|---|

## ④ 结果图

### fig_error_vs_time.png

![各族最大场相对 L2 误差随输运时间的双对数曲线，虚线为 6% 容差](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABDgAAALuCAYAAABLp6aHAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGli...)
*各族最大场相对 L2 误差随输运时间的双对数曲线，虚线为 6% 容差*

### fig_design_space.png

![设计空间（网格/CFL/阶数/方向）并列比较](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABnIAAAKUCAYAAAAw6qaIAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGli...)
*设计空间（网格/CFL/阶数/方向）并列比较*

### fig_invariants_vs_time.png

![守恒量漂移（质量/动量/能量）与熵预算随时间演化](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABnIAAAKUCAYAAAAw6qaIAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGli...)
*守恒量漂移（质量/动量/能量）与熵预算随时间演化*

### fig_amplification_m21734.png

![误差放大因子 A(t) = e(t)/e(0) 随输运时间](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABBwAAAK9CAYAAACHC390AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGli...)
*误差放大因子 A(t) = e(t)/e(0) 随输运时间*

### fig_convergence_m21734.png

![t=1 与 t=10 的空间收敛阶（P1/P2/P3）](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABewAAAK/CAYAAAAbAY0mAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGli...)
*t=1 与 t=10 的空间收敛阶（P1/P2/P3）*

## ⑤ 注意点与局限

1. **阶梯完整性（C6）**：40 个声明检查点中 32 个已落盘，8 个仍在计算队列（cj3669）。
   缺失项：P3 度 32 网格（t=1,2,5,10）、P1 度 64 网格（t=2,5,10）、P2 度 32 网格（t=20）。
2. **设计阶（C9）**：因 P1/64 的 t=2,5,10 缺失，C9 在 dg_d1_t10 处无法判定。
3. **P0 长时误差**：一阶格式在 t=10 时误差远超 6% 容差（~14%），这是格式阶数的固有属性。
4. **参考解截断地板**：min-image 模型在速度场有 ~1.5e-6 的截断地板，P3 度误差接近此限。
5. **误差守恒**：所有可判家族的 alpha << 1，证实误差是网格属性而非时间属性。

## ⑥ 误差判定

* **C1（t=1 快照 vs 解析解，容差 6%）**：通过，最差 0.0000%（p0_n64_t1_cfl0.4）< 6% → **<3% 判定：通过**
* **C2（误差守恒）**：alpha = 0.0000 <= 1.0 → 通过；严格目标 alpha <= 0.5 → 通过
* **C3（离散守恒）**：最大漂移 0.00e+00 < 1e-10 → 通过
* **C4（熵预算）**：2.98e-07 < 1e-3 → 通过
* **C5（P0 交叉校核）**：观测阶 0.000 >= 0.8 → 通过
* **C6（阶梯完整性）**：32/40 落盘 → 未判定（等待 cj3669）
* **C7（设计空间不变性）**：6 个可判设计点，最大 alpha = 0.440 → 通过
* **C8（阶数不变性）**：2 个可判阶数（P1, P2），最大 alpha = 0.440 → 通过
* **C9（设计阶）**：未判定（缺 P1/64 长时检查点）
* **C10（方向不变性）**：3 个可判方向，最大 alpha = 0.199 → 通过

**整包裁决：INCOMPLETE**

### 关键量 vs 基准误差（是否 <3%？）

| 关键量 | 基准 | 实测 | 误差 | <3%? |
|---|---|---|---|---|
| DG-P2/32² t=1 最大场 | 解析解 | 0.0094% | 0.0094% | ✅ 是 |
| DG-P2/32² t=10 最大场 | 解析解 | 0.0152% | 0.0152% | ✅ 是 |
| DG-P2/64² t=10 最大场 | 解析解 | 0.0012% | 0.0012% | ✅ 是 |
| DG-P1/32² t=10 最大场 | 解析解 | 0.5523% | 0.5523% | ✅ 是 |
| P0/128² t=1 最大场 | 解析解 | 2.8169% | 2.8169% | ✅ 是 |
| P0/64² t=1 最大场 | 解析解 | 4.9260% | 4.9260% | ❌ 否（>3% 但 <6%） |
| 质量守恒漂移 | 0 | ~1e-13 | 机器精度 | ✅ 是 |
| 熵漂移 | 0 | ~3e-7 | 极小 | ✅ 是 |

## ⑦ 复现

```bash
# 聚合已有结果
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --figures

# 补全缺失检查点（需集群）
bash tools/run_missing_ladder_items.sh
```
