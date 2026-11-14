# CASE-DG-ADVECT-ORDER（终态统一报告）：P0-P4 线性对流收敛率复测

**锚点** CASE-DG-ADVECT-ORDER | **判据** 收敛率 | **总判定** **PASS**
（未过项：无）

## 0. 误差判定（是否 <3%）

本算例判据是**收敛率**（观测阶 vs 文献/理论），不是绝对误差。

### P1-P4 收敛率（vs 文献同阶列）

| p | 文献拟合阶 | TensorDG 拟合阶 | 绝对偏差 | 相对偏差 | <3% |
|--:|--:|--:|--:|--:|--:|
| P1 | 1.990 | 2.037 | +0.047 | 2.37% | ✅ |
| P2 | 2.999 | 3.000 | +0.001 | 0.04% | ✅ |
| P3 | 4.000 | 4.000 | -0.000 | 0.01% | ✅ |
| P4 | 5.025 | 4.995 | -0.031 | 0.61% | ✅ |

### P0 收敛率（vs 理论阶 1.0）

| 项 | 值 | 判定 |
|---|---|:--:|
| 渐近区阶（N>=1280） | 0.9992 | ✅ |
| 渐近常数闭式匹配 | 0.0236% | ✅ |
| 求解器 vs 精确半离散 | 0.000025% | ✅ |

### 五项判据总表

| 判据 | 描述 | 阈值 | 实测 | 判定 |
|---|---|---|---|:--:|
| U1 rate | P1-P4 观测阶 vs ref | +/-0.10 阶 | 最大偏差 0.047 阶 | ✅ |
| U2 P0 order | P0 渐近区阶 | +/-0.01 | 0.9992 | ✅ |
| U3 P0 constant | 渐近常数闭式匹配 | +/-0.1% | 0.0236% | ✅ |
| U4 time share | P1-P4 时间误差占比 | <=10% | 6.73% | ✅ |
| U5 solver exact | P0 求解器 vs 精确解 | <=0.001% | 0.000025% | ✅ |

**结论：是，全部五项判据通过。**

## 1. 案例介绍

一维周期线性对流 u_t + c u_x = 0，x in [0,1)，c=1，
u(x,0) = sin(2*pi*x)，t* = cT/L = 1。
精确解 u = sin(2*pi*(x-t)) 解析已知（manufactured solution）。

| 项 | 符号/定义 | 值 |
|---|---|---|
| 单元数（P1-P4） | N | 20, 40, 80, 160, 320 |
| 多项式阶 | p | 0, 1, 2, 3, 4 |
| CFL | c*dt/h | 0.2 |
| 时间格式 | P1-P4 | 阶数匹配表：p+1 阶 / p+1 级 |
| 时间格式 | P0 | RK4（默认） |
| 无量纲末时 | t* | 1.0 |

**ref 出处**：Zheng Sun,
*Reducing polynomial degree by one for inner-stage operators affects neither stability type nor accuracy order of the Runge-Kutta discontinuous Galerkin method*, Mathematics of Computation (AMS),
DOI 10.1090/mcom/4037, 表 `tab:acc-sin-1d-uni`。
由 `tools/fetch_ref_advorder.py` 机械解析（零手抄）。

## 2. 怎么做的

本统一报告整合三轮独立研究：

1. **Round 1**（m19976）：P1-P4 用 RK4 跑阶梯，与 ref 对比。
   发现 R2/R6（"RK4 时间阶封顶"假设）被数据否定。

2. **Round 2**（m20415）：P1-P4 用 p+1 阶/p+1 级 RK 表复测，
   并用 `expm(L*T)u0` 精确拆分时间/空间误差。

3. **P0 闭式分析**（m20605）：DG0 半离散误差闭式表达
   E(N) -> pi*sqrt(2*pi^2 + 1/6)/N，
   通过 N=20..40960 阶梯验证预渐近区与渐近区。

4. **统一整合**（本单）：五项判据（U1-U5）一次性验证，
   形成完整证据链。

## 3. 计算结果

### 3.1 P1-P4 收敛率（同阶匹配）

| p | 格式 | 级数 | ref 列 | ref 拟合阶 | TensorDG 拟合阶 | 差 | 判定 |
|--:|---|---:|---|---:|---:|---:|:--:|
| P1 | rk2 | 2 | RK2DG1 | 1.990 | 2.037 | +0.047 | ✅ |
| P2 | rk3 | 3 | RK3DG2 | 2.999 | 3.000 | +0.001 | ✅ |
| P3 | rk4 | 4 | RK4DG3 | 4.000 | 4.000 | -0.000 | ✅ |
| P4 | rk5 | 6 | RK5DG4 | 5.025 | 4.995 | -0.031 | ✅ |

### 3.2 P0 渐近常数

| 项 | 值 |
|---|---|
| 闭式常数 | 14.016530 |
| 实测常数（N=40960） | 14.013225 |
| 相对误差 | 0.0236% |
| 预渐近->渐近过渡 N | 1280（阶偏差 <1%） |

### 3.3 时间/空间误差拆分（cfl=0.2）

| p | N | E_total | E_space | E_time | E_time 占比 |
|--:|--:|--:|--:|--:|--:|
| P1 | 8 | 3.718e-02 | 3.688e-02 | 1.960e-03 | 5.2714% |
| P1 | 16 | 7.502e-03 | 7.464e-03 | 5.051e-04 | 6.7323% |
| P2 | 8 | 1.677e-03 | 1.676e-03 | 5.738e-06 | 0.3423% |
| P2 | 16 | 2.090e-04 | 2.089e-04 | 7.175e-07 | 0.3434% |
| P3 | 8 | 7.982e-05 | 7.982e-05 | 9.431e-09 | 0.0118% |
| P3 | 16 | 5.042e-06 | 5.042e-06 | 5.868e-10 | 0.0116% |
| P4 | 8 | 3.093e-06 | 3.093e-06 | 1.274e-12 | 0.0000% |
| P4 | 16 | 9.732e-08 | 9.732e-08 | 4.164e-14 | 0.0000% |

## 4. 结果图

- `advorder_unified_convergence.png`：P1-P4 收敛曲线（实线）vs ref 同阶列（虚线叉）
- `advorder_unified_error_split.png`：E_time vs E_space 散点
- `advorder_unified_p0_constant.png`：P0 渐近常数收敛

## 5. 注意点（踩坑/局限）

- **预渐近区**：P0 在 N<640 时阶 <0.98，是预渐近行为，不是缺陷。
- **Butcher 障碍**：不存在 5 级 5 阶显式 RK，P4 用 6 级 5 阶表（Butcher）。
- **expm 限制**：精确拆分仅适用小网格（N<=16 for P0, N<=8 for P4）。
- **文献局限**：ref 未公布 dt/CFL 与通量形式，L2 幅值无法判定。

## 6. 被证伪的假设

- H_P0_defect: 'the 0.85 observed P0 order on N=20..320 means the DG0 operator is below design order'. Falsified: the closed-form semidiscrete error (no time stepping, no solver) gives the same fitted order on the same ladder, and the design order 1.00 is recovered on the extended ladder.
- H_constant: 'N*E(N) -> pi^2*sqrt(2) = 13.9577' (the scheme-damping estimate alone). Falsified: the measured value at N=40960 is 14.01322, i.e. 0.40% above it, while the two-term closed form pi*sqrt(2*pi^2 + 1/6) = 14.01653 is matched to 0.024%. The gap is the O(h^2) piecewise-constant projection floor pi^2*h^2/6, not a solver error.
- H_time_error: 'the DG0 pool-ladder error is contaminated by RK4 time integration'. Falsified at cfl=0.2: time share <= 4.8e-07 on the pool ladder and the fully discrete solver matches the semidiscrete exact error to 2.5e-07 relative over N=20...5120.

## 7. 最终结论

**CASE-DG-ADVECT-ORDER 验收结论：PASS**

- P1-P4 收敛率全部通过文献对照判据（最大绝对偏差 0.047 阶）
- P0 渐近收敛阶 0.9992 与理论值 1.0 一致
- P0 渐近常数与闭式 pi*sqrt(2*pi^2 + 1/6) 匹配至 0.0236%
- 时间误差在 CFL=0.2 下对 P1-P4 可忽略（最大 6.73%）
- 求解器与精确半离散解匹配至 2.5e-7 相对误差
