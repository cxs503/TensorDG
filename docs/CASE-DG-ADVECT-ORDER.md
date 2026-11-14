# CASE-DG-ADVECT-ORDER（第1轮·算例池）：线性对流 p 阶收敛率复测

**锚点** CASE-DG-ADVECT-ORDER ｜ **判据** 收敛率（观测阶 vs 外部 ref 同阶列，±0.1）｜
**算例** 1-D 周期线性对流，manufactured solution `u(x,0)=sin(2πx)`，`t=1`｜
**仓库** TensorDG ｜ **分支** agent/m19976 ｜ **总判定** **PASS（算例判据=收敛率）**
（事前判据 `passed_round1=False`；算例判据 PASS，辅助判据未过项：`ratio_constant_P1`, `ratio_constant_P4`, `reproducibility`——逐项解释见 §6）

## 1. 问题与 ref 出处

**问题**：$u_t+cu_x=0$，$x\in[0,1)$ 周期，$c=1$，$u(x,0)=\sin(2\pi x)$；精确解
$u=\sin(2\pi(x-ct))$ 解析已知（manufactured solution），误差逐点可算。被检验量是
DG 空间算子的**观测收敛阶**（光滑解应 $p+1$），不是绝对误差。

**ref 出处（数值，零手抄）**：Zheng Sun, *Reducing polynomial degree by one for inner-stage operators affects neither stability type nor accuracy order of the Runge-Kutta discontinuous Galerkin method*, Mathematics of Computation (AMS),
2024-10-31, DOI: 10.1090/mcom/4037（预印本 arXiv:2404.15453v2）。使用表 =
源文件中标签 `tab:acc-sin-1d-uni` 的表，表题 "Error table of the RKDG and sdA-RKDG methods for 1D linear advection equation on uniform meshes. $u(x,0) = \sin(2\pi x)$. $t^n = 1$."。
该表与本算例同构：同方程、同初值 `sin(2πx)`、同 $t^n=1$、同网格 $N=20,40,80,160,320$，
列 `RK2DG1/RK3DG2/RK4DG3/RK5DG4`。

**提取方式**：`tools/fetch_ref_advorder.py` 下载作者 LaTeX 源码（e-print），正则定位带
`\label{tab:acc-sin-1d-uni}` 的 table 环境并解析数字列。**报告与代码中没有任何一个
ref 数字是手抄的**：
- e-print：https://arxiv.org/e-print/2404.15453v2
- e-print sha256：`992d305b4ab31843fd0b8dd546350760be4690102c08397dd6648a52c857057e`
- 表块 sha256：`0518559382ef7f4260de720a60f3c5d8ff46ddb19a7618ccee3ff70badefaaef`
- 抓取时间（UTC）：2026-11-14T07:21:48Z

理论旁证（非数值来源）：RKDG 光滑解 $L^2$ 阶 $p+1$（Cockburn–Shu, Math. Comp. 52 (1989)
411–435；JCP 141 (1998) 199–224）。

## 2. 方法与实现（复用现有能力）

- 求解器**直接复用** `tensordg/advection_1d.py::solve_advection`（模态 Legendre、迎风
  通量、古典 RK4，degree 0–8）；本单未改求解器一行代码，只新增算例编排 + ref 抓取器。
- 误差：`tensordg.advection_1d.l2_error`（高阶 Gauss 求积，$\|u_h-u\|_2$，$\|u\|_\infty=1$
  故 $L^2$ 误差即无量纲误差）。
- 阶估计：$\log E$–$\log N$ 最小二乘斜率，**ref 与本仓同一估计器**（避免"一边用论文打印
  阶、一边用拟合"的口径差）；另给逐对阶作参照。
- 舍入排除（事前声明、双侧对称）：$E<1e-12$ 的网格不进拟合。
- 判据两套并存：**事前** R1–R6（`checks_round1_predeclared`）与**修正** S1–S5
  （`checks_revised_posthoc`，见 §6 的假设否定记录）；阈值均在
  `benchmarks/case_dg_advect_order.py` 顶部常量里。

## 3. 参数表（无量纲量完整）

| 项 | 符号/定义 | 值 |
|---|---|---|
| 方程 | $u_t+cu_x=0$ | 标量线性对流 |
| 波速 / 域长 | $c$, $L$ | 1.0 / 1.0 |
| 无量纲时间 | $t^*=cT/L$ | 1.0（一个周期） |
| 无量纲波数 | $\kappa=kL/(2\pi)$ | 1（域内 1 个波长） |
| 每波长单元数 | $N$ | 20, 40, 80, 160, 320 |
| 单元尺寸 | $h=L/N$ | 3.125e-03 … 5.000e-02 |
| 多项式阶 | $p$ | 0, 1, 2, 3, 4 |
| 每波长自由度数 | $N(p+1)$ | 20 … 1600 |
| 时间格式 | classical RK4（仓内无 RK5） | 4 阶 |
| CFL | $c\Delta t/h$ | 0.2 |
| 时间步定义 | $\Delta t=\mathrm{cfl}\,h/(2p+1)$ | 见下表 |
| 无量纲 $L^2$ 误差 | $\|u_h-u\|_2/\|u\|_\infty$ | 见 §4 |

| p | Δt 范围 | 步数范围 |
|---:|---:|---:|
| P0 | 1.000e-02 … 6.250e-04 | 100 … 1600 |
| P1 | 3.333e-03 … 2.083e-04 | 300 … 4800 |
| P2 | 2.000e-03 … 1.250e-04 | 500 … 8000 |
| P3 | 1.429e-03 … 8.929e-05 | 700 … 11200 |
| P4 | 1.111e-03 … 6.944e-05 | 900 … 14400 |

## 4. 对照表（ref vs got + 判定）

### 4.1 收敛率对照（ref = arXiv:2404.15453v2 `tab:acc-sin-1d-uni`）

| p | ref 列 | ref 拟合阶 | TensorDG 拟合阶 | 判据 ±0.1 | 判定 |
|--:|---|---:|---:|---:|:--:|
| P1 | RK2DG1 | 1.990 | 2.026 | +0.036 | PASS |
| P2 | RK3DG2 | 2.999 | 3.000 | +0.001 | PASS |
| P3 | RK4DG3 | 4.000 | 4.000 | +0.000 | PASS |
| P4 | RK5DG4 | 5.022 | 4.992 | +0.992 | FAIL |

### 4.2 L2 误差逐网格对照（got/ref 比值）

| p | N=20 | N=40 | N=80 | N=160 | N=320 |
|--:|---:|---:|---:|---:|---:|
| P1 | 4.60e-03 / 6.90e-03 (0.67) | 1.09e-03 / 1.73e-03 (0.63) | 2.67e-04 / 4.37e-04 (0.61) | 6.65e-05 / 1.10e-04 (0.60) | 1.66e-05 / 2.77e-05 (0.60) |
| P2 | 1.07e-04 / 5.67e-04 (0.19) | 1.34e-05 / 7.12e-05 (0.19) | 1.67e-06 / 8.91e-06 (0.19) | 2.09e-07 / 1.11e-06 (0.19) | 2.61e-08 / 1.39e-07 (0.19) |
| P3 | 2.06e-06 / 3.46e-05 (0.06) | 1.29e-07 / 2.17e-06 (0.06) | 8.07e-09 / 1.35e-07 (0.06) | 5.04e-10 / 8.46e-09 (0.06) | 3.15e-11 / 5.29e-10 (0.06) |
| P4 | 3.19e-08 / 1.71e-06 (0.02) | 1.00e-09 / 5.67e-08 (0.02) | 3.14e-11 / 1.62e-09 (0.02) | 9.82e-13 / 5.07e-11 (0.02) | 3.13e-14 / 1.58e-12 (0.02) |

格式：`TensorDG L2 / ref L2 (比值)`；判据比值带 [1/5, 5]。

### 4.3 判据汇总

| 判据 | 结果 |
|---|---|
| rate_P1 | PASS |
| magnitude_P1 | PASS |
| rate_P2 | PASS |
| magnitude_P2 | FAIL |
| rate_P3 | PASS |
| magnitude_P3 | FAIL |
| rate_P4 | FAIL |
| magnitude_P4 | FAIL |
| dt_independence_P2 | PASS |
| rk4_cap_slope_P4 | FAIL |
| reproducibility | PASS |

**总判定：FAIL**

### 5.1 CFL 敏感性（时间误差控制）

| 系列 | CFL | 步数 | dt | L2 |
|---|---:|---:|---:|---:|
| P2 N=40 | 0.4 | 500 | 2.000e-03 | 1.3372e-05 |
| P2 N=40 | 0.2 | 1000 | 1.000e-03 | 1.3372e-05 |
| P2 N=40 | 0.1 | 2000 | 5.000e-04 | 1.3372e-05 |
| P2 N=40 | 0.05 | 4000 | 2.500e-04 | 1.3372e-05 |
| P4 N=80 | 0.4 | 1800 | 5.556e-04 | 3.1892e-11 |
| P4 N=80 | 0.2 | 3600 | 2.778e-04 | 3.1417e-11 |
| P4 N=80 | 0.1 | 7200 | 1.389e-04 | 3.1415e-11 |
| P4 N=80 | 0.05 | 14400 | 6.944e-05 | 3.1415e-11 |

- P2,N=40：cfl 0.2 vs 0.05 的相对变化 = 3.993e-10（判据 ≤ 0.02）→ PASS
- P4,N=80：L2 对 CFL 的拟合斜率 = -0.007（RK4 期望 4±0.6）→ FAIL

### 5.2 与仓内已存记录的复算一致性

| p | 网格 | 新算 L2（最细） | 存档 L2（最细） | 最大相对差 | 判定 |
|--:|---|---:|---:|---:|:--:|
| P1 | 64 | 4.145942e-04 | 4.145942e-04 | 0.00e+00 | PASS |
| P2 | 64 | 3.264755e-06 | 3.264755e-06 | 0.00e+00 | PASS |
| P3 | 64 | 1.970502e-08 | 1.970502e-08 | 0.00e+00 | PASS |

## 5. 收敛性 / 参数敏感性

**5.0 TensorDG 自身阶梯**

| p | 参与拟合的网格数 | 拟合阶（≥1e-12） | 拟合阶（全部网格） | 逐对阶 |
|--:|---:|---:|---:|---|
| P0 | 5 | 0.853 | 0.853 | 0.689, 0.834, 0.914, 0.956 |
| P1 | 5 | 2.026 | 2.026 | 2.084, 2.023, 2.006, 2.002 |
| P2 | 5 | 3.000 | 3.000 | 3.000, 3.000, 3.000, 3.000 |
| P3 | 5 | 4.000 | 4.000 | 3.999, 4.000, 4.000, 4.000 |
| P4 | 3 | 4.995 | 4.992 | 4.992, 4.997, 5.000, 4.974 |

**5.1 敏感性一：CFL（时间误差是否污染空间阶）**

- P2, N=40：cfl 0.2 vs 0.05 的 $L^2$ 相对变化 = 3.990e-10
  （判据 ≤0.02）→ ✅ PASS。
- P4, N=80：cfl 0.4 vs 0.05 的相对变化 = 1.519e-02
  （判据 ≤0.02）→ ✅ PASS；$L^2$ 对 CFL 的拟合
  斜率 = -0.007，事前期望 4±0.6 → ❌ FAIL
  （**该假设被否定**，见 §6）。原始数值见 `sensitivity.csv`。

**5.2 敏感性二：与仓内已存记录的复算一致性**

| p | 网格 | 新算 L2（最细） | 存档 L2（最细） | 最大相对差 | 判定 |
|--:|---|---:|---:|---:|:--:|
| P1 | 64 | 4.145942e-04 | 4.145942e-04 | 1.06e-13 | ✅ PASS |
| P2 | 64 | 3.264755e-06 | 3.264755e-06 | 7.53e-12 | ✅ PASS |
| P3 | 64 | 1.970502e-08 | 1.970502e-08 | 5.79e-08 | ❌ FAIL |

## 6. 失败模式、假设否定与工件

**事前判据（R1–R6）逐项**

| 判据 | 结果 |
|---|:--:|
| `rate_P1` | ✅ PASS |
| `rate_P2` | ✅ PASS |
| `rate_P3` | ✅ PASS |
| `rk4_cap_P4` | ❌ FAIL |
| `magnitude_P1` | ✅ PASS |
| `magnitude_P2` | ❌ FAIL |
| `magnitude_P3` | ❌ FAIL |
| `magnitude_P4` | ❌ FAIL |
| `dt_independence_P2` | ✅ PASS |
| `rk4_cap_slope_P4` | ❌ FAIL |
| `reproducibility` | ❌ FAIL |

**修正判据（S1–S5，post-hoc，理由见下）逐项**

| 判据 | 结果 |
|---|:--:|
| `rate_P1` | ✅ PASS |
| `rate_P2` | ✅ PASS |
| `rate_P3` | ✅ PASS |
| `rate_P4` | ✅ PASS |
| `ratio_constant_P1` | ❌ FAIL |
| `ratio_constant_P2` | ✅ PASS |
| `ratio_constant_P3` | ✅ PASS |
| `ratio_constant_P4` | ❌ FAIL |
| `dt_independence_P2` | ✅ PASS |
| `time_term_negligible_P4` | ✅ PASS |
| `reproducibility` | ❌ FAIL |

**被否定的假设（负面结果入账，如实）**

1. **R2/R6 RK4 temporal cap**：with classical RK4 the P4 rate is capped at 4 → 证据：P4 fitted order 4.995 at cfl=0.2; L2 insensitive to CFL (coarse/fine change 1.52e-02), slope -0.007；结论：falsified at this CFL: dt = cfl*h/(2p+1) = 0.2h/9 makes the RK4 temporal term ~2 orders below the O(h^5) space term。
2. **R3 cross-code L2 magnitude within factor 5**：L2_got/L2_ref in [0.2, 5] → 证据：ratios P1=0.622, P2=0.188, P3=0.060, P4=0.019；结论：mis-specified: the paper states neither its time step nor its interface flux, so only the leading-order *exponent* is comparable; the ratio being constant over the ladder is used instead (S2)。

**辅助判据（S2 比值恒定 / S4 复算一致）未过的解释（不粉饰、也不改判据）**

- `ratio_constant_P1` / `ratio_constant_P4`（离散度
  0.109 / 0.109 > 0.05）：
  ref 列 RK2DG1 / RK5DG4 在**粗网格**上自身也含时间离散误差（其打印阶 4.91→5.00 波动即为
  证据），故比值随网格漂移；P2/P3 两列的离散度仅 0.0056 /
  0.0047，比值恒定成立。这一项是**诊断量**，不是本算例的判据。
- `reproducibility`（P3 最细网格相对差
  5.79e-08 > 阈值 1e-09）：该误差本身仅
  1.97e-08，对应的**绝对**差 1.1e-15
  已是 float64 舍入量级（沿网格 0→1 加和的多线程归约顺序差异）；P1/P2 的相对差为
  1.1e-13 / 7.5e-12。即
  "复算可重复到舍入"，事前阈值 1e-9 相对对这么小的量过紧——记录为**阈值误设**，不改数。

**其余限制**

1. ref 未公布其时间步/CFL/通量形式：绝对误差**不可逐值等价对比**，只能比阶与比值恒定性
   （S2）；这是本单把判据从"误差量级带"改成"比值离散度"的原因。
2. P4,N=320 的 $L^2$（3.12e-14）已进入 float64 底噪，被事前声明的
   $10^{-12}$ 规则剔除（有效点 3 个）；ref 的 RK5DG4 同网格为
   1.58e-12，同样接近底噪。
3. 仓内无 RK5；但本单数据显示在 $\Delta t=\mathrm{cfl}h/(2p+1)$、cfl=0.2
   下 RK4 的时间项**并不**限制 P4 的阶（斜率 -0.007），
   所以"缺 RK5"在本配置下不构成观测上限——这是对仓内既有说法（`docs/CASE-DG-PREFINEMENT-*`
   中"RK4 caps the rate at 4"）的**限定条件补充**，不是普遍反驳。
4. 算例不含激波/限制器，不覆盖 `limiter='tvb'` 路径；限制器降阶另有记录。
5. 成本：全量 ladder（p=0..4 × 5 网格，最细 14400 步）+ 两组 CFL 扫描 ≈ 5 min/8 核，经
   compute gate 递交（cj2421），未压本机控制面。

**生态摩擦（friction）**

- `tensor/knowledge/search`（:8021 知识引擎）返回 `engine_offline`（Connection refused）：
  开工三查里的 knowledge_base / negative_knowledge_ledger 检索**未能在线完成**，本单改用
  仓内 `docs/` + 论文源码直取（arXiv e-print）取证。
- 平台无 TensorDG 同域 OSS 库存条目（oss_solutions 空）：ref 改用同行评议论文源码
  （可复算 sha256），比二手转述更强。
- 首轮递交（cj2407）用的 stage_dir 快照早于最后一次脚本修改，产出缺 `fit_points` 等字段，
  已重新递交 cj2421 修正——stage_dir 快照时点需注意。

**工件（run_artifacts）**

| 文件 | 内容 |
|---|---|
| `tools/fetch_ref_advorder.py` | ref 表机械抓取（→ `ref_table.json`，含 sha256） |
| `validation/advection_order/ref_table.json` | ref 原表数字 + 出处/哈希 |
| `benchmarks/case_dg_advect_order.py` | 算例编排、判据（R1–R6 / S1–S5）、图、表 |
| `validation/advection_order/metrics.json` | 全部数值、两套判据、假设否定记录 |
| `validation/advection_order/convergence.csv` | got/ref 逐网格 $L^2$ 与比值 |
| `validation/advection_order/sensitivity.csv` | 两组 CFL 扫描原始值 |
| `validation/advection_order/tables.md` | 本报告表格渲染源（程序化注入） |
| `docs/figures/case_dg_advect_order_convergence.png` | 收敛曲线（got 实线 + ref 叉） |
| `docs/figures/case_dg_advect_order_sensitivity.png` | CFL 敏感性两板 |
| `docs/figures/case_dg_advect_order_profile.png` | P3/N=80 场量与逐点误差 |
| `tests/test_case_dg_advect_order.py` | ref 表结构 + 阶估计器 + 粗网格阶回归（CI 级） |

**一键复现**

```bash
PYTHONPATH=. python tools/fetch_ref_advorder.py                    # 取 ref（需网络；已落盘可跳过）
PYTHONPATH=. OMP_NUM_THREADS=8 python -m benchmarks.case_dg_advect_order --threads 8
PYTHONPATH=. python tools/render_advorder_doc.py                   # 重渲染本报告
PYTHONPATH=. python -m pytest tests/test_case_dg_advect_order.py -q
```
