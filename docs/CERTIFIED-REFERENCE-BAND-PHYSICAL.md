# 物理算例上的认证参考解带宽校准（round-2 practice §5 的第二块证据）

> 生成器：`python -m tools.certify_physical_band`；本文全部数字由 `artifacts/certify_physical_band/results.json` 程序化注入（零手抄）。
> 引擎：`tensordg.certified_reference (imported, not re-implemented)`；门限 τ = 3 %（owner 精度标准）。
> 环境：torch 2.7.1 / numpy 2.4.6。

## ① 为什么做这件事（问题定义）

round-2 认证参考解的做法（`docs/CERTIFIED-REFERENCE-PRACTICE.md`）交付了一把尺：
Roache GCI 带宽 `U` + 门有效性判据 `U/τ ≤ 1/3` + 余量判据 `err + U < τ`。
但这把尺**自身**的可靠性只在一处被检验过——`band_coverage_study` 用**合成**阶梯
（误差模型 `A N^-p + B N^-q`）问「带宽是否覆盖认证解的真误差」；其 docstring 自己写明局限：
合成阶梯观测阶高、外推几乎吃掉全部误差，因此带宽**极度保守**，该结论「不迁移到激波算例」。
m17539 交差审计 L1 项把这条记为缺口：**泛化性只有 Shu–Osher 一个物理算例支撑**。

本文补的正是这块：在**有闭式解**的物理 PDE 算例上跑**同一把尺**（import，不重写），
用解析解测出认证解的真误差，从而把「带宽是否覆盖真误差」这个可证伪命题
从合成误差模型搬到真实离散上。

## ② 算例与参数（物理背景 / 几何 / 参数表）

| 算例 | PDE | 初值 | T | 破解时刻 | DG 次数 | 阶梯 N |
|---|---|---|---|---|---|---|
| `advection_sine_P1` | $u_t+u_x=0$ | $\sin 2\pi x$ | 1 | — | P1 | [16, 32, 64] |
| `advection_sine_P2` | $u_t+u_x=0$ | $\sin 2\pi x$ | 1 | — | P2 | [16, 32, 64] |
| `advection_sine_P3` | $u_t+u_x=0$ | $\sin 2\pi x$ | 1 | — | P3 | [16, 32, 64] |
| `advection_gaussian_P2` | $u_t+u_x=0$ | 窄 Gauss，$\sigma=0.06$ | 0.5 | — | P2 | [32, 64, 128] |
| `advection_gaussian_P3` | $u_t+u_x=0$ | 窄 Gauss，$\sigma=0.06$ | 0.5 | — | P3 | [32, 64, 128] |
| `burgers_preshock_P1` | $u_t+(u^2/2)_x=0$ | $0.2+0.8\sin 2\pi x$ | 0.12 | 0.1989 | P1 | [32, 64, 128] |
| `burgers_preshock_P2` | $u_t+(u^2/2)_x=0$ | $0.2+0.8\sin 2\pi x$ | 0.12 | 0.1989 | P2 | [32, 64, 128] |
| `burgers_nearbreak_P1` | $u_t+(u^2/2)_x=0$ | $0.2+0.8\sin 2\pi x$ | 0.18 | 0.1989 | P1 | [32, 64, 128] |
| `burgers_nearbreak_P2` | $u_t+(u^2/2)_x=0$ | $0.2+0.8\sin 2\pi x$ | 0.18 | 0.1989 | P2 | [32, 64, 128] |

* 全部算例为 1-D 周期，域 [0,1]；解由**本仓生产内核**给出（`tensordg.advection_1d.solve_advection` / `tensordg.burgers_1d.solve_burgers`），
  无限幅器（`limiter=None`），RK4 / SSP-RK3 与内核默认一致。
* 阶梯输入为**单元平均值**（模态系数 0 即单元均值，`coarsen` 对均值精确），
  故嵌套关系 `coarsen(fine, 2) ≡ coarse` 精确成立，不引入重建误差。
* 解析解按 64 点 Gauss 求积积分成单元均值（被积函数光滑 → 求积误差不入账）。

## ③ 结果（逐阶梯）

| 阶梯 | 判定 | p_obs | 带宽 U % | 最细原解误差 % | 认证解真误差 % | 真误差/U | Richardson 增益 |
|---|---|---|---|---|---|---|---|
| `advection_sine_P1` | `CERTIFIED` | 2.9876 | 0.010294 | 0.0083 | 2.478e-04 | 0.024 | 267.674 |
| `advection_sine_P2` | `CERTIFIED` | 4.9892 | 1.046e-06 | 8.332e-07 | 1.523e-08 | 0.015 | 1744.138 |
| `advection_sine_P3` | `WITHHELD_ORDER_TOO_HIGH` | 7.1240 | 4.903e-11 | 4.031e-11 | 7.171e-12 | 0.146 | 763.091 |
| `advection_gaussian_P2` | `CERTIFIED` | 4.8869 | 7.423e-05 | 5.964e-05 | 5.232e-06 | 0.070 | 355.930 |
| `advection_gaussian_P3` | `WITHHELD_ORDER_TOO_HIGH` | 6.9364 | 4.451e-08 | 3.589e-08 | 2.091e-09 | 0.047 | 2121.644 |
| `burgers_preshock_P1` | `CERTIFIED` | 2.5653 | 0.001533 | 0.0011 | 3.156e-04 | 0.206 | 22.660 |
| `burgers_preshock_P2` | `CERTIFIED` | 3.8044 | 9.737e-06 | 6.414e-06 | 3.278e-06 | 0.337 | 35.164 |
| `burgers_nearbreak_P1` | `CERTIFIED` | 2.0486 | 0.013301 | 0.0124 | 0.0026 | 0.197 | 15.952 |
| `burgers_nearbreak_P2` | `CERTIFIED` | 2.7230 | 5.816e-04 | 8.700e-04 | 1.930e-04 | 0.332 | 15.518 |

列定义：`最细原解误差` = 最细网格原始 DG 解对解析解的相对 L1 误差（同网格）；
`认证解真误差` = Richardson 外推解在参考网格（次细）上对解析解的相对 L1 误差；
`真误差/U` ≤ 1 即带宽覆盖真误差；`Richardson 增益` = 次细原解误差 / 认证解真误差。

## ④ 汇总判定

* 带宽可测阶梯 9/9，其中带宽覆盖真误差 **9** 条 → 覆盖率 100.0 %。
* 最差 `真误差/U` = 0.337（`burgers_preshock_P2`），中位数 0.1463。
* Richardson 增益 > 1 的阶梯：9/9（区间 15.518 – 2121.644）。
* 被引擎**拒绝**（不可用作认证者）的阶梯：['advection_sine_P3', 'advection_gaussian_P3']。

## ④b 引擎行为发现：`p_max = 6.0` 会拒绝**超收敛**的物理阶梯

* 被拒的 2 条阶梯观测阶分别为 `advection_sine_P3` p=7.1240、`advection_gaussian_P3` p=6.9364，均超过 `P_MAX = 6.0`。
* 这不是数据退化，而是**上风 DG 单元均值在 L1 下的超收敛**：
  其解析解真误差为 7.171e-12 %、2.091e-09 %，比最细原解还小 2 个数量级，且带宽仍覆盖真误差。
* 后果（诚实的负面结论）：引擎对这类型阶梯返回 `WITHHELD_ORDER_TOO_HIGH`，
  即**把一把本来可用的尺判成不可用**。`p_max` 是策略上限而非物理上界；
  改它属于引擎语义变更，本单**未改**，只把现象与数字登记在此，供后继单决策。

## ⑤ 注意点 / 局限

* 这是**代码验证**（对闭式解的离散误差），不是实验 validation。
* 算例均为一维标量；二维系统的同一问题未测（见 ⑥ 建议）。
* `advection_sine` 完全光滑、p_obs 接近设计阶，属最有利情形；
  `advection_gaussian` 粗端欠解析、`burgers_nearbreak` 梯度陡，属不利情形——
  两者放进同一张表，正是为了让「带宽保守」这个结论**有反例可证伪**。
* `U` 仍是 Roache 三网格约定（Fs = 1.25），不是置信区间；三网格只给一个阶估计，无冗余。
* 认证解真误差的归一化是相对 L1（分母为解析解的 L1 范数），与 round-2 表同口径。
* **未测到的正是最要紧的区间（honest）**：上表全部阶梯仍属光滑/预破解区（p_obs ≈ 2–5），
  `p ~ 1` 的限幅/激波区**没有**认证结论。用方波对流 + TVB 限幅做的 raw-error 探针（`tools/certify_physical_band.py::probe_nonasymptotic`，同一条代码路径）给出 N=32: 34.01 % / N=64: 19.53 %，两点表观阶 ≈ 0.80——确实进入慢收敛区，
  但三级阶梯未在预算内跑完（load15 > 100 / 128 核，按容量纪律只做轻量活），
  故该区间的带宽覆盖率**仍然未知**，不得记为已验证。
* 粘性 Burgers 陡内层算例（ν=0.005，Cole–Hopf 解析解）单腿试跑被显式 RK4 的抛物步长限制拖到超出单命令预算，未纳入本表。
* 本表由 `python -m tools.certify_physical_band --light` 产出（去掉 256 单元阶梯与探针，负载纪律模式）；全量模式见脚本 `--help`。

## ⑥ 误差判定（回答「是否 < 3 %」）

* 认证解 vs 解析解的最差阶梯 = `burgers_nearbreak_P1`：0.0026 % → **< 3 %**。
* 认证解真误差 < 3 % 的阶梯：9/9。
* 带宽覆盖真误差：9/9（这是**带宽本身**的判据，与解的精度分开报：不用解的精度给带宽背书）。

