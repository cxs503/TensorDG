# REPRO-DG-CKS1998 · 2-D 系统阶数：从「点估计」升级为「带不确定度的认证」

> **锚点**：`REPRO-DG-CKS1998`（Cockburn & Shu, *The Runge–Kutta discontinuous Galerkin
> method for conservation laws V: multidimensional systems*, JCP 141 (1998) 199–224；
> SIAM Review 43(1) 2001）
> **单号**：m21846（自找活·platform 席）　**分支**：`agent/m21846`
> **性质**：验证欠账闭合 —— 平台 L4 已 `VERIFIED` 的 `sm2d_order_p{1,2,3}` 三行，
> 此前只有**一个数**（观测阶）对 3 % 容差；本单补的是那个数**凭什么算通过**：
> 估计子离散、渐近区、外推一致性。**不改求解器、不改任何预登记容差、不重算数值。**

---

## ① 案例介绍（物理背景 / 几何 / 参数）

CKS1998 的 §5.1 二维光滑系统算例：均匀流携带的**熵波**

    rho(x, y, t) = 1 + 0.2 sin(pi (x + y - 2 t)),   u = v = 1,   p = 1,   gamma = 1.4

周期盒 `[0,2]²`。因为 `u, v, p` 均为常数，二维 Euler 的动量方程与能量方程都退化成同一条
线性对流方程，故上式是**精确解**（无制造源项）；`tests/test_cks1998_smooth_2d_order.py`
用自动微分把这条精确性钉在机器精度上。

论文对光滑解证明的收敛速率是 `p + 1`（`p` = 多项式次数）。平台把这条理论判据编码成
三行：`sm2d_order_p1/p2/p3`，ref = 2/3/4，容差 3 %（`validation/repro_cks1998/bench_spec.py`，
容差表在看结果之前冻结）。**本单的全部数值输入就是这三行背后的三级网格梯**，来自平台
L4 作业 stdout（mission 20922 / cluster cj3053，
`validation/repro_cks1998/artifacts/l4_2d/stdout.txt`，仓内已入库）。

| 组 | 网格 | `L1(rho_h - rho)`（平台 stdout 直读） |
|---|---|---|
| P1 | 16 / 24 / 32 | 6.111182e-03 / 2.736229e-03 / 1.542369e-03 |
| P2 | 12 / 16 / 24 | 4.284309e-04 / 1.794235e-04 / 5.282294e-05 |
| P3 | 8 / 12 / 16 | 6.758086e-05 / 1.325564e-05 / 4.216007e-06 |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

新增 `benchmarks/cks1998_order_certification.py`（纯分析 + 求解器驱动两用），把「通过」
拆成三条**预登记**判据（模块头写定，早于任何数字）：

| 量 | 定义 | 通过阈值 |
|---|---|---|
| δ | `|p_fine − (p+1)| / (p+1)`（最细网格对观测阶 vs 设计阶） | — |
| U | 估计子离散：`max |X − p_fine|`，`X ∈ {LSQ(全部层), LSQ(最细三层), 次细对阶}` | — |
| σ | 渐近区比 `p_次细对 / p_最细对` | 0.9 ≤ σ ≤ 1.1 |
| ρ_R | 外推残差 `|e_R| / e_fine`，`e_R = e_f + (e_f − e_c)/(r^p − 1)` | ρ_R ≤ 3 % |
| 判定 | **CERTIFIED** iff `δ + U/(p+1) ≤ 3 %` 且 σ 在带内 且 ρ_R ≤ 3 % | |

三值 verdict 里刻意保留 `UNVERIFIED_BAND`：点估计落在 3 % 内、但 `δ+U/(p+1)` 出界的梯
**不得**被称作「已验证」——这条正是本单与「只看一个数」的区别。

**为什么不用 GCI**：被外推的量是误差本身，对误差梯做 Roache GCI 恒为 O(1)，什么也没说。
换成外推残差 ρ_R（若阶估计正确，`e_R → 0`）才是有意义的自洽判据。这条判断写在模块头，
并在测试里用「错误阶 → 大残差」验证其非空洞。

**另加一条独立复核**：从平台 stdout 的 `(cells, l1_density)` 重新做最小二乘，与平台
`observed_order` 逐项对账（`platform_orders.reconciliation_rel`）。

工具链：`python -m benchmarks.cks1998_order_certification --from-platform-artifact`
（读平台 stdout → 认证 → `validation/cks1998_order_certification/certification.{json,md}`）→
`python -m gen_report_cks1998_order_certification --figure docs/figures/cks1998_order_certification.png`。
`benchmarks/cks1998_order_certification.py` 同时带集群分片驱动（`--config-index/--num-configs`）
用于更细网格梯与时空分离，见 §⑤-2。

## ③ 计算结果（程序化注入，零手抄）

| P^k | 设计阶 | 最细对观测阶 | 估计子离散 U | δ | δ+U/(k+1) | σ | 渐近区 | ρ_R | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| P1 | 2 | 1.9927 | 0.0109 | 0.37 % | **0.91 %** | 0.995 | 是 | 0.000 % | **CERTIFIED** |
| P2 | 3 | 3.0158 | 0.0097 | 0.53 % | **0.85 %** | 1.003 | 是 | 0.000 % | **CERTIFIED** |
| P3 | 4 | 3.9819 | 0.0354 | 0.45 % | **1.34 %** | 1.009 | 是 | 0.000 % | **CERTIFIED** |

平台存档阶 vs 本单独立重算（最小二乘）：相对差 5.6e-16 / 7.4e-16 / 2.0e-15 —— 平台的
`observed_order` 就是这条梯的最小二乘斜率，无口径差。

图：`docs/figures/cks1998_order_certification.png`（三阶的 `L1` 误差 vs `h` log-log，
叠设计阶与实测阶参考斜率，标题给 verdict/δ/U/σ）。

## ④ 结果图

见上引用 png（本单唯一图；三子图 = P1/P2/P3 误差梯 + 参考斜率 + 判定标注）。

## ⑤ 注意点 / 局限（honest）

1. **只认证了平台实际打分的那三级梯**（16/24/32、12/16/24、8/12/16）。三级梯上「估计子
   离散」只有两个自由度，U 必然偏乐观（p3 的 U = 0.0354 已经是最细对 3.9819 与 LSQ 4.0036
   之差）；四级以上的梯才能把 U 压到真正收敛的值。
2. **时空分离本轮 not_run**：更细网格梯 + 同一网格对在 `CFL=0.2` 与 `CFL=0.05` 上的重算
   （23 个配置）已作为集群作业 **cj3753（6 分片，stage_dir 递交）** 排入队列，本单收尾时
   仍在排队（队列 ~96 个在飞作业）。**本单不据未回作业下任何结论**，`assemble()` 里
   `time_separation.spatial_isolated` 因此缺席；`tests/test_cks1998_order_certification.py`
   用合成数据单独钉住了时空分离的算法（同一网格对、两档 CFL、η 与阶漂移阈值）。
3. **ρ_R = 0.000 % 是三级梯的巧合级现象**：三级、且相邻阶几乎相等的梯，用最细对阶外推
   会把残差压到 0 附近；它证明「阶能解释误差衰减」，但**不**证明网格已充分细。
4. **本单没有改任何容差、也没有改求解器**：`bench_spec.METRICS` 的 3 % 是 11-15 之前冻结
   的；本单只回答「这个 3 % 闸门在现有梯上算不算认证」。
5. **结论的适用范围**：CERTIFIED 是对**该算例 + 该网格梯**的认证，不外推到激波算例
   （dMR / forward-facing step 的定量判据仍缺，见
   `docs/CASE-REPRO-DG-CKS1998-DOUBLE-MACH.md` §5-1）。

## ⑥ 误差判定（显式回答「是否 < 3 %」）

| 关键量 | 基准 | 实测（判定档） | 相对偏差 δ | 是否 < 3 % |
|---|---|---|---|---|
| P1 观测阶 | 设计阶 2 | 1.9927 | 0.37 % | **是**（含不确定度 0.91 %） |
| P2 观测阶 | 设计阶 3 | 3.0158 | 0.53 % | **是**（含不确定度 0.85 %） |
| P3 观测阶 | 设计阶 4 | 3.9819 | 0.45 % | **是**（含不确定度 1.34 %） |
| 外推残差 ρ_R（三阶） | 0（阶解释误差衰减） | ≤ 0.000 % | — | **是** |

**逐项结论：3/3 度数的阶偏差 < 3 %，而且「偏差 + 估计子不确定度」也 < 3 %（最差 1.34 %）
—— 平台的 `sm2d_order_p*` 闸门在这条梯上是**认证**，不只是「点估计恰好压线」。**

## 复现（一条命令一段）

```bash
PYTHONPATH=. python -m benchmarks.cks1998_order_certification --from-platform-artifact
PYTHONPATH=. python -m gen_report_cks1998_order_certification \
    --figure docs/figures/cks1998_order_certification.png
PYTHONPATH=. python -m pytest tests/test_cks1998_order_certification.py -q   # 14 passed
```