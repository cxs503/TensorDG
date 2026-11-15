# 声明设计阶数解除 `WITHHELD_ORDER_TOO_HIGH` 误拒：带宽覆盖率实测

> 生成器：`python -m tools.certify_design_order_ceiling`（本文件全部数字由 `artifacts/design_order_ceiling/results.json` 程序化注入，零手抄）。
> 引擎：`tensordg.certified_reference`（import，未重写）。> 门限 τ = 3 %。> 超收敛窗半宽 = 0.5 阶。

## ① 问题（为什么这是一条要修的欠账）

round-2 认证引擎用 `P_MAX = 6.0` 作为观测阶的**绝对**上限。该值是策略常数，不是物理上限：DG 单元均值在光滑解上以 `O(h^{2k+1})` 超收敛（Cockburn–Luskin–Shu–Süli 2003），故 P3 阶梯**必然**给出 `p ≈ 7 > 6`，被引擎按构造误拒。上一轮（`docs/CERTIFIED-REFERENCE-BAND-PHYSICAL.md` ④b）只能把该负面结论登记在案。
本单的修法：调用方声明 `design_order=k` 时，绝对上限被**双侧**窗口 `2k+1 ± slack` 取代（`order_ceiling`）；不声明则逐位保持原行为，所有已发布数字仍可比。

本单要证伪的命题：**打开窗口是否以带宽覆盖率为代价**。阶数被噪声夸大时 `U ~ 1/(r^p - 1)` 会被压得过小，因此「阶梯被接受了」本身不值钱——必须用闭式解测出的认证解真误差仍在带宽之内。

## ② 做法

* 算例表与物理真解完全复用 `tools/certify_physical_band.py`（同一批 1-D 周期算例、同一生产内核 `tensordg.advection_1d` / `tensordg.burgers_1d`、同一 Gauss 求积真解），本单不另造算例；
* 每条阶梯认证**两次**：`shipped` = 不声明（原行为），`declared` = `design_order=degree`；
* 认证解真误差 = Richardson 外推解在参考网格（次细）上对闭式解单元均值的相对 L1；`真误差/U ≤ 1` 即带宽覆盖；
* 判定：`newly_usable` 非空且 `regressed` 为空且覆盖率不降 → 窗口可用，否则退回。

## ③ 结果（逐阶梯）

| 阶梯 | k | p_obs | 2k+1 | 窗口上界 | shipped 判定 | declared 判定 | U % | 认证解真误差 % | 真误差/U | 覆盖 |
|---|---|---|---|---|---|---|---|---|---|---|
| `advection_sine_P1` | 1 | 2.988 | 3 | 3.5 | `CERTIFIED` | `CERTIFIED` | 0.01029 | 2.478e-04 | 0.02407 | yes |
| `advection_sine_P2` | 2 | 4.989 | 5 | 5.5 | `CERTIFIED` | `CERTIFIED` | 1.046e-06 | 1.523e-08 | 0.01457 | yes |
| `advection_sine_P3` | 3 | 7.124 | 7 | 7.5 | `WITHHELD_ORDER_TOO_HIGH` | `CERTIFIED` | 4.903e-11 | 7.171e-12 | 0.1463 | yes |
| `advection_gaussian_P2` | 2 | 4.887 | 5 | 5.5 | `CERTIFIED` | `CERTIFIED` | 7.423e-05 | 5.232e-06 | 0.07048 | yes |
| `advection_gaussian_P3` | 3 | 6.936 | 7 | 7.5 | `WITHHELD_ORDER_TOO_HIGH` | `CERTIFIED` | 4.451e-08 | 2.091e-09 | 0.04697 | yes |
| `burgers_preshock_P1` | 1 | 2.565 | 3 | 3.5 | `CERTIFIED` | `CERTIFIED` | 0.001533 | 3.156e-04 | 0.2059 | yes |
| `burgers_preshock_P2` | 2 | 3.804 | 5 | 5.5 | `CERTIFIED` | `CERTIFIED` | 9.737e-06 | 3.278e-06 | 0.3366 | yes |
| `burgers_nearbreak_P1` | 1 | 2.049 | 3 | 3.5 | `CERTIFIED` | `CERTIFIED` | 0.0133 | 0.002623 | 0.1972 | yes |
| `burgers_nearbreak_P2` | 2 | 2.723 | 5 | 5.5 | `CERTIFIED` | `CERTIFIED` | 5.816e-04 | 1.930e-04 | 0.3319 | yes |
| `burgers_preshock_P2_fine` | 2 | 4.291 | 5 | 5.5 | `CERTIFIED` | `CERTIFIED` | 3.780e-07 | 1.910e-07 | 0.5053 | yes |

## ④ 汇总判定

* 阶梯总数 10；shipped 判为可用（`gate_usable`）的 **8** 条，declared 判为可用的 **10** 条；
* 新解除误拒的阶梯：`advection_sine_P3`, `advection_gaussian_P3`；
* 反向回归（原本可用、声明后变不可用）：无；
* 带宽可测 10 条，其中带宽覆盖真误差 **10** 条，覆盖率 100 %；
* 最差 `真误差/U` = 0.5053（≤ 1 即覆盖）。

## ⑤ 注意点 / 局限

* `P_MAX` 本身**未改**：不声明 `design_order` 的调用行为逐位不变，既有已发布数字（Shu–Osher 4 级梯次、DG 家族自证拒绝）不受影响；
* 窗口是**双侧**的：`p` 仅需「大于 6」不够，必须落在 `2k+1 ± slack`；测试 `tests/test_certified_reference_design_order.py` 用 p≈6.2 与 p≈7.6 两例锁住；
* 阶梯整体必须严格收缩（所有局部阶 > 0），单靠最细三点的偶然高阶不能换取认证；
* 门有效性判据 `U/τ ≤ 1/3` 仍然最后说话：窗口接受的阶梯若带宽吃掉门，仍判 `WITHHELD_BAND_CONSUMES_GATE`；
* 全部算例为 1-D；`2k+1` 是 1-D 单元均值的超收敛率，多维的对应率不同，声明多维设计阶前需另测；
* 这是**代码验证**（对闭式解的离散误差），不是实验 validation。

## ⑥ 误差判定（是否 < 3%）

* 关键量 1——认证解真误差 vs 闭式解：最大 0.002623 %（门限 3 %）→ **< 3% 成立**；
* 关键量 2——带宽自洽：最差 `真误差/U` = 0.5053 ≤ 1 → 带宽对全部阶梯覆盖真误差，**不因打开窗口而失效**；
* 关键量 3——引擎改动对既有发布数字的影响：无声明路径逐位不变（`tests/test_certified_reference.py` 的 Shu–Osher 逐位等价断言仍绿）。
