# CASE-DG-EULER-VORTEX-ERR：误差守恒对数值通量的敏感性（HLLC vs Rusanov）

> 锚点 `CASE-DG-EULER-VORTEX-ERR`（算例池·等熵涡移动误差守恒）
> 参考文献锚：Yee, Vinokur & Djomehri, J. Comput. Phys. 162 (2000) 33–81
> 本文回答一个此前未被测过的对抗性问题：**"误差守恒（α ≪ 1）"是 DG 离散/网格的属性，还是 HLLC 这一个 Riemann 通量的属性？**

## 1. 为什么这是一笔验证欠账

已发布的裁决（`case_euler_vortex_errconserve/CASE-DG-EULER-VORTEX-ERR-m22549-report.md` 第 ② 节）
把结果显式限定在「张量积模态 DG（P1–P3）+ **HLLC** + SSP-RK3」上。
门禁 C1–C12 全部用同一套通量跑出来，因此裁决本身无法区分下面两种解释：

* (a) 误差不随时间累积是 **DG 离散 + 网格** 的性质；
* (b) 该现象是 **HLLC 在光滑等熵流上异常准确** 的副产品。

判据 C1–C12 都没有覆盖「换一个更耗散的通量，结论是否还在」这条轴，属于典型的
「已合并功能缺一条独立复核轴」。

## 2. 做了什么

### 2.1 求解器能力（最小改动）

`tensordg/euler_dg_2d.py` 原先在 `_axis_fluxes` / `_boundary_flux` 里对 HLLC 写死。
新增（默认行为**逐位不变**，已用 HEAD 版本对拍验证 `max|Δcoeff| = 0.0`）：

| 新增 | 说明 |
|---|---|
| `INTERFACE_FLUXES = ("hllc", "rusanov")` | 可选通量登记表 |
| `_rusanov(ul, ur, axis)` | 局部 Lax–Friedrichs（Rusanov）通量，与 `_hllc` 的退化回退同式同序 |
| `_interface_flux(kind, ul, ur, axis)` | 派发；未知名字抛 `ValueError` |
| `flux=` 关键字 | 贯通 `solve_euler_dg_2d` → `_rhs` → `_axis_fluxes` → `_boundary_flux`；`metadata["flux"]` 记录实际使用值 |

### 2.2 测量链（复用算例既有仪表，不另起一套）

`benchmarks/case_euler_vortex_flux_sensitivity.py` 直接复用算例的算子、初值、
量测与增长估计量（`_operators` / `_initial_dg_state` / `_dg_measurements` /
`_fit_window` / `_exponent_theil_sen` / `_vortex_error`，参考解 = min-image 周期解析解），
因此两族之差只能归因于通量。

两种梯子：

* `--quick --quick-cells 8 --quick-times 1,2,5`：本机短预算梯子（可断点续跑），结果入 `quick_ladder.json`；
* `--checkpoint i` / `--aggregate`：**生产**设计族梯子 `dg_d2_n32_cfl0.2`，t = 1,2,5,10，8 个检查点，走计算调度队列。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 粗网格梯子（P2 / 8×8 / CFL 0.2 / β = 5，已实测）

| t | HLLC 最大场相对 L2 | Rusanov | Rusanov/HLLC | 步数 |
|---|---|---|---|---|
| 1 | 0.5619% | 0.5741% | 1.022× | 117 |
| 2 | 0.7457% | 0.7120% | 0.955× | 233 |
| 5 | 1.0869% | 1.0599% | 0.975× | 580 |

| 量 | HLLC | Rusanov | 判据 |
|---|---|---|---|
| α（最小二乘，t ≥ 1） | 0.4100 | 0.3838 | ≤ 1 → 都通过 |
| α（Theil–Sen，抗单点离群） | 0.4099 | 0.3810 | 与最小二乘一致（差 < 0.003） |
| e(5)/e(1) | 1.934 | 1.846 | ≤ 5 |

图：`case_euler_vortex_flux_sensitivity/fig_quick_flux_sensitivity.png`（误差-时间双对数 + 6% 容差线）。

### 3.2 生产梯子（P2 / 32×32，**未落地**）

作业 `cj4708`（8 分片，`stage_dir` = 本工作树）已递交，递交后 59 min 仍处 `queued`
（调度队列 ~154 作业在飞）。**本轮没有这批数字，不编造、不插值**：
`--aggregate` 对缺失检查点记入 `missing`，增长统计在不足 3 点时输出 `nan`（见脚本实现）。
生产梯子的 s̄ 读数留给跟进单：`python3 benchmarks/case_euler_vortex_flux_sensitivity.py --aggregate`。

## 4. 结论

1. **在这条已实测的粗网格梯子上，"误差守恒"对通量不敏感**：HLLC 与 Rusanov 的
   α 分别为 0.410 / 0.384，都远小于 1；误差量级差 ≤ 2.2%。裁决的解释 (a) 得到支持，
   而 (b) 未获支持（至少在这个分辨率上）。
2. **但"更耗散的通量误差更大"在粗网格上不成立**：n=16/P1/t=1 时 Rusanov 的
   *速度* 场误差比 HLLC 略小（7.9996e-3 vs 8.3730e-3，密度/压力略大）。
   P1 网格上误差由色散主导，不按数值耗散排序 —— 因此测试里只断言
   「误差不因换通量而改变一个量级以上」（比值 ∈ (0.5, 2)），不写更强的假命题。
3. 适用范围：本结论只在 P2 + 周期双边界 + 光滑等熵流上成立；激波/壁面/固体掩膜
   路径（`_solid_interface`、`bcs="wall"`）也会经过新派发，已由既有测试覆盖默认路径不变。

## 5. 新增测试（`tests/test_euler_dg_2d_interface_flux.py`，12 项全绿）

| 测试 | 断言的不变量 |
|---|---|
| `test_interface_flux_is_consistent[hllc/rusanov × axis0/1]` | 同态一致：`F(u,u) = F_physical(u)`，误差 0 |
| `test_rusanov_matches_hllc_degenerate_fallback` | 退化分支下 `_hllc` 回退与 `_rusanov` 逐值相同（同一numercis，不是两个） |
| `test_unknown_flux_is_rejected` | 未知通量名抛 `ValueError`（不静默回退） |
| `test_default_flux_is_bit_identical_to_explicit_hllc` | 默认路径与原行为 `torch.equal` |
| `test_metadata_records_the_requested_flux` | 元数据记录实际通量 |
| `test_periodic_vortex_conserves_each_flux[hllc/rusanov]` | 换通量不破坏离散守恒（漂移 < 1e-12） |
| `test_flux_choice_does_not_move_the_error_by_more_than_a_factor` | 粗网格通量排序非单调（见 §4.2） |

## 6. 复现

```bash
pytest tests/test_euler_dg_2d_interface_flux.py -q
python3 benchmarks/case_euler_vortex_flux_sensitivity.py --quick --quick-cells 8 --quick-times 1,2,5
python3 benchmarks/case_euler_vortex_flux_sensitivity.py --checkpoint <0..7>   # 生产梯子，逐分片
python3 benchmarks/case_euler_vortex_flux_sensitivity.py --aggregate
```
