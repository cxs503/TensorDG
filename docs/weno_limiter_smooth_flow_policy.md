# WENO 限器的光滑流裁决：平台是「策略」不是 WENO 接线缺陷（m14921）

> 类型：**验证欠账清算 + 被否定假设**（`tensordg/euler_dg_2d.py::apply_weno_limiter_2d`）
> 上游欠账：`docs/anisotropic_2d_verification.md` §7 的 8 条 honest FAIL
> （`limiter_{tvb,weno}_{anisotropic,isotropic_control}_P{1,2}_order`）；
> `docs/tvb_constant_adjudication.md` §5.4 明确遗留「未覆盖 `limiter="weno"` 的同类裁决」；
> `docs/weno_limiter_2d_mixed_mode_defect.md` §5.2 遗留 `not_run` 的「强制限制表示的收敛阶」。
> 工件：`validation/weno_limiter_policy/results.json`（生成于 2026-11-11T12:49:01Z）、
> `validation/weno_limiter_policy/tables.md`、
> `docs/figures/weno_limiter_policy/sweep.png`、`docs/figures/weno_limiter_policy/forced_order.png`。
> 复现：`PYTHONPATH=. python -m benchmarks.weno_limiter_smooth_vortex` 然后
> `PYTHONPATH=. python -m benchmarks.report_weno_limiter_policy`；
> 测试：`PYTHONPATH=. python -m pytest tests/test_weno_limiter_policy.py`。

## 1. 案例介绍

被裁决的是下面这条判据族：

> `limiter_*_*_order`：`worst EOC with the {tvb|weno} limiter active on the smooth vortex
> -- the limiter must not destroy the smooth order`（阈值 P1 ≥ 1.5，P2 ≥ 2.5）。

它们在 `docs/anisotropic_2d_verification.md` 里以 0.06–0.46（P1）/ −0.61–0.16（P2）FAIL，
即「打开限制器后光滑涡不收敛」。TVB 半边的裁决（`docs/tvb_constant_adjudication.md`）
已证明：判据本身量纲一致，`M` 是**数据相关**的单元平均二阶导绝对上界，故默认 `M=1`
在光滑涡上**合法地**标记单元。WENO 半边此前没做。

物理设置＝标准二维等熵涡（2-D Euler 的精确解，周期域上刚性平移，无制造源项、无数值参照）：

| 参数 | 值 |
|---|---|
| 域 | `[[-5.0, 5.0], [-5.0, 5.0]]`，周期 |
| 涡强度 / 背景 | 5.0 / `(rho,u,v,p) = (1,1,1,1)` |
| 终了时间 / CFL | 1.0 / 0.2 |
| 网格（各向同性） | [[8, 8], [12, 12], [16, 16]] |
| 网格（2:1 拉伸） | [[8, 16], [12, 24], [16, 32]] |
| `M` 扫描 | [1.0, 2.0, 10.0, 1000000.0] |
| 强制限制网格（初始投影） | [[8, 8], [16, 16], [32, 32], [64, 64]] |
| 次数 | P1、P2（时间推进）；P1–P3（强制限制） |

## 2. 怎么做的

三块互不依赖的证据，全部对**解析解**取相对 L2 误差（Gauss 节点、原始量 rho/u/v/p）：

* **sweep**：`limiter="weno"`、`M ∈ [1.0, 2.0, 10.0, 1000000.0]`、P1/P2，在等向与 2:1 拉伸两梯上
  与**同一网格的无限制基线**对比，记录误差比、被标记单元比例、与基线是否**逐位相同**。
  可伪证预测：一旦 `M` 超过数据自身的二阶导上界，限器全程惰性 ⇒ 轨迹应与无限制**逐位相等**。
* **forced**：把精确涡初值 L2 投影到模态基，用 `M=0`（凡 Cockburn–Shu 偏差非零的单元必标）
  施加**一次** WENO 限制，直接测「被限制表示本身」的 L2 误差与收敛阶。线性重构（只在被标
  单元保 mode≤1）只能给二阶，所以 EOC→2 就是**上限**——把「限器破坏设计阶」变成可量化陈述。
* **cross**：同一 `M`、同一网格上 TVB 与 WENO 的误差与标记集对照；两个限器共用同一个
  单元探测判据，故共享的误差平台属于**策略**性质。

## 3. 计算结果（全部由 `results.json` 程序化注入）

### 3.1 限器代价 vs `M`（rho 相对 L2 误差 / 无限制误差）

| ladder | deg | cells | M | rho rel-L2 | rho x unr. | worst field x unr. | flagged (x/y/either) | fraction | bitwise |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| isotropic | 1 | 8x8 | 1 | 0.0147 | 1.04 | 1.04 | 0/0/0 | 0.000 | no |
| isotropic | 1 | 12x12 | 1 | 0.0179 | 2.6 | 2.71 | 4/2/5 | 0.035 | no |
| isotropic | 1 | 16x16 | 1 | 0.0116 | 3.04 | 3.05 | 6/4/9 | 0.035 | no |
| isotropic | 1 | 8x8 | 2 | 0.0141 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 12x12 | 2 | 0.0080 | 1.16 | 1.17 | 0/0/0 | 0.000 | no |
| isotropic | 1 | 16x16 | 2 | 0.0077 | 2.02 | 2.02 | 1/3/4 | 0.016 | no |
| isotropic | 1 | 8x8 | 10 | 0.0141 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 12x12 | 10 | 0.0069 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 16x16 | 10 | 0.0038 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 8x8 | 1e+06 | 0.0141 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 12x12 | 1e+06 | 0.0069 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 16x16 | 1e+06 | 0.0038 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 8x8 | 1 | 0.0216 | 6.61 | 7.3 | 0/0/0 | 0.000 | no |
| isotropic | 2 | 12x12 | 1 | 0.0198 | 21.6 | 23.1 | 4/2/5 | 0.035 | no |
| isotropic | 2 | 16x16 | 1 | 0.0125 | 31.5 | 31.5 | 7/5/10 | 0.039 | no |
| isotropic | 2 | 8x8 | 2 | 0.0033 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 12x12 | 2 | 0.0073 | 7.98 | 9.07 | 0/0/0 | 0.000 | no |
| isotropic | 2 | 16x16 | 2 | 0.0077 | 19.3 | 19.3 | 2/3/5 | 0.020 | no |
| isotropic | 2 | 8x8 | 10 | 0.0033 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 12x12 | 10 | 9.1389e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 16x16 | 10 | 3.9756e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 8x8 | 1e+06 | 0.0033 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 12x12 | 1e+06 | 9.1389e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 16x16 | 1e+06 | 3.9756e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 8x16 | 1 | 0.0125 | 1.18 | 1.21 | 0/2/2 | 0.016 | no |
| anisotropic | 1 | 12x24 | 1 | 0.0142 | 2.77 | 2.77 | 8/7/13 | 0.045 | no |
| anisotropic | 1 | 16x32 | 1 | 0.0090 | 3.19 | 3.19 | 11/13/21 | 0.041 | no |
| anisotropic | 1 | 8x16 | 2 | 0.0107 | 1.01 | 1.05 | 0/1/1 | 0.008 | no |
| anisotropic | 1 | 12x24 | 2 | 0.0077 | 1.5 | 1.5 | 2/4/6 | 0.021 | no |
| anisotropic | 1 | 16x32 | 2 | 0.0053 | 1.87 | 1.87 | 4/4/8 | 0.016 | no |
| anisotropic | 1 | 8x16 | 10 | 0.0105 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 12x24 | 10 | 0.0051 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 16x32 | 10 | 0.0028 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 8x16 | 1e+06 | 0.0105 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 12x24 | 1e+06 | 0.0051 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 16x32 | 1e+06 | 0.0028 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 8x16 | 1 | 0.0124 | 5.2 | 5.2 | 0/2/2 | 0.016 | no |
| anisotropic | 2 | 12x24 | 1 | 0.0150 | 22.7 | 22.9 | 8/7/13 | 0.045 | no |
| anisotropic | 2 | 16x32 | 1 | 0.0092 | 32.3 | 36.4 | 11/14/22 | 0.043 | no |
| anisotropic | 2 | 8x16 | 2 | 0.0070 | 2.93 | 2.93 | 0/1/1 | 0.008 | no |
| anisotropic | 2 | 12x24 | 2 | 0.0081 | 12.2 | 12.2 | 3/3/6 | 0.021 | no |
| anisotropic | 2 | 16x32 | 2 | 0.0049 | 17.2 | 17.2 | 3/3/6 | 0.012 | no |
| anisotropic | 2 | 8x16 | 10 | 0.0024 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 12x24 | 10 | 6.6156e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 16x32 | 10 | 2.8501e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 8x16 | 1e+06 | 0.0024 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 12x24 | 1e+06 | 6.6156e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 16x32 | 1e+06 | 2.8501e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |


### 3.2 收敛阶 vs `M`

| ladder | deg | M | design order | worst EOC (fields) | worst error ratio | bitwise all |
|---|---:|---:|---:|---|---:|---|
| isotropic | 1 | 1 | 2 | 0.294 | 3.05 | no |
| isotropic | 1 | 2 | 2 | 0.915 | 2.02 | no |
| isotropic | 1 | 10 | 2 | 1.877 | 1 | yes |
| isotropic | 1 | 1e+06 | 2 | 1.877 | 1 | yes |
| isotropic | 2 | 1 | 3 | 0.361 | 31.5 | no |
| isotropic | 2 | 2 | 3 | -1.567 | 19.3 | no |
| isotropic | 2 | 10 | 3 | 2.738 | 1 | yes |
| isotropic | 2 | 1e+06 | 3 | 2.738 | 1 | yes |
| anisotropic | 1 | 1 | 2 | 0.416 | 3.19 | no |
| anisotropic | 1 | 2 | 2 | 0.994 | 1.87 | no |
| anisotropic | 1 | 10 | 2 | 1.891 | 1 | yes |
| anisotropic | 1 | 1e+06 | 2 | 1.891 | 1 | yes |
| anisotropic | 2 | 1 | 3 | -0.094 | 36.4 | no |
| anisotropic | 2 | 2 | 3 | -0.018 | 17.2 | no |
| anisotropic | 2 | 10 | 3 | 2.758 | 1 | yes |
| anisotropic | 2 | 1e+06 | 3 | 2.758 | 1 | yes |


### 3.3 强制限制（`M=0`）表示的收敛阶——补上 m14777 的 `not_run`

| deg | cells | unlimited rho | limited rho | ratio | flagged | mode-0 changed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 8x8 | 0.0051 | 0.0226 | 4.46 | 0.984 | 0 |
| 1 | 16x16 | 0.0022 | 0.0066 | 3.01 | 0.945 | 0 |
| 1 | 32x32 | 5.5595e-04 | 0.0016 | 2.87 | 0.942 | 0 |
| 1 | 64x64 | 1.3955e-04 | 3.4651e-04 | 2.48 | 0.924 | 0 |
| 2 | 8x8 | 0.0025 | 0.0228 | 9.1 | 0.984 | 0 |
| 2 | 16x16 | 2.3906e-04 | 0.0065 | 27.3 | 0.945 | 0 |
| 2 | 32x32 | 3.0743e-05 | 0.0016 | 51.2 | 0.942 | 0 |
| 2 | 64x64 | 3.8615e-06 | 3.4141e-04 | 88.4 | 0.924 | 0 |
| 3 | 8x8 | 8.9697e-05 | 0.0228 | 254 | 0.984 | 0 |
| 3 | 16x16 | 2.4182e-05 | 0.0065 | 270 | 0.945 | 0 |
| 3 | 32x32 | 1.5019e-06 | 0.0016 | 1.05e+03 | 0.942 | 0 |
| 3 | 64x64 | 9.4422e-08 | 3.4141e-04 | 3.62e+03 | 0.924 | 0 |


| deg | EOC rho | EOC u | EOC v | EOC p | worst limited/unlimited on finest |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.013 | 2.080 | 1.984 | 2.039 | 2.49 |
| 2 | 2.024 | 2.089 | 1.983 | 2.049 | 102 |
| 3 | 2.023 | 2.089 | 1.983 | 2.049 | 3.62e+03 |


**关键数字**：强制限制表示的最差 EOC = 1.983（期望 2.0，偏差
0.86 %）；P2 的 rho EOC = 2.024
（无限制投影 rho EOC = 3.098）。即：**被限制表示最多二阶**。

### 3.4 TVB vs WENO 同时刻对照

| cells | M | limiter | rho rel-L2 | flagged either |
|---|---:|---|---:|---:|
| 8x8 | 1 | tvb | 0.0177 | 0 |
| 8x8 | 1 | weno | 0.0216 | 0 |
| 16x16 | 1 | tvb | 0.0105 | 8 |
| 16x16 | 1 | weno | 0.0125 | 10 |
| 8x8 | 10 | tvb | 0.0033 | 0 |
| 8x8 | 10 | weno | 0.0033 | 0 |
| 16x16 | 10 | tvb | 3.9756e-04 | 0 |
| 16x16 | 10 | weno | 3.9756e-04 | 0 |


### 3.5 判据

| criterion | value | verdict |
|---|---|---|
| `inert_M_bitwise_matches_unlimited` | True | PASS |
| `flagged_fraction_bounded_at_M1` | 0.04513888888888889 | PASS |
| `flagged_count_monotone_in_M` | True | PASS |
| `design_order_recovered_when_M_above_data_bound` | isotropic-p1-M10: 1.884, isotropic-p2-M10: 3.049, anisotropic-p1-M10: 1.891, anisotropic-p2-M10: 3.067 | PASS |
| `forced_limiter_representation_order` | p1: {'density': 2.013, 'velocity_x': 2.08, 'velocity_y': 1.984, 'pressure': 2.039}, p2: {'density': 2.024, 'velocity_x': 2.089, 'velocity_y': 1.983, 'pressure': 2.049}, p3: {'density': 2.023, 'velocity_x': 2.089, 'velocity_y': 1.983, 'pressure': 2.049} | PASS |
| `forced_limiter_preserves_cell_average` | 0 | PASS |
| `forced_limiter_flags_all_cells` | 0.923583984375 | FAIL |


## 4. 结果图

* `docs/figures/weno_limiter_policy/sweep.png`：左＝误差比 vs `M`（对数横轴），
  右＝被标单元比例 vs `M`。所有曲线在 `M` 越过数据上界后落到 1（误差比）/ 0（标记），
  即限器惰性、轨迹与无限制重合。
* `docs/figures/weno_limiter_policy/forced_order.png`：强制限制（`M=0`）表示与无限制投影的 rho 误差 vs `h`；
  被限制表示的斜率贴着 `h²`，无限制投影贴着 `h^(p+1)`。

## 5. 注意点（局限/适用范围）

1. **标记统计取终了态**：`flagged_*` 由终了时刻的模态系数算出，因此「终了态 0 标记但误差比 ≠ 1」
   是正常的（演化中途出现过标记，例如 `isotropic p1 8x8 M=1`）。
2. **`M` 是数据相关量**：本单给出「惰性发生在 `M ≥ 1e+06`」这一充分条件的
   实证（12 个 run 全部逐位相同），但没有给出「M 的普适下界」——它不是网格无关常数
   （见 `docs/tvb_constant_adjudication.md`）。
3. **未做**：P3 的时间推进扫描（只在强制限制块里覆盖 P1–P3）；`limiter_stage="final"` 分支；
   强拉伸（aspect > 4）网格上的 `M` 扫描。
4. **`M=0` 并不标记全部单元（诚实 FAIL，不粉饰）**：判据 `forced_limiter_flags_all_cells` 判 **FAIL**，
   最小被标比例 0.924。原因在判据算术本身：`minmod(d_left, d_right)` 在单调段当
   `|d_left| <= |d_right|` 时**恒等于 `d_left`**，于是偏差 `|d_left - minmod|` 精确为 0
   （实测 16² 上 x 轴 181/256、y 轴 184/256 被标）。这是 `_troubled_1d` 的非对称语义，
   不是缺陷；被标比例 ≥ 0.94 已足以触发「被限制表示至多二阶」的结论。
5. **判据层面结论**：`limiter_*_*_order` 这一族判据对 P≥2 **不可满足**——只要被标单元占 O(1)
   比例，被限制表示就只能给 O(h²)，而判据要求 EOC ≥ 2.5。这是判据设计问题，不是求解器缺陷；
   本单不修改该判据的阈值（属上游文档的验收语义），只把它降级为「策略测量」并与
   `docs/anisotropic_2d_verification.md` §7 的 honest FAIL 并存。

## 6. 误差判定（逐项显式回答）

| key quantity | reference | deviation | < 3 %? |
|---|---|---|---|
| forced-limiter EOC (worst field) | 2.0 | 0.86 % | yes |
| inert-M trajectory vs unlimited | bit-identical | 0 (12/12 runs) | yes |
| cell-average conservation under the limiter | exact | 0 changed components | yes |
| limiter-on cost at default M=1 | 1.0 (would be 0 % if inert) | 32.3x = +3125 % | no (this is the measured policy cost, not a code error) |


**判决**：WENO 限器在光滑等熵涡上的非收敛平台 = **策略/标记**性质（`M=1` 合法标记 + 被限制
表示只有二阶），WENO 接线无缺陷（`M` 足够大时轨迹与无限制**逐位相同**）；m14777 遗留的
`not_run`（强制限制表示的收敛阶）已补齐为 EOC ≈ 2。
