# 一维 SWE 静水重构（Audusse）良平衡模式：实现与验证（tp-paper-2 / m7559）

> 仓：TensorDG ｜ 分支：`agent/m7559` ｜ 基线：`main 75f6216`（2026-11-08）
> 前序：`docs/thacker_validation_gap_corrected.md`（m6377）、
> `docs/tensordg-swe-bed-slope-source-fix.md`（m6504）。
> m6504 在 §6 明确点名本单要补的能力：**Audusse 静水重构（hydrostatic
> reconstruction）**——它把「曲率床面静水精确平衡」与「曲率床面高精度源项」
> 这对矛盾统一起来；本单实现该能力，并把 m6504 遗留的两处欠账一并处理。

---

## 1. 结论摘要（先给判定）

| 项 | 本单结果 |
|---|---|
| 新增能力 | `solve_swe_1d(..., well_balanced=True)`：界面 Audusse 静水重构 + 以**床面多项式导数**（同 Gauss 点）离散底坡源项 |
| 湖静平衡（曲率床 `0.4 sin(πx/2)`，N=64，t=1 s，P1/P2） | 离散解**逐系数不动**：max|Δc|=**1.83e-12**（P1）/ **9.63e-11**（P2）；伪动量 **7.68e-12** / **3.97e-10** |
| 同一算例默认模式（未开 WB） | 伪动量 **1.361e-3**（P1）/ **3.289**（P2，即 m6504 记录的 0.645 m 漂移的动量指纹） |
| P2 Thacker 精度 | WB 模式与 m6504 的 `zb_slope` 模式**逐步一致**（steps 442/883 相同，u 相对误差 0.0000%），因抛物床面在 p=2 空间内**可精确表示**，静水重构退化为恒等 |
| P1 Thacker 精度 | WB 模式**变差**（η 14.9985%，N=40 触发 `Nonphysical SWE state`）——P1 无法表示二次床面，源项重建误差回到 O(dx)。诚实结论：**P1 验证仍应用 `zb_slope`**（0.1582%/0.0395%） |
| 附带缺陷修复 | SSP-RK2 的 stage-2 残差**丢掉了 `time` 与解析边界通量回调**（静默回退透射闭合）→ Thacker 解析边界模式下体积守恒 O(dt) 破缺；修后相对误差 3.163e-4 → **5.629e-7**（同一算例 562×） |
| 回归 | `tests/test_swe_wellbalanced.py`+`test_swe_1d.py`+`test_swe_bed_slope_consistency.py`：**42 passed, 1 xfailed**；新增 `tests/test_swe_wellbalanced_hydrostatic.py`：**7 passed** |

**关键量是否 < 3%？** 湖静平衡（关键量=漂移）**是**（机器精度）；Thacker P2/P1(`zb_slope`) η 误差 0.1582%/0.0395% **是**；
P1 的 WB 模式 η 误差 15.0% **不是**——如实列出，不粉饰（见 §5 适用范围）。

---

## 2. 为什么要静水重构（问题结构）

一维 SWE（守恒变量 `U=(h,hu)`，通量 `F=(hu, hu²+gh²/2)`，源项 `S=(0,-g h ∂_x z_b)`）的 DG 弱形式残差为

```
R_k = (2k+1)/dx [ ∫_K F φ'_k dξ - F*_{i+1/2} φ_k(1) + F*_{i-1/2} φ_k(-1) ] + S_k .
```

静水 `h+z_b=const, hu=0` 要对每个模态都精确为零，需要两件事：

1. **界面压力通量**必须与单元自身迹上的压力 `g/2 h_m²` 相消。若两侧床面不同（分片多项式床面在界面处有跳变），普通 HLL 通量给出的是两侧压力的加权平均，相消失败 → 漂移 O(压力跳变)，且**随 p 增大而增大**（P2 的迹差异更大，故默认模式下 P2 比 P1 差）。
   → 修法：Audusse (2004) 静水重构，两迹都投到 `b*=max(b_L,b_R)`、`h*=max(0,h+b-b*)`，再把单侧修正 `g/2(h²-h*²)` 加回该项动量通量。静水时两侧重构深度相等 ⇒ HLL 退化为 `(0, g/2 h*²)`，加回修正即**恰好**等于 `g/2 h_m²`，与体积分边界项逐项相消。
2. **源项**必须用与 `h` 同一个离散表示下的 `∂_x z_b`。m6504 用**解析**坡度（Gauss 点采样）换取了非可表示床面上的高精度，但那与 `h` 的多项式导数不一致，静水平衡因此破缺（P1 sine 床漂移 3.2e-4 m → 0.27 m）。
   → 修法：WB 模式把床面投到**与 h 同阶的分片多项式** `b_h`，源项用它自己的导数 `∂_x b_h`（同 Gauss 点、同求积）。于是静水时 `h'=−b_h'` 在离散层**逐点恒成立**，残差恒为零（任意床形、任意 p），代价是床面力的精度受 `b_h` 表示能力限制。

两者合起来即 `well_balanced=True`：**精确良平衡**（任意床形/阶数）与**高精度**（床面可在 DG 空间内表示时）二者兼得；床面不可精确表示时二者只能择一（§5）。

## 3. 实现（最小改动面）

* `tensordg/swe_1d.py`
  - 新增 `_hydrostatic_interface_flux(coeff, phi_left, phi_right, bed_modal, flux_kind, periodic)`
    → 返回**单侧**通量对 `(F_{i,+}, F_{i+1,-})`；两端仍为零阶外推（透射）闭合，解析边界通量回调在其后覆盖。
  - `_rhs(..., bed_modal=None)`：`bed_modal` 非空时走静水重构通量 + 多项式坡度源项；为空时**逐行保持**原两条分支（legacy 常值坡度 / `zb_slope` 解析坡度），默认路径行为不变。
  - `solve_swe_1d(..., well_balanced=False)`：在 Gauss 点上把 `z_b` 投影成 `bed_modal`；`zb_slope` 与 `well_balanced` 互斥（`ValueError`）。
  - **缺陷修复**：SSP-RK2 的 stage-2 残差原为
    `_rhs(u1, ..., flux_kind, periodic, zb_slope_quad=zb_slope_quad)`，
    即 `time=0.0`、`bc_left_flux/bc_right_flux=None` —— 解析边界通量在第二阶段**静默失效**（回退透射闭合），RK3 分支无此问题。现改为 `(..., time + dt, bc_left_flux, bc_right_flux, zb_slope_quad, bed_modal)`。
* `tests/test_swe_wellbalanced_hydrostatic.py`（新增 7 项）：曲率床静水逐系数不动（p=1,2）、默认模式现状特性化（防止未来默认翻转静默发生）、线性床下 WB≡默认（1e-11）、互斥参数、解析边界通量在 RK2/RK3 两阶都真正生效（体积守恒 <1e-5 / <1e-10）。

## 4. 验证（数字全部程序化产出）

### 4.1 湖静平衡（曲率床 `z_b=0.4 sin(πx/2)`，域 [-2,2]，N=64，CFL 0.3，float64）

| 模式 | p | max\|hu\| | max\|Δc\|（t=1s − t=0） |
|---|---:|---:|---:|
| 默认（legacy 源项） | 1 | 1.361e-03 | — |
| `well_balanced` | 1 | **7.68e-12** | **1.83e-12** |
| 默认（legacy 源项） | 2 | 3.289e+00 | — |
| `well_balanced` | 2 | **3.97e-10** | **9.63e-11** |

即：曲率床静水从「P2 有 32% 水深的漂移」变为**离散不动点**（残差只剩 ~1e-16/步的舍入积累）。

### 4.2 Thacker (1981) 抛物盆地平面晃动（`z_b=−h₀(1−(x/L)²)`，h₀=1, A=0.5, L=1, t=0.5 s ≈ T/3，解析边界通量）

| 配置 | steps | η 相对 L2 | h 相对 L2 | u 相对 L2 |
|---|---:|---:|---:|---:|
| P1 `zb_slope` N=20 | 265 | 0.1582% | 0.0480% | 0.0134% |
| P1 `zb_slope` N=40 | 530 | 0.0395% | 0.0120% | 0.0021% |
| P1 `well_balanced` N=20 | 357 | 14.9985% | 9.0789% | 133.98% |
| P1 `well_balanced` N=40 | — | `FloatingPointError`（非物理态守卫） | — | — |
| P2 `zb_slope` N=20 / N=40 | 442 / 883 | 0.1581% / 0.0395% | 0.0478% / 0.0120% | 0.0000% / 0.0000% |
| P2 `well_balanced` N=20 / N=40 | 442 / 883 | 0.1581% / 0.0395% | 同上 | 0.0000% |

* P1 `zb_slope`：N=20→40 误差比 4.00 ⇒ **EOC≈2.0**（p=1 设计阶 2）。
* **P2 下 WB 与 `zb_slope` 逐步数、逐数字一致**：抛物床面是二次多项式，p=2 单元内可**精确表示**，静水重构退化为恒等（`b*=b_L=b_R`、`h*=h`、修正项=0）。这既是实现的正确性证据，也说明 WB 模式不是"另一个格式"，而是同一格式在可表示床面上的等价实现。
* P1 WB 变差的原因是纯表示误差（源项用 `b_h` 的一阶导数代替解析坡度），**不是**实现缺陷：与 m6504 记录的 legacy 分支同一失效模式（N=40 触发非物理态守卫）。

### 4.3 RK2 边界通量缺陷（修复前/后，同一算例：Thacker，N=20，P1，t=0.2，解析边界通量）

| 版本 | 体积相对误差 |
|---|---:|
| `main 75f6216`（缺陷版） | **−3.163e-04** |
| 本单（修复） | **−5.629e-07**（562×） |

RK3 分支同样用例：1e-13 量级（Simpson 型阶段平均，非缺陷）。残差 5.6e-7 来自 Heun 对边界通量积分的时间梯形，属阶数上限，已在测试注释中写明。

## 5. 注意点 / 适用范围（诚实边界）

1. **床面可表示性决定能否同时拿到良平衡与高精度**：`well_balanced` 的源项用 `b_h` 的导数，故对**分片可表示**床面（平底、线性、p≥2 的二次）既精确良平衡又达设计阶；对不可表示床面（P1+抛物、任意阶+正弦）只能精确良平衡、力精度退化到 O(dx)。要"两者都要"，正解是提高床面表示阶（p≥2）或做局部床面加密，而非混用解析坡度。
2. **默认行为未变**：`well_balanced=False` 时新代码走原分支；线性床用例证明 WB 与默认在 `b_L=b_R` 时残差一致到 1e-11。
3. **干湿处理只做了保守化**：重构用 `h*=max(0,·)`，速度用 `DRY_DEPTH` 守卫；本单**未**验证干床前沿（溃坝/爬滩）上的 WB 性质，属未覆盖面。
4. **2-D 未做**：`tensordg/swe_2d.py` 仍用原源项；横向界面重构（x/y 双向 `b*`）是下一步，本单不声称。
5. `thacker_planar_basin` 的返回值顺序是 `(h, u, η)`（首个返回是**水深**不是自由面）——本单在脚本里踩到该坑（一度把 h 当 η 比较，得出 41.8%/62.5% 的假误差），已在验证脚本中固定为 `h` 与 `η` 分别比较。使用该解析解时请留意。

## 6. 复现

```bash
PYTHONPATH=. python -m pytest tests/test_swe_wellbalanced_hydrostatic.py -q
PYTHONPATH=. python -m pytest tests/test_swe_wellbalanced.py tests/test_swe_1d.py \
    tests/test_swe_bed_slope_consistency.py -q      # 42 passed, 1 xfailed
# 湖静平衡 / Thacker 双模式对比数字（写本文件 §4 的程序化来源）
PYTHONPATH=. python validation/swe_thacker/run_hydrostatic_wb.py
```

## 7. 参考

* Audusse, Bouchut, Bristeau, Klein, Perthame (2004), *A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows*, SIAM J. Sci. Comput. 25(6).
* Xing, Shu (2006), *High order finite difference WENO schemes with the exact conservation property for the shallow water equations*, JCP 208; 及同作者 DG 良平衡系列。
* Thacker (1981), *Some exact solutions to the nonlinear shallow-water wave equations*, JFM 107.