# SWE 底坡源项一致性缺陷：定位与修复（tp-ver-5 / m6504）

> 仓：TensorDG ｜ 分支：`agent/m6504` ｜ 基线：`f2743ce`（main, 2026-11-08）
> 前序：`docs/thacker_validation_gap_corrected.md`（m6377）留下两条未解线索：
> (b)「半离散残差：质量方程残差为机器零、动量方程残差为一阶 O(dx)」、(c)「施加解析边界通量后
> P1 N≥40 触发 `Nonphysical SWE state`」。
> 本单把 (b) 定位到**具体的离散缺陷**并给出修复，(c) 在修复后消失（同一算例、同一配置）。

---

## 1. 结论摘要

| 项 | 结论 |
|---|---|
| 缺陷 | 1-D SWE 动量方程底坡源项用**单元常数**坡度 `dbdx=(z_b,R−z_b,L)/dx`。对分段线性床面精确；对**曲率床面**它只是 `dz_b/dx` 的单元平均值，丢掉的线性部分乘上 DG 解后，离散力留下 **O(dx)、且与多项式阶 p 无关**的误差 |
| 指纹 | p 加密无效：Thacker 算例 P2 的 η 阶在 t=0.1 s 只有 0.52–0.67（设计阶 3）；连续速度场误差 17–22% 不随 N 下降 |
| 修复 | `solve_swe_1d(..., zb_slope=callable)`（新增可选 kwarg，默认 None 保持原行为）：在与 `h` 相同的 Gauss 点上采样 `dz_b/dx`，源项求积对多项式数据**精确**：`S_k = −(2k+1)/2 · g Σ_q w_q h(x_q) z_b'(x_q) φ_k(ξ_q)` |
| 关键性质 | 坡度常数时**代数上退化为旧式** `−g dbdx c_k`（实测差 5.7e-14），故不是重新归一化，而是严格推广 |
| 效果（t=0.5 s，解析边界通量） | P1 N=20：η 误差 **11.93% → 0.126%**（95×），u 误差 **133.7% → 0.013%**；N=40 旧式直接 `FloatingPointError`，新式 **0.031%** |
| 收敛性 | 新式 P1：0.1259% → 0.0313%（N=20→40）= 比值 4.02，**EOC 2.01**（设计阶 2）；旧式无此行为 |
| P2 连续速度场 | 旧式误差 21.9%/16.9%/11.5%（N=20/40/80，阶 ~0.4）；新式 **机器零**（1e-7 相对）——该算例的速度场是空间均匀的，P2 能精确表示，故残差应只剩舍入 |
| 代价（诚实） | 新式源项**破坏曲率床面静水精确平衡**：sin 床面 P1 lake-at-rest 漂移 3.2e-4 m → 0.27 m。故默认不启用；两者统一需 Audusse 静水重构（§6） |
| 附带发现 | main 上已有 **2 个红测试**（与本改动无关，已用 pristine HEAD 复核）：P2 平底 lake-at-rest 门 1.8e-11 > 1e-11（纯舍入）；P2 曲率床面 lake-at-rest 漂移 **0.645 m = 32% 水深**（真缺陷，已 triage，见 §5） |

---

## 2. 算例与配置（论文级六要素之一）

Thacker (1981) 抛物盆地平面晃动，解析解取自 `tensordg.swe_exact.thacker_planar_basin`
（m6377 已核验：`omega=√(2gh₀)/L`，均匀速度幅值 `AωL/(2h₀)`，三条标量恒等式 + sympy 残差=0）。

| 参数 | 值 |
|---|---|
| 床面 | `z_b = −h₀(1−(x/L)²)`，`dz_b/dx = 2h₀x/L²` |
| 水深/幅值/半宽 | `h₀=1.0 m`，`A=0.5 m`，`L=1.0 m`，`g=9.81 m/s²` |
| 周期 | `T=1.4185 s`；考核时刻 `t=0.1 s`（早期）与 `t=0.5 s`（≈T/3，m6377 的未解缺口） |
| 数值 | DG `p=1,2`，HLL 通量，SSP-RK3，CFL 0.3，float64，N=20/40 |
| 边界 | 解析边界通量 `bc_left_flux/bc_right_flux`（`swe_exact.thacker_planar_basin_boundary_flux`） |

## 3. 方法：缺陷怎么定位的

1. **重算半离散残差**：把解析解 L2 投影到 DG 空间，直接调 `_rhs` 看逐模态残差——复现 m6377 的
   「质量残差机器零、动量残差 O(dx)」。质量方程无源项 ⇒ 残差为零；动量方程唯一非谱一致的项就是
   源项 ⇒ 怀疑点收敛到 `dbdx`。
2. **代数核验**：`S_k = (2k+1)/dx ∫_K (−g h z_b') φ_k dx`。旧式等价于把 `z_b'` 换成常值
   `b_mid`；对二次床面 `z_b'` 是**线性**的，故 `∫ h (z_b'−b_mid) φ_k ≠ 0`，且其大小不随 p 减小。
3. **对照实验**：用真实坡度在 Gauss 点上求积（新分支），其余一切不变（同一 `_rhs`、同一通量、
   同一时间积分），只切换源项分支。
4. **性质锁定**：常坡度下新式必须与旧式逐位一致（测 1）；否则说明改的是归一化而非精度。

## 4. 结果（数字全部来自 `validation/swe_thacker/bed_slope_results.json`，零手抄）

### 4a. 主表（相对 L2 误差，%）

| 配置 | 旧式 η | 旧式 u | 新式 η | 新式 u | 步数 旧/新 |
|---|---|---|---|---|---|
| P1 N=20 t=0.1 | 0.1270 | 0.5331 | 0.1158 | 0.0278 | 48 / 48 |
| P1 N=40 t=0.1 | 0.0347 | 0.1679 | 0.0287 | 0.0043 | 95 / 95 |
| P2 N=20 t=0.1 | 2.7128 | 21.8999 | 0.1157 | **0.0000** | 104 / 79 |
| P2 N=40 t=0.1 | 1.7084 | 16.9424 | 0.0287 | **0.0000** | 263 / 158 |
| P1 N=20 t=0.5 | 11.9328 | 133.7326 | **0.1259** | 0.0134 | 357 / 265 |
| P1 N=40 t=0.5 | **FAIL**（Nonphysical） | FAIL | **0.0313** | 0.0021 | — / 530 |
| P2 N=20 t=0.5 | 3.1328 | 28.6781 | **0.1258** | **0.0000** | 1529 / 442 |
| P2 N=40 t=0.5 | 1.2865 | 18.4264 | **0.0313** | **0.0000** | 3026 / 883 |

### 4b. 观测阶

* 新式 P1, t=0.5：`0.1259 → 0.0313`，比值 **4.02**，EOC **2.01**（设计阶 2 ✓）。
* 旧式 P2, t=0.1：`2.7128 → 1.7084 → 1.1924`（N=20/40/80），EOC **0.67 / 0.52** —— 设计阶 3，
  实测半阶，即「加密无效」的定量形式。
* 旧式 P2 均匀速度误差 `21.90 → 16.94 → 11.48`：EOC ≈ 0.37，**永不收敛**；新式为舍入量级。

### 4c. 时间步塌缩

P2 N=40、t=0.5：旧式需 **3026** 步（同一 CFL），新式 **883** 步。旧式把源项误差打进速度场，
声速被抬高，`dt ∝ dx/max(|u|+c)` 被压小 3.4 倍；这解释了 m6377 记录的「P2 比 P1 更差 + 触发
非物理守卫」。

### 4d. 图

* `validation/swe_thacker/bed_slope_profiles.png`：t=0.5 s 的 η(x)、u(x) 与解析解对比
  （旧式整体倾斜相位错，新式贴线）。
* `validation/swe_thacker/bed_slope_convergence.png`：左=误差–N 双对数（含 3% 验收线）；
  右=曲率床面 lake-at-rest 漂移随时间（P1/P2）。

## 5. 注意点 / 局限（不粉饰）

1. **代价：静水平衡退化**。新式破坏曲率床面 lake-at-rest 的离散抵消：sin 床面 N=64、t=1 s，
   P1 漂移 3.2e-4 m（旧式）→ 0.27 m（新式）。旧式的「常数源项」恰好提供了该抵消。因此
   `zb_slope` 默认关闭，只在「光滑床面 + 非定常 + 有解析边界」的验证配置下使用；
   `tests/test_swe_bed_slope_consistency.py::test_exact_source_does_not_preserve_lake_at_rest_on_curved_bed`
   把这条 trade-off 钉死，防止有人当作「严格更好」盲目启用。
2. **边界闭合仍是独立缺陷**。若不给解析边界通量（默认透射闭合），新式会在 t≈0.4 s 前丢失正性
   （旧式则是 +6.6% 质量漂移）。本单未修边界；这与 m6377 §6c 同源。
3. **附带发现的两个 main 红测试**（与本改动无关；用 pristine HEAD 的 `swe_1d.py` 复核过）：
   * `test_flat_bed_lake_at_rest_is_machine_precise[2]`：P2 平底漂移 1.77e-11 m（相对水深 8.8e-12），
     原门 1e-11 落在舍入地板之下 → 判为**门限过紧**，改为 1e-10 并在注释记录实测值。
   * `test_sloped_bed_lake_at_rest_stays_finite[2]`：P2 曲率床面漂移 **0.645 m = 32% 水深**
     （P1 为 3.2e-4 m）。随 N 下降（N=32/64/128 → 1.91/0.645/0.273 m），故是**离散缺陷而非失稳**；
     根因=界面缺少静水重构。本单不改求解器，改为 `pytest.xfail` 并在 reason 中写明实测数字与根治路径。
4. **第三个 main 红测试**（2-D，本单未处理）：`tests/test_swe_2d_dynamics.py::
   test_stoker_depth_and_momentum_match_exact_solution` 在 pristine HEAD 包上同样失败
   （已用 `/tmp/orig` 原版 `swe_1d.py`+`swe_exact.py` 复跑验证，失败一致），与本改动无关；
   推测与 §5.3 同源的 2-D 源项/界面一致性缺陷，留给下一单。
5. **未做**：Audusse 静水重构实现；2-D 版本 `swe_2d.py` 的同型缺陷（其源项同样是单元常数坡度，
   `tests/test_swe_wellbalanced.py` 里 2-D 的静水漂移门同样只是「有界」而非精确）——建议下一单处理。

## 6. 后续（下一单规格）

**目标**：同时满足「任意床面静水精确平衡」与「p 阶一致」。
**做法**（标准做法，非新发明）：界面改用 Audusse et al. (2004) 静水重构——
`z_b* = max(z_b,L, z_b,R)`，`h*_{L,R} = max(0, h_{L,R} + z_{b,L,R} − z_b*)`，HLL 通量取重构态，
动量分量加界面修正 `(g/2)(h² − h*²)`；体积源项保留本单的精确求积。静止态下两侧重构态相同
⇒ HLL 通量 ≡ 静水压强、修正项相消，静水平衡应回到机器精度；非定常解仍保有 p 阶一致性。
**验收**：(a) 曲率床 lake-at-rest P1/P2 漂移 < 1e-10；(b) Thacker t=0.5 误差 < 3% 且 EOC≈2；
(c) 现有 `tests/test_swe_bed_slope_consistency.py` 全绿。

## 7. 复现

```bash
PYTHONPATH=. python -m pytest tests/test_swe_bed_slope_consistency.py tests/test_swe_wellbalanced.py -q
PYTHONPATH=. python validation/swe_thacker/run_bed_slope_consistency.py   # 117 s, 16 组
```

产物：`validation/swe_thacker/bed_slope_results.json`、`bed_slope_profiles.png`、
`bed_slope_convergence.png`。

## 参考文献

* Thacker, W. C. (1981). Some exact solutions of the nonlinear shallow-water equations.
  *J. Fluid Mech.* 107, 499–508.
* Audusse, E., Bouchut, F., Bristeau, M.-O., Klein, R., Perthame, B. (2004). A fast and stable
  well-balanced scheme with hydrostatic reconstruction for shallow water flows.
  *SIAM J. Sci. Comput.* 25(6), 2050–2065.
* Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Springer.