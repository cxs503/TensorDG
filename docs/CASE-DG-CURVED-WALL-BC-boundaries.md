# 曲线 Euler DG 的壁面/远场边界：能力实现与诚实负结果（REPRO-DG-CKS1998）

> 锚点 `REPRO-DG-CKS1998`（Cockburn & Shu, JCP **141** (1998) 199--224, Part V）。
> 本单（m21389）做的是该锚点**最后一类缺失表示手段**：曲边壁面 + 远场边界。
> 一句话结论：**边界通量在“不推进时间”的层面全部精确可验（自由流保持 7.6e-15、
> 壁面质量/能量通量 1e-15、闭合环域质量/能量守恒 9e-16、几何面积 5e-5），
> 但显式 SSP-RK3 推进是线性不稳定的（第 197 步发散），因此
> CKS1998 的“圆柱绕流”算例**没有做成**——honest FAIL，不做任何绕流数值声明。**

## 1. 缺口（为什么这单值得做）

`tensordg/euler_curved.py` 的模块文档原本自认：

> “Not implemented (honest scope): reflective/characteristic curved walls (the
> `CurvedMesh.walls` flags are ignored -- only periodic edges are supported)”

而 `curved_geometry.annulus_mesh` 早已能造出圆柱极坐标环域网格，`CurvedMesh`
也已有 `walls` 标志——**网格有壁面语义，求解器却没有壁面边界**。JCP 141 的四个
算例里，双马赫反射（`docs/CASE-REPRO-DG-CKS1998-DOUBLE-MACH.md`）、前向台阶
（`docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md`）、二维 Riemann、光滑二维精度阶
都已落地，唯一没有表示手段的就是**曲壁物体绕流**。所以本单补的是边界条件。

## 2. 做了什么（最小改动，全部在 `tensordg/euler_curved.py`）

| 组件 | 内容 |
|---|---|
| `resolve_boundaries(mesh, boundaries)` | 四个边界面族 `(axis0_lo, axis0_hi, axis1_lo, axis1_hi)` 解析为 `periodic/wall/farfield/outflow`；`auto` 由网格语义推导（周期轴→周期，`walls` 标志→壁面，否则沿用历史的零梯度外插）；周期性与网格不一致、未知种类、缺远场态都**报错**而不是静默角落 |
| `wall_normal_flux(inner, n, gamma)` | 镜像 ghost（`u_n → -u_n`，`rho/p/u_t` 保持）+ 同一个 HLLC；对称 Riemann 扇给出 `S_M = 0`，于是壁面通量是**不可穿透且绝热**的 `(0, p n, 0)`（到舍入），同时继承内部面的 HLLC 耗散 |
| `_boundary_normal_flux` + `rhs` 覆盖 | 只有**真正的边界面单元**（`neighbours < 0`）被覆盖，内部面路径逐位不变；`boundaries=None` 时全部行为与历史一致（`tests/test_euler_curved.py` 原 9 条仍全绿） |
| `cylinder_annulus_mesh(nr, ntheta, r_in, r_out, degree, growth, inner_curve)` | 贴体极坐标 O 网格，**径向几何加密**（`growth>1`），`inner_curve(theta)` 可接 CAD 曲线；θ 周期、`walls=(True,False,False,False)` |
| `sample/primitive`、`wall_pressure`、`wall_force`、`steady_residual` | 诊断：体积/壁面积分点取值、壁面压强（含弧长权重）、流体对物体的压力合力（闭合面闭合性检验）、稳态残差 |

## 3. 结果（数字全部来自 `validation/curved_wall_bc/metrics.json`，程序化生成）

命令：`PYTHONPATH=. python3 benchmarks/curved_wall_bc_verification.py`

| 度量 | 值 |
|---|---|
| `W1_wall_fsp_operator` | 7.623e-15 |
| `W1n_flipped_wall_flux_operator` | 36 |
| `W2_wall_mass_flux` | 2.58e-16 |
| `W2_wall_energy_flux` | 1.032e-15 |
| `W2_wall_momentum_minus_pn` | 6.661e-16 |
| `W3_mass_drift_rel` | 9.397e-16 |
| `W3_energy_drift_rel` | 9.047e-16 |
| `W4_area_rel_nr4_nt8_g1` | 0.09968 |
| `W4_area_rel_nr8_nt16_g2` | 4.932e-05 |
| `W4_area_rel_nr12_nt24_g2` | 9.767e-06 |
| `W5_blew_up_at_step` | 197 |
| `W5_dt` | 0.004214 |

- **W1**：壁面有界槽道内均匀滑移流（`u=0.3` 平行于壁面）的半离散算子 `sup|dU/dt| = 7.6e-15`
  ——壁面通量的符号/朝向与 DG 面项一致。**W1n** 是门自身的空洞性对照：把壁面通量整体取反，
  同一算子变成 `36.0`（差 15 个量级），说明 W1 不是自动成立的。
- **W2**：随机可容许壁面态（法向速度严格为 0）上，质量/能量通量 `2.6e-16` / `1.0e-15`，
  动量通量与 `p n` 差 `6.7e-16`。
- **W3**：双壁闭合环域 20 步后总质量/总能量相对漂移 `9.4e-16` / `9.0e-16`。
- **W4**：离散环域面积 vs 解析 `π(r_o²-r_i²)`：`nr=8,nt=16,g=2` → `4.9e-5`；
  `nr=12,nt=24,g=2` → `9.8e-6`（`g=1` 直线边 10% 是**预期的几何欠解析**，不是缺陷）。
- **W5（诚实负结果）**：一个**精确稳态**（均匀滑移流）的显式推进中，舍入扰动按每 10 步
  ~5--6 倍放大，第 **197** 步超出 1e3。两种壁面通量形式（裸 `(0,pn)` 与镜像 ghost）都一样。

残差历史（`sup|dU/dt|`）：

| 步 | 残差 |
|---:|---:|
| 10 | 4.928e-14 |
| 20 | 2.351e-13 |
| 30 | 1.244e-12 |
| 40 | 9.244e-12 |
| 50 | 6.792e-11 |
| 60 | 5.003e-10 |
| 70 | 3.679e-09 |
| 80 | 2.691e-08 |
| 90 | 1.956e-07 |
| 100 | 1.411e-06 |
| 110 | 1.010e-05 |
| 120 | 7.181e-05 |
| 130 | 5.067e-04 |
| 140 | 3.551e-03 |
| 150 | 2.473e-02 |
| 160 | 1.714e-01 |
| 170 | 1.190e+00 |
| 180 | 8.618e+00 |
| 190 | 8.225e+01 |

## 4. 误差判定（逐项，显式回答“是否 < 3%”）

| 判据 | 实测 | 基准 | 误差 | < 3%？ |
|---|---:|---:|---:|---|
| 自由流保持（壁面 + 均匀滑移） | 7.6e-15 | 0（机器精度） | ~1e-15 | 是 |
| 壁面质量通量 | 2.6e-16 | 0 | ~1e-16 | 是 |
| 壁面能量通量 | 1.0e-15 | 0 | ~1e-15 | 是 |
| 闭合域质量守恒（20 步） | 9.4e-16 | 0 | ~1e-15 | 是 |
| 环域面积（g=2, 12×24） | — | 解析 π(r_o²-r_i²) | 9.8e-6 相对 | 是 |
| **绕流 Cp（CKS1998 §5 圆柱）** | **未取得** | Kármán--Tsien / 文献 | **无数字** | **不适用（honest FAIL）** |

最后一行是本单的核心：**壁面边界不稳定，绕流算例没有跑出可信解，因此不给 Cp 误差、
不给云图，也不把它记成能力**。

## 5. 注意点 / 局限

1. **稳定性是未解问题**，不是我把它调参藏起来了：W5 的失稳发生在**精确稳态**上
   （初值就是解），所以它只能是离散算子的不稳定模式，而不是流动本身不稳定。
   候选修法：壁面改用真正的两点通量、边界面上 BR1/Bassi--Rebay 型接口形式、
   或加熵稳定 SAT 罚项。已写进模块文档的“do not delete the honest test”提醒。
2. **远场类型只是 Dirichlet 型 ghost**，没有特征辐射条件，反射无界；即便壁面修好，
   圆柱算例也应做外半径 R 研究把阻塞误差量化（本单没有走到那一步）。
3. **`test_wall_boundary_march_is_unstable_honest` 故意断言“会炸”**：它是负结果的门，
   谁修好了稳定性它就会红，这正是触发删除文档警告的信号。
4. **不要用本模块声明 CKS1998 圆柱算例已完成**；锚点的这一项仍然是欠账。

## 6. 复现

```bash
PYTHONPATH=. python3 -m pytest tests/test_euler_curved_boundaries.py -q   # 13 条门
PYTHONPATH=. python3 benchmarks/curved_wall_bc_verification.py            # 写 metrics.json
```
