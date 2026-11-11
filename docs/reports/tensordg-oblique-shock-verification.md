# 二维 DG Euler 斜激波精确验证 & PR #247「假基准」审计（m13761）

> 仓：TensorDG；分支 `agent/m13761`；上游 `main@5f9b865`。
> 全部数字由 `benchmarks/oblique_shock.py` 程序化写入 `docs/benchmarks/results/oblique_shock/results.json`，
> 本文表格由该 JSON 生成（零手抄）。测试：`tests/test_oblique_shock.py`、`tests/test_euler_verification_chain.py`。

## ① 案例介绍（物理背景 / 几何 / 参数）

**被审计对象**：PR #247（mission m11149）合入 main 的「Sod → 双 Mach 反射 → 超声速楔形绕流」验证链
（`benchmarks/supersonic_wedge.py`、`benchmarks/double_mach_reflection.py`、
`docs/dg_euler_verification_report.md`、`tests/test_euler_verification_chain.py`）。
该报告声称「楔形绕流激波角误差 <5°」「双 Mach 反射密度 1.4~22+ 与 Woodward–Colella 一致」。

**审计发现（复现，非推断）**：

1. `solve_supersonic_wedge(...)` 对**任何**参数都抛 `RuntimeError`：`_initial_wedge` 里
   `torch.full_like(y, V_INF)` 与 `torch.full_like(x, ...)` 形状不同（`[1,ny,4,1]` vs `[nx,1,4,1]`），
   `_conservative` 的 `torch.stack` 失败。`tests/test_euler_verification_chain.py::TestSupersonicWedge` 因此在 main 上 **FAIL**。
2. `solve_double_mach(...)` 同样 FAIL：`torch.where` 的条件是 `torch.float64` 而非 `bool`
   （`RuntimeError: where expected condition to be a boolean tensor`）。
3. 即使修掉崩溃，`_initial_wedge` 返回的是**均匀来流**（`rho=1, u=2, v=0, p=1` 全场），
   没有楔面、没有激波——「测得的激波角」无从谈起（旧 `_compute_shock_angle` 的固定阈值 1.5
   高于 M=2/15° 的精确下游密度 1.3457，连一个穿越点都找不到，`angle_error_deg` 静默为 `None`）。
4. 双 Mach 初始状态**不是 Euler 方程的解**：M=10/60° 斜激波的下游速度必须沿激波法向
   （幅值 8.25 m/s），旧代码写成沿 +x 的 (8.25, 0)；且其 ρ₂=8、p₂=116.5 来自*正*激波关系，
   对 60° 斜激波的法向马赫数不成立（本单算得法向 M_n=10 时二者一致，前提是速度方向正确）。
5. 真正的阻塞在**边界条件**：`solve_euler_dg_2d` 只提供 periodic / 零梯度出流，
   没有斜滑移壁（楔面）与运动后激波边界（双 Mach 的 y=1 跟踪边界），
   所以「壁面楔形绕流」与「Woodward–Colella 三相点图形」在当前求解器上**物理上不可达**，
   现有报告的数字没有工件支撑（无 JSON、无运行 ID）。

**本单交付**：不粉饰上述结论，改为提供当前求解器**能**做且**有解析锚**的验证：斜激波精确解。

| 项 | 设置 |
|---|---|
| 方程 | 二维可压缩 Euler，γ=1.4 |
| 精确解 | 上游均匀流 + 平面激波（与 x 轴夹角 β）+ Rankine–Hugoniot 下游态；两态均匀 ⇒ **定常精确解** |
| 锚 | θ–β–M 关系（Anderson《Modern Compressible Flow》3/e 第 4 章）+ RH 四条跳跃条件 |
| 数值 | `solve_euler_dg_2d`：模态 Legendre DG、HLLC、SSP-RK3、TVB 限制器，零梯度出流边界 |
| 网格 | 24²–40²，p=1，CFL 0.1、TVB 常数 1.0（强限制） |
| 测量窗 | 距边界 0.12、距精确激波面 4h 以外（出流面截断激波，必须排除） |

## ② 怎么做的（方法 / 流程 / 关键设置）

1. `benchmarks/oblique_shock.py`：`beta_from_theta`（θ–β–M 弱解分支二分）、`oblique_shock`（RH 跳跃，
   自检四条 RH 残差 + 由两个速度矢量独立反算的偏转角）、`oblique_shock_initial`（两态初值）、
   `measure_oblique_shock`（从**数值场本身**反演激波角：沿精确法向做双线性采样、取密度中值穿越、
   对穿越点做总体最小二乘直线拟合；并量测平台比与保持误差）。
2. 判据**先声明后计算**（`GATES`，未事后调参）：β 误差 ≤ 1.0°、
   ρ 比与 p 比误差 ≤ 3%、偏转角误差 ≤ 0.5°、
   保持误差 ≤ 5%。
3. 三个对照：网格对齐正激波（β=90°）、负对照（同两态放在偏离 RH 角 5° 的激波面上）、
   M=10/β=60°（Woodward–Colella 类强激波）。
4. 修 `benchmarks/double_mach_reflection.py`：布尔 mask；用 `oblique_shock` 在激波静止系构造精确下游态
   再 Galilei 变换回实验室系，得到 ρ₂=8、p₂=116.5、速度幅值 8.25 沿法向（与 Woodward–Colella 一致），
   并在文件头写明缺失的反射壁/运动边界。
5. 修 `benchmarks/supersonic_wedge.py`：改形状 bug；初值改为精确斜激波（M=2、θ=15°）；
   阈值改为 RH 中值（旧固定阈值 1.5 永远找不到穿越）。
6. 测试：新增 `tests/test_oblique_shock.py`（理论自洽 + RH 残差 + 正激波极限 + 一个 DG 反演门）；
   `tests/test_euler_verification_chain.py` 的两个「只测不崩」用例改为**对照解析值**断言，
   双 Mach 全量推进标记 `xfail` 并注明这是诚信缺口。

## ③ 计算结果（全部来自 results.json）

| case | M | β_exact(°) | β̂(°) | Δβ(°) | ρ₂/ρ₁ exact | ρ₂/ρ₁ hat | err | p err | Δθ(°) | 保持误差 |
|---|---|---|---|---|---|---|---|---|---|---|
| wedge_M2_theta15 | 2.0 | 45.344 | 45.647 | 0.303 | 1.7289 | 1.7289 | 0.00% | 0.00% | 0.0002 | 2.31% |
| normal_M4_beta90 | 4.0 | 90.000 | 90.000 | 0.000 | 4.5714 | 4.5714 | 0.00% | 0.00% | 0.0000 | 2.03% |
| control_M2_theta15_wrong_beta | 2.0 | 45.344 | 40.408 | 0.065 | 1.7289 | 1.7289 | 0.00% | 0.00% | 0.0003 | 3.73% |

- 网格对齐正激波（control）：ρ 比 4.57138 vs 精确 4.57143（误差 1e-5），p 比 18.49974 vs 18.5 —— 测量管线本身精确。
- 楔形类（M=2、θ=15°、β=45.3436°）：40² 网格上 β̂ 误差 0.303°，ρ/p 比误差 <1e-4，Δθ=0.0002°，
  保持误差 2.31%。
- **M=10/β=60°**：`FloatingPointError: nonphysical cell average`——TVB=100 与 TVB=1/CFL=0.1 均如此，
  强激波工况在现有限制器设置下**不成立**（诚信 FAIL，未调参掩盖）。
- **负对照 FAIL**：把同一对状态放在偏离 5° 的激波面上，t=0.04 时保持误差仅 3.73%，未达 10% 的判别阈值
  —— 说明 `preserve_l1` 在短时下对*激波面几何*不敏感（平台比本来就由 RH 构造，天然与几何无关）。
  判别力来自正激波极限（1e-5）与楔形反演（0.30°），不来自该负对照。

## ④ 结果图

本单未生成位图：运行预算全部用于精确锚与两个已合并基准的修复；绘图通路已实装
（`oblique_shock._plot_case`：密度云图 + 精确激波面 + 法向剖面），可 `python -m benchmarks.oblique_shock`
一次生成全部 PNG。此处如实说明而非用占位图冒充。

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **边界条件是硬约束**：出流面会把激波截断，所有量测必须在内部窗口；壁面楔形绕流、双 Mach 反射图形
  需要斜滑移壁与运动边界，当前不可达——这是**真实缺口**，已登记为后续单线索。
- **测量管线的独立性有边界**：平台比与精确值来自同一套 RH 代数，只有**激波角**与**保持误差**是真正的
  数值反演量；因此判据以 β̂ 与正激波极限为主，平台比只作一致性检查。
- **限制器代价**：激波贯穿全域时几乎所有单元被标记为 troubled cell，TVB 路径使单步代价约 3 倍于光滑算例
  （32² 网格 80 步 ≈ 24 s），这是本单只能用 24²–40² 的直接原因。
- 适用范围：定常平面激波（斜/正），M ≤ 4 在当前设置下稳定；M=10 不稳定。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 量 | 基准 | 实测误差 | <3%？ |
|---|---|---|---|
| 激波角（M=2、θ=15°、40²） | θ–β–M 精确 β=45.3436° | 0.303° ≈ 0.67% | **是** |
| 密度比（同上） | RH 1.72892 | <1e-4 | **是** |
| 压力比（同上） | RH 2.19465 | <1e-4 | **是** |
| 偏转角（同上） | 15° | 0.0002° | **是** |
| 正激波密度比（M=4） | RH 4.57143 | 1e-5 | **是** |
| 正激波压力比（M=4） | RH 18.5 | 1e-5 | **是** |
| 保持误差（M=2 斜激波） | 0（定常精确解） | 2.31% | **是** |
| M=10 强激波 | RH 5.625 / 87.3 | 未取得（发散） | **否（诚信 FAIL）** |
| 负对照判别 | 应 >10% | 3.73% | **否（判据不敏感，已披露）** |

**结论**：斜激波角度与跳跃量的验证面成立（<3%，多数量在 1e-4 量级），但
(a) 壁面楔形绕流与 Woodward–Colella 双 Mach 反射仍**未验证**——PR #247 的相关声明应予撤回，
(b) M=10 强激波工况不稳定，(c) 负对照判据不敏感——三项均如实入账，不做粉饰。
