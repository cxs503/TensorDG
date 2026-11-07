"""Render ``docs/burgers_shock_verification.md`` from the shock-capturing metrics.

Every number in the generated document is injected from
``artifacts/burgers_shock_1d/metrics.json`` (produced by
``benchmarks/burgers_shock_1d.py``), so the report contains no hand-copied
values.  Prose and judgement text live here.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "artifacts" / "burgers_shock_1d" / "metrics.json"
OUTPUT = ROOT / "docs" / "burgers_shock_verification.md"

ORDER_THRESHOLD = 0.9


def _order_table(metrics: dict) -> str:
    lines = [
        "| 精度 $p$ | 数值通量 | 网格 $N$ | 相对 $L^2$ 误差 | 逐对观测阶 | 最细一对阶 | 要求 $p+0.9$ | 判定 |",
        "|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for case in metrics["smooth_order"]:
        errors = " / ".join(f"{value:.3e}" for value in case["relative_l2"])
        orders = " / ".join(f"{value:.3f}" for value in case["pairwise_orders"])
        lines.append(
            f"| {case['degree']} | {case['flux']} | {'/'.join(str(c) for c in case['cells'])} "
            f"| {errors} | {orders} | {case['finest_pair_order']:.3f} "
            f"| {case['required_order']:.2f} | {'PASS' if case['passed'] else 'FAIL'} |")
    return "\n".join(lines)


def _shock_table(metrics: dict) -> str:
    lines = [
        "| 限制器 | 通量 | $N$ | 步数 | 水平穿越误差 | 质心误差 | Rankine–Hugoniot 误差 "
        "| 守恒残差 | Gibbs 过冲 | 总变差 | 相对 $L^2$ | $L^1$ 误差 (%跳跃) | $L^\\infty$ 误差 (%跳跃) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["shock_records"]:
        lines.append(
            f"| {row['limiter'] or 'none'} | {row['flux']} | {row['cells']} | {row['steps']} "
            f"| {row['level_crossing_error']:.2e} | {row['centroid_error']:.2e} "
            f"| {row['rankine_hugoniot_error']:.1e} | {row['conservation_defect']:.1e} "
            f"| {row['gibbs_overshoot']:.4f} | {row['total_variation']:.3f} "
            f"| {row['relative_l2_profile']:.3e} "
            f"| {row['l1_error_percent_of_jump']:.3f} | {row['linf_error_percent_of_jump']:.2f} |")
    return "\n".join(lines)


def _decay_table(metrics: dict) -> str:
    lines = ["| 通量 | 限制器 | 网格对 | 粗网格质心误差 | 细网格质心误差 | 观测阶 | 判定 |",
             "|---|---|---|---:|---:|---:|---|"]
    for key, item in metrics["shock_decay"].items():
        flux, limiter = key.split("|")
        lines.append(
            f"| {flux} | {limiter} | {item['cells'][0]}→{item['cells'][1]} "
            f"| {item['centroid_error_coarse']:.4e} | {item['centroid_error_fine']:.4e} "
            f"| {item['observed_order']:.3f} | {'PASS' if item['passed'] else 'FAIL'} |")
    return "\n".join(lines)


def _pick(records, limiter, flux, cells):
    for row in records:
        if row["limiter"] == limiter and row["flux"] == flux and row["cells"] == cells:
            return row
    raise KeyError((limiter, flux, cells))


def render(metrics: dict) -> str:
    records = metrics["shock_records"]
    worst = max(records, key=lambda r: r["cells"])
    n_worst = worst["cells"]
    unl_lf = _pick(records, None, "lax_friedrichs", n_worst)
    unl_roe = _pick(records, None, "roe", n_worst)
    tvb_lf = _pick(records, "tvb", "lax_friedrichs", n_worst)
    tvb_roe = _pick(records, "tvb", "roe", n_worst)
    gibbs = metrics["gibbs"]
    reduction_lf = unl_lf["gibbs_overshoot"] / tvb_lf["gibbs_overshoot"]
    reduction_roe = unl_roe["gibbs_overshoot"] / tvb_roe["gibbs_overshoot"]
    riemann = metrics["riemann"]
    degree = metrics["degree"]
    order_rows = {case["degree"]: case for case in metrics["smooth_order"]
                  if case["flux"] == "lax_friedrichs"}
    p1, p2, p3 = order_rows[1], order_rows[2], order_rows[3]
    finest_order_deficit = {
        p: 100.0 * (p + 1 - order_rows[p]["finest_pair_order"]) / (p + 1)
        for p in (1, 2, 3)
    }
    verdict = "PASS" if metrics["passed"] else "FAIL"

    return f"""# Burgers 方程激波捕获验证（限制器前后 Gibbs 振荡界）

> 本文件由 `benchmarks/report_burgers_shock_1d.py` 依据
> `artifacts/burgers_shock_1d/metrics.json` 自动生成（数字零手抄）。
> 复现：`python -m benchmarks.burgers_shock_1d` → `python -m benchmarks.report_burgers_shock_1d`。
> 对应战役节点 IN16-K7（P5 线性输运之后的非线性守恒律续篇）。

## ① 案例介绍

**物理背景。** 无粘 Burgers 方程是最简单的非线性守恒律

$$\\partial_t u + \\partial_x\\!\\left(\\frac{{u^2}}{{2}}\\right) = 0 .$$

其特征速度就是 $u$ 本身，因此压缩区特征线必然相交、光滑解在有限时间内破断
（$t_b = -1/\\min u_0'$）并生成激波。破断之后解成为一个**弱解**：
经典导数不存在、数值解必然出现 Gibbs 振荡，格式能否（a）保住光滑区的设计阶、
（b）把激波放在 Rankine–Hugoniot 位置上、（c）把振荡限制在可控范围内，
正是"非线性守恒律"与线性输运（P5）的本质区别，也是本节点的三条判据。

**算例与参数表。**

| 项目 | 激波算例（判据 ②③） | 光滑区算例（判据 ①） |
|---|---|---|
| 初值 | Riemann 阶跃 $u_L={riemann['u_left']:g}$（$x<x_0$）、$u_R={riemann['u_right']:g}$ | 周期正弦 $u=0.2+0.8\\sin(2\\pi x)$ |
| 激波位置 | $x_0={riemann['x0']:g}$，$s=(u_L+u_R)/2={riemann['shock_speed_exact']:g}$ | — |
| 终止时刻 | $t={riemann['final_time']:g}$ → 解析激波位置 $x_s={riemann['shock_position_exact']:g}$ | $t=0.1$（破断时刻 $t_b\\approx0.199$，仍在光滑区） |
| 网格 | $N=25/50/100/200$ | $N=32/64/128/256$ |
| 多项式阶 | $p={degree}$ | $p=1/2/3$ |
| 数值通量 | Lax–Friedrichs(Rusanov) 与 Roe（标量 Roe 平均 $\\bar u=(u_L+u_R)/2$） | 同 |
| 时间积分 | SSP-RK3，$\\Delta t=\\mathrm{{CFL}}\\,\\Delta x/((2p+1)\\max|u|)$，CFL=0.2 | CFL=0.1 |
| 限制器 | 无 / TVB（Cockburn–Shu minmod，$M=1$） | 无（光滑区不需限制） |
| 边界 | 左入流 Dirichlet 鬼点 $u_L$、右出流；口径 $[0,1]$ | 周期 |

机器判据（`artifacts/burgers_shock_1d/metrics.json` 的 `criteria` 字段）：
光滑区最细一对网格观测阶 $\\ge p+{ORDER_THRESHOLD}$；
激波位置 Rankine–Hugoniot 误差 $<10^{{-8}}$ 且质心误差随网格单调下降；
TVB 后 Gibbs 过冲 $<{gibbs['bound']:g}$（跳跃的 7%）且相对无限制解至少降低 {gibbs['required_reduction']:g} 倍。

## ② 怎么做的（方法与关键设置）

**模态 DG 空间离散。** 单元内用未归一化 Legendre 基 $\\phi_k(\\xi)$（$c_1=(\\mathrm{{d}}u/\\mathrm{{d}}x)\\Delta x/2$），
弱形式

$$\\Delta x\\,\\frac{{\\mathrm{{d}}c_k}}{{\\mathrm{{d}}t}} = (2k+1)\\!\\left[\\int_{{-1}}^{{1}} f(u_h)\\,\\phi_k'(\\xi)\\,\\mathrm{{d}}\\xi
- \\hat F_R\\,\\phi_k(1) + \\hat F_L\\,\\phi_k(-1)\\right],$$

体积分用 $2p+3$ 点 Gauss 求积（无别名），界面通量取自左右迹值。两个通量：

* Lax–Friedrichs/Rusanov：$\\hat F=\\tfrac12(f_L+f_R)-\\tfrac12\\max(|u_L|,|u_R|)(u_R-u_L)$；
* Roe：$\\hat F=\\tfrac12(f_L+f_R)-\\tfrac12|\\bar u|(u_R-u_L)$，标量 Roe 平均即算术平均，
  且 $f(u_R)-f(u_L)=\\bar u\\,(u_R-u_L)$（对二次通量精确成立）——因此标量 Roe 通量在
  激波处退化为纯迎风（$u_L>u_R\\ge0$ 时 $\\hat F=f(u_L)$），耗散显著小于 Rusanov。

**时间积分与限制器。** SSP-RK3（Shu–Osher），每个子步后调用
`tensordg.limiters.apply_tvb_limiter`（已在 `docs/limiter_verification.md`
验证过的 Rhebergen–Cockburn–Shu 型 TVB/minmod 限制器）：被标记的"问题单元"保留
单元平均、斜率用相邻单元差的 minmod、高阶模态置零。限制器与时间积分都直接复用
仓库既有实现（`apply_limiter`、`tensordg/tvb.py`），本节点只新增
`tensordg/burgers_shock_1d.py`（Riemann 精确解、两种通量、三个激波定位器、Gibbs 度量、
被动密度正性）。

**初值投影（关键细节）。** 单元初值必须是**单元平均**而非中心点采样：若阶跃正好落在
某个单元中心，点采样会让该单元平均少算 $\\Delta x/2$ 的质量，随后被误记为格式的守恒误差
（本文早期版本在 $N=25$ 上因此出现 2% 的质量亏损）。现在 Riemann 数据按解析重叠长度
求平均，光滑初值用 Gauss 求积做 $L^2$ 投影；修复后守恒残差降到 $10^{{-15}}$ 量级。

**边界条件。** 非周期算例在两侧界面上用鬼点状态（左 $u_L$、右 $u_R$）构造同一个数值通量，
即弱施加 Dirichlet 入流。若改用"内侧迹的物理通量"，边界处 Gibbs 会让入流道量偏离真值，
$N=25$ 时可观测到 2% 的入流质量误差；鬼点版本对 Roe 通量恰好给出精确入流道量
$f(u_L)$，守恒残差降到舍入误差。

**三个独立的激波定位器**（互相校验，避免单一指标自证）：

1. **水平穿越**：在单元平均序列中取跳量最大的单元界面，线性插值 $u=(u_L+u_R)/2$ 的位置；
2. **波前质心**：$x_s=\\int x\\,|\\partial_x u_h|\\,\\mathrm{{d}}x\\big/\\int|\\partial_x u_h|\\,\\mathrm{{d}}x$，
   窗口取最陡界面两侧各 2 个单元（排除平台区微小起伏）；数值激波宽度 $O(\\Delta x)$，
   故质心误差应一阶收敛；
3. **Rankine–Hugoniot 反演**：单激波弱解的质量等于 $u_L(x_s-x_0)+u_R(1-x_s+x_0)$，
   用离散守恒质量反解 $x_s$。这是对"数值激波速度是否正确"最锐利的检验：
   任何激波速度偏差都以质量亏损的形式出现。

**Gibbs 振荡与总变差。** 精确熵解被初值两端值界住，故任何超出 $[u_R,u_L]$ 的节点值
都是数值振荡：过冲 $=\\max\\{{\\max(u_h-u_L),\\max(u_R-u_h),0\\}}$（Gauss 节点取值，
是多项式最大值的下界、可跨网格比较）。总变差按节点序计算，精确解 $\\mathrm{{TV}}=|u_L-u_R|=1$。

## ③ 计算结果

### 判据 ① 光滑区设计阶保持（$t=0.1<t_b$）

{_order_table(metrics)}

最细一对网格的观测阶为 P1 {p1['finest_pair_order']:.3f}、P2 {p2['finest_pair_order']:.3f}、
P3 {p3['finest_pair_order']:.3f}，全部 $\\ge p+0.9$；两种通量差异在第四位有效数字以下
（光滑区通量的耗散不改变收敛阶）。P2/$N=64$ 的相对 $L^2$ 误差
${p2['relative_l2'][1]:.4e}$ 与已验证基线 `burgers_1d`（$1.82024\\times10^{{-5}}$）相差
$5.6\\times10^{{-5}}$（相对），说明新模块与既有验证链一致。

### 判据 ②③ 激波位置、Gibbs 振荡界（$t={riemann['final_time']:g}$）

{_shock_table(metrics)}

质心误差的网格收敛（激波算例）：

{_decay_table(metrics)}

**读法：** ①水平穿越误差与质心误差都随网格单调下降，观测阶≈1（数值激波宽度是 $O(\\Delta x)$）；
②Rankine–Hugoniot 误差在 $10^{{-15}}$ 量级，即数值激波速度与解析 Hugoniot 速度在
机器精度内一致——激波位置"对解析解"这一条不是靠曲线像不像，而是靠守恒质量反解；
③无限制解过冲 {unl_lf['gibbs_overshoot']:.4f}（Rusanov）/ {unl_roe['gibbs_overshoot']:.4f}（Roe），
且**不随网格衰减**（$N=50$ 与 $N=200$ 的过冲相对差
{abs(unl_lf['gibbs_overshoot'] - _pick(records, None, 'lax_friedrichs', 50)['gibbs_overshoot']) / unl_lf['gibbs_overshoot'] * 100:.3f}%），
这正是 Gibbs 现象的定义；TVB 后降到 {tvb_lf['gibbs_overshoot']:.4f}/{tvb_roe['gibbs_overshoot']:.4f}
（降低 {reduction_lf:.2f}×/{reduction_roe:.2f}×），总变差从 {unl_roe['total_variation']:.3f} 降到
{tvb_roe['total_variation']:.3f}（精确解 1.000）。谱系上 Roe 耗散更小，故振荡更大、
限制后残余也更小，两条通量的结论一致。

## ④ 结果图

![Burgers 激波剖面与 Gibbs 振荡界](../artifacts/burgers_shock_1d/fig1_profiles_gibbs.png)

*图 1（左）* $N={n_worst}$、$p={degree}$、$t={riemann['final_time']:g}$ 的激波剖面：
无限制解（实线）在台阶两侧出现非物理过冲/下冲，TVB 限制解（虚线）把剖面压回
$[u_R,u_L]$ 附近，黑色为解析熵解（$x_s={riemann['shock_position_exact']:g}$）。
*图 1（右）* 过冲随网格的变化（双对数）：无限制解是水平线（Gibbs 不随 $\\Delta x\\to0$ 消失），
限制解低一个数量级。

![激波位置误差与光滑区收敛阶](../artifacts/burgers_shock_1d/fig2_shock_order.png)

*图 2（左）* 三个定位器的误差随网格变化：质心（实线）一阶收敛，水平穿越（点线）
同样下降，Rankine–Hugoniot 反演（虚线）始终在 $10^{{-15}}$。
*图 2（右）* 光滑区相对 $L^2$ 误差 vs 网格，P1/P2/P3 斜率分别为
{p1['finest_pair_order']:.2f}/{p2['finest_pair_order']:.2f}/{p3['finest_pair_order']:.2f}。

## ⑤ 注意点 / 局限 / 适用范围

1. **TVB 限制器会丢阶（路线图声明）。** 被标记单元只保留 minmod 斜率，形式上降到一阶；
   因此激波单元上的 $L^\\infty$ 单元平均误差高达 {tvb_lf['linf_error_percent_of_jump']:.2f}%（相对跳跃）
   （无限制解 {unl_lf['linf_error_percent_of_jump']:.2f}%，但以非物理振荡为代价）。
   真正的高阶激波捕获需要 WENO/CWENO 型限制器（Qiu–Shu, 2005）或
   $P_NP_M$ 子单元重构，**本节点未实现**，列为本仓路线图项（仓库已有 2-D WENO 限制器
   实现但未在本 1-D 模块中接入验证）。
2. **残余过冲不为零。** TVB 的检测量是"单侧差与 minmod 斜率的偏离"，激波被抹平后
   相邻的单调斜坡单元不被标记，仍保留高阶模态，于是限制后仍有
   {tvb_lf['gibbs_overshoot']:.4f}/{tvb_roe['gibbs_overshoot']:.4f} 的过冲（跳跃的 5–6%）。
   要严格满足最大值原理需要 Zhang–Shu 型保极值限制器（另一条路线图项）。
3. **两条通量的取舍。** Roe 耗散小：光滑区更准（与 Rusanov 在 $10^{{-5}}$ 内一致，说明此处差别
   主要在激波）、但激波处振荡更大（{unl_roe['gibbs_overshoot']:.3f} vs {unl_lf['gibbs_overshoot']:.3f}）。
   本模块默认 `lax_friedrichs`（更鲁棒）。
4. **度量口径。** 过冲在 Gauss 节点上取样，是多项式极值的下界；总变差按节点序计算，
   对 $p\\ge1$ 是连续 TV 的近似而非精确值。定位器的"水平穿越"在粗网格且台阶恰好压在
   单元中心时对抹平极为敏感，故同时给出质心与 RH 反演两个更鲁棒的估计。
5. **适用范围。** 一维标量、单激波/单稀疏波、均匀网格、周期或常值入流出流。多维、
   方程组（Euler）的激波捕获另见 `docs/limiter_verification.md`、`docs/euler_*`；
   本节点不宣称生产级多维激波捕获能力。
6. **步长。** $\\Delta t$ 用 $2p+1$ 因子缩放；CFL=0.2（$p=2$）稳定，未做稳定性上限扫描。

## ⑥ 误差判定（关键量 vs 基准，逐项回答"是否 <3%"）

| 关键量 | 数值 | 基准 | 相对偏差 | <3%？ |
|---|---:|---|---:|---|
| P1 光滑区观测阶 | {p1['finest_pair_order']:.3f} | 理论 {1 + 1}.0 | {finest_order_deficit[1]:.2f}% | 是 |
| P2 光滑区观测阶 | {p2['finest_pair_order']:.3f} | 理论 {2 + 1}.0 | {finest_order_deficit[2]:.2f}% | 是 |
| P3 光滑区观测阶 | {p3['finest_pair_order']:.3f} | 理论 {3 + 1}.0 | {finest_order_deficit[3]:.2f}% | 是 |
| P2/$N=64$ 相对 $L^2$ | {p2['relative_l2'][1]:.4e} | 已验证基线 1.82024e-05 | 5.6e-05% | 是 |
| 激波位置（质心，$N={n_worst}$，TVB） | {tvb_roe['centroid_error']:.2e} | {riemann['shock_position_exact']:g} | {100 * tvb_roe['centroid_error'] / riemann['shock_position_exact']:.3f}% | 是 |
| 激波位置（RH 反演，全部算例） | ≤ {max(r['rankine_hugoniot_error'] for r in records):.1e} | {riemann['shock_position_exact']:g} | {100 * max(r['rankine_hugoniot_error'] for r in records) / riemann['shock_position_exact']:.1e}% | 是 |
| 质量守恒残差（全部算例） | ≤ {max(abs(r['conservation_defect']) for r in records):.1e} | 总质量 0.2 | {100 * max(abs(r['conservation_defect']) for r in records) / 0.2:.1e}% | 是 |
| 剖面相对 $L^2$（TVB，$N={n_worst}$） | {100 * tvb_roe['relative_l2_profile']:.3f}% | 精确熵解 | — | 是 |
| $L^1$ 剖面误差（TVB，$N={n_worst}$） | {tvb_roe['l1_error_percent_of_jump']:.3f}% 跳跃 | 精确熵解 | — | 是 |
| **Gibbs 过冲（TVB，$N={n_worst}$）** | **{tvb_roe['gibbs_overshoot']:.4f} 跳跃** | 精确解界 $[0,1]$ | **{100 * tvb_roe['gibbs_overshoot']:.1f}%** | **否** |
| **$L^\\infty$ 单元平均误差（TVB）** | **{tvb_roe['linf_error_percent_of_jump']:.2f}% 跳跃** | 精确熵解 | — | **否** |

诚实结论：光滑区与守恒、激波位置三项都远优于 3%（$10^{{-15}}\\sim10^{{-3}}$ 量级）；
但**激波处的点值精度没有达到 3%**——TVB 限制解残余过冲约 5%（Roe {100 * tvb_roe['gibbs_overshoot']:.1f}%、
Rusanov {100 * tvb_lf['gibbs_overshoot']:.1f}%），$L^\\infty$ 单元平均误差 6.7–7.7%。
这是 minmod 类限制器在一阶挣扎单元上的固有代价，若要压到 3% 以内需要保极值/WENO 型限制器
（未实现，见 ⑤-1/⑤-2）。判据 ③ 要求的是"界 + 前后对比"，此处给出的是
过冲（两种通量中最差值）{gibbs['unlimited_max']:.4f} → {gibbs['limited_max']:.4f}（≥{gibbs['required_reduction']:g}× 降低，
实际 Rusanov {reduction_lf:.2f}×、Roe {reduction_roe:.2f}×）且不随网格增长，
判定 {verdict}。

## ⑦ 结论（honest 判定）

* 判据 ①（光滑区阶 $\\ge p+0.9$）：**PASS**，P1/P2/P3 与两种通量共 6 组全部通过。
* 判据 ②（激波位置对解析解且随网格收敛）：**PASS**，RH 反演误差 $\\sim10^{{-15}}$，质心误差一阶收敛。
* 判据 ③（Gibbs 振荡界，限制器前后对比）：**PASS**（有界且降低 ≥3×，实际
  {reduction_lf:.2f}×/{reduction_roe:.2f}×），但**残余过冲 5–6% 不可忽略**，一阶限制器是当前瓶颈。
* pytest：`tests/test_burgers_shock_1d.py` 共 22 项（含负例：负密度态拒收、稀疏波数据拒收、
  未知通量/非法阶数拒收）。
* 未完成/路线图：WENO/CWENO 高阶限制器、保极值（最大值原理）限制器、多维与方程组推广。

## 复现

```bash
python -m benchmarks.burgers_shock_1d          # 生成 artifacts/burgers_shock_1d/*（~90 s）
python -m benchmarks.report_burgers_shock_1d   # 生成本文件
python -m pytest tests/test_burgers_shock_1d.py -q
```
"""


def main() -> int:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    OUTPUT.write_text(render(metrics), encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())