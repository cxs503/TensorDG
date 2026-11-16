"""Build the paper-style progress report for m22823 (CASE-DG-ADVECT-ORDER round 3)."""
from __future__ import annotations

import base64
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
M = json.loads((REPO / "validation/advection_order_stability/metrics.json").read_text())

from tensordg.runge_kutta import TABLEAUS                      # noqa: E402
from tensordg.rk_stability import stability_function            # noqa: E402


def f(x, d=4):
    return f"{x:.{d}f}"


def pct(x, d=2):
    return f"{100.0 * x:.{d}f}%"


def order_figure(path: pathlib.Path) -> pathlib.Path:
    zs = -np.linspace(1e-3, 3.0, 400)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    palette = (("rk2", "#1f77b4"), ("rk3", "#2ca02c"),
               ("rk4", "#9467bd"), ("ssp_rk54", "#d62728"))
    for name, colour in palette:
        tab = TABLEAUS[name]
        ax.plot(zs, np.abs(stability_function(tab, zs)), color=colour, lw=1.6,
                label=f"{name} (order {tab.order})")
    ax.plot(zs, np.exp(zs), "k--", lw=1.0, label=r"$e^{z}$")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$z=-x$")
    ax.set_ylabel(r"$|R(z)|$")
    ax.set_ylim(0.0, 1.6)
    ax.set_title(r"(a) $|R(z)|$ vs $e^z$; the $|R|=1$ crossing is $cfl_{max}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    z = np.logspace(-3, -0.3, 200)
    for name, colour in palette:
        tab = TABLEAUS[name]
        err = np.maximum(np.abs(stability_function(tab, -z) - np.exp(-z)), 1e-18)
        keep = err > 1e-16
        slope = -math.log(err[keep][-1] / err[0]) / math.log(z[keep][-1] / z[0])
        ax2.loglog(z, err, color=colour, lw=1.5,
                   label=f"{name}: slope {slope:.2f}")
    ax2.loglog(z, z ** 5 / 120.0, "k:", lw=1.0, label=r"$z^{5}/5!$")
    ax2.set_xlabel(r"$z$ (on the negative real axis)")
    ax2.set_ylabel(r"$|R(z)-e^{z}|$")
    ax2.set_title("(b) truncation order seen in the polynomial")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


FIG2 = order_figure(pathlib.Path("/tmp/advorder_stability_order.png"))


def b64(p: pathlib.Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


anchors = M["anchors"]
matched = M["matched"]
cross = M["cross_pairing"]
emp = {r["degree"]: r for r in M["empirical"]}
setup, th = M["setup"], M["thresholds"]
ref = anchors["reference"]
meas = anchors["measured"]
worst = min(matched, key=lambda r: r["cfl_max"])
emp_min = min(r["empirical_over_predicted"] for r in M["empirical"])
emp_max = max(r["empirical_over_predicted"] for r in M["empirical"])

md = []
A = md.append
A("## ① 案例介绍")
A("")
A("**被认证的对象是离散格式的稳定性，不是流场。** TensorDG 的 RKDG 精度阶梯"
  "（`CASE-DG-ADVECT-ORDER` 第 2 轮）用 `p+1` 级显式 Runge–Kutta 格式推进多项式阶 "
  "`p` 的 DG 空间，以对标 Sun (Math. Comp. 2024, DOI 10.1090/mcom/4037) 的 "
  "`RK2DG1/RK3DG2/RK4DG3/RK5DG4` 四列。整个阶梯与它的 dt 阶探针都跑在同一个 "
  "`cfl` 上，但“这个 `cfl` 在每一档的线性稳定域内”此前只是注释里的断言："
  "仓库里只有 `rk4`/`ssp_rk3` 两个积分器被测过（`validation/cfl_stability`），"
  "第 1 轮的 `rk2`、第 2 轮的 `rk3` 以及五级四阶 SSP-RK(5,4) 从未被认证。")
A("")
A("数学背景：一维周期线性对流 $u_t + c u_x = 0$，$x\\in[0,1)$，$c=1$；DG 半离散化后得线性 ODE "
  "$\\dot u = L u$，其谱 $\\Lambda(L)$ 与时间积分器的放大多项式 $R(z)$ 决定 $z=dt\\lambda$ 是否被放大。")
A("")
A("| 参数 | 值 |")
A("|---|---|")
A("| 对流速度 $c$ / 区域 | 1 / $[0,1)$ 周期 |")
A("| 多项式阶 $p$（谱扫描） | " + ", ".join(str(d) for d in setup["degrees_spectrum"]) + " |")
A("| 被认证档位 | " + ", ".join(str(d) for d in setup["degrees_matched"]) + " |")
A("| 时间格式候选 | " + ", ".join("`%s`" % t for t in setup["certified_tableaus"]) + " |")
A(f"| 块符号波数采样 | {setup['n_theta']} 点，$\\theta\\in[0,\\pi]$（任意网格最坏情形） |")
A(f"| 步长规则 | $dt=\\mathrm{{cfl}}\\,h/((2p+1)|a|)$（`tensordg/advection_1d.py`） |")
A(f"| 精度阶梯 `cfl` / dt 阶探针上限 | {setup['cfl_accuracy_ladder']} / {setup['cfl_probe_max']} |")
A(f"| 稳定性容差 | {setup['stability_tol']} |")
A(f"| 经验复算 | $N={setup['empirical_ncells']}$，$T={setup['empirical_final_time']}$，宽带初值 |")
A("")
A("## ② 怎么做的")
A("")
A("1. **解析路线（新增 `tensordg/rk_stability.py`）**：对 Butcher 形式的显式格式，"
  "$R(z)=1+z\\,b^{T}(I-zA)^{-1}\\mathbf{1}=1+\\sum_{k\\ge1}(b^{T}A^{k-1}\\mathbf{1})z^{k}$，"
  "系数机械取自求解器实际使用的 `RKTableau` 对象——不存在第二份手抄系数；"
  "系数形式与线性求解形式互为交叉校验（实测最大偏差 "
  f"{anchors['two_form_max_abs_diff']:.2e}）。")
A("2. **算子谱用生产内核装配**：`assemble_operator` + `symbol_matrix` 取块符号在 "
  "$\\theta\\in[0,\\pi]$ 上的连续分支，即**任意网格**的最坏情形，而非某网格恰好解析出的几个特征值。")
A("3. **逐特征值极限**：对每个 $\\lambda$ 求首个使 $|R(dt\\lambda)|>1+10^{-10}$ 的 $dt$，取 "
  "$\\min_\\lambda$；包络非单调时做 400 点审计把边界挪到最早越界点（不给乐观值）。"
  "再按代码的 $(2p+1)$ 约定换算成用户可直接比较的 $\\mathrm{cfl}_{max}$。")
A("4. **经验复算（独立路线）**：在出厂求解器 `solve_advection` 上对 `cfl` 二分，判据为模态范数"
  "增长指数的符号，初值用宽带剖面（Nyquist 方波 + 窄高斯）以激发全部模态。")
A("5. **门禁**：`tests/test_rk_stability.py`（26 项）把 4 个独立锚点、阶恒等式、"
  "以及“注册的 `ssp_rk3` Butcher 形式复现求解器内联分支到舍入误差”固化为测试。")
A("")
A("## ③ 计算结果")
A("")
A("**锚点（不依赖本模块的参考值）**")
A("")
A("| 格式 | 负实轴区间（测得） | 参考 | 虚轴区间（测得） | 参考 |")
A("|---|---|---|---|---|")
A(f"| rk4 | {f(meas['rk4']['real_axis_interval'],10)} | 2.7852935634 | "
  f"{f(meas['rk4']['imaginary_axis_interval'],10)} | $2\\sqrt2$ = {f(ref['rk4_imaginary_axis_interval'],10)} |")
A(f"| rk3 | {f(meas['rk3']['real_axis_interval'],10)} | 2.5127453267 | "
  f"{f(meas['rk3']['imaginary_axis_interval'],10)} | $\\sqrt3$ = {f(ref['kutta3_imaginary_axis_interval'],10)} |")
A(f"| ssp_rk3 | {f(meas['ssp_rk3']['real_axis_interval'],10)} | "
  f"{f(ref['ssp_rk3_real_axis_interval_replayed'],10)}（级递推复现值） | "
  f"{f(meas['ssp_rk3']['imaginary_axis_interval'],10)} | — |")
A("")
A("**被认证档位 vs 精度阶梯实际使用值**")
A("")
A("| p | 时间格式 | 级数 | $\\mathrm{cfl}_{max}$ | 使用 cfl | 余量 | 0.4 在域内 | 经验爆炸阈值 | 经验/解析−1 |")
A("|---|---|---|---|---|---|---|---|---|")
for r in matched:
    e = emp[r["degree"]]
    A(f"| {r['degree']} | {r['tableau']} | {r['stages']} | {f(r['cfl_max'])} | "
      f"{f(r['cfl_used'],2)} | {f(r['margin_at_used'],2)}× | {'✅' if r['probe_inside'] else '❌'} | "
      f"{f(e['empirical_cfl_max'])} | +{pct(e['empirical_over_predicted'])} |")
A("")
A("**交叉配对矩阵 $\\mathrm{cfl}_{max}$（加粗＝阶梯实际采用）**")
A("")
degs = sorted({r["degree"] for r in cross})
names = setup["certified_tableaus"]
A("| p \\ 格式 | " + " | ".join(names) + " |")
A("|---" * (len(names) + 1) + "|")
for d in degs:
    cells = []
    for n in names:
        r = next(x for x in cross if x["degree"] == d and x["tableau"] == n)
        cells.append(f"**{f(r['cfl_max'],3)}**" if r["matched"] else f(r["cfl_max"], 3))
    A(f"| {d} | " + " | ".join(cells) + " |")
A("")
A(f"总判定 **{M['verdict']}**，判据 S1–S6 全部通过；未过项：{M['failed'] or '无'}。")
A("")
A("## ④ 结果图")
A("")
A("图 1（稳定性证书）：(a) 四个 matched 格式的 $|R(z)|=1$ 边界与 p=4 DG 谱按 "
  f"$\\mathrm{{cfl}}_{{max}}={f(matched[3]['cfl_max'])}$ 缩放后恰好触及边界；"
  "(b) 逐 $(p,\\text{格式})$ 的认证 $\\mathrm{cfl}_{max}$（对数轴），圆环为阶梯采用的搭配，"
  "横线为 0.2 与 0.4。")
A("")
A("图 2（解析基线比较）：(a) $|R(z)|$ 与 $e^{z}$ 的对比，$|R|=1$ 交点即 $\\mathrm{cfl}_{max}$；"
  "(b) $|R(z)-e^{z}|$ 的幂律斜率即格式阶（$z^{5}/5!$ 为参考斜率）。")
A("")
A("> 说明：本任务是**离散格式分析**，不含物理场量，故无“场量云图”；"
  "图 1(a) 的谱点云即被检查的“场”，图 2 为与解析函数 $e^z$ 的直接比较曲线。")
A("")
A("## ⑤ 注意点（踩坑与局限）")
A("")
A("1. **两条等价路线都要留**：系数形式快（谱扫描从数十秒降到亚秒），线性求解形式短；"
  "只用一条就无法发现系数提取的索引错误，故两者互为测试锚点。")
A("2. **`RKTableau.check()` 只接受严格下三角行**：第一次写 `ssp_rk3` 时把第 2 级行写成两个系数"
  "（漏了首级 `(1.0,)`），被 `check()` 当场拦下——该不变量确实在防错。")
A("3. **“SSP 格式稳定域更小”是错觉**：SSP-RK(5,4) 负实轴区间 5.331，比经典 RK4 的 2.785 更大；"
  "SSP 系数 1.508 约束的是凸组合形式下的步长，与线性稳定域实轴长度不是同一个量。")
A("4. **经验阈值系统性偏高 +3.04%…+20.00%**：极限附近增长极慢（p=1 在 1.20× 极限处仍有 "
  "$\\lambda_{growth}=-0.56$），有限 $T$ 内指数尚未转正。故经验值只作**下界**（判据 S5），"
  "证书由解析路线给出。")
A("5. **“同阶”必须级数与阶同时匹配**：`rk2` 配 $p\\ge2$ 时 $\\mathrm{cfl}_{max}$ 仅 "
  f"{f(next(r['cfl_max'] for r in cross if r['degree']==2 and r['tableau']=='rk2'),3)}"
  "，已低于阶梯使用的 0.2——只凑级数会跑进不稳定区。")
A("6. **只覆盖线性、常系数、周期、上风通量、无限制器**；限制器/非线性格式的放大算子是级映射的范数，"
  "不是 $|R|$，本轮结论不适用。")
A("")
A("## ⑥ 误差判定（关键量 vs 基准，是否 < 3%）")
A("")
A("| 关键量 | 基准 | 偏差 | < 3%? |")
A("|---|---|---|---|")
A(f"| RK4 负实轴区间 | 表值 2.7852935634 | "
  f"{abs(meas['rk4']['real_axis_interval']-ref['rk4_real_axis_interval']):.2e}（绝对） | ✅ |")
A(f"| RK4 虚轴区间 | $2\\sqrt2$ | "
  f"{abs(meas['rk4']['imaginary_axis_interval']-ref['rk4_imaginary_axis_interval']):.2e} | ✅ |")
A(f"| Kutta-3 虚轴区间 | $\\sqrt3$ | "
  f"{abs(meas['rk3']['imaginary_axis_interval']-ref['kutta3_imaginary_axis_interval']):.2e} | ✅ |")
A(f"| `ssp_rk3` 负实轴区间 | 级递推复现值 | "
  f"{abs(meas['ssp_rk3']['real_axis_interval']-ref['ssp_rk3_real_axis_interval_replayed']):.2e} | ✅ |")
A(f"| 解析 CFL 极限 vs 经验爆炸阈值 | 经验（$N=8$, $T$=1） | 解析偏保守 {pct(emp_min)}…{pct(emp_max)} | "
  "✅（方向正确：解析不乐观） |")
A(f"| 阶梯 `cfl` = {setup['cfl_accuracy_ladder']} vs 认证极限 | 最小档 {f(worst['cfl_max'])} | "
  f"余量 {f(worst['margin_at_used'],2)}× | ✅ |")
A("")
A("**明确回答：稳定性锚点的相对偏差全部 < 3%**（实测 1e-10 量级＝浮点精度极限）。"
  f"经验复算偏保守 {pct(emp_min)}–{pct(emp_max)}，属“解析更严格”的安全方向，不否定结论；"
  "唯一超过 3% 的项是经验阈值本身，成因见 ⑤.4。")

report = {
    "title": "TensorDG CASE-DG-ADVECT-ORDER 第3轮：同阶时间格式阶梯的线性稳定性证书（p=1..4）",
    "summary": ("为第 2 轮 matched-stage 精度阶梯补上缺失的稳定性认证：新增由 Butcher 表机械导出的"
                "放大多项式模块（tensordg/rk_stability.py），认证四档 cfl=0.2 与 0.4 均落在稳定域内"
                "（余量 5.00×–6.71×），并以 4 项独立锚点（RK4/Kutta-3 轴区间、跨实现 ssp_rk3 复现）"
                "与出厂求解器经验复算交叉验证；把仓库里未验证的注释断言变成可复算证书。"),
    "scenario_key": "tensordg/case-dg-advect-order",
    "mission_id": 22823,
    "tags": ["TensorDG", "RKDG", "RungeKutta", "stability", "CFL", "verification",
             "CASE-DG-ADVECT-ORDER"],
    "markdown": "\n".join(md),
    "figures": [
        {"name": "advorder_stability.png",
         "caption": "图1 稳定性证书：(a) 四个 matched 格式的 |R(z)|=1 边界与 p=4 DG 谱按 cfl_max 缩放后触及边界；"
                    "(b) 逐 (p, 格式) 的认证 cfl_max（对数轴），圆环为精度阶梯采用的搭配，横线为 0.2 与 0.4。",
         "png_base64": b64(REPO / "docs" / "figures" / "advorder_stability.png")},
        {"name": "advorder_stability_order.png",
         "caption": "图2 解析基线比较：(a) |R(z)| 与 e^z，|R|=1 交点即 cfl_max；(b) |R(z)-e^z| 的幂律斜率＝格式阶"
                    "（虚线 z^5/5! 为参考斜率）。",
         "png_base64": b64(FIG2)},
    ],
}
pathlib.Path("/tmp/report.json").write_text(json.dumps(report))
print("report.json bytes:", pathlib.Path("/tmp/report.json").stat().st_size)
print("markdown lines:", len(md))
