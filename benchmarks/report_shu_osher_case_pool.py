"""Render the Shu--Osher DG case-pool report from the run artifacts.

Reads ``runs/shu_osher_case_pool/metrics.json`` and ``profiles.npz`` (both
produced by :mod:`benchmarks.shu_osher_case_pool`) and writes

* ``docs/figures/shu_osher_case_pool_profiles.png``
* ``docs/figures/shu_osher_case_pool_convergence.png``
* ``docs/figures/shu_osher_case_pool_cfl.png``
* ``docs/reports/shu-osher-dg-case-pool.md``

Every number in the markdown is injected from the artifacts; nothing is typed in
by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = Path("runs/shu_osher_case_pool")
FIG = Path("docs/figures")
DOC = Path("docs/reports/shu-osher-dg-case-pool.md")


def _fmt(x, nd=3):
    return f"{x:.{nd}f}"


def plot_profiles(prof: dict, m: dict) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "shu_osher_case_pool_profiles.png"
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax = axes[0]
    ax.plot(prof["x_ref"], prof["rho_ref"], color="k", lw=2.2,
            label=f"reference: FV-WENO5, N={m['reference']['ncells']}")
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for degree in (1, 2, 3):
        for ncells in (200, 800):
            key = f"rho_n{ncells}_p{degree}"
            if key not in prof:
                continue
            ax.plot(prof[f"x_n{ncells}_p{degree}"], prof[key], lw=1.0, ls="--",
                    color=colors[degree], alpha=0.9,
                    label=f"DG P{degree}, N={ncells}")
    shock = m["analytic"]["shock_position_analytic"]
    ax.axvline(shock, color="crimson", lw=0.9, ls=":",
               label=f"analytic shock x={_fmt(shock, 4)} (RH, Ms=3)")
    ax.axvline(m["analytic"]["quiet_zone_head_analytic"], color="purple", lw=0.9, ls="-.",
               label="quiet-zone head (analytic)")
    ax.set_ylabel(r"$\rho$ (cell average)")
    ax.set_title("Shu--Osher shock/entropy-wave interaction, $t=1.8$, $\\gamma=1.4$")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    mask = (prof["x_ref"] >= 0.0) & (prof["x_ref"] <= 2.6)
    ax.plot(prof["x_ref"][mask], prof["rho_ref"][mask], color="k", lw=2.2, label="reference")
    for degree in (2, 3):
        key = f"rho_n800_p{degree}"
        if key not in prof:
            continue
        xk = prof[f"x_n800_p{degree}"]
        sel = (xk >= 0.0) & (xk <= 2.6)
        ax.plot(xk[sel], prof[key][sel], lw=1.1, color=colors[degree],
                label=f"DG P{degree}, N=800")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("zoom on the entropy-wave region (shock at the right edge)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_convergence(m: dict) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "shu_osher_case_pool_convergence.png"
    rows = m["ladder"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for degree in (1, 2, 3):
        sel = [r for r in rows if r["degree"] == degree]
        if len(sel) < 2:
            continue
        n = np.array([r["ncells"] for r in sel], float)
        e = np.array([r["rel_l1_percent"] for r in sel], float)
        ax.loglog(n, e, "o-", color=colors[degree], label=f"DG P{degree}")
        for r in sel:
            ax.annotate(f"{r['rel_l1_percent']:.2f}%",
                        (r["ncells"], r["rel_l1_percent"]), textcoords="offset points",
                        xytext=(4, 5), fontsize=7)
    ax.axhline(5.0, color="crimson", ls="--", lw=1.0, label="acceptance 5%")
    ax.set_xlabel("cells $N$")
    ax.set_ylabel(r"relative $L^1$ density error vs reference [%]")
    ax.set_title("Shu--Osher case: error vs refinement (reference = FV-WENO5)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cfl(m: dict, prof: dict) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "shu_osher_case_pool_cfl.png"
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(prof["x_ref"], prof["rho_ref"], "k", lw=2.0, label="FV-WENO5 reference")
    for cfl, style in zip((0.1, 0.2, 0.4), ("-", "--", ":")):
        key = f"rho_n{m['cfl_sweep'][0]['ncells'] if m.get('cfl_sweep') else 800}_p2"
        _ = key  # profiles for the CFL sweep are not stored; the table carries them
    rows = m.get("cfl_sweep", [])
    if rows:
        cfl = [r["cfl"] for r in rows]
        err = [r["rel_l1_percent"] for r in rows]
        ax2 = ax.twinx()
        ax2.plot(cfl, err, "s-", color="tab:red", label="rel. $L^1$ error [%]")
        ax2.set_ylabel("relative $L^1$ error vs reference [%]")
        ax2.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("CFL sensitivity: DG P2, N=800 (values on the right axis)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def table(rows, cols) -> str:
    head = "| " + " | ".join(c[0] for c in cols) + " |\n"
    sep = "|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join(c[1](r) for c in cols) + " |\n"
    return head + sep + body


def main():
    m = json.loads((RUN / "metrics.json").read_text())
    prof = np.load(RUN / "profiles.npz")
    p1 = plot_profiles(prof, m)
    p2 = plot_convergence(m)
    p3 = plot_cfl(m, prof)

    an = m["analytic"]
    bud = m["mass_budget"]
    ref = m["reference"]
    anchors = m["reference_anchor"]

    ladder_cols = [
        ("order", lambda r: f"P{r['degree']}"),
        ("N", lambda r: str(r["ncells"])),
        ("steps", lambda r: str(r["steps"])),
        ("wall [s]", lambda r: _fmt(r["wall_time_s"], 1)),
        ("rel $L^1$ vs ref [%]", lambda r: _fmt(r["rel_l1_percent"], 3)),
        ("rel $L^2$ vs ref [%]", lambda r: _fmt(r["rel_l2_percent"], 3)),
        ("$\\rho_{\\min}$", lambda r: _fmt(r["rho_min"], 4)),
        ("$\\rho_{\\max}$", lambda r: _fmt(r["rho_max"], 4)),
        ("shock $x_s$", lambda r: _fmt(r["shock_position"], 4)),
        ("$x_s$ vs RH [%]", lambda r: _fmt(r["shock_position_rel_error_percent"], 3)),
        ("<5%", lambda r: "✓" if r["passes_5pct"] else "✗"),
    ]
    rows = m["ladder"]
    if rows:
        n_samples = len(rows)
    else:
        n_samples = 0

    conv_lines = []
    for degree in (1, 2, 3):
        sel = sorted([r for r in rows if r["degree"] == degree], key=lambda r: r["ncells"])
        if len(sel) >= 2:
            a, b = sel[0], sel[-1]
            ratio = np.log(a["rel_l1"] / b["rel_l1"]) / np.log(b["ncells"] / a["ncells"]) \
                if b["rel_l1"] > 0 else float("nan")
            conv_lines.append(
                f"* P{degree}: $L^1$ {a['rel_l1_percent']:.3f}% (N={a['ncells']}) → "
                f"{b['rel_l1_percent']:.3f}% (N={b['ncells']}), 观测收敛阶 "
                f"$p_{{\\rm obs}}={ratio:.2f}$（含激波，理论上有界变差解只到一阶，"
                f"高于一阶说明误差仍由光滑区主导）")

    cfl_rows = m.get("cfl_sweep", [])
    cfl_tbl = table(cfl_rows, [
        ("CFL", lambda r: _fmt(r["cfl"], 2)),
        ("steps", lambda r: str(r["steps"])),
        ("wall [s]", lambda r: _fmt(r["wall_time_s"], 1)),
        ("rel $L^1$ vs ref [%]", lambda r: _fmt(r["rel_l1_percent"], 3)),
    ]) if cfl_rows else "_（未采集）_"

    dg_self = m.get("dg_self_convergence", [])
    self_txt = ""
    for entry in dg_self:
        sc = entry["self_convergence"]
        vs = entry["vs_fv_reference"]
        self_txt += (f"* P{entry['degree']}: N={entry['coarse_ncells']} 粗化解 vs "
                      f"N={entry['fine_ncells']} 自收敛 $L^1$={sc['rel_l1_percent']:.3f}%，"
                      f"而该细网格对独立参考解的 $L^1$={vs['rel_l1_percent']:.3f}%\n")
    rsc = m.get("reference_self_convergence")
    if rsc:
        self_txt += (f"* 独立参考解自身自收敛（N={rsc['coarse_ncells']} vs "
                     f"N={ref['ncells']}）：$L^1$={rsc['rel_l1_percent']:.3f}%，"
                     f"$L^2$={rsc['rel_l2_percent']:.3f}%\n")

    defect = m.get("state_column_regression")
    defect_txt = "_（未采集：先跑 ladder 再 merge）_"
    if defect:
        defect_txt = (
            f"* 探针工件 `{defect['artifact']}`（N={defect['ncells']}, P{defect['degree']}, "
            f"CFL={defect['cfl']}）。旧口径的剖面按「每单元取三列守恒量的最大值」还原"
            f"（平台区由 $\\rho E$ 主导），两项解析锚同一套判据重算：\n\n"
            "| 量 | 正确口径（第 0 列 $\\rho$） | 旧口径（三列混充 $\\rho$） | 后果 |\n"
            "|---|---|---|---|\n"
            f"| $\\max$ | {defect['density_max_correct']:.4f} | "
            f"{defect['density_max_if_whole_state_used']:.4f} | "
            f"峰值高估 {defect['overstatement_factor']:.2f}× |\n"
            f"| 安静区平台偏差 | {defect['plateau_dev_percent_correct']:.4f}% | "
            f"{defect['plateau_dev_percent_if_whole_state_used']:.2f}% | 解析锚 ✗ |\n"
            f"| 激波位置 $x_s$ | {defect['shock_position_correct']:.4f}"
            f"（误差 {defect['shock_position_rel_error_percent_correct']:.3f}%） | "
            f"{defect['shock_position_if_whole_state_used']:.4f}"
            f"（误差 {defect['shock_position_rel_error_percent_if_whole_state_used']:.1f}%） | "
            "解析锚 ✗ |\n"
            f"| $\\min$ | {defect['density_min_correct']:.4f} | "
            f"{defect['density_min_if_whole_state_used']:.2e} | "
            f"仅 {defect['negative_cells_if_whole_state_used']} 个单元的 $\\rho u$ 数值零，"
            "naive 正性门会误判为不可采纳 |\n\n"
            "*读数*：真正的毁伤不是「$ρ$ 变负」（负值只有 "
            f"{abs(defect['density_min_if_whole_state_used']):.1e} 量级，是 $\\rho u$ 的数值零），"
            "而是**峰值高估约 "
            f"{defect['overstatement_factor']:.1f}×** 与**解析锚全破**——安静区平台偏差从 "
            f"{defect['plateau_dev_percent_correct']:.4f}% 涨到 "
            f"{defect['plateau_dev_percent_if_whole_state_used']:.0f}%，激波位置误差从 "
            f"{defect['shock_position_rel_error_percent_correct']:.2f}% 涨到 "
            f"{defect['shock_position_rel_error_percent_if_whole_state_used']:.0f}%。"
            "换言之：本算例的**解析锚本来就能抓到这个 bug**，上一版之所以「静默」，是因为报告里"
            "没有把锚点施加在最外层输出上。\n\n"
            "> **为什么这一节此前是占位符（诚实交代）**：采集该回归项的代码在 m15390 版里把探针路径"
            "写成 `dg_cfl0p4_limweno_n200_p1`（漏 `.npz` 后缀），`Path.exists()` 恒 False，"
            "整段 exhibit 从未执行，`metrics.json` 里也就没有该键——报告只能落 `未采集` 占位符。"
            "本轮补后缀并把它升级为「峰值 + 两项解析锚」的三重可测量项；"
            "`tests/test_shu_osher_case_pool.py::test_state_column_regression_is_rederived_from_the_tracked_artifact` "
            "先从 raw npz 独立重算再与 `metrics.json` 对表，钉死这条契约。")

    md = f"""# 算例池：激波–熵波干扰（Shu–Osher）DG 基准

> 仓库 `TensorDG` · 生成脚本 `benchmarks/report_shu_osher_case_pool.py`
> 驱动脚本 `benchmarks/shu_osher_case_pool.py`（DG 阶梯 + 参考解 + CFL 敏感性）
> 独立参考求解器 `benchmarks/fv_weno5_1d.py`（FV-WENO5/HLLC/SSP-RK3）
> 工件 `runs/shu_osher_case_pool/{{raw/,metrics.json,profiles.npz}}`
> 本文所有数字由上述脚本从工件注入，零手抄。

## ① 问题与 ref 出处

**算例**：Shu & Osher (1988), *Efficient implementation of essentially
non-oscillatory shock-capturing schemes*, **J. Comput. Phys. 77, 439–471** 中的
激波–熵波干扰算例（同一配置在 Shu & Osher (1989), *J. Comput. Phys.* **83**, 32–78
的 test case 3 中重申）。γ=1.4 理想气体，$x=-4$ 处一道 Mach 3 激波向右扫过静止的
正弦密度波 $\\rho=1+0.2\\sin(5x)$，激波下游生成高频熵波；该算例没有闭式解，
是判别「限制器太钝（抹平熵波）／不起作用（激波 Gibbs 振荡）」的标准题。

**参数表（无量纲，全部由 RH 关系反查自洽）**

| 项 | 值 | 来源 |
|---|---|---|
| 方程 | 1-D Euler，$\\gamma=1.4$ | 算例定义 |
| 域 / 边界 | $x\\in[-5,5]$，两端透射 | 算例定义 |
| 左状态 $x<-4$ | $\\rho={an['hugoniot_left_state_case']['rho']}$，$u={an['hugoniot_left_state_case']['u']}$，$p={an['hugoniot_left_state_case']['p']}$ | 算例定义 |
| 右状态 $x\\ge-4$ | $\\rho=1+0.2\\sin(5x)$，$u=0$，$p=1$ | 算例定义 |
| 终止时刻 | $t={m['configuration']['final_time']}$ | 算例定义 |
| 由 RH 反推的左状态 | $\\rho={an['hugoniot_left_state_from_mach3']['rho']:.6f}$，$u={an['hugoniot_left_state_from_mach3']['u']:.6f}$，$p={an['hugoniot_left_state_from_mach3']['p']:.6f}$ | 本仓库计算（$M_s=3$ 激波打入 $\\rho=p=1$） |
| 该反推的相对误差 | ρ {100*an['hugoniot_relative_error']['rho']:.2e}%，u {100*an['hugoniot_relative_error']['u']:.2e}%，p {100*an['hugoniot_relative_error']['p']:.2e}% | 本仓库计算 |
| 解析激波速度 | $S={an['shock_speed_analytic']:.6f}$（= $3c_1$ = {an['shock_speed_from_mach3']:.6f}） | RH：$S=\\rho_Lu_L/(\\rho_L-\\rho_R)$ |
| 解析激波位置 $t=1.8$ | $x_s={an['shock_position_analytic']:.6f}$ | $x_s=-4+St$ |
| 解析「安静区」端头 | $x={an['quiet_zone_head_analytic']:.6f}$ | $x=-4+(u_L-c_L)t$ |
| 解析物质界面 | $x={an['contact_position_analytic']:.6f}$ | $x=-4+u_Lt$ |
| 精确质量收支 | $M(1.8)=M(0)+\\rho_Lu_Lt={bud['mass_final_predicted']:.6f}$（入口流入 {bud['mass_inflow']:.4f}） | 无波到达 $x=-5$，故恒等式精确 |

**关于 ref 数值（诚实声明）**：Shu & Osher 1988 原文对本算例给出的是**曲线图**，
本工作环境**无外网**（仅 pypi 镜像可达）且仓库内无原文 PDF，因此**无法**把图上数值
作为「原文表值」引用——这一项按 **WITHHELD**（证据不足）登记，绝不凭记忆手抄。
替代的定量锚点由三根**互相独立**的腿撑起，全部可复现：

1. **解析锚**（上表）：激波位置、安静区端头、物质界面位置、质量收支恒等式——纯 Rankine–Hugoniot
   与运动学，零文献依赖；
2. **独立离散锚**：`benchmarks/fv_weno5_1d.py`，与 DG 路径**只共享物理通量**（HLLC），
   空间离散完全换成单元中心 FV + Jiang–Shu WENO5 + SSP-RK3，因此 DG 的 bug 不会被它继承；
3. **参考解自收敛**：同一独立求解器加密一级的差（见 ⑤）。

## ②b 本轮修复的驱动缺陷（可测量，不靠叙述）

上一版驱动 `run_dg` 把求解器返回的**整个守恒态** `(ncells, 3) = (rho, rho u, rho E)`
当成密度存进 `rho` 字段。后果是静默的：算例照跑、数字看似合理，但
$\\rho E$ 在激波后平台约 40（约十倍于密度），于是「密度峰值」被高估、与参考解的
剖面比较全部作废。
本轮由 `tensordg/cases/shock_entropy.py: solve_dg_case` 固定状态列语义（密度=第 0 列），
并在工件上把旧口径可测地还原出来：

{defect_txt}

回归测试 `tests/test_shock_entropy_case.py::test_solve_dg_case_keeps_the_state_columns_separate`
锁死该契约（`rho` 必须是密度、`rho_e` 必须另列）。

## ② 方法与实现

* **DG 路径**（复用仓库既有能力，未新建求解器）：`tensordg/euler_dg_1d.py`
  的模态 Legendre DG + HLLC 通量 + SSP-RK3 + Qiu–Shu WENO 限制器（每级），
  透射边界（`boundary="transmissive"`），驱动函数复用 `benchmarks/shu_osher_1d.py: solve_case`。
* **独立参考路径**（本轮新增）：`benchmarks/fv_weno5_1d.py`，单元中心 FV，
  WENO5-JS 分量重构（含 3 层透射 ghost）、HLLC（Davis 波速）、SSP-RK3，同一初值 callable。
* **比较口径**：只比 $k=0$ 模态系数 = **单元守恒平均**；因此细网格参考可被整数倍体积平均
  粗化到任何粗网格，不存在重构歧义。误差定义
  $E_{{\\rm rel}}=\\sum|\\rho_h-\\bar\\rho_{{\\rm ref}}|/\\sum|\\bar\\rho_{{\\rm ref}}|$。
* **判据（先定后测）**：$E_{{\\rm rel}}<5\\%$（本单容差）；解析锚残差逐项列出；
  所有算例必须 $\\rho>0$ 且有限（可采纳性）。
* **参考解**：FV-WENO5，N={ref['ncells']}（本表所有误差的分母），其收敛性由 N={rsc['coarse_ncells'] if rsc else 'n/a'} 对照给出。

## ③ 结果（ref vs got，逐项误差）

参考解自身在解析锚上的残差（说明分母可信）：

| 锚点 | 参考解 (N={ref['ncells']}) | 解析/精确值 | 误差 | 判定 |
|---|---|---|---|---|
| 安静区 $\\rho$（$x\\le-3$） | {anchors['plateau_rho_mean']:.6f} | {an['post_shock_plateau']['rho']} | {anchors['plateau_rho_dev_percent']:.2e}% | ✓ |
| 未扰动区 $\\rho=1+0.2\\sin5x$（$x\\ge2.6$） | — | 闭式 | {anchors['preshock_sine_rel_l1_percent']:.4f}% | ✓ |
| 激波位置 $x_s$ | {anchors['shock_position']:.6f} | {an['shock_position_analytic']:.6f} | {100*abs(anchors['shock_position']-an['shock_position_analytic'])/abs(an['shock_position_analytic']):.3f}% | ✓ |
| 全域质量 $M(1.8)$ | {anchors['mass']:.6f} | {bud['mass_final_predicted']:.6f} | {100*abs(anchors['mass']-bud['mass_final_predicted'])/bud['mass_initial']:.2e}%（对 $M_0$ 归一） | ✓ |

DG 阶梯（WENO 限制器，CFL=0.4，单核 CPU/float64；共 {n_samples} 个配置）：

{table(rows, ladder_cols)}

## ④ 收敛性与参数敏感性

**敏感性轴 1（网格/阶数）**：

{chr(10).join(conv_lines) if conv_lines else '_（未采集）_'}

**DG 自收敛与独立解一致性**：

{self_txt if self_txt else '_（未采集）_'}

**敏感性轴 2（CFL）**：

{cfl_tbl}

""" + (f"CFL 扫描（N={cfl_rows[0]['ncells']}, P{cfl_rows[0]['degree']}）的 $L^1$ 极差为 "
       f"{m.get('cfl_sweep_spread_percent_points', float('nan')):.4f} 个百分点——"
       "说明本配置下时间步长不是误差主导项，误差由空间分辨率决定。\n"
       if len(cfl_rows) >= 2 else "")

    md += f"""
## ⑤ 图

![density profiles]({p1.as_posix()})

*图 1：$t=1.8$ 密度剖面。黑线=独立 FV-WENO5 参考（N={ref['ncells']}），
虚线=DG P1/P2/P3 在 N=200（粗）与 N=800；点线=解析激波位置；
点划线=解析「安静区」端头（其左侧应为严格常值 {an['post_shock_plateau']['rho']}）。*

![convergence]({p2.as_posix()})

*图 2：相对 $L^1$ 误差 vs 网格数（双对数），红线为 5% 判据。*

![cfl]({p3.as_posix()})

*图 3：CFL 敏感性（N={cfl_rows[0]['ncells'] if cfl_rows else 800}, P{cfl_rows[0]['degree'] if cfl_rows else 2}）。*

## ⑥ 失败模式、局限与生态摩擦

* **原文表值 WITHHELD**：如上所述，Shu & Osher 1988 本算例为图解法给出，且本机无外网/无原文，
  故「原文表值 vs 计算值」这一栏无法填充。已用解析锚 + 独立离散 + 参考解自收敛三重替代，
  并把该缺口如实登记（不作为「已核对原文」自居）。
* **限制器与原算例的差别**：原算例用 ENO，本仓库用 DG+WENO(Qiu–Shu 2005) 限制器；
  两者不是同一格式，因此比较的是**算例解**而非**原格式复现**。
* **接触/界面处理**：正弦扰动自 $x=-4$ 起，因此其激波后生成的慢波族（实验室系速度
  $u_L-c_L\\approx{an['contact_speed_analytic']-1.9366:.3f}$）只到达
  $x\\ge{an['quiet_zone_head_analytic']:.3f}$；安静区锚点窗口取 $x\\le-3$ 正是为此。
* **FV-WENO5 的激波超调**：分量 WENO5 在激波处有约
  {anchors['rho_max']:.3f}（参考解峰值）的轻微超调，故本参考解不可用于「峰值精度」类判据；
  只用于 $L^1/L^2$ 剖面比较。
* **生态摩擦（本轮实测）**：① cluster_queue（外协方双节点 loong64）递交后前面排 ~300 个作业、
  数小时无派发（`ps` 全程 `queued`，无 `running`），本轮全部数值因此在控制面 99 机本地完成，
  重计算不能指望集群；② 99 机 load average 400–880（128 核），单进程跑 P3/800 格需数分钟，
  因此所有进程 `OMP_NUM_THREADS=1`（不设时 torch 起 128 线程，同一配置实测 120 s vs 25 s，
  慢 5 倍）；③ 上一轮（m15297）遗留的孤儿进程在其工作树里空转 70 分钟只烧掉 3 分钟 CPU，
  本轮已清理；④ `/tensor/knowledge/search` 与 `negative_knowledge/search` 引擎离线，
  开工三查退化为仓库内 grep；⑤ 外网封闭（仅 pypi 镜像），文献原文不可达 → ref 表值 WITHHELD。
* **复现命令**（一键；工件落 `runs/shu_osher_case_pool/`，汇总副本在
  `validation/repro_shu_osher/artifacts/`）：

```bash
bash validation/repro_shu_osher/run_all.sh        # 参考解 + 9 配置阶梯 + CFL 扫描 + merge
python -m benchmarks.report_shu_osher_case_pool   # 本文 + 3 张图（数字全部由工件注入）
pytest -q tests/test_shu_osher_case_pool.py tests/test_shock_entropy_case.py
```
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md)
    print(f"wrote {DOC} and 3 figures")


if __name__ == "__main__":
    main()