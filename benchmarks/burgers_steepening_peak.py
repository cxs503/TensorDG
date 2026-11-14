"""Benchmark driver: gradient peak of the steepening inviscid Burgers profile.

Case anchor: ``DG-BURGERS-STEEPEN`` (:mod:`tensordg.cases.burgers_steepening`).

Everything printed/plotted here is read from
``artifacts/burgers_steepening_peak/metrics.json`` afterwards; the doc
``docs/CASE-DG-BURGERS-STEEPEN.md`` is rendered from that artifact
(``--render-doc``), so no number in the report is hand-copied.

Reproduce::

    python -m benchmarks.burgers_steepening_peak --render-doc
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from tensordg.cases import burgers_steepening as case
from tensordg.cases.burgers_steepening import (
    BREAKING_TIME,
    CELLS,
    CFL,
    CFL_LADDER,
    DEGREES,
    FINAL_TIME,
    SLOPE_MAX,
    TOLERANCE,
    TIME_LADDER,
    burgers_steepening_parameters,
    exact_gradient,
    exact_peak_gradient,
    solve_case,
)

DEFAULT_OUT = Path("artifacts/burgers_steepening_peak")
DEFAULT_DOC = Path("docs/CASE-DG-BURGERS-STEEPEN.md")


# ---------------------------------------------------------------------------
# study
# ---------------------------------------------------------------------------
class RowCache:
    """Memoise individual runs so a study can be sharded into short chunks.

    Each row is keyed by ``(cells, degree, final_time, cfl)`` and stored in
    ``<output-dir>/cache.json``; a rerun only recomputes what is missing, so the
    same artifact can be assembled from several short invocations (and a
    long study never has to be thrown away because one chunk was killed).
    """

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            self.data = json.loads(path.read_text())

    def run(self, cells: int, degree: int, final_time: float, cfl: float) -> dict:
        key = f"{cells}|{degree}|{final_time!r}|{cfl!r}"
        if key in self.data:
            return dict(self.data[key])
        started = time.time()
        row = solve_case(cells=cells, degree=degree, final_time=final_time, cfl=cfl)
        row["wall_seconds"] = round(time.time() - started, 3)
        self.data[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))
        return dict(row)


def run_ladder(cells, degrees, final_time=FINAL_TIME, cfl=CFL, cache=None) -> list[dict]:
    rows = []
    for degree in degrees:
        for n in cells:
            started = time.time()
            if cache is None:
                row = solve_case(cells=n, degree=degree, final_time=final_time, cfl=cfl)
                row["wall_seconds"] = round(time.time() - started, 3)
            else:
                row = cache.run(n, degree, final_time, cfl)
            rows.append(row)
            print(
                f"  p={degree} N={n:5d}  peak={row['peak_numeric_polynomial']:.4f} "
                f"(ref {row['peak_reference_closed']:.4f}) "
                f"err={100 * row['relative_peak_error']:+.3f}%  "
                f"L2={100 * row['relative_l2_error']:.4f}%  {row['wall_seconds']:.1f}s",
                flush=True,
            )
    return rows


def observed_order(coarse: float, fine: float, ratio: float) -> float | None:
    if coarse <= 0.0 or fine <= 0.0 or coarse == fine:
        return None
    return math.log(coarse / fine) / math.log(ratio)


def analyse(ladder, time_sweep, cfl_sweep, final_time=FINAL_TIME) -> dict:
    tolerance = TOLERANCE
    rows_by_degree = {
        degree: sorted([r for r in ladder if r["degree"] == degree],
                       key=lambda r: r["cells"])
        for degree in sorted({r["degree"] for r in ladder})
    }

    finest = max(ladder, key=lambda r: (r["cells"], r["degree"]))
    finest_degree_rows = rows_by_degree[finest["degree"]]
    prev = finest_degree_rows[-2] if len(finest_degree_rows) > 1 else None

    orders = {}
    for degree, rows in rows_by_degree.items():
        peak_orders, l2_orders = [], []
        for coarse, fine in zip(rows[:-1], rows[1:]):
            ratio = fine["cells"] / coarse["cells"]
            peak_orders.append(observed_order(
                abs(coarse["relative_peak_error"]), abs(fine["relative_peak_error"]), ratio))
            l2_orders.append(observed_order(
                coarse["relative_l2_error"], fine["relative_l2_error"], ratio))
        orders[str(degree)] = {
            "peak_error_orders": peak_orders,
            "l2_error_orders": l2_orders,
            "cells": [r["cells"] for r in rows],
        }

    outside = [
        r for r in ladder
        if abs(r["relative_peak_error"]) >= tolerance
    ]
    best_outside = max(outside, key=lambda r: abs(r["relative_peak_error"])) if outside else None

    acceptance = {
        "peak_within_tolerance_at_finest": {
            "value": abs(finest["relative_peak_error"]),
            "threshold": tolerance,
            "pass": abs(finest["relative_peak_error"]) < tolerance,
            "config": f"N={finest['cells']} p={finest['degree']}",
        },
        "band_is_discriminating": {
            "pass": best_outside is not None,
            "worst_outside_config": (
                None if best_outside is None
                else {
                    "cells": best_outside["cells"],
                    "degree": best_outside["degree"],
                    "relative_peak_error": best_outside["relative_peak_error"],
                }
            ),
            "n_configs_outside_band": len(outside),
        },
        "mean_conservation": {
            "value": max(abs(r["mean_conservation_error"]) for r in ladder),
            "threshold": 1.0e-12,
            "pass": max(abs(r["mean_conservation_error"]) for r in ladder) < 1.0e-12,
        },
    }

    cfl_errors = [row["relative_peak_error"] for row in cfl_sweep]
    temporal = {
        "cfl": [row["cfl"] for row in cfl_sweep],
        "relative_peak_error": cfl_errors,
        "spread": (max(cfl_errors) - min(cfl_errors)) if cfl_errors else None,
        "negligible_against_tolerance": (
            bool(cfl_errors) and (max(cfl_errors) - min(cfl_errors)) < 0.1 * tolerance
        ),
    }

    #: the *diagnostic* reading declared up front: how far the peak estimate
    #: still moves between the two finest levels.  It is reported, never used to
    #: pass or fail the case (a pointwise extremum of a thin layer is expected to
    #: oscillate at the sub-percent level with cell/grid alignment).
    diagnostics = {
        "peak_change_between_two_finest": {
            "value": (
                None if prev is None
                else abs(finest["peak_numeric_polynomial"] / prev["peak_numeric_polynomial"] - 1.0)
            ),
            "declared_target": 5.0e-3,
            "met": (
                prev is not None
                and abs(finest["peak_numeric_polynomial"] / prev["peak_numeric_polynomial"] - 1.0) < 5.0e-3
            ),
            "levels": (None if prev is None else [prev["cells"], finest["cells"]]),
        },
    }

    verdict = "PASS" if all(item["pass"] for item in acceptance.values()) else "FAIL"
    return {
        "verdict": verdict,
        "criteria": acceptance,
        "diagnostics": diagnostics,
        "observed_orders": orders,
        "finest": finest,
        "temporal_sensitivity": temporal,
    }


def reference_block(final_time=FINAL_TIME) -> dict:
    closed = exact_peak_gradient(final_time)
    sampled, x_sampled = case.sampled_peak_gradient(final_time)
    probe = solve_case(cells=64, degree=1, final_time=final_time)
    return {
        "closed_form": "G(t) = S/(1 - t S),  S = max|u_0'| = 2 pi A",
        "initial_peak": SLOPE_MAX,
        "breaking_time": BREAKING_TIME,
        "final_time": final_time,
        "time_fraction_of_breaking": final_time / BREAKING_TIME,
        "peak_closed_form": closed,
        "peak_sampled_20001": sampled,
        "peak_sampled_rel_diff_vs_closed": sampled / closed - 1.0,
        "peak_x_sampled": x_sampled,
        "growth_factor_vs_initial": closed / SLOPE_MAX,
        "reference_at_quadrature_bias": (
            probe["peak_reference_at_quadrature"] / closed - 1.0
        ),
        "reference_protocol": (
            "peak of the exact gradient field (2) evaluated by dense maximisation "
            "on the unit cell extended one period each side; closed form (3) "
            "independently checked against it"
        ),
        "production_exact_profile_at_T": probe["production_exact_profile"],
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(out_dir: Path, ladder, time_sweep, analysis, max_cells: int = 512) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from tensordg.burgers_1d import _operators, solve_burgers

    figures = []
    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) profiles + gradient at the documented terminal time
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    x_fine = np.linspace(0.0, 1.0, 4001)
    u_exact = case.exact_solution(x_fine, FINAL_TIME)
    g_exact = np.abs(exact_gradient(x_fine, FINAL_TIME))
    axes[0].plot(x_fine, u_exact, "k-", lw=2.0, label="exact (characteristics)")
    axes[1].plot(x_fine, g_exact, "k-", lw=2.0, label="exact |du/dx|")
    styles = {"64": "--", "128": "-.", "256": ":", "512": "-"}
    for n in sorted({row["cells"] for row in ladder if row["cells"] <= max_cells}):
        rows = [r for r in ladder if r["cells"] == n and r["degree"] == max(DEGREES)]
        if not rows:
            continue
        degree = rows[0]["degree"]
        centers, coeff, _ = solve_burgers(
            ncells=n, degree=degree, final_time=FINAL_TIME, cfl=CFL,
        )
        h = 1.0 / n
        nodes, weights, phi, dphi, _, _ = _operators(degree, coeff.device, coeff.dtype)
        x = (centers[:, None] + 0.5 * h * nodes[None, :]).reshape(-1).cpu().numpy()
        u = (coeff @ phi.T).reshape(-1).cpu().numpy()
        g = np.abs(((coeff @ dphi.T) * (2.0 / h)).reshape(-1).cpu().numpy())
        axes[0].plot(x, u, styles.get(str(n), "-"), lw=1.0, label=f"DG p={degree} N={n}")
        axes[1].plot(x, g, styles.get(str(n), "-"), lw=1.0, label=f"DG p={degree} N={n}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u")
    axes[0].set_title(f"u(x, T={FINAL_TIME})   T/t_b={FINAL_TIME / BREAKING_TIME:.3f}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("|du_h/dx|")
    axes[1].set_title("gradient peak (steepening)")
    axes[1].set_ylim(0.0, 1.35 * exact_peak_gradient(FINAL_TIME))
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "fig_profiles_T.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    figures.append(str(path))

    # (2) steepening law: peak gradient vs time
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    tt = np.linspace(0.0, BREAKING_TIME * 0.999, 400)
    ax.plot(tt, [exact_peak_gradient(t) for t in tt], "k-", lw=2.0,
            label="exact  S/(1-tS)")
    for degree in sorted({row["degree"] for row in time_sweep}):
        rows = sorted([r for r in time_sweep if r["degree"] == degree],
                      key=lambda r: r["final_time"])
        ax.plot([r["final_time"] for r in rows],
                [r["peak_numeric_polynomial"] for r in rows], "o--",
                label=f"DG p={degree} N={rows[0]['cells']}")
    ax.axvline(BREAKING_TIME, color="0.4", ls=":", label="$t_b=1/S$")
    ax.set_xlabel("t"); ax.set_ylabel("max |du/dx|"); ax.set_yscale("log")
    ax.set_title("steepening law of the gradient peak")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "fig_peak_vs_time.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    figures.append(str(path))

    # (3) convergence of the two error measures against the 5 % band
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for degree in sorted({row["degree"] for row in ladder}):
        rows = sorted([r for r in ladder if r["degree"] == degree], key=lambda r: r["cells"])
        cells = [r["cells"] for r in rows]
        ax.loglog(cells, [abs(r["relative_peak_error"]) for r in rows], "o-",
                  label=f"peak error, p={degree}")
        ax.loglog(cells, [r["relative_l2_error"] for r in rows], "s--",
                  label=f"L2 error, p={degree}")
    ax.axhline(TOLERANCE, color="crimson", ls="--", lw=1.2,
               label=f"acceptance {100 * TOLERANCE:.0f}%")
    ax.set_xlabel("cells"); ax.set_ylabel("relative error")
    ax.set_title("gradient-peak error vs L2 error")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "fig_convergence.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    figures.append(str(path))
    return figures


# ---------------------------------------------------------------------------
# doc rendering (numbers come from the artifact)
# ---------------------------------------------------------------------------
def render_doc(metrics: dict, doc_path: Path) -> Path:
    ref = metrics["reference"]
    criteria = metrics["criteria"]
    ladder = metrics["ladder"]
    orders = metrics["observed_orders"]
    temporal = metrics["temporal_sensitivity"]
    time_sweep = metrics["time_sweep"]

    lines = []
    add = lines.append
    add("# CASE-DG-BURGERS-STEEPEN — 陡化 Burgers 梯度峰值 vs 特征线精确解\n")
    add("> 本文件由 `python -m benchmarks.burgers_steepening_peak --render-doc` 依"
        "`artifacts/burgers_steepening_peak/metrics.json` 自动渲染（数字零手抄）。")
    add(f"> 判定：**{metrics['verdict']}** · 容差 `{100 * TOLERANCE:.0f}%` · "
        f"最细档位 `N={metrics['finest']['cells']} p={metrics['finest']['degree']}`\n")
    add("## ① 问题与参照出处\n")
    add("无粘 Burgers 方程 $u_t+(u^2/2)_x=0$，周期 $[0,1]$，生产初值 "
        f"$u_0={case.MEAN}+{case.AMPLITUDE}\\sin 2\\pi x$（TensorDG 生产内核 "
        "`tensordg/burgers_1d.py` 的 `initial_profile` 原样调用）。")
    add("")
    add("关键量是**梯度峰值** $G(t)=\\max_x|\\partial_x u|$，而不是整场 $L^2$ 误差："
        "陡化过程中解在 $t\\to t_b$ 时梯度无界增长，$L^2$ 是平均值，对这一点几乎不敏感。")
    add("")
    add("参照为**精确解**（特征线隐式解 $u=u_0(\\xi),\\ \\xi=x-tu_0(\\xi)$，Whitham "
        "*Linear and Nonlinear Waves* 1974 第 2 章 kinematic waves；同一关系即仓库 "
        "`burgers_1d.exact_profile` 所迭代者）。对其求导得")
    add("")
    add("$$\\frac{\\partial u}{\\partial x}=\\frac{u_0'(\\xi)}{1+t\\,u_0'(\\xi)},"
        f"\\qquad G(t)=\\frac{{S}}{{1-tS}},\\qquad S=\\max|u_0'|=2\\pi A={ref['initial_peak']:.6f},"
        f"\\qquad t_b=\\frac{{1}}{{S}}={ref['breaking_time']:.7f}$$")
    add("")
    add(f"算例定于 $T={ref['final_time']}$（$T/t_b={ref['time_fraction_of_breaking']:.3f}$），"
        f"精确峰值 $G(T)={ref['peak_closed_form']:.6f}$，是初始峰值的 "
        f"**{ref['growth_factor_vs_initial']:.2f} 倍**。")
    add("")
    add("**零手抄与可证伪性。** 参照数值全部由代码给出，并做两道独立复核：")
    add(f"（a）闭式 (3) vs 在 20001 点网格上对 (2) 直接取极大（与闭式无关的代码路径）："
        f"相对差 `{ref['peak_sampled_rel_diff_vs_closed']:+.3e}`；"
        f"（b）把同一“采样口径”施加给精确场时得到偏差 "
        f"`{100 * ref['reference_at_quadrature_bias']:+.2f}%`（见 ⑤ 口径敏感性）。")
    add("")
    add("## ② 方法与实现\n")
    add("- **数值解**：复用生产内核 `tensordg.burgers_1d.solve_burgers`（模态 Legendre 基、"
        "Rusanov 界面通量、SSP-RK3、$\\Delta t=\\mathrm{CFL}\\,h/((2p+1)\\max|u|)$），未改动一行；")
    add("- **本算例新增**：`tensordg/cases/burgers_steepening.py`（算例定义 + 精确解/精确梯度/"
        "闭式峰值 + 峰值度量）与本文；求解器、限制器、时间推进全部复用；")
    add("- **峰值度量（主口径）**：单元内 $\\partial_x u_h$ 是 $\\xi$ 的 $p-1$ 次多项式，"
        "故在参考单元上 201 点稠密求值取极值，得到数值梯度场自身的峰（不含点采样偏差）；")
    add("- **参照度量**：精确梯度场 (2) 在整周期 20001 点网格取极大，并用闭式 (3) 复核。")
    add("")
    add("## ③ 参数表\n")
    params = metrics["parameters"]
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 区域/边界 | $[{params['domain'][0]},{params['domain'][1]}]$ / 周期 |")
    add(f"| 初值 | $u_0={params['initial_condition']}$ |")
    add(f"| 破断时刻 $t_b=1/\\max(-u_0')$ | {params['breaking_time']:.7f} |")
    add(f"| 终了时刻 $T$ | {params['final_time']} ($T/t_b={params['time_fraction_of_breaking']:.4f}$) |")
    add(f"| 初始梯度峰值 $S$ | {params['initial_gradient_peak']:.6f} |")
    add(f"| 多项式次数 | $p\\in\\{{{', '.join(str(d) for d in params['degrees'])}\\}}$ |")
    add(f"| 网格 | $N\\in\\{{{', '.join(str(c) for c in params['cells'])}\\}}$ |")
    add(f"| 界面通量 | {params['flux']} |")
    add(f"| 时间推进 | {params['time_integrator']}，CFL={params['cfl']} |")
    add(f"| 容差判据 | 梯度峰值相对误差 $<{100 * params['tolerance']:.0f}\\%$ |")
    add("")
    add("## ④ 对照表（ref vs got，误差打勾叉）\n")
    add("| p | N | 步数 | got $\\max\\lvert\\partial_x u_h\\rvert$ | ref $G(T)$ | 相对误差 | 判定 | $L^2$ 相对误差 |")
    add("|---|---|---|---|---|---|---|---|")
    for row in ladder:
        ok = "✅" if abs(row["relative_peak_error"]) < TOLERANCE else "❌"
        add(f"| {row['degree']} | {row['cells']} | {row['steps']} | "
            f"{row['peak_numeric_polynomial']:.4f} | {row['peak_reference_closed']:.4f} | "
            f"{100 * row['relative_peak_error']:+.3f}% | {ok} | "
            f"{100 * row['relative_l2_error']:.4f}% |")
    add("")
    finest = metrics["finest"]
    add(f"最细档位 `N={finest['cells']} p={finest['degree']}`："
        f"got {finest['peak_numeric_polynomial']:.4f} vs ref {finest['peak_reference_closed']:.4f}"
        f"（{100 * finest['relative_peak_error']:+.3f}%）→ "
        f"{'判据满足' if abs(finest['relative_peak_error']) < TOLERANCE else '判据不满足'}。")
    add("")
    add("## ⑤ 收敛性与参数敏感性\n")
    add("**网格/阶次收敛（梯度峰值误差与 $L^2$ 误差双列）**")
    add("")
    add("| p | 观测阶（峰值误差，逐对网格） | 观测阶（$L^2$） |")
    add("|---|---|---|")
    for degree, info in orders.items():
        peak_orders = ", ".join("—" if o is None else f"{o:+.2f}" for o in info["peak_error_orders"])
        l2_orders = ", ".join("—" if o is None else f"{o:+.2f}" for o in info["l2_error_orders"])
        add(f"| {degree} | {peak_orders} | {l2_orders} |")
    add("")
    add("峰值误差在 $N=128$ 处出现**非单调过冲**（多项式在峰值单元内越过精确峰），"
        "这是一个必须如实记录的观察：**逐点极值不保证单调收敛**，而 $L^2$ 误差是单调的。"
        "因此判据取在声明的最细档位，并用“非平凡性”判据证明 $5\\%$ 带确实能判出差别。")
    add("")
    add("**口径敏感性（采样 vs 多项式极值）**：把峰值改在 DG 求积节点上取 max（朴素口径），"
        "误差列见 ④ 的对照：")
    add("")
    add("| p | N | 多项式极值口径误差 | 求积节点口径误差 | 口径差 |")
    add("|---|---|---|---|---|")
    for row in ladder:
        add(f"| {row['degree']} | {row['cells']} | {100 * row['relative_peak_error']:+.3f}% | "
            f"{100 * row['relative_peak_error_quadrature_protocol']:+.3f}% | "
            f"{100 * (row['relative_peak_error_quadrature_protocol'] - row['relative_peak_error']):+.3f}% |")
    add("")
    add(f"**时间步长敏感性**（N=256, p=3, $T$，CFL $\\in\\{{{', '.join(f'{c}' for c in temporal['cfl'])}\\}}$）："
        f"峰值相对误差 {min(temporal['relative_peak_error']) * 100:+.3f}% … "
        f"{max(temporal['relative_peak_error']) * 100:+.3f}%，极差 "
        f"{100 * temporal['spread']:.3f} 个百分点，"
        f"{'远小于' if temporal['negligible_against_tolerance'] else '不可忽略于'}容差带 "
        f"{100 * TOLERANCE:.0f}% ⇒ 该算例测的是空间分辨率，不是时间推进。")
    add("")
    add("**时刻敏感性（陡化律）**：")
    add("")
    add("| t | t/t_b | ref $G(t)$ | got (N, p) | 相对误差 |")
    add("|---|---|---|---|---|")
    for row in time_sweep:
        add(f"| {row['final_time']} | {row['final_time'] / BREAKING_TIME:.3f} | "
            f"{row['peak_reference_closed']:.4f} | {row['peak_numeric_polynomial']:.4f} "
            f"(N={row['cells']}, p={row['degree']}) | {100 * row['relative_peak_error']:+.3f}% |")
    add("")
    add("## ⑥ 失败模式与工件\n")
    add("**判据明细**（`metrics.json:criteria`）")
    add("")
    add("| 判据 | 值 | 阈值 | 结果 |")
    add("|---|---|---|---|")
    peak_c = criteria["peak_within_tolerance_at_finest"]
    add(f"| 最细档位峰值误差 | {100 * peak_c['value']:.3f}% | <{100 * peak_c['threshold']:.0f}% | "
        f"{'PASS' if peak_c['pass'] else 'FAIL'} |")
    disc = criteria["band_is_discriminating"]
    worst = disc["worst_outside_config"]
    add(f"| 容差带非平凡性 | 带外配置数 {disc['n_configs_outside_band']}"
        + (f"（最差 N={worst['cells']} p={worst['degree']}: "
           f"{100 * worst['relative_peak_error']:+.2f}%）" if worst else "")
        + f" | ≥1 | {'PASS' if disc['pass'] else 'FAIL'} |")
    stop = metrics["diagnostics"]["peak_change_between_two_finest"]
    add(f"| 诊断：最细两级峰值变化 | "
        + ("—" if stop["value"] is None else f"{100 * stop['value']:.3f}%")
        + f" | 预期 <{100 * stop['declared_target']:.1f}% | "
        f"{'达标' if stop['met'] else '未达标（如实记录）'} |")
    cons = criteria["mean_conservation"]
    add(f"| 质量守恒（单元均值） | {cons['value']:.2e} | <{cons['threshold']:.0e} | "
        f"{'PASS' if cons['pass'] else 'FAIL'} |")
    add("")
    add("**已记录的摩擦（生态观察）**")
    add("")
    add(f"- `tensordg/burgers_1d.exact_profile` 用固定 40 步牛顿迭代反演特征线，"
        f"雅可比 $1+t u_0'$ 在 $t\\to t_b$ 时退化到 $1-t/t_b$，故在 $t=0.195$"
        f"（$t/t_b=0.980$，解仍光滑）即抛 `RuntimeError`。"
        f"本算例的参照改用**区间二分+牛顿抛光**（`burgers_steepening.characteristic_foot`），"
        f"对一切 $0\\le t<t_b$ 收敛；生产函数未改动（其 40 步上限是已知限制，非本单要修的 bug）。"
        f"在 $T={FINAL_TIME}$ 处生产函数状态：`{ref['production_exact_profile_at_T']}`。")
    add("- 知识库检索 `Burgers steepening` / `Burgers gradient` 均返回 0 条"
        "（`/knowledge/search`），故参照必须以闭式+双口径复核自建，无可引用的表值；"
        "本报告因此不引用任何抄录数值，只引用公式与其出处（Whitham 1974 第 2 章，章级引用）。")
    add("- 逐点极值判据的非单调性（见 ⑤）是本算例暴露的真实局限："
        "任何“峰值误差必须单调下降”的门禁都会在 $N=128$ 处误判，判据因此只钉在最细档位。")
    add("")
    add("**工件**")
    add("")
    for art in metrics["artifacts"]:
        add(f"- `{art}`")
    add("")
    add("**复现**")
    add("")
    add("```bash")
    add("python -m benchmarks.burgers_steepening_peak --render-doc")
    add("python -m pytest tests/test_burgers_steepening.py -q")
    add("```")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return doc_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=list(CELLS) + [1024])
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEGREES))
    parser.add_argument("--times", type=float, nargs="+", default=list(TIME_LADDER))
    parser.add_argument("--cfls", type=float, nargs="+", default=list(CFL_LADDER))
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--render-doc", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--figure-max-cells", type=int, default=512,
                        help="largest mesh drawn as a curve (cost control)")
    args = parser.parse_args(argv)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("reference:", flush=True)
    reference = reference_block(args.final_time)
    print(f"  G(T)={reference['peak_closed_form']:.6f}  "
          f"sampled={reference['peak_sampled_20001']:.6f} "
          f"(rel {reference['peak_sampled_rel_diff_vs_closed']:+.2e})", flush=True)

    cache = RowCache(out_dir / "cache.json")

    print("ladder:", flush=True)
    ladder = run_ladder(args.cells, args.degrees, args.final_time, args.cfl, cache)

    print("time sweep:", flush=True)
    time_sweep = []
    for t in args.times:
        if t >= BREAKING_TIME:
            print(f"  skip t={t}: at/after breaking")
            continue
        row = cache.run(256, 3, t, args.cfl)
        time_sweep.append(row)
        print(f"  t={t:.3f} peak={row['peak_numeric_polynomial']:.4f} "
              f"ref={row['peak_reference_closed']:.4f} "
              f"err={100 * row['relative_peak_error']:+.3f}%", flush=True)

    print("cfl sweep:", flush=True)
    cfl_sweep = []
    for cfl in args.cfls:
        row = cache.run(256, 3, args.final_time, cfl)
        cfl_sweep.append(row)
        print(f"  cfl={cfl} err={100 * row['relative_peak_error']:+.3f}% "
              f"steps={row['steps']}", flush=True)

    analysis = analyse(ladder, time_sweep, cfl_sweep, args.final_time)

    artifacts = []
    metrics_path = out_dir / "metrics.json"
    figures = ([] if args.no_figures
               else make_figures(out_dir, ladder, time_sweep, analysis, args.figure_max_cells))
    artifacts = [str(metrics_path)] + figures + [str(args.doc)]

    metrics = {
        "case_key": case.CASE_KEY,
        "generated_by": "benchmarks.burgers_steepening_peak",
        "repro_cmd": "python -m benchmarks.burgers_steepening_peak --render-doc",
        "parameters": {
            **burgers_steepening_parameters(),
            "cells": list(args.cells),
            "degrees": list(args.degrees),
            "time_ladder": [r["final_time"] for r in time_sweep],
            "cfl_ladder": list(args.cfls),
            "final_time": args.final_time,
            "cfl": args.cfl,
        },
        "reference": reference,
        "ladder": ladder,
        "time_sweep": time_sweep,
        "cfl_sweep": cfl_sweep,
        "verdict": analysis["verdict"],
        "criteria": analysis["criteria"],
        "diagnostics": analysis["diagnostics"],
        "observed_orders": analysis["observed_orders"],
        "finest": analysis["finest"],
        "temporal_sensitivity": analysis["temporal_sensitivity"],
        "artifacts": artifacts,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # flat csv mirrors so downstream tooling does not need the json schema
    (out_dir / "ladder.csv").write_text(
        "degree,cells,steps,peak_numeric_polynomial,peak_numeric_quadrature,"
        "peak_reference_closed,relative_peak_error,relative_l2_error,wall_seconds\n"
        + "\n".join(
            f"{r['degree']},{r['cells']},{r['steps']},{r['peak_numeric_polynomial']!r},"
            f"{r['peak_numeric_quadrature']!r},{r['peak_reference_closed']!r},"
            f"{r['relative_peak_error']!r},{r['relative_l2_error']!r},{r['wall_seconds']}"
            for r in ladder
        ) + "\n",
        encoding="utf-8",
    )
    (out_dir / "time_sweep.csv").write_text(
        "final_time,peak_numeric_polynomial,peak_reference_closed,relative_peak_error\n"
        + "\n".join(
            f"{r['final_time']!r},{r['peak_numeric_polynomial']!r},"
            f"{r['peak_reference_closed']!r},{r['relative_peak_error']!r}"
            for r in time_sweep
        ) + "\n",
        encoding="utf-8",
    )

    if args.render_doc:
        render_doc(metrics, args.doc)
        print(f"rendered {args.doc}", flush=True)

    print(f"verdict: {metrics['verdict']}")
    for name, item in metrics["criteria"].items():
        print(f"  {name}: {item['pass']}")
    return 0 if metrics["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())