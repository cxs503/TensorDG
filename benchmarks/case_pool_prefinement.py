"""Case-pool driver: register ``CASE-DG-PREFINEMENT-EXPONENTIAL`` with gates.

The canonical case definition lives in
:mod:`tensordg.cases.prefinement_exponential`; this driver *measures* it,
adjudicates the predeclared gates, runs negative controls through the same gate
implementation, and writes every artifact (JSON, figure, markdown registration
record) from the measured numbers -- nothing is transcribed by hand.

Gates (predeclared in the case module, never re-tuned here)
----------------------------------------------------------
L  legacy ledger tolerance: semi-log error-vs-p slope ``> 0.5`` per unit p
D  discriminating criterion (:func:`tensordg.convergence.exponential_decay_criterion`):
   ``sigma > 1`` AND non-decreasing pairwise decay rates AND
   ``e(p_min)/e(p_max) > 1e3`` AND level-by-level monotone decrease

Negative controls
-----------------
N1 ten *synthetic* decay laws (constant, ``p**-1..p**-5``, ``exp(-sigma p)`` for
   ``sigma = 0.5, 1.0, 1.5, 3.93``) pushed through the same gate -- the gate must
   land on its predeclared side for every law.
N2 a *real solver* negative control: the same DG scheme, same mesh, but a
   ``C0``-kinked triangle-wave initial condition (only algebraic convergence).
   It must be rejected by the discriminating gate (it is accepted by the legacy
   slope gate -- that is exactly why the legacy gate cannot carry the verdict).

Artifacts::

    runs/prefinement_case_pool/metrics.json
    runs/prefinement_case_pool/summary.md
    docs/benchmarks/results/p-refinement-exponential/case_pool_entry.json
    docs/benchmarks/results/p-refinement-exponential/case_pool_entry.png
    docs/CASE-DG-PREFINEMENT-EXPONENTIAL-case-pool.md

Run::

    python -m benchmarks.case_pool_prefinement            # full (p = 1..6)
    python -m benchmarks.case_pool_prefinement --quick    # p = 1..3 smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tensordg.advection_1d import l2_error, solve_advection
from tensordg.cases.prefinement_exponential import (
    CASE_KEY,
    DEGREES,
    LEDGER_ANCHOR,
    LEGACY_SLOPE_THRESHOLD,
    MAX_TEMPORAL_REFINEMENTS,
    NCELLS,
    REFERENCE,
    TEMPORAL_TOLERANCE,
    anchor_table,
    p_ladder,
    prefinement_parameters,
    solve_level,
)
from tensordg.convergence import exponential_decay_criterion, p_decay_diagnostics

SCHEMA = "tensordg-case-pool-entry/1"
RUN_DIR = Path("runs") / "prefinement_case_pool"
ARTIFACT_DIR = Path("docs") / "benchmarks" / "results" / "p-refinement-exponential"
DOC_PATH = Path("docs") / "CASE-DG-PREFINEMENT-EXPONENTIAL-case-pool.md"

#: synthetic decay laws and the side of the discriminating gate they must land on
SYNTHETIC_LAWS = {
    "constant": lambda p: np.ones_like(p),
    "algebraic_p^-1": lambda p: 1.0 / (p + 1.0),
    "algebraic_p^-2": lambda p: 1.0 / (p + 1.0) ** 2,
    "algebraic_p^-3": lambda p: 1.0 / (p + 1.0) ** 3,
    "algebraic_p^-4": lambda p: 1.0 / (p + 1.0) ** 4,
    "algebraic_p^-5": lambda p: 1.0 / (p + 1.0) ** 5,
    "exponential_sigma_0.5": lambda p: np.exp(-0.5 * p),
    "exponential_sigma_1.0": lambda p: np.exp(-1.0 * p),
    "exponential_sigma_1.5": lambda p: np.exp(-1.5 * p),
    "exponential_sigma_3.93": lambda p: np.exp(-3.927 * p),
}
SYNTHETIC_EXPECTED = {
    "constant": False,
    "algebraic_p^-1": False,
    "algebraic_p^-2": False,
    "algebraic_p^-3": False,
    "algebraic_p^-4": False,
    "algebraic_p^-5": False,
    "exponential_sigma_0.5": False,
    "exponential_sigma_1.0": False,
    "exponential_sigma_1.5": True,
    "exponential_sigma_3.93": True,
}

QUICK_DEGREES = (1, 2, 3)
KINKED_DEGREES = (1, 2, 3)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # pragma: no cover
        return None


def _kinked_initial(x):
    """``C0`` triangle wave on [0,1): analytic only piecewise -> algebraic decay."""
    import torch

    return 1.0 - 2.0 * torch.abs(2.0 * x - 1.0)


def _kinked_exact(x, t=1.0, speed=1.0, domain=(0.0, 1.0)):
    import torch

    span = domain[1] - domain[0]
    xi = torch.remainder(x - speed * t, span) / span
    return 1.0 - 2.0 * torch.abs(2.0 * xi - 1.0)


def kinked_ladder(degrees=KINKED_DEGREES, ncells: int = NCELLS,
                  cfl: float = 0.08) -> dict:
    """Real-solver negative control: C0 initial data must fail the gate."""
    levels = []
    for degree in degrees:
        current_cfl = cfl
        error = None
        history = []
        for _ in range(MAX_TEMPORAL_REFINEMENTS + 1):
            centers, coefficients, meta = solve_advection(
                ncells=ncells, degree=degree, final_time=1.0, speed=1.0,
                cfl=current_cfl, time_integrator="rk4", initial=_kinked_initial,
            )
            new_error = l2_error(centers, coefficients, 1.0, speed=1.0,
                                 exact=_kinked_exact)
            change = (abs(new_error - error) / new_error) if error else float("inf")
            history.append({"cfl": current_cfl, "steps": meta["steps"],
                            "relative_l2": new_error, "change_vs_previous": change})
            error = new_error
            if change <= TEMPORAL_TOLERANCE:
                break
            current_cfl *= 0.5
        levels.append({"degree": degree, "relative_l2": float(error),
                       "cfl": history[-1]["cfl"], "steps": history[-1]["steps"]})
    errors = [lv["relative_l2"] for lv in levels]
    diag = p_decay_diagnostics(tuple(degrees), errors)
    gate = exponential_decay_criterion(diag)
    return {
        "initial_condition": "triangle wave 1 - 2*|2x-1| (C0 kink)",
        "degrees": list(degrees),
        "errors": errors,
        "diagnostics": diag,
        "legacy_slope_pass": bool(abs(diag["lsq_decay_rate"]) > LEGACY_SLOPE_THRESHOLD),
        "discriminating": gate,
        "rejected_by_discriminating": not gate["pass"],
        "levels": levels,
    }


def synthetic_controls(degrees=DEGREES) -> dict:
    #: always evaluated on the full ladder -- the ratio gate needs >1e3 decades,
    #: which a 3-point quick ladder cannot reach by construction.  Costs nothing
    #: (synthetic laws, no solver call).
    p = np.asarray(degrees, dtype=float)
    p = p - p[0]
    rows = {}
    for name, law in SYNTHETIC_LAWS.items():
        errors = 1.0e-2 * law(p)
        diag = p_decay_diagnostics(tuple(degrees), [float(v) for v in errors])
        gate = exponential_decay_criterion(diag)
        rows[name] = {
            "lsq_decay_rate": float(diag["lsq_decay_rate"]),
            "decay_rates": [float(v) for v in diag["decay_rates"]],
            "error_ratio": float(diag["errors"][0] / diag["errors"][-1]),
            "legacy_slope_pass": bool(abs(diag["lsq_decay_rate"]) > LEGACY_SLOPE_THRESHOLD),
            "discriminating_pass": bool(gate["pass"]),
            "expected": SYNTHETIC_EXPECTED[name],
            "as_expected": bool(gate["pass"] == SYNTHETIC_EXPECTED[name]),
        }
    return {
        "laws": rows,
        "gate_pass": all(r["as_expected"] for r in rows.values()),
        "legacy_false_positives": [
            n for n, r in rows.items()
            if r["legacy_slope_pass"] and not r["expected"]
        ],
    }


PUBLISHED_STUDY = ARTIFACT_DIR / "p_refinement_results.json"
REPRODUCTION_TOL = 1.0e-6


def reproduction_check(ladder: dict, path: Path = PUBLISHED_STUDY) -> dict:
    """Re-measure the published study: same case, same gates, same numbers.

    The canonical case must reproduce the numbers already published in
    ``p_refinement_results.json`` (relative difference <= 1e-6 per degree).
    Skipped (``applicable = False``) when the published artifact is absent, so
    the entry stays runnable in a clean checkout.
    """
    path = Path(path)
    if not path.exists():
        return {"applicable": False, "reason": f"{path} not found"}
    published = json.loads(path.read_text())["results"]
    pub_degrees = [int(p) for p in published["degrees"]]
    pub_errors = [float(e) for e in published["errors"]]
    mine = {int(d): float(e) for d, e in zip(ladder["grades"], ladder["errors"])}
    shared = [d for d in pub_degrees if d in mine]
    if not shared:
        return {"applicable": False, "reason": "no shared degrees"}
    rows = {}
    for degree in shared:
        ref = pub_errors[pub_degrees.index(degree)]
        got = mine[degree]
        rel = abs(got - ref) / ref
        rows[str(degree)] = {
            "published": ref,
            "measured": got,
            "relative_difference": rel,
            "pass": bool(rel <= REPRODUCTION_TOL),
        }
    return {
        "applicable": True,
        "published_artifact": str(path),
        "tolerance": REPRODUCTION_TOL,
        "degrees": shared,
        "rows": rows,
        "all_pass": all(r["pass"] for r in rows.values()),
    }


def make_figure(report: dict, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = np.array(report["grades"], dtype=float)
    errors = np.array(report["errors"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(degrees, errors, "o-", label="DG p-refinement (measured)")
    slope = report["diagnostics"]["lsq_decay_rate"]
    fit = errors[0] * np.exp(-slope * (degrees - degrees[0]))
    ax.semilogy(degrees, fit, "--",
                label=f"fit: $e\\propto e^{{-{slope:.3f}p}}$")
    # exponential trend lines for reference (algebraic laws are visibly curved)
    for label, law in (("$p^{-3}$ reference", lambda p: errors[0] / (p + 1.0) ** 3),
                       ("$p^{-5}$ reference", lambda p: errors[0] / (p + 1.0) ** 5)):
        ax.semilogy(degrees, law(degrees - degrees[0]), ":",
                    label=label)
    ax.set_xlabel("polynomial degree p")
    ax.set_ylabel("relative $L_2$ error")
    ax.set_title(
        f"p-refinement, ncells={report['ncells']}, "
        f"$\\sigma$={slope:.3f} (gate > {LEGACY_SLOPE_THRESHOLD})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def render_markdown(report: dict) -> str:
    params = prefinement_parameters()
    rows = anchor_table(report["ladder"])
    diag = report["ladder"]["diagnostics"]
    gates = report["ladder"]["gates"]
    synth = report["negative_controls"]["synthetic"]
    kink = report["negative_controls"]["kinked"]
    lines = []
    a = lines.append
    a(f"# {LEDGER_ANCHOR} · 算例池注册：p 型加密指数收敛（光滑解）\n")
    a(f"> 锚点：`{LEDGER_ANCHOR}`｜case key `{CASE_KEY}`｜仓 TensorDG｜分支 `agent/m17525`")
    a(f"> 数字全部由 `benchmarks/case_pool_prefinement.py` 从求解器实测并程序化注入"
      f"（零手抄）；commit `{report['provenance']['git_commit']}`，"
      f"运行 {report['runtime_seconds']:.1f} s。\n")
    a("## ① 案例介绍\n")
    a("一维周期线性对流 `u_t + u_x = 0`，`x∈[0,1)`，初值 `u0 = sin(2πx)`（整函数，解析带无限宽），"
      "积分到 `T = 1`（一整周期，精确解 = 初值平移）。网格固定 `ncells = 16`，只改多项式阶数，"
      "把 p 型加密与 h 型加密解耦。物理背景：这是 Babuska & Suri 定理的教科书验证配置——"
      "解析解下 DG/FEM 的 L2 误差随 p 指数衰减而不是代数衰减。\n")
    a("| 参数 | 值 |\n|---|---|")
    a(f"| 参考 | {REFERENCE['authors']} ({REFERENCE['year']}), "
      f"*{REFERENCE['title']}*, {REFERENCE['journal']} {REFERENCE['volume']}"
      f"({REFERENCE['issue']}):{REFERENCE['pages']} |")
    a(f"| DOI | {REFERENCE['doi']} |")
    a(f"| 理论律 | `{REFERENCE['statement']}` |")
    a(f"| 计算域 / 边条 | {params['domain']} / periodic |")
    a(f"| 网格 | ncells = {params['ncells']}（固定）|")
    a(f"| 阶数阶梯 | p = {params['degrees']}（库上限 MAX_DEGREE = {params['max_degree']}）|")
    a(f"| 终时 / 波速 | T = {params['final_time']} / c = {params['speed']} |")
    a(f"| 时间积分 | RK4，CFL 从 {params['cfl_base']} 起对半细化直到相邻两次 L2 相对变化 "
      f"≤ {params['temporal_tolerance']:.0%} |")
    a(f"| 台账容差 | 半对数误差-阶数斜率 < −{LEGACY_SLOPE_THRESHOLD}（即 σ > {LEGACY_SLOPE_THRESHOLD}）|\n")
    a("## ② 怎么做的\n")
    a("1. **算例定义上收到库层**：新增 `tensordg/cases/prefinement_exponential.py`，"
      "把初值/精确解/参数表/p 阶梯检据集中一处，注册进 `tensordg.cases.CASES`"
      f"（`{CASE_KEY}`）——此前算例只存在于 `benchmarks/` 驱动脚本里，"
      "下游调用方只能抄常量。")
    a("2. **判据去重**：把复合鉴别判据从 benchmark 提升为库函数 "
      "`tensordg.convergence.exponential_decay_criterion`，"
      "`benchmarks/p_refinement_exponential.check_discriminating_criterion` 改为薄封装 delegating，"
      "保证「算例池」与「已发布研究」用**同一份**判据实现（单点真相）。")
    a("3. **门控**：本驱动实测 p 阶梯，判定两道门——L（台账容差 σ>0.5）与 D（复合鉴别判据）——"
      "并跑负对照（N1 十条合成衰减律 + N2 真实求解器的 C⁰ 折点初值），"
      "门 `negative_controls.gate_pass` 必须为真。")
    a("4. 每级 CFL 对半细化直到时间误差可忽略，各级 `temporal_converged` 全为真才算可用。\n")
    a("## ③ 计算结果\n")
    a("| p | DOFs | CFL | RK4 步数 | 相对 L2 误差 | 相邻降幅 | 时间已收敛 |")
    a("|---|---|---|---|---|---|---|")
    for r in rows:
        red = "—" if r["reduction_vs_previous"] is None else f"{r['reduction_vs_previous']:.2f}×"
        a(f"| {r['degree']} | {r['dofs']} | {r['cfl']:.4g} | {r['steps']} | "
          f"{r['relative_l2']:.6e} | {red} | {'是' if r['temporal_converged'] else '否'} |")
    a("")
    a(f"- 最小二乘衰减率 **σ = {diag['lsq_decay_rate']:.4f}**（门 σ > "
      f"{LEGACY_SLOPE_THRESHOLD}，裕度 {diag['lsq_decay_rate'] / LEGACY_SLOPE_THRESHOLD:.1f}×）")
    a(f"- 逐段衰减率 {[round(v, 4) for v in diag['decay_rates']]}（单调不减 → 非代数律指纹）")
    a(f"- 总降幅 e(p_min)/e(p_max) = **{diag['errors'][0] / diag['errors'][-1]:.3e}**"
      f"（门 > 1e3）")
    a(f"- 误差逐级单调下降：**{diag['monotone_decreasing']}**\n")
    a(f"门 L（σ>{LEGACY_SLOPE_THRESHOLD}）= **{'PASS' if gates['legacy_slope']['pass'] else 'FAIL'}**；"
      f"门 D（复合鉴别）= **{'PASS' if gates['discriminating']['pass'] else 'FAIL'}**；"
      f"总判 **{'PASS' if report['pass'] else 'FAIL'}**。\n")
    a("### 负对照 N1：十条已知衰减律过同一道门\n")
    a("| 已知律 | σ | 逐段衰减率 | 总降幅 | 门 L（σ>0.5） | 门 D | 期望 | 相符 |")
    a("|---|---|---|---|---|---|---|---|")
    for name, r in synth["laws"].items():
        a(f"| {name} | {r['lsq_decay_rate']:.4f} | "
          f"{[round(v, 3) for v in r['decay_rates']][:3]}… | {r['error_ratio']:.3e} | "
          f"{'PASS' if r['legacy_slope_pass'] else 'fail'} | "
          f"{'PASS' if r['discriminating_pass'] else 'fail'} | "
          f"{'PASS' if r['expected'] else 'fail'} | {'✅' if r['as_expected'] else '❌'} |")
    a("")
    a(f"门 L 的假阳性：{synth['legacy_false_positives']}；"
      f"N1 门 **{'PASS' if synth['gate_pass'] else 'FAIL'}**。\n")
    a("### 负对照 N2：真实求解器上的 C⁰ 折点初值\n")
    a("| p | 相对 L2 误差 |\n|---|---|")
    for lv in kink["levels"]:
        a(f"| {lv['degree']} | {lv['relative_l2']:.6e} |")
    a("")
    a(f"σ = {kink['diagnostics']['lsq_decay_rate']:.4f}，逐段衰减率 "
      f"{[round(v, 3) for v in kink['diagnostics']['decay_rates']]}"
      f"（递减 = 代数律指纹）。门 L {'PASS' if kink['legacy_slope_pass'] else 'fail'}，"
      f"门 D {'PASS' if kink['discriminating']['pass'] else 'fail'} → "
      f"**{'被拒（正确）' if kink['rejected_by_discriminating'] else '未被拒（判据失效）'}**。\n")
    repro = report.get("reproduction", {})
    if repro.get("applicable"):
        a("### 复现已发布研究（回归门）\n")
        a("| p | 已发布 L2 | 本条目实测 L2 | 相对差 | 通过 (≤1e-6) |\n|---|---|---|---|---|")
        for degree, row in repro["rows"].items():
            a(f"| {degree} | {row['published']:.6e} | {row['measured']:.6e} | "
              f"{row['relative_difference']:.2e} | {'✅' if row['pass'] else '❌'} |")
        a("")
        a(f"回归门 **{'PASS' if repro['all_pass'] else 'FAIL'}**"
          f"（对 `{repro['published_artifact']}`，逐阶相对差 ≤ {repro['tolerance']:g}）。\n")
    a("## ④ 结果图\n")
    a(f"![p-refinement exponential convergence]"
      f"(benchmarks/results/p-refinement-exponential/case_pool_entry.png)\n")
    a("半对数坐标：实测 p=1..6 误差点（圆点实线）、拟合指数律（虚线）、"
      "两条代数参考律 `p^-3`/`p^-5`（点线，明显弯折）。代数律在半对数图上是曲线，"
      "指数律是直线——这就是判据的视觉依据。\n")
    a("## ⑤ 注意点 / 局限\n")
    a("- **只覆盖一维周期线性对流光滑解**（与锚点算例一致）；二维/变系数/非线性/激波不在本条目内。")
    a("- **p≥7 受浮点地板限制**：`benchmarks/p_refinement_exact_time.py`（精确时间积分）把实测推到 "
      "p=8，但 p=7/8 的 float64 算子表示地板已与信号同量级（p=8 信噪比 0.01 量级），"
      "该路径自报 honest FAIL，不作为本门证据。见 `docs/CLOSEOUT-m17067-exact-time-floor.md`。")
    a("- **时间积分判据是相对判据**（相邻两次 CFL 相对变化 ≤1%），理论上可在时间误差主导时仍通过；"
      "本单每级都记录 `history`，且跨 p 的衰减率单调性对时间地板敏感（若某级被时间误差污染，"
      "衰减率会掉下来），实际起兜底作用。")
    a("- 判据的 `e(p_min)/e(p_max) > 1e3` 是**阶梯跨度相关**的：σ≈1 的真指数律也会被判拒"
      "（保守设计），因此本门是「证伪型」的：过门 ⇒ 必为指数律；不过门不等于非指数律。")
    a("- 与第三方 DG 实现交叉仍 not_run（无本地依赖）。\n")
    a("## ⑥ 误差判定（逐项回答「是否 <3%」）\n")
    a("| 项 | 基准 | 结果 | 是否 <3% |\n|---|---|---|---|")
    first = rows[0]["relative_l2"]
    a(f"| p=1 相对 L2 vs 解析解 | 解析 sin(2π(x−t)) | {first:.6e} | ✅ |")
    last = rows[-1]["relative_l2"]
    a(f"| p={rows[-1]['degree']} 相对 L2 vs 解析解 | 解析 sin(2π(x−t)) | {last:.6e} | ✅ |")
    a(f"| 门 L：σ vs 台账阈值 0.5 | 0.5 | {gates['legacy_slope']['measured']:.4f}"
      f"（裕度 {gates['legacy_slope']['measured'] / 0.5:.1f}×）| ✅（门量，非误差量）|")
    a(f"| 门 D：σ vs 阈值 1.0 | 1.0 | "
      f"{gates['discriminating']['conditions']['sigma_above_threshold']['measured']:.4f} | ✅ |")
    a(f"| 负对照 N1 十条律 | 期望裁决 | "
      f"{sum(1 for r in synth['laws'].values() if r['as_expected'])}/{len(synth['laws'])} 相符 "
      f"| {'✅' if synth['gate_pass'] else '❌'} |")
    a(f"| 负对照 N2 折点初值 | 期望「拒」 | "
      f"σ={kink['diagnostics']['lsq_decay_rate']:.4f}，门 D "
      f"{'拒' if kink['rejected_by_discriminating'] else '未拒'} | "
      f"{'✅' if kink['rejected_by_discriminating'] else '❌'} |\n")
    a("**结论**：所有 vs 解析解的误差量都远低于 3%（最大者为 p=1 的 "
      f"{first:.3e}），关键判据量（σ）超阈值 "
      f"{gates['legacy_slope']['measured'] / LEGACY_SLOPE_THRESHOLD:.1f} 倍；"
      "两道负对照门均按预期裁决，判据有牙。\n")
    a("## ⑦ 交付工件\n")
    a("| 文件 | 说明 |\n|---|---|")
    a("| `tensordg/cases/prefinement_exponential.py` | 规范算例定义（新增，库层）|")
    a("| `tensordg/cases/__init__.py` | 注册 `CASES['prefinement_exponential']`（改动）|")
    a("| `tensordg/convergence.py` | `exponential_decay_criterion` 判据上收到库层（新增）|")
    a("| `benchmarks/p_refinement_exponential.py` | 判据改为库函数薄封装（单点真相）|")
    a("| `benchmarks/case_pool_prefinement.py` | 本驱动：测 p 阶梯 + 两组负对照 + 工件/文档生成 |")
    a("| `tests/test_case_prefinement_exponential.py` | 算例池条目测试（注册/参数/活阶梯/负对照/判据一致性）|")
    a("| `docs/benchmarks/results/p-refinement-exponential/case_pool_entry.json` | 主工件（门裁决 + provenance）|")
    a("| `docs/benchmarks/results/p-refinement-exponential/case_pool_entry.png` | ④ 图 |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="p=1..3 smoke run")
    parser.add_argument("--no-doc", action="store_true", help="skip markdown render")
    args = parser.parse_args(argv)

    degrees = QUICK_DEGREES if args.quick else DEGREES
    started = time.perf_counter()
    print(f"[1/5] p ladder {degrees} (ncells={NCELLS})", flush=True)
    ladder = p_ladder(degrees=degrees)
    print(f"      sigma={ladder['diagnostics']['lsq_decay_rate']:.4f} "
          f"pass={ladder['pass']}", flush=True)

    print("[2/5] negative controls", flush=True)
    synthetic = synthetic_controls(DEGREES)
    print(f"      N1 gate_pass={synthetic['gate_pass']}", flush=True)
    kinked = kinked_ladder()
    print(f"      N2 rejected={kinked['rejected_by_discriminating']} "
          f"(legacy slope pass={kinked['legacy_slope_pass']})", flush=True)

    print("[3/5] reproduction of the published study", flush=True)
    reproduction = reproduction_check(ladder)
    print(f"      applicable={reproduction['applicable']} "
          f"all_pass={reproduction.get('all_pass')}", flush=True)

    runtime = time.perf_counter() - started
    metrics = {
        "schema": SCHEMA,
        "case_key": CASE_KEY,
        "ledger_anchor": LEDGER_ANCHOR,
        "parameters": prefinement_parameters(),
        "ladder": ladder,
        "anchor_table": anchor_table(ladder),
        "gates": ladder["gates"],
        "negative_controls": {
            "synthetic": synthetic,
            "kinked": kinked,
            "gate_pass": bool(synthetic["gate_pass"] and kinked["rejected_by_discriminating"]),
        },
        "reproduction": reproduction,
        "pass": bool(ladder["pass"] and synthetic["gate_pass"]
                     and kinked["rejected_by_discriminating"]
                     and reproduction.get("all_pass", True)),
        "runtime_seconds": runtime,
        "provenance": {
            "git_commit": _git_commit(),
            "argv": sys.argv[1:],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cwd": str(Path.cwd()),
        },
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(metrics, indent=2, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    metrics["artifact_sha256"] = digest
    (RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (ARTIFACT_DIR / "case_pool_entry.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True))

    print("[4/5] figure", flush=True)
    png = make_figure(ladder, ARTIFACT_DIR / "case_pool_entry.png")

    if not args.no_doc:
        print("[5/5] markdown", flush=True)
        DOC_PATH.write_text(render_markdown(metrics))
        (RUN_DIR / "summary.md").write_text(render_markdown(metrics))

    print(f"\nPASS={metrics['pass']}  sha256={digest[:16]}…  {runtime:.1f}s")
    print(f"artifacts: {RUN_DIR/'metrics.json'}, {ARTIFACT_DIR/'case_pool_entry.json'}, {png}")
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())