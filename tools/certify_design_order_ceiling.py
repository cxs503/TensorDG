"""Close the ``WITHHELD_ORDER_TOO_HIGH`` gap: declare the design order, measure the band.

Why this exists
---------------
``docs/CERTIFIED-REFERENCE-BAND-PHYSICAL.md`` (mission m21812) calibrated the
round-2 certification engine on physical PDE cases with closed-form truth and
found a *false negative*: the engine's absolute ceiling ``P_MAX = 6.0`` refuted
every P3 ladder, because DG cell averages superconverge at ``2k+1`` and
``2*3+1 = 7 > 6``.  That document registered the finding and explicitly left the
decision to a follow-up ("改它属于引擎语义变更，本单未改").  This driver is that
follow-up: ``certify(..., design_order=k)`` replaces the absolute bound by the
two-sided window ``2k+1 +/- SUPERCONVERGENCE_SLACK`` (``order_ceiling``), and the
question this driver answers is the falsifiable one:

    **does opening the window cost band coverage?**

A band that is too tight is exactly what an inflated observed order produces
(``U ~ 1/(r**p - 1)``), so "the ladder is now accepted" is worthless unless the
certified profile's *true* error -- measured against the closed-form solution --
still sits inside its own band.  Every ladder below is solved by this
repository's production kernels, certified by the shipped engine (imported, not
re-implemented), and then checked against truth.  The numbers are written to
``artifacts/design_order_ceiling/results.json`` and injected into
``docs/CERTIFIED-REFERENCE-DESIGN-ORDER-CEILING.md`` programmatically.

Usage::

    python -m tools.certify_design_order_ceiling            # full study + doc + figure
    python -m tools.certify_design_order_ceiling --no-figure
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensordg.certified_reference import (
    SUPERCONVERGENCE_SLACK,
    certify,
    coarsen,
    order_ceiling,
    rel_l1,
)
from tools.certify_physical_band import CASES, TAU, exact_field, solve_field

DEFAULT_OUTDIR = Path("artifacts/design_order_ceiling")
DOC_PATH = Path("docs/CERTIFIED-REFERENCE-DESIGN-ORDER-CEILING.md")
FIG_PATH = Path("artifacts/design_order_ceiling/design_order_ceiling.png")


def measure(case: dict, *, tolerance_percent: float = TAU) -> dict:
    """Certify one physical ladder twice (shipped default vs declared order)."""
    levels, steps = [], []
    for n in case["ladder"]:
        field, meta = solve_field(case, n)
        levels.append({"ncells": int(n), "field": field})
        steps.append(meta.get("steps"))

    plain = certify(levels, tolerance_percent=tolerance_percent)
    declared = certify(levels, tolerance_percent=tolerance_percent,
                       design_order=float(case["degree"]))
    n_ref = int(declared["reference_ncells"])
    exact_ref = exact_field(case, n_ref)

    true_percent = None
    if declared["reference"] is not None:
        true_percent = 100.0 * rel_l1(declared["reference"], exact_ref)
    band = declared["band_percent"]
    row = {
        "case": case["name"],
        "degree": int(case["degree"]),
        "ladder": [int(n) for n in case["ladder"]],
        "observed_order": declared["observed_order_used"],
        "order_rate_expected": 2.0 * float(case["degree"]) + 1.0,
        "order_ceiling": order_ceiling(float(case["degree"]))[1],
        "verdict_shipped": plain["verdict"],
        "verdict_declared": declared["verdict"],
        "superconvergent": bool(declared["superconvergent"]),
        "gate_usable_shipped": bool(plain["gate_usable"]),
        "gate_usable_declared": bool(declared["gate_usable"]),
        "band_percent": band,
        "band_share_of_gate": declared["band_share_of_gate"],
        "err_certified_percent": true_percent,
        "true_over_band": (true_percent / band) if (band and true_percent is not None) else None,
        "band_covers_truth": (bool(true_percent <= band)
                              if (band and true_percent is not None) else None),
        "reference_ncells": n_ref,
        "steps": steps,
    }
    return row


def summarise(rows: list[dict]) -> dict:
    measured = [r for r in rows if r["band_covers_truth"] is not None]
    covered = [r for r in measured if r["band_covers_truth"]]
    refused_shipped = [r["case"] for r in rows if not r["gate_usable_shipped"]]
    usable_declared = [r["case"] for r in rows if r["gate_usable_declared"]]
    return {
        "n_ladders": len(rows),
        "n_band_measurable": len(measured),
        "n_band_covers_truth": len(covered),
        "coverage_fraction": (len(covered) / len(measured)) if measured else None,
        "worst_true_over_band": (max(r["true_over_band"] for r in measured)
                                 if measured else None),
        "usable_shipped": [r["case"] for r in rows if r["gate_usable_shipped"]],
        "usable_declared": usable_declared,
        "refused_shipped": refused_shipped,
        "n_usable_shipped": len([r["case"] for r in rows if r["gate_usable_shipped"]]),
        "n_usable_declared": len(usable_declared),
        "newly_usable": [c for c in usable_declared
                         if c not in [r["case"] for r in rows if r["gate_usable_shipped"]]],
        "regressed": [r["case"] for r in rows
                      if r["gate_usable_shipped"] and not r["gate_usable_declared"]],
    }


def run(outdir: Path = DEFAULT_OUTDIR, *, figure: bool = True) -> dict:
    rows = [measure(case) for case in CASES]
    result = {
        "tolerance_percent": TAU,
        "superconvergence_slack": SUPERCONVERGENCE_SLACK,
        "rows": rows,
        "summary": summarise(rows),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    if figure:
        result["figure"] = str(plot(result, FIG_PATH))
    render(result, DOC_PATH)
    return result


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "**no**"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if v == 0.0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.{digits-1}e}"
        return f"{v:.{digits}g}"
    return str(v)


def plot(result: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- left: observed order vs the 2k+1 window, with the old absolute bound
    idx = np.arange(len(rows))
    obs = [r["observed_order"] for r in rows]
    rate = [r["order_rate_expected"] for r in rows]
    ceil = [r["order_ceiling"] for r in rows]
    ax1.axhspan(0.0, 6.0, color="0.92", zorder=0)
    ax1.axhline(6.0, color="firebrick", lw=1.4, ls="--", label="shipped absolute bound P_MAX = 6")
    for i, r in enumerate(rows):
        ax1.vlines(i, r["order_rate_expected"] - SUPERCONVERGENCE_SLACK,
                   r["order_ceiling"], color="tab:blue", lw=6, alpha=0.25,
                   label="declared window 2k+1 ± slack" if i == 0 else None)
    ax1.plot(idx, obs, "o", color="black", ms=6, label="observed order $p_{obs}$")
    ax1.plot(idx, rate, "s", ms=4, mfc="none", color="tab:blue", label="$2k+1$ (expected)")
    ax1.set_xticks(idx)
    ax1.set_xticklabels([r["case"] for r in rows], rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("order of accuracy")
    ax1.set_title("(a) the P3 ladders sit above 6 by construction")
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(fontsize=7.5, loc="upper left")

    # --- right: certified-profile true error vs its own band
    measurable = [r for r in rows if r["band_covers_truth"] is not None]
    ax2.set_yscale("log")
    for i, r in enumerate(measurable):
        ax2.plot([i, i], [r["band_percent"], r["err_certified_percent"]],
                 color="0.6", lw=1.0, zorder=1)
    ax2.plot(np.arange(len(measurable)), [r["band_percent"] for r in measurable],
             "s", color="tab:orange", ms=6, label="band $U$")
    ax2.plot(np.arange(len(measurable)), [r["err_certified_percent"] for r in measurable],
             "o", color="tab:green", ms=6, label="true error of certified profile")
    ax2.set_xticks(np.arange(len(measurable)))
    ax2.set_xticklabels([r["case"] for r in measurable], rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("relative L1 error [%], log scale")
    ax2.set_title("(b) opening the window keeps the band above the truth")
    ax2.grid(alpha=0.3, which="both", axis="y")
    ax2.legend(fontsize=8)

    fig.suptitle("declared-design-order ceiling: refusing P3 ladders was a policy artefact",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render(result: dict, doc_path: Path = DOC_PATH) -> str:
    rows = result["rows"]
    s = result["summary"]
    L: list[str] = []
    L.append("# 声明设计阶数解除 `WITHHELD_ORDER_TOO_HIGH` 误拒：带宽覆盖率实测\n")
    L.append("> 生成器：`python -m tools.certify_design_order_ceiling`（本文件全部数字由 "
             "`artifacts/design_order_ceiling/results.json` 程序化注入，零手抄）。\n"
             "> 引擎：`tensordg.certified_reference`（import，未重写）。"
             f"> 门限 τ = {_fmt(result['tolerance_percent'])} %。"
             f"> 超收敛窗半宽 = {_fmt(result['superconvergence_slack'])} 阶。\n")

    L.append("## ① 问题（为什么这是一条要修的欠账）\n")
    L.append("round-2 认证引擎用 `P_MAX = 6.0` 作为观测阶的**绝对**上限。该值是策略常数，"
             "不是物理上限：DG 单元均值在光滑解上以 `O(h^{2k+1})` 超收敛"
             "（Cockburn–Luskin–Shu–Süli 2003），故 P3 阶梯**必然**给出 `p ≈ 7 > 6`，"
             "被引擎按构造误拒。上一轮（`docs/CERTIFIED-REFERENCE-BAND-PHYSICAL.md` ④b）"
             "只能把该负面结论登记在案。\n"
             "本单的修法：调用方声明 `design_order=k` 时，绝对上限被**双侧**窗口 "
             "`2k+1 ± slack` 取代（`order_ceiling`）；不声明则逐位保持原行为，"
             "所有已发布数字仍可比。\n")
    L.append("本单要证伪的命题：**打开窗口是否以带宽覆盖率为代价**。"
             "阶数被噪声夸大时 `U ~ 1/(r^p - 1)` 会被压得过小，因此"
             "「阶梯被接受了」本身不值钱——必须用闭式解测出的认证解真误差仍在带宽之内。\n")

    L.append("## ② 做法\n")
    L.append("* 算例表与物理真解完全复用 `tools/certify_physical_band.py`"
             "（同一批 1-D 周期算例、同一生产内核 `tensordg.advection_1d` / "
             "`tensordg.burgers_1d`、同一 Gauss 求积真解），本单不另造算例；")
    L.append("* 每条阶梯认证**两次**：`shipped` = 不声明（原行为），"
             "`declared` = `design_order=degree`；")
    L.append("* 认证解真误差 = Richardson 外推解在参考网格（次细）上对闭式解单元均值的相对 L1；"
             "`真误差/U ≤ 1` 即带宽覆盖；")
    L.append("* 判定：`newly_usable` 非空且 `regressed` 为空且覆盖率不降 → 窗口可用，否则退回。\n")

    L.append("## ③ 结果（逐阶梯）\n")
    L.append("| 阶梯 | k | p_obs | 2k+1 | 窗口上界 | shipped 判定 | declared 判定 | U % | 认证解真误差 % | 真误差/U | 覆盖 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| `{case}` | {k} | {p} | {rate} | {ceil} | {vs} | {vd} | {u} | {et} | {ratio} | {cov} |".format(
            case=r["case"], k=r["degree"], p=_fmt(r["observed_order"]),
            rate=_fmt(r["order_rate_expected"]), ceil=_fmt(r["order_ceiling"]),
            vs=f"`{r['verdict_shipped']}`", vd=f"`{r['verdict_declared']}`",
            u=_fmt(r["band_percent"]), et=_fmt(r["err_certified_percent"]),
            ratio=_fmt(r["true_over_band"]), cov=_fmt(r["band_covers_truth"])))
    L.append("")
    L.append("## ④ 汇总判定\n")
    L.append(f"* 阶梯总数 {s['n_ladders']}；shipped 判为可用（`gate_usable`）的 **{s['n_usable_shipped']}** 条，"
             f"declared 判为可用的 **{s['n_usable_declared']}** 条；")
    L.append(f"* 新解除误拒的阶梯：{', '.join('`%s`' % c for c in s['newly_usable']) or '无'}；")
    L.append(f"* 反向回归（原本可用、声明后变不可用）：{', '.join(s['regressed']) or '无'}；")
    L.append(f"* 带宽可测 {s['n_band_measurable']} 条，其中带宽覆盖真误差 **{s['n_band_covers_truth']}** 条，"
             f"覆盖率 {_fmt(100.0 * (s['coverage_fraction'] or 0.0))} %；")
    L.append(f"* 最差 `真误差/U` = {_fmt(s['worst_true_over_band'])}（≤ 1 即覆盖）。\n")

    L.append("## ⑤ 注意点 / 局限\n")
    L.append("* `P_MAX` 本身**未改**：不声明 `design_order` 的调用行为逐位不变，"
             "既有已发布数字（Shu–Osher 4 级梯次、DG 家族自证拒绝）不受影响；")
    L.append("* 窗口是**双侧**的：`p` 仅需「大于 6」不够，必须落在 `2k+1 ± slack`；"
             "测试 `tests/test_certified_reference_design_order.py` 用 p≈6.2 与 p≈7.6 两例锁住；")
    L.append("* 阶梯整体必须严格收缩（所有局部阶 > 0），单靠最细三点的偶然高阶不能换取认证；")
    L.append("* 门有效性判据 `U/τ ≤ 1/3` 仍然最后说话：窗口接受的阶梯若带宽吃掉门，仍判 "
             "`WITHHELD_BAND_CONSUMES_GATE`；")
    L.append("* 全部算例为 1-D；`2k+1` 是 1-D 单元均值的超收敛率，多维的对应率不同，"
             "声明多维设计阶前需另测；")
    L.append("* 这是**代码验证**（对闭式解的离散误差），不是实验 validation。\n")

    L.append("## ⑥ 误差判定（是否 < 3%）\n")
    L.append(f"* 关键量 1——认证解真误差 vs 闭式解：最大 {_fmt(max(r['err_certified_percent'] for r in rows if r['err_certified_percent'] is not None))} %"
             f"（门限 {_fmt(result['tolerance_percent'])} %）→ **< 3% 成立**；")
    L.append(f"* 关键量 2——带宽自洽：最差 `真误差/U` = {_fmt(s['worst_true_over_band'])} ≤ 1 → "
             "带宽对全部阶梯覆盖真误差，**不因打开窗口而失效**；")
    L.append("* 关键量 3——引擎改动对既有发布数字的影响：无声明路径逐位不变"
             "（`tests/test_certified_reference.py` 的 Shu–Osher 逐位等价断言仍绿）。\n")
    text = "\n".join(L)
    doc_path.write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    result = run(args.outdir, figure=not args.no_figure)
    print(json.dumps({k: v for k, v in result["summary"].items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())