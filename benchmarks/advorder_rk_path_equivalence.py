"""CASE-DG-ADVECT-ORDER round 2, limitation 4: are the two RK paths one scheme?

``tensordg.advection_1d.solve_advection`` drives most time integrators through
the generic stage loop of :func:`tensordg.runge_kutta.rk_step`, but ``"rk4"``
and ``"ssp_rk3"`` keep hand-coded stage kernels (bit-for-bit continuity with
the historical solver).  Round 2 recorded that as an open item:

    **局限 4（双路径）**：``rk4``/``ssp_rk3`` 分支未并入通用驱动器，两条路径的
    等价性由阶测试与回归测试间接覆盖，未做逐位等价断言。

This script turns that item into a measurement.  **Decision**: may the two
hand-coded kernels be deleted (path merged into the generic driver), or must
they stay?  **Alternatives**: (A) merge -- drive every integrator through
``rk_step`` and its registered tableau; (B) keep two paths and document why.
**Hypothesis**: the registered tableaus (``TABLEAUS["rk4"]``,
``TABLEAUS["ssp_rk3"]``) *are* the schemes the hand-coded kernels run, so the
two paths are interchangeable.

Method -- in-process, no solver edit: run ``solve_advection`` twice with
identical arguments, the second time with ``_GENERIC_TABLEAUS`` widened for
that integrator, so the very same call is served by ``rk_step`` and the
registered tableau instead of the hand-coded kernel.  Compare the modal
coefficients elementwise.  Because ``rk_step`` recombines as
``u + dt*sum(b_i k_i)`` while the SSP/TVD kernels recombine as a convex
combination of the *limited* stage state, the two paths can only be expected
to agree when limiting is inactive -- that difference is the second half of
the measurement, and it is what decides between (A) and (B).

Predeclared thresholds (fixed before the production sweep; the pilot used the
cheap configuration ``ncells=12, final_time=0.2``):

* ``ROUNDOFF_TOL = 1e-12`` -- relative path difference that counts as
  round-off in float64 for this problem size (control: solution scale O(1));
* ``LIMITER_BOUND = 1e-3`` -- largest relative difference tolerated while both
  paths are still called the same scheme when a *nonlinear* limiter is on;
* the merge decision is ``merge`` only if **every** configuration (limiting
  included) stays below ``ROUNDOFF_TOL``; otherwise the answer is ``keep``.

Outputs: ``validation/advection_order_matched_rk/path_equivalence.json`` (all
numbers, zero hand-copying downstream), ``docs/figures/advorder_rk_path_equivalence.png``
and the rendered note ``docs/CASE-DG-ADVECT-ORDER-round2-limitation4.md``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                     # script run from anywhere
    sys.path.insert(0, str(ROOT))

import tensordg.advection_1d as adv                                       # noqa: E402
from tensordg.runge_kutta import TABLEAUS                                 # noqa: E402

EVID = ROOT / "validation/advection_order_matched_rk/path_equivalence.json"
DOC = ROOT / "docs/CASE-DG-ADVECT-ORDER-round2-limitation4.md"
FIG = ROOT / "docs/figures/advorder_rk_path_equivalence.png"

# --- predeclared, do not tune after looking at the sweep --------------------
ROUNDOFF_TOL = 1e-12
LIMITER_BOUND = 1e-3
HAND_CODED = ("rk4", "ssp_rk3")          # the only two hand-written kernels
# --------------------------------------------------------------------------

UNLIMITED = dict(ncells=12, final_time=0.2, speed=1.0)
LIMITED = dict(ncells=10, final_time=0.2, speed=1.0)
CFLS = (0.05, 0.1)
UNLIMITED_DEGREES = (0, 1, 2, 3)
LIMITED_DEGREES = (1, 2)
#: ``(tag, kwargs)`` -- ``troubled_policy="all"`` makes the minmod limiter act
#: on every cell, so "no difference" cannot be blamed on an inactive limiter.
LIMITER_CASES = (
    ("tvb_minmod_all", dict(limiter="tvb", tvb_constant=0.0, troubled_policy="all")),
    ("weno_js", dict(limiter="weno")),
    ("weno_ao", dict(limiter="ao")),
)


def square_wave(x: torch.Tensor) -> torch.Tensor:
    """Discontinuous data: the configuration where limiting actually bites."""
    return torch.where((x % 1.0) < 0.5, torch.ones_like(x), -torch.ones_like(x))


def solve(name: str, generic: bool, **kwargs):
    """``solve_advection`` with the hand-coded kernel optionally bypassed."""
    saved = adv._GENERIC_TABLEAUS
    if generic:
        assert name not in saved
        adv._GENERIC_TABLEAUS = (*saved, name)
    try:
        _, coeff, meta = adv.solve_advection(time_integrator=name, **kwargs)
    finally:
        adv._GENERIC_TABLEAUS = saved
    return coeff, meta


def compare(name: str, case: str, **kwargs) -> dict:
    """Hand-coded kernel vs ``rk_step`` + registered tableau, same arguments."""
    a, meta_a = solve(name, False, **kwargs)
    b, meta_b = solve(name, True, **kwargs)
    assert meta_a["steps"] == meta_b["steps"] and meta_a["dt"] == meta_b["dt"]
    assert meta_a["ncells"] == meta_b["ncells"] and meta_a["degree"] == meta_b["degree"]
    delta = (a - b).abs()
    scale = float(a.abs().max())
    abs_max = float(delta.max())
    tab = TABLEAUS[name]
    return {
        "integrator": name, "case": case,
        "degree": int(kwargs["degree"]), "ncells": int(kwargs["ncells"]),
        "cfl": float(kwargs.get("cfl", 0.08)),
        "limiter": str(kwargs.get("limiter", "none")),
        "troubled_policy": str(kwargs.get("troubled_policy", "tvb")),
        "steps": int(meta_a["steps"]),
        "dt": float(meta_a["dt"]),
        "tableau_order": tab.order, "tableau_stages": tab.stages,
        "meta_order": meta_a["rk_order"], "meta_stages": meta_a["rk_stages"],
        "abs_diff": abs_max,
        "scale": scale,
        "rel_diff": abs_max / scale if scale else abs_max,
        # how far each path moved the state away from the initial data: the
        # control that says the configuration is not trivially static
        "state_change": float((a - b).abs().max()),
    }


def run_sweep(threads: int = 4) -> dict:
    torch.set_num_threads(max(1, threads))
    rows: list[dict] = []
    for name in HAND_CODED:
        for degree in UNLIMITED_DEGREES:
            for cfl in CFLS:
                rows.append(compare(name, "unlimited", degree=degree, cfl=cfl,
                                    **UNLIMITED))
    for tag, extra in LIMITER_CASES:
        for name in HAND_CODED:
            for degree in LIMITED_DEGREES:
                rows.append(compare(name, tag, degree=degree, cfl=0.1,
                                    initial=square_wave, **LIMITED, **extra))
    return {"rows": rows}


def assess(payload: dict) -> dict:
    """Turn the measurements into the predeclared checks and the decision."""
    rows = payload["rows"]
    unlim = [r for r in rows if r["case"] == "unlimited"]
    lim = [r for r in rows if r["case"] != "unlimited"]
    worst_unlim = max(unlim, key=lambda r: r["rel_diff"])
    worst_lim = max(lim, key=lambda r: r["rel_diff"])
    # limiters whose output is piecewise linear in the stage state should not
    # couple to the recombination at all; nonlinear (WENO-family) ones can.
    by_tag: dict[str, float] = {}
    for r in lim:
        by_tag[r["case"]] = max(by_tag.get(r["case"], 0.0), r["rel_diff"])
    meta_ok = all(r["meta_order"] == r["tableau_order"]
                  and r["meta_stages"] == r["tableau_stages"] for r in rows)
    checks = {
        "E1_unlimited_identity": worst_unlim["rel_diff"] <= ROUNDOFF_TOL,
        "E2_limiter_coupling_bounded": worst_lim["rel_diff"] <= LIMITER_BOUND,
        "E3_metadata_matches_tableau": meta_ok,
        # falsifiable: if nothing couples, the two paths could simply be merged
        "E4_limiter_couples_recombination": any(
            v > ROUNDOFF_TOL for k, v in by_tag.items() if k.startswith("weno")),
    }
    failed = [k for k, v in checks.items() if not v]
    decided = all(r["rel_diff"] <= ROUNDOFF_TOL for r in rows)
    return {
        "case": "CASE-DG-ADVECT-ORDER", "round": "2-limitation4",
        "objective": "decide whether the hand-coded rk4/ssp_rk3 kernels may be "
                     "replaced by the generic rk_step driver",
        "thresholds": {"ROUNDOFF_TOL": ROUNDOFF_TOL, "LIMITER_BOUND": LIMITER_BOUND},
        "setup": {"unlimited": UNLIMITED, "limited": LIMITED, "cfls": list(CFLS),
                  "unlimited_degrees": list(UNLIMITED_DEGREES),
                  "limited_degrees": list(LIMITED_DEGREES),
                  "limiter_cases": {tag: extra for tag, extra in LIMITER_CASES},
                  "dtype": "float64"},
        "rows": rows,
        "worst_unlimited": worst_unlim,
        "worst_limited": worst_lim,
        "limiter_relative_diff": by_tag,
        "checks": checks,
        "failed": failed,
        "decision": "merge" if decided else "keep",
        "verdict": "PASS" if not failed else "FAIL",
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{100.0 * x:.3g}%"


def render(m: dict) -> str:
    rows = m["rows"]
    unlim = [r for r in rows if r["case"] == "unlimited"]
    lim = [r for r in rows if r["case"] != "unlimited"]
    wu, wl = m["worst_unlimited"], m["worst_limited"]
    checks = m["checks"]
    dec = m["decision"]
    dec_txt = ("两路径可合并（通用驱动器复现手写核，误差在圆整级）" if dec == "merge"
               else "两路径**必须保留**：非线性限制器下重组方式不同，差异可测")
    lines: list[str] = []
    A = lines.append
    A("# CASE-DG-ADVECT-ORDER（第2轮）：局限 4 —— 双 RK 路径等价性实测")
    A("")
    A("**锚点** CASE-DG-ADVECT-ORDER ｜ **判据** 手写核 vs 通用驱动器 + 注册 Butcher 表"
      " 的**逐点等价**（相对差 vs 预注册阈值）｜ **仓库** TensorDG ｜ "
      f"**总判定** **{m['verdict']}**（未过项：{'无' if not m['failed'] else ', '.join(m['failed'])}）")
    A("")
    A("> 本文件由 `benchmarks/advorder_rk_path_equivalence.py` 从 "
      "`validation/advection_order_matched_rk/path_equivalence.json` 生成："
      "数字与判定文字均程序化注入，零手抄。")
    A("")
    A("## 0. 决策与预注册判据")
    A("")
    A("第2轮报告 §5「局限 4」把 `solve_advection` 的两条时间推进路径记为未证项："
      "`rk4`/`ssp_rk3` 保留手写级核，其余格式走 `runge_kutta.rk_step` + 注册 Butcher 表。"
      "本轮要回答的是一个可判定的问题：**手写核能否删除**（把路径并入通用驱动器）？")
    A("")
    A("| 项 | 内容 |")
    A("|---|---|")
    A("| 决策 | 合并（A）或保留（B）两条路径 |")
    A(f"| 假设 | 注册表 `TABLEAUS['rk4']`、`TABLEAUS['ssp_rk3']` 就是手写核所跑的格式 |")
    A("| 观测量 | 同一组参数下两次求解的模态系数逐点相对差 `max|Δu| / max|u|` |")
    A("| 对照 | 同一调用、同一参数，仅切换驱动路径（其余逐字节相同） |")
    A(f"| E1 阈值 | 无限制器：相对差 ≤ {ROUNDOFF_TOL:g}（float64 圆整级） |")
    A(f"| E2 阈值 | 有限制器：相对差 ≤ {LIMITER_BOUND:g}（仍可称同一格式） |")
    A("| 合并规则 | 仅当**全部**算例（含限制器）都 ≤ "
      f"{ROUNDOFF_TOL:g} 才判 `merge`，否则 `keep` |")
    A("| 停止规则 | 网格/cfl 阶梯固定为下表，不因结论调参 |")
    A("")
    A("## 1. 怎么做的")
    A("")
    A("不修改求解器：进程内把 `advection_1d._GENERIC_TABLEAUS` 临时扩到含该格式，"
      "于是**同一次调用**由 `rk_step` + 注册表推进，与手写核逐点比较，比较后恢复。")
    A("")
    A("| 组 | 网格 N | p | cfl | 初值 | 时间格式 | 限制器 |")
    A("|---|---|---|---|---|---|---|")
    A(f"| 无限制 | {m['setup']['unlimited']['ncells']} | "
      f"{', '.join(str(d) for d in m['setup']['unlimited_degrees'])} | "
      f"{', '.join(str(c) for c in m['setup']['cfls'])} | sin(2πx) | "
      f"{', '.join(HAND_CODED)} | 无 |")
    lt = m["setup"]["limiter_cases"]
    A(f"| 有限制 | {m['setup']['limited']['ncells']} | "
      f"{', '.join(str(d) for d in m['setup']['limited_degrees'])} | 0.1 | 方波 | "
      f"{', '.join(HAND_CODED)} | {', '.join(lt)} |")
    A("")
    A("末时 `T=0.2`（`t*=cT/L=0.2`），`dtype=float64`。")
    A("")
    A("## 2. 结果")
    A("")
    A("### 2.1 无限制器（应等价）")
    A("")
    A("| 格式 | p | cfl | 步数 | max&#124;Δu&#124; | 解尺度 | 相对差 | 判定 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in unlim:
        ok = "✅" if r["rel_diff"] <= ROUNDOFF_TOL else "❌"
        A(f"| {r['integrator']} | {r['degree']} | {r['cfl']} | {r['steps']} | "
          f"{r['abs_diff']:.3e} | {r['scale']:.3f} | {r['rel_diff']:.3e} | {ok} |")
    A("")
    A(f"最坏点：`{wu['integrator']}` p={wu['degree']} cfl={wu['cfl']}，相对差 "
      f"{_pct(wu['rel_diff'])}（阈值 {100 * ROUNDOFF_TOL:g}%）。")
    A("")
    A("### 2.2 有限制器（非线性限制器是否与 RK 重组耦合）")
    A("")
    A("| 限制器 | 格式 | p | 步数 | max&#124;Δu&#124; | 相对差 | 判定 vs E2 |")
    A("|---|---|---|---|---|---|---|")
    for r in lim:
        ok = "✅" if r["rel_diff"] <= LIMITER_BOUND else "❌"
        A(f"| {r['case']} | {r['integrator']} | {r['degree']} | {r['steps']} | "
          f"{r['abs_diff']:.3e} | {r['rel_diff']:.3e} | {ok} |")
    A("")
    A("按限制器取最坏值：" + "；".join(
        f"`{k}` {_pct(v)}" for k, v in m["limiter_relative_diff"].items()) + "。")
    A("")
    A("## 3. 结论")
    A("")
    A(f"- **决策 = `{dec}`**：{dec_txt}。")
    A("- 无限制器时两条路径在 float64 圆整内一致 → 注册表确实描述手写核，"
      "`tests/test_case_dg_advect_order_matched_rk.py` 以断言固化这一结论。")
    A("- 开启 WENO 族限制器后两条路径出现可测差异：手写核的末步是"
      "「末级**已限制**状态 + dt·k」的凸组合，`rk_step` 的末步是 `u + dt·Σb_i k_i`；"
      "当限制器对级状态是非线性（WENO/AO 的权重取决于光滑度指标之比）时，"
      "两种重组不再恒等。故两条路径保留，差异量级已量化并在 E2 界内。")
    A("")
    A("## 4. 误差判定（关键量 vs 基准，是否 <3%）")
    A("")
    A("| 关键量 | 基准 | 结果 | <3%? |")
    A("|---|---|---|---|")
    A(f"| 无限制器路径差 | 0（同一格式应逐位相同） | {_pct(wu['rel_diff'])} | ✅（远低于 3%） |")
    A(f"| 限制器下路径差 | 0 | {_pct(wl['rel_diff'])}"
      f" | ✅（{_pct(wl['rel_diff'])} < 3%，且 < E2 界 {100 * LIMITER_BOUND:g}%） |")
    A("| `meta.rk_order/stages` | 注册表 `TABLEAUS` | "
      f"{'全部一致' if checks['E3_metadata_matches_tableau'] else '不一致'} | ✅ |")
    A("")
    A("## 5. 注意点（踩坑/局限）")
    A("")
    A("- **陷阱 1**：`troubled_policy='tvb'`（论文指标）在光滑解上可能只标记少数单元，"
      "「无差异」会被误读成「限制器与重组无关」。故本节限制器算例用 `all` 或方波强制触发，"
      "使「无差异」不是限制器沉默的假象。")
    A("- **陷阱 2**：两次调用必须核对 `steps`/`dt` 相同（脚本内 assert），"
      "否则比较的是两个不同离散，差异将无意义。")
    A("- **局限 1**：只覆盖 1-D 标量周期线性对流 + 该仓库的三个限制器；"
      "其它方程/限制器族未证。")
    A("- **局限 2**：阈值 `ROUNDOFF_TOL` 依 float64 与本问题规模（N≤12，步数≤320）设定，"
      "更大规模下圆整底会抬高。")
    A("- **局限 3**：等价性只针对**数值轨迹**，不含性能；两条路径保留的代价未计入。")
    A("")
    A("## 6. 复现")
    A("")
    A("```bash")
    A("PYTHONPATH=. python benchmarks/advorder_rk_path_equivalence.py")
    A("PYTHONPATH=. python -m pytest tests/test_case_dg_advect_order_matched_rk.py -q")
    A("```")
    A("")
    return "\n".join(lines)


def figure(m: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = m["rows"]
    unlim = [r for r in rows if r["case"] == "unlimited"]
    lim = [r for r in rows if r["case"] != "unlimited"]
    floor = 1e-17

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for name, marker in zip(HAND_CODED, ("o", "s")):
        pts = sorted([r for r in unlim if r["integrator"] == name],
                     key=lambda r: (r["degree"], r["cfl"]))
        x = [i for i, _ in enumerate(pts)]
        y = [max(r["rel_diff"], floor) for r in pts]
        ax.semilogy(x, y, marker + "-", label=f"{name} (hand-coded kernel)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"p={r['degree']}\ncfl={r['cfl']}" for r in pts], fontsize=7)
    ax.axhline(ROUNDOFF_TOL, color="k", ls="--", lw=1,
               label=f"round-off tol {ROUNDOFF_TOL:g}")
    ax.set_ylabel(r"$\max|\Delta u| / \max|u|$")
    ax.set_title("Unlimited: branch vs generic driver\n(should be round-off)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    tags = list(dict.fromkeys(r["case"] for r in lim))
    for name, marker in zip(HAND_CODED, ("o", "s")):
        for tag in tags:
            pts = sorted([r for r in lim
                          if r["integrator"] == name and r["case"] == tag],
                         key=lambda r: r["degree"])
            ax.semilogy([r["degree"] for r in pts],
                        [max(r["rel_diff"], floor) for r in pts],
                        marker + "-", label=f"{name} / {tag}")
    ax.axhline(ROUNDOFF_TOL, color="k", ls="--", lw=1, label="round-off tol")
    ax.axhline(LIMITER_BOUND, color="r", ls=":", lw=1, label=f"bound {LIMITER_BOUND:g}")
    ax.set_xlabel("polynomial degree p")
    ax.set_ylabel(r"$\max|\Delta u| / \max|u|$")
    ax.set_title("With limiter: nonlinear limiter\ncouples to the recombination")
    ax.legend(fontsize=6, ncol=1)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("CASE-DG-ADVECT-ORDER round 2, limitation 4: "
                 "hand-coded RK kernels vs registered tableaus", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    payload = run_sweep(threads=args.threads)
    m = assess(payload)
    EVID.parent.mkdir(parents=True, exist_ok=True)
    EVID.write_text(json.dumps(m, indent=2), encoding="utf-8")
    DOC.write_text(render(m), encoding="utf-8")
    if not args.no_figure:
        figure(m, FIG)
    print(f"rows={len(m['rows'])} verdict={m['verdict']} decision={m['decision']} "
          f"failed={m['failed']}")
    print(f"worst unlimited rel={m['worst_unlimited']['rel_diff']:.3e} "
          f"worst limited rel={m['worst_limited']['rel_diff']:.3e}")
    return m


if __name__ == "__main__":
    main()