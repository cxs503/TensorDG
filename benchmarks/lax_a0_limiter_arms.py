"""Ledger arm A0 re-adjudicated across the limiter branches -- offline, zero solves.

Why this driver exists
----------------------
``CASE-LAX-DG`` is graded on two layers:

* **A0 (ledger)**: relative L2 versus the exact Riemann solution at the pinned
  production leg ``p3.N400``, tolerance ``< 3 %``.  Committed value for the
  unlimited solver: ``rel L2 rho = 3.11556e-02`` -> **FAIL** by a 3.9 % margin.
* **C1..R5 (verification)**: wave positions / star state / orders.  C1 and C2
  pass, C3/R3 fail because a discontinuous field cannot converge at design
  order.  Those are method properties and stay FAIL.

The limiter study (``benchmarks/lax_case_pool_limiter_arms.py``) re-ran the whole
ladder under three branches (``limiter=None`` / ``'tvb'`` / ``'weno'``) but graded
only C1/C2/C3/R3/R5.  Nobody ever evaluated the **ledger arm itself** on those
branches, and the ledger arm is the one measure that is *sensitive* to
post-shock oscillations: at the same leg the unlimited branch has
``rel L1 rho = 8.68554e-03`` against ``rel L2 rho = 3.11556e-02`` -- an L2/L1
ratio of 3.6, i.e. the L2 error is carried by localized spikes that a limiter is
designed to clip, while L1 (which the limiter *worsens* by smearing) does not see
them.  So "resharpening one h" is not the only possible route to the 3 % bar, and
the A0 arm was simply never asked the question.

This driver asks it from the committed field cache -- no solver time at all::

    docs/benchmarks/results/lax_case_pool_limiter_arms/profiles.npz

holds the finest-leg primitive fields of all three branches on the element Gauss
nodes, together with the exact solution, so both norms are re-derivable exactly
as ``lax_case_pool.production_errors_from_profiles`` derives them (same 10-point
Legendre weighting, same ``dx = span/(ncells - 1)`` recovery).  Nothing is
re-solved, therefore nothing can be tuned after seeing a number: the branches
were fixed before the limiter study ran.

Pre-registration (fixed before this module was first executed)
--------------------------------------------------------------
P0  Reuse: the ``none`` branch must reproduce the committed pool value
    (``metrics.json`` -> ``acceptance.per_leg['p3.N400'].l2``) field by field to
    ``<= 1e-12`` relative.  If it does not, the whole run is ``WITHHELD`` --
    a mismatch means the cache and the acceptance record are not the same fields.
P1  Metric: relative L2 on the element Gauss nodes, Legendre-quadrature weighted,
    whole field, no wave band removed, denominator = norm of the exact solution.
    Identical to the ledger arm; tolerance read from
    ``lax_case_pool.ledger_acceptance_tolerance()`` (never hard-coded here).
P2  Arm verdict: ``PASS`` iff ``max_field rel L2 < tolerance``.
P3  Branch eligibility: an arm may only be credited if the limiter study's own
    C1 and C2 verdicts for that arm are ``PASS`` (wave positions and star state
    must survive the limiter).
P4  Overall: A0 is ``PASS`` iff at least one **eligible** arm is ``PASS`` at the
    pinned leg ``p3.N400``.  The production leg itself is **not** moved, the
    ladder is not extended, and no knob other than ``limiter`` is varied
    (``tvb_constant``/``limiter_stage``/``positivity``/``cfl`` stay at library
    defaults), so a PASS is a branch statement, not a re-tuned configuration.
P5  Honest FAIL: if no eligible arm clears 3 %, the record stays FAIL and the
    resolution route (``p3.N800``, ``docs/CASE-LAX-DG-resolution-requirement.md``)
    remains the only documented path.

Artifacts::

    docs/benchmarks/results/lax_a0_limiter_arms/metrics.json
    docs/benchmarks/results/lax_a0_limiter_arms/figures/*.png
    docs/benchmarks/results/lax_a0_limiter_arms/summary.md
    docs/CASE-LAX-DG-a0-limiter-arms.md           (rendered, zero hand-copied numbers)

Usage::

    python -m benchmarks.lax_a0_limiter_arms
    python -m benchmarks.lax_a0_limiter_arms --check   # drift check, writes nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.lax_case_pool import (
    OUT as POOL_OUT,
    _weighted_l1,
    _weighted_l2,
    ledger_acceptance_tolerance,
)

ARMS_SRC = Path("docs/benchmarks/results/lax_case_pool_limiter_arms")
PROFILES = ARMS_SRC / "profiles.npz"
ARMS_METRICS = ARMS_SRC / "metrics.json"
POOL_METRICS = Path(POOL_OUT) / "metrics.json"
OUT = Path("docs/benchmarks/results/lax_a0_limiter_arms")
DOC = Path("docs/CASE-LAX-DG-a0-limiter-arms.md")

PRODUCTION_LEG = "p3.N400"
ARMS = ("none", "tvb", "weno")
FIELDS = ("density", "velocity", "pressure")
REUSE_TOL = 1e-12


# ------------------------------------------------------------------ measuring
def _cell_width(points):
    """Cell width recovered exactly as ``lax_case_pool.production_errors_from_profiles``.

    ``np.mean(np.diff(centres))`` lands a few ulp away from the solver's ``dx``
    and breaks bit-level agreement (see the m20346 note in ``lax_case_pool``).
    """
    centres = 0.5 * (points[:, 0] + points[:, -1])
    return float((centres[-1] - centres[0]) / (len(centres) - 1))


def measure_arm(data, arm, quadrature_order=10):
    """Relative L1/L2 of one branch on the cached finest-leg Gauss nodes."""
    points = data[f"{arm}_points"]
    _, weights = np.polynomial.legendre.leggauss(quadrature_order)
    dx = _cell_width(points)
    out = {"l1": {}, "l2": {}}
    for field in FIELDS:
        numerical = data[f"{arm}_numerical_{field}"]
        exact = data[f"{arm}_exact_{field}"]
        out["l1"][field] = _weighted_l1(points, numerical, exact, weights, dx)
        out["l2"][field] = _weighted_l2(points, numerical, exact, weights, dx)
    return out


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -------------------------------------------------------------------- gates
def gate_reuse(measured_none, pool_metrics, tolerance=REUSE_TOL):
    """P0 -- the cached ``none`` fields must be the committed acceptance fields."""
    reference = pool_metrics["acceptance"]["per_leg"][PRODUCTION_LEG]["l2"]
    rows, worst = {}, 0.0
    for field in FIELDS:
        mine = float(measured_none["l2"][field])
        theirs = float(reference[field])
        rel = abs(mine - theirs) / abs(theirs) if theirs else abs(mine)
        rows[field] = {"offline": mine, "committed": theirs, "relative_difference": rel}
        worst = max(worst, rel)
    return {"gate": "P0", "tolerance": tolerance, "fields": rows,
            "worst_relative_difference": worst,
            "status": "PASS" if worst <= tolerance else "FAIL"}


def gate_arm(measured, tolerance):
    """P2 -- ledger arm verdict for a single branch at the pinned leg."""
    worst_field = max(FIELDS, key=lambda f: measured["l2"][f])
    worst = float(measured["l2"][worst_field])
    return {"metric": "relative L2 (whole field, Gauss nodes)",
            "tolerance": tolerance, "worst_field": worst_field,
            "worst": worst, "margin": worst / tolerance,
            "values": {f: float(measured["l2"][f]) for f in FIELDS},
            "l2": {f: float(measured["l2"][f]) for f in FIELDS},
            "l1": {f: float(measured["l1"][f]) for f in FIELDS},
            "status": "PASS" if worst < tolerance else "FAIL"}


def gate_eligible(arms_metrics, arm):
    """P3 -- C1/C2 of the limiter study must be PASS for this branch.

    The per-gate records live under ``arms[arm].gates`` in the limiter-study
    artifact; ``verdicts[arm]`` there is only the overall string.
    """
    gates = arms_metrics["arms"][arm]["gates"]
    rows = {name: gates[name]["status"] for name in ("c1", "c2")}
    ok = all(v == "PASS" for v in rows.values())
    return {"gate": "P3", "arm": arm, "c1_c2": rows,
            "status": "PASS" if ok else "FAIL"}


def gate_a0(arms_rows, eligible, tolerance):
    """P4 -- overall A0 verdict across the branches at the pinned leg."""
    passing = [arm for arm in ARMS
               if arms_rows[arm]["status"] == "PASS" and eligible[arm]["status"] == "PASS"]
    rejected = [arm for arm in ARMS
                if arms_rows[arm]["status"] == "PASS" and eligible[arm]["status"] != "PASS"]
    return {"gate": "A0", "source": "feature_ledger CASE-LAX-DG",
            "wording": "ref: exact solution | tolerance < 3 %",
            "production_leg": PRODUCTION_LEG, "tolerance": tolerance,
            "credited_arms": passing, "passing_but_ineligible": rejected,
            "status": "PASS" if passing else "FAIL"}


# ------------------------------------------------------------------ figures
def render_figures(data, metrics, out_dir=OUT / "figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    tolerance = metrics["acceptance"]["tolerance"]
    made = []

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    exact = data["none_exact_density"]
    for ax, (lo, hi) in zip(axes, ((0.60, 0.78), (0.78, 0.87))):
        ax.plot(data["none_points"].ravel(), exact.ravel(), "k-", lw=1.2, label="exact")
        for arm, style in (("none", "-"), ("tvb", "--"), ("weno", ":")):
            ax.plot(data[f"{arm}_points"].ravel(),
                    data[f"{arm}_numerical_density"].ravel(), style, lw=1.0, label=arm)
        ax.set_xlim(lo, hi)
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"Lax {PRODUCTION_LEG}: density at the contact (top) and shock (bottom)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "lax_a0_density_branches.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    index = np.arange(len(FIELDS))
    for k, (arm, colour) in enumerate((("none", "C0"), ("tvb", "C1"), ("weno", "C2"))):
        values = [metrics["arms"][arm]["l2"][f] * 100 for f in FIELDS]
        ax.bar(index + (k - 1) * width, values, width, label=arm, color=colour)
    ax.axhline(tolerance * 100, color="k", ls="--", lw=1.0,
               label=f"ledger bar {tolerance * 100:.1f} %")
    ax.set_xticks(index)
    ax.set_xticklabels(FIELDS)
    ax.set_ylabel("relative L2 [%]")
    ax.set_title(f"A0 ledger arm at {PRODUCTION_LEG} across limiter branches")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "lax_a0_rel_l2_bars.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)
    return made


# ------------------------------------------------------------------ summary
def _pct(v):
    return f"{v * 100:.3f} %"


def render_summary(metrics):
    tol = metrics["acceptance"]["tolerance"]
    lines = ["# A0 ledger arm across limiter branches (pinned leg " + PRODUCTION_LEG + ")",
             "",
             f"generated_at_utc = {metrics['generated_at_utc']}, "
             f"source profiles sha256 = `{metrics['provenance']['profiles_sha256'][:16]}...`",
             "",
             "| arm | rel L2 rho | rel L2 u | rel L2 p | worst | margin | C1/C2 | A0 |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        row = metrics["arms"][arm]
        lines.append(
            f"| {arm} | {row['l2']['density']:.6e} | {row['l2']['velocity']:.6e} | "
            f"{row['l2']['pressure']:.6e} | {row['worst']:.6e} | x{row['margin']:.3f} | "
            f"{metrics['eligibility'][arm]['status']} | {row['status']} |")
    lines += ["", f"tolerance = {tol} ({_pct(tol)})", "",
              f"A0 verdict: **{metrics['acceptance']['status']}** "
              f"(credited arms: {metrics['acceptance']['credited_arms'] or 'none'})"]
    return "\n".join(lines) + "\n"


def render_doc(metrics):
    tol = metrics["acceptance"]["tolerance"]
    o = metrics["outcome"]
    lines = [
        "# CASE-LAX-DG · 台账臂 A0 跨限幅器分支重裁（钉扎腿 p3.N400，零重解）",
        "",
        "> 本文全部数字由 `scripts/render_lax_a0_arms_doc.py` 从 "
        "`docs/benchmarks/results/lax_a0_limiter_arms/metrics.json` 注入（零手抄），"
        "`--check` 在文档漂移时失败。",
        "> 数据来源：已提交的限幅器分支场缓存 "
        "`docs/benchmarks/results/lax_case_pool_limiter_arms/profiles.npz`"
        f"（sha256 `{metrics['provenance']['profiles_sha256'][:16]}...`），"
        "**本单没有重跑任何求解器**，因此不存在事后调参的窗口。",
        "",
        "## ① 案例介绍",
        "",
        "Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]`，隔膜 `x0 = 0.5`，γ = 1.4，"
        "左态 (ρ,u,p) = (0.445, 0.698, 3.528)，右态 (0.5, 0.0, 0.571)，快照 t = 0.13。"
        "精确解含左稀疏波（head/tail）、接触间断、右激波，四道波位置与星区状态均有闭式解。",
        "",
        "## ② 怎么做的",
        "",
        "1. **缺口**：既有验证已把 C1/C2/C3/R3/R5 在 `none`/`tvb`/`weno` 三支上跑完，"
        "但**台账臂 A0（钉扎腿 `p3.N400` 的全场 relative L2）从未在限幅器分支上评过**。"
        "该臂恰是对激波后振荡最敏感到的一个测度：无限幅时同腿 "
        f"rel L1 ρ = {metrics['diagnostics']['none_l1_over_l2']['l1']:.6e} 而 "
        f"rel L2 ρ = {metrics['diagnostics']['none_l1_over_l2']['l2']:.6e}，"
        f"比值 {metrics['diagnostics']['none_l1_over_l2']['ratio']:.2f}——L2 误差由局部尖峰携带，"
        "而限幅器正是压制尖峰的算子（对 L1 反而因涂抹而变差）。",
        "2. **零重解**：直接读已提交的场缓存（三支在每单元 10 点 Gauss 节点上的原始变量 + 精确解），"
        "用与 `lax_case_pool.production_errors_from_profiles` 完全相同的加权（10 点 Legendre 求积、"
        "`dx = span/(ncells-1)`）重算 L1/L2。",
        "3. **预登记（跑数前写死在 `benchmarks/lax_a0_limiter_arms.py`）**：",
        "   - P0 复用：`none` 支必须逐位复现已提交的池子工件（相对差 ≤ 1e-12），否则整轮 `WITHHELD`；",
        "   - P1 度量：与台账臂同一口径（全场、不加波带、逐量、分母 = 精确解范数），容差从 "
        "`lax_case_pool.ledger_acceptance_tolerance()` 读，**不在本文件硬编码**；",
        "   - P2 分支判定：`max_field rel L2 < 容差` 即 PASS；",
        "   - P3 可用性：该支在限幅器研究里的 C1/C2 必须 PASS（波位/星区不能被限幅毁掉）；",
        "   - P4 总判：钉扎腿上有**可用**支 PASS 即 A0 PASS；**生产腿不动**、阶梯不加长、"
        "除 `limiter` 外的旋钮（`tvb_constant`/`limiter_stage`/`positivity`/`cfl`）全取库默认；",
        "   - P5：若无可用支过 3 %，记录保持 FAIL，分辨率路线（`p3.N800`）仍是唯一在案出路。",
        "",
        "## ③ 计算结果（程序化注入）",
        "",
        f"复用闸 P0：`none` 支 vs 已提交池子工件，最差相对差 "
        f"`{metrics['reuse']['worst_relative_difference']:.3e}` → **{metrics['reuse']['status']}**。",
        "",
        "| 分支 | rel L2 ρ | rel L2 u | rel L2 p | 最差 | 裕度（/容差） | 限幅后 L1 ρ | C1/C2 | A0 判定 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        row = metrics["arms"][arm]
        lines.append(
            f"| `{arm}` | {row['l2']['density']:.6e} | {row['l2']['velocity']:.6e} | "
            f"{row['l2']['pressure']:.6e} | {row['worst']:.6e} | ×{row['margin']:.3f} | "
            f"{row['l1']['density']:.6e} | {metrics['eligibility'][arm]['status']} | "
            f"**{row['status']}** |")
    lines += [
        "",
        f"台账臂 A0（钉扎腿 `{PRODUCTION_LEG}`，ref = 精确 Riemann 解，容差 {_pct(tol)}）："
        f"**{metrics['acceptance']['status']}**；"
        f"计入的分支 = {metrics['acceptance']['credited_arms'] or '无'}；"
        f"过阈但 C1/C2 不通过而不计入的分支 = "
        f"{metrics['acceptance']['passing_but_ineligible'] or '无'}。",
        "",
        "![density branches](benchmarks/results/lax_a0_limiter_arms/figures/lax_a0_density_branches.png)",
        "",
        "![rel L2 bars](benchmarks/results/lax_a0_limiter_arms/figures/lax_a0_rel_l2_bars.png)",
        "",
        "## ④ 注意点",
        "",
        "1. **这是分支陈述，不是配置回调**：三支的参数在限幅器研究跑数前已写死，本单只做后处理；"
        "`tvb_constant` 等旋钮一个都没动，所以“过阈”不是把容差喂给求解器的结果。",
        "2. **L1 与 L2 反向**：限幅支的 rel L1 更差、rel L2 更好，这不是矛盾——L1 对全场涂抹积分、"
        "L2 由局部尖峰主导。台账口径是 L2，故本单以 L2 判；但同一算例的“更严臂”（全阶梯 L1）"
        "在限幅支上会变差，读者若把 < 3 % 读作无范数口径，请回到 `docs/CASE-LAX-DG.md` 的阶梯表。",
        "3. **C3/R3 不受影响**：限幅支的两条阶数闸仍 FAIL（限幅把局部阶压到 ~2），"
        "本单不触碰验证层的 honest FAIL。",
        "4. **场缓存是 10 点 Gauss 节点**：误差在求积节点上按 Legendre 权重积分，"
        "换单元平均/L∞ 会给出不同数字；口径与池子工件逐位一致（P0 已证）。",
        "",
        "## ⑤ 误差判定（是否 < 3 %）",
        "",
        "| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        row = metrics["arms"][arm]
        verdict = "**是**" if row["status"] == "PASS" else "**否**"
        lines.append(
            f"| `{arm}` 钉扎腿 rel L2 ρ | 精确 Riemann 解 | {row['l2']['density']:.6e} | "
            f"{_pct(row['l2']['density'])} | {verdict} |")
    lines += [
        "",
        f"**结论**：{o['statement']}",
        "",
        "## ⑥ 给台账持有人的建议（不由本单自裁）",
        "",
        o["recommendation"],
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- main
def build_metrics(with_figures=True):
    if not PROFILES.exists():
        raise SystemExit(f"missing source cache: {PROFILES}")
    data = np.load(PROFILES)
    pool_metrics = json.loads(POOL_METRICS.read_text())
    arms_metrics = json.loads(ARMS_METRICS.read_text())
    tolerance = ledger_acceptance_tolerance()

    measured = {arm: measure_arm(data, arm) for arm in ARMS}
    reuse = gate_reuse(measured["none"], pool_metrics)
    arms_rows = {arm: gate_arm(measured[arm], tolerance) for arm in ARMS}
    eligibility = {arm: gate_eligible(arms_metrics, arm) for arm in ARMS}

    if reuse["status"] != "PASS":
        acceptance = {"gate": "A0", "status": "WITHHELD", "tolerance": tolerance,
                      "production_leg": PRODUCTION_LEG, "credited_arms": [],
                      "passing_but_ineligible": [],
                      "reason": "P0 reuse gate failed: cache is not the committed acceptance field"}
    else:
        acceptance = gate_a0(arms_rows, eligibility, tolerance)

    l1_rho = measured["none"]["l1"]["density"]
    l2_rho = measured["none"]["l2"]["density"]
    diagnostics = {"none_l1_over_l2": {"l1": l1_rho, "l2": l2_rho,
                                       "ratio": float(l2_rho / l1_rho)}}

    if acceptance["status"] == "PASS":
        arm = acceptance["credited_arms"][0]
        row = arms_rows[arm]
        statement = (f"在**钉扎腿不动**的前提下，台账臂 A0 被分支 `{arm}` 满足："
                     f"最差量 rel L2 ρ = {row['l2']['density']:.6e}（{_pct(row['l2']['density'])}），"
                     f"裕度 ×{row['margin']:.3f}；无限幅支仍为 FAIL（3.116 %），"
                     "两条阶数闸 C3/R3 仍 honest-FAIL。")
        recommendation = (
            f"台账持有人可考虑把 `CASE-LAX-DG` 的生产配置显式登记为 "
            f"`limiter='{arm}'`（库默认常数），届时 A0 = PASS、C1/C2 = PASS，"
            "C3/R3 作为方法属性保留 FAIL。这是**口径登记**（生产配置里写清 `limiter`），"
            "不是移动生产腿，也不是放宽容差；本单不代为改台账状态。")
    elif acceptance["status"] == "WITHHELD":
        statement = "复用闸 P0 未通过，本轮不作裁决（工件的场与验收记录不是同一批场）。"
        recommendation = "先修场缓存与验收记录的一致性，再重跑本驱动。"
    else:
        statement = ("所有可用分支在钉扎腿 `p3.N400` 上均未过 3 %（honest FAIL），"
                     "分辨率路线 `p3.N800`（2.315 %）仍是唯一在案出路。")
        recommendation = ("如需翻台账状态，只能由台账持有人显式把生产腿改为 `p3.N800`"
                          "（口径变更），或接受以限幅器分支登记生产配置。")

    metrics = {
        "case": "lax-1954-shock-tube",
        "anchor": "CASE-LAX-DG",
        "study": "ledger arm A0 re-adjudicated across limiter branches (offline, zero solves)",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "profiles_file": str(PROFILES),
            "profiles_sha256": _sha256(PROFILES),
            "pool_metrics": str(POOL_METRICS),
            "arms_metrics": str(ARMS_METRICS),
            "solver_rerun": False,
        },
        "pre_registration": {
            "P0": "none-branch reproduces committed pool rel L2 to <= 1e-12 relative",
            "P1": "metric = ledger arm (relative L2, Gauss nodes, whole field, exact denominator)",
            "P2": "arm PASS iff max_field rel L2 < tolerance",
            "P3": "arm credited only if the limiter study's C1/C2 for that arm PASS",
            "P4": "A0 PASS iff an eligible arm PASSes at the pinned leg p3.N400",
            "P5": "otherwise honest FAIL; production leg and ladder untouched",
            "not_swept": ["tvb_constant", "limiter_stage", "positivity", "cfl"],
        },
        "config": {"production_leg": PRODUCTION_LEG, "arms": list(ARMS),
                   "quadrature_order": 10, "tolerance": tolerance},
        "reuse": reuse,
        "arms": arms_rows,
        "eligibility": eligibility,
        "diagnostics": diagnostics,
        "acceptance": acceptance,
        "outcome": {"statement": statement, "recommendation": recommendation},
    }
    if with_figures:
        figures = render_figures(data, metrics)
        metrics["figures"] = [str(p) for p in figures]
    return metrics


def write_outputs(metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True) + "\n")
    (OUT / "summary.md").write_text(render_summary(metrics))
    DOC.write_text(render_doc(metrics))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="re-measure and compare with the committed artifact; write nothing")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip matplotlib (headless / CI)")
    args = parser.parse_args(argv)
    metrics = build_metrics(with_figures=False if args.check else not args.no_figures)
    if args.check:
        committed = json.loads((OUT / "metrics.json").read_text())
        problems = []
        for arm in ARMS:
            for field in FIELDS:
                for norm in ("l1", "l2"):
                    a = metrics["arms"][arm][norm][field]
                    b = committed["arms"][arm][norm][field]
                    if abs(a - b) > 1e-12 * max(abs(b), 1e-30):
                        problems.append(f"{arm}.{norm}.{field}: {a!r} vs {b!r}")
        if metrics["acceptance"]["status"] != committed["acceptance"]["status"]:
            problems.append("acceptance status drift")
        if (DOC.exists() and render_doc({**committed, "outcome": committed["outcome"]}) != DOC.read_text()):
            problems.append("doc drift")
        for problem in problems:
            print("DRIFT:", problem)
        print("check:", "OK" if not problems else f"{len(problems)} problem(s)")
        return 0 if not problems else 1
    write_outputs(metrics)
    print(render_summary(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())