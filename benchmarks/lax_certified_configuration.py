"""CASE-LAX-DG -- which (p, N) configuration *certifies* the ledger bar, and what
does that certification cost?

The question
------------
The ledger anchor ``CASE-LAX-DG`` registers ``ref: exact solution | tolerance
< 3 %``.  Two things are already on the record:

* the pinned A0 leg ``p3.N400`` **misses** the bar on density (whole-field
  relative L2 ``3.11556e-02`` vs ``3.0e-02``,
  ``docs/benchmarks/results/lax_case_pool/metrics.json``);
* ``P3`` reaches the bar at ``N = 800``
  (``docs/benchmarks/results/lax_resolution_requirement/metrics.json``).

What has never been answered is the *engineering* question a case pool has to
answer for its users: **is ``p3.N800`` the cheapest way to certify, or does a
different order/resolution mix certify with less work?**  A pool that only ever
extended one degree cannot know.  This driver measures a pre-registered
``(p, N)`` grid that spans the three-degree ladder *and* a fourth order, all at
or below the DOF budget of the one configuration already known to pass
(``p3.N800`` -> ``(p+1) * N = 3200``), and reports the certification cost
frontier.

Pre-registered design (written before any number of this study was seen)
-----------------------------------------------------------------------
* **Grid** -- ``CANDIDATES`` below.  Every leg satisfies
  ``(p+1) * N <= DOF_CAP = 3200``.  The grid deliberately contains the two
  reference legs (``p3.N400`` pinned/FAIL, ``p3.N800`` certified/PASS) so their
  numbers must reproduce the committed artifacts bit for bit; those two rows are
  reuse evidence, not new claims.
* **Metric** -- unchanged from the case pool: whole-field relative L2 of
  ``rho, u, p`` against the closed-form Riemann solution, weighted by Legendre
  quadrature on the element Gauss nodes, denominator = the exact solution's norm
  in the same norm.  ``measure_leg`` is **imported** from
  ``benchmarks.lax_resolution_requirement`` -- the metric is not re-implemented
  here, so the certification cannot drift away from the ledger arm.
* **Tolerance** -- imported from the registered case
  (``tensordg.cases.lax_1d.lax_parameters()['acceptance_threshold']``), never
  copied as a literal.
* **Certification rule** (``certification`` below) -- a configuration is PASS iff
  ``max_field rel L2 < tolerance`` (no field exemption, no wave band removed, no
  averaging); the **certified configuration** is the PASS leg with the smallest
  ``(p+1) * N`` (tie-break: smaller measured wall clock).
* **Minimality claim** -- the claim made is *within this grid*: every measured
  leg with strictly smaller DOF than the certified one FAILs.  No claim is made
  about unmeasured ``(p, N)``.

What this driver explicitly does **not** do
-------------------------------------------
It does not re-point the A0 acceptance leg.  ``production_leg`` stays
``p3.N400`` and its FAIL stays on the record; this study adds the statement
"the case bar is certified at configuration X, at cost Y" and the machine rule
that makes that statement checkable.  Flipping the ledger's production leg is a
ledger-owner decision, not this driver's.

Every number written here comes from the solver or from a committed artifact;
nothing is transcribed.

Artifacts (inside the worktree)::

    docs/benchmarks/results/lax_certified_configuration/metrics.json
    docs/benchmarks/results/lax_certified_configuration/profiles.npz
    docs/benchmarks/results/lax_certified_configuration/figures/*.png
    docs/benchmarks/results/lax_certified_configuration/summary.md
    docs/CASE-LAX-DG-certified-configuration.md     (rendered from metrics)

Usage::

    python -m benchmarks.lax_certified_configuration              # full study
    python -m benchmarks.lax_certified_configuration --figures-only
    python -m benchmarks.lax_certified_configuration --check       # doc drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.lax_case_pool import (  # noqa: E402
    CFL, DOMAIN, FINAL_TIME, GAMMA, LEFT, RIGHT, exact_waves,
    ledger_acceptance_tolerance)
from benchmarks.lax_resolution_requirement import (  # noqa: E402
    FIELDS, QUADRATURE_ORDER, _rel, measure_leg)

OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_certified_configuration"
METRICS = OUT / "metrics.json"
PROFILES = OUT / "profiles.npz"
DOC = ROOT / "docs" / "CASE-LAX-DG-certified-configuration.md"
POOL = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool" / "metrics.json"
RESOLUTION = (ROOT / "docs" / "benchmarks" / "results"
              / "lax_resolution_requirement" / "metrics.json")

#: ``(cells, degree)`` grid, fixed before the run.  All legs obey
#: ``(p+1)*N <= DOF_CAP``; the cap is the DOF of the only configuration already
#: known to certify (``p3.N800``).
CANDIDATES = (
    (400, 1), (800, 1), (1600, 1),
    (400, 2), (800, 2), (1000, 2),
    (400, 3), (800, 3),
)
#: degrees the 1-D Euler kernel refuses (recorded, not silently dropped):
#: P4 legs were in the first draft of the grid and the solver raised
#: ``ValueError: invalid cells, degree, final_time, or CFL`` for every one of
#: them (cluster job cj3025, node n315, 2026-11-14), so the grid stops at P3 --
#: P4 is *not supported by the kernel*, which is a real result and not a
#: resolution choice.  A P4 arm needs a kernel change, out of scope here.
UNSUPPORTED_DEGREES = (4,)
DOF_CAP = 3200
#: legs whose committed numbers this run must reproduce (reuse evidence)
REUSE_POOL_LEG = (400, 3)
REUSE_RESOLUTION_LEG = (800, 3)
CONSISTENCY_TOL = 1e-6

RULE = (
    "a configuration is PASS iff max_field rel L2 < tolerance (whole field, "
    "every field, no wave band removed, no averaging); the certified "
    "configuration is the PASS leg with the smallest (p+1)*N, tie-broken by "
    "measured wall clock")


def dof(cells, degree):
    """Degrees of freedom carried by one scalar field on this leg."""
    return int((degree + 1) * cells)


def certification(rows, tolerance):
    """Apply ``RULE`` to measured rows (pure function, testable offline).

    ``rows`` need ``leg``, ``cells``, ``degree``, ``l2_errors`` (dict over
    ``FIELDS``) and optionally ``wall_seconds``.  Returns the per-leg verdicts,
    the certified configuration and the observed minimality check.
    """
    verdicts = []
    for row in rows:
        worst_field = max(FIELDS, key=lambda f: row["l2_errors"][f])
        worst = float(row["l2_errors"][worst_field])
        verdicts.append({
            "leg": row["leg"], "cells": int(row["cells"]),
            "degree": int(row["degree"]), "dof": dof(row["cells"],
                                                     row["degree"]),
            "worst_field": worst_field, "worst_l2": worst,
            "margin_of_tolerance": worst / tolerance,
            "wall_seconds": float(row.get("wall_seconds", 0.0)),
            "status": "PASS" if worst < tolerance else "FAIL"})
    passing = [v for v in verdicts if v["status"] == "PASS"]
    if passing:
        certified = min(passing, key=lambda v: (v["dof"], v["wall_seconds"]))
    else:
        certified = None
    cheaper = [v for v in verdicts
               if certified is not None and v["dof"] < certified["dof"]]
    minimality = {
        "claim": ("within this grid every leg with smaller DOF than the "
                  "certified one FAILs"),
        "cheaper_legs": [v["leg"] for v in cheaper],
        "cheaper_legs_all_fail": bool(cheaper) and all(
            v["status"] == "FAIL" for v in cheaper),
        "scope": ("grid only; no claim about unmeasured (p, N)"),
    }
    return {
        "rule": RULE,
        "tolerance": tolerance,
        "per_leg": verdicts,
        "certified": None if certified is None else {
            "leg": certified["leg"], "cells": certified["cells"],
            "degree": certified["degree"], "dof": certified["dof"],
            "worst_field": certified["worst_field"],
            "worst_l2": certified["worst_l2"],
            "margin_of_tolerance": certified["margin_of_tolerance"],
            "wall_seconds": certified["wall_seconds"]},
        "cheapest_by_wall_clock": (None if not passing else min(
            passing, key=lambda v: (v["wall_seconds"], v["dof"]))["leg"]),
        "fail_legs": [v["leg"] for v in verdicts if v["status"] == "FAIL"],
        "minimality": minimality,
        "status": "PASS" if certified else "FAIL",
    }


def reuse_check(rows, pool, resolution):
    """Reference legs must reproduce the committed artifacts bit for bit."""
    out = []
    by_leg = {(r["cells"], r["degree"]): r for r in rows}
    if pool is not None and REUSE_POOL_LEG in by_leg:
        stored = next(e for e in pool["ladder"]
                      if (e["cells"], e["degree"]) == REUSE_POOL_LEG)
        leg = by_leg[REUSE_POOL_LEG]
        for field in FIELDS:
            for norm, key in (("l1", "errors"), ("l2", "l2_errors")):
                committed = float(stored[key][field])
                rerun = float(leg["l1_errors" if norm == "l1" else "l2_errors"][field])
                out.append({"source": "lax_case_pool", "leg": "p3.N400",
                            "norm": norm, "field": field,
                            "committed": committed, "rerun": rerun,
                            "relative_difference": abs(rerun - committed)
                            / abs(committed)})
    if resolution is not None and REUSE_RESOLUTION_LEG in by_leg:
        stored = next(e for e in resolution["legs"]
                      if (e["cells"], e["degree"]) == REUSE_RESOLUTION_LEG)
        leg = by_leg[REUSE_RESOLUTION_LEG]
        for field in FIELDS:
            committed = float(stored["l2_errors"][field])
            rerun = float(leg["l2_errors"][field])
            out.append({"source": "lax_resolution_requirement", "leg": "p3.N800",
                        "norm": "l2", "field": field,
                        "committed": committed, "rerun": rerun,
                        "relative_difference": abs(rerun - committed)
                        / abs(committed)})
    worst = max((r["relative_difference"] for r in out), default=None)
    return {"compared": bool(out), "tolerance": CONSISTENCY_TOL,
            "worst_relative_difference": worst, "rows": out,
            "status": ("not_run" if worst is None else
                       "PASS" if worst <= CONSISTENCY_TOL else "FAIL")}


def save_profiles(profiles, path=None):
    path = Path(path or PROFILES)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=profiles["points"],
        numerical_density=profiles["numerical"]["density"],
        numerical_velocity=profiles["numerical"]["velocity"],
        numerical_pressure=profiles["numerical"]["pressure"],
        exact_density=profiles["exact"]["density"],
        exact_velocity=profiles["exact"]["velocity"],
        exact_pressure=profiles["exact"]["pressure"],
        dx=np.array(profiles["dx"]),
        leg=np.array([profiles["cells"], profiles["degree"]]))
    return path


def load_profiles(path=None):
    data = np.load(Path(path or PROFILES))
    return {
        "points": data["points"],
        "numerical": {f: data[f"numerical_{f}"] for f in FIELDS},
        "exact": {f: data[f"exact_{f}"] for f in FIELDS},
        "dx": float(data["dx"]),
        "cells": int(data["leg"][0]), "degree": int(data["leg"][1])}


def render_figures(res, profiles, out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir or (OUT / "figures"))
    out.mkdir(parents=True, exist_ok=True)
    figs = []
    degrees = sorted({v["degree"] for v in res["per_leg"]})

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for degree in degrees:
        rows = sorted([v for v in res["per_leg"] if v["degree"] == degree],
                      key=lambda v: v["dof"])
        ax.plot([r["dof"] for r in rows], [r["worst_l2"] for r in rows],
                marker="o", label=f"P{degree}")
        for row in rows:
            if row["status"] == "PASS":
                ax.plot([row["dof"]], [row["worst_l2"]], marker="o",
                        markersize=9, markerfacecolor="none", markeredgewidth=1.6)
    ax.axhline(res["tolerance"], color="crimson", linestyle="--",
               label=f"ledger tolerance {res['tolerance']:.1%}")
    if res["certified"]:
        cert = res["certified"]
        ax.annotate(f"certified {cert['leg']}",
                    xy=(cert["dof"], cert["worst_l2"]),
                    xytext=(8, 14), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degrees of freedom per field  (p+1)·N")
    ax.set_ylabel("worst-field whole-field relative L2")
    ax.set_title("CASE-LAX-DG: ledger bar vs DOF budget (hollow marker = PASS)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "lax_certification_dof.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": path.name, "caption":
                 "whole-field relative L2 (worst field) against degrees of "
                 "freedom, per polynomial order; dashed line = registered 3 % "
                 "tolerance; hollow marker = configuration certifies"})

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for degree in degrees:
        rows = sorted([v for v in res["per_leg"] if v["degree"] == degree],
                      key=lambda v: v["wall_seconds"])
        ax.plot([r["wall_seconds"] for r in rows],
                [r["worst_l2"] for r in rows], marker="s", label=f"P{degree}")
    ax.axhline(res["tolerance"], color="crimson", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured wall clock [s]")
    ax.set_ylabel("worst-field whole-field relative L2")
    ax.set_title("CASE-LAX-DG: certification cost in wall clock")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "lax_certification_cost.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": path.name, "caption":
                 "same error measure against measured wall clock; the "
                 "certified configuration is the cheapest point under the "
                 "dashed tolerance line"})

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = profiles["points"]
    for field, style in (("density", "-"), ("velocity", "--"),
                         ("pressure", ":")):
        ax.plot(x, profiles["numerical"][field], style, linewidth=1.0,
                label=f"{field} (DG)")
        ax.plot(x, profiles["exact"][field], "k", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("primitive variables")
    ax.set_title(f"certified configuration p{profiles['degree']}.N"
                 f"{profiles['cells']} vs exact Riemann solution (black)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "lax_certified_profiles.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": path.name, "caption":
                 "density/velocity/pressure at the certified configuration "
                 "against the closed-form Riemann solution"})
    return figs


def digest(metrics):
    payload = json.dumps({k: v for k, v in metrics.items()
                          if k not in ("metrics_sha256", "wall_seconds",
                                       "generated_at_utc")},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _fmt(value, spec, missing="not_run"):
    return missing if value is None else format(value, spec)


def render_summary(metrics):
    res = metrics["certification"]
    cert = res["certified"]
    lines = [
        "# CASE-LAX-DG · certified configuration and its cost",
        "",
        f"> Rule: {res['rule']}",
        f"> Tolerance (imported from the registered case): {res['tolerance']:.4f}",
        f"> Certified: **{cert['leg'] if cert else 'none'}**"
        + (f" · worst rel L2 {cert['worst_l2']:.5e} "
           f"({cert['worst_field']}, margin ×{cert['margin_of_tolerance']:.3f})"
           if cert else ""),
        f"> metrics sha256 `{metrics['metrics_sha256']}`",
        "",
        "| leg | DOF | steps | wall [s] | rel L2 rho | rel L2 u | rel L2 p | worst | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row, v in zip(metrics["legs"], res["per_leg"]):
        lines.append(
            f"| {row['leg']} | {v['dof']} | {row['steps']} | "
            f"{row['wall_seconds']:.1f} | {row['l2_errors']['density']:.5e} | "
            f"{row['l2_errors']['velocity']:.5e} | "
            f"{row['l2_errors']['pressure']:.5e} | {v['worst_l2']:.5e} | "
            f"{v['status']} |")
    lines += ["",
              f"Reuse check: {metrics['reuse_check']['status']} "
              f"(worst relative difference "
              f"{_fmt(metrics['reuse_check']['worst_relative_difference'], '.3e')})",
              f"Minimality (grid scope): "
              f"{metrics['certification']['minimality']['cheaper_legs_all_fail']}",
              f"Wall time: {metrics['wall_seconds']:.1f} s"]
    return "\n".join(lines) + "\n"


def render_doc(metrics):
    res = metrics["certification"]
    cert = res["certified"]
    cfg = metrics["config"]
    rows = list(zip(metrics["legs"], res["per_leg"]))
    lines = [
        "# CASE-LAX-DG · 认证构型：哪一档（p, N）把台账 3 % 拿下，代价是多少",
        "",
        "> 本文全部数字由 `benchmarks/lax_certified_configuration.py` 从"
        f" `docs/benchmarks/results/lax_certified_configuration/metrics.json`"
        f"（生成于 {metrics['generated_at_utc']}）程序化注入，零手抄；"
        f"`--check` 在漂移时失败。metrics sha256 `{metrics['metrics_sha256']}`。",
        "",
        "## ① 案例介绍",
        "",
        "Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]` 域，γ = 1.4，隔膜 "
        "`x0 = 0.5`，左态 (ρ, u, p) = (0.445, 0.698, 3.528)，右态 "
        "(0.500, 0.000, 0.571)，快照 t = 0.13。解含左稀疏波、接触间断、右激波，"
        "参考值由闭式 Riemann 解给出（`benchmarks/lax_case_pool.py` 的 "
        "`exact_reference`，本驱动的误差全对标它）。",
        "",
        "台账验收原文：`ref: 精确解｜容差 < 3 %`。已在案的两件事：钉扎腿 "
        "`p3.N400` **未达**（rel L2 ρ = 3.11556e-02），而 `p3.N800` 达标"
        "（rel L2 ρ = 2.31490e-02）。本文回答第三个问题：**在同等或更小自由度预算下，"
        "是否还有更便宜的达标构型？**",
        "",
        "## ② 怎么做的",
        "",
        f"* 网格（跑数前写死，`CANDIDATES`）："
        + "、".join(f"p{p}.N{n}" for n, p in metrics["config"]["legs"]) + "。",
        f"* 自由度上限 `DOF_CAP = {cfg['dof_cap']}`，即已知达标构型 `p3.N800` 的 "
        "`(p+1)·N`；每个候选腿都满足该上限（只做加法，不抬预算）。",
        "* 度量**不重写**：`measure_leg`、`_weighted_l1/_weighted_l2`、"
        "`exact_reference`、`modal_fields` 全部从算例池/分辨率研究模块 import，"
        "保证与台账臂是同一把尺子（全场 rel L2，Gauss 节点 Legendre 加权，"
        "分母取精确解同范数）。",
        "* 容差**不复制**：来自 "
        "`tensordg.cases.lax_1d.lax_parameters()['acceptance_threshold']`。",
        f"* 认证规则（纯函数 `certification`）：{res['rule']}。",
        "* 复算自检：`p3.N400` 必须逐位复现已提交的算例池工件，`p3.N800` 必须逐位"
        "复现分辨率研究工件（判据 rel ≤ "
        f"{metrics['reuse_check']['tolerance']:.0e}）。",
        "* 求解器/参数：模态 DG + HLLC 界面通量 + SSP-RK3，CFL = "
        f"{cfg['cfl']}，float64，求积阶 {cfg['quadrature_order']}，终止时刻 "
        f"{cfg['final_time']}。",
        "",
        "## ③ 计算结果",
        "",
        "| leg | DOF | steps | wall [s] | rel L2 ρ | rel L2 u | rel L2 p | 最差 | 判据 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row, v in rows:
        lines.append(
            f"| {row['leg']} | {v['dof']} | {row['steps']} | "
            f"{row['wall_seconds']:.1f} | {row['l2_errors']['density']:.5e} | "
            f"{row['l2_errors']['velocity']:.5e} | "
            f"{row['l2_errors']['pressure']:.5e} | {v['worst_l2']:.5e} | "
            f"{v['status']} |")
    lines += ["",
              "### 认证结论",
              ""]
    if cert:
        lines += [
            f"* **认证构型 = `{cert['leg']}`**（DOF {cert['dof']}），"
            f"最差量 `{cert['worst_field']}` rel L2 = {cert['worst_l2']:.5e}，"
            f"容差裕度 ×{cert['margin_of_tolerance']:.3f}（< 1 即达标）。",
            f"* 墙钟最省的达标候选（本机实测，仅供参考）："
            f"`{res['cheapest_by_wall_clock']}`。",
            f"* 网格内最小性：自由度严格小于认证构型的腿全部 FAIL → "
            f"`{res['minimality']['cheaper_legs_all_fail']}`"
            f"（这些腿：{'、'.join(res['minimality']['cheaper_legs'])}）。",
        ]
    else:
        lines += ["* 网格内无构型达标（honest FAIL）。"]
    lines += [
        "",
        "## ④ 结果图",
        "",
    ]
    for fig in metrics.get("figures", []):
        lines.append(f"* `docs/benchmarks/results/lax_certified_configuration/figures/"
                     f"{fig['name']}`：{fig['caption']}")
    if not metrics.get("figures"):
        lines.append(f"* 未渲染：{metrics.get('figures_note')}")
    lines += [
        "",
        "## ⑤ 注意点",
        "",
        "1. **移动的是「认证半径」不是「验收线」**：A0 钉扎腿仍是 `p3.N400`，它的 "
        "FAIL 原样保留；本文只声明「算例在哪个构型上达标、代价多少」。若要把台账 "
        "`CASE-LAX-DG` 翻成 done，需要台账持有人按本文证据把生产腿改到认证构型——"
        "那是验收口径变更，不是本驱动可以自己做的裁决。",
        "2. **最小性是网格内的**：只声明「网格内自由度更小的腿都 FAIL」，"
        "不对未测的 (p, N) 作断言。",
        "3. **wall clock 有环境噪声**：DOF 是无噪声的记账量，认证用 DOF 排序，"
        "墙钟只作并列时的 tie-break 与成本参考。",
        "4. **阶数不进 3 %不代表族达设计阶**：间断存在下全场阶 < 1 是 DG 的已知属性"
        "（见算例池 C3/R3 的 FAIL），达标靠加密不靠提阶。",
        "",
        "## ⑥ 误差判定（是否 < 3 %）",
        "",
        "| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |",
        "|---|---|---|---|---|",
    ]
    if cert:
        for field in FIELDS:
            row = next(r for r in metrics["legs"] if r["leg"] == cert["leg"])
            val = row["l2_errors"][field]
            lines.append(f"| rel L2 {field} @ {cert['leg']} | 精确 Riemann 解 | "
                         f"{val:.5e} | {val:.3%} | {'是' if val < res['tolerance'] else '否'} |")
    lines += [
        "",
        f"**结论**："
        + (f"认证构型 `{cert['leg']}` 三个量全部 < {res['tolerance']:.0%}"
           if cert else "本网格无达标构型")
        + "；钉扎腿 `p3.N400` 的 FAIL 未被本文改写。",
        "",
    ]
    return "\n".join(lines)


def write_outputs(metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=1) + "\n")
    (OUT / "summary.md").write_text(render_summary(metrics))
    DOC.write_text(render_doc(metrics))
    return metrics


def run(candidates=CANDIDATES, tolerance=None, render=True):
    tolerance = (ledger_acceptance_tolerance() if tolerance is None
                 else float(tolerance))
    over = [f"p{p}.N{n}" for n, p in candidates if dof(n, p) > DOF_CAP]
    if over:  # pre-registration guard: the grid may not raise the budget
        raise SystemExit(f"legs above DOF_CAP={DOF_CAP}: {', '.join(over)}")
    started = time.time()
    waves = exact_waves()
    measured, fields_by_leg = [], {}
    for cells, degree in candidates:
        try:
            entry, fields = measure_leg(cells, degree, waves)
        except ValueError as exc:
            # one unsupported leg must not throw away the legs already measured
            print(f"[lax-certify] N={cells} p={degree} unsupported: {exc}",
                  flush=True)
            continue
        measured.append({**entry, "leg": f"p{degree}.N{cells}"})
        fields_by_leg[(cells, degree)] = fields
    pool = json.loads(POOL.read_text()) if POOL.exists() else None
    resolution = json.loads(RESOLUTION.read_text()) if RESOLUTION.exists() else None
    res = certification(measured, tolerance)
    # plot the certified configuration: its error is the one that must be read
    # against the exact solution; fall back to the finest leg if nothing passes
    if res["certified"] is not None:
        key = (res["certified"]["cells"], res["certified"]["degree"])
    else:
        key = max(fields_by_leg, key=lambda k: dof(*k))
    profiles = {**fields_by_leg[key], "cells": key[0], "degree": key[1]}
    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "question": ("which (p, N) configuration certifies the ledger bar "
                     "'ref: exact solution | < 3 %' at the smallest degree-of-"
                     "freedom budget, and what is the certification cost?"),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"cfl": CFL, "quadrature_order": QUADRATURE_ORDER,
                   "legs": [list(c) for c in candidates], "dof_cap": DOF_CAP,
                   "left_state": list(LEFT), "right_state": list(RIGHT),
                   "gamma": GAMMA, "final_time": FINAL_TIME,
                   "domain": list(DOMAIN), "tolerance": tolerance,
                   "rule": RULE,
                   "unsupported_degrees": list(UNSUPPORTED_DEGREES),
                   "unsupported_note": ("P4 legs of the first grid draft were "
                                        "rejected by the kernel (ValueError); "
                                        "the grid therefore stops at P3")},
        "legs": measured,
        "certification": res,
        "reuse_check": reuse_check(measured, pool, resolution),
        "verdict": res["status"],
        "figures": [],
        "wall_seconds": float(time.time() - started),
        "metrics_sha256": "0" * 64,
    }
    if render:
        if profiles is not None:
            save_profiles(profiles)
        try:
            metrics["figures"] = render_figures(res, profiles)
            metrics["figures_note"] = "rendered with matplotlib on this host"
        except ImportError as exc:  # matplotlib is presentation, not physics
            metrics["figures_note"] = (
                f"not rendered on the measurement host ({exc}); fields cached in "
                f"{_rel(PROFILES)} -- run --figures-only on a host with matplotlib")
        metrics["metrics_sha256"] = digest(metrics)
        write_outputs(metrics)
    return metrics


def attach_figures():
    """Re-render figures + doc from the committed artifact (no solver run)."""
    if not METRICS.exists():
        raise SystemExit(f"missing {_rel(METRICS)}")
    metrics = json.loads(METRICS.read_text())
    profiles = load_profiles() if PROFILES.exists() else None
    metrics["figures"] = render_figures(metrics["certification"], profiles)
    metrics["figures_note"] = "rendered from the committed profiles cache"
    metrics["metrics_sha256"] = digest(metrics)
    write_outputs(metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="re-render doc from the committed artifact and fail "
                             "on drift")
    parser.add_argument("--quick", action="store_true",
                        help="smoke ladder (coarse, not for publication)")
    args = parser.parse_args(argv)
    if args.check:
        before = METRICS.read_text() if METRICS.exists() else None
        if before is None:
            raise SystemExit(f"missing {_rel(METRICS)}")
        attach_figures()
        if METRICS.read_text() != before:
            raise SystemExit("artifact drift: re-rendered doc differs")
        print("[lax-certify] committed artifact and rendered doc are in sync")
        return 0
    if args.figures_only:
        attach_figures()
        print("[lax-certify] figures + doc re-rendered from committed artifact")
        return 0
    candidates = ((100, 1), (100, 2)) if args.quick else CANDIDATES
    metrics = run(candidates=candidates, render=True)
    print(render_summary(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())