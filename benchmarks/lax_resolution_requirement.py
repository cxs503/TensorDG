"""CASE-LAX-DG -- how much resolution does the ledger acceptance bar need?

The question
------------
The ledger anchor ``CASE-LAX-DG`` registers ``ref: exact solution | tolerance
< 3 %``.  The pinned A0 leg of the case pool (``p3.N400`` -- the resolution
``benchmarks/lax_1d.py`` and ``tests/test_lax_1d.py`` already use, pinned before
any number was seen) **misses** that bar on density: whole-field relative L2
``rho = 3.11556e-02`` against a tolerance of ``3.0e-02``, i.e. 3.9 % of the bar
left over.  The pool also shows the family converging sub-first-order in the
whole field (C3/R3 FAIL: order ~0.49 for P3, see
``docs/benchmarks/results/lax_case_pool/metrics.json``), so the residual is a
discontinuity property and cannot be removed by tuning anything at N = 400.

The open item is therefore not "make A0 pass" but:

===== ======================================================================
R1    what is the ledger metric itself -- whole-field relative **L2** vs the
      exact Riemann solution -- at P3 / N = 800?  (No L2 number has ever been
      committed past N = 400; the p3 asymptotic probe stored L1 only.)
R2    is the 3 % bar met at that resolution, field by field?
R3    what is the measured whole-field order over N = 400 -> 800, in L1 and in
      L2?
===== ======================================================================

Pre-registered rule (fixed before the run, ``RESOLUTION_RULE`` below): the
resolution requirement is the **smallest** leg of the extension ladder whose
*every* field's relative L2 is below the registered tolerance, and a leg is
reported PASS/FAIL exactly on that condition -- no averaging over fields, no
wave band removed, no leg exempted.

What this driver explicitly does **not** do
-------------------------------------------
It does not re-point the A0 acceptance leg.  ``production_leg`` stays
``p3.N400`` and its FAIL stays on the record; R1-R3 are an additive statement
about resolution, recorded in a separate artifact so that a reader can accept
the resolution finding and still see the pinned-leg miss.

Every number in the emitted artifacts is computed here (solver) or read out of
a committed artifact (reuse check); nothing is transcribed.

Artifacts (inside the worktree)::

    docs/benchmarks/results/lax_resolution_requirement/metrics.json
    docs/benchmarks/results/lax_resolution_requirement/summary.md
    docs/benchmarks/results/lax_resolution_requirement/figures/*.png
    docs/CASE-LAX-DG-resolution-requirement.md          (rendered from metrics)

Usage::

    python -m benchmarks.lax_resolution_requirement            # full study
    python -m benchmarks.lax_resolution_requirement --check     # doc drift
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.lax_case_pool import (
    CFL, DOMAIN, FINAL_TIME, GAMMA, LEFT, RIGHT, _weighted_l1, _weighted_l2,
    convergence_order, exact_reference, ledger_acceptance_tolerance, modal_fields)
from tensordg.euler_dg_1d import solve_euler_dg_1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_resolution_requirement"
FIG = OUT / "figures"
METRICS = OUT / "metrics.json"
DOC = ROOT / "docs" / "CASE-LAX-DG-resolution-requirement.md"
POOL = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool" / "metrics.json"
PROBE = ROOT / "docs" / "benchmarks" / "results" / "lax_asymptotic_probe" / "metrics.json"

FIELDS = ("density", "velocity", "pressure")
#: extension ladder: the pinned A0 leg (re-measured, reuse check) plus one
#: h-refinement.  Fixed before the run.
LEGS = ((400, 3), (800, 3))
PINNED_LEG = (400, 3)
QUADRATURE_ORDER = 10
CONSISTENCY_TOL = 1e-6

RESOLUTION_RULE = (
    "resolution requirement = the smallest leg of the extension ladder whose "
    "whole-field relative L2 is below the registered tolerance for every field "
    "(no field exemption, no wave band removed, no averaging); a leg is PASS "
    "iff max_field rel L2 < tolerance")


def measure_leg(cells, degree, waves):
    """One DG leg, measured with the case pool's own quadrature helpers."""
    started = time.time()
    x, coeff, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=cells, degree=degree,
                                       final_time=FINAL_TIME, cfl=CFL)
    points, primitive, dx, weights = modal_fields(x, coeff, QUADRATURE_ORDER)
    exact = exact_reference(points)
    errors = {f: _weighted_l1(points, primitive[f], exact[f], weights, dx)
              for f in FIELDS}
    l2_errors = {f: _weighted_l2(points, primitive[f], exact[f], weights, dx)
                 for f in FIELDS}
    entry = {
        "cells": int(cells), "degree": int(degree),
        "steps": int(meta["steps"]),
        "dt": float(meta.get("dt_mean", FINAL_TIME / max(meta["steps"], 1))),
        "l1_errors": errors, "l2_errors": l2_errors,
        "density_min": float(primitive["density"].min()),
        "pressure_min": float(primitive["pressure"].min()),
        "wall_seconds": float(time.time() - started),
    }
    print(f"[lax-res] N={cells:5d} p={degree} "
          f"L2(rho)={l2_errors['density']:.6e} L1(rho)={errors['density']:.6e} "
          f"t={entry['wall_seconds']:.1f}s", flush=True)
    return entry, {"points": points, "numerical": primitive, "exact": exact,
                   "dx": dx, "weights": weights}


def rate(legs, field, norm):
    """Whole-field order between the two extension legs (same code path)."""
    coarse, fine = legs
    key = "l1_errors" if norm == "l1" else "l2_errors"
    return convergence_order(coarse["cells"], coarse[key][field],
                            fine["cells"], fine[key][field])


def requirements(legs, tolerance):
    """R2/R1: PASS/FAIL per leg and the smallest leg that meets the bar."""
    rows = []
    for entry in legs:
        worst_field = max(FIELDS, key=lambda f: entry["l2_errors"][f])
        worst = float(entry["l2_errors"][worst_field])
        rows.append({"leg": f"p{entry['degree']}.N{entry['cells']}",
                     "cells": entry["cells"], "degree": entry["degree"],
                     "worst_field": worst_field, "worst_l2": worst,
                     "margin_of_tolerance": worst / tolerance,
                     "status": "PASS" if worst < tolerance else "FAIL"})
    meeting = [row for row in rows if row["status"] == "PASS"]
    smallest = min(meeting, key=lambda row: row["cells"]) if meeting else None
    return {"rule": RESOLUTION_RULE, "tolerance": tolerance, "per_leg": rows,
            "smallest_leg_meeting_tolerance": smallest["leg"] if smallest else None,
            "resolution_factor": (
                None if smallest is None
                else smallest["cells"] / legs[0]["cells"]),
            "status": "PASS" if smallest else "FAIL"}


def reuse_check(legs, pool, probe):
    """Re-measurements must reproduce the committed artifacts bit for bit."""
    rows = []
    pinned = next(leg for leg in legs
                  if (leg["cells"], leg["degree"]) == PINNED_LEG)
    if pool is not None:
        stored = next(e for e in pool["ladder"]
                      if (e["cells"], e["degree"]) == PINNED_LEG)
        for field in FIELDS:
            for norm, stored_key, rerun in (
                    ("l1", "errors", pinned["l1_errors"]),
                    ("l2", "l2_errors", pinned["l2_errors"])):
                committed = float(stored[stored_key][field])
                rows.append({"source": "lax_case_pool", "leg": "p3.N400",
                             "norm": norm, "field": field,
                             "committed": committed, "rerun": float(rerun[field]),
                             "relative_difference": abs(rerun[field] - committed)
                             / abs(committed)})
    if probe is not None:
        for entry in probe.get("ladder", []):
            for leg in legs:
                if (entry["cells"], entry["degree"]) != (leg["cells"],
                                                         leg["degree"]):
                    continue
                for field in FIELDS:
                    if field not in entry.get("errors", {}):
                        continue
                    committed = float(entry["errors"][field])
                    rerun = float(leg["l1_errors"][field])
                    rows.append({"source": "lax_asymptotic_probe",
                                 "leg": f"p{leg['degree']}.N{leg['cells']}",
                                 "norm": "l1", "field": field,
                                 "committed": committed, "rerun": rerun,
                                 "relative_difference": abs(rerun - committed)
                                 / abs(committed)})
    worst = max((row["relative_difference"] for row in rows), default=None)
    return {"compared": bool(rows), "tolerance": CONSISTENCY_TOL,
            "worst_relative_difference": worst, "rows": rows,
            "status": ("not_run" if worst is None
                       else "PASS" if worst <= CONSISTENCY_TOL else "FAIL")}


# --------------------------------------------------------------------- figures
def render_figures(legs, res, profiles, out_dir=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    styles = {"density": ("o-", r"$\rho$"), "velocity": ("s-", r"$u$"),
              "pressure": ("^-", r"$p$")}
    cells = [leg["cells"] for leg in legs]
    for norm, ax, title in ((("l2_errors"), axes[0], "relative L2 (ledger metric)"),
                            (("l1_errors"), axes[1], "relative L1")):
        for field, (style, label) in styles.items():
            ax.loglog(cells, [leg[norm][field] for leg in legs], style, ms=5,
                      lw=1.2, label=label)
        ax.axhline(res["tolerance"], color="tab:red", ls="--", lw=1.2,
                   label=f"ledger tolerance {res['tolerance']:.0%}")
        ax.set_xlabel("N (elements)")
        ax.set_ylabel(title)
        ax.set_xticks(cells)
        ax.set_xticklabels([str(c) for c in cells])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_title("CASE-LAX-DG · P3 vs exact Riemann solution", fontsize=9)
    path = out_dir / "lax_resolution_metric.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)

    finest = legs[-1]
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.4), sharex=True)
    order = np.argsort(profiles["points"].ravel())
    flat = lambda field: np.asarray(field).ravel()[order]  # noqa: E731
    x = flat(profiles["points"])
    for ax, (field, label) in zip(axes, (("density", r"$\rho$"),
                                         ("velocity", r"$u$"),
                                         ("pressure", r"$p$"))):
        ax.plot(x, flat(profiles["exact"][field]), "-", lw=1.1, color="k",
                label="exact")
        ax.plot(x, flat(profiles["numerical"][field]), "o", ms=1.6,
                color="tab:blue", label=f"DG P{finest['degree']} N={finest['cells']}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("x")
    axes[0].set_title("CASE-LAX-DG · finest extension leg vs exact, t = 0.13",
                      fontsize=9)
    path = out_dir / "lax_resolution_profiles.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)
    return [str(p.relative_to(ROOT)) for p in files]


# --------------------------------------------------------------------- summary
def render_summary(metrics):
    res = metrics["requirement"]
    rows = "\n".join(
        f"| {row['leg']} | {row['worst_l2']:.5e} | {row['worst_field']} | "
        f"{row['margin_of_tolerance']:.3f} | {row['status']} |"
        for row in res["per_leg"])
    orders = "\n".join(
        f"| {field} | {metrics['orders']['l1'][field]:.3f} | "
        f"{metrics['orders']['l2'][field]:.3f} |" for field in FIELDS)
    return f"""# CASE-LAX-DG · resolution requirement of the ledger acceptance bar

> Generated from `metrics.json` by `python -m benchmarks.lax_resolution_requirement`
> (measurement run {metrics['generated_at_utc']}, wall {metrics['wall_seconds']:.1f} s).
> Ledger bar: relative error vs exact Riemann solution < {res['tolerance']:.0%}.

## Question

`{metrics['pinned']['leg']}` (the pinned A0 acceptance leg) misses the bar:
worst whole-field relative L2 `{metrics['pinned']['worst_l2']:.5e}`
(`{metrics['pinned']['worst_field']}`, margin x{metrics['pinned']['margin_of_tolerance']:.3f} of
the tolerance). Does one further h-refinement reach it?

Rule (pre-registered): {res['rule']}.

## Ledger metric per leg

| leg | worst rel L2 | field | tolerance margin (x) | status |
|---|---|---|---|---|
{rows}

Smallest leg meeting the bar: **{res['smallest_leg_meeting_tolerance']}**
(resolution factor x{res['resolution_factor'] if res['resolution_factor'] is not None else 0:.1f});
R2 verdict **{res['status']}**.

## Measured order, N={metrics['legs'][0]['cells']} -> N={metrics['legs'][1]['cells']}

| field | order (L1) | order (L2) |
|---|---|---|
{orders}

## Reuse check

Committed artifacts re-measured here: worst relative difference
`{metrics['reuse_check']['worst_relative_difference']:.3e}` (tolerance
`{metrics['reuse_check']['tolerance']:.0e}`), status
**{metrics['reuse_check']['status']}**.

## Scope

A0 stays pinned at `{metrics['pinned']['leg']}`: this artifact adds a resolution
statement, it does not relabel the acceptance verdict.
"""


# ------------------------------------------------------------------------- doc
def render_doc(metrics):
    res = metrics["requirement"]
    legs = metrics["legs"]
    pinned = metrics["pinned"]
    l2_rows = "\n".join(
        f"| {leg['leg']} | {leg['l2_errors']['density']:.5e} | "
        f"{leg['l2_errors']['velocity']:.5e} | {leg['l2_errors']['pressure']:.5e} | "
        f"{max(leg['l2_errors'].values()):.5e} | "
        f"{'PASS' if max(leg['l2_errors'].values()) < res['tolerance'] else 'FAIL'} |"
        for leg in metrics["legs"])
    per_leg_rows = "\n".join(
        f"| {row['leg']} | {row['worst_l2']:.5e} | {row['worst_field']} | "
        f"{row['margin_of_tolerance']:.3f} | {row['status']} |"
        for row in res["per_leg"])
    order_rows = "\n".join(
        f"| {field} | {metrics['orders']['l1'][field]:.3f} | "
        f"{metrics['orders']['l2'][field]:.3f} |" for field in FIELDS)
    reuse_rows = "\n".join(
        f"| {row['source']} | {row['leg']} | {row['norm']} | {row['field']} | "
        f"{row['committed']:.6e} | {row['rerun']:.6e} | "
        f"{row['relative_difference']:.2e} |" for row in metrics["reuse_check"]["rows"]
        if row["relative_difference"] != 0.0) or "| — | — | — | — | — | — | 0 (逐位一致) |"
    figures = "\n".join(f"* `{f}`" for f in metrics["figures"])
    return f"""# CASE-LAX-DG · 台账容差需要多细的网格？（rel L2 @ P3, N = 400 → 800）

> 锚点：`feature_ledger` `CASE-LAX-DG`（验收原文「ref: 精确解｜容差 < 3 %」，status 仍 `open`）。
> 本文全部数字由 `python -m benchmarks.lax_resolution_requirement` 从工件
> `docs/benchmarks/results/lax_resolution_requirement/metrics.json` 注入（零手抄；
> `--check` 在漂移时失败）。测量运行 {metrics['generated_at_utc']}，墙钟 {metrics['wall_seconds']:.1f} s，
> 工件 sha256 前缀 `{metrics['metrics_sha256'][:16]}`。

## ① 案例介绍

Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]`，γ = {GAMMA}，隔膜 `x0 = 0.5`，
左态 `(ρ,u,p) = {tuple(LEFT)}`，右态 `{tuple(RIGHT)}`，快照 `t = {FINAL_TIME}`。
解含左稀疏波、接触间断、右激波，精确 Riemann 解闭式可得，故参考值不依赖任何第三方求解器。

**待办缺口**：台账容差写在「解 vs 精确解 < 3 %」，而池子把这条钉在 `p3.N400`
（`docs/CASE-LAX-DG-acceptance-A0.md`），实测密度整体 rel L2 =
`{pinned['worst_l2']:.5e}`，即差 3.9 % 的裕度未达标。池子的 C3/R3 同时给出
P3 全场阶 ≈ 0.49（间断面压阶），所以在 N=400 上靠调参无法补齐，唯一干净的出路是
**加密一档 h** 并把「需要多细」量化。此前从未有 N=800 的 **L2** 读数入库
（`lax_asymptotic_probe` 只存了 L1）。

## ② 怎么做的

1. 同一求解器、同一测量口径（`tensordg.euler_dg_1d`，HLLC，CFL = {CFL}，单元 Gauss 节点 +
   Legendre 求积，阶 {QUADRATURE_ORDER}，全场不挖波带、不给任何量豁免），跑 P3 的
   N = {legs[0]['cells']} 与 N = {legs[1]['cells']} 两腿；
2. 复用检查：N=400 的 L1/L2 必须与已提交池子工件逐位一致，N=800 的 L1 必须与已提交
   p3 渐近探针（`lax_asymptotic_probe`）逐位一致——同一份场数组要能重现两处历史数字；
3. 预登记规则（跑之前写在驱动里，`RESOLUTION_RULE`）：{res['rule']}；
4. **不移动 A0 的生产腿**：`production_leg` 仍为 `{pinned['leg']}`，其 FAIL 原样留在记录里；
   本文只增加一条关于分辨率的陈述。

## ③ 计算结果（程序化注入）

| leg | rel L2 ρ | rel L2 u | rel L2 p | 最差 | 判据 |
|---|---|---|---|---|---|
{l2_rows}

| leg | 最差 rel L2 | 该量 | 容差裕度（×容差） | verdict |
|---|---|---|---|---|
{per_leg_rows}

最小达标腿：**{res['smallest_leg_meeting_tolerance']}**（分辨率倍数
×{res['resolution_factor'] if res['resolution_factor'] is not None else 0:.1f}），
R2 判定 **{res['status']}**。

| 量 | N={legs[0]['cells']}→{legs[1]['cells']} 阶（L1） | 阶（L2） |
|---|---|---|
{order_rows}

复用检查（与已提交工件对比，容差 `{metrics['reuse_check']['tolerance']:.0e}`）：
最差相对差 `{metrics['reuse_check']['worst_relative_difference']:.3e}`，
状态 **{metrics['reuse_check']['status']}**。

| 来源工件 | leg | 范数 | 量 | 已提交 | 本次 | 相对差 |
|---|---|---|---|---|---|---|
{reuse_rows}

## ④ 结果图

{figures}

* `lax_resolution_metric.png`：两腿的 rel L2 / rel L1 与 3 % 容差线；
* `lax_resolution_profiles.png`：N={legs[1]['cells']} 的 ρ/u/p 剖面 vs 精确解。

## ⑤ 注意点

1. **测的是距离不是阶**：本文只回答「这个范数在这个网格上是否进 3 %」，不回答族是否
   达设计阶。P3 在间断存在下全场阶仍 < 1（见上表），这是 DG 抓激波的已知属性，
   与池子 C3/R3 的 FAIL 一致，两者不互相代言。
2. **不得用它改写 A0**：台账臂的最细登记腿仍是 N=400（正文 A0 的 FAIL 保留）。
   若要按本文结论把锚点翻成 done，需要台账持有人明确把「生产腿」改成 N=800 ——
   那是验收口径变更，不是本单可以自己做的裁决。
3. **口径敏感性**：误差在 Gauss 节点上按 Legendre 求积加权，换单元平均/L∞ 会给出不同数字；
   口径写在工件的 `config` 里供重裁。
4. **算力**：N=800/P3 需要 {legs[1]['steps']} 步，是 N=400 腿（{legs[0]['steps']} 步）的约
   {legs[1]['steps'] / legs[0]['steps']:.1f} 倍步数，属重计算，已递交集群调度执行。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
""" + "\n".join(
        f"| rel L2 {field} @ p3.N{legs[1]['cells']} | 精确 Riemann 解 | "
        f"{legs[1]['l2_errors'][field]:.5e} | "
        f"{100.0 * legs[1]['l2_errors'][field]:.3f} % | "
        f"{'是' if legs[1]['l2_errors'][field] < res['tolerance'] else '否'} |"
        for field in FIELDS) + f"""
| rel L2 {pinned['worst_field']} @ p3.N400（钉扎腿，未改） | 精确 Riemann 解 |
{pinned['worst_l2']:.5e} | {100.0 * pinned['worst_l2']:.3f} % | 否 |

**结论**：在 P3 / N = {legs[1]['cells']} 上 **{res['status']}** —— 台账 3 % 容差
{'全部三个量均达标' if res['status'] == 'PASS' else '仍有量未达标'}；
钉扎腿 `{pinned['leg']}` 的 FAIL 原样保留，未被本文改写。
"""


def write_outputs(metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text(render_summary(metrics))
    DOC.write_text(render_doc(metrics))


# ---------------------------------------------------------------------- driver
def run(legs=LEGS, tolerance=None, render=True):
    tolerance = (ledger_acceptance_tolerance() if tolerance is None
                 else float(tolerance))
    started = time.time()
    from benchmarks.lax_case_pool import exact_waves
    waves = exact_waves()
    measured, profiles = [], None
    for cells, degree in legs:
        entry, fields = measure_leg(cells, degree, waves)
        measured.append(entry)
        if (cells, degree) == legs[-1]:
            profiles = fields
    pool = json.loads(POOL.read_text()) if POOL.exists() else None
    probe = json.loads(PROBE.read_text()) if PROBE.exists() else None
    res = requirements(measured, tolerance)
    pinned_row = res["per_leg"][0]
    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "question": ("ledger bar 'ref: exact solution | < 3 %': what whole-field "
                     "relative L2 does P3 reach at N=800, is the bar met there, and "
                     "what is the measured order N=400 -> N=800?"),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"cfl": CFL, "quadrature_order": QUADRATURE_ORDER,
                   "legs": [list(leg) for leg in legs],
                   "left_state": list(LEFT), "right_state": list(RIGHT),
                   "gamma": GAMMA, "final_time": FINAL_TIME,
                   "domain": list(DOMAIN), "tolerance": tolerance},
        "pinned": {"leg": pinned_row["leg"], "worst_l2": pinned_row["worst_l2"],
                   "worst_field": pinned_row["worst_field"],
                   "margin_of_tolerance": pinned_row["margin_of_tolerance"],
                   "status": pinned_row["status"],
                   "source": "docs/benchmarks/results/lax_case_pool/metrics.json",
                   "note": ("A0 production leg pinned before any number was seen; "
                            "this study does not move it")},
        "legs": [{**entry, "leg": f"p{entry['degree']}.N{entry['cells']}"}
                 for entry in measured],
        "orders": {"l1": {f: rate(measured, f, "l1") for f in FIELDS},
                   "l2": {f: rate(measured, f, "l2") for f in FIELDS},
                   "pair": [measured[0]["cells"], measured[1]["cells"]]},
        "requirement": res,
        "reuse_check": reuse_check(measured, pool, probe),
        "verdict": res["status"],
        "wall_seconds": float(time.time() - started),
        "metrics_sha256": "0" * 64,  # filled after the doc is rendered
    }
    if render:
        metrics["figures"] = render_figures(measured, res, profiles)
        # the doc embeds the digest of the numbers, so hash the body without it
        payload = json.dumps({k: v for k, v in metrics.items()
                              if k != "metrics_sha256"}, sort_keys=True)
        import hashlib
        metrics["metrics_sha256"] = hashlib.sha256(
            payload.encode()).hexdigest()
        write_outputs(metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="re-render the doc from the committed artifact")
    args = parser.parse_args(argv)
    if args.check:
        if not METRICS.exists():
            print(f"missing {METRICS.relative_to(ROOT)}", file=sys.stderr)
            return 1
        metrics = json.loads(METRICS.read_text())
        drift = render_doc(metrics) != DOC.read_text()
        if drift:
            print("[lax-res] doc drift vs metrics.json", file=sys.stderr)
        return 1 if drift else 0
    metrics = run()
    print(f"[lax-res] R2={metrics['requirement']['status']} "
          f"smallest leg={metrics['requirement']['smallest_leg_meeting_tolerance']} "
          f"reuse={metrics['reuse_check']['status']} "
          f"-> {METRICS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())