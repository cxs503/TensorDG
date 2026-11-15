"""CASE-LAX-DG -- does the certified resolution claim survive a temporal budget?

The question
------------
Two numbers stand in the case-pool record (``docs/benchmarks/results/
lax_case_pool/metrics.json``):

* the **pinned A0 leg** ``p3.N400`` misses the ledger bar on density
  (whole-field relative L2 ``3.11556e-02`` vs ``< 3 %``);
* the **resolution statement** (``docs/CASE-LAX-DG-resolution-requirement.md``)
  shows ``p3.N800`` meets it (``2.31490e-02``).

Both are quoted at one time step (``CFL = 0.35``, SSP-RK3).  The pool does carry
a temporal gate (``benchmarks/lax_case_pool.py::gate_time``, tolerance
``TOL_TIME_ERROR_SHARE = 5 %`` of the baseline-CFL error, from three exact
``dt`` halvings on a fixed mesh) -- but it has only ever been evaluated on the
cheap legs ``TIME_LEGS = ((100,1),(200,2),(200,3))`` and on the **L1** norm,
i.e. on neither of the two legs above, and on neither of the two norms the
ledger bar is stated in (relative L2).

So the record has an unmeasured confound: neither the PASS nor the FAIL at the
certified resolutions has a temporal budget.  If the ``dt`` term were a
non-trivial part of ``2.31490e-02`` the PASS would be partly an artifact of
``dt``; if the pinned FAIL could be removed by shrinking ``dt`` alone the FAIL
would be mis-attributed to resolution.

Pre-registered rule (fixed before any run, ``TIME_BUDGET_RULE`` below)
---------------------------------------------------------------------
T1  temporal budget -- for every leg and every field, the temporal share of
    the baseline-CFL **relative L2** error must be below the pool's registered
    5 % (order-identifiable -> the model share ``temporal_share``; otherwise
    the measured 4x-``dt`` sensitivity ``temporal_bound``, which is the honest
    bound when the series is flat).
T2  the resolution PASS is dt-robust -- at **every** CFL level, ``p3.N800``
    max-field relative L2 < the ledger tolerance.
T3  the pinned FAIL is not a dt artifact -- at **every** CFL level,
    ``p3.N400`` max-field relative L2 >= the ledger tolerance.
T4  reuse -- the baseline-CFL L2 readings reproduce the committed
    ``lax_resolution_requirement`` artifact.
T5  the refinement actually refines -- ``steps`` and ``dt_mean = T/steps``
    halve with the CFL divisor (ratio in [1.9, 2.1] per halving).

Nothing here re-points the A0 acceptance leg or edits a committed number: this
driver writes its own artifact.

Artifacts (inside the worktree)::

    docs/benchmarks/results/lax_time_budget_certified/metrics.json
    docs/benchmarks/results/lax_time_budget_certified/summary.md
    docs/benchmarks/results/lax_time_budget_certified/profiles.npz
    docs/benchmarks/results/lax_time_budget_certified/figures/*.png
    docs/CASE-LAX-DG-time-budget-certified-legs.md      (rendered from metrics)

Usage::

    python -m benchmarks.lax_time_budget_certified               # full study
    python -m benchmarks.lax_time_budget_certified --check       # doc drift
    python -m benchmarks.lax_time_budget_certified --figures-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.lax_case_pool import (
    CFL, DOMAIN, FINAL_TIME, GAMMA, LEFT, RIGHT, TOL_TIME_ERROR_SHARE,
    _weighted_l1, _weighted_l2, exact_reference, ledger_acceptance_tolerance,
    modal_fields, temporal_bound, temporal_order, temporal_share)
from tensordg.euler_dg_1d import solve_euler_dg_1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_time_budget_certified"
FIG = OUT / "figures"
METRICS = OUT / "metrics.json"
PROFILES = OUT / "profiles.npz"
DOC = ROOT / "docs" / "CASE-LAX-DG-time-budget-certified-legs.md"
RESOLUTION = ROOT / "docs" / "benchmarks" / "results" / "lax_resolution_requirement" / "metrics.json"

FIELDS = ("density", "velocity", "pressure")
#: the two legs the certified record stands on: the pinned A0 leg and the
#: smallest leg of the resolution ladder that meets the bar.  Fixed before the run.
LEGS = ((400, 3), (800, 3))
#: exact halvings of dt at fixed mesh -- same divisors the pool's own temporal
#: study registers (``TIME_CFL_DIVISORS``), so the readings are comparable.
DIVISORS = (1, 2, 4)
QUADRATURE_ORDER = 10
REUSE_TOL = 1e-9
STEP_RATIO_BAND = (1.9, 2.1)

TIME_BUDGET_RULE = (
    "T1 temporal share of the baseline-CFL whole-field relative L2 error < 5 % "
    "for every leg and every field (order-identifiable -> temporal_share, else "
    "the measured 4x-dt bound); T2 p3.N800 max-field rel L2 < ledger tolerance "
    "at every CFL level; T3 p3.N400 max-field rel L2 >= ledger tolerance at "
    "every CFL level (the FAIL is not a dt artifact); T4 baseline-CFL L2 "
    "reproduces the committed resolution artifact; T5 steps and dt_mean halve "
    "with the CFL divisor (ratio in [1.9, 2.1]). Overall PASS iff T1-T5 hold.")


# ------------------------------------------------------------------ measuring
def measure_level(cells, degree, divisor):
    """One (mesh, dt) point, measured with the case pool's own helpers."""
    started = time.time()
    cfl = CFL / divisor
    x, coeff, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=cells, degree=degree,
                                       final_time=FINAL_TIME, cfl=cfl)
    points, primitive, dx, weights = modal_fields(x, coeff, QUADRATURE_ORDER)
    exact = exact_reference(points)
    l2 = {f: _weighted_l2(points, primitive[f], exact[f], weights, dx) for f in FIELDS}
    l1 = {f: _weighted_l1(points, primitive[f], exact[f], weights, dx) for f in FIELDS}
    entry = {"cells": int(cells), "degree": int(degree), "cfl": float(cfl),
             "cfl_divisor": int(divisor), "steps": int(meta["steps"]),
             "dt_mean": float(meta.get("dt_mean", FINAL_TIME / max(meta["steps"], 1))),
             "dt_last_step": float(meta["dt"]),
             "l2_errors": l2, "l1_errors": l1,
             "density_min": float(primitive["density"].min()),
             "pressure_min": float(primitive["pressure"].min()),
             "wall_seconds": float(time.time() - started)}
    print(f"[lax-tbudget] N={cells:4d} p={degree} cfl={cfl:.5f} "
          f"L2(rho)={l2['density']:.6e} steps={entry['steps']} "
          f"t={entry['wall_seconds']:.1f}s", flush=True)
    cache = {"points": points.ravel(), "dx": dx, "weights": weights.ravel(),
             "exact": {f: np.asarray(exact[f]).ravel() for f in FIELDS},
             "numerical": {f: np.asarray(primitive[f]).ravel() for f in FIELDS}}
    return entry, cache


def temporal_fields(levels):
    """Pool-registered temporal diagnostics, applied to the L2 series."""
    fields = {}
    for f in FIELDS:
        series = [level["l2_errors"][f] for level in levels]
        order = temporal_order(series)
        fields[f] = {"relative_l2": series, "temporal_order": order,
                     "temporal_share": temporal_share(series, order),
                     "temporal_bound": temporal_bound(series)}
    return fields


def temporal_budget(fields):
    """T1, per legacy the pool's own reading: share if identifiable, else bound."""
    worst, worst_where = 0.0, None
    for f, data in fields.items():
        candidate = data["temporal_share"]
        reading = "share"
        if candidate is None:
            candidate, reading = data["temporal_bound"], "bound"
        if candidate is None:
            continue
        if candidate > worst:
            worst, worst_where = float(candidate), {"field": f, "reading": reading}
    return {"tolerance": TOL_TIME_ERROR_SHARE, "worst": worst,
            "worst_where": worst_where,
            "status": "PASS" if worst_where is not None and worst < TOL_TIME_ERROR_SHARE
                      else "FAIL"}


# --------------------------------------------------------------------- checks
def check_ledger_verdicts(rows, tolerance):
    """T2 / T3: is the PASS/FAIL at each leg stable under dt refinement?"""
    out = {}
    for row in rows:
        key = f"p{row['degree']}.N{row['cells']}"
        per_level = {}
        for level in row["levels"]:
            worst_field, worst = max(((f, level["l2_errors"][f]) for f in FIELDS),
                                     key=lambda item: item[1])
            per_level[f"cfl={level['cfl']:.5f}"] = {
                "worst_l2": float(worst), "worst_field": worst_field,
                "margin_of_tolerance": float(worst / tolerance),
                "status": "PASS" if worst < tolerance else "FAIL"}
        statuses = {entry["status"] for entry in per_level.values()}
        out[key] = {"per_level": per_level,
                    "stable": len(statuses) == 1,
                    "status_at_baseline_cfl": per_level[f"cfl={row['levels'][0]['cfl']:.5f}"]["status"],
                    "uniform_status": (statuses.pop() if len(statuses) == 1 else "MIXED")}
    t2 = ("PASS" if all(entry["uniform_status"] == "PASS"
                        for key, entry in out.items() if key.endswith("N800")) else "FAIL")
    t3 = ("PASS" if all(entry["uniform_status"] == "FAIL"
                        for key, entry in out.items() if key.endswith("N400")) else "FAIL")
    return {"per_leg": out,
            "T2_resolution_pass_is_dt_robust": t2,
            "T3_pinned_fail_is_not_dt_artifact": t3}


def check_reuse(rows):
    """T4: the baseline-CFL readings must reproduce the committed artifact."""
    if not RESOLUTION.exists():
        return {"source": str(RESOLUTION.relative_to(ROOT)), "rows": [],
                "worst_relative_difference": None, "tolerance": REUSE_TOL,
                "status": "not_run"}
    committed = json.loads(RESOLUTION.read_text())
    by_leg = {f"p{leg['degree']}.N{leg['cells']}": leg for leg in committed["legs"]}
    rows_out, worst, skipped = [], 0.0, []
    for row in rows:
        key = f"p{row['degree']}.N{row['cells']}"
        baseline = row["levels"][0]
        if key not in by_leg:
            skipped.append(key)
            continue
        for f in FIELDS:
            a = float(by_leg[key]["l2_errors"][f])
            b = float(baseline["l2_errors"][f])
            diff = abs(a - b) / max(abs(a), 1e-300)
            worst = max(worst, diff)
            rows_out.append({"leg": key, "field": f, "committed": a,
                             "rerun": b, "relative_difference": diff})
    return {"source": str(RESOLUTION.relative_to(ROOT)), "rows": rows_out,
            "worst_relative_difference": worst, "tolerance": REUSE_TOL,
            "skipped_legs": skipped,
            "status": ("PASS" if worst <= REUSE_TOL else "FAIL") if rows_out else "not_run"}


def check_dt_refinement(rows):
    """T5: steps and the mean step must halve with the CFL divisor."""
    out, ok = {}, True
    for row in rows:
        key = f"p{row['degree']}.N{row['cells']}"
        steps = [level["steps"] for level in row["levels"]]
        dt_mean = [level["dt_mean"] for level in row["levels"]]
        step_ratios = [steps[i + 1] / steps[i] for i in range(len(steps) - 1)]
        dt_ratios = [dt_mean[i] / dt_mean[i + 1] for i in range(len(dt_mean) - 1)]
        inside = all(STEP_RATIO_BAND[0] <= r <= STEP_RATIO_BAND[1]
                     for r in step_ratios + dt_ratios)
        ok = ok and inside
        out[key] = {"steps": steps, "dt_mean": dt_mean,
                    "step_ratios": step_ratios, "dt_mean_ratios": dt_ratios,
                    "band": list(STEP_RATIO_BAND),
                    "status": "PASS" if inside else "FAIL"}
    return {"per_leg": out, "status": "PASS" if ok else "FAIL"}


# ------------------------------------------------------------------- figures
def _rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def render_figures(rows, checks, cache):
    """Two presentation figures; physics verdicts do not depend on them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tolerance = checks["ledger"]["tolerance"]
    FIG.mkdir(parents=True, exist_ok=True)
    files = []

    fig, axes = plt.subplots(1, len(rows), figsize=(4.6 * len(rows), 3.4), squeeze=False)
    for ax, row in zip(axes[0], rows):
        cfls = [level["cfl"] for level in row["levels"]]
        for f, color in zip(FIELDS, ("tab:blue", "tab:green", "tab:red")):
            ax.plot(cfls, [level["l2_errors"][f] for level in row["levels"]],
                    "o-", color=color, label=f)
        ax.axhline(tolerance, color="k", ls="--", lw=1.0,
                   label=f"ledger bar {tolerance:.0%}")
        ax.set_xscale("log")
        ax.set_xlabel("CFL")
        ax.set_ylabel("whole-field relative L2")
        ax.set_title(f"P{row['degree']} N={row['cells']} · dt refinement", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIG / "lax_time_budget_l2_vs_cfl.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    labels, values = [], []
    for row in rows:
        for f in FIELDS:
            data = row["fields"][f]
            candidate = data["temporal_share"]
            if candidate is None:
                candidate = data["temporal_bound"]
            labels.append(f"p{row['degree']}.N{row['cells']}\n{f}")
            values.append(0.0 if candidate is None else candidate)
    ax.bar(range(len(values)), values, color="tab:purple")
    ax.axhline(TOL_TIME_ERROR_SHARE, color="k", ls="--", lw=1.0,
               label=f"registered {TOL_TIME_ERROR_SHARE:.0%}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("temporal share of baseline L2 error")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_title("CASE-LAX-DG · temporal budget at the certified legs", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIG / "lax_time_budget_share.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)

    # profiles: does shrinking dt move the solution at all?  (baseline vs 4x finer)
    fig, axes = plt.subplots(1, len(rows), figsize=(4.6 * len(rows), 3.4), squeeze=False)
    for ax, row in zip(axes[0], rows):
        key_base = _cache_key(row, 0)
        key_fine = _cache_key(row, len(row["levels"]) - 1)
        base, fine = cache[key_base], cache[key_fine]
        order = np.argsort(base["points"])
        x = base["points"][order]
        ax.plot(x, base["exact"]["density"][order], "-", lw=1.4, color="k", label="exact")
        ax.plot(x, base["numerical"]["density"][order], ".", ms=2.0,
                color="tab:blue", label=f"cfl={row['levels'][0]['cfl']:.3f}")
        ax.plot(x, fine["numerical"]["density"][order], "-", lw=0.8,
                color="tab:orange", label=f"cfl={row['levels'][-1]['cfl']:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(f"P{row['degree']} N={row['cells']} · t = {FINAL_TIME}", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    path = FIG / "lax_time_budget_profiles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    files.append(path)
    return [_rel(p) for p in files]


def _cache_key(row, index):
    return f"p{row['degree']}_N{row['cells']}_d{row['levels'][index]['cfl_divisor']}"


def save_profiles(rows, cache):
    PROFILES.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for row in rows:
        for index, level in enumerate(row["levels"]):
            key = _cache_key(row, index)
            entry = cache[key]
            payload[f"{key}_points"] = entry["points"]
            payload[f"{key}_weights"] = entry["weights"]
            payload[f"{key}_dx"] = np.array([entry["dx"]])
            for f in FIELDS:
                payload[f"{key}_exact_{f}"] = entry["exact"][f]
                payload[f"{key}_numerical_{f}"] = entry["numerical"][f]
    np.savez_compressed(PROFILES, **payload)


def load_profiles():
    data = np.load(PROFILES)
    cache = {}
    for name in data.files:
        if not name.endswith("_points"):
            continue
        key = name[: -len("_points")]
        cache[key] = {"points": data[f"{key}_points"], "weights": data[f"{key}_weights"],
                      "dx": float(data[f"{key}_dx"][0]),
                      "exact": {f: data[f"{key}_exact_{f}"] for f in FIELDS},
                      "numerical": {f: data[f"{key}_numerical_{f}"] for f in FIELDS}}
    return cache


# --------------------------------------------------------------- doc / summary
def _fmt(value, spec, missing="not_run"):
    return missing if value is None else format(value, spec)


def digest(metrics):
    payload = json.dumps({k: v for k, v in metrics.items() if k != "metrics_sha256"},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def render_summary(metrics):
    rows = []
    for row in metrics["legs"]:
        budget = metrics["checks"]["temporal_budget"]["per_leg"][row["leg"]]
        rows.append(f"| {row['leg']} | {row['levels'][0]['l2_errors']['density']:.5e} | "
                    f"{row['levels'][-1]['l2_errors']['density']:.5e} | "
                    f"{budget['worst']:.3e} | {budget['status']} |")
    body = "\n".join(rows)
    checks = metrics["checks"]
    return f"""# CASE-LAX-DG · temporal budget at the certified legs

> Generated from `metrics.json` (measurement run {metrics['generated_at_utc']},
> wall {metrics['wall_seconds']:.1f} s). Rule: {metrics['rule']}

| leg | rel L2 ρ @ cfl={metrics['config']['cfl']} | rel L2 ρ @ cfl={metrics['legs'][-1]['levels'][-1]['cfl']:.5f} | temporal share (worst field) | T1 |
|---|---|---|---|---|
{body}

* T1 temporal budget: **{checks['temporal_budget']['status']}**
* T2 resolution PASS is dt-robust: **{checks['ledger']['T2_resolution_pass_is_dt_robust']}**
* T3 pinned FAIL is not a dt artifact: **{checks['ledger']['T3_pinned_fail_is_not_dt_artifact']}**
* T4 reuse of committed baseline: **{checks['reuse']['status']}**
* T5 dt refinement actually refines: **{checks['dt_refinement']['status']}**

Overall: **{metrics['verdict']}**
"""


def leg_by_name(metrics, name):
    for row in metrics["legs"]:
        if row["leg"] == name:
            return row
    return None


def render_doc(metrics):
    tol = metrics['config']['tolerance']
    legs_label = " / ".join(f"P{row['degree']} N = {row['cells']}" for row in metrics["legs"])
    t1 = metrics['checks']['temporal_budget']
    budgets = t1.get("per_leg") or {}
    budget = (max(budgets.values(), key=lambda b: b["worst"]) if budgets else
              {"worst": float("nan"), "worst_where": None, "status": "not_run",
               "tolerance": TOL_TIME_ERROR_SHARE})
    ledger = metrics['checks']['ledger']

    level_rows = []
    for row in metrics["legs"]:
        for level in row["levels"]:
            worst_field, worst = max(((f, level["l2_errors"][f]) for f in FIELDS),
                                     key=lambda item: item[1])
            level_rows.append(
                f"| {row['leg']} | {level['cfl']:.5f} | {level['steps']} | "
                f"{level['dt_mean']:.4e} | {level['l2_errors']['density']:.5e} | "
                f"{level['l2_errors']['velocity']:.5e} | "
                f"{level['l2_errors']['pressure']:.5e} | {worst:.5e} (`{worst_field}`) | "
                f"{'PASS' if worst < tol else 'FAIL'} |")
    level_table = "\n".join(level_rows)

    field_rows = []
    for row in metrics["legs"]:
        for f in FIELDS:
            data = row["fields"][f]
            share = data["temporal_share"]
            bound = data["temporal_bound"]
            reading = (f"share {share:.3e}" if share is not None
                       else f"bound {bound:.3e}" if bound is not None else "not_run")
            field_rows.append(
                f"| {row['leg']} | {f} | {data['relative_l2'][0]:.6e} | "
                f"{data['relative_l2'][-1]:.6e} | "
                f"{'null（不可辨识）' if data['temporal_order'] is None else format(data['temporal_order'], '.3f')} | {reading} | "
                f"{'PASS' if (share if share is not None else bound or 1.0) < TOL_TIME_ERROR_SHARE else 'FAIL'} |")
    field_table = "\n".join(field_rows)

    verdict_rows = []
    for key, entry in ledger["per_leg"].items():
        for level_key, level in entry["per_level"].items():
            verdict_rows.append(
                f"| {key} | {level_key} | {level['worst_l2']:.5e} | "
                f"{level['worst_field']} | {level['margin_of_tolerance']:.3f} | "
                f"{level['status']} |")
    verdict_table = "\n".join(verdict_rows)

    reuse = metrics["checks"]["reuse"]
    reuse_row = "\n".join(
        f"| {r['leg']} | {r['field']} | {r['committed']:.6e} | {r['rerun']:.6e} | "
        f"{r['relative_difference']:.2e} |" for r in reuse["rows"]) or "| — | — | — | — | not_run |"

    dt_rows = "\n".join(
        f"| {key} | {entry['steps']} | "
        f"{', '.join(f'{r:.4f}' for r in entry['step_ratios'])} | "
        f"{', '.join(f'{r:.4f}' for r in entry['dt_mean_ratios'])} | {entry['status']} |"
        for key, entry in metrics["checks"]["dt_refinement"]["per_leg"].items())

    if metrics.get("figures"):
        figures = "\n".join(f"* `{f}`" for f in metrics["figures"])
        figures += ("\n\n* `lax_time_budget_l2_vs_cfl.png`：两腿的 rel L2（ρ/u/p）随 dt 细化，"
                    "虚线为台账 3 % 容差；\n"
                    "* `lax_time_budget_share.png`：各腿各量的时间误差份额 vs 5 % 注册门；\n"
                    "* `lax_time_budget_profiles.png`：基线 CFL 与 4× 细化 dt 的密度剖面叠加（vs 精确解），"
                    "看细化 dt 是否真的移动了场。")
        figures += f"\n\n渲染来源：{metrics.get('figures_note', 'n/a')}。"
    else:
        figures = ("* 本工件未附图：" + metrics.get("figures_note", "figures missing")
                   + "。判定不依赖图。")

    worst_field_name = budget["worst_where"]["field"] if budget["worst_where"] else "not_run"
    worst_field_reading = budget["worst_where"]["reading"] if budget["worst_where"] else "not_run"
    n800 = ledger["per_leg"].get("p3.N800", {})
    n400 = ledger["per_leg"].get("p3.N400", {})
    leg800 = leg_by_name(metrics, "p3.N800") or metrics["legs"][-1]
    leg400 = leg_by_name(metrics, "p3.N400") or metrics["legs"][0]
    fine800 = leg800["levels"][-1]
    base400 = leg400["levels"][0]
    return f"""# CASE-LAX-DG · 认证腿上的时间误差预算（{legs_label}）

> 锚点：`feature_ledger` `CASE-LAX-DG`（验收原文「ref: 精确解｜容差 < 3 %」，status 仍 `open`）。
> 本文全部数字由 `python -m benchmarks.lax_time_budget_certified` 从工件
> `docs/benchmarks/results/lax_time_budget_certified/metrics.json` 注入（零手抄；
> `--check` 在漂移时失败）。测量运行 {metrics['generated_at_utc']}，墙钟 {metrics['wall_seconds']:.1f} s，
> 工件 sha256 前缀 `{metrics['metrics_sha256'][:16]}`。
> 总体判定：**{metrics['verdict']}**。

## ① 案例介绍

Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]`，γ = {metrics['config']['gamma']}，隔膜 `x0 = 0.5`，
左态 `(ρ,u,p) = {tuple(metrics['config']['left_state'])}`，右态 `{tuple(metrics['config']['right_state'])}`，
快照 `t = {metrics['config']['final_time']}`。参考值 = 精确 Riemann 解（闭式，不依赖任何第三方求解器）。

**待办缺口（本单要回答的）**：算例池记录里两条相反方向的关键读数——钉扎腿 `p3.N400`
密度 rel L2 = `{n400.get('per_level', {}).get('cfl=' + format(metrics['config']['cfl'], '.5f'), {}).get('worst_l2', float('nan')):.5e}`（未达标）、
分辨率声明腿 `p3.N800` = `{n800.get('per_level', {}).get('cfl=' + format(metrics['config']['cfl'], '.5f'), {}).get('worst_l2', float('nan')):.5e}`（达标）
——**都是在单一 CFL = {metrics['config']['cfl']} 下取的**，而池子的时间门 `gate_time`
（注册容差 {TOL_TIME_ERROR_SHARE:.0%}）从未在这两条腿上、也从未在台账所用的 rel L2 范数上评过。
若不测，「N=800 达标」有被 `dt` 误差注水的风险，而「N=400 不达标」有被误判为分辨率问题的风险。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. 网格固定（P3, N = 400 与 800），只动 `dt`：CFL ∈ {{ {', '.join(f"{metrics['config']['cfl']/d:.5f}" for d in metrics['config']['divisors'])} }}，
   即三次精确 2× 细化（与池子 `TIME_CFL_DIVISORS = (1,2,4)` 同口径）。
2. 误差口径沿用池子本身：单元 Gauss 节点（求积阶 {metrics['config']['quadrature_order']}）、Legendre 求积加权、
   全场、不挖波带、不给任何量豁免；参考值由 `benchmarks.lax_case_pool.exact_reference` 供给。
3. 时间诊断直接复用池子的注册实现：`temporal_order`（三次精确对分给出闭式阶，不做拟合）、
   `temporal_share`（可辨识时的模型份额）、`temporal_bound`（序列平/非单调时的 4× dt 实测敏感度，作为诚实上界）。
4. 预登记规则（跑之前写死在驱动里 `TIME_BUDGET_RULE`）：{metrics['rule']}
5. **不改任何已提交数字、不移动 A0 生产腿**：本文只在独立工件里追加时间维度的陈述。

## ③ 计算结果（程序化注入）

### 3.1 全部测量点（rel L2，台账度量）

| leg | CFL | steps | dt_mean | rel L2 ρ | rel L2 u | rel L2 p | 最差 | 对 {tol:.0%} 判据 |
|---|---|---|---|---|---|---|---|---|
{level_table}

### 3.2 时间误差预算（T1）

| leg | 量 | rel L2 @ 最粗 dt | rel L2 @ 最细 dt | 观测时间阶 | 时间份额/上界 | T1 |
|---|---|---|---|---|---|---|
{field_table}

最坏读数（两腿取最坏）：`{worst_field_name}`，按 **{worst_field_reading}** 读 =
`{budget['worst']:.3e}`，注册门 `{budget['tolerance']:.0%}` → 整体 T1 = **{t1['status']}**。

### 3.3 台账判定的 dt 稳健性（T2 / T3）

| leg | 层级 | 最差 rel L2 | 最差量 | 容差裕度（×） | verdict |
|---|---|---|---|---|---|
{verdict_table}

* T2「N = 800 达标不是 dt 假象」：**{ledger['T2_resolution_pass_is_dt_robust']}**
  （三档 CFL 的 verdict 一致性：`{n800.get('uniform_status', 'not_run')}`）；
* T3「N = 400 不达标不是时间步问题」：**{ledger['T3_pinned_fail_is_not_dt_artifact']}**
  （`{n400.get('uniform_status', 'not_run')}`）。

### 3.4 复用检查（T4）与细化有效性（T5）

基线 CFL 的读数 vs 已提交 `lax_resolution_requirement` 工件（相对容差 `{reuse['tolerance']:.0e}`）：
最差相对差 `{_fmt(reuse['worst_relative_difference'], '.3e')}` → **{reuse['status']}**。

| leg | field | 已提交 | 本次 | 相对差 |
|---|---|---|---|---|
{reuse_row}

| leg | steps | 步数比（相邻 CFL 对分） | dt_mean 比 | T5 |
|---|---|---|---|---|
{dt_rows}

## ④ 结果图

{figures}

## ⑤ 注意点

1. **只回答时间维度**：本文不重算空间阶、不重判 A0 生产腿；`p3.N400` 的 FAIL 原样保留。
   若要把锚点翻成 done，仍需台账持有人显式变更「生产腿」口径。
2. **份额 vs 上界**：激波主导的 norm 在 dt 细化下常常是平的（序列不可辨识 ⇒ `temporal_order = null`），
   此时 `temporal_share` 无定义，只能报 4× dt 实测敏感度作为上界；本文按池子注册的读法取上界，
   不把「测不出」写成「等于零」。
3. **口径敏感性**：误差在 Gauss 节点上按 Legendre 求积加权；换单元平均 / L∞ 会给出不同数字，
   口径写在工件 `config` 里供重裁。
4. **算力**：N = 800 / CFL = {fine800['cfl']:.5f} 需 {fine800['steps']} 步，
   属重计算，本机不起，已递交集群调度执行。

## ⑥ 误差判定（是否 < {tol:.0%}）

| 关键量 | 基准 | 数值 | 相对误差 | < {tol:.0%} ? |
|---|---|---|---|---|
| rel L2 density @ p3.N800, 最细 dt | 精确 Riemann 解 | {fine800['l2_errors']['density']:.5e} | {fine800['l2_errors']['density']*100:.3f} % | {'是' if fine800['l2_errors']['density'] < tol else '否'} |
| rel L2 velocity @ p3.N800, 最细 dt | 精确 Riemann 解 | {fine800['l2_errors']['velocity']:.5e} | {fine800['l2_errors']['velocity']*100:.3f} % | {'是' if fine800['l2_errors']['velocity'] < tol else '否'} |
| rel L2 pressure @ p3.N800, 最细 dt | 精确 Riemann 解 | {fine800['l2_errors']['pressure']:.5e} | {fine800['l2_errors']['pressure']*100:.3f} % | {'是' if fine800['l2_errors']['pressure'] < tol else '否'} |
| rel L2 density @ p3.N400（钉扎腿，未改） | 精确 Riemann 解 | {base400['l2_errors']['density']:.5e} | {base400['l2_errors']['density']*100:.3f} % | {'是' if base400['l2_errors']['density'] < tol else '否'} |
| 时间误差份额（最坏字段） | 池子注册门 | {budget['worst']:.3e} | {budget['worst']*100:.3f} % | {'是' if budget['worst'] < TOL_TIME_ERROR_SHARE else '否'} |

**结论**：在 P3 / N = 800 上、且在最细 dt 下，三量 rel L2 全部
{'达标（< ' + format(tol, '.0%') + '）' if max(fine800['l2_errors'].values()) < tol else '未达标'}；
钉扎腿 `p3.N400` 的 FAIL {'在三档 dt 下都不被时间细化救回' if ledger['T3_pinned_fail_is_not_dt_artifact'] == 'PASS' else '在某一档 dt 下的判定发生了变化（见 3.3）'}。
时间误差份额 {'低于' if t1['status'] == 'PASS' else '不低于'}注册门 {TOL_TIME_ERROR_SHARE:.0%}，
因此上述空间分辨率结论{'不是' if t1['status'] == 'PASS' else '可能部分是'} `dt` 误差造成的。
"""


def write_outputs(metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    (OUT / "summary.md").write_text(render_summary(metrics))
    DOC.write_text(render_doc(metrics))


# -------------------------------------------------------------------- driver
def run(legs=LEGS, divisors=DIVISORS, tolerance=None, render=True):
    tolerance = ledger_acceptance_tolerance() if tolerance is None else float(tolerance)
    started = time.time()
    rows, cache = [], {}
    for cells, degree in legs:
        level_entries = []
        for divisor in divisors:
            entry, fields = measure_level(cells, degree, divisor)
            level_entries.append(entry)
            cache[f"p{degree}_N{cells}_d{divisor}"] = fields
        rows.append({"cells": int(cells), "degree": int(degree),
                     "leg": f"p{degree}.N{cells}",
                     "levels": level_entries,
                     "fields": temporal_fields(level_entries),
                     "wall_seconds": float(sum(l["wall_seconds"] for l in level_entries))})

    budgets = {row["leg"]: temporal_budget(row["fields"]) for row in rows}
    t1 = ("PASS" if all(b["status"] == "PASS" for b in budgets.values()) else "FAIL")
    checks = {
        "temporal_budget": {"tolerance": TOL_TIME_ERROR_SHARE, "per_leg": budgets,
                            "reading": ("temporal share of the baseline-CFL whole-field "
                                        "relative L2 error, from three exact dt halvings "
                                        "on a fixed mesh (pool semantics)"),
                            "status": t1},
        "ledger": {"tolerance": tolerance, **check_ledger_verdicts(rows, tolerance)},
        "reuse": check_reuse(rows),
        "dt_refinement": check_dt_refinement(rows),
    }
    verdicts = [checks["temporal_budget"]["status"],
                checks["ledger"]["T2_resolution_pass_is_dt_robust"],
                checks["ledger"]["T3_pinned_fail_is_not_dt_artifact"],
                checks["reuse"]["status"],
                checks["dt_refinement"]["status"]]
    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "question": ("does the certified resolution record (p3.N800 PASS, p3.N400 FAIL) "
                     "survive a temporal error budget, in the ledger's relative-L2 metric?"),
        "rule": TIME_BUDGET_RULE,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"cfl": CFL, "divisors": list(divisors), "legs": [list(l) for l in legs],
                   "quadrature_order": QUADRATURE_ORDER, "gamma": GAMMA,
                   "left_state": list(LEFT), "right_state": list(RIGHT),
                   "final_time": FINAL_TIME, "domain": list(DOMAIN),
                   "tolerance": tolerance, "time_error_share": TOL_TIME_ERROR_SHARE},
        "legs": rows,
        "checks": checks,
        "verdict": "PASS" if all(v == "PASS" for v in verdicts) else "FAIL",
        "wall_seconds": float(time.time() - started),
        "metrics_sha256": "0" * 64,
    }
    if render:
        save_profiles(rows, cache)
        try:
            metrics["figures"] = render_figures(rows, checks, cache)
            metrics["figures_note"] = "rendered with matplotlib on this host"
        except ImportError as exc:  # presentation, not physics
            metrics["figures"] = []
            metrics["figures_note"] = (
                f"not rendered on the measurement host ({exc}); run "
                "`python -m benchmarks.lax_time_budget_certified --figures-only` "
                "on a host with matplotlib")
        metrics["metrics_sha256"] = digest(metrics)
        write_outputs(metrics)
    return metrics


def attach_figures():
    if not METRICS.exists():
        raise SystemExit(f"missing {METRICS.relative_to(ROOT)}")
    if not PROFILES.exists():
        raise SystemExit(f"missing {PROFILES.relative_to(ROOT)}")
    metrics = json.loads(METRICS.read_text())
    cache = load_profiles()
    metrics["figures"] = render_figures(metrics["legs"], metrics["checks"], cache)
    metrics["figures_note"] = "rendered from the committed profiles cache"
    metrics["metrics_sha256"] = digest(metrics)
    write_outputs(metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the doc still matches the committed artifact")
    parser.add_argument("--figures-only", action="store_true",
                        help="attach figures from the profiles cache (no solver run)")
    parser.add_argument("--legs", default=None,
                        help="comma list of cells:degree, e.g. 400:3,800:3 (default both)")
    args = parser.parse_args(argv)
    if args.figures_only:
        metrics = attach_figures()
        print(f"[lax-tbudget] figures attached: {metrics['figures']}")
        return 0
    if args.check:
        if not METRICS.exists():
            print(f"missing {METRICS.relative_to(ROOT)}", file=sys.stderr)
            return 1
        metrics = json.loads(METRICS.read_text())
        drift = render_doc(metrics) != DOC.read_text()
        if drift:
            print("[lax-tbudget] doc drift vs metrics.json", file=sys.stderr)
        return 1 if drift else 0
    legs = LEGS
    if args.legs:
        legs = tuple(tuple(int(v) for v in part.split(":")) for part in args.legs.split(","))
    metrics = run(legs=legs)
    print(f"[lax-tbudget] verdict={metrics['verdict']} "
          f"T1={metrics['checks']['temporal_budget']['status']} "
          f"T2={metrics['checks']['ledger']['T2_resolution_pass_is_dt_robust']} "
          f"T3={metrics['checks']['ledger']['T3_pinned_fail_is_not_dt_artifact']} "
          f"T4={metrics['checks']['reuse']['status']} "
          f"T5={metrics['checks']['dt_refinement']['status']} "
          f"-> {METRICS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(main())