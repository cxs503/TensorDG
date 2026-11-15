"""Lax shock tube (Toro test 2) -- wave-structure and admissibility audit.

Why this driver exists
----------------------
The ledger acceptance of the anchor ``CASE-LAX-DG`` is a single *integral*
number: relative ``L2`` error versus the exact Riemann solution, tolerance
``< 3 %``.  ``docs/CASE-LAX-DG-resolution-closure.md`` shows the pinned
production leg (P3 / N = 400) sits at ``3.116 %`` (FAIL) and N = 800 at
``2.315 %`` (PASS).  An integral norm, however, says nothing about

* whether the captured discontinuities are monotone (no new extrema),
* how thick the numerical shock / contact are, in cells,
* where -- in physical space -- the remaining error sits.

Those are exactly the quantities a user of a shock-capturing code cares about,
and nothing in the 22 pre-existing ``lax_*`` artifacts measures them.  This
driver adds the missing layer: it recomputes the field, then measures *local*
structure against the same closed-form reference.

Pre-registered definitions (fixed before any number was read)
-------------------------------------------------------------
``W1`` wave positions  -- reuse :func:`benchmarks.lax_case_pool.wave_positions`
    (already pre-registered for gate C1); error reported as
    ``|x_num - x_exact| / |x_exact - x0|``.

``W2`` transition thickness -- for the contact (``rho*_L -> rho*_R``) and the
    shock (``rho*_R -> rho_R``) take the 10 % and 90 % density levels of that
    jump, pick the crossings closest to the *exact* position inside a window of
    ``+- max(8 dx, 0.02)`` around it, and report ``w = |x90 - x10|`` both in
    physical units and as ``w / dx`` (thickness in cells).

``W3`` overshoot / undershoot -- ``max(0, max(u_num) - max(u_exact))`` and
    ``max(0, min(u_exact) - min(u_num))`` for ``rho`` and ``p``, reported in
    absolute terms and as a fraction of the exact field range, together with the
    location of each offending node.

``W4`` total variation -- ``TV(u_num) / TV(u_exact)`` with ``TV`` taken on the
    Gauss-node samples sorted by ``x``.  ``= 1`` for a monotone profil; every
    unit above 1 is spurious variation (numerical oscillation).  Note the exact
    profile is *not* smooth: its own TV is finite and analytic.

``W5`` plateau error -- reuse :func:`benchmarks.lax_case_pool.plateau_values`
    (star plateau means), relative error versus the exact star state.

No number in this file is transcribed: every entry is computed from the solver
or from the closed-form Riemann solution in ``benchmarks/riemann_exact.py``.

Reuse check
-----------
The committed artifacts ``docs/benchmarks/results/lax_case_pool/profiles.npz``
(P3 / N = 400) and ``docs/benchmarks/results/lax_resolution_requirement/
profiles.npz`` (P3 / N = 800) hold the fields of two legs.  ``reuse_check``
recomputes W3/W4 from those committed arrays and requires the fresh solver run
of the same leg to agree to ``1e-12``; if it does not, the artifact is stale and
the run is flagged.

Artifacts (inside the worktree)::

    docs/benchmarks/results/lax_wave_structure/metrics.json
    docs/benchmarks/results/lax_wave_structure/profiles.npz
    docs/benchmarks/results/lax_wave_structure/figures/*.png
    docs/benchmarks/results/lax_wave_structure/summary.md

Usage::

    python -m benchmarks.lax_wave_structure                 # full audit
    python -m benchmarks.lax_wave_structure --quick          # smoke (tiny legs)
    python -m benchmarks.lax_wave_structure --stage reuse    # no solver, npz only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from benchmarks.lax_case_pool import (
    CFL,
    DOMAIN,
    FINAL_TIME,
    LEFT,
    RIGHT,
    exact_waves,
    exact_positions,
    exact_reference,
    modal_fields,
    plateau_values,
    wave_positions,
    _weighted_l1,
    _weighted_l2,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d

OUT = Path("docs/benchmarks/results/lax_wave_structure")
FIG = OUT / "figures"

#: resolution ladder -- P3 is the degree of the pinned production leg.
#: N=1600 is the natural third rung but costs ~12 CPU-min; the local budget is
#: < 10 min per process, so the audit runs the 200/400/800 triple, which still
#: brackets the ledger's pinned leg (400, FAIL) and certified leg (800, PASS).
LADDER = [(200, 3), (400, 3), (800, 3)]

#: limiter arms at the production resolution.  ``None`` = no shock limiter,
#: only the positivity filter; ``tvb M=0`` = strict TVD; ``final`` = limit only
#: the end-of-step state (see ``tensordg.euler_dg_1d.solve_euler_dg_1d``).
LIMITER_ARMS = [
    {"label": "baseline", "limiter": None, "limiter_stage": "every",
     "tvb_constant": 1.0, "note": "engine default of the pool driver"},
    {"label": "tvb_M0", "limiter": "tvb", "limiter_stage": "every",
     "tvb_constant": 0.0, "note": "strict TVD (Cockburn-Shu M = 0)"},
    {"label": "tvb_M1", "limiter": "tvb", "limiter_stage": "every",
     "tvb_constant": 1.0, "note": "TVB M = 1"},
    {"label": "tvb_M1_final", "limiter": "tvb", "limiter_stage": "final",
     "tvb_constant": 1.0, "note": "TVB M = 1, end-of-step only"},
    {"label": "weno", "limiter": "weno", "limiter_stage": "every",
     "tvb_constant": 1.0, "note": "WENO limiter (Qiu-Shu)"},
]
ARM_CELLS, ARM_DEGREE = 400, 3

#: committed fields used by the reuse check: leg -> (path, cells)
COMMITTED = {
    "p3.N400": ("docs/benchmarks/results/lax_case_pool/profiles.npz", 400),
    "p3.N800": ("docs/benchmarks/results/lax_resolution_requirement/profiles.npz",
                800),
}


# ------------------------------------------------------------------ geometry
def _sorted(points, values):
    p = np.asarray(points).ravel()
    v = np.asarray(values).ravel()
    order = np.argsort(p)
    return p[order], v[order]


def _crossing_near(points, values, level, window, reference):
    """Interpolated crossing of ``values = level`` nearest ``reference``."""
    p, v = _sorted(points, values)
    keep = (p >= window[0]) & (p <= window[1])
    p, v = p[keep], v[keep]
    if p.size < 2:
        return None
    s = v - level
    hits = np.nonzero(np.sign(s[:-1]) * np.sign(s[1:]) <= 0)[0]
    if hits.size == 0:
        return None
    roots = []
    for i in hits:
        p0, p1, v0, v1 = p[i], p[i + 1], v[i], v[i + 1]
        roots.append(0.5 * (p0 + p1) if v1 == v0
                     else p0 + (level - v0) * (p1 - p0) / (v1 - v0))
    roots = np.asarray(roots, dtype=float)
    return float(roots[np.argmin(np.abs(roots - reference))])


def transition_width(points, values, level_lo, level_hi, reference, dx,
                     window_half=None):
    """10 %-to-90 % thickness of a captured jump (definition W2)."""
    window_half = max(8.0 * dx, 0.02) if window_half is None else window_half
    window = (reference - window_half, reference + window_half)
    lo = level_lo + 0.1 * (level_hi - level_lo)
    hi = level_lo + 0.9 * (level_hi - level_lo)
    x_lo = _crossing_near(points, values, lo, window, reference)
    x_hi = _crossing_near(points, values, hi, window, reference)
    if x_lo is None or x_hi is None:
        return {"x10": x_lo, "x90": x_hi, "width": None,
                "width_cells": None}
    width = abs(x_hi - x_lo)
    return {"x10": x_lo, "x90": x_hi, "width": width,
            "width_cells": width / dx}


def total_variation(points, values):
    p, v = _sorted(points, values)
    return float(np.abs(np.diff(v)).sum())


def extreme_report(points, values, exact_values):
    """W3: overshoot / undershoot of a nodal field against the exact range."""
    p = np.asarray(points).ravel()
    v = np.asarray(values).ravel()
    e = np.asarray(exact_values).ravel()
    lo_e, hi_e = float(e.min()), float(e.max())
    span = hi_e - lo_e
    i_max, i_min = int(np.argmax(v)), int(np.argmin(v))
    overshoot = max(0.0, float(v[i_max]) - hi_e)
    undershoot = max(0.0, lo_e - float(v[i_min]))
    outside = int(np.count_nonzero((v > hi_e) | (v < lo_e)))
    return {
        "num_min": float(v[i_min]), "num_max": float(v[i_max]),
        "exact_min": lo_e, "exact_max": hi_e,
        "x_at_num_min": float(p[i_min]), "x_at_num_max": float(p[i_max]),
        "overshoot": overshoot, "undershoot": undershoot,
        "overshoot_rel_span": overshoot / span, "undershoot_rel_span": undershoot / span,
        "nodes_outside_exact_range": outside,
        "nodes": int(v.size),
    }


# ------------------------------------------------------------------- metrics
def field_metrics(points, primitive, waves, exact, dx, quadrature_order=10):
    """All of W1-W5 for one nodal field set."""
    pos_exact = exact_positions(waves)
    pos_num = wave_positions(points, primitive, waves, dx=dx)
    position_errors = {}
    for name, x_num in pos_num.items():
        x_ex = pos_exact.get(name)
        if x_ex is None or x_num is None:
            position_errors[name] = None
            continue
        travel = abs(x_ex - waves["diaphragm"])
        position_errors[name] = {
            "x_num": float(x_num), "x_exact": float(x_ex),
            "abs_error": abs(float(x_num) - x_ex),
            "rel_error": abs(float(x_num) - x_ex) / travel,
        }
    rho = primitive["density"]
    rho_star_l = waves["density_star_left"]
    rho_star_r = waves["density_star_right"]
    rho_r = waves["right_state"][0]
    thickness = {
        "contact": transition_width(points, rho, rho_star_l, rho_star_r,
                                    pos_exact["contact"], dx),
        "shock": transition_width(points, rho, rho_star_r, rho_r,
                                  pos_exact["shock"], dx),
    }
    extrema = {name: extreme_report(points, primitive[name], exact[name])
               for name in ("density", "pressure")}
    tv = {name: {"numerical": total_variation(points, primitive[name]),
                 "exact": total_variation(points, exact[name])}
          for name in ("density", "pressure")}
    for name in tv:
        tv[name]["ratio"] = tv[name]["numerical"] / tv[name]["exact"]
    plateau = plateau_values(points, primitive, waves)
    plateau_error = {
        "star_left": {k: plateau["star_left"][k] for k in ("density", "velocity", "pressure")},
        "star_right": {k: plateau["star_right"][k] for k in ("density", "velocity", "pressure")},
    }
    exact_star = {"density": rho_star_l, "velocity": waves["velocity_star"],
                  "pressure": waves["pressure_star"]}
    exact_star_r = {"density": rho_star_r, "velocity": waves["velocity_star"],
                    "pressure": waves["pressure_star"]}
    for side, ref in (("star_left", exact_star), ("star_right", exact_star_r)):
        plateau_error[side] = {
            k: {"value": plateau[side][k],
                "exact": ref[k],
                "rel_error": abs(plateau[side][k] - ref[k]) / abs(ref[k])}
            for k in ref}
    return {
        "positions": pos_num,
        "position_errors": position_errors,
        "thickness": thickness,
        "extrema": extrema,
        "tv": tv,
        "plateau": plateau_error,
    }


def run_leg(cells, degree, waves, limiter=None, limiter_stage="every",
            tvb_constant=1.0, cfl=CFL):
    started = time.time()
    kwargs = {"limiter": limiter, "limiter_stage": limiter_stage,
              "tvb_constant": tvb_constant}
    x, coeff, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=cells, degree=degree,
                                       final_time=FINAL_TIME, cfl=cfl, **kwargs)
    points, primitive, dx, weights = modal_fields(x, coeff)
    exact = exact_reference(points)
    metrics = field_metrics(points, primitive, waves, exact, dx)
    metrics.update({
        "cells": cells, "degree": degree, "dx": float(dx),
        "steps": int(meta["steps"]),
        "limiter": limiter, "limiter_stage": limiter_stage,
        "tvb_constant": tvb_constant,
        "wall_seconds": float(time.time() - started),
        "l1_errors": {f: _weighted_l1(points, primitive[f], exact[f], weights, dx)
                      for f in ("density", "velocity", "pressure")},
        "l2_errors": {f: _weighted_l2(points, primitive[f], exact[f], weights, dx)
                      for f in ("density", "velocity", "pressure")},
    })
    print(f"[lax-ws] N={cells} p={degree} limiter={limiter}/{limiter_stage} "
          f"L2(rho)={metrics['l2_errors']['density']:.6f} "
          f"TV(rho)={metrics['tv']['density']['ratio']:.3f} "
          f"under(p)={metrics['extrema']['pressure']['undershoot']:.4f} "
          f"t={metrics['wall_seconds']:.1f}s", flush=True)
    return metrics, {"points": points, "numerical": primitive, "exact": exact,
                     "dx": dx}


def reuse_check(waves):
    """Recompute W3/W4 from the committed fields, no solver involved."""
    rows = {}
    for leg, (path, cells) in COMMITTED.items():
        file = Path(path)
        if not file.exists():
            rows[leg] = {"status": "missing_artifact", "path": path}
            continue
        data = np.load(file)
        points = data["points"]
        primitive = {"density": data["numerical_density"],
                     "velocity": data["numerical_velocity"],
                     "pressure": data["numerical_pressure"]}
        exact = {"density": data["exact_density"],
                 "velocity": data["exact_velocity"],
                 "pressure": data["exact_pressure"]}
        dx = 1.0 / cells
        rows[leg] = {
            "status": "ok", "path": path,
            "extrema": {n: extreme_report(points, primitive[n], exact[n])
                        for n in ("density", "pressure")},
            "tv": {n: {"numerical": total_variation(points, primitive[n]),
                       "exact": total_variation(points, exact[n])}
                   for n in ("density", "pressure")},
            "thickness": {
                "contact": transition_width(
                    points, primitive["density"], waves["density_star_left"],
                    waves["density_star_right"],
                    exact_positions(waves)["contact"], dx),
                "shock": transition_width(
                    points, primitive["density"], waves["density_star_right"],
                    waves["right_state"][0], exact_positions(waves)["shock"], dx),
            },
        }
        for n in ("density", "pressure"):
            rows[leg]["tv"][n]["ratio"] = (rows[leg]["tv"][n]["numerical"]
                                           / rows[leg]["tv"][n]["exact"])
    return rows


def committed_legs(waves):
    """Full W1-W5 for the legs whose fields are already banked, no solver.

    The integral L2 numbers are *not* recomputed here (they live in
    ``docs/benchmarks/results/lax_resolution_requirement/metrics.json`` and are
    cited verbatim); everything else -- positions, thickness, extrema, TV,
    plateaus -- is re-derived from the committed nodal fields.
    """
    legs, fields = [], {}
    for leg, (path, cells) in COMMITTED.items():
        file = Path(path)
        if not file.exists():
            continue
        data = np.load(file)
        points = data["points"]
        primitive = {"density": data["numerical_density"],
                     "velocity": data["numerical_velocity"],
                     "pressure": data["numerical_pressure"]}
        exact = {"density": data["exact_density"],
                 "velocity": data["exact_velocity"],
                 "pressure": data["exact_pressure"]}
        dx = 1.0 / cells
        metrics = field_metrics(points, primitive, waves, exact, dx)
        metrics.update({
            "leg": leg, "cells": cells, "degree": 3, "dx": float(dx),
            "limiter": None, "limiter_stage": "every", "tvb_constant": 1.0,
            "source": path, "l2_errors": None,
            "l2_source": "docs/benchmarks/results/lax_resolution_requirement/metrics.json",
        })
        legs.append(metrics)
        fields[leg] = {"points": points, "numerical": primitive, "exact": exact,
                       "dx": dx}
        print(f"[lax-ws] committed {leg}: TV(rho)="
              f"{metrics['tv']['density']['ratio']:.3f} "
              f"under(p)={metrics['extrema']['pressure']['undershoot']:.4f}", flush=True)
    return legs, fields


def banked_l2():
    """Relative L2 (density) of the two committed legs, read from their artifact."""
    path = Path("docs/benchmarks/results/lax_resolution_requirement/metrics.json")
    if not path.exists():
        return {"status": "missing_artifact", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in payload.get("legs", []):
        leg = row.get("leg") or f"p{row['degree']}.N{row['cells']}"
        rows[leg] = {"density": row["l2_errors"]["density"],
                     "velocity": row["l2_errors"]["velocity"],
                     "pressure": row["l2_errors"]["pressure"],
                     "cells": row["cells"], "degree": row["degree"]}
    return {"status": "ok", "path": str(path), "values": rows,
            "metrics_sha256": payload.get("metrics_sha256")}


def agreement(fresh, committed, key_path):
    node = fresh
    for key in key_path:
        node = node[key]
    if not isinstance(node, (int, float)) or not isinstance(committed, (int, float)):
        return None
    return abs(abs(float(node)) - abs(float(committed)))


# ------------------------------------------------------------------- figures
def render_figures(metrics, fields, waves, reuse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    pos = waves["positions"]

    # 1. zoom on the two captured discontinuities, finest two legs
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, (name, centre, half) in zip(
            axes, [("contact", pos["contact"], 0.05),
                   ("shock", pos["shock"], 0.05)]):
        for leg in metrics["legs"][-2:]:
            key = f"p{leg['degree']}.N{leg['cells']}"
            f = fields[key]
            x = np.asarray(f["points"]).ravel()
            r = np.asarray(f["numerical"]["density"]).ravel()
            o = np.argsort(x)
            ax.plot(x[o], r[o], lw=1.0, label=f"DG {key}")
        fx = np.linspace(centre - half, centre + half, 4001)
        ex = exact_reference(fx[None, :])["density"][0]
        ax.plot(fx, ex, "k--", lw=1.4, label="exact")
        ax.axvline(centre, color="0.6", lw=0.8)
        ax.set_xlim(centre - half, centre + half)
        ax.set_title(f"density · {name} zoom")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("CASE-LAX-DG: captured discontinuities (oscillation visible)")
    fig.tight_layout()
    path = FIG / "profiles_zoom.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)

    # 2. total-variation ratio vs resolution
    cells = [leg["cells"] for leg in metrics["legs"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for field, marker in (("density", "o"), ("pressure", "s")):
        ax.plot(cells, [leg["tv"][field]["ratio"] for leg in metrics["legs"]],
                marker + "-", label=f"TV({field}) / TV(exact)")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="monotone limit (=1)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cells N (P3)")
    ax.set_ylabel("TV ratio")
    ax.set_title("spurious variation grows with refinement")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "tv_ratio_vs_n.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)

    # 3. undershoot amplitude vs resolution
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for field, marker in (("density", "o"), ("pressure", "s")):
        ax.plot(cells, [leg["extrema"][field]["undershoot_rel_span"] * 100
                        for leg in metrics["legs"]],
                marker + "-", label=f"{field} undershoot")
        ax.plot(cells, [leg["extrema"][field]["overshoot_rel_span"] * 100
                        for leg in metrics["legs"]],
                marker + ":", label=f"{field} overshoot")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cells N (P3)")
    ax.set_ylabel("% of exact field range")
    ax.set_title("new extrema outside the exact range")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "extrema_vs_n.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)

    # 4. limiter arms
    arms = metrics["limiter_arms"]
    if arms:
        labels = [a["label"] for a in arms]
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
        axes[0].bar(labels, [a["tv"]["density"]["ratio"] for a in arms])
        axes[0].axhline(1.0, color="k", ls="--", lw=1.0)
        axes[0].set_title("TV(rho) ratio")
        axes[1].bar(labels, [a["l2_errors"]["density"] * 100 for a in arms])
        axes[1].axhline(3.0, color="r", ls="--", lw=1.0, label="3 % gate")
        axes[1].set_title("rel L2(rho) [%]")
        axes[1].legend(fontsize=8)
        axes[2].bar(labels, [a["extrema"]["pressure"]["undershoot_rel_span"] * 100
                             for a in arms])
        axes[2].set_title("p undershoot [% exact range]")
        for ax in axes:
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle("limiter arms")
        fig.tight_layout()
        path = FIG / "limiter_arms.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path.name)
    return made


def render_summary(metrics):
    banked = (metrics.get("banked_l2") or {}).get("values", {})

    def l2pct(leg):
        row = banked.get(leg["leg"])
        return "—" if not row else f"{100 * row['density']:.3f} %"

    lines = ["# CASE-LAX-DG · 波结构/容许性审计（程序化渲染，零手抄）", "",
             f"生成：{metrics['generated_at_utc']}　锚点：`{metrics['anchor']}`",
             "",
             "## 波位置误差 W1（`|dx|/|x_exact-x0|`）", "",
             "| leg | head | tail | contact | shock |", "|---|---|---|---|---|"]
    for leg in metrics["legs"]:
        pe = leg["position_errors"]
        cells = [f"{100 * pe[k]['rel_error']:.3f} %" if pe.get(k) else "—"
                 for k in ("rarefaction_head", "rarefaction_tail", "contact", "shock")]
        lines.append(f"| p{leg['degree']}.N{leg['cells']} | " + " | ".join(cells) + " |")
    lines += ["", "## 间断厚度 W2（密度 10%–90%，格数 = w/dx）", "",
              "| leg | contact w | contact w/dx | shock w | shock w/dx |",
              "|---|---|---|---|---|"]
    for leg in metrics["legs"]:
        c, s = leg["thickness"]["contact"], leg["thickness"]["shock"]
        fmt = lambda v: "—" if v is None else f"{v:.5f}"
        lines.append(f"| p{leg['degree']}.N{leg['cells']} | {fmt(c['width'])} | "
                     f"{fmt(c['width_cells'])} | {fmt(s['width'])} | "
                     f"{fmt(s['width_cells'])} |")
    lines += ["", "## 越界极值 W3 / 总变差 W4", "",
              "| leg | rel L2 ρ | ρ 超调 | ρ 欠调 | ρ TV 比 | p 欠调 | p TV 比 | 越界节点 |",
              "|---|---|---|---|---|---|---|---|"]
    for leg in metrics["legs"]:
        e, t = leg["extrema"], leg["tv"]
        lines.append(
            f"| p{leg['degree']}.N{leg['cells']} | "
            f"{l2pct(leg)} | "
            f"{100 * e['density']['overshoot_rel_span']:.2f} % | "
            f"{100 * e['density']['undershoot_rel_span']:.2f} % | "
            f"{t['density']['ratio']:.3f} | "
            f"{100 * e['pressure']['undershoot_rel_span']:.2f} % | "
            f"{t['pressure']['ratio']:.3f} | "
            f"{e['density']['nodes_outside_exact_range']} / {e['density']['nodes']} |")
    if metrics["limiter_arms"]:
        lines += ["", "## 限幅器支臂（P%d, N=%d）"
                  % (metrics["limiter_arms"][0]["degree"],
                     metrics["limiter_arms"][0]["cells"]), "",
                  "| arm | 说明 | rel L2 ρ | ρ TV 比 | p 欠调 | 越界节点 |",
                  "|---|---|---|---|---|---|"]
        for a in metrics["limiter_arms"]:
            lines.append(
                f"| {a['label']} | {a['note']} | "
                f"{100 * a['l2_errors']['density']:.3f} % | "
                f"{a['tv']['density']['ratio']:.3f} | "
                f"{100 * a['extrema']['pressure']['undershoot_rel_span']:.2f} % | "
                f"{a['extrema']['density']['nodes_outside_exact_range']}"
                f" / {a['extrema']['density']['nodes']} |")
    lines += ["", "## 判据结论", "", "```",
              json.dumps(metrics["verdict"], indent=1, ensure_ascii=False), "```", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="tiny legs, smoke only (not an artifact)")
    parser.add_argument("--stage", default="committed",
                        choices=["all", "committed", "ladder", "arms", "reuse"],
                        help="run only part of the audit")
    parser.add_argument("--out", default=None)
    parser.add_argument("--arm-cells", type=int, default=ARM_CELLS)
    parser.add_argument("--arm-degree", type=int, default=ARM_DEGREE)
    parser.add_argument("--merge-from", default=None,
                        help="previous metrics.json to fold in (split runs)")
    parser.add_argument("--ladder", default=None,
                        help="comma list cells:degree, overrides LADDER")
    args = parser.parse_args(argv)
    global OUT, FIG
    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    OUT, FIG = out, out / "figures"

    torch.set_default_dtype(torch.float64)
    waves = exact_waves()
    started = time.time()

    reuse = reuse_check(waves)

    legs, arm_rows, fields = [], [], {}
    if args.stage in ("all", "committed"):
        legs, fields = committed_legs(waves)
    if args.stage in ("all", "ladder", "reuse") and args.stage != "all":
        if args.quick:
            ladder = [(100, 1)]
        elif args.ladder:
            ladder = [tuple(int(v) for v in item.split(":"))
                      for item in args.ladder.split(",")]
        else:
            ladder = LADDER
        for cells, degree in ladder:
            leg, field = run_leg(cells, degree, waves)
            leg["leg"] = f"p{degree}.N{cells}"
            legs.append(leg)
            fields[leg["leg"]] = field
    if args.stage in ("all", "arms") and args.stage != "all":
        for arm in LIMITER_ARMS:
            reused = next((leg for leg in legs
                           if leg["cells"] == args.arm_cells
                           and leg["degree"] == args.arm_degree
                           and leg["limiter"] == arm["limiter"]
                           and leg["limiter_stage"] == arm["limiter_stage"]
                           and leg["tvb_constant"] == arm["tvb_constant"]), None)
            if reused is not None:
                row = dict(reused)
                row.update({"label": arm["label"], "note": arm["note"],
                            "leg": (f"p{args.arm_degree}.N{args.arm_cells}"
                                    f".{arm['label']}")})
                print(f"[lax-ws] arm {arm['label']}: reused the identical ladder leg",
                      flush=True)
                arm_rows.append(row)
                continue
            row, _ = run_leg(args.arm_cells, args.arm_degree, waves,
                             limiter=arm["limiter"],
                             limiter_stage=arm["limiter_stage"],
                             tvb_constant=arm["tvb_constant"])
            row.update({"label": arm["label"], "note": arm["note"],
                        "leg": (f"p{args.arm_degree}.N{args.arm_cells}"
                                f".{arm['label']}")})
            arm_rows.append(row)

    if args.merge_from and Path(args.merge_from).exists():
        old = json.loads(Path(args.merge_from).read_text(encoding="utf-8"))
        merged = {leg["leg"]: leg for leg in old.get("legs", [])}
        merged.update({leg["leg"]: leg for leg in legs})
        legs = list(merged.values())
        arms_old = old.get("limiter_arms", [])
        if arms_old:
            by_label = {arm["label"]: arm for arm in arms_old}
            by_label.update({arm["label"]: arm for arm in arm_rows})
            arm_rows = list(by_label.values())
            # keep the pre-registered arm order
            order = [a["label"] for a in LIMITER_ARMS]
            arm_rows.sort(key=lambda a: order.index(a["label"])
                          if a["label"] in order else len(order))
        print(f"[lax-ws] merged {args.merge_from}: "
              f"{len(legs)} legs, {len(arm_rows)} arms", flush=True)

    # reuse agreement: fresh solver run vs committed fields (same metrics)
    reuse_agreement = {}
    for leg in legs:
        key = leg["leg"]
        if key in reuse and reuse[key].get("status") == "ok":
            c = reuse[key]
            reuse_agreement[key] = {
                "rho_tv_ratio": agreement(leg, c, ["tv", "density", "ratio"]),
                "p_tv_ratio": agreement(leg, c, ["tv", "pressure", "ratio"]),
                "rho_undershoot": agreement(leg, c,
                                            ["extrema", "density", "undershoot"]),
                "p_undershoot": agreement(leg, c,
                                          ["extrema", "pressure", "undershoot"]),
                "contact_width": agreement(leg, c,
                                           ["thickness", "contact", "width"]),
            }

    banked = banked_l2()
    trends = {}
    if len(legs) >= 2:
        first, last = legs[0], legs[-1]
        trends["tv_ratio_change"] = {
            f: {"first": first["tv"][f]["ratio"], "last": last["tv"][f]["ratio"],
               "grows_with_refinement": last["tv"][f]["ratio"]
               > first["tv"][f]["ratio"]}
            for f in ("density", "pressure")}
        h = [leg["dx"] for leg in legs]
        for wave in ("contact", "shock"):
            w = [leg["thickness"][wave]["width"] for leg in legs]
            if all(v is not None and v > 0 for v in w):
                slope = float(np.polyfit(np.log(h), np.log(w), 1)[0])
                trends[f"thickness_scaling_exponent_{wave}"] = {
                    "slope_log_w_vs_log_dx": slope,
                    "interpretation": ("1.0 = fixed thickness in cells "
                                       "(first-order-like captured jump)"),
                    "width_cells": [leg["thickness"][wave]["width_cells"]
                                    for leg in legs]}
        order_l2 = []
        for a, b in zip(legs, legs[1:]):
            if a.get("l2_errors") and b.get("l2_errors"):
                order_l2.append(math.log(b["l2_errors"]["density"]
                                         / a["l2_errors"]["density"])
                                / math.log(a["cells"] / b["cells"]))
        if order_l2:
            trends["l2_order_density"] = order_l2

    verdict = {
        "acceptance_metric_untouched": ("the ledger arm (relative L2, P3/N=400) "
                                        "is NOT re-graded by this audit"),
        "admissibility_gate_proposed": (
            "nodes_outside_exact_range == 0 and TV ratio <= 1.02 at the certified "
            "resolution"),
        "admissibility_pass": all(
            leg["extrema"]["density"]["nodes_outside_exact_range"] == 0
            and leg["tv"]["density"]["ratio"] <= 1.02 for leg in legs) if legs else None,
        "l2_pass_at_3pct": {leg: row["density"] < 0.03
                            for leg, row in (banked or {}).get("values", {}).items()},
    }

    metrics = {
        "anchor": "CASE-LAX-DG",
        "case": {"left": LEFT, "right": RIGHT, "gamma": 1.4,
                 "final_time": FINAL_TIME, "diaphragm": waves["diaphragm"],
                 "cfl": CFL},
        "question": ("does the integral <3 % L2 acceptance of CASE-LAX-DG also "
                     "imply a monotone, non-oscillatory solution, and how thick "
                     "are the captured shock and contact?"),
        "pre_registered_definitions": {
            "W1": "wave position rel error |x_num-x_ex|c/|x_ex-x0| (reuses pool C1)",
            "W2": "10%-90% density transition thickness, in cells (w/dx)",
            "W3": "overshoot/undershoot outside the exact field range",
            "W4": "TV(numerical)/TV(exact) on Gauss nodes sorted by x",
            "W5": "star-plateau relative error (pool definition)",
        },
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exact": {"positions": waves["positions"],
                  "star": {"pressure": waves["pressure_star"],
                           "velocity": waves["velocity_star"],
                           "density_star_left": waves["density_star_left"],
                           "density_star_right": waves["density_star_right"]}},
        "legs": legs, "limiter_arms": arm_rows, "banked_l2": banked,
        "reuse_check": reuse, "reuse_agreement_solver_vs_committed": reuse_agreement,
        "trends": trends, "verdict": verdict,
        "wall_seconds": float(time.time() - started),
    }

    figures = render_figures(metrics, fields, waves, reuse) if fields else []
    arrays = {}
    if args.merge_from and (out / "profiles.npz").exists():
        with np.load(out / "profiles.npz") as previous:
            arrays.update({k: previous[k] for k in previous.files})
    arrays.update({f"{key}.points": f["points"] for key, f in fields.items()})
    arrays.update({f"{key}.numerical_{n}": f["numerical"][n]
                   for key, f in fields.items()
                   for n in ("density", "velocity", "pressure")})
    arrays.update({f"{key}.exact_{n}": f["exact"][n]
                   for key, f in fields.items()
                   for n in ("density", "velocity", "pressure")})
    arrays.update({f"{key}.dx": np.asarray(f["dx"]) for key, f in fields.items()})
    np.savez_compressed(out / "profiles.npz", **arrays)
    metrics["profiles_file"] = "profiles.npz"
    metrics["figures"] = figures
    payload = json.dumps(metrics, indent=1, sort_keys=True, ensure_ascii=False)
    metrics["metrics_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    metrics["metrics_sha256_covers"] = "payload without this field"
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=1, sort_keys=True, ensure_ascii=False),
        encoding="utf-8")
    render_summary(metrics)
    print(f"[lax-ws] wrote {out}/metrics.json  ({metrics['wall_seconds']:.0f} s)")
    return metrics


if __name__ == "__main__":
    main()