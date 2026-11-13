"""CASE-LAX-DG · is the ``p3`` order failure intrinsic, or pre-asymptotic?

Why this probe exists
---------------------
``benchmarks/lax_case_pool.py`` leaves the ledger anchor ``CASE-LAX-DG`` at
``FAIL``: the two order gates ``C3`` (whole-field relative-L1 order >= 1.0) and
``R3`` (analytic-region order >= 0.85 (p+1)) are missed, and they are missed
*worst at p3* -- density order ``0.27``/``0.48`` against requirements of
``3.4``/``1.0`` on the committed ``N = 100/200/400`` ladder.

Two very different readings survive the committed evidence:

* **intrinsic** -- the method (unlimited DG, only Zhang-Shu positivity scaling)
  cannot exceed ~O(h^0.5) in this configuration, so the gate failure is a
  property of the scheme at *any* resolution;
* **pre-asymptotic** -- the L1 error under a *fixed* physical band is still
  dominated by the dissipation layer of the three discontinuities.  Smearing
  width scales with ``h`` but has an ``O(h)``-independent core, so a band that
  is only 8 cells wide at ``N=400`` still contains the oscillatory ring of the
  p3 solution.  Under this reading the order must *rise* once the band holds
  many cells per ripple wavelength.

The committed ladder stops at ``N=400`` -- precisely where the band contains
34 cells, i.e. only ~4 ripple wavelengths.  This module extends the ladder to
``N = 800`` and answers the question with the pool's own gate functions
(``gate_c3`` / ``gate_r3`` imported verbatim), so a verdict here means the same
thing as a verdict there.

Pre-registered design (written before the numbers were seen)
-----------------------------------------------------------
* **Ladder** -- ``N = 400, 800`` for ``p = 1, 2, 3``, plus the committed
  ``N = 200, p = 3`` leg so that p3 has a three-point ladder and the
  ``200->400`` order can be compared with the committed ``200->400`` order
  (reuse evidence, not assumption).
* **Gates** -- ``gate_c3`` / ``gate_r3`` verbatim, ``R3_BAND_WIDTH = 0.02``
  from the pool.  Nothing is retuned; a configuration that fails keeps its
  FAIL.
* **H1 (pre-asymptotic)** -- for ``p=3`` the finest-pair R3 order exceeds the
  previous-pair R3 order by a factor of at least ``ESCALATION_FACTOR`` (1.5)
  and the analytic-region error keeps dropping.  Reading: the gate failure is
  a resolution artefact, the criterion is reachable but not yet reached.
* **H2 (intrinsic)** -- the finest-pair order does not escalate by that
  factor.  Reading: the order gate is a genuine scheme-property FAIL and the
  honest verdict of the pool stands for all ``N`` in reach.
* **Ripple diagnostic** -- the p3 residual ``rho_h - rho_exact`` is reduced
  over the star-right plateau (contact -> shock, pulled in by the wave band):
  ``max|r|``, ``rms``, and the zero-crossing count of the detrended residual,
  which gives the ripple wavelength in **cells**.  If the ripple wavelength is
  comparable to ``R3_BAND_WIDTH / dx`` the band is inside the ripple and H1 is
  the reading to test; this is measured, not assumed.
* **Reuse check** -- the ``N=400, p=3`` leg is re-run in this process and
  compared digit-wise with the committed pool metrics; the comparison is
  written into the artifact, so "the ladder is the same ladder" has evidence.

Result contract
---------------
Every number written here comes from the solver, from the closed-form Riemann
solution in ``benchmarks.riemann_exact``/``lax_case_pool``, or from the
committed metrics JSON.  Nothing is transcribed by hand.

Artifacts::

    docs/benchmarks/results/lax_asymptotic_probe/metrics.json
    docs/benchmarks/results/lax_asymptotic_probe/metrics.sha256
    docs/benchmarks/results/lax_asymptotic_probe/figures/*.png
    docs/benchmarks/results/lax_asymptotic_probe/summary.md
    docs/CASE-LAX-DG-p3-asymptotic-probe.md   (rendered by ``--render``)

Usage::

    PYTHONPATH=. python -m benchmarks.lax_asymptotic_probe            # full
    PYTHONPATH=. python -m benchmarks.lax_asymptotic_probe --quick     # smoke
    PYTHONPATH=. python -m benchmarks.lax_asymptotic_probe --render    # doc only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.lax_case_pool import (
    CFL,
    DIAPHRAGM,
    FINAL_TIME,
    GAMMA,
    LEFT,
    OUT as POOL_OUT,
    R3_BAND_WIDTH,
    R3_ORDER_FRACTION,
    RIGHT,
    _weighted_l1,
    convergence_order,
    exact_reference,
    exact_waves,
    gate_c3,
    gate_r3,
    modal_fields,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d

#: ``N = 400 -> 800`` for every degree; p3 also gets the committed ``N=200``
#: leg so that a three-point ladder exists where the pool failed hardest.
LADDER = [(400, 1), (800, 1), (400, 2), (800, 2), (200, 3), (400, 3), (800, 3)]
QUICK_LADDER = [(100, 1), (200, 1), (100, 2), (200, 2), (100, 3), (200, 3)]
#: pre-registered escalation factor for H1
ESCALATION_FACTOR = 1.5
#: the star-right plateau, pulled in from the analytic waves by the wave band
PLATEAU = ("contact", "shock")

OUT = Path(os.environ.get("LAX_ASYMPTOTIC_OUT",
                          "docs/benchmarks/results/lax_asymptotic_probe"))
POOL_METRICS = Path(POOL_OUT) / "metrics.json"
DOC = Path("docs") / "CASE-LAX-DG-p3-asymptotic-probe.md"


# ------------------------------------------------------------------ measuring
def analytic_mask(points, waves, band=R3_BAND_WIDTH):
    """Nested region rule: every point farther than ``band`` from every wave."""
    mask = np.ones_like(np.asarray(points, dtype=float), dtype=bool)
    for name in ("rarefaction_head", "rarefaction_tail", "contact", "shock"):
        mask &= np.abs(points - waves["positions"][name]) > band
    return mask


def plateau_window(points, waves, names=PLATEAU, shrink=R3_BAND_WIDTH, mask=None):
    """Points strictly between the first and last named waves, band pulled in."""
    flat = np.asarray(points, dtype=float).ravel()
    keep = (flat > waves["positions"][names[0]] + shrink) & \
           (flat < waves["positions"][names[1]] - shrink)
    if mask is not None:
        keep &= mask.ravel()
    return np.flatnonzero(keep)


def ripple_diagnostics(points, residual, waves, dx):
    """Ripple amplitude and wavelength in the star-right plateau.

    ``wavelength_cells`` comes from the zero crossings of the detrended
    residual: ``n`` crossings over a window of length ``L`` means ``n/2``
    half-periods, hence ``wavelength = 2L/n``.
    """
    flat_x = np.asarray(points, dtype=float).ravel()
    flat_r = np.asarray(residual, dtype=float).ravel()
    index = plateau_window(flat_x, waves)
    if index.size < 8:
        return {"points": int(index.size)}
    order = np.argsort(flat_x[index])
    x = flat_x[index][order]
    r = flat_r[index][order] - flat_r[index].mean()
    crossings = int(np.count_nonzero(np.diff(np.signbit(r))))
    length = float(x[-1] - x[0])
    wavelength = None if crossings < 2 else 2.0 * length / crossings
    # the raw crossing count is aliasing-sensitive at p1 (sub-cell wiggle);
    # a 5-point moving average isolates the O(8 cell) dissipation ripple
    smoothed = np.convolve(r, np.ones(5) / 5.0, mode="valid") if r.size >= 5 else r
    crossings_smoothed = int(np.count_nonzero(np.diff(np.signbit(smoothed))))
    wavelength_smoothed = (None if crossings_smoothed < 2
                           else 2.0 * length / crossings_smoothed)
    return {
        "points": int(index.size),
        "window": [float(x[0]), float(x[-1])],
        "max_abs_residual": float(np.abs(flat_r[index]).max()),
        "rms_residual": float(np.sqrt(np.mean(flat_r[index] ** 2))),
        "zero_crossings": crossings,
        "wavelength": wavelength,
        "wavelength_cells": None if wavelength is None else wavelength / dx,
        "zero_crossings_smoothed": crossings_smoothed,
        "wavelength_smoothed": wavelength_smoothed,
        "wavelength_cells_smoothed": (None if wavelength_smoothed is None
                                      else wavelength_smoothed / dx),
        "band_cells": R3_BAND_WIDTH / dx,
    }


def measure_leg(cells, degree, waves, exact_positions_cache=None):
    started = time.time()
    x, coeff, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=cells, degree=degree,
                                       final_time=FINAL_TIME, cfl=CFL)
    points, primitive, dx, weights = modal_fields(x, coeff)
    exact = exact_reference(points)
    mask = analytic_mask(points, waves)
    errors = {field: _weighted_l1(points, primitive[field], exact[field], weights, dx)
              for field in ("density", "velocity", "pressure")}
    analytic_errors = {
        field: _weighted_l1(points, primitive[field], exact[field], weights, dx, mask=mask)
        for field in ("density", "velocity", "pressure")}
    residual = primitive["density"] - exact["density"]
    leg = {
        "cells": cells, "degree": degree,
        "steps": int(meta["steps"]), "dt": float(meta["dt"]),
        "limiter_reported": meta.get("limiter"),
        "cfl": float(meta.get("cfl", CFL)),
        "wall_seconds": float(time.time() - started),
        "analytic_fraction": float(mask.mean()),
        "density_min": float(primitive["density"].min()),
        "pressure_min": float(primitive["pressure"].min()),
        "errors": errors,
        "analytic_errors": analytic_errors,
        "ripple": ripple_diagnostics(points, residual, waves, dx),
    }
    print(f"[lax-probe] N={cells:5d} p={degree} L1(rho)={errors['density']:.6e} "
          f"analytic={analytic_errors['density']:.6e} "
          f"ripple_cells={leg['ripple'].get('wavelength_cells')} "
          f"t={leg['wall_seconds']:.1f}s", flush=True)
    return leg, {"points": points, "numerical": primitive, "exact": exact}


def order_table(entries, key):
    """Orders between consecutive resolutions of the same degree."""
    table = {}
    for entry in entries:
        table.setdefault(entry["degree"], {})[entry["cells"]] = entry[key]
    out = {}
    for degree, by_cells in table.items():
        cells = sorted(by_cells)
        per_field = {}
        for field in ("density", "velocity", "pressure"):
            pairs = []
            for coarse, fine in zip(cells[:-1], cells[1:]):
                pairs.append({
                    "coarse": coarse, "fine": fine,
                    "coarse_error": by_cells[coarse][field],
                    "fine_error": by_cells[fine][field],
                    "order": convergence_order(coarse, by_cells[coarse][field],
                                               fine, by_cells[fine][field]),
                })
            per_field[field] = pairs
        out[degree] = per_field
    return out


def escalate(orders, degree=3, field="density"):
    """Finest-pair order vs previous-pair order for one degree/field."""
    pairs = orders.get(degree, {}).get(field)
    if not pairs or len(pairs) < 2:
        return {"available": False}
    previous, finest = pairs[-2], pairs[-1]
    if previous["order"] is None or finest["order"] is None:
        return {"available": False, "pairs": pairs}
    ratio = finest["order"] / previous["order"] if previous["order"] > 0 else None
    return {
        "available": True,
        "previous_pair": [previous["coarse"], previous["fine"]],
        "previous_order": previous["order"],
        "finest_pair": [finest["coarse"], finest["fine"]],
        "finest_order": finest["order"],
        "requirement": R3_ORDER_FRACTION * (degree + 1),
        "escalation_factor": ratio,
        "meets_requirement": finest["order"] >= R3_ORDER_FRACTION * (degree + 1),
        "escalates": ratio is not None and ratio >= ESCALATION_FACTOR,
    }


# ------------------------------------------------------------------- figures
def make_figures(entries, profiles, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    files = []

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for degree, style in ((1, "o-"), (2, "s-"), (3, "^-")):
        cells = sorted({e["cells"] for e in entries if e["degree"] == degree})
        if len(cells) < 2:
            continue
        for key, label, ls in (("errors", "whole field", "-"),
                               ("analytic_errors", "analytic region (band 0.02)", "--")):
            errors = [next(e[key]["density"] for e in entries
                           if e["degree"] == degree and e["cells"] == c)
                      for c in cells]
            axes[0].loglog(cells, errors, style.replace("-", ls), label=f"p{degree} {label}")
    axes[0].set_xlabel("cells")
    axes[0].set_ylabel(r"relative $L_1$ (density)")
    axes[0].set_title("Lax tube: whole field vs analytic region")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)

    # right panel: observed order per degree and pair, with the requirement line
    for degree, colour in ((1, "tab:blue"), (2, "tab:orange"), (3, "tab:red")):
        cells = sorted({e["cells"] for e in entries if e["degree"] == degree})
        pairs = list(zip(cells[:-1], cells[1:]))
        if not pairs:
            continue
        orders = []
        for coarse, fine in pairs:
            ec = next(e["analytic_errors"]["density"] for e in entries
                      if e["degree"] == degree and e["cells"] == coarse)
            ef = next(e["analytic_errors"]["density"] for e in entries
                      if e["degree"] == degree and e["cells"] == fine)
            orders.append(convergence_order(coarse, ec, fine, ef))
        axes[1].plot(range(len(orders)), orders, "o-", color=colour, label=f"p{degree} measured")
        axes[1].axhline(R3_ORDER_FRACTION * (degree + 1), color=colour, ls=":",
                        label=f"p{degree} requirement")
    axes[1].set_xlabel("ladder pair index (coarse -> fine)")
    axes[1].set_ylabel("analytic-region order (density)")
    axes[1].set_title("order vs R3 requirement")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    path = directory / "lax_asymptotic_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)

    finest = profiles.get("finest")
    if finest is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
        x = finest["points"].ravel()
        order = np.argsort(x)
        rho_n = finest["numerical"]["density"].ravel()[order]
        rho_e = finest["exact"]["density"].ravel()[order]
        axes[0].plot(x[order], rho_e, "k-", lw=1.0, label="exact")
        axes[0].plot(x[order], rho_n, "r.", ms=1.0, label=f"DG p{finest['degree']} N={finest['cells']}")
        axes[0].set_xlim(0.6, 0.9)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel(r"$\rho$")
        axes[0].set_title("star plateau + discontinuities (zoom)")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(x[order], (rho_n - rho_e), "b-", lw=0.7)
        axes[1].axhline(0.0, color="k", lw=0.5)
        for name in ("contact", "shock"):
            axes[1].axvline(exact_waves()["positions"][name], color="tab:red", ls=":", lw=0.8)
        axes[1].set_xlim(0.6, 0.9)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel(r"$\rho_h - \rho_{exact}$")
        axes[1].set_title("residual (the ripple the band must exclude)")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        path = directory / "lax_asymptotic_residual.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        files.append(path)

    return files


# --------------------------------------------------------------------- runner
def run(ladder=LADDER, quick=False):
    started = time.time()
    waves = exact_waves()
    entries, profiles = [], {}
    for cells, degree in ladder:
        leg, profile = measure_leg(cells, degree, waves)
        entries.append(leg)
        profiles[(cells, degree)] = profile
        if (cells, degree) == ladder[-1]:
            profiles["finest"] = {**profile, "cells": cells, "degree": degree}

    orders = order_table(entries, "errors")
    analytic_orders = order_table(entries, "analytic_errors")
    p3 = escalate(analytic_orders, 3)
    p3_c3 = escalate(orders, 3)

    finest_leg = entries[-1]
    gates = {
        "c3": gate_c3(entries),
        "r3": gate_r3(entries, band={"width": R3_BAND_WIDTH,
                                     "rule": "fixed physical band (nested)"}),
        "evaluated_on": [f"N={e['cells']},p={e['degree']}" for e in entries],
    }

    reuse = {"compared": False}
    if POOL_METRICS.exists() and not quick:
        pool = json.loads(POOL_METRICS.read_text())
        matches = [e for e in pool["ladder"]
                   if (e["cells"], e["degree"]) in profiles and not isinstance(
                       profiles[(e["cells"], e["degree"])], type(None))]
        rows, worst = [], 0.0
        for committed in matches:
            key = (committed["cells"], committed["degree"])
            mine = next(e for e in entries if (e["cells"], e["degree"]) == key)
            for field, value in committed["errors"].items():
                rel = abs(mine["errors"][field] - value) / max(abs(value), 1e-30)
                worst = max(worst, rel)
                rows.append({"cells": key[0], "degree": key[1], "field": field,
                             "committed": value, "rerun": mine["errors"][field],
                             "relative_difference": rel})
        reuse = {"compared": bool(rows), "rows": rows, "worst_relative_difference": worst,
                 "source": str(POOL_METRICS)}

    if quick:
        hypothesis = "quick smoke: hypothesis evaluation suppressed"
    elif p3.get("available") and p3["meets_requirement"]:
        hypothesis = ("H1 supported (gate reachable): the extended ladder's finest pair "
                      f"reaches R3 order {p3['finest_order']:.3f} >= {p3['requirement']:.2f}")
    elif p3.get("available") and p3["escalates"]:
        hypothesis = ("H1 supported as a trend: order still escalating "
                      f"(x{p3['escalation_factor']:.2f}) but "
                      f"{p3['finest_order']:.3f} < {p3['requirement']:.2f}")
    elif p3.get("available"):
        hypothesis = ("H2 supported: order did not escalate (x"
                      f"{p3['escalation_factor']:.2f}), the p3 order gate is a scheme "
                      "property and the pool's honest FAIL stands for every N in reach")
    else:
        hypothesis = "undetermined: ladder has no previous-pair comparison"

    payload = {
        "anchor": "CASE-LAX-DG",
        "case": "lax-1954-shock-tube",
        "question": ("p3 C3/R3 order FAIL: intrinsic scheme property or pre-asymptotic "
                     "resolution artefact?"),
        "hypotheses": {
            "H1_pre_asymptotic": (f"finest-pair R3 order >= {ESCALATION_FACTOR}x the previous "
                                  "pair (criterion reachable, not yet reached)"),
            "H2_intrinsic": (f"finest-pair R3 order < {ESCALATION_FACTOR}x the previous pair "
                             "(criterion misses for every N in reach)"),
            "resolution": hypothesis,
        },
        "config": {
            "ladder": [list(leg) for leg in ladder],
            "cfl": CFL,
            "final_time": FINAL_TIME,
            "diaphragm": DIAPHRAGM,
            "gamma": GAMMA,
            "left_state": list(LEFT),
            "right_state": list(RIGHT),
            "r3_band_width": R3_BAND_WIDTH,
            "r3_order_fraction": R3_ORDER_FRACTION,
            "escalation_factor": ESCALATION_FACTOR,
            "plateau_window": list(PLATEAU),
            "quick": quick,
        },
        "reference_definition": {
            "solution": "exact Riemann solution (benchmarks.riemann_exact.star_state)",
            "wave_positions": "closed form Rankine-Hugoniot / isentropic, not read back",
            "source": "Toro (2009) Riemann Solvers, 3rd ed., section 10.6 test 2",
        },
        "waves": {"positions": waves["positions"],
                  "pressures": waves.get("pressures"), "speeds": waves.get("speeds")},
        "ladder": entries,
        "orders": orders,
        "analytic_orders": analytic_orders,
        "p3_resolution": {"r3": p3, "c3": p3_c3},
        "gates": gates,
        "reuse_check": reuse,
        "finest_leg": {k: finest_leg[k] for k in ("cells", "degree", "steps", "errors",
                                                  "analytic_errors", "ripple")},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "figures": [],
        "wall_seconds": float(time.time() - started),
    }
    figures = make_figures(entries, profiles, OUT / "figures")
    payload["figures"] = [str(p) for p in figures]
    payload["verdict"] = ("PASS" if gates["c3"]["status"] == "PASS" and
                          gates["r3"]["status"] == "PASS" else "FAIL")
    return payload


def write(payload):
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True)
    target = OUT / "metrics.json"
    target.write_text(text + "\n")
    digest = hashlib.sha256((text + "\n").encode()).hexdigest()
    (OUT / "metrics.sha256").write_text(f"{digest}  metrics.json\n")
    (OUT / "summary.md").write_text(render_summary(payload, digest) + "\n")
    return target, digest


def render_summary(payload, digest=None):
    lines = []
    a = lines.append
    cfg = payload["config"]
    a("# CASE-LAX-DG · p3 order probe (N = 400 -> 800)\n")
    a(f"> Question: {payload['question']}")
    a(f"> Resolution: {payload['hypotheses']['resolution']}")
    a(f"> Verdict (pool gates on this ladder): **{payload['verdict']}**"
      + (f" · metrics sha256 `{digest}`" if digest else ""))
    a("\n## Ladder (programmatic, zero transcription)\n")
    a("| p | N | steps | whole-field L1 rho | analytic L1 rho | band cells | ripple wavelength (cells) |")
    a("|---|---|---|---|---|---|---|")
    for e in payload["ladder"]:
        ripple = e["ripple"]
        wl = ripple.get("wavelength_cells")
        a(f"| {e['degree']} | {e['cells']} | {e['steps']} | "
          f"{e['errors']['density']:.5e} | {e['analytic_errors']['density']:.5e} | "
          f"{ripple.get('band_cells'):.1f} | {'' if wl is None else f'{wl:.2f}'} |")
    a("\n## Orders (analytic region, nested band)\n")
    a("| p | pair | order rho | requirement |")
    a("|---|---|---|---|")
    for degree in sorted(payload["analytic_orders"], key=int):
        for pair in payload["analytic_orders"][degree]["density"]:
            req = R3_ORDER_FRACTION * (int(degree) + 1)
            order = pair["order"]
            a(f"| {degree} | {pair['coarse']} -> {pair['fine']} | "
              f"{'' if order is None else f'{order:.3f}'} | {req:.2f} |")
    a("\n## p3 resolution\n")
    for key in ("r3", "c3"):
        block = payload["p3_resolution"][key]
        if block.get("available"):
            a(f"* {key}: previous pair {block['previous_pair'][0]}->{block['previous_pair'][1]} "
              f"order {block['previous_order']:.3f}; finest pair "
              f"{block['finest_pair'][0]}->{block['finest_pair'][1]} "
              f"order {block['finest_order']:.3f} (escalation x{block['escalation_factor']:.3f})")
        else:
            a(f"* {key}: not available on this ladder")
    reuse = payload["reuse_check"]
    if reuse.get("compared"):
        a(f"\n## Reuse check\n* worst relative difference vs committed pool metrics: "
          f"{reuse['worst_relative_difference']:.3e} ({reuse['source']})")
    a(f"\nWall time: {payload['wall_seconds']:.1f} s")
    return "\n".join(lines)


def render_doc(payload, digest=None):
    cfg = payload["config"]
    a = []
    p = a.append
    p("# CASE-LAX-DG · p3 阶失效归因探针（N = 400 → 800）\n")
    p(f"> 锚点 `CASE-LAX-DG`（Lax 激波管 / Toro 2009 §10.6 test 2，ref = 精确解，容差 <3 %）")
    p("> 本文全部数字由 `benchmarks/lax_asymptotic_probe.py --render` 从 "
      "`docs/benchmarks/results/lax_asymptotic_probe/metrics.json` 程序化注入，零手抄。"
      + (f" 工件 sha256 `{digest}`。" if digest else ""))
    p("\n## ① 案例介绍\n")
    p("Lax (1954) 激波管：`[0,1]` 域，隔膜 `x0 = 0.5`，γ = 1.4，快照 t = 0.13，"
      "含左稀疏波（head/tail）、接触间断、右激波四道波。精确解由 `benchmarks.riemann_exact` "
      "的 Newton 迭代给出，波位为闭式 Rankine–Hugoniot / 等熵关系，参考值不依赖任何数值解。")
    p("| 参数 | 左态 | 右态 |\n|---|---|---|")
    p(f"| (ρ, u, p) | {tuple(cfg['left_state'])} | {tuple(cfg['right_state'])} |")
    p("\n解析波位（本算例的基准）：\n")
    p("| 波 | x |\n|---|---|")
    for name, value in payload["waves"]["positions"].items():
        p(f"| {name} | {value:.6f} |")
    p("\n## ② 怎么做的\n")
    p("算例池原阶梯止于 `N = 400`，其 `C3`（全场 L1 阶 ≥ 1.0）与 `R3`（解析区阶 ≥ 0.85(p+1)）"
      "在 p3 上离阈值最远。两种读法都能解释同一批数字：**方法内禀**（无限制 DG + 正性缩放，"
      "阶数上限就在 O(h^0.5) 量级）或 **未进渐近区**（固定物理带宽 0.02 在 N=400 只含 34 个单元，"
      "仍覆盖间断耗散层的振荡环）。本单把阶梯延长到 `N = 800`，用池子**自己的** "
      "`gate_c3` / `gate_r3` 裁决，判据不做任何改动。")
    p(f"* 阶梯：`{cfg['ladder']}`（p3 额外保留 N=200 腿，形成三点阶梯）")
    p(f"* CFL = {cfg['cfl']}，t = {cfg['final_time']}，重构每单元 10 点 Gauss；"
      f"R3 带宽固定 {cfg['r3_band_width']}（嵌套规则）")
    p(f"* 预登记假设：H1 若最细对阶 ≥ 前一对阶 × {cfg['escalation_factor']} → 判「未进渐近区」；"
      "否则 H2 → 判「方法内禀」，池子的 honest FAIL 原样保留")
    p("* 波纹诊断：星区右平台（接触 → 激波，按带宽内缩）上残差的 `max|r|`、rms 与"
      "过零点数；过零点数给波纹波长（单元数），用来判断带宽是否落在波纹内部\n")
    p("## ③ 计算结果\n")
    p("### 3.1 阶梯\n")
    p("| p | N | steps | 全场 L1 ρ | 解析区 L1 ρ | 解析区测度 | 带宽（单元） | 波纹波长（单元） |")
    p("|---|---|---|---|---|---|---|---|")
    for e in payload["ladder"]:
        ripple = e["ripple"]
        wl = ripple.get("wavelength_cells")
        p(f"| {e['degree']} | {e['cells']} | {e['steps']} | {e['errors']['density']:.5e} | "
          f"{e['analytic_errors']['density']:.5e} | {e['analytic_fraction']:.4f} | "
          f"{ripple.get('band_cells'):.1f} | "
          f"{'—' if wl is None else f'{wl:.2f}'} |")
    p("\n### 3.2 实测阶（解析区，嵌套带宽）\n")
    p("| p | 对 | 阶 ρ | 要求 0.85(p+1) | 达标 |\n|---|---|---|---|---|")
    for degree in sorted(payload["analytic_orders"], key=int):
        for pair in payload["analytic_orders"][degree]["density"]:
            req = R3_ORDER_FRACTION * (int(degree) + 1)
            order = pair["order"]
            ok = "—" if order is None else ("是" if order >= req else "否")
            p(f"| {degree} | {pair['coarse']} → {pair['fine']} | "
              f"{'—' if order is None else f'{order:.3f}'} | {req:.2f} | {ok} |")
    p("\n### 3.3 H1/H2 裁决\n")
    for key, label in (("r3", "R3（解析区阶）"), ("c3", "C3（全场阶）")):
        block = payload["p3_resolution"][key]
        if block.get("available"):
            p(f"* {label}：p3 前一对 `{block['previous_pair'][0]}→{block['previous_pair'][1]}` "
              f"阶 {block['previous_order']:.3f} → 最细对 "
              f"`{block['finest_pair'][0]}→{block['finest_pair'][1]}` 阶 "
              f"{block['finest_order']:.3f}，升级因子 ×{block['escalation_factor']:.3f}，"
              f"阈值 {block['requirement']:.2f} → "
              f"{'达标' if block['meets_requirement'] else '未达标'}")
        else:
            p(f"* {label}：本阶梯不可比")
    p(f"\n**结论**：{payload['hypotheses']['resolution']}")
    p(f"\n本阶梯上池子判据裁决 = **{payload['verdict']}**"
      f"（C3 {payload['gates']['c3']['status']} / R3 {payload['gates']['r3']['status']}）。")
    reuse = payload["reuse_check"]
    if reuse.get("compared"):
        p(f"\n### 3.4 复用/复现核对\n* 与已提交池子工件的最大相对差："
          f"**{reuse['worst_relative_difference']:.3e}**（源 `{reuse['source']}`）")
    p("\n## ④ 图\n")
    for path in payload["figures"]:
        p(f"* `{path}`")
    p("\n## ⑤ 注意点\n")
    p("* 判据未改写：`gate_c3` / `gate_r3` 与带宽 0.02 原样从池子导入；本单只延长阶梯。")
    p("* 波纹波长为过零点估计（半周期计数），对残差均值敏感；因此同时报 `max|r|` 与 rms。")
    p("* 未做的项：未与第三方 DG 实现交叉；未做 p4+（求解器只支持 p ≤ 3）；"
      "未把 N 推到 1600（单腿代价按 N² 估算超单节点预算）。")
    p("* 本单只改「能测到的阶」，不改「池子的验收阈值」——阈值变更须走台账流程。")
    p("\n## ⑥ 误差判定（关键量 vs 基准，是否 <3 %）\n")
    p("判据是**阶**不是百分比误差，故对「<3 %」的回答分两类：")
    p("\n1. 波位/星区量（池子 `C1`/`C2` 已在 `docs/CASE-LAX-DG.md` 判定全部 <3 %）：本单未重测，"
      "沿用前单结论并注明出处，不在此重复声明。")
    finest = payload["finest_leg"]
    p(f"2. 本单新增的关键量是最细腿（p{finest['degree']} / N={finest['cells']}）的解析区相对 L1："
      f"ρ {finest['analytic_errors']['density']:.3e}、u {finest['analytic_errors']['velocity']:.3e}、"
      f"p {finest['analytic_errors']['pressure']:.3e}"
      f" → 均 **{'<3 %' if max(finest['analytic_errors'].values()) < 0.03 else '≥3 %'}**"
      f"（最大 {max(finest['analytic_errors'].values())*100:.4f} %）。")
    stalled = payload["p3_resolution"]["r3"].get("escalates") is False
    p(f"3. 阶判据（C3/R3）不是百分比误差：其回答是"
      f"「{'阶未随加密升级，归因方法内禀' if stalled else '阶随加密升级，属未进渐近区'}」，"
      f"本阶梯裁决 **{payload['verdict']}**。")
    return "\n".join(a) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="coarse smoke ladder")
    parser.add_argument("--render", action="store_true", help="re-render doc from metrics")
    parser.add_argument("--ladder", default=None, help='e.g. "400,800:1;400,800:2"')
    args = parser.parse_args()

    if args.render:
        payload = json.loads((OUT / "metrics.json").read_text())
        digest = (OUT / "metrics.sha256").read_text().split()[0] \
            if (OUT / "metrics.sha256").exists() else None
        DOC.write_text(render_doc(payload, digest))
        print(f"rendered {DOC}")
        return

    ladder = QUICK_LADDER if args.quick else LADDER
    if args.ladder:
        ladder = []
        for chunk in args.ladder.split(";"):
            cells, degree = chunk.split(":")
            for value in cells.split(","):
                ladder.append((int(value), int(degree)))
        ladder.sort(key=lambda leg: (leg[1], leg[0]))
    payload = run(ladder, quick=args.quick)
    target, digest = write(payload)
    DOC.write_text(render_doc(payload, digest))
    print(f"verdict={payload['verdict']} resolution={payload['hypotheses']['resolution']}")
    print(f"artifacts: {target}, {DOC}")


if __name__ == "__main__":
    main()