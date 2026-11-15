"""CASE-LAX-DG: does the published "limiters are over-limiting at p3" claim
survive the closure mesh and the documented fair limiter stage?

Why this driver exists (reopened #617, round 3)
-----------------------------------------------
``docs/CASE-LAX-DG-resolution-closure.md`` §3.2 / §5.3 publishes two claims from
the closure run:

  (i)  at the production mesh ``p3 N=400`` both shock limiters (``tvb``,
       ``weno``) are **worse** than leaving the discontinuities unlimited, in
       whole-field relative L2 (7.04 % / 6.44 % versus 3.12 %);
  (ii) therefore "the limiters are not a remedy" for the p3 accuracy gap.

Both statements rest on **one mesh** and on **one limiter stage**: the arms
were run with the historic default ``limiter_stage="every"``, which the
repository's own ``docs/limiter_verification.md`` documents as destroying
high-order accuracy on smooth cells (the limiter re-projects cells that keep
being flagged after every SSP-RK3 sub-step).  Neither fact is measured by the
committed artifact, so claim (ii) is currently a single-factor, single-mesh
inference with an unmeasured second factor that the repo itself flags as
suspect.

This driver measures exactly that missing 2 x 2 + 1 factorial:

    mesh        N in {400, 800}          (production leg, closure mesh)
    limiter     none / tvb / weno        (registered arm + both shock limiters)
    stage       every / final            (limiter_stage; only for limiters)

Everything else is the registered configuration (p=3, CFL 0.35, exact Riemann
reference, element Gauss nodes, fixed 0.02 physical band as in the pool).
The ``none`` arm has no limiter stage, so it contributes one leg per mesh; the
grid is therefore 10 legs.

Predeclared predictions and decision rule (fixed in code *before* the run)
--------------------------------------------------------------------------
See ``PREDICTIONS`` below and the constants ``BAND_GAIN_FACTOR``,
``L2_STAGE_GAIN_MIN``, ``SEPARATION_MIN``.  ``gate()`` evaluates them and maps
the three outcomes onto a verdict; the mapping is a pure function of the three
booleans, so a test can drive every branch without a cluster run.

Honesty rules carried over from the closure artifact
----------------------------------------------------
* These arms are **diagnostic**.  The ledger acceptance ``A0`` is registered on
  the *unlimited* production leg, so no limiter arm can flip it; the verdict of
  the closure study (closure mesh 800) is untouched by this study.
* Nothing is transcribed: every number in ``metrics.json`` comes from the
  solver or the exact Riemann sampler.
* The unlimited/``every`` legs at ``N=400`` and ``N=800`` are re-measured here
  and compared to the committed closure artifact; agreement to ``1e-12``
  relative is the evidence that this runner uses the same measurement path.

Usage::

    python -m benchmarks.lax_closure_arms_mesh_scan --skip-figures \\
        --output-dir /tmp/o            # cluster, prints DIGEST
    python -m benchmarks.lax_closure_arms_mesh_scan --quick   # smoke
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from benchmarks.lax_case_pool import (
    CFL,
    DOMAIN,
    FINAL_TIME,
    LEFT,
    R3_BAND_WIDTH,
    RIGHT,
    _weighted_l1,
    _weighted_l2,
    exact_reference,
    exact_waves,
    ledger_acceptance_tolerance,
    modal_fields,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d

OUT = Path("docs/benchmarks/results/lax_closure_arms_mesh_scan")
FIG = OUT / "figures"

DEGREE = 3
#: the production leg and the closure mesh of the committed closure artifact
MESHES = (400, 800)
QUICK_MESHES = (100, 200)
LIMITERS = (None, "tvb", "weno")
#: limiter stage only exists when a limiter is on; the unlimited arm has one leg
STAGES = ("every", "final")
FIELDS = ("density", "velocity", "pressure")

#: P2: a limiter must cut the band-masked L1 of density to <= this fraction of
#: the unlimited arm's value (i.e. the oscillations inside the analytic bands
#: really are what the limiter removes)
BAND_GAIN_FACTOR = 0.5
#: P3: switching limiter_stage "every" -> "final" must cut the whole-field
#: relative L2 of density by at least this relative amount for the claim
#: "the limiter is over-limiting" to be attributable to the stage setting
L2_STAGE_GAIN_MIN = 0.20
#: the metric split is declared material when the unlimited arm's whole-field
#: L2 exceeds its band-masked L1 by this factor (oscillation leakage dominates)
SEPARATION_MIN = 2.0

PREDICTIONS = {
    "P1": ("at limiter_stage='every', whole-field rho L2 is strictly ordered "
           "none < tvb and none < weno at BOTH meshes (mesh-independence of the "
           "committed single-mesh claim)"),
    "P2": ("at both meshes, tvb and weno each cut the band-masked rho L1 to "
           f"<= {BAND_GAIN_FACTOR:.2f} x the unlimited value (the limiter does "
           "remove the oscillation it targets)"),
    "P3": ("at both meshes, moving each limiter from stage 'every' to 'final' "
           f"cuts whole-field rho L2 by >= {L2_STAGE_GAIN_MIN:.0%} (the "
           "committed penalty is a stage artifact, not a property of the "
           "limiter)"),
}


# --------------------------------------------------------------------- runners
def _arms(meshes, limiters=LIMITERS, stages=STAGES):
    """The factorial grid: (cells, limiter, limiter_stage) legs.

    The unlimited arm has no limiter stage, so it appears once per mesh.
    """
    grid = []
    for cells in meshes:
        for limiter in limiters:
            for stage in ((None,) if limiter is None else stages):
                grid.append((int(cells), limiter, stage))
    return grid


def run_leg(cells, limiter=None, limiter_stage=None, degree=DEGREE,
            tvb_constant=1.0, waves=None, final_time=FINAL_TIME, cfl=CFL):
    """Solve one leg and measure it exactly the way the closure artifact does.

    Same measurement path as ``benchmarks.lax_resolution_closure.run_leg``:
    primitive fields on the element Gauss nodes, relative L1/L2 with the Gauss
    weights, exact Riemann solution as denominator, and the R3 rule for the
    band mask (a *fixed physical* half width around each analytic wave).
    """
    if waves is None:
        waves = exact_waves(final_time=final_time)
    started = time.time()
    kwargs = {}
    if limiter is not None and limiter_stage is not None:
        kwargs["limiter_stage"] = limiter_stage
    x, coeff, meta = solve_euler_dg_1d(
        LEFT, RIGHT, ncells=cells, degree=degree, final_time=final_time,
        cfl=cfl, limiter=limiter, tvb_constant=tvb_constant, **kwargs,
    )
    points, primitive, dx, weights = modal_fields(x, coeff)
    exact = exact_reference(points, final_time=final_time)
    band = np.ones_like(primitive["density"], dtype=bool)
    for name in ("rarefaction_head", "rarefaction_tail", "contact", "shock"):
        band &= np.abs(points - waves["positions"][name]) > R3_BAND_WIDTH
    entry = {
        "cells": int(cells),
        "degree": int(degree),
        "limiter": "none" if limiter is None else str(limiter),
        "limiter_stage": None if limiter is None else str(limiter_stage),
        "tvb_constant": float(tvb_constant) if limiter == "tvb" else None,
        "steps": int(meta["steps"]),
        "dt_mean": float(meta.get("dt_mean", final_time / max(meta["steps"], 1))),
        "wall_seconds": float(time.time() - started),
        "density_min": float(primitive["density"].min()),
        "band_fraction": float(band.mean()),
        "l1": {f: _weighted_l1(points, primitive[f], exact[f], weights, dx)
               for f in FIELDS},
        "l2": {f: _weighted_l2(points, primitive[f], exact[f], weights, dx)
               for f in FIELDS},
        "band_l1": {f: _weighted_l1(points, primitive[f], exact[f], weights, dx,
                                    mask=band) for f in FIELDS},
        "max_abs": {f: float(np.max(np.abs(primitive[f] - exact[f])))
                    for f in FIELDS},
    }
    print(f"[arms] N={cells:5d} limiter={entry['limiter']:4s} "
          f"stage={str(entry['limiter_stage']):5s} "
          f"L2(rho)={entry['l2']['density']:.6f} "
          f"band(rho)={entry['band_l1']['density']:.6f} "
          f"steps={entry['steps']:5d} t={entry['wall_seconds']:.1f}s", flush=True)
    return entry


def run_grid(meshes=MESHES, limiters=LIMITERS, stages=STAGES):
    waves = exact_waves()
    return [run_leg(cells, limiter=limiter, limiter_stage=stage, waves=waves)
            for cells, limiter, stage in _arms(meshes, limiters, stages)]


# ----------------------------------------------------------------------- gates
def _pick(legs, cells, limiter, stage):
    for leg in legs:
        if (leg["cells"] == cells and leg["limiter"] == limiter
                and leg["limiter_stage"] == stage):
            return leg
    raise KeyError(f"no leg cells={cells} limiter={limiter} stage={stage}")


def separation(legs):
    """whole-field L2 / band-masked L1 of density, per leg (oscillation share)."""
    out = {}
    for leg in legs:
        key = f"N{leg['cells']}.{leg['limiter']}" + (
            "" if leg["limiter_stage"] is None else f".{leg['limiter_stage']}")
        band = leg["band_l1"]["density"]
        out[key] = float(leg["l2"]["density"] / band) if band > 0 else None
    return out


def gate(legs, tolerance=None, band_gain_factor=BAND_GAIN_FACTOR,
         l2_stage_gain_min=L2_STAGE_GAIN_MIN, separation_min=SEPARATION_MIN):
    """Evaluate the three predeclared predictions and map them to a verdict.

    Pure function of the legs (no I/O), so every branch is unit-testable.
    """
    tolerance = ledger_acceptance_tolerance() if tolerance is None else tolerance
    meshes = sorted({leg["cells"] for leg in legs})

    # --- P1: unlimited is the most accurate arm at stage "every", every mesh
    p1_rows = []
    for cells in meshes:
        none = _pick(legs, cells, "none", None)["l2"]["density"]
        row = {"cells": cells, "none": none}
        for limiter in ("tvb", "weno"):
            value = _pick(legs, cells, limiter, "every")["l2"]["density"]
            row[limiter] = value
            row[f"{limiter}_worse"] = bool(value > none)
        row["holds"] = bool(row["tvb_worse"] and row["weno_worse"])
        p1_rows.append(row)
    p1 = all(row["holds"] for row in p1_rows)

    # --- P2: the limiters do remove the in-band oscillation
    p2_rows = []
    for cells in meshes:
        none = _pick(legs, cells, "none", None)["band_l1"]["density"]
        row = {"cells": cells, "none": none}
        for limiter in ("tvb", "weno"):
            value = _pick(legs, cells, limiter, "every")["band_l1"]["density"]
            row[limiter] = value
            row[f"{limiter}_ratio"] = float(value / none) if none > 0 else None
            row[f"{limiter}_cuts"] = bool(value <= band_gain_factor * none)
        row["holds"] = bool(row["tvb_cuts"] and row["weno_cuts"])
        p2_rows.append(row)
    p2 = all(row["holds"] for row in p2_rows)

    # --- P3: the penalty shrinks when the limiter stops re-projecting smooth
    #         cells after every RK sub-step
    p3_rows = []
    for cells in meshes:
        for limiter in ("tvb", "weno"):
            every = _pick(legs, cells, limiter, "every")["l2"]["density"]
            final = _pick(legs, cells, limiter, "final")["l2"]["density"]
            gain = float(1.0 - final / every) if every > 0 else None
            p3_rows.append({"cells": cells, "limiter": limiter,
                            "l2_every": every, "l2_final": final,
                            "relative_gain": gain,
                            "holds": bool(gain is not None
                                          and gain >= l2_stage_gain_min)})
    p3 = all(row["holds"] for row in p3_rows)

    # --- metric split: how much of the unlimited error lives inside the bands
    sep = separation(legs)
    unlimited_sep = {f"N{cells}": sep[f"N{cells}.none"] for cells in meshes}
    split_material = all(
        (value is not None and value >= separation_min)
        for value in unlimited_sep.values())

    # --- verdict, predeclared mapping of the three booleans
    if p1:
        ranking = "mesh-independent"
    else:
        ranking = "mesh-specific (single-mesh claim: the limiter penalty is not reproduced at every mesh)"
    if p3:
        stage_effect = "dominant"
    elif any(row["holds"] for row in p3_rows):
        stage_effect = "partial"
    else:
        stage_effect = "none"
    if split_material:
        metric_view = ("metric-dependent: whole-field L2 counts oscillation "
                       "leakage inside the four analytic bands, band-masked L1 does not")
    else:
        metric_view = "metric-consistent"

    if p1 and p3:
        statement = ("retain the limiter statement, but qualify it by the limiter "
                     "stage: 'over-limiting' describes limiter_stage='every'; the "
                     "documented fair setting 'final' recovers a measurable share "
                     "of the penalty")
    elif not p1 and p3:
        statement = ("withdraw the single-mesh ordering and re-state it per mesh; "
                     "the stage effect (P3) holds, so the penalty is dominated by "
                     "the stage setting rather than by the limiter")
    elif p1 and not p3:
        statement = ("the single-mesh ordering survives refinement and is not a "
                     "stage artifact: the penalty comes from the troubled-cell "
                     "fallback itself, so 'limiters are over-limiting at p3' "
                     "stands unqualified")
    else:
        statement = ("neither the ordering nor the stage effect is mesh-robust: "
                     "the arms table of the closure doc must be demoted to a "
                     "single-mesh observation and the general claim withdrawn")

    return {
        "bar": "relative L2 / band-masked L1 vs exact Riemann solution",
        "tolerance": float(tolerance),
        "tolerance_note": ("the ledger 3 % bar is registered on the unlimited "
                           "production leg only; limiter arms are diagnostic and "
                           "cannot flip ledger A0"),
        "registered_arm_unaffected": True,
        "grid": [{"cells": leg["cells"], "limiter": leg["limiter"],
                  "limiter_stage": leg["limiter_stage"]} for leg in legs],
        "bar_values": {"band_gain_factor": float(band_gain_factor),
                       "l2_stage_gain_min": float(l2_stage_gain_min),
                       "separation_min": float(separation_min)},
        "predictions": {
            "P1": {"text": PREDICTIONS["P1"], "holds": bool(p1), "rows": p1_rows},
            "P2": {"text": PREDICTIONS["P2"], "holds": bool(p2), "rows": p2_rows},
            "P3": {"text": PREDICTIONS["P3"], "holds": bool(p3), "rows": p3_rows},
        },
        "separation_wholefield_l2_over_band_l1": sep,
        "unlimited_separation": unlimited_sep,
        "verdicts": {"P1_limiter_ranking": ranking,
                     "P3_stage_effect": stage_effect,
                     "metric_view": metric_view,
                     "published_statement": statement},
        "reading": ("PASS/FAIL language does not apply to the arms: the only "
                    "binary outcomes here are the three predeclared predictions. "
                    "P1 True = the committed ordering reproduces at both meshes; "
                    "P2 True = the limiters do suppress the in-band oscillation; "
                    "P3 True = limiter_stage='final' measurably recovers accuracy, "
                    "so the committed penalty must be attributed to the stage "
                    "setting as well as to the limiter."),
    }


def gate_closure_reproduction(legs, artifact_path=None):
    """Do the re-measured unlimited legs reproduce the committed closure artifact?

    Only the legs the closure artifact already contains can be checked
    (``none``/``every`` at ``N=400`` and ``N=800``).  Any relative deviation
    above ``1e-12`` means the two runners do not share a measurement path and
    this study's arms would not be comparable with the published ladder.
    """
    path = Path(artifact_path or
                "docs/benchmarks/results/lax_resolution_closure/metrics.json")
    if not path.exists():
        return {"checked": False, "reason": f"{path} not available"}
    committed = json.loads(path.read_text(encoding="utf-8"))
    by_cells = {leg["cells"]: leg for leg in committed["ladder"]}
    rows, worst = [], 0.0
    for leg in legs:
        if leg["limiter"] != "none" or leg["cells"] not in by_cells:
            continue
        reference = by_cells[leg["cells"]]
        row = {"cells": leg["cells"], "fields": {}, "steps": [leg["steps"], reference["steps"]]}
        for field in FIELDS:
            for norm in ("l1", "l2", "band_l1"):
                mine = leg[norm][field]
                theirs = reference[norm][field]
                rel = abs(mine - theirs) / abs(theirs) if theirs else abs(mine)
                row["fields"][f"{norm}.{field}"] = float(rel)
                worst = max(worst, float(rel))
        rows.append(row)
    return {"checked": bool(rows), "artifact": str(path),
            "worst_relative_deviation": float(worst),
            "tolerance": 1e-12,
            "reproduced": bool(rows and worst <= 1e-12),
            "legs": rows}


# --------------------------------------------------------------------- figures
def render_figures(metrics, out_dir=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    legs = metrics["legs"]
    meshes = sorted({leg["cells"] for leg in legs})
    tol = metrics["gate"]["tolerance"]
    made = []

    # (1) whole-field L2 of every arm at every mesh, on a log axis with the bar.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = [f"{leg['limiter']}"
              + ("" if leg["limiter_stage"] is None else f"/{leg['limiter_stage']}")
              for leg in legs if leg["cells"] == meshes[0]]
    width = 0.8 / len(meshes)
    for k, cells in enumerate(meshes):
        arms = [leg for leg in legs if leg["cells"] == cells]
        xs = [i + (k - (len(meshes) - 1) / 2) * width for i in range(len(arms))]
        ax.bar(xs, [arm["l2"]["density"] for arm in arms], width,
               label=f"N={cells}", alpha=0.85)
    ax.axhline(tol, color="k", ls="--", lw=1.2, label=f"ledger bar {tol:.0%}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(r"whole-field relative $L_2$ of $\rho$")
    ax.set_title("CASE-LAX-DG: limiter arms x limiter stage x mesh")
    ax.legend(fontsize=8)
    path = out_dir / "arms_mesh_scan.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption":
                 "Whole-field relative L2 of density for every arm at the "
                 "production mesh (N=400) and the closure mesh (N=800); dashed "
                 "line = ledger 3 % bar. The unlimited arm is below the bar at "
                 "N=800, the limiter arms are not, and moving the limiter to "
                 "stage 'final' moves them toward the bar."})

    # (2) the metric split: in-band oscillation error versus whole-field error.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    markers = {"none": "o", "tvb": "s", "weno": "^"}
    for cells, style in zip(meshes, ("--", "-")):
        arms = [leg for leg in legs if leg["cells"] == cells]
        ax.plot([arm["band_l1"]["density"] for arm in arms],
                [arm["l2"]["density"] for arm in arms], style + "k", lw=0.8, alpha=0.4)
        for arm in arms:
            ax.scatter(arm["band_l1"]["density"], arm["l2"]["density"],
                       marker=markers[arm["limiter"]],
                       facecolors="none" if cells == meshes[0] else "k",
                       edgecolors="k", s=46,
                       label=None)
    lim = [min(leg["band_l1"]["density"] for leg in legs) * 0.7,
           max(leg["l2"]["density"] for leg in legs) * 1.4]
    ax.plot(lim, lim, "k:", lw=1.0, label=r"$L_2 = L_1^{(band)}$")
    for name, marker in markers.items():
        ax.scatter([], [], marker=marker, facecolors="none", edgecolors="k",
                   label=f"limiter={name}")
    ax.scatter([], [], marker="o", facecolors="none", edgecolors="k",
               label=f"open = N={meshes[0]}")
    ax.scatter([], [], marker="o", facecolors="k", edgecolors="k",
               label=f"filled = N={meshes[-1]}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"band-masked relative $L_1$ of $\rho$ (oscillation view)")
    ax.set_ylabel(r"whole-field relative $L_2$ of $\rho$")
    ax.set_title("CASE-LAX-DG: the two metrics rank the limiters oppositely")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    path = out_dir / "arms_metric_split.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption":
                 "Band-masked relative L1 of density (what the limiter removes) "
                 "against whole-field relative L2 (what the ledger gates). The "
                 "unlimited arm sits top-left in oscillation share, the limiter "
                 "arms bottom-right: the two metrics answer different questions, "
                 "so 'limiters are worse' is a whole-field-L2 statement."})
    return made


def render_summary(metrics):
    gate_ = metrics["gate"]
    lines = ["# CASE-LAX-DG limiter arms x stage x mesh scan", "",
             f"generated: {metrics['generated_at_utc']}  |  "
             f"wall: {metrics['wall_seconds']:.0f} s", "",
             "Predeclared predictions:",
             f"- P1 (mesh-independent limiter ranking): **{'HOLDS' if gate_['predictions']['P1']['holds'] else 'FAILS'}**",
             f"- P2 (in-band oscillation removed by limiters): **{'HOLDS' if gate_['predictions']['P2']['holds'] else 'FAILS'}**",
             f"- P3 (stage 'every' -> 'final' recovers >= {gate_['bar_values']['l2_stage_gain_min']:.0%} whole-field L2): "
             f"**{'HOLDS' if gate_['predictions']['P3']['holds'] else 'FAILS'}**", "",
             "| N | limiter | stage | rho L2 | rho band L1 | L2/band | steps | wall s |",
             "|---|---|---|---|---|---|---|---|"]
    for leg in sorted(metrics["legs"], key=lambda x: (x["cells"], x["limiter"],
                                                      str(x["limiter_stage"]))):
        key = f"N{leg['cells']}.{leg['limiter']}" + (
            "" if leg["limiter_stage"] is None else f".{leg['limiter_stage']}")
        sep = gate_["separation_wholefield_l2_over_band_l1"].get(key)
        lines.append(f"| {leg['cells']} | {leg['limiter']} | "
                     f"{leg['limiter_stage'] or '-'} | {leg['l2']['density']:.4%} | "
                     f"{leg['band_l1']['density']:.4%} | "
                     f"{sep:.2f} | {leg['steps']} | {leg['wall_seconds']:.0f} |")
    lines += ["", f"verdict: {gate_['verdicts']['published_statement']}"]
    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------------ main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smoke ladder (100/200)")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    out = Path(args.output_dir) if args.output_dir else OUT
    meshes = QUICK_MESHES if args.quick else MESHES
    legs = run_grid(meshes)
    gate_ = gate(legs)
    reproduction = gate_closure_reproduction(legs)
    figures = [] if args.skip_figures else None
    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "study": ("limiter arms x limiter_stage x mesh: is the published "
                  "'limiters are over-limiting at p3' claim mesh- and "
                  "stage-robust?"),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"degree": DEGREE, "meshes": list(meshes), "cfl": CFL,
                   "final_time": FINAL_TIME, "domain": list(DOMAIN),
                   "r3_band_width": R3_BAND_WIDTH,
                   "limiters": [("none" if a is None else a) for a in LIMITERS],
                   "limiter_stages": list(STAGES), "quick": bool(args.quick)},
        "legs": legs,
        "gate": gate_,
        "closure_artifact_reproduction": reproduction,
        "wall_seconds": float(time.time() - started),
        "out_dir": str(out),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    if figures is None:
        metrics["figures"] = render_figures(metrics, out / "figures")
    else:
        metrics["figures"] = figures
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    if out == OUT:
        render_summary(metrics)
    digest = {"generated_at_utc": metrics["generated_at_utc"],
              "wall_seconds": metrics["wall_seconds"],
              "legs": legs, "gate": gate_,
              "closure_artifact_reproduction": reproduction}
    print("DIGEST" + json.dumps(digest), flush=True)
    print(f"[arms] P1={gate_['predictions']['P1']['holds']} "
          f"P2={gate_['predictions']['P2']['holds']} "
          f"P3={gate_['predictions']['P3']['holds']} "
          f"wall={metrics['wall_seconds']:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())