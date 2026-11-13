"""CASE-LAX-DG · does shock capturing restore the pool's failing gates?

Why this study exists
---------------------
``benchmarks/lax_case_pool.py`` registered five pre-registered gates for the
ledger anchor ``CASE-LAX-DG`` and its latest committed run
(``docs/benchmarks/results/lax_case_pool/metrics.json``) leaves the pool
**FAIL**: ``C1``/``C2``/``R5``/``T`` pass, ``C3`` (whole-field relative L1
order ``>= 1.0``) and ``R3`` (analytic-region order ``>= 0.85 (p+1)``) fail,
worst at ``p3`` (density order ``0.484`` against a ``3.4`` requirement).

The pool ran **without any shock-capturing mechanism** (``limiter=None``; only
the Zhang-Shu positivity scaling is active).  The obvious alternative reading
of the two FAIL verdicts is therefore *not* "the solver is wrong" but "the P3
leg integrates an unregularised oscillatory solution".  That reading is
falsifiable with the artifacts already in the tree: the committed
``profiles.npz`` (finest leg, N=400, P3) carries a residual whose error is
**not** at the waves but in the star plateau behind the contact, as a
wavelength-~8-cell ripple; the rarefaction fan is accurate to ~1e-4 and the
undisturbed right state to ~4e-8.

This module answers the falsification question directly and with the *pool's
own gate functions* (``gate_c3`` / ``gate_r3`` imported from
``benchmarks.lax_case_pool``), so a verdict here means the same thing as a
verdict there.

Pre-registered design (written before the numbers)
--------------------------------------------------
* **Mechanisms** -- ``none`` (reference, taken from the committed pool
  metrics), ``tvb-every`` / ``tvb-final`` (TVB limiter, Cockburn-Shu, applied
  after every SSP-RK3 sub-step or only at the end of the step) and
  ``av-every`` (sensor-gated modal artificial viscosity, ``kind="av"``, with
  the library defaults from ``CASE-DG-AV-SHOCK-CAPTURE`` -- *not* tuned on
  Lax).
* **Ladder** -- the two failing degrees only, ``p2`` and ``p3``, at
  ``N = 100 / 200 / 400``.  The pooled ``p1`` legs are left out to bound the
  cost; the scope reduction is declared rather than hidden, and the ``p1``
  legs of the committed pool are quoted next to the results.
* **Gates** -- ``gate_c3`` / ``gate_r3`` verbatim, with ``R3_BAND_WIDTH``
  as in the pool.  Nothing is retuned here; a mechanism that fails keeps its
  FAIL.
* **Oscillation metric** -- for every leg the residual ``rho_h - rho_exact``
  is reduced over three windows: the rarefaction fan, the star-left plateau
  and the star-right plateau (contact -> shock).  The post-contact plateau is
  where unlimited DG piles up its Gibbs ripple, so ``max|residual|`` there is
  the quantity a shock-capturing mechanism must shrink.
* **Reference-reuse check** -- the committed ``p1``/``N=100`` leg is re-run in
  this process and compared digit-wise with the committed value, so reusing
  the committed ``none`` ladder has evidence instead of an assumption.

Result contract
---------------
Every number printed or written here comes from the solver, from the exact
Riemann solution, or from the committed metrics JSON.  Nothing is transcribed.

Artifacts::

    validation/lax_limiter_study/metrics.json    measured values + gate verdicts
    validation/lax_limiter_study/metrics.sha256  artifact digest
    validation/lax_limiter_study/figures/*.png   profiles, convergence, ripple
    validation/lax_limiter_study/summary.md      tables rendered from metrics.json

Usage::

    PYTHONPATH=. python -m benchmarks.lax_limiter_study                 # full
    PYTHONPATH=. python -m benchmarks.lax_limiter_study --mechanisms tvb-final
    PYTHONPATH=. python -m benchmarks.lax_limiter_study --skip-reference
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.lax_case_pool import (
    CFL,
    FINAL_TIME,
    LEFT,
    R3_BAND_WIDTH,
    RIGHT,
    _weighted_l1,
    exact_positions,
    exact_reference,
    exact_waves,
    gate_c3,
    gate_r3,
    modal_fields,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d

# ------------------------------------------------------------------ registry
#: shock-capturing mechanisms under test.  ``limiter_kwargs`` is empty for the
#: ``av`` entry on purpose: the library defaults (from the AV case) are used
#: as-is, so nothing is tuned against the Lax numbers.
MECHANISMS = {
    "none": {"limiter": None, "limiter_stage": "every", "limiter_kwargs": {}},
    "tvb-every": {"limiter": "tvb", "limiter_stage": "every", "limiter_kwargs": {}},
    "tvb-final": {"limiter": "tvb", "limiter_stage": "final", "limiter_kwargs": {}},
    "av-every": {"limiter": "av", "limiter_stage": "every", "limiter_kwargs": {}},
}
DEFAULT_MECHANISMS = ("tvb-every", "tvb-final", "av-every")

#: the pool's failing degrees only (p1 is quoted from the committed metrics)
LADDER = [(100, 2), (200, 2), (400, 2), (100, 3), (200, 3), (400, 3)]
QUICK_LADDER = [(100, 2), (200, 2), (200, 3)]
#: leg re-run to prove the committed ``none`` ladder can be reused as reference
REPRO_LEG = (100, 1)
REFERENCE_METRICS = Path("docs/benchmarks/results/lax_case_pool/metrics.json")
OUT = Path(os.environ.get("LAX_LIMITER_OUT", "validation/lax_limiter_study"))
FIG = OUT / "figures"

#: residual windows (physical coordinates, fixed before the runs).  The star
#: plateaus are pulled in by ``R3_BAND_WIDTH`` from the analytic wave
#: positions, i.e. the same wave band the pool masks out.
WINDOW_SHRINK = R3_BAND_WIDTH


# ---------------------------------------------------------------- measuring
def residual_windows(waves, shrink=WINDOW_SHRINK):
    """Fixed physical windows used by the residual reduction."""
    pos = waves["positions"]
    return {
        "rarefaction_fan": (pos["rarefaction_head"] + shrink,
                            pos["rarefaction_tail"] - shrink),
        "star_left": (pos["rarefaction_tail"] + shrink, pos["contact"] - shrink),
        "star_right": (pos["contact"] + shrink, pos["shock"] - shrink),
        "undisturbed_right": (pos["shock"] + shrink, 1.0 - shrink),
    }


def oscillation_metrics(points, values, exact_values, waves, shrink=WINDOW_SHRINK):
    """Residual statistics per window (``values`` is one primitive field).

    ``sign_flips`` is the fraction of neighbouring node pairs whose residual
    changes sign: a smooth solution gives ~0, a cell-scale ripple gives a
    large fraction.  Together with ``max_abs`` it separates "smeared but
    monotone" from "oscillatory".
    """
    x = np.asarray(points, dtype=float).ravel()
    num = np.asarray(values, dtype=float).ravel()
    ref = np.asarray(exact_values, dtype=float).ravel()
    order = np.argsort(x)
    x, num, ref = x[order], num[order], ref[order]
    residual = num - ref
    out = {}
    for name, (lo, hi) in residual_windows(waves, shrink).items():
        mask = (x >= lo) & (x <= hi)
        if not mask.any():
            out[name] = {"nodes": 0, "max_abs": None, "mean_abs": None,
                         "bias": None, "sign_flips": None}
            continue
        local = residual[mask]
        signs = np.sign(local)
        flips = float((signs[1:] * signs[:-1] < 0).sum()) / max(local.size - 1, 1)
        out[name] = {"nodes": int(mask.sum()),
                     "max_abs": float(np.abs(local).max()),
                     "mean_abs": float(np.abs(local).mean()),
                     "bias": float(local.mean()),
                     "sign_flips": float(flips)}
    return out


def infinite_norm(points, values, exact_values, weights, dx, mask=None):
    """Relative L-infinity norm on the quadrature nodes."""
    weight = 0.5 * dx * weights[None, :] * np.ones_like(values)
    if mask is not None:
        weight = weight * mask
    active = weight > 0.0
    if not active.any():
        return None
    return float(np.abs(values - exact_values)[active].max()
                 / np.abs(exact_values)[active].max())


# --------------------------------------------------------------------- legs
def run_leg(cells, degree, mechanism, waves=None):
    """One (N, p, mechanism) leg: fields, errors, gate inputs, diagnostics."""
    config = MECHANISMS[mechanism] if mechanism in MECHANISMS else mechanism
    waves = waves if waves is not None else exact_waves()
    started = time.time()
    x, coeff, meta = solve_euler_dg_1d(
        LEFT, RIGHT, ncells=cells, degree=degree, final_time=FINAL_TIME,
        cfl=CFL, limiter=config["limiter"],
        limiter_stage=config["limiter_stage"],
        limiter_kwargs=config.get("limiter_kwargs") or None,
    )
    points, primitive, dx, weights = modal_fields(x, coeff)
    exact = exact_reference(points)
    errors = {field: _weighted_l1(points, primitive[field], exact[field],
                                 weights, dx)
              for field in ("density", "velocity", "pressure")}
    mask = np.ones_like(primitive["density"], dtype=bool)
    for name in ("rarefaction_head", "rarefaction_tail", "contact", "shock"):
        distance = np.abs(points - waves["positions"][name])
        mask &= distance > R3_BAND_WIDTH
    analytic_errors = {
        field: _weighted_l1(points, primitive[field], exact[field], weights, dx,
                            mask=mask)
        for field in ("density", "velocity", "pressure")}
    entry = {
        "mechanism": mechanism,
        "cells": cells, "degree": degree,
        "steps": int(meta["steps"]), "dt": float(meta["dt"]),
        "wall_seconds": float(time.time() - started),
        "analytic_fraction": float(mask.mean()),
        "density_min": float(primitive["density"].min()),
        "limiter_reported": meta.get("limiter"),
        "limiter_stage": meta.get("limiter_stage"),
        "positivity_stats": meta.get("positivity_stats", {}),
        "errors": errors,
        "analytic_errors": analytic_errors,
        "linf": {field: infinite_norm(points, primitive[field], exact[field],
                                      weights, dx, mask=mask)
                 for field in ("density", "velocity", "pressure")},
        "oscillation": oscillation_metrics(points, primitive["density"],
                                           exact["density"], waves),
    }
    print(f"[{mechanism}] N={cells:4d} p={degree} L1(rho)={errors['density']:.6f} "
          f"analytic={analytic_errors['density']:.6f} "
          f"ripple={entry['oscillation']['star_right']['max_abs']:.3e} "
          f"t={entry['wall_seconds']:.1f}s", flush=True)
    return entry, {"points": points, "numerical": primitive, "exact": exact}


def run_ladder(mechanism, ladder=LADDER, waves=None):
    entries, profiles = [], {}
    for cells, degree in ladder:
        entry, fields = run_leg(cells, degree, mechanism, waves=waves)
        entries.append(entry)
        if (cells, degree) == ladder[-1]:
            profiles = fields
    return entries, profiles


# -------------------------------------------------------- reference handling
def load_reference_ladder():
    """The committed ``none`` ladder restricted to the degrees under test."""
    data = json.loads(REFERENCE_METRICS.read_text())
    degrees = {degree for _, degree in LADDER}
    kept = [entry for entry in data["ladder"] if entry["degree"] in degrees]
    for entry in kept:
        entry["mechanism"] = "none"
    return data, kept


def reference_reuse_check(waves=None):
    """Re-run one committed leg and measure the discrepancy.

    If the re-run reproduces the committed relative L1 to round-off, reusing
    the committed ``none`` ladder as the reference is justified; if it does
    not, the comparison below would mix a mechanism effect with a drift.
    """
    committed = json.loads(REFERENCE_METRICS.read_text())
    cells, degree = REPRO_LEG
    match = [entry for entry in committed["ladder"]
             if entry["cells"] == cells and entry["degree"] == degree]
    if not match:
        return {"leg": {"cells": cells, "degree": degree}, "status": "unknown",
                "reason": "leg absent from the committed metrics"}
    entry, _ = run_leg(cells, degree, "none", waves=waves)
    out = {"leg": {"cells": cells, "degree": degree}, "rerun": {}, "committed": {}}
    worst = 0.0
    for field in ("density", "velocity", "pressure"):
        now = entry["errors"][field]
        then = float(match[0]["errors"][field])
        relative = abs(now - then) / abs(then)
        worst = max(worst, relative)
        out["rerun"][field] = now
        out["committed"][field] = then
        out["relative_difference"] = None
    out["worst_relative_difference"] = float(worst)
    out["steps_rerun"] = int(entry["steps"])
    out["steps_committed"] = int(match[0]["steps"])
    # 1e-6 is the declared bar: the committed numbers carry 6 significant
    # digits in the docs, so anything below it is a formatting-level match.
    out["status"] = "PASS" if worst < 1e-6 else "FAIL"
    return out


def reference_definition_diagnostic(waves=None):
    """Does gate C1 compare against the *analytic* wave positions?

    ``gate_c1`` compares the numerical level crossing with the level crossing
    of the exact field, both computed by the same locator.  For the contact
    and the shock the locator lands on the discontinuity itself, so the two
    definitions coincide; inside the rarefaction fan the locator returns the
    ``RAREFACTION_LEVEL_FRACTION`` crossing, which is *offset* from the
    analytic characteristic edge.  The offset is measured here in the gate's
    own normalisation ``|dx| / |x_exact - x0|`` so a reader can see whether
    the PASS number in the pool is a wave-edge accuracy or a
    fan-self-consistency number.
    """
    waves = waves if waves is not None else exact_waves()
    located = exact_positions(waves)
    rows, worst = {}, 0.0
    for name, analytic in waves["positions"].items():
        crossing = located.get(name)
        if crossing is None:
            rows[name] = {"analytic": float(analytic), "locator": None,
                          "normalised_offset": None}
            continue
        normalised = abs(crossing - analytic) / abs(analytic - 0.5)
        worst = max(worst, normalised)
        rows[name] = {"analytic": float(analytic), "locator": float(crossing),
                      "absolute_offset": float(abs(crossing - analytic)),
                      "normalised_offset": float(normalised)}
    same = [name for name, row in rows.items()
            if row["normalised_offset"] is not None
            and row["normalised_offset"] < 1e-3]
    return {"normalisation": "|x_locator - x_analytic| / |x_analytic - x0|",
            "tolerance_of_gate_c1": 0.02,
            "rows": rows, "worst_normalised_offset": float(worst),
            "coincident_rows": sorted(same),
            "reading": ("C1 compares numerical and exact fields through the same "
                        "level locator; fan rows are a level-set measure, not the "
                        "analytic characteristic edge"),
            }


# ------------------------------------------------------------------- gates
def mechanism_gates(entries):
    """Run the pool's own C3/R3 gates on one mechanism's ladder."""
    return {"c3": gate_c3(entries),
            "r3": gate_r3(entries, band={"width": R3_BAND_WIDTH,
                                         "rule": "fixed physical half width"}),
            "verdict": ("PASS" if gate_c3(entries)["status"] == "PASS"
                        and gate_r3(entries, band={"width": R3_BAND_WIDTH})["status"] == "PASS"
                        else "FAIL")}


def order_by_mechanism(entries):
    """(degree, field) finest-pair orders, plus the ripple trend."""
    table = {}
    for entry in entries:
        table.setdefault(entry["mechanism"], {}).setdefault(
            entry["degree"], []).append(entry)
    out = {}
    for mechanism, by_degree in table.items():
        rows = {}
        for degree, legs in by_degree.items():
            legs = sorted(legs, key=lambda e: e["cells"])
            if len(legs) < 2:
                continue
            coarse, fine = legs[-2], legs[-1]
            rows[f"p{degree}"] = {
                "cells": [coarse["cells"], fine["cells"]],
                "density": _order(coarse["cells"], coarse["errors"]["density"],
                                  fine["cells"], fine["errors"]["density"]),
                "analytic_density": _order(
                    coarse["cells"], coarse["analytic_errors"]["density"],
                    fine["cells"], fine["analytic_errors"]["density"]),
                # committed pool legs carry no oscillation block; the ripple
                # trend is then reported as n/a instead of crashing the run
                "ripple_star_right": [
                    (coarse.get("oscillation") or {}).get("star_right", {}).get("max_abs"),
                    (fine.get("oscillation") or {}).get("star_right", {}).get("max_abs")],
            }
        out[mechanism] = rows
    return out


def _order(coarse_cells, coarse_error, fine_cells, fine_error):
    if not coarse_error or not fine_error or coarse_error <= 0.0 or fine_error <= 0.0:
        return None
    return float(math.log(coarse_error / fine_error)
                 / math.log(fine_cells / coarse_cells))


# ------------------------------------------------------------------ figures
def render_figures(profiles, entries, orders, out_dir=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    colours = {"none": "tab:blue", "tvb-every": "tab:red",
               "tvb-final": "tab:orange", "av-every": "tab:green"}

    if profiles:
        reference = next(iter(profiles.values()))
        points = np.asarray(reference["points"], dtype=float).ravel()
        order = np.argsort(points)
        fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)
        for field, ax in zip(("density", "pressure"), axes):
            ax.plot(points[order],
                    np.asarray(reference["exact"][field], dtype=float).ravel()[order],
                    "-", color="k", lw=1.2, label="exact")
            for mechanism, data in profiles.items():
                ax.plot(points[order],
                        np.asarray(data["numerical"][field], dtype=float).ravel()[order],
                        ".", ms=1.8, mfc="none", color=colours.get(mechanism, "C0"),
                        label=mechanism)
            ax.set_ylabel(field)
        axes[-1].set_xlim(0.6, 0.9)
        axes[-1].set_xlabel("x")
        axes[0].legend(loc="upper right", fontsize=8, ncol=2)
        axes[0].set_title("Lax tube, N=400 P3: contact/shock zoom, per mechanism")
        fig.tight_layout()
        path = out_dir / "lax_mechanisms_profiles.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        files.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, key, title in ((axes[0], "analytic_errors", "analytic-region rel L1"),
                           (axes[1], "errors", "whole-field rel L1")):
        for mechanism in sorted({e["mechanism"] for e in entries}):
            legs = sorted([e for e in entries
                           if e["mechanism"] == mechanism and e["degree"] == 3],
                          key=lambda e: e["cells"])
            if not legs:
                continue
            cells = np.array([e["cells"] for e in legs], dtype=float)
            err = np.array([e[key]["density"] for e in legs])
            ax.loglog(cells, err, "o-", color=colours.get(mechanism, "C0"),
                      label=mechanism)
        ax.set_xlabel("cells (P3 legs)")
        ax.set_ylabel(r"relative $L_1$ (density)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "lax_mechanisms_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for mechanism in sorted({e["mechanism"] for e in entries}):
        legs = sorted([e for e in entries
                       if e["mechanism"] == mechanism and e["degree"] == 3],
                      key=lambda e: e["cells"])
        if not legs:
            continue
        cells = np.array([e["cells"] for e in legs], dtype=float)
        ripple = np.array([e["oscillation"]["star_right"]["max_abs"] for e in legs])
        ax.loglog(cells, ripple, "o-", color=colours.get(mechanism, "C0"),
                  label=mechanism)
    ax.set_xlabel("cells (P3 legs)")
    ax.set_ylabel(r"max $|\rho_h-\rho|$ on post-contact plateau")
    ax.set_title("Gibbs ripple the mechanisms have to remove")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "lax_mechanisms_ripple.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)
    return files


# ------------------------------------------------------------------ summary
def render_summary(payload):
    rows = []
    for mechanism, gates in payload["gates"].items():
        rows.append((mechanism, gates["c3"]["status"], gates["r3"]["status"],
                     gates["verdict"]))
    lines = ["# CASE-LAX-DG · 机制对照（程序化注入）", "",
             f"generated_at_utc = {payload['generated_at_utc']}", "",
             "| mechanism | C3 | R3 | verdict |", "|---|---|---|---|"]
    lines += [f"| {m} | {c} | {r} | {v} |" for m, c, r, v in rows]
    lines += ["", "| mechanism | degree | density order | analytic order | ripple (coarse->fine) |",
              "|---|---|---|---|---|"]
    for mechanism, by_degree in payload["orders"].items():
        for degree, row in by_degree.items():
            ripple = row["ripple_star_right"]
            lines.append(f"| {mechanism} | {degree} | {_fmt(row['density'])} | "
                         f"{_fmt(row['analytic_density'])} | "
                         f"{_sci(ripple[0])} -> {_sci(ripple[1])} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    return OUT / "summary.md"


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _sci(value):
    return "n/a" if value is None else f"{value:.3e}"


# --------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mechanisms", default=",".join(DEFAULT_MECHANISMS),
                        help="comma separated subset of " + ", ".join(MECHANISMS))
    parser.add_argument("--quick", action="store_true",
                        help="reduced ladder, smoke only (verdicts not final)")
    parser.add_argument("--skip-reference", action="store_true",
                        help="do not re-run the committed leg / load the pool ladder")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    mechanisms = [name.strip() for name in args.mechanisms.split(",") if name.strip()]
    for name in mechanisms:
        if name not in MECHANISMS:
            raise SystemExit(f"unknown mechanism {name!r}")
    ladder = QUICK_LADDER if args.quick else LADDER
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    waves = exact_waves()

    payload = {"case": "lax-1954-shock-tube", "anchor": "CASE-LAX-DG",
               "config": {"ladder": ladder, "cfl": CFL, "final_time": FINAL_TIME,
                          "left_state": LEFT, "right_state": RIGHT,
                          "mechanicsm": {name: MECHANISMS[name] for name in mechanisms},
                          "quick": bool(args.quick),
                          "r3_band_width": R3_BAND_WIDTH,
                          "residual_windows": residual_windows(waves)},
               "waves": waves["positions"],
               "environment": {"python": platform.python_version(),
                               "numpy": np.__version__,
                               "torch": torch.__version__,
                               "platform": platform.platform()},
               "reference_definition": reference_definition_diagnostic(waves)}

    entries, profiles, gates, orders = [], {}, {}, {}
    if not args.skip_reference:
        payload["reference_reuse_check"] = reference_reuse_check(waves)
        _, reference_entries = load_reference_ladder()
        entries.extend(reference_entries)
        gates["none"] = mechanism_gates(reference_entries)
    for mechanism in mechanisms:
        mechanism_entries, mechanism_profiles = run_ladder(mechanism, ladder, waves=waves)
        entries.extend(mechanism_entries)
        gates[mechanism] = mechanism_gates(mechanism_entries)
        profiles[mechanism] = mechanism_profiles
    payload["ladder"] = entries
    payload["gates"] = gates
    payload["orders"] = order_by_mechanism(entries)
    payload["verdict"] = ("PASS" if gates and all(g["verdict"] == "PASS"
                                                 for g in gates.values()) else "FAIL")
    payload["wall_seconds"] = float(time.time() - started)

    if not args.no_figures:
        payload["figures"] = [str(path) for path in
                              render_figures(profiles, entries, orders)]
    metrics = OUT / "metrics.json"
    metrics.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
    (OUT / "metrics.sha256").write_text(f"{digest}  metrics.json\n")
    render_summary(payload)
    print(f"[lax-limiter] verdict={payload['verdict']} wall={payload['wall_seconds']:.1f}s "
          f"sha256={digest[:16]}")
    return payload


if __name__ == "__main__":
    main()