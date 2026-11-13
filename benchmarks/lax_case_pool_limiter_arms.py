"""CASE-LAX-DG -- limiter arms: is the ``C3``/``R3`` deficit a method property?

Motivation (attribution debt on an existing honest FAIL)
-------------------------------------------------------
``benchmarks/lax_case_pool.py`` publishes ``verdict = FAIL`` driven by two
*order* gates on the Lax shock tube: ``C3`` (whole-field relative L1 order,
requirement ``>= 1.0``) and ``R3`` (analytic-region order, requirement
``>= 0.85 (p + 1)``).  The published run calls the solver **without a shock
limiter** (``limiter=None``: only the Zhang--Shu positivity scaling is active),
so the FAIL is currently attributable to "this DG discretisation on this case"
without separating *which* ingredient is responsible.  Two very different
readings fit the same numbers:

* (M) **method property** -- an unlimited modal DG solution of a strong shock
  carries Gibbs oscillations over the whole domain; their amplitude decays
  slower than ``h**(p+1)``, so neither the whole-field nor the analytic-region
  order can reach the design order.  No runtime knob changes that.
* (D) **driver default** -- the library ships TVB (Cockburn--Shu 1989) and WENO
  (Qiu--Shu 2005) limiters, and the pool simply never switched one on.  Then
  ``C3``/``R3`` would flip to PASS with a limiter and the ledger verdict would
  be limited by a default, not by the scheme.

This driver separates (M) from (D) by changing **exactly one knob**, the
limiter, and re-adjudicating the *same* pre-registered gates.

Pre-registration (written before the run; never extended afterwards)
--------------------------------------------------------------------
Arms (all other settings are library defaults, identical to the published
pool: ``cfl = 0.35``, ``positivity = "scale"``, ``limiter_stage = "every"``,
``tvb_constant = 1.0``, HLLC, SSP-RK3, float64, ``10``-point Gauss output):

=====  ====================================  ==================================
arm    solver argument                       reading
=====  ====================================  ==================================
none   ``limiter=None``                      baseline, must reproduce the pool
tvb    ``limiter="tvb"``                     TVB with the shipped default M=1.0
weno   ``limiter="weno"``                    WENO, no tunable parameter exposed
=====  ====================================  ==================================

Tolerances are **imported** from :mod:`benchmarks.lax_case_pool`
(``TOL_C1 = 2 %``, ``TOL_C2 = 5 %``, ``MIN_C3_ORDER = 1.0``,
``R3_ORDER_FRACTION = 0.85``, ``R5_ORDER_FRACTION = 3 %``) and the R3 region
rule is the fixed physical band ``R3_BAND_WIDTH = 0.02`` (m16606 F1).  No arm
argument is swept or re-tuned after this file's first run; ``tvb_constant`` is
never touched, so the study cannot cherry-pick ``M``.

Reproduction check
------------------
Arm ``none`` must reproduce the tracked pool record
(``docs/benchmarks/results/lax_case_pool/metrics.json``) leg by leg; the
comparison is emitted as ``reproduction`` in this driver's metrics so a
reviewer sees a number, not a promise.

Artifacts (tracked paths only -- ``runs/`` is not carried into ``main``)::

    docs/benchmarks/results/lax_case_pool_limiter_arms/metrics.json
    docs/benchmarks/results/lax_case_pool_limiter_arms/profiles.npz
    docs/benchmarks/results/lax_case_pool_limiter_arms/summary.md
    docs/benchmarks/results/lax_case_pool_limiter_arms/figures/*.png
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.lax_case_pool import (
    CFL, DOMAIN, ENTROPY_AMPLITUDE, ENTROPY_CELLS, ENTROPY_CFL, ENTROPY_DEGREES,
    ENTROPY_PRESSURE, ENTROPY_SPEEDS, ENTROPY_TIME, FINAL_TIME, LADDER, LEFT,
    MIN_C3_ORDER, QUICK_LADDER, R3_BAND_WIDTH, R3_LEGACY_BAND_CELLS,
    R3_ORDER_FRACTION, R5_ORDER_FRACTION, RIGHT, TOL_C1, TOL_C2,
    _weighted_l1, entropy_wave_exact, exact_positions, exact_reference,
    exact_waves, gate_c1, gate_c2, gate_c3, gate_r3, gate_r5, modal_fields,
    plateau_values, wave_positions,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d

#: pre-registered arms -- one knob, three values, library defaults everywhere
ARMS = {
    "none": {},
    "tvb": {"limiter": "tvb"},
    "weno": {"limiter": "weno"},
}

OUT = Path("docs/benchmarks/results/lax_case_pool_limiter_arms")
POOL_RECORD = Path("docs/benchmarks/results/lax_case_pool/metrics.json")


def _leg(cells, degree, arm):
    """One ladder leg for one arm -- identical post-processing for every arm."""
    kwargs = ARMS[arm]
    x, coeff, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=cells, degree=degree,
                                       final_time=FINAL_TIME, cfl=CFL, **kwargs)
    points, primitive, dx, weights = modal_fields(x, coeff)
    exact = exact_reference(points)
    errors = {field: _weighted_l1(points, primitive[field], exact[field], weights, dx)
              for field in ("density", "velocity", "pressure")}
    mask = np.ones_like(primitive["density"], dtype=bool)
    mask_legacy = np.ones_like(mask, dtype=bool)
    waves = exact_waves()
    for name in ("rarefaction_head", "rarefaction_tail", "contact", "shock"):
        distance = np.abs(points - waves["positions"][name])
        mask &= distance > R3_BAND_WIDTH
        mask_legacy &= distance > R3_LEGACY_BAND_CELLS * dx
    analytic = {field: _weighted_l1(points, primitive[field], exact[field],
                                    weights, dx, mask=mask)
                for field in ("density", "velocity", "pressure")}
    analytic_legacy = {field: _weighted_l1(points, primitive[field], exact[field],
                                           weights, dx, mask=mask_legacy)
                       for field in ("density", "velocity", "pressure")}
    entry = {"cells": cells, "degree": degree, "arm": arm,
             "steps": int(meta["steps"]), "dt": float(meta["dt"]),
             "limiter": meta["limiter"],
             "positivity_stats": meta.get("positivity_stats"),
             "analytic_fraction": float(mask.mean()),
             "analytic_fraction_legacy": float(mask_legacy.mean()),
             "density_min": float(primitive["density"].min()),
             "errors": errors, "analytic_errors": analytic,
             "analytic_errors_legacy_band": analytic_legacy}
    return entry, (points, primitive, dx, weights), exact


def run_ladder(arm, ladder=LADDER):
    """Solves the pool ladder for one arm; keeps the finest leg for C1/C2."""
    waves = exact_waves()
    entries, finest, profiles = [], None, None
    for cells, degree in ladder:
        started = time.time()
        entry, (points, primitive, dx, weights), exact = _leg(cells, degree, arm)
        entry["wall_seconds"] = float(time.time() - started)
        entries.append(entry)
        print(f"[lax-arms] arm={arm:4s} N={cells:4d} p={degree} "
              f"L1(rho)={entry['errors']['density']:.6e} "
              f"ana={entry['analytic_errors']['density']:.6e} "
              f"t={entry['wall_seconds']:.1f}s", flush=True)
        if (cells, degree) == ladder[-1]:
            finest = {"cells": cells, "degree": degree,
                      "positions": wave_positions(points, primitive, waves, dx=dx),
                      "plateaus": plateau_values(points, primitive, waves)}
            profiles = {"points": points, "numerical": primitive, "exact": exact}
    return waves, exact_positions(waves), entries, finest, profiles


def run_entropy(arm, ladder=ENTROPY_CELLS, degrees=ENTROPY_DEGREES,
                speeds=ENTROPY_SPEEDS):
    """Smooth entropy-wave ladder (R5) per arm: does the limiter keep order?"""
    kwargs = ARMS[arm]
    entries = []
    for degree in degrees:
        for speed in speeds:
            measured = {}
            for cells in ladder:
                def initial_state(node_x, speed=speed):
                    rho = 1.0 + ENTROPY_AMPLITUDE * np.sin(2.0 * np.pi * node_x)
                    return (rho, torch.full_like(rho, speed),
                            torch.full_like(rho, ENTROPY_PRESSURE))

                x, coeff, _ = solve_periodic_euler_dg_1d(
                    initial_state, ncells=cells, degree=degree,
                    final_time=ENTROPY_TIME, domain=DOMAIN, cfl=ENTROPY_CFL,
                    **kwargs)
                points, primitive, dx, weights = modal_fields(x, coeff)
                exact = entropy_wave_exact(points, ENTROPY_TIME, speed)
                measured[cells] = _weighted_l1(points, primitive["density"], exact,
                                               weights, dx)
            cells = sorted(measured)
            order = None
            if measured[cells[-2]] > 0.0 and measured[cells[-1]] > 0.0:
                order = (np.log(measured[cells[-2]] / measured[cells[-1]])
                         / np.log(cells[-1] / cells[-2]))
            entries.append({"arm": arm, "degree": degree, "speed": speed,
                            "cells": list(cells),
                            "relative_l1": {str(c): measured[c] for c in cells},
                            "order": None if order is None else float(order)})
            print(f"[lax-arms] entropy arm={arm:4s} u={speed} p={degree} "
                  f"order={order}", flush=True)
    return entries


def reproduction_check(entries, record=POOL_RECORD):
    """Arm ``none`` vs the tracked pool record: max relative difference."""
    if not record.exists():
        return {"status": "NOT_RUN", "reason": f"{record} missing"}
    published = {(int(e["cells"]), int(e["degree"])): e
                 for e in json.loads(record.read_text())["ladder"]}
    worst, worst_key, rows = 0.0, None, []
    for entry in entries:
        key = (entry["cells"], entry["degree"])
        if key not in published:
            continue
        row = {"cells": key[0], "degree": key[1]}
        for kind in ("errors", "analytic_errors"):
            for field, value in entry[kind].items():
                ref = published[key][kind][field]
                rel = abs(value - ref) / abs(ref)
                row[f"{kind}.{field}"] = {"published": ref, "measured": value,
                                          "relative_difference": float(rel)}
                if rel > worst:
                    worst, worst_key = rel, f"{key}.{kind}.{field}"
        rows.append(row)
    return {"status": "PASS" if worst < 1e-9 else "FAIL",
            "metric": "max relative difference of every ladder leg and field",
            "worst_relative_difference": float(worst), "worst_key": worst_key,
            "record": str(record), "legs": rows}


def run_arm(arm, ladder=LADDER, entropy_ladder=ENTROPY_CELLS):
    started = time.time()
    waves, exact_pos, entries, finest, profiles = run_ladder(arm, ladder)
    entropy = run_entropy(arm, entropy_ladder)
    c1 = gate_c1(finest["positions"], exact_pos)
    c2 = gate_c2(finest["plateaus"], waves)
    c3 = gate_c3(entries)
    r3 = gate_r3(entries, "analytic_errors",
                 {"width": R3_BAND_WIDTH,
                  "rule": "fixed physical band (nested), m16606 F1"})
    r3_legacy = gate_r3(entries, "analytic_errors_legacy_band",
                        {"legacy_cells": R3_LEGACY_BAND_CELLS,
                         "rule": "retired 4-cell band (not nested)"})
    r5 = gate_r5(entropy)
    gates = {"c1": c1, "c2": c2, "c3": c3, "r3": r3, "r5": r5}
    verdict = "PASS" if all(g["status"] == "PASS" for g in gates.values()) else "FAIL"
    out = {"arm": arm, "arguments": ARMS[arm], "ladder": entries,
           "finest": finest, "exact_positions": exact_pos, "waves": waves,
           "entropy_wave": entropy, "gates": gates,
           "diagnostics": {"r3_legacy_band": r3_legacy},
           "verdict": verdict, "wall_seconds": float(time.time() - started)}
    print(f"[lax-arms] arm={arm} verdict={verdict} "
          f"C3={c3['gated_order']:.3f} R3(p)="
          f"{[round(r3['per_leg'].get(f'p{d}.density', float('nan')), 3) for d in (1, 2, 3)]}",
          flush=True)
    return out, profiles


def _analytic_order_table(arm_results):
    rows = {}
    for arm, result in arm_results.items():
        rows[arm] = {}
        for degree in sorted({e["degree"] for e in result["ladder"]}):
            legs = sorted([e for e in result["ladder"] if e["degree"] == degree],
                          key=lambda e: e["cells"])
            rows[arm][f"p{degree}"] = {
                "cells": [e["cells"] for e in legs],
                "whole_field": [e["errors"]["density"] for e in legs],
                "analytic": [e["analytic_errors"]["density"] for e in legs],
                "steps": [e["steps"] for e in legs],
                "r3_order": result["gates"]["r3"]["per_leg"].get(f"p{degree}.density"),
                "r3_requirement": R3_ORDER_FRACTION * (degree + 1),
                "c3_order": result["gates"]["c3"]["per_leg"].get(f"p{degree}.density"),
            }
    return rows


def render_figures(arm_results, profiles, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    table = _analytic_order_table(arm_results)
    colors = {"none": "tab:blue", "tvb": "tab:orange", "weno": "tab:green"}

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    for ax, key, title in ((axes[0], "whole_field", "whole field"),
                           (axes[1], "analytic", "analytic regions (band 0.02)")):
        for arm in table:
            for degree in table[arm]:
                cells = np.array(table[arm][degree]["cells"], dtype=float)
                err = np.array(table[arm][degree][key])
                style = {"p1": "-o", "p2": "-s", "p3": "-^"}.get(degree, "-.")
                ax.loglog(cells, err, style, ms=3.0, color=colors.get(arm, "k"),
                          alpha=0.9, label=f"{arm} {degree}")
        ax.set_xlabel("cells")
        ax.set_ylabel(r"relative $L_1$ (density)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=5.5, ncol=2, loc="lower left")
    axes[2].set_axis_off()
    axes[2].text(0.0, 0.95, "R3 measured order / requirement\n", fontsize=8,
                 va="top", family="monospace")
    lines = []
    for arm in table:
        for degree in table[arm]:
            got = table[arm][degree]["r3_order"]
            need = table[arm][degree]["r3_requirement"]
            ok = "PASS" if got is not None and got >= need else "FAIL"
            lines.append(f"{arm:4s} {degree}: {got if got is None else round(got, 3)}"
                         f" / {need:.2f}  {ok}")
    axes[2].text(0.0, 0.86, "\n".join(lines), fontsize=7, va="top",
                 family="monospace")
    fig.tight_layout()
    path = out_dir / "lax_limiter_arms_orders.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)

    fig, axes = plt.subplots(len(profiles), 1, figsize=(7.4, 2.6 * len(profiles)),
                             sharex=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, arm in zip(axes, profiles):
        data = profiles[arm]
        points = np.asarray(data["points"]).ravel()
        order = np.argsort(points)
        ax.plot(points[order], np.asarray(data["numerical"]["density"]).ravel()[order],
                "o", ms=2.0, mfc="none", color=colors.get(arm, "k"),
                label=f"DG ({arm})")
        ax.plot(points[order], np.asarray(data["exact"]["density"]).ravel()[order],
                "-", lw=1.0, color="k", label="exact")
        for name, position in arm_results[arm]["waves"]["positions"].items():
            ax.axvline(position, color="tab:red", lw=0.5, ls=":")
        ax.set_ylabel(r"$\rho$")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"arm = {arm}", fontsize=8)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    path = out_dir / "lax_limiter_arms_profiles.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    files.append(path)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default=",".join(ARMS),
                        help="comma separated subset of %s" % ",".join(ARMS))
    parser.add_argument("--quick", action="store_true",
                        help="reduced ladder (smoke test, not a verdict run)")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--ladder", default=None,
                        help="restrict the ladder, e.g. '100:1,200:1,400:1' "
                             "(a reduced ladder must be declared in the report)")
    parser.add_argument("--entropy-quick", action="store_true",
                        help="short entropy ladder (16, 32) -- not a verdict run")
    parser.add_argument("--skip-entropy", action="store_true",
                        help="skip the R5 smooth ladder (not a verdict run)")
    args = parser.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; pre-registered: {list(ARMS)}")
    if args.ladder:
        ladder = [tuple(int(v) for v in pair.split(":"))
                  for pair in args.ladder.split(",") if pair.strip()]
    else:
        ladder = QUICK_LADDER if args.quick else LADDER
    entropy_ladder = ((16, 32) if (args.quick or args.entropy_quick)
                      else ENTROPY_CELLS)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    arm_results, profiles, metrics_arms = {}, {}, {}
    for arm in arms:
        result, prof = run_arm(arm, ladder, entropy_ladder)
        if args.skip_entropy:
            result["entropy_wave"] = []
            result["gates"].pop("r5")
            result["verdict"] = ("PASS" if all(g["status"] == "PASS"
                                              for g in result["gates"].values())
                                 else "FAIL")
        arm_results[arm] = result
        profiles[arm] = prof
        metrics_arms[arm] = {k: v for k, v in result.items() if k != "waves"}

    reproduction = (reproduction_check(arm_results["none"]["ladder"])
                    if "none" in arm_results else {"status": "NOT_RUN"})
    table = _analytic_order_table(arm_results)
    figures: list[Path] = []
    np.savez(out / "profiles.npz", **{
        f"{arm}_{which}_{field}": np.asarray(value)
        for arm, prof in profiles.items()
        for which, bundle in (("numerical", prof["numerical"]),
                              ("exact", prof["exact"]))
        for field, value in bundle.items()
    } | {f"{arm}_points": np.asarray(prof["points"])
         for arm, prof in profiles.items()})

    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "study": "limiter arms (single-knob attribution of C3/R3)",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pre_registration": {
            "arms": ARMS,
            "changed_knob": "limiter (kernel argument), nothing else",
            "inherited_tolerances": {
                "c1": TOL_C1, "c2": TOL_C2, "c3_min_order": MIN_C3_ORDER,
                "r3_order_fraction": R3_ORDER_FRACTION,
                "r5_order_fraction": R5_ORDER_FRACTION,
                "r3_band_width": R3_BAND_WIDTH},
            "declared_before_run": True,
            "not_swept": ["tvb_constant (library default 1.0)",
                          "limiter_stage (library default 'every')",
                          "positivity (library default 'scale')",
                          "cfl (0.35)"],
        },
        "config": {"cfl": CFL, "ladder": ladder, "quick": args.quick,
                   "reduced_ladder": bool(args.ladder),
                   "entropy_ladder": list(entropy_ladder),
                   "final_time": FINAL_TIME},
        "arms": metrics_arms,
        "analytic_orders": table,
        "reproduction": reproduction,
        "figures": [str(p) for p in figures],
        "verdicts": {arm: result["verdict"] for arm, result in arm_results.items()},
        "wall_seconds": float(time.time() - started),
    }
    path = out / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    (out / "summary.md").write_text(render_summary(metrics))
    if not args.skip_figures:
        try:
            figures = render_figures(arm_results, profiles, out / "figures")
        except Exception as error:  # never lose the record to a plotting bug
            print(f"[lax-arms] WARNING figures failed: {error!r}", flush=True)
        metrics["figures"] = [str(fig) for fig in figures]
        path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    print(f"[lax-arms] wrote {path} verdicts={metrics['verdicts']}", flush=True)
    return metrics


def render_summary(metrics):
    rows = ["# CASE-LAX-DG limiter arms -- machine summary", "",
            f"generated: {metrics['generated_at_utc']}", "",
            "| arm | C1 | C2 | C3 | R3 | R5 | verdict |",
            "|---|---|---|---|---|---|---|"]
    for arm, result in metrics["arms"].items():
        gates = result["gates"]
        rows.append("| " + " | ".join([arm] + [
            gates[key]["status"] if key in gates else "not run"
            for key in ("c1", "c2", "c3", "r3", "r5")] + [result["verdict"]]) + " |")
    rows += ["", "| arm | degree | whole-field order | analytic order | "
                 "R3 requirement | R3 |", "|---|---|---|---|---|---|"]
    for arm, per_degree in metrics["analytic_orders"].items():
        for degree, data in per_degree.items():
            got = data["r3_order"]
            rows.append(f"| {arm} | {degree} | "
                        f"{'n/a' if data['c3_order'] is None else round(data['c3_order'], 4)} | "
                        f"{'n/a' if got is None else round(got, 4)} | "
                        f"{data['r3_requirement']:.2f} | "
                        f"{'PASS' if got is not None and got >= data['r3_requirement'] else 'FAIL'} |")
    rep = metrics["reproduction"]
    rows += ["", f"reproduction (arm=none vs tracked pool record): "
                 f"{rep.get('status')} "
                 f"max rel diff = {rep.get('worst_relative_difference')}", ""]
    return "\n".join(rows)


if __name__ == "__main__":
    main()