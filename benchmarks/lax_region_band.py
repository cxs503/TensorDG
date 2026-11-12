"""CASE-LAX-DG -- does the ``R3`` analytic-region order survive a *nested* region?

Motivation (verification debt, not a new physics case)
------------------------------------------------------
``benchmarks/lax_case_pool.py`` gates ``R3`` on a relative L1 order measured
outside a band of ``4 * h`` around each analytic wave (``h`` = cell width).
That band **shrinks with refinement**, so the measured region is *not nested*:
at ``N = 100`` (P1) the excluded band is ``0.04`` wide, at ``N = 400`` it is
``0.01``.  Points that are *excluded* on the coarse leg are therefore *included*
on the fine leg, and those points sit closest to the smeared discontinuity where
the local error decays slowest.  A measured order below ``p + 1`` can be either

* (M) a genuine method property (limiter / discontinuity-driven pollution), or
* (A) an artifact of a non-nested measurement region.

The published ``R3 = FAIL`` cannot tell (M) from (A).  This driver re-measures
the *same* DG solutions under four region rules that differ **only** in how the
wave bands are cut:

===== ====================================================================  ==========================
rule  region                                                                  independent of ``h``?
===== ====================================================================  ==========================
A     baseline: exclude ``4 * h`` around each analytic wave (published rule)  no  (band -> 0)
B     nested: exclude a fixed physical ``0.02`` around each analytic wave    yes
C     nested, narrower: fixed physical ``0.01``                              yes
D     nested, wide: fixed physical ``0.05``                                  yes
===== ====================================================================  ==========================

The four masks are applied to the *same* solver output (one solve per ladder
leg), so differences in the measured order are attributable to the region rule
alone -- no other knob is touched.  Wave positions come from the analytic
solution (``lax_case_pool.exact_positions``), never from the numerical field,
so the regions cannot drift with the solution.

Artifacts (inside the worktree)::

    runs/lax_region_band/metrics.json
    runs/lax_region_band/summary.md
    runs/lax_region_band/figures/region_order.png
    runs/lax_region_band/figures/region_measure.png

Usage::

    python -m benchmarks.lax_region_band            # full ladder (~5 CPU minutes)
    python -m benchmarks.lax_region_band --quick    # smoke only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.lax_case_pool import (CFL, FINAL_TIME, LADDER, LEFT,
                                      R3_BAND_CELLS, R3_ORDER_FRACTION, RIGHT,
                                      convergence_order, exact_positions,
                                      exact_reference, exact_waves, modal_fields,
                                      solve_euler_dg_1d, _weighted_l1)

OUT = Path("runs/lax_region_band")
WAVES = ("rarefaction_head", "rarefaction_tail", "contact", "shock")

#: band half-width in *physical* units for the nested rules
NESTED_BANDS = {"nested_0p01": 0.01, "nested_0p02": 0.02, "nested_0p05": 0.05}
BASELINE_RULE = "band_4h"

#: smoke ladder -- every degree keeps a full coarse->fine pair so that orders exist
QUICK_LADDER = [(100, 1), (200, 1), (100, 2), (200, 2)]


def region_masks(points, cell_width, positions):
    """Boolean masks (and retained physical measure) for every region rule."""
    masks, measures = {}, {}
    base = np.ones(points.shape, dtype=bool)
    for name in WAVES:
        base &= np.abs(points - positions[name]) > R3_BAND_CELLS * cell_width
    masks[BASELINE_RULE] = base
    for label, half_width in NESTED_BANDS.items():
        mask = np.ones(points.shape, dtype=bool)
        for name in WAVES:
            mask &= np.abs(points - positions[name]) > half_width
        masks[label] = mask
    # retained physical length (measure of the region actually integrated over)
    for label, mask in masks.items():
        # weights are detached below by the caller; here reuse uniform cell weight
        measures[label] = float(mask.mean())
    return masks, measures


def run_ladder(ladder=LADDER):
    waves = exact_waves()
    positions = exact_positions(waves)
    entries = []
    for cells, degree in ladder:
        started = time.time()
        x, coeff, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=cells, degree=degree,
                                           final_time=FINAL_TIME, cfl=CFL)
        points, primitive, cell_width, weights = modal_fields(x, coeff)
        exact = exact_reference(points)
        masks, _ = region_masks(points, cell_width, positions)
        cell_weight = 0.5 * cell_width * weights[None, :] * np.ones_like(points)
        rules = {}
        for label, mask in masks.items():
            rules[label] = {
                "band_width": float(R3_BAND_CELLS * cell_width) if label == BASELINE_RULE
                else float(NESTED_BANDS[label]),
                "retained_fraction": float(mask.mean()),
                "retained_length": float(cell_weight.sum() * mask.mean()),
                "errors": {field: _weighted_l1(points, primitive[field], exact[field],
                                               weights, cell_width, mask=mask)
                           for field in ("density", "velocity", "pressure")},
            }
        entries.append({"cells": cells, "degree": degree,
                        "steps": int(meta["steps"]),
                        "wall_seconds": float(time.time() - started),
                        "rules": rules})
        print(f"[band] N={cells:4d} p={degree} "
              + " ".join(f"{k}:{v['errors']['density']:.3e}"
                         for k, v in rules.items())
              + f" t={entries[-1]['wall_seconds']:.1f}s", flush=True)
    return entries


def rule_orders(entries):
    """Finest-pair order of every rule for every degree (density is gated)."""
    out = {}
    for label in (BASELINE_RULE, *NESTED_BANDS):
        per_degree = {}
        for degree in sorted({e["degree"] for e in entries}):
            legs = sorted((e for e in entries if e["degree"] == degree),
                          key=lambda e: e["cells"])
            if len(legs) < 2:
                continue
            coarse, fine = legs[-2], legs[-1]
            per_degree[f"p{degree}"] = {
                "coarse_cells": coarse["cells"], "fine_cells": fine["cells"],
                "order": {field: convergence_order(
                    coarse["cells"],
                    coarse["rules"][label]["errors"][field],
                    fine["cells"],
                    fine["rules"][label]["errors"][field])
                    for field in ("density", "velocity", "pressure")},
                "retained_fraction": {
                    "coarse": coarse["rules"][label]["retained_fraction"],
                    "fine": fine["rules"][label]["retained_fraction"]},
                "band_width": {"coarse": coarse["rules"][label]["band_width"],
                               "fine": fine["rules"][label]["band_width"]},
            }
        out[label] = per_degree
    return out


def region_verdicts(orders):
    verdicts = {}
    for label, per_degree in orders.items():
        detail, ok = {}, True
        for key, data in per_degree.items():
            degree = int(key[1:])
            requirement = R3_ORDER_FRACTION * (degree + 1)
            order = data["order"]["density"]
            passed = order is not None and order >= requirement
            detail[key] = {"order": order, "requirement": requirement,
                           "ratio": None if order is None else order / requirement,
                           "status": "PASS" if passed else "FAIL"}
            ok = ok and passed
        verdicts[label] = {"per_degree": detail, "status": "PASS" if ok else "FAIL"}
    return verdicts


def nestedness(entries):
    """How much the retained region moves between the coarsest and finest leg."""
    rows = {}
    for label in (BASELINE_RULE, *NESTED_BANDS):
        for degree in sorted({e["degree"] for e in entries}):
            legs = sorted((e for e in entries if e["degree"] == degree),
                          key=lambda e: e["cells"])
            if len(legs) < 2:
                continue
            coarse = legs[0]["rules"][label]["retained_fraction"]
            fine = legs[-1]["rules"][label]["retained_fraction"]
            rows.setdefault(label, {})[f"p{degree}"] = {
                "coarse_cells": legs[0]["cells"], "fine_cells": legs[-1]["cells"],
                "coarse_retained": coarse, "fine_retained": fine,
                "relative_change": abs(fine - coarse) / coarse}
    return rows


def render_figure(orders, nested, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    labels = [BASELINE_RULE, *NESTED_BANDS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    degrees, designs = [], []
    for label in labels:
        per_degree = orders[label]
        degs = sorted(int(k[1:]) for k in per_degree)
        ords = [per_degree[f"p{d}"]["order"]["density"] for d in degs]
        axes[0].plot(degs, ords, "o-", label=label)
        if label == BASELINE_RULE:
            degrees, designs = degs, [d + 1 for d in degs]
    axes[0].plot(degrees, designs, "k--", lw=1, label="design p+1")
    axes[0].plot(degrees, [R3_ORDER_FRACTION * d for d in designs], "k:",
                 lw=1, label="gate 0.85(p+1)")
    axes[0].set_xlabel("polynomial degree p")
    axes[0].set_ylabel("measured relative L1 order (density)")
    axes[0].set_title("Lax R3 order vs region rule")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for label in labels:
        degs = sorted(int(k[1:]) for k in nested[label])
        change = [nested[label][f"p{d}"]["relative_change"] for d in degs]
        axes[1].plot(degs, change, "o-", label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("polynomial degree p")
    axes[1].set_ylabel("relative change of retained region\n(coarsest -> finest leg)")
    axes[1].set_title("nestedness of the measurement region")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "region_order.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(str(path))
    return figures


def render_summary(metrics):
    lines = ["# CASE-LAX-DG · R3 region-rule sensitivity", ""]
    lines.append(f"generated_at_utc: `{metrics['generated_at_utc']}`")
    lines.append("")
    lines.append("## orders (density, finest ladder pair)")
    lines.append("")
    header = "| rule | " + " | ".join(
        f"p{int(k[1:])}" for k in metrics["orders"][BASELINE_RULE]) + " | verdict |"
    lines.append(header)
    lines.append("|" + "---|" * (len(header.split("|")) - 2))
    for label, per_degree in metrics["orders"].items():
        cells = [f"{v['order']['density']:.3f}" if v["order"]["density"] is not None
                 else "n/a" for v in per_degree.values()]
        lines.append(f"| {label} | " + " | ".join(cells)
                     + f" | {metrics['verdicts'][label]['status']} |")
    lines.append("")
    lines.append("## nestedness of the region (retained fraction, coarsest -> finest)")
    lines.append("")
    lines.append("| rule | " + " | ".join(metrics["nestedness"][BASELINE_RULE]) + " |")
    lines.append("|" + "---|" * (len(metrics["nestedness"][BASELINE_RULE]) + 1))
    for label, rows in metrics["nestedness"].items():
        cells = [f"{v['coarse_retained']:.3f} -> {v['fine_retained']:.3f} "
                 f"({v['relative_change'] * 100:.1f} %)" for v in rows.values()]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args(argv)
    ladder = QUICK_LADDER if args.quick else LADDER
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    entries = run_ladder(ladder)
    orders = rule_orders(entries)
    verdicts = region_verdicts(orders)
    nested = nestedness(entries)
    figures = render_figure(orders, nested, out / "figures")
    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "purpose": "R3 region-rule sensitivity: is the published analytic-region "
                   "order failure a method property or a measurement artifact?",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"cfl": CFL, "ladder": ladder, "quick": args.quick,
                   "baseline_band_cells": R3_BAND_CELLS,
                   "nested_bands": NESTED_BANDS,
                   "order_fraction": R3_ORDER_FRACTION},
        "ladder": entries, "orders": orders, "verdicts": verdicts,
        "nestedness": nested, "figures": figures,
        "wall_seconds": float(time.time() - started),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "summary.md").write_text(render_summary(metrics))
    print("[band] verdicts: " + ", ".join(f"{k}={v['status']}"
                                          for k, v in verdicts.items()))
    return metrics


if __name__ == "__main__":
    main()