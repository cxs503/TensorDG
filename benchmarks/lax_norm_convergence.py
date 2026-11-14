"""CASE-LAX-DG -- is the A0 relative-L2 acceptance number converged in *its own*
post-processing quadrature?

Motivation (measurement debt on a published acceptance verdict)
--------------------------------------------------------------
``benchmarks/lax_case_pool.gate_a0`` judges the ledger anchor ``CASE-LAX-DG``
(``ref: exact solution | tolerance < 3 %``) on the relative L2 error at the
production leg ``p3.N400``.  That number is produced by ``_weighted_l2`` on
``quadrature_order = 10`` Gauss nodes per element -- a **single-panel** rule
applied to an integrand ``(u_dg - u_exact)^2`` that is *discontinuous inside an
element* wherever a wave (rarefaction head/tail, contact, shock) crosses it.

A single-panel Gauss rule is not exact for a function with an interior jump,
and its error does **not** decrease monotonically in the number of nodes: it
oscillates with the node placement relative to the jump.  So the published A0
margin (``3.11556e-02`` vs ``3.0e-02``, a 3.9 % miss) carries an unquantified
systematic component, and a reader cannot tell whether the FAIL is a property
of the DG solution or of the norm evaluation.

What this module measures
-------------------------
``breakpoint_resolved``
    Each element interval is split at any analytic wave position that falls
    inside it; every resulting sub-interval is smooth, so a Gauss rule on it
    converges spectrally.  This is the *reference* evaluation of the same
    discrete solution against the same exact solution -- no solver re-run, the
    DG element polynomial is recovered from the committed ``profiles.npz``
    samples (the modal field is degree ``p``, the stored rule has 10 nodes, so
    the interpolant is exact to round-off; asserted in the tests).
``single_panel_sweep``
    The same norms computed with the shipped single-panel rule at many orders,
    which is how the bias/oscillation of the published choice is exposed.
``error_share``
    How much of the L2 numerator lives in the elements the waves actually
    cross -- i.e. how much of the acceptance verdict is decided by a handful of
    cells whose integrand the shipped rule resolves worst.

Nothing here re-runs the solver, so the audit is cheap and re-runnable offline
from the committed artifact.  Numbers are written to
``docs/benchmarks/results/lax_norm_convergence/metrics.json`` and the tables in
``docs/CASE-LAX-DG-norm-quadrature-budget.md`` are rendered from that file.

Scope (honest)
--------------
``profiles.npz`` exists only for the finest ladder leg, so the audit covers the
**ledger arm** ``production_l2`` (which is judged on exactly that leg).  The
``ladder_l1`` arm's worst value sits on a *coarse* leg (``p1.N100``) whose
element fields were never committed; that leg is reported ``not_run`` here
rather than assumed.

Usage::

    python -m benchmarks.lax_norm_convergence
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from benchmarks.lax_case_pool import (exact_reference, exact_waves,
                                      production_errors_from_profiles,
                                      gate_a0)

PROFILES = Path("docs/benchmarks/results/lax_case_pool/profiles.npz")
METRICS = Path("docs/benchmarks/results/lax_case_pool/metrics.json")
OUT = Path("docs/benchmarks/results/lax_norm_convergence")
FIELDS = ("density", "velocity", "pressure")
SWEEP_ORDERS = (4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96)
REFERENCE_ORDER = 16        # sub-interval rule of the reference evaluation
CONVERGENCE_ORDERS = (12, 16, 24, 32)
FIELD_LABEL = {"density": "rho", "velocity": "u", "pressure": "p"}


# --------------------------------------------------------------------------- #
# profile loading / element-polynomial recovery
# --------------------------------------------------------------------------- #
def load_profiles(path=PROFILES):
    """Committed finest-leg fields + the element geometry they live on."""
    data = np.load(path)
    points = np.asarray(data["points"], dtype=float)      # (ncells, n_nodes)
    centres = 0.5 * (points[:, 0] + points[:, -1])
    dx = float(np.mean(np.diff(centres)))
    xi = 2.0 * (points - centres[:, None]) / dx           # reference variable
    return {
        "points": points,
        "centres": centres,
        "dx": dx,
        "xi": xi,
        "degree": int(math.ceil(points.shape[1] - 0.5)) - 1,
        "fields": {f: {"numerical": np.asarray(data[f"numerical_{f}"], dtype=float),
                       "exact": np.asarray(data[f"exact_{f}"], dtype=float)}
                   for f in FIELDS},
    }


def cell_polynomials(prof, field):
    """Degree-(n_nodes-1) interpolant of the committed element samples.

    The DG field is degree ``p`` and the stored rule has ``n_nodes`` nodes with
    ``n_nodes > p``, so this reproduces the element polynomial itself (the
    residual is asserted at round-off level by the tests), which is what lets
    the same discrete solution be re-integrated on a different rule.
    """
    xi, values = prof["xi"], prof["fields"][field]["numerical"]
    return [np.polyfit(xi[c], values[c], xi.shape[1] - 1)
            for c in range(xi.shape[0])]


def evaluate_cells(coeffs, xi_query):
    out = np.empty_like(xi_query)
    for c, coef in enumerate(coeffs):
        out[c] = np.polyval(coef, xi_query[c])
    return out


def _physical_to_xi(prof, x):
    return 2.0 * (x - prof["centres"][:, None]) / prof["dx"]


# --------------------------------------------------------------------------- #
# the two evaluations of the same norms
# --------------------------------------------------------------------------- #
def _grid(prof, nodes, weights, breaks):
    """Per-cell abscissae/weights, splitting at any wave inside the element."""
    xs, ws = [], []
    half_width = 0.5 * prof["dx"]
    for c in range(prof["points"].shape[0]):
        # true element extent: the committed samples are Gauss nodes *inside*
        # the element, so the sample extremes are NOT the element edges.
        lo = prof["centres"][c] - half_width
        hi = prof["centres"][c] + half_width
        cuts = [lo] + [b for b in breaks if lo < b < hi] + [hi]
        cx, cw = [], []
        for a, b in zip(cuts[:-1], cuts[1:]):
            mid, sub_half = 0.5 * (a + b), 0.5 * (b - a)
            cx.append(mid + sub_half * nodes)
            cw.append(sub_half * weights)
        xs.append(np.concatenate(cx))
        ws.append(np.concatenate(cw))
    width = max(x.shape[0] for x in xs)
    grid_x = np.zeros((len(xs), width))
    grid_w = np.zeros((len(xs), width))
    for c, (x, w) in enumerate(zip(xs, ws)):
        grid_x[c, :x.shape[0]] = x
        grid_w[c, :w.shape[0]] = w
    return grid_x, grid_w


def _norms(prof, coeffs, x, w, exact):
    numerical = evaluate_cells(coeffs, _physical_to_xi(prof, x))
    reference = exact
    num_sq = float(np.sum(w * (numerical - reference) ** 2))
    den_sq = float(np.sum(w * reference ** 2))
    l1 = float(np.sum(w * np.abs(numerical - reference))
               / np.sum(w * np.abs(reference)))
    return {"l2": math.sqrt(num_sq / den_sq), "l1": l1,
            "l2_numerator": num_sq, "l2_denominator": den_sq}


def breakpoint_resolved_norms(prof, order=REFERENCE_ORDER, breaks=None):
    """Norms with every element split at the analytic wave positions."""
    if breaks is None:
        breaks = sorted(exact_waves()["positions"].values())
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, w = _grid(prof, nodes, weights, breaks)
    exact = exact_reference(x)
    out = {"l1": {}, "l2": {}}
    for field in FIELDS:
        out[field] = _norms(prof, cell_polynomials(prof, field), x, w,
                            exact[field])
    out["l1"] = {f: out[f]["l1"] for f in FIELDS}
    out["l2"] = {f: out[f]["l2"] for f in FIELDS}
    out["order"] = order
    out["breakpoints"] = [float(b) for b in breaks]
    return out


def single_panel_norms(prof, order=10):
    """Norms with the shipped choice: one Gauss panel per element."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, w = _grid(prof, nodes, weights, [])
    exact = exact_reference(x)
    out = {"l1": {}, "l2": {}}
    for field in FIELDS:
        out[field] = _norms(prof, cell_polynomials(prof, field), x, w,
                            exact[field])
    out["l1"] = {f: out[f]["l1"] for f in FIELDS}
    out["l2"] = {f: out[f]["l2"] for f in FIELDS}
    out["order"] = order
    return out


def convergence_study(prof, orders=CONVERGENCE_ORDERS):
    """Does the reference evaluation itself converge?  (sub-interval rule)"""
    values = {f: [] for f in FIELDS}
    for order in orders:
        norms = breakpoint_resolved_norms(prof, order=order)
        for field in FIELDS:
            values[field].append({"order": int(order),
                                  "l2": norms["l2"][field],
                                  "l1": norms["l1"][field]})
    worst = 0.0
    for field in FIELDS:
        l2s = [row["l2"] for row in values[field]]
        spread = (max(l2s) - min(l2s)) / l2s[-1]
        worst = max(worst, spread)
    return {"orders": [int(o) for o in orders], "per_field": values,
            "worst_l2_spread": float(worst),
            "converged_at_1e_6": bool(worst < 1e-6)}


def error_share(prof, order=10, band_cells=2.0):
    """Fraction of the L2 numerator carried by elements a wave crosses."""
    breaks = np.asarray(sorted(exact_waves()["positions"].values()))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, w = _grid(prof, nodes, weights, [])
    exact = exact_reference(x)
    distance = np.min(np.abs(prof["centres"][:, None] - breaks[None, :]), axis=1)
    near = distance < band_cells * prof["dx"]
    out = {"band_cells": band_cells, "cells": int(near.sum()),
           "cells_total": int(prof["centres"].shape[0]), "per_field": {}}
    for field in FIELDS:
        coeffs = cell_polynomials(prof, field)
        numerical = evaluate_cells(coeffs, _physical_to_xi(prof, x))
        sq = (w * (numerical - exact[field]) ** 2)
        share = float(np.sum(sq[near]) / np.sum(sq))
        den = (w * exact[field] ** 2)
        out["per_field"][field] = {
            "numerator_share": share,
            "denominator_share": float(np.sum(den[near]) / np.sum(den))}
    return out


# --------------------------------------------------------------------------- #
# verdict re-adjudication
# --------------------------------------------------------------------------- #
def re_adjudicate(prof):
    """Re-judge the *ledger* arm with the corrected norm, via ``gate_a0``.

    The committed ladder entries (``metrics.json``) are handed to ``gate_a0``
    unchanged so the ladder table keeps its published values; only the
    ``production_l2`` arm is overwritten with the corrected measurement --
    exactly the input ``gate_a0`` offers for the profiles path.
    """
    committed = production_errors_from_profiles(PROFILES, quadrature_order=10)
    reference = breakpoint_resolved_norms(prof, order=REFERENCE_ORDER)
    stored = json.loads(METRICS.read_text())
    entries = [dict(leg) for leg in stored["ladder"]]
    as_committed = gate_a0(entries, production_l2=dict(committed["l2"]))
    corrected_entry = dict(entries[-1]) if entries else {}
    corrected = gate_a0(entries, production_l2=dict(reference["l2"]))
    tolerance = as_committed["tolerance"]
    worst_committed = as_committed["arms"]["production_l2"]["worst"]
    worst_corrected = corrected["arms"]["production_l2"]["worst"]
    return {
        "tolerance": float(tolerance),
        "production_leg": as_committed["production_leg"],
        "committed": {"l1": dict(committed["l1"]), "l2": dict(committed["l2"]),
                      "worst": float(worst_committed),
                      "status": as_committed["arms"]["production_l2"]["status"],
                      "miss_factor": float(worst_committed / tolerance),
                      "excess_over_bar": float(worst_committed / tolerance - 1.0)},
        "corrected": {"l1": dict(reference["l1"]), "l2": dict(reference["l2"]),
                      "worst": float(worst_corrected),
                      "status": corrected["arms"]["production_l2"]["status"],
                      "miss_factor": float(worst_corrected / tolerance),
                      "excess_over_bar": float(worst_corrected / tolerance - 1.0)},
        "verdict_flips": bool(as_committed["arms"]["production_l2"]["status"]
                              != corrected["arms"]["production_l2"]["status"]),
        "ladder_l1_arm": {
            "worst": float(as_committed["arms"]["ladder_l1"]["worst"]),
            "status": as_committed["arms"]["ladder_l1"]["status"],
            "re_measured": False,
            "reason": ("profiles.npz holds only the finest leg; the ladder arm's "
                       "worst leg (p1.N100) has no committed element fields, so "
                       "its quadrature bias is not quantified here — reported "
                       "not_run, not assumed")},
        "corrected_entry_marker": sorted(corrected_entry)[:3],
    }


def bias_table(prof, orders=SWEEP_ORDERS):
    reference = breakpoint_resolved_norms(prof, order=REFERENCE_ORDER)
    rows = []
    for order in orders:
        norms = single_panel_norms(prof, order=order)
        row = {"order": int(order)}
        for field in FIELDS:
            row[field] = {
                "l2": norms["l2"][field],
                "l1": norms["l1"][field],
                "l2_bias": float(norms["l2"][field] / reference["l2"][field] - 1.0),
                "l1_bias": float(norms["l1"][field] / reference["l1"][field] - 1.0),
            }
        rows.append(row)
    return rows


def oscillation_metrics(rows):
    """Monotonicity + envelope of the single-panel sweep (the actual defect)."""
    out = {}
    for field in FIELDS:
        biases = np.array([row[field]["l2_bias"] for row in rows])
        diffs = np.diff(biases)
        out[field] = {
            "max_abs_bias": float(np.max(np.abs(biases))),
            "sign_changes": int(np.sum(np.sign(diffs[:-1]) * np.sign(diffs[1:]) < 0)),
            "committed_10pt_bias": float(
                next(r[field]["l2_bias"] for r in rows if r["order"] == 10)),
            "monotone_under_refinement": bool(np.all(diffs <= 0)
                                              or np.all(diffs >= 0)),
        }
    return out


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def _figure(bias_rows, prof, share, reference, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    colours = {"density": "#1f77b4", "velocity": "#d62728", "pressure": "#2ca02c"}

    ax = axes[0]
    for field in FIELDS:
        qs = [row["order"] for row in bias_rows]
        vals = [row[field]["l2"] for row in bias_rows]
        ax.semilogx(qs, vals, "o-", ms=4, lw=1.2, color=colours[field],
                    label=f"{FIELD_LABEL[field]}: single panel")
        ax.axhline(reference["l2"][field], color=colours[field], ls="--", lw=1.0,
                   alpha=0.75)
    ax.axvline(10, color="k", ls=":", lw=1.0)
    ax.text(10.3, ax.get_ylim()[0], " shipped q=10", fontsize=8, va="bottom")
    ax.axhline(0.03, color="0.4", ls="-.", lw=0.9)
    ax.text(bias_rows[0]["order"], 0.0302, " ledger bar 3 %", fontsize=8, color="0.4")
    ax.set_xlabel("single-panel Gauss order per element")
    ax.set_ylabel("relative L2 vs exact solution")
    ax.set_title("A0 norm vs the evaluation rule\n(dashed = breakpoint-resolved)")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = [FIELD_LABEL[f] for f in FIELDS]
    at10 = [next(r[f]["l2_bias"] for r in bias_rows if r["order"] == 10) * 100
            for f in FIELDS]
    worst = [max(abs(r[f]["l2_bias"]) for r in bias_rows) * 100 for f in FIELDS]
    pos = np.arange(len(labels))
    ax.bar(pos - 0.2, at10, 0.4, label="at shipped q=10")
    ax.bar(pos + 0.2, worst, 0.4, label="worst over q=4..96")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(pos, labels)
    ax.set_ylabel("relative L2 bias vs reference  [%]")
    ax.set_title("Quadrature bias of the published norm")
    ax.legend(fontsize=8)
    for i, v in enumerate(at10):
        ax.text(i - 0.2, v, f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)

    ax = axes[2]
    breaks = np.asarray(reference["breakpoints"])
    coeffs = cell_polynomials(prof, "velocity")
    nodes, weights = np.polynomial.legendre.leggauss(10)
    x, w = _grid(prof, nodes, weights, [])
    exact = exact_reference(x)
    numerical = evaluate_cells(coeffs, _physical_to_xi(prof, x))
    sq = (w * (numerical - exact["velocity"]) ** 2)
    per_cell = sq.sum(axis=1) / sq.sum()
    ax.bar(prof["centres"], per_cell / per_cell.max(), width=prof["dx"] * 0.9,
           color="#4c72b0")
    for b in breaks:
        ax.axvline(b, color="#d62728", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("cell share of L2 error (normalised)")
    ax.set_title(f"Error lives in {share['cells']} wave cells "
                 f"({100 * share['per_field']['velocity']['numerator_share']:.0f}% "
                 "of numerator)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _summary_markdown(metrics):
    lens = metrics["re_adjudication"]
    c, r = lens["committed"], lens["corrected"]
    lines = [
        "# renders → docs/CASE-LAX-DG-norm-quadrature-budget.md",
        "",
        f"| field | committed q=10 L2 | corrected L2 | bias |",
        "|---|---|---|---|",
    ]
    for field in FIELDS:
        lines.append(
            f"| {FIELD_LABEL[field]} | {c['l2'][field]:.6e} | "
            f"{r['l2'][field]:.6e} | "
            f"{(c['l2'][field] / r['l2'][field] - 1) * 100:+.2f} % |")
    lines += ["", "| arm | worst | tolerance | verdict |",
              "|---|---|---|---|"]
    lines.append(f"| production_l2 (committed) | {c['worst']:.6e} | "
                 f"{lens['tolerance']:.2e} | {c['status']} |")
    lines.append(f"| production_l2 (corrected) | {r['worst']:.6e} | "
                 f"{lens['tolerance']:.2e} | {r['status']} |")
    lines.append("")
    lines.append(f"verdict flips: **{lens['verdict_flips']}**")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prof = load_profiles()
    reference = breakpoint_resolved_norms(prof, order=REFERENCE_ORDER)
    study = convergence_study(prof)
    rows = bias_table(prof)
    share = error_share(prof)
    lens = re_adjudicate(prof)
    oscillation = oscillation_metrics(rows)

    interp_residual = {}
    for field in FIELDS:
        coeffs = cell_polynomials(prof, field)
        xi = prof["xi"]
        interp_residual[field] = float(max(
            np.max(np.abs(np.polyval(coef, xi[c]) - prof["fields"][field]["numerical"][c]))
            for c, coef in enumerate(coeffs)))

    metrics = {
        "anchor": "CASE-LAX-DG",
        "question": ("is the A0 relative-L2 acceptance number converged with "
                     "respect to its own output quadrature?"),
        "profiles": str(PROFILES),
        "reference_order": REFERENCE_ORDER,
        "sweep_orders": list(SWEEP_ORDERS),
        "degree_recovered": prof["degree"],
        "interpolation_residual": interp_residual,
        "reference_norms": {"l1": reference["l1"], "l2": reference["l2"],
                            "breakpoints": reference["breakpoints"]},
        "reference_convergence": study,
        "single_panel_sweep": rows,
        "oscillation": oscillation,
        "error_share": share,
        "re_adjudication": lens,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (OUT / "summary.md").write_text(_summary_markdown(metrics) + "\n")
    _figure(rows, prof, share, reference, OUT / "figures" / "norm_quadrature.png")
    print(json.dumps({"verdict_flips": lens["verdict_flips"],
                      "committed_worst": lens["committed"]["worst"],
                      "corrected_worst": lens["corrected"]["worst"],
                      "converged": study["converged_at_1e_6"],
                      "oscillation": oscillation}, indent=2))
    return metrics


if __name__ == "__main__":
    main()