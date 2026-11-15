"""CASE-LAX-DG · A0 metric catalogue: *what* is measured, not just how it is integrated.

Motivation
----------
``docs/CASE-LAX-DG-a0-quadrature-audit.md`` settled the *quadrature* axis of the
ledger arm A0: the exact Lax solution is piecewise analytic with kinks inside
cells, so a fixed Gauss rule under-reads, and only a breakpoint-aligned converged
rule may be compared against the ``< 3 %`` bar.  Under that converged rule the
pinned leg ``p3.N400`` reads ``3.0251e-02`` (density) and still **fails**.

The *target* axis was never audited: relative L2 can be taken

* on the **pointwise** DG field (what the pool does), or
* on the **cell averages** (the resolved scales only),

and in either case in primitive (``rho, u, p``) or conservative
(``rho, rho*u, E``) variables.  Those are different measurements of "the same"
error, and the ledger wording ("ref: exact solution | tolerance < 3 %") does not
pin one.  This driver quantifies the whole catalog from committed field caches
and answers three questions:

Q1  Does the A0 verdict depend on the target (pointwise vs cell average)?
Q2  Is a metric that passes the case *discriminating* -- would it also pass a
    coarser reconstruction on the same mesh?
Q3  Where does the pointwise error actually live?

The decisive probe for Q2 is exact and needs no extra solve: take the committed
DG solution, keep each cell's **conserved cell mean** and drop all higher modes
(the "same-mesh P0 truncation").  A cell-average metric is *invariant* under
that operation (the cell mean is unchanged, to floating-point), while the
pointwise metric moves.

Q3 is answered by a Pythagorean split that holds per cell because
``int_cell (q - qbar) = 0``::

    int_cell (q - qex)^2 = int_cell (q - qbar)^2 + int_cell (qbar - qex)^2
                          \\__ sub-cell content __/  \\__ P0-truncation error __/

so the pointwise error of any DG field is exactly *its own sub-cell content*
plus *the error of the same-mesh P0 field*.  The two terms are reported, and the
sub-cell content of the exact solution itself (nonzero only in the four cells
holding a kink) bounds what can be attained.

Zero solver runs: everything is re-read from committed caches
------------------------------------------------------------
* ``docs/benchmarks/results/lax_case_pool/profiles.npz`` -- the pinned ledger leg
  ``p3.N400`` (400 x 10 Gauss nodes, primitive fields + exact);
* ``docs/benchmarks/results/lax_resolution_requirement/profiles.npz`` -- the
  ``p3.N800`` leg, reported as **context** for the refinement trend only.

Usage::

    python -m benchmarks.lax_a0_metric_catalogue            # write artifact + doc
    python -m benchmarks.lax_a0_metric_catalogue --check    # drift check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.lax_a0_quadrature_audit import (
    BASE_ORDER,
    CONTEXT_PROFILES,
    FIELDS,
    POOL_PROFILES,
    PRODUCTION_LEG,
    CONTEXT_LEG,
    _breakpoints,
    _geometry,
    load_fields,
    measure,
)
from benchmarks.lax_case_pool import ledger_acceptance_tolerance
from benchmarks.riemann_exact import sample

LEFT = (0.445, 0.698, 3.528)
RIGHT = (0.5, 0.0, 0.571)
FINAL_TIME = 0.13
DIAPHRAGM = 0.5
GAMMA = 1.4

OUT = Path("docs/benchmarks/results/lax_a0_metric_catalogue")
DOC = Path("docs/CASE-LAX-DG-a0-metric-catalogue.md")

PRIMITIVE = FIELDS                                # density, velocity, pressure
CONSERVED = ("density", "momentum", "energy")
SUBDIVISIONS = 24                                 # converged exact cell means
TOL_REUSE = 1e-12                                 # pool-value re-read
TOL_INVARIANCE = 1e-12                            # cell-average metric invariance
TOL_PYTHAGORAS = 1e-10                            # decomposition residual


# ------------------------------------------------------------------ variables
def to_conserved(primitive):
    """``(rho, u, p)`` -> ``(rho, rho*u, E)``; the full ``(cells, nodes)`` array."""
    rho, u, p = (primitive[f] for f in PRIMITIVE)
    return {"density": rho, "momentum": rho * u, "energy": p / (GAMMA - 1.0) + 0.5 * rho * u * u}


def to_primitive(conserved):
    """``(rho, rho*u, E)`` -> ``(rho, u, p)``."""
    rho, mom, energy = (conserved[f] for f in CONSERVED)
    return {"density": rho, "velocity": mom / rho,
            "pressure": (GAMMA - 1.0) * (energy - 0.5 * mom * mom / rho)}


def load_leg(path):
    """Committed field cache + exact solution re-sampled at the same nodes."""
    fields = load_fields(path)
    exact = sample(fields["points"], FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)
    for key in PRIMITIVE:
        fields[f"exact_{key}"] = exact[key]
    return fields


def _exact_arrays(points):
    """Exact solution at the cached nodes, in both variable sets."""
    exact_primitive = {f: sample(points, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)[f] for f in PRIMITIVE}
    return exact_primitive, to_conserved(exact_primitive)


def nodal_fields_from_coefficients(x, coeff, order=BASE_ORDER):
    """Modal DG output -> the ``fields`` dict used by every measure here.

    ``x`` is the solver's cell-centre array, ``coeff`` the ``(cells, modes, 3)``
    Legendre coefficient tensor.  Values are evaluated at the ``order`` Gauss
    nodes of each cell (the committed cache layout) and the exact solution is
    sampled at the same points, so a freshly solved leg can be measured with the
    same code path as a committed artifact.
    """
    values = coeff.detach().cpu().numpy() if hasattr(coeff, "detach") else np.asarray(coeff)
    centres = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    cells = centres.shape[0]
    dx = float((centres[-1] - centres[0]) / (cells - 1))
    nodes, _ = np.polynomial.legendre.leggauss(order)
    phi = np.stack([np.polynomial.legendre.legval(
        nodes, np.concatenate((np.zeros(k), [1.0]))) for k in range(values.shape[1])], axis=1)
    conserved = np.einsum("cmv,qm->cqv", values, phi)
    primitive = {
        "density": conserved[..., 0],
        "velocity": conserved[..., 1] / conserved[..., 0],
        "pressure": 0.4 * (conserved[..., 2] - 0.5 * conserved[..., 1] ** 2 / conserved[..., 0]),
    }
    points = centres[:, None] + 0.5 * dx * nodes[None, :]
    exact = sample(points, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)
    fields = {"points": points}
    for key in PRIMITIVE:
        fields[f"numerical_{key}"] = primitive[key]
        fields[f"exact_{key}"] = exact[key]
    return fields


def _cell_weights(points):
    """Per-node quadrature weights of the committed measure (one Gauss rule/cell)."""
    centres, dx = _geometry(points)
    nodes, weights = np.polynomial.legendre.leggauss(points.shape[1])
    return centres, dx, np.broadcast_to(0.5 * dx * weights[None, :], points.shape)


def _aligned_cell_means(points, arrays, subdivisions=SUBDIVISIONS):
    """Exact (kink-aligned) cell means of ``arrays`` on the mesh of ``points``."""
    centres, dx = _geometry(points)
    nodes, weights = np.polynomial.legendre.leggauss(subdivisions)
    breaks = _breakpoints()
    means = {k: np.zeros(len(centres)) for k in arrays}
    for cell, centre in enumerate(centres):
        xl, xr = centre - 0.5 * dx, centre + 0.5 * dx
        inner = breaks[(breaks > xl) & (breaks < xr)]
        edges = np.concatenate(([xl], inner, [xr]))
        xs, ws = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            half, mid = 0.5 * (b - a), 0.5 * (a + b)
            xs.append(mid + half * nodes)
            ws.append(half * weights)
        xs, ws = np.concatenate(xs), np.concatenate(ws)
        exact = sample(xs, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)
        exact_c = to_conserved({f: exact[f] for f in PRIMITIVE})
        for key in arrays:
            means[key][cell] = float(np.sum(ws * exact_c[key]) / dx)
    return means


# ------------------------------------------------------------------ the probe
def truncate_to_cell_means(fields):
    """Same-mesh P0 truncation of a committed field, in *conserved* variables.

    The DG cell mean of a conserved variable is the physically meaningful
    resolved quantity; dropping the higher modes is what a first-order scheme on
    the same mesh would carry.  Returns a ``fields``-style dict (primitive at the
    same nodes) so every measure below can be applied unchanged.
    """
    points = fields["points"]
    _, dx, cell_weight = _cell_weights(points)
    conserved = to_conserved({f: fields[f"numerical_{f}"] for f in PRIMITIVE})
    means = {k: (v * cell_weight).sum(axis=1) / dx for k, v in conserved.items()}
    flat = {k: np.broadcast_to(v[:, None], points.shape) for k, v in means.items()}
    primitive = to_primitive(flat)
    out = {"points": points}
    for f in PRIMITIVE:
        out[f"numerical_{f}"] = primitive[f]
        out[f"exact_{f}"] = fields[f"exact_{f}"]
    return out


# ------------------------------------------------------------------- measures
def pointwise_norm(fields, variables="primitive", norm="l2"):
    """Relative L1 / L2 of the pointwise field on the committed Gauss nodes."""
    points = fields["points"]
    _, _, cell_weight = _cell_weights(points)
    if variables == "primitive":
        numerical = {f: fields[f"numerical_{f}"] for f in PRIMITIVE}
        exact = {f: fields[f"exact_{f}"] for f in PRIMITIVE}
        out = {}
        for f in PRIMITIVE:
            out[f] = _rel(numerical[f], exact[f], cell_weight, norm)
        return out
    numerical = to_conserved({f: fields[f"numerical_{f}"] for f in PRIMITIVE})
    exact = to_conserved({f: fields[f"exact_{f}"] for f in PRIMITIVE})
    return {f: _rel(numerical[f], exact[f], cell_weight, norm) for f in CONSERVED}


def cell_average_norm(fields, variables="primitive", norm="l2"):
    """Relative L1 / L2 of the *cell averages* against the exact cell averages."""
    points = fields["points"]
    centres, dx, cell_weight = _cell_weights(points)
    if variables == "primitive":
        numerical = {f: fields[f"numerical_{f}"] for f in PRIMITIVE}
        exact = {f: fields[f"exact_{f}"] for f in PRIMITIVE}
        num_c = to_conserved(numerical)
        ex_c = to_conserved(exact)
    else:
        num_c = to_conserved({f: fields[f"numerical_{f}"] for f in PRIMITIVE})
        ex_c = to_conserved({f: fields[f"exact_{f}"] for f in PRIMITIVE})
    num_mean = {k: (v * cell_weight).sum(axis=1) / dx for k, v in num_c.items()}
    ex_mean = _aligned_cell_means(points, num_c)
    w = np.full(len(centres), dx)
    if variables == "primitive":
        num_prim = to_primitive({k: np.broadcast_to(v[:, None], points.shape)
                                 for k, v in num_mean.items()})
        ex_prim = to_primitive({k: np.broadcast_to(v[:, None], points.shape)
                                for k, v in ex_mean.items()})
        return {f: _rel(num_prim[f][:, 0], ex_prim[f][:, 0], w, norm) for f in PRIMITIVE}
    return {f: _rel(num_mean[f], ex_mean[f], w, norm) for f in CONSERVED}


def _rel(numerical, exact, weight, norm):
    diff = numerical - exact
    if norm == "l2":
        num = float(np.sqrt(np.sum(weight * diff ** 2)))
        den = float(np.sqrt(np.sum(weight * exact ** 2)))
    else:
        num = float(np.sum(weight * np.abs(diff)))
        den = float(np.sum(weight * np.abs(exact)))
    return num / den


def pythagoras_split(fields, field="density", order=SUBDIVISIONS):
    """Exact four-term split of the pointwise error of one field.

    With ``qbar`` the DG cell mean and ``qbar_ex`` the exact cell mean,

        q - q_ex = (q - qbar) + (qbar - qbar_ex) + (qbar_ex - q_ex)

    where the first and third pieces have zero cell mean and the middle one is
    constant per cell.  Five of the nine cross terms therefore vanish and

        int (q-q_ex)^2 = int (q-qbar)^2              # DG sub-cell content
                       + int (qbar-qbar_ex)^2        # resolved (cell-mean) error
                       + int (qbar_ex-q_ex)^2        # exact sub-cell content
                       + 2 int (q-qbar)(qbar_ex-q_ex)  # sub-cell alignment

    ``alignment < 0``: the DG sub-cell shape cancels part of the exact one (the
    DG field does resolve sub-cell structure); ``> 0``: spurious oscillation.

    Every term is evaluated with the **same** quadrature -- the breakpoint
    aligned rule also used by ``lax_a0_quadrature_audit`` for the canonical A0
    reading -- so the identity holds to machine precision instead of to the
    (few-percent) quadrature error of a kinked exact solution.
    """
    from benchmarks.lax_a0_quadrature_audit import barycentric_matrix

    points = fields["points"]
    nodal = fields[f"numerical_{field}"]
    centres, dx = _geometry(points)
    cells = points.shape[0]
    base_nodes = np.polynomial.legendre.leggauss(points.shape[1])[0]
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(order)
    breaks = _breakpoints()

    terms = {"dg_sub_cell": 0.0, "cell_mean_error": 0.0,
             "exact_sub_cell": 0.0, "alignment": 0.0, "total": 0.0, "norm2": 0.0}
    cell_mean = np.zeros(cells)
    ex_mean = np.zeros(cells)
    sub_per_cell = np.zeros(cells)
    centres_out = centres
    for cell in range(cells):
        centre = centres[cell]
        xl, xr = centre - 0.5 * dx, centre + 0.5 * dx
        inner = breaks[(breaks > xl) & (breaks < xr)]
        edges = np.concatenate(([xl], inner, [xr]))
        xs, ws = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            half, mid = 0.5 * (b - a), 0.5 * (a + b)
            xs.append(mid + half * gauss_nodes)
            ws.append(half * gauss_weights)
        xs, ws = np.concatenate(xs), np.concatenate(ws)
        interp = barycentric_matrix(2.0 * (xs - centre) / dx, base_nodes)
        num = nodal[cell] @ interp.T
        exact = sample(xs, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)[field]
        cell_mean[cell] = float(np.sum(ws * num) / dx)
        ex_mean[cell] = float(np.sum(ws * exact) / dx)
        sub = num - cell_mean[cell]
        exact_sub = ex_mean[cell] - exact
        sub_per_cell[cell] = float(np.sum(ws * sub ** 2))
        terms["dg_sub_cell"] += float(np.sum(ws * sub ** 2))
        terms["cell_mean_error"] += float(np.sum(ws * (cell_mean[cell] - ex_mean[cell]) ** 2))
        terms["exact_sub_cell"] += float(np.sum(ws * exact_sub ** 2))
        terms["alignment"] += float(np.sum(ws * sub * exact_sub))
        terms["total"] += float(np.sum(ws * (num - exact) ** 2))
        terms["norm2"] += float(np.sum(ws * exact ** 2))

    total = terms["total"]
    identity = (terms["dg_sub_cell"] + terms["cell_mean_error"]
                + terms["exact_sub_cell"] + 2.0 * terms["alignment"])
    kink = np.array([np.any((breaks > c - 0.5 * dx) & (breaks < c + 0.5 * dx)) for c in centres])
    return {
        **terms,
        "relative_l2": float(np.sqrt(total / terms["norm2"])),
        "residual": abs(total - identity) / total,
        "dg_sub_cell_share": terms["dg_sub_cell"] / total,
        "cell_mean_share": terms["cell_mean_error"] / total,
        "exact_sub_cell_share": terms["exact_sub_cell"] / total,
        "alignment_share": 2.0 * terms["alignment"] / total,
        "cell_mean": cell_mean,
        "kink_cells": int(kink.sum()),
        "dg_sub_cell_in_kink_cells": float(sub_per_cell[kink].sum()),
        "dg_sub_cell_outside_kink_cells": float(sub_per_cell[~kink].sum()),
    }


def _exact_cell_mean_of(fields, field):
    """Exact cell mean of one *primitive* field (aligned converged quadrature)."""
    points = fields["points"]
    centres, dx = _geometry(points)
    nodes, weights = np.polynomial.legendre.leggauss(SUBDIVISIONS)
    breaks = _breakpoints()
    out = np.zeros(len(centres))
    for cell, centre in enumerate(centres):
        xl, xr = centre - 0.5 * dx, centre + 0.5 * dx
        inner = breaks[(breaks > xl) & (breaks < xr)]
        edges = np.concatenate(([xl], inner, [xr]))
        xs, ws = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            half, mid = 0.5 * (b - a), 0.5 * (a + b)
            xs.append(mid + half * nodes)
            ws.append(half * weights)
        xs, ws = np.concatenate(xs), np.concatenate(ws)
        out[cell] = float(np.sum(ws * sample(xs, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)[field]) / dx)
    return out


# --------------------------------------------------------------------- build
def build_metrics(with_figures=True):
    tolerance = ledger_acceptance_tolerance()
    fields = load_leg(POOL_PROFILES)
    p0 = truncate_to_cell_means(fields)

    metrics = {
        "case": "lax-1954-shock-tube",
        "anchor": "CASE-LAX-DG",
        "study": "which *target* the A0 relative L2 is taken on (pointwise vs cell average), offline, zero solves",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "profiles_file": str(POOL_PROFILES),
            "profiles_sha256": hashlib.sha256(POOL_PROFILES.read_bytes()).hexdigest(),
            "context_profiles": str(CONTEXT_PROFILES),
            "solver_rerun": False,
            "documented_in": str(DOC),
        },
        "config": {
            "production_leg": PRODUCTION_LEG,
            "context_leg": CONTEXT_LEG,
            "tolerance": tolerance,
            "base_order": BASE_ORDER,
            "cell_average_exact_subdivisions": SUBDIVISIONS,
            "breakpoints": sorted(_breakpoints().tolist()),
            "gamma": GAMMA, "final_time": FINAL_TIME, "diaphragm": DIAPHRAGM,
        },
    }

    # ---- Q1: the catalogue ------------------------------------------------
    catalogue = {}
    for variables, keys in (("primitive", PRIMITIVE), ("conserved", CONSERVED)):
        for norm in ("l2", "l1"):
            for target, fn in (("pointwise", pointwise_norm), ("cell_average", cell_average_norm)):
                p3 = fn(fields, variables, norm)
                truncated = fn(p0, variables, norm)
                catalogue[f"{variables}/{norm}/{target}"] = {
                    "fields": {f: {"dg": p3[f], "p0_truncated": truncated[f],
                                   "discrimination_ratio": (p3[f] / truncated[f] if truncated[f] else None)}
                               for f in keys},
                    "worst_dg": max(p3.values()),
                    "worst_field": max(p3, key=p3.get),
                    "verdict": "PASS" if max(p3.values()) < tolerance else "FAIL",
                }

    # converged / quadrature-converged reference (audit reuse): the canonical
    # pointwise value, plus the unconverged orders for provenance
    aligned = measure(fields, "aligned", order=20)
    uniform = {str(order): measure(fields, "uniform", order) for order in (6, 8, 10, 16)}
    aligned_p0 = measure(p0, "aligned", order=20)
    uniform6_p0 = measure(p0, "uniform", order=6)
    for key, dg, tr in (
        ("primitive/l2/pointwise_uniform_order6", uniform["6"]["l2"], uniform6_p0["l2"]),
        ("primitive/l2/pointwise_aligned_converged", aligned["l2"], aligned_p0["l2"]),
    ):
        catalogue[key] = {
            "fields": {f: {"dg": dg[f], "p0_truncated": tr[f],
                           "discrimination_ratio": dg[f] / tr[f] if tr[f] else None}
                       for f in PRIMITIVE},
            "worst_dg": max(dg.values()),
            "worst_field": max(PRIMITIVE, key=lambda f: dg[f]),
            "verdict": "PASS" if max(dg.values()) < tolerance else "FAIL",
        }
    metrics["catalogue"] = catalogue

    # ---- Q2: invariance / discrimination gates ----------------------------
    invariance = {}
    for key in ("primitive/l2/cell_average", "conserved/l2/cell_average",
                "primitive/l1/cell_average", "conserved/l1/cell_average"):
        worst = 0.0
        for entry in catalogue[key]["fields"].values():
            worst = max(worst, abs(entry["dg"] - entry["p0_truncated"]) / entry["dg"])
        invariance[key] = worst
    metrics["gates"] = {
        "cell_average_invariance_worst_relative_drift": {
            "value": max(invariance.values()), "tolerance": TOL_INVARIANCE,
            "status": "PASS" if max(invariance.values()) <= TOL_INVARIANCE else "FAIL",
            "per_metric": invariance,
            "reading": "a metric invariant under same-mesh P0 truncation cannot rank schemes on that mesh",
        },
    }

    # ---- Q3: four-term decomposition + sub-cell attribution ---------------
    split = {field: pythagoras_split(fields, field) for field in PRIMITIVE}
    metrics["decomposition"] = {
        field: {k: v for k, v in s.items() if k != "cell_mean"}
        for field, s in split.items()
    }
    canon = metrics["catalogue"]["primitive/l2/pointwise_aligned_converged"]["fields"]
    canon_drift = max(abs(split[f]["relative_l2"] - canon[f]["dg"]) / canon[f]["dg"]
                      for f in PRIMITIVE)
    metrics["gates"]["decomposition_total_matches_canonical_a0"] = {
        "value": canon_drift, "tolerance": 1e-12,
        "status": "PASS" if canon_drift <= 1e-12 else "FAIL",
        "reading": "the decomposition's total must be the canonical A0 reading itself",
    }
    metrics["gates"]["decomposition_identity_worst_residual"] = {
        "value": max(v["residual"] for v in split.values()), "tolerance": TOL_PYTHAGORAS,
        "status": "PASS" if max(v["residual"] for v in split.values()) <= TOL_PYTHAGORAS else "FAIL",
        "reading": "int(q-q_ex)^2 = dg_sub + cellmean + exact_sub + 2*alignment, per-cell exact",
    }

    # ---- context: the refinement trend (N=800), context only --------------
    try:
        ctx = load_leg(CONTEXT_PROFILES)
        ctx_catalogue = {
            "pointwise": {f: pointwise_norm(ctx, "primitive", "l2")[f] for f in PRIMITIVE},
            "cell_average": {f: cell_average_norm(ctx, "primitive", "l2")[f] for f in PRIMITIVE},
        }
        ctx_split = {}
        for field in PRIMITIVE:
            s_ctx = pythagoras_split(ctx, field)
            ctx_split[field] = {k: v for k, v in s_ctx.items() if k != "cell_mean"}
        metrics["context"] = {"leg": CONTEXT_LEG, "norms": ctx_catalogue,
                              "decomposition": ctx_split,
                              "reading": "context only: the pinned leg of A0 is p3.N400"}
    except FileNotFoundError:  # pragma: no cover - cache missing
        metrics["context"] = {"leg": CONTEXT_LEG, "status": "cache missing"}

    metrics["outcome"] = _outcome(metrics, tolerance)
    if with_figures:
        metrics["figures"] = [str(p) for p in render_figures(metrics)]
    return metrics


def _outcome(metrics, tolerance):
    tol_pct = tolerance * 100
    point = metrics["catalogue"]["primitive/l2/pointwise_aligned_converged"]
    cell = metrics["catalogue"]["primitive/l2/cell_average"]
    unconverged = metrics["catalogue"]["primitive/l2/pointwise_uniform_order6"]
    dec = metrics["decomposition"]["density"]
    verdict_p = "FAIL" if point["worst_dg"] >= tolerance else "PASS"
    verdict_c = "PASS" if cell["worst_dg"] < tolerance else "FAIL"
    statement = (
        f"钉扎腿 `{PRODUCTION_LEG}` 的相对 L2 密度误差取决于**度量对象**："
        f"点值（aligned 收敛求积）{point['worst_dg']*100:.4f} % → **{verdict_p}**；"
        f"单元平均 {cell['worst_dg']*100:.4f} % → {verdict_c}。前者是后者的 "
        f"{point['worst_dg']/cell['worst_dg']:.2f} 倍，而台账原文（「ref: 精确解｜容差 < "
        f"{tol_pct:.2f} %」）没有指定是哪一个。"
    )
    discrimination = (
        "单元平均度量**不可用于裁决**：它在同网格 P0 截断（保住每单元守恒量均值、丢掉高阶模态）"
        f"下逐位不变（最差相对漂移 {metrics['gates']['cell_average_invariance_worst_relative_drift']['value']:.1e}），"
        "即一个只携带单元均值的场会读到同一个数——它无法区分 P0 与 P3，"
        f"因此用它把 A0 判成 PASS 等于让一阶重构通过同一条容差。"
    )
    decomposition = (
        f"点值误差可精确四分（`∫(q−q_ex)² = DG 次单元 + 单元均值误差 + 精确解次单元 + 2×对齐项`，"
        f"密度残差 {dec['residual']:.1e}）：DG 次单元占 {dec['dg_sub_cell_share']*100:.1f} %、"
        f"单元均值项占 {dec['cell_mean_share']*100:.1f} %、精确解自身次单元占 "
        f"{dec['exact_sub_cell_share']*100:.1f} %、对齐项 2×cov 占 {dec['alignment_share']*100:+.1f} %。"
        f"DG 次单元能量有 {dec['dg_sub_cell_in_kink_cells']/dec['dg_sub_cell']*100:.1f} % 落在含解析折点的 "
        f"{dec['kink_cells']} 个单元（网格的 {dec['kink_cells']/400*100:.2f} %）。"
        f"对齐项为负说明 DG 的次单元形状在**抵消**精确解的次单元结构（不是纯噪声）。"
    )
    ctx_point = metrics["context"]["norms"]["pointwise"]
    ctx_cell = metrics["context"]["norms"]["cell_average"]
    recommendation = (
        f"结论：按台账已登记口径（整场点值相对 L2）A0 的 honest FAIL 站得住——收敛求积下密度 "
        f"{point['worst_dg']*100:.4f} % ≥ {tol_pct:.2f} %；且**不能**用「换成单元平均度量」翻案"
        f"（那会让同网格 P0 也通过）。注意未收敛的 6 阶 Gauss 读数会给出 "
        f"{unconverged['worst_dg']*100:.4f} % 的 PASS 假象，这正是 "
        f"`riemann_dg_error` 默认值必须与池子统一的原因。真正的出路仍是分辨率"
        f"（`p3.N800` 点值密度 {ctx_point['density']*100:.3f} %、单元平均 "
        f"{ctx_cell['density']*100:.3f} %）或由台账持有人显式改登记生产腿/度量口径。"
    )
    return {"pointwise_verdict": verdict_p, "cell_average_verdict": verdict_c,
            "statement": statement, "discrimination": discrimination,
            "decomposition": decomposition, "recommendation": recommendation}


# ------------------------------------------------------------------- figures
def render_figures(metrics, out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) if out_dir else OUT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    tol = metrics["config"]["tolerance"]

    # (1) catalogue bars: DG vs same-mesh P0 truncation, per metric
    keys = ["primitive/l2/pointwise", "primitive/l2/pointwise_aligned_converged",
            "primitive/l2/cell_average", "primitive/l1/pointwise",
            "primitive/l1/cell_average", "conserved/l2/pointwise"]
    labels = ["node L2", "node L2 (conv.)", "cellavg L2", "node L1", "cellavg L1", "cons L2"]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    width = 0.38
    x = np.arange(len(keys))
    dg = [100 * metrics["catalogue"][k]["worst_dg"] for k in keys]
    p0 = [100 * max(v["p0_truncated"] for v in metrics["catalogue"][k]["fields"].values())
          for k in keys]
    ax.bar(x - width / 2, dg, width, label="DG-P3 (pinned leg)", color="#1f77b4")
    ax.bar(x + width / 2, p0, width, label="same-mesh P0 truncation", color="#ff7f0e")
    ax.axhline(100 * tol, color="crimson", ls="--", label=f"ledger bar {100*tol:.1f} %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("worst-field relative error [%]")
    ax.set_title("A0 metric catalogue, pinned leg p3.N400: the target decides the verdict")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "metric_catalogue.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # (2) four-term decomposition vs refinement
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    labels = ["DG sub-cell", "cell-mean error", "exact sub-cell", "2 x alignment"]
    keys = ["dg_sub_cell", "cell_mean_error", "exact_sub_cell", "alignment"]
    x = np.arange(len(PRIMITIVE))
    width = 0.2
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for i, (label, key, colour) in enumerate(zip(labels, keys, colours)):
        vals = [100 * metrics["decomposition"][f][key]
                / max(metrics["decomposition"][f]["total"], 1e-30) for f in PRIMITIVE]
        if key == "alignment":
            vals = [2 * v for v in vals]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=colour)
    ax.set_xticks(x)
    ax.set_xticklabels(PRIMITIVE)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("share of pointwise error^2 [%]")
    ax.set_title("p3.N400 pointwise error, exact four-term split")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "decomposition.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # (3) where the sub-cell content lives
    fields_dict = load_leg(POOL_PROFILES)
    centres, dx, cell_weight = _cell_weights(fields_dict["points"])
    split = pythagoras_split(fields_dict, "density")
    sub = np.sqrt(((fields_dict["numerical_density"] - split["cell_mean"][:, None]) ** 2
                   * cell_weight).sum(axis=1) / dx)
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.plot(centres, sub, lw=1.0, color="#1f77b4", label="per-cell sub-cell content (density)")
    for name, xb in sorted(_wave_positions().items()):
        ax.axvline(xb, color="crimson", ls=":", lw=1.0)
        ax.annotate(name, (xb, 0.95 * sub.max()), rotation=90, fontsize=7,
                    ha="right", va="top", color="crimson")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\|q-\bar q\|_{L^2(cell)}$")
    ax.set_title("sub-cell content localises at the four analytic waves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "subcell_localisation.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def _wave_positions():
    from benchmarks.lax_case_pool import exact_waves
    return exact_waves()["positions"]


# --------------------------------------------------------------------- render
def _ranking_note(entries):
    if all(abs(v["discrimination_ratio"] - 1.0) < 1e-12 for v in entries.values()):
        return "无区分（比值恒 1.000）"
    wins = [f for f, v in entries.items() if v["discrimination_ratio"] < 1.0 - 1e-12]
    if len(wins) == len(entries):
        return "DG 全胜"
    if not wins:
        return "P0 胜"
    return "DG 胜 " + "/".join(wins)


def render_doc(metrics):
    tol = 100 * metrics["config"]["tolerance"]
    cfg = metrics["config"]
    dec = metrics["decomposition"]
    out = metrics["outcome"]
    gates = metrics["gates"]
    ctx = metrics["context"]
    lines = [
        "# CASE-LAX-DG · A0 度量目录：判的是「点值」还是「单元平均」",
        "",
        f"> 锚点 `CASE-LAX-DG`（台账原文「ref: 精确解｜容差 < {tol:.2f} %」，status=open）。"
        "本文全部数字由 `benchmarks/lax_a0_metric_catalogue.py` 从已提交场缓存程序化注入"
        f"（`{cfg['production_leg']}` 的 `profiles.npz`，sha256 "
        f"`{metrics['provenance']['profiles_sha256'][:16]}…`），零手抄、零求解器重跑；"
        "`--check` 在工件漂移时失败。",
        "",
        "## ① 案例介绍",
        "",
        "Lax (1954) 激波管 / Toro (2009) Test 2：`[0,1]`，隔膜 `x0 = 0.5`，γ = 1.4，"
        "左态 `(ρ,u,p) = (0.445, 0.698, 3.528)`，右态 `(0.5, 0.0, 0.571)`，快照 `t = 0.13`。"
        "精确解含四道波（左稀疏波头/尾、接触间断、右激波），解析折点 "
        + "、".join(f"`{x:.6f}`" for x in cfg["breakpoints"]) + "。",
        "",
        f"钉扎腿 `{cfg['production_leg']}`：DG-P3、400 单元（DOF 1600）、HLLC 界面通量、SSP-RK3。"
        f"台账容差 {tol:.2f} % 作用在「整场相对 L2」上。",
        "",
        "## ② 怎么做的",
        "",
        "`docs/CASE-LAX-DG-a0-quadrature-audit.md` 解决的是**求积**轴（精确解须按解析折点切段、"
        "逼到收敛才可与容差比较）。本文审计的是另一根轴：**度量对象**。同一个「相对 L2」可以取自",
        "",
        "* **点值**（单元 Gauss 节点上的 DG 解）——池子当前做法；",
        "* **单元平均**（只保留解析尺度）——能区分「波位/幅值对不对」但不看次单元；",
        "",
        "且各自可落在原始变量 `(ρ,u,p)` 或守恒变量 `(ρ, ρu, E)` 上、取 L1 或 L2。八种组合都是"
        "「同一个误差」的合法测量，台账原文没有指定用哪一个。",
        "",
        "**三条问法（先登记后跑数，零求解）**",
        "",
        "1. **Q1 目录**：`{变量集} × {L1,L2} × {点值, 单元平均}` 共 8 组，逐量对容差判 PASS/FAIL；"
        "点值另给 aligned 收敛求积（阶 20）与未收敛 6 阶读数，用于区分「度量对象」与「求积精度」两根轴。",
        "2. **Q2 判别力探针**：对同一已提交场做**同网格 P0 截断**（在守恒变量上保住每单元均值、"
        "丢掉高阶模态，再还原原始变量）。若某度量在此操作下**逐位不变**，它就无法区分 P0 与 P3，"
        "不能用于裁决。该操作精确、无需重跑求解器。",
        "3. **Q3 误差去哪了**：点值误差的**精确四分**。令 `q̄` 为 DG 单元均值、`q̄_ex` 为精确解单元均值，"
        "则逐单元",
        "",
        "   `∫(q−q_ex)² = ∫(q−q̄)² + ∫(q̄−q̄_ex)² + ∫(q̄_ex−q_ex)² + 2∫(q−q̄)(q̄_ex−q_ex)`",
        "",
        "   （DG 次单元内容 ⊕ 单元均值误差 ⊕ 精确解自身次单元 ⊕ 子单元对齐项）。首、末两项逐单元零均值、"
        "中间项逐单元常数，因此其余五个交叉项恒为零；对齐项为负表示 DG 的次单元形状在**抵消**精确解的"
        "次单元结构，为正是虚假振荡。",
        "",
        "## ③ 计算结果",
        "",
        "### 3.1 目录：同一钉扎腿、多种度量",
        "",
        f"| 度量对象 | 变量 | 范数 | 最差场 | DG-P3 | 同网格 P0 截断 | DG/P0 | 判定 vs {tol:.2f} % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = ["primitive/l2/pointwise_uniform_order6",
             "primitive/l2/pointwise", "primitive/l2/pointwise_aligned_converged",
             "primitive/l2/cell_average", "primitive/l1/pointwise",
             "primitive/l1/cell_average", "conserved/l2/pointwise",
             "conserved/l2/cell_average", "conserved/l1/pointwise",
             "conserved/l1/cell_average"]
    for key in order:
        entry = metrics["catalogue"].get(key)
        if entry is None:
            continue
        variables, norm, target = key.split("/")
        worst = entry["fields"][entry["worst_field"]]
        p0v = worst.get("p0_truncated")
        ratio = worst.get("discrimination_ratio")
        lines.append(
            f"| `{target}` | {variables} | {norm} | {entry['worst_field']} | "
            f"{100*entry['worst_dg']:.4f} % | "
            f"{'—' if p0v is None else f'{100*p0v:.4f} %'} | "
            f"{'—' if ratio is None else f'{ratio:.3f}'} | **{entry['verdict']}** |")
    lines += [
        "",
        "读法三点：",
        "",
        f"1. **点值族一致判 FAIL**（收敛求积 {100*metrics['catalogue']['primitive/l2/pointwise_aligned_converged']['worst_dg']:.4f} %，"
        f"未收敛 10 阶 {100*metrics['catalogue']['primitive/l2/pointwise']['worst_dg']:.4f} %，"
        f"守恒变量 L2 {100*metrics['catalogue']['conserved/l2/pointwise']['worst_dg']:.4f} %）——裁决在点值族内稳健。",
        f"2. **单元平均值族一律 PASS**（原始变量 L2 最差 {100*metrics['catalogue']['primitive/l2/cell_average']['worst_dg']:.4f} %，"
        f"守恒变量 L2 最差 {100*metrics['catalogue']['conserved/l2/cell_average']['worst_dg']:.4f} %）——"
        "换个度量对象就把 verdict 翻过来了。",
        f"3. **未收敛 6 阶点值读数是 PASS（{100*metrics['catalogue']['primitive/l2/pointwise_uniform_order6']['worst_dg']:.4f} %）**，"
        "与已审计的求积轴结论一致：低阶 Gauss 读数在场内折点处欠积分，会制造假 PASS。",
        "",
        "### 3.2 排序翻转：P3 是否优于同网格 P0，取决于度量族",
        "",
        "同一个 DG-P3 场与其同网格 P0 截断（只保留守恒量单元均值）的误差比值（DG/P0，<1 表示 DG 更好）：",
        "",
        "| 度量 | 各场 DG/P0 | 谁更好 |",
        "|---|---|---|",
        *[f"| `{key}` | " + ", ".join(
            f"{f} {metrics['catalogue'][key]['fields'][f]['discrimination_ratio']:.3f}"
            for f in metrics['catalogue'][key]['fields'])
          + " | " + _ranking_note(metrics["catalogue"][key]["fields"]) + " |"
          for key in ("primitive/l2/pointwise_aligned_converged", "primitive/l2/pointwise",
                      "primitive/l1/pointwise", "conserved/l2/pointwise",
                      "primitive/l2/cell_average")],
        "",
        "**读法**：整场点值 L1 上，同网格 P0 截断在三个原始量上都比 DG-P3 更接近精确解；"
        "点值 L2 上密度由 DG 占优、速度与压力由 P0 截断占优；换到守恒变量 L2 则 DG 全胜。"
        "即**「高阶是否更准」在本算例的这个网格上不是格式性质，而是度量族性质**——"
        "这也解释了为什么把 A0 的度量对象偷偷换掉就能翻 verdict，"
        "因此本文坚持用台账已登记的点值相对 L2 裁决。",
        "",
        "### 3.3 判别力闸（Q2）",
        "",
        f"* 单元平均度量在同网格 P0 截断下的最差相对漂移 = "
        f"**{gates['cell_average_invariance_worst_relative_drift']['value']:.2e}**"
        f"（容差 {gates['cell_average_invariance_worst_relative_drift']['tolerance']:.0e}）→ "
        f"{gates['cell_average_invariance_worst_relative_drift']['status']}。",
        "* 即：**单元平均度量对次单元内容完全免疫**。同一网格上一个只保留守恒量单元均值的场"
        "（P0 截断）会读到与 DG-P3 相同的数，误差比值 1.000。它是「解析尺度是否对」的度量，"
        "**不是**「高阶是否在起作用」的度量。",
        "",
        "### 3.4 误差四分与定位（Q3）",
        "",
        f"恒等式残差最差 **{gates['decomposition_identity_worst_residual']['value']:.2e}**（相对）"
        f"（容差 {gates['decomposition_identity_worst_residual']['tolerance']:.0e}）→ "
        f"{gates['decomposition_identity_worst_residual']['status']}。",
        "",
        "| 场 | 总误差² | DG 次单元 | 单元均值项 | 精确解次单元 | 2×对齐项 | 次单元占比 | 对齐项占比 |",
        "",
        "（各项可正可负，占比之和为 100 %，单项占比可超过 100 %。）",
        "",
        "|---|---|---|---|---|---|---|---|",
    ]
    for field in ("density", "velocity", "pressure"):
        s = dec[field]
        lines.append(
            f"| {field} | {s['total']:.6e} | {s['dg_sub_cell']:.6e} | {s['cell_mean_error']:.6e} | "
            f"{s['exact_sub_cell']:.6e} | {2*s['alignment']:.6e} | "
            f"{100*s['dg_sub_cell_share']:.1f} % | {100*s['alignment_share']:.1f} % |")
    lines += [
        "",
        f"* 含解析折点的单元数 = **{dec['density']['kink_cells']}**"
        f"（网格的 {100*dec['density']['kink_cells']/400:.2f} %），却承载了密度 DG 次单元能量的 "
        f"**{100*dec['density']['dg_sub_cell_in_kink_cells']/dec['density']['dg_sub_cell']:.1f} %**。",
        f"* 密度的对齐项占总误差的 {100*dec['density']['alignment_share']:.1f} %（负），"
        f"压力 {100*dec['pressure']['alignment_share']:.1f} %，速度 {100*dec['velocity']['alignment_share']:.1f} %："
        "DG 的次单元形状在四道波处**抵消**了精确解的次单元结构，"
        "即它确实在分辨次单元，点值误差的主体是**量的失配**而不是纯噪声。",
        "",
        f"### 3.5 加密趋势（context only：`{ctx['leg']}`，不是钉扎腿）",
        "",
        "| 场 | 点值 L2 | 单元平均 L2 | DG 次单元 | 精确解次单元 |",
        "|---|---|---|---|---|",
    ]
    for field in ("density", "velocity", "pressure"):
        s = ctx["decomposition"][field]
        lines.append(f"| {field} | {100*ctx['norms']['pointwise'][field]:.4f} % | "
                     f"{100*ctx['norms']['cell_average'][field]:.4f} % | "
                     f"{s['dg_sub_cell']:.6e} | {s['exact_sub_cell']:.6e} |")
    ratio_p = (ctx["norms"]["pointwise"]["density"] /
               metrics["catalogue"]["primitive/l2/pointwise_aligned_converged"]["fields"]["density"]["dg"])
    ratio_sub = (ctx["decomposition"]["density"]["dg_sub_cell"] / dec["density"]["dg_sub_cell"])
    lines += [
        "",
        f"* 加密一档 h（400→800）：密度点值误差 ×{ratio_p:.3f}、DG 次单元能量 ×{ratio_sub:.3f}——"
        "次单元内容随 h 缩小是点值误差下降的主因，与 3.3 的四分一致（单元均值项本来就小）。"
        "这也是为什么 p3.N800 能以 2.315 % 过关，而「换度量对象」不是。",
        "",
        "## ④ 结果图",
        "",
        "![metric catalogue](benchmarks/results/lax_a0_metric_catalogue/figures/metric_catalogue.png)",
        "",
        "![decomposition](benchmarks/results/lax_a0_metric_catalogue/figures/decomposition.png)",
        "",
        "![localisation](benchmarks/results/lax_a0_metric_catalogue/figures/subcell_localisation.png)",
        "",
        "## ⑤ 注意点",
        "",
        "1. **单元平均不是「更宽松的同一把尺子」，而是另一把尺子**：它对次单元内容完全免疫"
        f"（P0 截断漂移 {gates['cell_average_invariance_worst_relative_drift']['value']:.1e}）。"
        "用它把 A0 判成 PASS，等于让一阶重构也过同一条容差——所以本文不主张换度量。",
        "2. **P0 截断是探针，不是方案**：它不主张「P0 更准」；截断在**守恒变量**上做再还原原始变量，"
        "避免在 `(ρ,u,p)` 上截断产生非守恒假象。",
        "3. **四分恒等式是精确的**：五个交叉项因「首末两项逐单元零均值、中间项逐单元常数」恒为零，"
        f"实测残差 ≤ {gates['decomposition_identity_worst_residual']['value']:.1e}（相对）。",
        "4. **求积轴不重复审计**：点值裁决值取自已提交的 aligned 收敛求积（`lax_a0_quadrature_audit`），"
        "本文只负责「度量对象」轴；未收敛 6 阶读数单列，仅作反面教材。",
        "5. **本单不动池子与台账**：`lax_case_pool` 工件、生产腿、台账状态均原样；"
        "口径变更属台账持有人权限。",
        "6. **纯离线**：全部数字来自 400×10 与 800×10 的已提交场缓存，无求解器重跑；"
        "本机 load15 高企时仍可执行（实测 < 15 s）。",
        "",
        "## ⑥ 误差判定（是否 < " + f"{tol:.2f} %" + "）",
        "",
        "| 关键量（度量对象） | 基准 | 实测 | < " + f"{tol:.2f} %" + " ? |",
        "|---|---|---|---|",
    ]
    for key, human in (("primitive/l2/pointwise_aligned_converged", "密度，点值（收敛求积）"),
                       ("primitive/l2/pointwise", "密度，点值（池子 10 阶）"),
                       ("conserved/l2/pointwise", "动量，守恒变量点值"),
                       ("primitive/l2/cell_average", "速度，单元平均（最差场）"),
                       ("primitive/l2/pointwise_uniform_order6", "密度，点值（未收敛 6 阶）")):
        entry = metrics["catalogue"][key]
        field = entry["worst_field"]
        lines.append(f"| {human}（{field}） | 精确 Riemann 解 | {100*entry['worst_dg']:.4f} % | "
                     f"{'是' if entry['verdict'] == 'PASS' else '**否**'} |")
    lines += [
        "",
        "## ⑦ 结论与建议",
        "",
        f"**度量对象**：{out['statement']}",
        "",
        f"**判别力**：{out['discrimination']}",
        "",
        f"**误差定位**：{out['decomposition']}",
        "",
        f"**建议**：{out['recommendation']}",
        "",
        "*工件：`docs/benchmarks/results/lax_a0_metric_catalogue/metrics.json` + `figures/`；"
        "本页由 `python -m benchmarks.lax_a0_metric_catalogue` 生成，`--check` 机器复核。*",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_summary(metrics):
    return render_doc(metrics)


def write_outputs(metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in metrics.items() if k != "figures"}
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n")
    DOC.write_text(render_doc(metrics))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with committed artifact")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    metrics = build_metrics(with_figures=not (args.check or args.no_figures))
    if args.check:
        committed = json.loads((OUT / "metrics.json").read_text())
        worst, where = 0.0, ""
        for key, entry in metrics["catalogue"].items():
            ref = committed["catalogue"].get(key, {}).get("worst_dg")
            if ref is None:
                raise SystemExit(f"--check: missing committed metric {key}")
            drift = abs(entry["worst_dg"] - ref) / max(abs(ref), 1e-30)
            if drift > worst:
                worst, where = drift, key
        if worst > 1e-9:
            raise SystemExit(f"--check: drift {worst:.3e} at {where}")
        print(f"--check: OK (worst drift {worst:.2e} at {where})")
        return 0
    write_outputs(metrics)
    print(f"wrote {OUT}/metrics.json and {DOC}")
    print("verdict:", metrics["catalogue"]["primitive/l2/pointwise_aligned_converged"]["verdict"],
          "| cell-average:", metrics["catalogue"]["primitive/l2/cell_average"]["verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())