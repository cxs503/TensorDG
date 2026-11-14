"""CASE-DG-EULER-VORTEX-ERR: error *conservation* of the transported isentropic vortex.

Sibling of :mod:`benchmarks.case_euler_vortex` (case-pool entry ``EULER-VORTEX``,
"等熵涡平移"), which fixes one transport time ``t = 1`` and one question: is the
field error below tolerance *at that instant*?  That snapshot check cannot
distinguish a scheme whose error is a fixed discretisation residue from one
whose error keeps accumulating while the vortex crosses the periodic box.

This entry (``EULER-VORTEX-ERR``, 等熵涡移动误差守恒) asks the transport
question.  The exact problem has a symmetry: uniform translation in a periodic
box leaves the *shape* of the vortex invariant, hence the discretisation error
is a property of the mesh/flux pair, not of the elapsed time.  Four
machine-decidable criteria:

============ ==========================================================
C1 tolerance every checkpoint's max primitive-field relative L2 < 6 %
             (case-pool tolerance, ref. Yee et al. 2000).
C2 error     at the design gate config the error must not be secular:
   conservation   ``e(t_last)/e(t_ref) <= 2`` **and** the least-squares
             exponent of ``e ~ t^alpha`` satisfies ``alpha <= 0.5``.
C3 discrete  total mass / x-momentum / y-momentum / energy drift by
   conservation < 1e-10 relative to the *t=0 discrete* state (periodic
             interface fluxes telescope, so this is a round-off-level
             identity, not a modelling assumption).
C4 entropy   ``S = int p rho^{-gamma} dA`` (exact value = box area, the
   budget    vortex is isentropic) drifts by < 1e-3 relative; physical
             dissipation may only *decrease* it.
============ ==========================================================

Two independent discrete families are measured, so the criteria are not
self-certified by one code path:

* ``dg`` -- modal DG (:func:`tensordg.euler_dg_2d.solve_euler_dg_2d_vortex`,
  HLLC + SSP-RK3, L2-projected initial state, volume-quadrature metric);
* ``p0`` -- P0 finite volume (:func:`tensordg.euler_2d.solve_isentropic_vortex`,
  Rusanov + SSP-RK3, exact cell-average metric).

Both are compared with the *exact* solution (:func:`tensordg.euler_2d.
vortex_primitive`), whose exactness this package already verifies independently
(``case_euler_vortex.py --task residual``: ``U_t + F_x + G_y = 0`` to 1e-10).

One process per configuration, so the ladder is embarrassingly parallel::

    python benchmarks/case_euler_vortex_errconserve.py --shard 3/16 --output-dir DIR
    python benchmarks/case_euler_vortex_errconserve.py --aggregate --output-dir DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_1d import GAMMA
from tensordg.euler_2d import _cell_averages, solve_isentropic_vortex, vortex_error, vortex_primitive

from benchmarks.verify_euler_dg_2d import _reconstruct, _vortex_error as dg_vortex_error

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
BOX_AREA = (DOMAIN[0][1] - DOMAIN[0][0]) * (DOMAIN[1][1] - DOMAIN[1][0])

#: C1 -- case-pool tolerance for this entry (ref: Yee et al. 2000).
TOLERANCE = 0.06
#: C2 -- allowed error growth from ``T_REF`` to the last checkpoint.
GROWTH_RATIO_MAX = 2.0
MAX_GROWTH_EXPONENT = 0.5
T_REF = 1.0
#: C3 -- round-off level for a global invariant (1e-10 ~ 4.5e5 eps).
INVARIANT_TOLERANCE = 1e-10
#: Gauss points per direction of ``solve_euler_2d``'s cell-average projection.
#: Must mirror that solver's default, otherwise the t=0 reference state (and
#: hence every conservation drift) would be measured against a different state.
P0_QUADRATURE = 4
#: C4 -- entropy-functional drift at the gate configuration.
ENTROPY_TOLERANCE = 1e-3

#: Long-time transport ladder.  With |u_inf| = sqrt(2) and a period of 10,
#: ``t = 10`` is one full box crossing per axis and the P0 family reaches
#: ``t = 20`` (two crossings).  The high-order families are capped at ``t = 10``
#: by the per-run budget of the compute gate; the report states that scope
#: explicitly instead of extrapolating.
LADDER = (
    *({"task": "dg", "degree": 2, "cells": 32, "final_time": t, "cfl": 0.2}
      for t in (1.0, 2.0)),
    *({"task": "p0", "cells": 128, "final_time": t, "cfl": 0.4}
      for t in (1.0, 2.0, 5.0, 10.0)),
)

#: Configuration carrying C2/C4 (same pair of meshes as the sibling entry's gate).
GATE_FAMILY = "dg_d2_n32"


def task_key(config: dict) -> str:
    if config["task"] == "dg":
        return (f"dg_d{config['degree']}_n{config['cells']}"
                f"_t{config['final_time']:g}_cfl{config['cfl']:g}")
    if config["task"] == "p0":
        return (f"p0_n{config['cells']}_t{config['final_time']:g}"
                f"_cfl{config['cfl']:g}")
    raise ValueError(f"unknown task {config['task']!r}")


def family_key(record: dict) -> str:
    if record["task"] == "dg":
        return f"dg_d{record['degree']}_n{record['cells']}"
    return f"p0_n{record['cells']}"


# ---------------------------------------------------------------------------
# exact-solution invariants
# ---------------------------------------------------------------------------

def exact_invariants(points: int = 320, strength: float = STRENGTH,
                     background=BACKGROUND):
    """Global invariants of the exact initial state via Gauss-Legendre quadrature.

    The vortex is smooth (core scale O(1) in a 10x10 box), so 320 points per
    direction are converged far below the 1e-12 resolution of the drift columns.
    """
    nodes, weights = np.polynomial.legendre.leggauss(points)
    (xmin, xmax), (ymin, ymax) = DOMAIN
    nodes_t = torch.as_tensor(nodes, dtype=torch.float64)
    weights_t = torch.as_tensor(weights, dtype=torch.float64)
    x = 0.5 * (xmin + xmax) + 0.5 * (xmax - xmin) * nodes_t
    y = 0.5 * (ymin + ymax) + 0.5 * (ymax - ymin) * nodes_t
    rho, u, v, p = vortex_primitive(x[None, :], y[:, None], 0.0, strength, background)
    energy = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
    jacobian = 0.25 * (xmax - xmin) * (ymax - ymin)
    weight = weights_t[:, None] * weights_t[None, :] * jacobian
    return {
        "mass": float((rho * weight).sum()),
        "momentum_x": float((rho * u * weight).sum()),
        "momentum_y": float((rho * v * weight).sum()),
        "energy": float((energy * weight).sum()),
        "entropy": float(((p / rho ** GAMMA) * weight).sum()),
        "density_min": float(rho.min()),
    }


def exact_core_density(points: int = 4001, strength: float = STRENGTH):
    """Minimum core density, sampled from ``vortex_primitive`` on a fine ray.

    No closed form is hand-transcribed: the value comes from the same analytic
    function the solvers are tested against.
    """
    radius = torch.linspace(0.0, 3.0, points, dtype=torch.float64)
    rho, _, _, _ = vortex_primitive(radius, torch.zeros_like(radius), 0.0,
                                   strength, (0.0, 0.0))
    return float(rho.min())


# ---------------------------------------------------------------------------
# per-configuration measurement
# ---------------------------------------------------------------------------

def _dg_measurements(coeff, basis, nodes, weights):
    """Invariants of one DG state, on the same quadrature as the case gate."""
    nx, ny = coeff.shape[0], coeff.shape[1]
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, dtype=coeff.dtype) + 0.5) * dx
    y = ymin + (torch.arange(ny, dtype=coeff.dtype) + 0.5) * dy
    state = _reconstruct(coeff, basis)
    weight = (weights[:, None] * weights[None, :]) * (0.25 * dx * dy)
    rho, _, _, p = dg2._primitive(state)
    entropy = p / rho ** GAMMA
    field = lambda component: float((state[..., component] * weight).sum())
    s_total = float((entropy * weight).sum())
    return {
        "invariants": {
            "mass": field(0),
            "momentum_x": field(1),
            "momentum_y": field(2),
            "energy": field(3),
            "entropy": s_total,
        },
        "entropy_mean": s_total / float(weight.sum()),
        "density_min": float(rho.min()),
        "pressure_min": float(p.min()),
    }


def _p0_measurements(fields):
    """Invariants of one P0 (cell-average) state; quadrature-free and exact."""
    nx, ny = fields.shape[0], fields.shape[1]
    (xmin, xmax), (ymin, ymax) = DOMAIN
    cell_area = ((xmax - xmin) / nx) * ((ymax - ymin) / ny)
    rho = fields[..., 0]
    mom_x = fields[..., 1]
    mom_y = fields[..., 2]
    energy = fields[..., 3]
    u = mom_x / rho
    v = mom_y / rho
    p = (GAMMA - 1.0) * (energy - 0.5 * rho * (u ** 2 + v ** 2))
    entropy = p / rho ** GAMMA
    total = float(entropy.sum())
    return {
        "invariants": {
            "mass": float(rho.sum() * cell_area),
            "momentum_x": float(mom_x.sum() * cell_area),
            "momentum_y": float(mom_y.sum() * cell_area),
            "energy": float(energy.sum() * cell_area),
            "entropy": float(total * cell_area),
        },
        "entropy_mean": float(entropy.mean()),
        "density_min": float(rho.min()),
        "pressure_min": float(p.min()),
    }


def _operators(degree: int):
    return dg2.legendre_operators(degree, max(degree + 2, 3))


def _initial_dg_state(config: dict):
    """The exact t=0 discrete state the solver starts from (same code path)."""
    degree, cells = config["degree"], config["cells"]
    nodes, weights, basis, _, _ = _operators(degree)
    nx = ny = cells
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    strength = config.get("strength", STRENGTH)
    background = tuple(config.get("background_velocity", BACKGROUND))

    def initial(xx, yy):
        return vortex_primitive(xx, yy, 0.0, strength, background)

    coeff = dg2._initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis,
                                      torch.device("cpu"), torch.float64)
    signs = _operators(degree)[4]
    coeff = dg2._positivity_filter(coeff, basis, signs)
    return coeff, basis, nodes, weights


def _initial_p0_state(config: dict):
    cells = config["cells"]
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / cells, (ymax - ymin) / cells
    x = xmin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dy
    strength = config.get("strength", STRENGTH)
    background = tuple(config.get("background_velocity", BACKGROUND))

    def initial(xx, yy):
        return vortex_primitive(xx, yy, 0.0, strength, background)

    return _cell_averages(initial, x, y, dx, dy, P0_QUADRATURE,
                          torch.device("cpu"), torch.float64)


def run_dg(config: dict) -> dict:
    started = time.time()
    degree = config["degree"]
    strength = config.get("strength", STRENGTH)
    background = tuple(config.get("background_velocity", BACKGROUND))
    final_time = config.get("final_time", 1.0)
    _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=config["cells"],
        degree=degree,
        final_time=final_time,
        cfl=config.get("cfl", 0.2),
        strength=strength,
        background_velocity=background,
        domain=DOMAIN,
    )
    nodes, weights, basis, _, _ = _operators(degree)
    errors = dg_vortex_error(coeff, basis, nodes, DOMAIN, final_time, strength,
                            background)
    final = _dg_measurements(coeff, basis, nodes, weights)
    coeff0, basis0, nodes0, weights0 = _initial_dg_state(config)
    initial = _dg_measurements(coeff0, basis0, nodes0, weights0)
    return {
        "task": "dg",
        "degree": degree,
        "cells": config["cells"],
        "final_time": final_time,
        "cfl": config.get("cfl", 0.2),
        "steps": meta["steps"],
        "dt": meta["dt"],
        "dofs": meta["dofs"],
        "errors": {name: float(errors[name]) for name in FIELDS},
        "metric": "volume-quadrature relative L2 (L2-projected modal state)",
        "final": final,
        "initial": initial,
        "wall_seconds": time.time() - started,
    }


def run_p0(config: dict) -> dict:
    started = time.time()
    strength = config.get("strength", STRENGTH)
    background = tuple(config.get("background_velocity", BACKGROUND))
    final_time = config.get("final_time", 1.0)
    x, y, fields, meta = solve_isentropic_vortex(
        ncells=config["cells"],
        final_time=final_time,
        cfl=config.get("cfl", 0.4),
        domain=DOMAIN,
        strength=strength,
        background_velocity=background,
    )
    conservative = fields["conservative"] if isinstance(fields, dict) else fields
    raw = vortex_error(x, y, conservative, final_time, strength, background)
    errors = {name: raw[name]["relative_l2"] for name in FIELDS}
    final = _p0_measurements(conservative)
    initial = _p0_measurements(_initial_p0_state(config))
    return {
        "task": "p0",
        "degree": 0,
        "cells": config["cells"],
        "final_time": final_time,
        "cfl": config.get("cfl", 0.4),
        "steps": meta["steps"],
        "dt": meta.get("dt"),
        "dofs": int(conservative.numel()),
        "errors": {name: float(errors[name]) for name in FIELDS},
        "metric": "exact cell-average relative L2 (P0 finite volume)",
        "final": final,
        "initial": initial,
        "wall_seconds": time.time() - started,
    }


def run_task(config: dict) -> dict:
    record = (run_dg if config["task"] == "dg" else run_p0)(config)
    record["task_key"] = task_key(config)
    record["family"] = family_key(record)
    record["config"] = dict(config)
    record["max_relative_l2"] = max(record["errors"].values())
    record["max_percent"] = 100.0 * record["max_relative_l2"]
    drift = {}
    for name, value in record["final"]["invariants"].items():
        reference = record["initial"]["invariants"][name]
        drift[name] = abs(value - reference) / abs(reference)
    record["invariant_drift"] = drift
    record["entropy_production"] = (
        record["final"]["invariants"]["entropy"] / record["initial"]["invariants"]["entropy"]
        - 1.0)
    record["entropy_drift"] = abs(record["entropy_production"])
    record["density_min_relative_error"] = (
        abs(record["final"]["density_min"] - record["initial"]["density_min"])
        / abs(record["initial"]["density_min"]))
    record["gate_pass"] = record["max_relative_l2"] < TOLERANCE
    return record


# ---------------------------------------------------------------------------
# aggregation / gates / tables
# ---------------------------------------------------------------------------

def _fit_exponent(times, errors, t_min=T_REF):
    """Least-squares exponent of ``e ~ t^alpha`` over ``t >= t_min``.

    Fewer than three checkpoints is reported as ``nan``: a two-point "fit" through
    an error curve that starts at the projection floor says nothing about
    accumulation, and turning it into a gate would be a vacuous claim.
    """
    points = [(t, e) for t, e in zip(times, errors) if t >= t_min and e > 0.0]
    if len(points) < 3:
        return float("nan")
    lt = np.log([p[0] for p in points])
    le = np.log([p[1] for p in points])
    return float(np.polyfit(lt, le, 1)[0])


def _group(records):
    families = {}
    for record in records:
        families.setdefault(record["family"], []).append(record)
    for rows in families.values():
        rows.sort(key=lambda r: r["final_time"])
    return families


def evaluate_gates(records: list, exact: dict) -> dict:
    families = _group(records)
    gate_rows = [r for r in families.get(GATE_FAMILY, []) if r["final_time"] >= T_REF]
    gates = {}

    worst = max(r["max_relative_l2"] for r in records)
    gates["C1_tolerance"] = {
        "criterion": f"max over checkpoints of max-field relative L2 < {TOLERANCE:.0%}",
        "measured": worst,
        "passed": bool(worst < TOLERANCE),
    }

    exponents = {name: _fit_exponent([r["final_time"] for r in rows],
                                    [r["max_relative_l2"] for r in rows])
                 for name, rows in families.items()}
    ratio = float("nan")
    if gate_rows:
        reference = [r for r in gate_rows if abs(r["final_time"] - T_REF) < 1e-12]
        if reference:
            ratio = gate_rows[-1]["max_relative_l2"] / reference[0]["max_relative_l2"]
    gate_exponent = exponents.get(GATE_FAMILY, float("nan"))
    gates["C2_error_conservation"] = {
        "criterion": (f"{GATE_FAMILY}: e(t_last)/e(t={T_REF:g}) <= {GROWTH_RATIO_MAX} "
                      f"and alpha <= {MAX_GROWTH_EXPONENT}"),
        "growth_ratio": ratio,
        "exponent": gate_exponent,
        "exponents_all_families": exponents,
        "status": ("evaluated" if np.isfinite(gate_exponent) else "not_evaluated"),
        "passed": bool(np.isfinite(ratio) and ratio <= GROWTH_RATIO_MAX
                       and np.isfinite(gate_exponent)
                       and gate_exponent <= MAX_GROWTH_EXPONENT),
    }

    worst_drift, worst_at = 0.0, None
    for record in records:
        for name in ("mass", "momentum_x", "momentum_y", "energy"):
            value = record["invariant_drift"][name]
            if value > worst_drift:
                worst_drift, worst_at = value, f"{record['task_key']}:{name}"
    gates["C3_discrete_conservation"] = {
        "criterion": ("relative drift of mass/x-mom/y-mom/energy vs the t=0 discrete "
                      f"state < {INVARIANT_TOLERANCE:g}"),
        "worst_drift": worst_drift,
        "worst_at": worst_at,
        "passed": bool(worst_drift < INVARIANT_TOLERANCE),
    }

    entropy_drift = max((r["entropy_drift"] for r in gate_rows), default=float("nan"))
    gates["C4_entropy_budget"] = {
        "criterion": (f"{GATE_FAMILY}: |Delta S|/S(0) < {ENTROPY_TOLERANCE:g} "
                      "(S = int p rho^-gamma)"),
        "measured": entropy_drift,
        "exact_entropy": exact["entropy"],
        "passed": bool(np.isfinite(entropy_drift) and entropy_drift < ENTROPY_TOLERANCE),
    }
    gates["all_passed"] = all(v["passed"] for v in gates.values() if isinstance(v, dict))
    not_evaluated = [k for k, v in gates.items()
                     if isinstance(v, dict) and v.get("status") == "not_evaluated"]
    gates["not_evaluated"] = not_evaluated
    gates["verdict"] = ("PASS" if gates["all_passed"]
                        else ("INCOMPLETE" if not_evaluated else "FAIL"))
    return {"gates": gates, "families": {k: len(v) for k, v in families.items()}}


def aggregate(records: list, exact: dict, output_dir: Path) -> dict:
    evaluation = evaluate_gates(records, exact)
    metrics = {
        "case": "EULER-VORTEX-ERR",
        "ledger_slug": "CASE-DG-EULER-VORTEX-ERR",
        "reference": (
            "Case-pool entry EULER-VORTEX-ERR cites Yee, Sandham & Djomehri (2000) "
            "with a <6 % tolerance; the analytic isentropic-vortex solution itself "
            "is the standard form documented in Shu (1998) sec. 4.4.2 "
            "(domain [-5,5]^2, beta=5, u_inf=(1,1), gamma=1.4, periodic). The "
            "tolerance is taken from the case-pool entry verbatim and is NOT "
            "re-derived from the paper text (the paper is not available in-repo); "
            "the acceptance numbers below are therefore measured against the "
            "analytic solution, which is independent of which paper introduced it."),
        "tolerance_relative_l2": TOLERANCE,
        "ladder": list(LADDER),
        "exact_invariants": exact,
        "gates": evaluation["gates"],
        "family_count": evaluation["families"],
        "records": [
            {
                "task_key": r["task_key"],
                "family": r["family"],
                "config": r["config"],
                "final_time": r["final_time"],
                "cfl": r["cfl"],
                "steps": r["steps"],
                "errors": r["errors"],
                "max_percent": r["max_percent"],
                "invariant_drift": r["invariant_drift"],
                "entropy_drift": r["entropy_drift"],
                "entropy_production": r["entropy_production"],
                "density_min_relative_error": r["density_min_relative_error"],
                "initial_invariants": r.get("initial", {}).get("invariants"),
                "final_invariants": r.get("final", {}).get("invariants"),
                "gate_pass": r["gate_pass"],
                "wall_seconds": r["wall_seconds"],
            }
            for r in sorted(records, key=lambda r: (r["family"], r["final_time"]))
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "error_conservation.csv").write_text(_csv(records))
    (output_dir / "summary.md").write_text(_markdown(records, metrics, exact))
    return metrics


def _csv(records: list) -> str:
    header = ("family,final_time,steps,max_percent,mass_drift,momentum_x_drift,"
              "momentum_y_drift,energy_drift,entropy_drift,density_min_relative_error\n")
    lines = []
    for r in sorted(records, key=lambda r: (r["family"], r["final_time"])):
        d = r["invariant_drift"]
        lines.append(
            f"{r['family']},{r['final_time']:g},{r['steps']},{r['max_percent']:.6f},"
            f"{d['mass']:.3e},{d['momentum_x']:.3e},{d['momentum_y']:.3e},"
            f"{d['energy']:.3e},{r['entropy_drift']:.3e},"
            f"{r['density_min_relative_error']:.6e}\n")
    return header + "".join(lines)


def _markdown(records: list, metrics: dict, exact: dict) -> str:
    gates = metrics["gates"]

    def flag(name):
        return "通过" if gates[name]["passed"] else "未通过"

    lines = [
        "# CASE-EULER-VORTEX-ERR：等熵涡移动的误差守恒",
        "",
        "本文件由 `benchmarks/case_euler_vortex_errconserve.py --aggregate` 从 "
        "`results/*.json` 程序化生成，数字零手抄。",
        "",
        "## 门禁",
        "",
        "| 判据 | 内容 | 实测 | 判定 |",
        "|---|---|---|---|",
        f"| C1 容差 | 各检查点最大场相对 L2 < {TOLERANCE:.0%} | "
        f"{gates['C1_tolerance']['measured']:.4%} | {flag('C1_tolerance')} |",
        f"| C2 误差守恒 | e(t_last)/e(t={T_REF:g}) <= {GROWTH_RATIO_MAX}，"
        f"alpha <= {MAX_GROWTH_EXPONENT} | 比值 "
        f"{gates['C2_error_conservation']['growth_ratio']:.3f}，alpha "
        f"{gates['C2_error_conservation']['exponent']:.3f} | "
        f"{flag('C2_error_conservation')} |",
        f"| C3 离散守恒 | 质量/动量/能量漂移 < {INVARIANT_TOLERANCE:g} | "
        f"{gates['C3_discrete_conservation']['worst_drift']:.3e} | "
        f"{flag('C3_discrete_conservation')} |",
        f"| C4 熵预算 | \\|dS\\|/S(0) < {ENTROPY_TOLERANCE:g} | "
        f"{gates['C4_entropy_budget']['measured']:.3e} | "
        f"{flag('C4_entropy_budget')} |",
        "",
        "## 误差随时间（族内逐检查点）",
        "",
        "| 族 | t | 步数 | rho % | u % | p % | 最大场 % | 族 alpha |",
        "|---|---|---|---|---|---|---|---|",
    ]
    families = _group(records)
    for name, rows in families.items():
        exponent = _fit_exponent([r["final_time"] for r in rows],
                                 [r["max_relative_l2"] for r in rows])
        for r in rows:
            e = r["errors"]
            lines.append(
                f"| {name} | {r['final_time']:g} | {r['steps']} | "
                f"{100 * e['density']:.4f} | {100 * e['velocity_x']:.4f} | "
                f"{100 * e['pressure']:.4f} | {r['max_percent']:.4f} | "
                f"{exponent:.3f} |")
    lines += [
        "",
        "## 守恒与熵",
        "",
        "| 配置 | t | 质量漂移 | x 动量漂移 | 能量漂移 | 熵漂移 | 核密度相对误差 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: (r["family"], r["final_time"])):
        d = r["invariant_drift"]
        lines.append(
            f"| {r['task_key']} | {r['final_time']:g} | {d['mass']:.2e} | "
            f"{d['momentum_x']:.2e} | {d['energy']:.2e} | {r['entropy_drift']:.2e} | "
            f"{r['density_min_relative_error']:.3e} |")
    lines += [
        "",
        "精确解初始不变量（320 点 Gauss 每方向）：",
        "",
        "| 量 | 值 |",
        "|---|---|",
        f"| 质量 | {exact['mass']:.12f} |",
        f"| x 动量 | {exact['momentum_x']:.12f} |",
        f"| 能量 | {exact['energy']:.12f} |",
        f"| 熵 S = ∫p·rho^-gamma（应等于面积 {BOX_AREA:g}）| {exact['entropy']:.12f} |",
        f"| 核最小密度 | {exact['density_min']:.9f} |",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(records: list, output_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    families = _group(records)
    figures = []

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, rows in sorted(families.items()):
        times = [r["final_time"] for r in rows]
        values = [100.0 * r["max_relative_l2"] for r in rows]
        ax.loglog(times, values, "o-", label=name)
    ax.axhline(100.0 * TOLERANCE, color="crimson", ls="--",
               label=f"tolerance {TOLERANCE:.0%} (Yee et al. 2000 / case pool)")
    ax.set_xlabel("dimensionless time t  (|u_inf|t = travel distance, period 10)")
    ax.set_ylabel("max primitive-field relative L2 error [%]")
    ax.set_title("Isentropic vortex, long-time transport: does the error accumulate?")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_error_vs_time.png", dpi=150)
    plt.close(fig)
    figures.append({"name": "fig_error_vs_time.png",
                    "caption": "各离散族最大场相对 L2 误差随无量纲时间的变化（双对数轴，"
                               "斜率即累积指数 alpha）；红虚线为算例池 6% 容差。"})

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for name, rows in sorted(families.items()):
        times = [r["final_time"] for r in rows]
        axes[0].semilogy(times, [max(r["invariant_drift"]["mass"], 1e-18) for r in rows],
                         "o-", label=f"{name} mass")
        axes[0].semilogy(times, [max(r["invariant_drift"]["energy"], 1e-18) for r in rows],
                         "s--", label=f"{name} energy")
        axes[1].semilogy(times, [max(r["entropy_drift"], 1e-18) for r in rows],
                         "o-", label=f"{name} |dS|/S")
        axes[1].semilogy(times, [r["density_min_relative_error"] for r in rows],
                         "s--", label=f"{name} core-density error")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("relative drift of a global invariant")
    axes[0].set_title("C3 discrete conservation (periodic interface fluxes telescope)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("relative measure")
    axes[1].set_title("C4 entropy budget and core-density dissipation")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_invariants_vs_time.png", dpi=150)
    plt.close(fig)
    figures.append({"name": "fig_invariants_vs_time.png",
                    "caption": "左：C3 质量/能量守恒漂移（应处于舍入水平）；"
                               "右：C4 熵泛函漂移与涡核最小密度的相对误差随时间的增长。"})
    return figures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def select_shard(index: int, total: int) -> list:
    """Configurations of shard ``index`` out of ``total``.

    The comparison uses ``index % total`` so that *both* 0-based (``0..n-1``) and
    1-based (``1..n``) shard conventions used by different schedulers partition
    the ladder exactly once -- neither convention can silently drop config 0.
    """
    if total < 1:
        raise ValueError("total must be >= 1")
    if index < 0:
        raise ValueError("index must be non-negative")
    return [config for position, config in enumerate(LADDER)
            if position % total == index % total]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("dg", "p0"), default=None)
    parser.add_argument("--cells", type=int, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--strength", type=float, default=STRENGTH)
    parser.add_argument("--output-dir", default="case_euler_vortex_errconserve")
    parser.add_argument("--shard", default=None, help="i/n: run every n-th ladder entry")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    output_dir = Path(args.output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    if args.task is not None:
        config = {"task": args.task, "cells": args.cells or 32,
                  "final_time": args.final_time if args.final_time is not None else 1.0,
                  "cfl": args.cfl if args.cfl is not None else (0.2 if args.task == "dg" else 0.4),
                  "strength": args.strength}
        if args.task == "dg":
            config["degree"] = args.degree if args.degree is not None else 2
        record = run_task(config)
        path = output_dir / "results" / f"{record['task_key']}.json"
        path.write_text(json.dumps(record, indent=2))
        print(f"{record['task_key']}: max {record['max_percent']:.4f}% "
              f"({record['wall_seconds']:.1f} s) -> {path}")
        return 0

    if args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        for config in select_shard(index, total):
            path = output_dir / "results" / f"{task_key(config)}.json"
            if path.exists():
                print(f"skip existing {path.name}", flush=True)
                continue
            record = run_task(config)
            path = output_dir / "results" / f"{record['task_key']}.json"
            path.write_text(json.dumps(record, indent=2))
            print(f"{record['task_key']}: max {record['max_percent']:.4f}% "
                  f"({record['wall_seconds']:.1f} s) -> {path}", flush=True)
        return 0

    # aggregate whatever checkpoints exist
    records = []
    for path in sorted((output_dir / "results").glob("*.json")):
        records.append(json.loads(path.read_text()))
    if not records:
        raise SystemExit(f"no records in {output_dir / 'results'}")
    exact = exact_invariants()
    exact["core_density"] = exact_core_density()
    metrics = aggregate(records, exact, output_dir)
    if args.figures:
        make_figures(records, output_dir)
    print(json.dumps(metrics["gates"], indent=2))
    return 0 if metrics["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())