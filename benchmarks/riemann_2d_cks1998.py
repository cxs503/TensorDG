"""CKS1998 2-D Riemann problem benchmark (Lax & Liu 1998, config 3).

This is the 2-D system benchmark referenced in Cockburn & Shu (JCP 141, 1998,
"V: Multidimensional systems") for the RKDG scheme on the 2-D Euler equations.

The four-quadrant Riemann problem (Lax & Liu 1998, config 3) features the
interaction of two shock waves and is a standard test for 2-D shock-capturing.

Reference:
    Lax, P. D. & Liu, X.-D. (1998). "Solution of Two-Dimensional Riemann
    Problems of Gas Dynamics by Positive Schemes on a Rectangular Grid."
    SIAM J. Sci. Comput. 19(2), 319-340.

    Cockburn, B. & Shu, C.-W. (1998/2001). "The Runge-Kutta discontinuous
    Galerkin method for conservation laws V: Multidimensional systems."
    J. Comput. Phys. 141, 199-224; SIAM Review 43(1), 2001.

Gate history (important, do not silently re-baseline)
-----------------------------------------------------
The first version of this module declared four gates, two of which were
**ill-posed** and were therefore reported as honest FAILs in
``docs/CASE-DG-CKS1998-2D-RIEMANN.md``:

* ``C4 mass conservation < 1 %`` was declared for "periodic BCs", but the case
  driver runs the physical configuration with *transmissive* boundaries
  (``periodic=(False, False)`` in ``solve_euler_dg_2d_riemann``), so total mass
  legitimately changes by the boundary flux (25 % at t=0.1).  Reporting that as
  a limiter defect -- "the TVB limiter's modal scaling is not conservative" --
  was wrong: the limiter never touches the cell average.  The well-posed
  replacement is the **discrete flux-balance identity**
  ``Q(t) - Q(0) == integral of the boundary flux``, which closes to round-off
  for a conservative scheme on *any* boundary condition.  The raw (physical)
  mass change is still reported, it is just no longer scored as "conservation".
* ``C2 self-convergence`` compared the *total masses* of two meshes, which is a
  scalar, mesh-independent quantity for a fixed final time -- it measures
  nothing about convergence.  The replacement is a real relative ``L1`` distance
  between the coarse mesh and the block-averaged fine mesh.

See ``docs/CASE-DG-CKS1998-2D-RIEMANN-CONSERVATION.md`` and
``validation/riemann2d_conservation/audit.py`` for the measurements.

Quantitative gates (pre-declared, before any run):
    C1  Positivity: rho > 0 and p > 0 at all quadrature points and cell averages.
    C2  Mesh self-convergence: relative L1(density) between 64x64 and the
        block-averaged 32x32 < 0.05.
    C3  Finite solution: no NaN/Inf in the final state.
    C4  Discrete flux balance: |Q(t) - Q(0) - boundary flux| / scale < 1e-10
        for mass, both momenta and energy (transmissive boundaries).
    C5  Periodic control: on a periodic domain the same data conserves mass,
        momentum and energy to |change| / scale < 1e-12.

Usage:
    # smoke profile = QUICK_CONFIG (32x32 P1, t=0.1) + periodic control + figure,
    # then render the case document's number table from the record
    OMP_NUM_THREADS=1 python -m benchmarks.riemann_2d_cks1998 --quick --sync-doc
    # full case time t=0.25 with the 32x32/64x64 self-convergence (C2) ladder
    OMP_NUM_THREADS=1 python -m benchmarks.riemann_2d_cks1998
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_2d import (
    LAX_LIU_CONFIG_3,
    cell_fields,
    legendre_operators,
    solve_euler_dg_2d,
)

# Pre-declared gates (fixed before any run)
GATES = {
    "C1_positivity": "rho > 0, p > 0 at all volume quadrature points and cell averages",
    "C2_self_convergence_l1": 0.05,     # relative L1(density), 64x64 vs block-mean 32x32
    "C3_finite": "no NaN/Inf",
    "C4_flux_balance_rel": 1e-10,       # |Q(t)-Q(0)-boundary flux| / scale
    "C5_periodic_drift_rel": 1e-12,     # same data, periodic domain, no boundary at all
    "C6_mass_flux_safety": 5.0,         # |dM| <= factor * rho_max0 * s_max0 * perimeter * t
    "C6_extremum_factor": 1.5,          # max(rho,p) <= factor * max over the initial data
}

# Pre-registered C6 constants (written before any C6 run; see
# docs/CASE-DG-CKS1998-2D-RIEMANN-DIVERGENCE-GATE.md).
#   * the mass bound is *a priori*: it is built from the four initial Riemann
#     states and the geometry only, never from a measured run, so it cannot be
#     fitted to whatever the divergence happened to produce;
#   * ``MASS_FLUX_SAFETY = 5`` is the slack for the fact that the boundary value
#     is not exactly an initial state once waves reach the wall;
#   * ``EXTREMUM_SAFETY = 1.5`` is a *plausibility* factor, not a theorem: for
#     2-D Euler there is no rigorous maximum principle for an entropy solution
#     (shock focusing can compress above the initial max), so C6 only claims
#     "the discrete solution stays within 50 % of the initial extrema".
MASS_FLUX_SAFETY = 5.0
EXTREMUM_SAFETY = 1.5

# The ``--quick`` profile is a *smoke* configuration and is declared here rather
# than assembled from CLI defaults: the committed record, the case document and
# the doc-binding test all quote this one dict, so "quick" cannot mean two things.
# 32x32 P1 at ``t = 0.1`` keeps the whole profile (physical run + periodic control
# + figure) inside a *single* bounded command (a minute of CPU on one core); the
# full case time ``t = 0.25`` is what the ``full`` profile runs, together with the
# 32x32/64x64 self-convergence ladder (C2).
QUICK_CONFIG = {
    "ncells": 32,
    "degree": 1,
    "final_time": 0.1,
    "cfl": 0.2,
}

CONSERVED = ("mass", "x_momentum", "y_momentum", "energy")

# Lax & Liu 1998 config 3 states (rho, u, v, p)
# These are the exact initial states; no closed-form solution exists at t > 0.
CONFIG_3 = LAX_LIU_CONFIG_3


# ---------------------------------------------------------------------------
# Initial data / measurement helpers
# ---------------------------------------------------------------------------

def initial_condition(x, y):
    """Four-quadrant Lax & Liu config-3 data, ``(rho, u, v, p)`` on any shape."""
    shape = torch.broadcast_shapes(x.shape, y.shape)
    lower = torch.broadcast_to(y < 0.5, shape)
    left = torch.broadcast_to(x < 0.5, shape)
    rho = torch.zeros(shape, dtype=x.dtype)
    velocity_x = torch.zeros(shape, dtype=x.dtype)
    velocity_y = torch.zeros(shape, dtype=x.dtype)
    pressure = torch.zeros(shape, dtype=x.dtype)
    quadrants = (
        ("lower-left", lower & left),
        ("lower-right", lower & ~left),
        ("upper-left", ~lower & left),
        ("upper-right", ~lower & ~left),
    )
    for name, mask in quadrants:
        state = CONFIG_3[name]
        rho = torch.where(mask, torch.full(shape, state[0], dtype=x.dtype), rho)
        velocity_x = torch.where(mask, torch.full(shape, state[1], dtype=x.dtype), velocity_x)
        velocity_y = torch.where(mask, torch.full(shape, state[2], dtype=x.dtype), velocity_y)
        pressure = torch.where(mask, torch.full(shape, state[3], dtype=x.dtype), pressure)
    return rho, velocity_x, velocity_y, pressure


def a_priori_bounds(final_time, states=None, domain=((0.0, 1.0), (0.0, 1.0))):
    """C6 bounds derived from the *initial data and geometry only*.

    Nothing in here reads a computed solution, so the same numbers are produced
    before and after any run; that is what makes C6 a gate rather than a
    post-hoc descriptor.

    Returns a dict with

    ``mass0``            initial total mass (exact, from the quadrant areas),
    ``rho_max0/p_max0``  initial extrema,
    ``speed_max0``       bound on the fastest signal speed, ``|u|+|v|+a``,
    ``perimeter``        length of the domain boundary,
    ``mass_flux_bound``  ``MASS_FLUX_SAFETY * rho_max0 * speed_max0 * P * t``,
    ``rho_cap/p_cap``    ``EXTREMUM_SAFETY`` times the initial extrema.
    """
    states = CONFIG_3 if states is None else states
    (x0, x1), (y0, y1) = domain
    area = (x1 - x0) * (y1 - y0)
    perimeter = 2.0 * ((x1 - x0) + (y1 - y0))
    rho_max0 = max(state[0] for state in states.values())
    p_max0 = max(state[3] for state in states.values())
    speed_max0 = max(
        math.hypot(state[1], state[2]) + math.sqrt(GAMMA * state[3] / state[0])
        for state in states.values()
    )
    # Each quadrant covers a quarter of the domain.
    mass0 = 0.25 * area * sum(state[0] for state in states.values())
    return {
        "final_time": float(final_time),
        "mass0": float(mass0),
        "rho_max0": float(rho_max0),
        "p_max0": float(p_max0),
        "speed_max0": float(speed_max0),
        "perimeter": float(perimeter),
        "mass_flux_bound": float(MASS_FLUX_SAFETY * rho_max0 * speed_max0 * perimeter * final_time),
        "rho_cap": float(EXTREMUM_SAFETY * rho_max0),
        "p_cap": float(EXTREMUM_SAFETY * p_max0),
    }


def _check_positivity(coeff, degree):
    """rho/p at the volume quadrature nodes and at the cell averages.

    Measured with an independent quadrature reconstruction (not the solver's own
    admissibility test) so that a bug in ``_admissible_cells`` cannot hide here.
    """
    nodes, _weights, basis, _derivative, _signs = legendre_operators(
        degree, max(degree + 2, 3), device=coeff.device, dtype=coeff.dtype
    )
    state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho = state[..., 0]
    pressure = (GAMMA - 1.0) * (
        state[..., 3] - 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / rho.clamp_min(1e-30)
    )
    mean = coeff[..., 0, 0, :]
    mean_pressure = (GAMMA - 1.0) * (
        mean[..., 3]
        - 0.5 * (mean[..., 1] ** 2 + mean[..., 2] ** 2) / mean[..., 0].clamp_min(1e-30)
    )
    return {
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "p_min": float(pressure.min()),
        "p_max": float(pressure.max()),
        "rho_min_mean": float(mean[..., 0].min()),
        "p_min_mean": float(mean_pressure.min()),
        "rho_positive": bool(torch.all(rho > 0.0)),
        "p_positive": bool(torch.all(pressure > 0.0)),
        "rho_positive_mean": bool(torch.all(mean[..., 0] > 0.0)),
        "p_positive_mean": bool(torch.all(mean_pressure > 0.0)),
        "quadrature_nodes": int(nodes.numel()),
    }


def _total_mass(coeff, dx, dy):
    """Total mass = integral of rho over the domain (from the cell averages)."""
    return float((coeff[:, :, 0, 0, 0] * dx * dy).sum())


def block_average(means, factor):
    """Average an ``(n, n)`` array of cell means down by an integer ``factor``."""
    n = int(means.shape[0])
    if factor < 1 or n % factor:
        raise ValueError(f"cannot coarsen {n} by {factor}")
    m = n // factor
    return means.reshape(m, factor, m, factor).mean(dim=(1, 3))


def self_convergence_l1(coarse_means, fine_means):
    """Relative L1 distance between a coarse mesh and the block-averaged fine mesh."""
    factor = int(fine_means.shape[0]) // int(coarse_means.shape[0])
    folded = block_average(fine_means, factor)
    return float(torch.abs(folded - coarse_means).sum()) / max(
        float(torch.abs(folded).sum()), 1e-30
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_riemann_2d(ncells=32, degree=1, final_time=0.25, cfl=0.2,
                   limiter="tvb", tvb_constant=1.0, device="cpu",
                   periodic=(False, False)):
    """Run one 2-D Riemann configuration and measure every gate quantity.

    ``periodic`` defaults to the *physical* configuration of the case
    (transmissive boundaries); the ``(True, True)`` variant is the
    conservation control with no boundary at all.
    """
    started = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        initial_condition,
        ncells=ncells,
        degree=degree,
        domain=((0.0, 1.0), (0.0, 1.0)),
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        tvb_constant=tvb_constant,
        periodic=periodic,
        initial_projection="l2",
        device=device,
        track_boundary_flux=True,
    )
    wall = time.time() - started
    bounds = a_priori_bounds(final_time)

    fields = cell_fields(coeff)
    pos = _check_positivity(coeff, degree)
    means = coeff[..., 0, 0, 0].detach().cpu().clone()

    conservation = {}
    for name in CONSERVED:
        row = meta["conservation_residual"][name]
        conservation[name] = {
            "change": row["change"],
            "boundary_flux": row["boundary_flux"],
            "residual": row["residual"],
            "relative_residual": row["relative"],
        }

    return {
        "ncells": ncells,
        "degree": degree,
        "final_time": final_time,
        "cfl": cfl,
        "periodic": tuple(bool(flag) for flag in periodic),
        "steps": meta["steps"],
        "dt": meta.get("dt"),
        "wall_seconds": round(wall, 2),
        "rho_min": float(fields["density"].min()),
        "rho_max": float(fields["density"].max()),
        "p_min": float(fields["pressure"].min()),
        "p_max": float(fields["pressure"].max()),
        "positivity": pos,
        "total_mass": _total_mass(coeff, 1.0 / ncells, 1.0 / ncells),
        "initial_mass": bounds["mass0"],
        "c6_bounds": bounds,
        "conservation": conservation,
        "max_flux_balance_residual": max(
            conservation[name]["relative_residual"] for name in CONSERVED
        ),
        "max_abs_conservation_change": max(
            abs(conservation[name]["change"]) for name in CONSERVED
        ),
        "finite": bool(torch.isfinite(coeff).all()),
        "dofs": meta.get("dofs", ncells * ncells * (degree + 1) ** 2 * 4),
        "cell_means_rho": means,
        "x": x,
        "y": y,
        "coeff": coeff,
    }


def _slim(record):
    return {k: v for k, v in record.items() if k not in ("cell_means_rho", "x", "y", "coeff")}


# ---------------------------------------------------------------------------
# Gate verdicts (pure function so the tests can probe its sensitivity)
# ---------------------------------------------------------------------------

def judge(rows, convergence=None, gates=None):
    """Score every gate from measured rows; never returns a hardcoded verdict."""
    gates = GATES if gates is None else gates
    verdicts = {}

    verdicts["C1_positivity"] = {
        "passed": all(
            r["positivity"]["rho_positive"]
            and r["positivity"]["p_positive"]
            and r["positivity"]["rho_positive_mean"]
            and r["positivity"]["p_positive_mean"]
            for r in rows
        ),
        "worst_rho_min": min(r["positivity"]["rho_min"] for r in rows),
        "worst_p_min": min(r["positivity"]["p_min"] for r in rows),
        "detail": "rho > 0 and p > 0 at volume nodes and cell averages",
    }

    verdicts["C3_finite"] = {
        "passed": all(r["finite"] for r in rows),
        "detail": "no NaN/Inf",
    }

    open_rows = [r for r in rows if not (r["periodic"][0] and r["periodic"][1])]
    if open_rows:
        worst = max(r["max_flux_balance_residual"] for r in open_rows)
        verdicts["C4_flux_balance"] = {
            "passed": worst < gates["C4_flux_balance_rel"],
            "worst_relative_residual": worst,
            "tolerance": gates["C4_flux_balance_rel"],
            "detail": (
                "Q(t) - Q(0) equals the time-integrated boundary flux; the "
                "transmissive mass change itself is physical and is reported "
                "under 'mass change' rather than being scored as conservation"
            ),
        }
    else:
        verdicts["C4_flux_balance"] = {
            "passed": False,
            "worst_relative_residual": None,
            "tolerance": gates["C4_flux_balance_rel"],
            "detail": "NOT_RUN: no non-periodic row in this profile",
            "check_state": "not_run",
        }

    periodic_rows = [r for r in rows if r["periodic"][0] and r["periodic"][1]]
    if periodic_rows:
        worst_scale = max(
            abs(r["conservation"][name]["change"])
            / max(
                abs(r["conservation"][name]["boundary_flux"]),
                abs(r["total_mass"]),
                1e-30,
            )
            for r in periodic_rows
            for name in CONSERVED
        )
        verdicts["C5_periodic_drift"] = {
            "passed": worst_scale < gates["C5_periodic_drift_rel"],
            "worst_relative_change": worst_scale,
            "tolerance": gates["C5_periodic_drift_rel"],
            "detail": "periodic control: no boundary flux is possible",
        }
    else:
        verdicts["C5_periodic_drift"] = {
            "passed": False,
            "worst_relative_change": None,
            "tolerance": gates["C5_periodic_drift_rel"],
            "detail": "NOT_RUN: no periodic control row in this profile",
            "check_state": "not_run",
        }

    if convergence:
        worst = max(convergence.values())
        verdicts["C2_self_convergence"] = {
            "passed": worst < gates["C2_self_convergence_l1"],
            "worst_relative_l1": worst,
            "tolerance": gates["C2_self_convergence_l1"],
            "per_pair": dict(convergence),
        }
    else:
        verdicts["C2_self_convergence"] = {
            "passed": False,
            "worst_relative_l1": None,
            "tolerance": gates["C2_self_convergence_l1"],
            "detail": "NOT_RUN: quick profile has a single mesh",
            "check_state": "not_run",
        }

    # --- C6: is the solution still physical? ---------------------------------
    # The C1/C4/C5 trio is blind to divergence: a diverging run keeps rho,p > 0
    # (C1), keeps closing the flux-balance identity to round-off (C4) and, on a
    # periodic control, keeps every conserved quantity flat (C5) while the mass
    # itself runs to 1e+26.  C6 is the missing "is this still a solution of the
    # PDE" gate and is scored against bounds fixed by the *initial data*.
    c6_rows = [r for r in rows if c6_row_is_scorable(r)]
    if c6_rows:
        worst_mass = None
        worst_extremum = None
        per_row = {}
        for r in c6_rows:
            ratio = c6_mass_bound_ratio(r)
            extremum = c6_extremum_factor(r)
            per_row[f"p{r['degree']}_{r['ncells']}_{'periodic' if all(r['periodic']) else 'open'}"] = {
                "mass_bound_ratio": ratio,
                "extremum_factor": extremum,
            }
            worst_mass = ratio if worst_mass is None else max(worst_mass, ratio)
            worst_extremum = extremum if worst_extremum is None else max(worst_extremum, extremum)
        verdicts["C6_physical_magnitude"] = {
            "passed": bool(
                worst_mass <= 1.0 and worst_extremum <= gates["C6_extremum_factor"]
            ),
            "worst_mass_bound_ratio": worst_mass,
            "worst_extremum_factor": worst_extremum,
            "mass_tolerance": 1.0,
            "extremum_tolerance": gates["C6_extremum_factor"],
            "per_row": per_row,
            "detail": (
                "|mass(t)-mass(0)| must stay inside an a-priori flux bound "
                "(MASS_FLUX_SAFETY * rho_max0 * speed_max0 * perimeter * t) and "
                "max(rho,p) must stay within EXTREMUM_SAFETY x the initial "
                "extrema; bounds come from the initial data, not from a run"
            ),
        }
    else:
        verdicts["C6_physical_magnitude"] = {
            "passed": False,
            "worst_mass_bound_ratio": None,
            "worst_extremum_factor": None,
            "detail": "NOT_RUN: no row carries a complete set of a-priori bounds",
            "check_state": "not_run",
        }

    return verdicts


def c6_row_is_scorable(row):
    """True when the row carries everything C6 needs (no silent defaults)."""
    if not row.get("c6_bounds"):
        return False
    positivity = row.get("positivity") or {}
    if "rho_max" not in positivity or "p_max" not in positivity:
        return False
    mass = (row.get("conservation") or {}).get("mass")
    if not mass or mass.get("change") is None:
        return False
    return math.isfinite(float(mass["change"]))


def c6_mass_bound_ratio(row):
    """``|mass(t) - mass(0)| / mass_flux_bound`` for one measured row.

    Both operands are pre-existing fields of the row; no fitting is involved.
    """
    bounds = row["c6_bounds"]
    change = row["conservation"]["mass"]["change"]
    return float(abs(change)) / max(bounds["mass_flux_bound"], 1e-30)


def c6_extremum_factor(row):
    """``max(rho_max/rho_max0, p_max/p_max0)`` for one measured row."""
    bounds = row["c6_bounds"]
    return float(
        max(
            row["positivity"]["rho_max"] / max(bounds["rho_max0"], 1e-30),
            row["positivity"]["p_max"] / max(bounds["p_max0"], 1e-30),
        )
    )


def run_convergence_study(degrees=(1, 2), meshes=(32, 64), final_time=0.25,
                          cfl=0.2, device="cpu"):
    """Mesh self-convergence study: coarse mesh vs block-averaged fine mesh."""
    results = {}
    for degree in degrees:
        rows = []
        for ncells in meshes:
            record = run_riemann_2d(ncells=ncells, degree=degree,
                                    final_time=final_time, cfl=cfl, device=device)
            rows.append(record)
            print(f"  P{degree} {ncells}x{ncells}: {record['steps']} steps, "
                  f"rho=[{record['positivity']['rho_min']:.4e}, "
                  f"{record['positivity']['rho_max']:.4f}], "
                  f"flux-balance resid={record['max_flux_balance_residual']:.2e}, "
                  f"{record['wall_seconds']:.1f}s", flush=True)

        convergence = {}
        for coarse, fine in zip(rows, rows[1:]):
            key = f"{coarse['ncells']}_vs_{fine['ncells']}"
            convergence[key] = self_convergence_l1(
                coarse["cell_means_rho"], fine["cell_means_rho"]
            )
        results[f"p{degree}"] = {
            "rows": [_slim(r) for r in rows],
            "full_rows": rows,
            "self_convergence_l1": convergence,
        }
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_density_field(x, y, coeff, degree, ncells, output_path):
    """Density contour plot for the 2-D Riemann problem."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = cell_fields(coeff)
    density = fields["density"].cpu().numpy()
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full domain
    im0 = axes[0].contourf(x_np, y_np, density.T, levels=30, cmap="viridis")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"2-D Riemann (Lax-Liu Config 3) | DG P{degree}, {ncells}x{ncells}")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], label="Density")

    # Zoom on interaction region
    mask_x = (x_np > 0.3) & (x_np < 0.7)
    mask_y = (y_np > 0.3) & (y_np < 0.7)
    x_zoom = x_np[mask_x]
    y_zoom = y_np[mask_y]
    if len(x_zoom) > 0 and len(y_zoom) > 0:
        density_zoom = density[np.ix_(mask_x, mask_y)]
        im1 = axes[1].contourf(x_zoom, y_zoom, density_zoom.T, levels=30, cmap="viridis")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Zoom: wave interaction region")
        axes[1].set_aspect("equal")
        fig.colorbar(im1, ax=axes[1], label="Density")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Document binding: the case document quotes only numbers rendered from the record
# ---------------------------------------------------------------------------
#
# The failure this closes: ``docs/CASE-DG-CKS1998-2D-RIEMANN.md`` used to quote a
# transmissive mass change of ``0.3494`` while ``metrics.json`` in the same
# directory said ``0.0828``, and a 16x16 configuration that the documented
# reproduce command no longer produced (the driver had moved to 32x32, t=0.25).
# Nothing failed, because every number in the document was typed by hand.
#
# Here the document's number table is *rendered* from the record.  ``--sync-doc``
# writes it into the document, and
# ``tests/test_riemann_2d_cks1998_doc_binding.py`` compares the two byte for byte,
# so a hand edit, a record regenerated at another configuration, or a record whose
# verdicts no longer follow from its own rows all fail the build.

DOC_BLOCK_BEGIN = "<!-- DOC-NUMBERS:BEGIN"
DOC_BLOCK_END = "<!-- DOC-NUMBERS:END -->"
DOC_VALUES_HEADER = "| 量 | metrics.json 路径 | 值 |"

# ``(label, dotted path into metrics.json, formatting)``.  Paths are explicit so
# the document states where every number comes from.
DOC_NUMBER_FIELDS = (
    ("profile", "profile", "str"),
    ("物理构型 ncells", "runs[0].ncells", "int"),
    ("物理构型 degree", "runs[0].degree", "int"),
    ("物理构型 final_time", "runs[0].final_time", "num"),
    ("物理构型 cfl", "runs[0].cfl", "num"),
    ("物理构型 时间步数", "runs[0].steps", "int"),
    ("C1 rho_min（格点最低值，含周期控制行）",
     "verdicts.C1_positivity.worst_rho_min", "sci"),
    ("C1 p_min（格点最低值，含周期控制行）",
     "verdicts.C1_positivity.worst_p_min", "sci"),
    ("C1 判定", "verdicts.C1_positivity", "verdict"),
    ("C3 判定", "verdicts.C3_finite", "verdict"),
    ("物理构型 初始质量", "runs[0].initial_mass", "num"),
    ("物理构型 末态质量", "runs[0].total_mass", "num"),
    ("物理构型 质量变化（物理边界通量，不是守恒误差）",
     "runs[0].conservation.mass.change", "num"),
    ("C4 通量平衡最坏相对残差",
     "verdicts.C4_flux_balance.worst_relative_residual", "sci"),
    ("C4 判定", "verdicts.C4_flux_balance", "verdict"),
    ("周期控制 时间步数", "runs[1].steps", "int"),
    ("C5 周期控制最坏相对漂移",
     "verdicts.C5_periodic_drift.worst_relative_change", "sci"),
    ("C5 判定", "verdicts.C5_periodic_drift", "verdict"),
    ("C6 最坏质量界比（|dM| / 先验通量界）",
     "verdicts.C6_physical_magnitude.worst_mass_bound_ratio", "sci"),
    ("C6 最坏极值因子（max(rho,p) / 初值极值）",
     "verdicts.C6_physical_magnitude.worst_extremum_factor", "num"),
    ("C6 判定", "verdicts.C6_physical_magnitude", "verdict"),
    ("C2 判定", "verdicts.C2_self_convergence", "verdict"),
    ("结果图文件", "provenance.figure", "str"),
    ("结果图取自的记录行", "provenance.figure_source_row", "int"),
)

_PATH_TOKEN = re.compile(r"([^.\[\]]+)(?:\[(\d+)\])?")


def _resolve(record, path):
    """Dotted-path lookup with ``[i]`` indices; ``KeyError`` when absent."""
    node = record
    for name, index in _PATH_TOKEN.findall(path):
        node = node[int(name)] if isinstance(node, list) else node[name]
        if index:
            node = node[int(index)]
    return node


def _format_value(value, kind):
    if kind == "verdict":
        if value.get("check_state") == "not_run":
            return "NOT_RUN"
        return "PASS" if value.get("passed") else "FAIL"
    if kind == "int":
        return str(int(value))
    if kind == "num":
        return f"{float(value):.6g}"
    if kind == "sci":
        return f"{float(value):.3e}"
    return str(value)


def doc_number_rows(record):
    """``(label, path, rendered value)`` triples quoted by the case document."""
    rows = []
    for label, path, kind in DOC_NUMBER_FIELDS:
        try:
            value = _resolve(record, path)
        except (KeyError, IndexError, TypeError):
            # A profile that legitimately lacks a row (no periodic control, no
            # convergence study) says so instead of silently dropping the line.
            rows.append((label, path, "NOT RECORDED"))
            continue
        rows.append((label, path, _format_value(value, kind)))
    return rows


def render_doc_numbers_block(record):
    """Markdown block the case document must contain, rendered from ``record``."""
    header = (
        DOC_BLOCK_BEGIN
        + " (由 `python -m benchmarks.riemann_2d_cks1998 --quick --sync-doc`"
        + " 从 metrics.json 渲染，禁止手改) -->"
    )
    lines = [header, DOC_VALUES_HEADER, "|---|---:|:---:|"]
    for label, path, text in doc_number_rows(record):
        lines.append(f"| {label} | `{path}` | {text} |")
    lines.append(DOC_BLOCK_END)
    return "\n".join(lines)


def sync_document(record, doc_path):
    """Replace the ``DOC-NUMBERS`` block of ``doc_path`` with the rendered one."""
    doc_path = Path(doc_path)
    text = doc_path.read_text(encoding="utf-8")
    start = text.find(DOC_BLOCK_BEGIN)
    end = text.find(DOC_BLOCK_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"{doc_path}: DOC-NUMBERS markers not found")
    end += len(DOC_BLOCK_END)
    updated = text[:start] + render_doc_numbers_block(record) + text[end:]
    if updated != text:
        doc_path.write_text(updated, encoding="utf-8")
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--time", type=float, default=0.25)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="docs/benchmarks/results/riemann_2d_cks1998")
    parser.add_argument("--sync-doc", action="store_true",
                        help="rewrite the DOC-NUMBERS block of the case document "
                             "from the record just produced")
    parser.add_argument("--doc", default=str(REPO / "docs" / "CASE-DG-CKS1998-2D-RIEMANN.md"))
    args = parser.parse_args()

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CKS1998 2-D Riemann Problem Benchmark (Lax & Liu 1998, Config 3)")
    print("=" * 60)
    print("  Initial states (rho, u, v, p):")
    for name, state in CONFIG_3.items():
        print(f"    {name}: {state}")
    print(f"  Gates: {GATES}")
    print()

    if args.quick:
        config = QUICK_CONFIG
        print(f"[Quick] {config['ncells']}x{config['ncells']} P{config['degree']}, "
              f"t={config['final_time']}, CFL={config['cfl']}: physical "
              "(transmissive) + periodic control ...")
        if (args.time, args.cfl) != (QUICK_CONFIG["final_time"], QUICK_CONFIG["cfl"]):
            print("  note: --time/--cfl apply to the full profile; the quick "
                  "profile is fixed by QUICK_CONFIG", flush=True)
        physical = run_riemann_2d(ncells=config["ncells"], degree=config["degree"],
                                  final_time=config["final_time"], cfl=config["cfl"],
                                  device=args.device)
        control = run_riemann_2d(ncells=config["ncells"], degree=config["degree"],
                                 final_time=config["final_time"], cfl=config["cfl"],
                                 device=args.device, periodic=(True, True))
        rows = [physical, control]
        convergence = None
        profile = "quick"
        quick_config = {
            "ncells": physical["ncells"],
            "degree": physical["degree"],
            "final_time": physical["final_time"],
            "cfl": physical["cfl"],
        }
        figure_row = physical
        command = "python -m benchmarks.riemann_2d_cks1998 --quick"
    else:
        print("[Full] convergence study P1/P2, 32x32 and 64x64 ...")
        conv = run_convergence_study(degrees=(1, 2), meshes=(32, 64),
                                     final_time=args.time, cfl=args.cfl,
                                     device=args.device)
        rows = [dict(r, periodic=(False, False)) for deg in conv.values() for r in deg["rows"]]
        convergence = {}
        for deg in conv.values():
            convergence.update(deg["self_convergence_l1"])
        profile = "full"
        quick_config = None
        figure_row = next(r for r in conv["p1"]["full_rows"] if r["ncells"] == 32)
        command = "python -m benchmarks.riemann_2d_cks1998"

    # The figure is drawn from an *already measured* row: the previous version
    # re-solved the same configuration a third time just to plot it, which both
    # wasted a full solve and made the picture and the recorded row two different
    # trajectories.
    figure_name = (
        f"riemann_2d_density_p{figure_row['degree']}_{figure_row['ncells']}.png"
    )
    figure_row_index = next(
        index
        for index, row in enumerate(rows)
        if row["ncells"] == figure_row["ncells"]
        and row["degree"] == figure_row["degree"]
        and tuple(row["periodic"]) != (True, True)
    )
    plot_density_field(figure_row["x"], figure_row["y"], figure_row["coeff"],
                       figure_row["degree"], figure_row["ncells"],
                       output / figure_name)
    print(f"  figure: {output / figure_name} (from runs[{figure_row_index}], "
          f"not a duplicate solve)", flush=True)

    verdicts = judge(rows, convergence=convergence)

    record = {
        "benchmark": "riemann_2d_cks1998",
        "profile": profile,
        "config": "Lax_Liu_1998_config_3",
        "gates": GATES,
        "runs": [_slim(r) for r in rows],
        "verdicts": verdicts,
        "provenance": {
            "command": command,
            "quick_config": quick_config,
            "figure": figure_name,
            "figure_source_row": figure_row_index,
            "note": (
                "figure and rows come from the same solves; no row is "
                "recomputed for plotting"
            ),
        },
        "reference": "Lax & Liu (1998), SIAM J. Sci. Comput. 19(2), 319-340",
        "notes": (
            "The transmissive mass change is a physical boundary flux, not a "
            "conservation error; C4 scores the flux-balance identity instead."
        ),
    }
    with open(output / "metrics.json", "w") as handle:
        json.dump(record, handle, indent=2, default=str)

    if args.sync_doc:
        doc_path = Path(args.doc)
        changed = sync_document(record, doc_path)
        print(f"\n  case document {'updated' if changed else 'already in sync'}: {doc_path}")

    print()
    print("Verdicts:")
    all_pass = True
    for gate, row in verdicts.items():
        if row.get("check_state") == "not_run":
            status = "NOT_RUN"
        else:
            status = "PASS" if row["passed"] else "FAIL"
            if not row["passed"]:
                all_pass = False
        print(f"  {gate}: {status}  {json.dumps({k: v for k, v in row.items() if k != 'detail'})}")

    print(f"\nAll scored gates: {'PASS' if all_pass else 'FAIL'}")
    print(f"Results written to: {output}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
