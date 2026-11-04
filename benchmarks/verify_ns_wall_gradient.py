"""Second-order Dirichlet wall gradient for ``tensordg.ns_dg_2d``.

The m2589 audit (``docs/reports/tensordg-ns-wall-verification.md``) measured the
wall traction/heat flux as **first order** and the strong-form wall residual as
**O(1)** for a curved analytic state, and left the fix as an open item.  This
benchmark verifies the fix and decomposes what is left:

1. ``gradient`` -- the wall heat flux reconstructed from the two innermost cells
   (``g_face = 1.5 g_cell - 0.5 g_neighbour``, no SIPG penalty) against the
   analytic ``-kappa dT/dy``.  Pre-fix: slope 1.00; post-fix: slope 2.00.
2. ``penalty``  -- the two contributions to the flux the scheme actually uses:
   the reconstruction error and the symmetric-interior-penalty term
   ``(sigma/h) (T_wall - T|-)``.  The penalty term is O(h) because the trace
   jump is O(h^2); it is the dominant residual source and it is *not* a defect
   of the gradient (it is the stencil that makes SIPG coercive).
3. ``residual`` -- the strong-form residual of the exact steady Couette state,
   split into wall and interior rows, before vs after the fix.
4. ``transient`` -- the impulsively started Couette case (m2496 P0 case).  It
   still fails at t ~ 0.033: the gradient error was **not** the instability
   driver, so m2589's candidate mechanism (i) is retired.

All numbers are written to ``validation/ns_dg_2d_wall/gradient_fix.json`` and the
figure to ``docs/figures/ns_dg_2d_wall_gradient.png``.
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import tensordg.ns_dg_2d as ns  # noqa: E402
from tools import diag_ns_wall as diag  # noqa: E402

MU = diag.MU
KAPPA = diag.KAPPA
WALL_SPEED = diag.WALL_SPEED
AMPLITUDE = MU * WALL_SPEED**2 / (2.0 * KAPPA)
ANALYTIC_DT_DY_WALL = AMPLITUDE          # T = T_w + A y (1 - y), T'(0) = A
ANALYTIC_HEAT_FLUX_BOTTOM = -KAPPA * ANALYTIC_DT_DY_WALL
MESHES = (8, 16, 32, 64)


def _wall_traces(ny):
    d = diag.build_state((2, ny), 1)
    h = d["dy"]
    left = ns._face_primitives(d["coeff"], d["basis"], d["derivative"], d["signs"], 1, "left")
    return d, h, left, ns._shift_along(left, 1, -1)


def _row(ny):
    d, h, left, neighbour = _wall_traces(ny)
    penalty = 2.0 * 2.0**2
    gradient = float(ns._extrapolate_trace_derivative(
        left["T_n"], neighbour["T_n"],
    )[0, 0, 0]) * (2.0 / h)
    one_sided = float(left["T_n"][0, 0, 0]) * (2.0 / h)
    # penalty-free reconstruction == the flux with sigma = 0
    clean = ns._boundary_flux(left, d["boundary"]["ymin"], 1, h, MU, KAPPA, 0.0,
                              2.0 / h, 2.0 / d["dx"], neighbour=neighbour)
    total = ns._boundary_flux(left, d["boundary"]["ymin"], 1, h, MU, KAPPA, penalty,
                              2.0 / h, 2.0 / d["dx"], neighbour=neighbour)
    flux_clean = float(clean[0, 0, 0, 3])
    flux_total = float(total[0, 0, 0, 3])
    jump = float(d["boundary"]["ymin"]["temperature"])
    trace_T = float(left["temperature"][0, 0, 0])
    penalty_flux = -KAPPA * (-(penalty / h) * (trace_T - jump))
    return dict(
        ncells=[2, ny], degree=1, h=h,
        dT_dy_one_sided=one_sided,
        dT_dy_extrapolated=gradient,
        dT_dy_analytic=ANALYTIC_DT_DY_WALL,
        dT_dy_error_one_sided=abs(one_sided - ANALYTIC_DT_DY_WALL),
        dT_dy_error_extrapolated=abs(gradient - ANALYTIC_DT_DY_WALL),
        flux_clean=flux_clean,
        flux_clean_error=abs(flux_clean - ANALYTIC_HEAT_FLUX_BOTTOM),
        flux_total=flux_total,
        flux_total_error=abs(flux_total - ANALYTIC_HEAT_FLUX_BOTTOM),
        penalty_flux=penalty_flux,
        trace_temperature_error=abs(trace_T - jump),
    )


def section_gradient():
    rows = [_row(ny) for ny in MESHES]
    for previous, current in zip(rows, rows[1:]):
        current["slope_dT_dy_one_sided"] = math.log2(
            previous["dT_dy_error_one_sided"] / current["dT_dy_error_one_sided"])
        current["slope_dT_dy_extrapolated"] = math.log2(
            previous["dT_dy_error_extrapolated"] / current["dT_dy_error_extrapolated"])
        current["slope_flux_clean"] = math.log2(
            previous["flux_clean_error"] / current["flux_clean_error"])
        current["slope_flux_total"] = math.log2(
            previous["flux_total_error"] / current["flux_total_error"])
        current["slope_penalty_flux"] = math.log2(
            abs(previous["penalty_flux"]) / abs(current["penalty_flux"]))
    return rows


def section_residual():
    """Strong-form residual of the exact steady state, wall vs interior rows."""
    rows = []
    for ny in (8, 16, 32):
        d = diag.build_state((2, ny), 1)
        rhs = ns._rhs_ns(d["coeff"], d["basis"], d["derivative"], d["weights"],
                         d["signs"], d["dx"], d["dy"], (True, False), d["boundary"],
                         MU, KAPPA, 2.0 * 2.0**2)
        rows.append(dict(
            ncells=[2, ny], degree=1,
            rhs_max=float(rhs.abs().max()),
            energy=float(rhs[..., 3].abs().max()),
            momentum_x=float(rhs[..., 1].abs().max()),
            wall_row=float(rhs[:, [0, -1]].abs().max()),
            interior=float(rhs[:, 1:-1].abs().max()),
        ))
    for previous, current in zip(rows, rows[1:]):
        current["slope_interior"] = math.log2(previous["interior"] / current["interior"])
    return rows


def section_transient():
    """Impulsively started Couette: m2496 P0 case, still an honest open defect."""
    result = dict(ncells=[2, 8], degree=1, final_time=0.1)
    try:
        ns.solve_couette_dg_2d(ncells=(2, 8), degree=1, mu=MU, final_time=0.1,
                               device=torch.device("cpu"), dtype=torch.float64)
        result.update(survived=True, failure=None)
    except FloatingPointError as error:
        _, failure = diag.march(ncells=(2, 8), degree=1, max_steps=400)
        result.update(survived=False, failure=failure, error=str(error))
    return result


def make_figure(gradient, residual):
    h = [row["h"] for row in gradient]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.loglog(h, [row["dT_dy_error_one_sided"] for row in gradient], "o-",
              label="one-sided cell gradient (pre-fix, slope 1.00)")
    ax.loglog(h, [row["dT_dy_error_extrapolated"] for row in gradient], "s-",
              label="1.5g_cell - 0.5g_neighbour (this fix, slope 2.00)")
    ax.loglog(h, [abs(row["penalty_flux"]) / KAPPA for row in gradient], "^-",
              label="SIPG penalty term (sigma/h)[[T]] (O(h))")
    ax.loglog(h, [row["flux_clean_error"] for row in gradient], "d--",
              label="reconstructed wall heat flux error")
    ax.set_xlabel("h = 1/ny")
    ax.set_ylabel("error in dT/dy (K per unit length)")
    ax.set_title("(a) wall heat-flux reconstruction")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ny = [row["ncells"][1] for row in residual]
    ax.loglog(ny, [row["energy"] for row in residual], "o-", label="energy channel (wall row)")
    ax.loglog(ny, [row["interior"] for row in residual], "s-", label="interior rows")
    ax.loglog(ny, [0.3045 for _ in ny], "k:", label="energy residual pre-fix (0.3045)")
    ax.set_xlabel("ny (cells across the channel)")
    ax.set_ylabel("|rhs|_inf of the exact steady state")
    ax.set_title("(b) strong-form residual: wall vs interior")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    gradient = section_gradient()
    residual = section_residual()
    transient = section_transient()
    payload = dict(
        mu=MU, kappa=KAPPA, wall_speed=WALL_SPEED, amplitude=AMPLITUDE,
        gradient=gradient, residual=residual, transient=transient,
    )
    out = root / "validation" / "ns_dg_2d_wall" / "gradient_fix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    fig = make_figure(gradient, residual)
    fig_path = root / "docs" / "figures" / "ns_dg_2d_wall_gradient.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130)
    print(json.dumps(dict(
        gradient=[{k: (round(v, 10) if isinstance(v, float) else v)
                   for k, v in row.items()} for row in gradient],
        residual=residual, transient=transient,
    ), indent=1))


if __name__ == "__main__":
    main()