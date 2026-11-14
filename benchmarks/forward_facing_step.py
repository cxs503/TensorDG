"""Internal solid obstacles in the 2-D DG solver + the CKS1998 forward-facing step.

Until this driver existed ``tensordg.euler_dg_2d`` could only put a wall on the
outer boundary of the box: the ``bcs`` mapping keys on ``left/right/bottom/top``.
A body that is *inside* the mesh -- the forward-facing step of Cockburn & Shu,
JCP 141 (1998) 199--224 (the 2-D system benchmark of that paper, after
Woodward & Colella, JCP 54 (1984)) -- therefore had no representation at all.

The solver now takes ``solid``: a boolean ``(nx, ny)`` cell mask.  Every face
between a solid and a fluid cell carries the exact slip-wall flux
``(0, p, 0, 0)`` built from the *fluid* trace, faces between two solid cells
carry nothing, the outer-face flux of a solid cell is zero and the solid cells
are frozen at their projected initial data.  No ghost layer, no cut cells: the
obstacle must be grid aligned, which is exactly what the step is.

Verification (all references are independent of the solver):

**G1 free-stream preservation across an embedded wall.**  A uniform stream
parallel to every solid face must not be disturbed: the wall flux then equals
the physical flux and the residual is zero to round-off, so this is a hard gate
(``< 1e-10`` relative), not a small-error claim.

**G2 closed box with an obstacle.**  All four outer faces are walls and the
obstacle is inside.  Neither carries mass nor energy, so both domain integrals
must be constant to round-off.  Momentum is *not* conserved -- the obstacle
pushes back -- which is why the solver also reports ``obstacle_impulse``: the
time integral of the solid-face momentum flux, obtained by the same RK-weighted
quadrature as the outer-surface ledger.  With it the momentum identity closes
too (``conservation_residual_total``).

**G3 shock reflecting off an embedded wall.**  The classical exact two-shock
solution (``benchmarks.euler_dg_2d_wall_bc.WallReflection``: the incident shock
is initialised from the Rankine--Hugoniot relations, the reflected shock Mach
number comes from bisecting the jump conditions).  The wall is now an *internal*
solid block instead of a domain face, so the same reference validates the new
face treatment against a closed-form solution.

**G4 the paper case itself** (Mach 3 step, gamma = 1.4, unit-length channel).
Quantitative content: the still-undisturbed upstream state (exact), the step
drag coefficient from ``obstacle_impulse``, the leading-shock position and the
peak density at two resolutions, and the free-stream error of the region the
shock has not reached.  No digitised literature curve is available in this
repository, so the paper's contour plots are compared *qualitatively*
(triple point / slip line / recirculation present) and the case is documented as
a demonstration, not as an L4 quantitative certificate.

Usage::

    python -m benchmarks.forward_facing_step --profile quick   # gates + 60x20
    python -m benchmarks.forward_facing_step --profile full    # gates + 60x20/90x30
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

from tensordg.euler_dg_2d import (
    GAMMA,
    conservative_state,
    normalise_solid,
    solve_euler_dg_2d,
)
from benchmarks.euler_dg_2d_wall_bc import WallReflection

STEP_X = 0.6   # leading face of the step        (CKS1998 / Woodward--Colella)
STEP_Y = 0.2   # step height
LENGTH = 3.0   # channel length
HEIGHT = 1.0   # channel height
MACH = 3.0     # inflow Mach number
RHO_INF = 1.4
P_INF = 1.0
U_INF = MACH * math.sqrt(GAMMA * P_INF / RHO_INF)


def step_mask(ncells, domain=((0.0, LENGTH), (0.0, HEIGHT)), step=(STEP_X, STEP_Y)):
    """Boolean cell mask of the forward-facing step (``True`` = solid)."""
    nx, ny = ncells
    (xmin, xmax), (ymin, ymax) = domain
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    return (x[:, None] < step[0]) & (y[None, :] < step[1])


def free_stream(x, y):
    return (
        torch.full_like(x, RHO_INF),
        torch.full_like(x, U_INF),
        torch.zeros_like(x),
        torch.full_like(x, P_INF),
    )


def _pressure(field):
    rho = field[..., 0]
    return (GAMMA - 1.0) * (
        field[..., 3] - 0.5 * (field[..., 1] ** 2 + field[..., 2] ** 2) / rho.clamp_min(1e-30)
    )


# ---------------------------------------------------------------------------
# gates on the new capability
# ---------------------------------------------------------------------------

def gate_free_stream(ncells=(24, 12), degree=2, final_time=0.05, cfl=0.2):
    """G1: a stream parallel to every embedded face must stay uniform."""
    nx, ny = ncells
    solid = torch.zeros(nx, ny, dtype=torch.bool)
    solid[:, ny // 3: ny // 3 + 2] = True  # horizontal wall strip: faces are y-normal
    _, _, ref, _ = solve_euler_dg_2d(free_stream, ncells=ncells, degree=degree,
                                     final_time=0.0)
    _, _, got, meta = solve_euler_dg_2d(
        free_stream, ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
        periodic=(False, False), solid=solid, bcs={"bottom": "wall"},
    )
    deviation = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-30)
    return {
        "gate": "G1 free-stream preservation across an embedded wall",
        "ncells": list(ncells), "degree": degree, "steps": meta["steps"],
        "relative_deviation": deviation, "tolerance": 1e-10,
        "pass": deviation < 1e-10,
    }


def gate_closed_box(ncells=(24, 24), degree=2, final_time=0.1, cfl=0.15):
    """G2: a closed box with an obstacle keeps mass and energy to round-off."""
    nx, ny = ncells
    solid = torch.zeros(nx, ny, dtype=torch.bool)
    solid[nx // 3: 2 * nx // 3, ny // 3: ny // 2] = True

    def bump(x, y):
        r = (x - 0.5) ** 2 + (y - 0.5) ** 2
        return (1.0 + 0.3 * torch.exp(-40.0 * r), torch.zeros_like(x), torch.zeros_like(x),
                1.0 + 0.5 * torch.exp(-40.0 * r))

    _, _, _, meta = solve_euler_dg_2d(
        bump, ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
        periodic=(False, False), solid=solid, track_boundary_flux=True,
        bcs={"left": "wall", "right": "wall", "bottom": "wall", "top": "wall"},
    )
    total = meta["conservation_residual_total"]
    out = {
        "gate": "G2 closed box with an interior obstacle",
        "ncells": list(ncells), "degree": degree, "steps": meta["steps"],
        "solid_cells": meta["solid_cells"],
    }
    ok = True
    for name in ("mass", "energy"):
        value = abs(total[name]["relative"])
        out[f"{name}_relative_residual"] = value
        ok = ok and value < 1e-12
    out["obstacle_impulse_x_momentum"] = meta["obstacle_impulse"]["x_momentum"]
    # momentum is *not* conserved (the obstacle pushes back); with the impulse it closes
    out["momentum_relative_residual_with_impulse"] = abs(total["y_momentum"]["relative"])
    out["tolerance"] = 1e-12
    out["pass"] = bool(ok and abs(total["y_momentum"]["relative"]) < 1e-12)
    return out


def gate_wall_reflection(ncells=(64, 1), degree=2, mach=2.0, cfl=0.12,
                         incident_mach=mach, extra_time=0.08):
    """G3: shock reflecting off an *internal* solid wall vs the exact two-shock solution."""
    nx, ny = ncells
    xmax, ymax = 1.2, 0.1
    wr = WallReflection(x_shock=0.4, x_wall=1.0, incident_mach=incident_mach)
    dx = xmax / nx
    xc = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    solid = torch.zeros(nx, ny, dtype=torch.bool)
    solid[xc > wr.x_wall] = True
    pre_rho, pre_u, pre_p = wr.pre
    post_rho, post_u, post_p = wr.post

    def initial(x, y):
        ahead = x < wr.x_zero
        return (
            torch.where(ahead, torch.full_like(x, pre_rho), torch.full_like(x, post_rho)),
            torch.where(ahead, torch.full_like(x, pre_u), torch.full_like(x, post_u)),
            torch.zeros_like(x),
            torch.where(ahead, torch.full_like(x, pre_p), torch.full_like(x, post_p)),
        )

    t_end = wr.wall_time + extra_time
    x, y, coeff, meta = solve_euler_dg_2d(
        initial, ncells=ncells, degree=degree, final_time=t_end, cfl=cfl,
        domain=((0.0, xmax), (0.0, ymax)), periodic=(False, False), solid=solid,
        track_boundary_flux=True,
        bcs={"left": "outflow", "right": "wall", "bottom": "wall", "top": "wall"},
    )
    avg = coeff[..., 0, 0, :]
    rho = avg[..., 0]
    pressure = _pressure(avg)
    exact_rho, exact_p = wr.reflected[0], wr.reflected[2]
    # The reflected plateau is the thin slab between the reflected front (which
    # runs left from the wall) and the wall itself; sample *inside* it.
    front = wr.front(t_end)
    samples = []
    for offset in (0.12, 0.3, 0.5, 0.7, 0.85):
        xx = front + offset * (wr.x_wall - front)
        if xx >= wr.x_wall:
            continue
        i = min(int(xx / dx), nx - 1)
        samples.append({
            "x": xx,
            "rho": float(rho[i, 0]), "rho_exact": exact_rho,
            "rho_error_percent": 100.0 * (float(rho[i, 0]) / exact_rho - 1.0),
            "p": float(pressure[i, 0]), "p_exact": exact_p,
            "p_error_percent": 100.0 * (float(pressure[i, 0]) / exact_p - 1.0),
        })
    worst = max(max(abs(s["rho_error_percent"]), abs(s["p_error_percent"])) for s in samples)
    return {
        "gate": "G3 shock reflection off an embedded wall (exact two-shock reference)",
        "ncells": list(ncells), "degree": degree, "steps": meta["steps"],
        "incident_mach": incident_mach, "wall_time": wr.wall_time, "final_time": t_end,
        "reflected_rho_exact": exact_rho, "reflected_p_exact": exact_p,
        "reflected_front_exact": wr.front(t_end),
        "samples": samples, "worst_relative_error_percent": worst,
        "tolerance_percent": 3.0, "pass": worst < 3.0,
        "mass_relative_residual": abs(meta["conservation_residual_total"]["mass"]["relative"]),
        "energy_relative_residual": abs(meta["conservation_residual_total"]["energy"]["relative"]),
    }


# ---------------------------------------------------------------------------
# the paper case
# ---------------------------------------------------------------------------

def run_step_case(ncells=(60, 20), degree=1, final_time=4.0, cfl=0.2, tvb_constant=1.0,
                  limiter="tvb", log=print):
    """Mach 3 forward-facing step (CKS1998 / Woodward--Colella)."""
    nx, ny = ncells
    domain = ((0.0, LENGTH), (0.0, HEIGHT))
    solid = step_mask(ncells, domain)
    inflow = conservative_state(
        torch.tensor(RHO_INF, dtype=torch.float64),
        torch.tensor(U_INF, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(P_INF, dtype=torch.float64),
    )

    def inflow_ghost(X, Y, t):
        return inflow  # conservative (rho, rho u, rho v, E), broadcast by the solver

    started = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        free_stream, ncells=ncells, degree=degree, domain=domain, final_time=final_time,
        cfl=cfl, periodic=(False, False), solid=solid, limiter=limiter,
        tvb_constant=tvb_constant, track_boundary_flux=True,
        bcs={"left": inflow_ghost, "bottom": "wall", "top": "outflow", "right": "outflow"},
    )
    elapsed = time.time() - started
    avg = coeff[..., 0, 0, :]
    rho = avg[..., 0]
    pressure = _pressure(avg)
    velocity = avg[..., 1] / rho.clamp_min(1e-30)
    xs = x.numpy()
    ys = y.numpy()

    # the inflow corner region is untouched by the shock at every t: exact gate
    upstream = (xs[:, None] < 0.5) & (ys[None, :] > 0.5)
    up_dev = float((rho[upstream] / RHO_INF - 1.0).abs().max())

    # leading shock: first x where rho exceeds 2 at mid channel height
    j = ny // 2
    column = rho[:, j]
    above = torch.nonzero(column > 2.0, as_tuple=False)
    front = float(xs[int(above[0])]) if above.numel() else float("nan")

    drag = meta["obstacle_impulse"]["x_momentum"]
    force = -drag / final_time if final_time > 0 else float("nan")
    cd = force / (0.5 * RHO_INF * U_INF ** 2 * STEP_Y)

    result = {
        "case": "CKS1998 forward-facing step (Mach 3)",
        "ncells": list(ncells), "degree": degree, "final_time": final_time,
        "cfl": cfl, "limiter": limiter, "tvb_constant": tvb_constant,
        "steps": meta["steps"], "dt_last": meta["dt"], "wall_clock_seconds": elapsed,
        "solid_cells": meta["solid_cells"],
        "upstream_free_stream_max_relative_error": up_dev,
        "rho_min": float(rho.min()), "rho_max": float(rho.max()),
        "pressure_min": float(pressure.min()), "pressure_max": float(pressure.max()),
        "leading_shock_x_at_midheight": front,
        "step_drag_force": force, "step_drag_coefficient": cd,
        "conservation": {
            name: {
                "relative": meta["conservation_residual_total"][name]["relative"],
                "obstacle_impulse": meta["obstacle_impulse"][name],
            }
            for name in ("mass", "energy")
        },
        "rho_field": rho.tolist(),
    }
    log(f"    {ncells[0]}x{ncells[1]} P{degree}: {meta['steps']} steps, "
        f"{elapsed:.0f} s, rho {float(rho.min()):.3f}..{float(rho.max()):.3f}, "
        f"front x={front:.3f}, Cd={cd:.3f}")
    return result, (xs, ys, rho, pressure)


def _figure(path, cases):
    """Density contours of the step case at every resolution (`cases` maps label->(xs,ys,rho,p))."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 2.6), squeeze=False)
    for ax, (label, (xs, ys, rho, pressure)) in zip(axes[0], cases.items()):
        art = ax.pcolormesh(xs, ys, rho.numpy().T, cmap="inferno", shading="auto")
        ax.set_title(f"density, {label}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(art, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--outdir", default="validation/forward_step")
    parser.add_argument("--final-time", type=float, default=None)
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    gates = [gate_free_stream(), gate_closed_box(), gate_wall_reflection()]
    for gate in gates:
        print(f"  {gate['gate']}: pass={gate['pass']}")

    final_time = args.final_time if args.final_time is not None else 4.0
    grids = [(60, 20)] if args.profile == "quick" else [(60, 20), (90, 30)]
    cases, fields = [], {}
    for ncells in grids:
        print(f"  running Mach 3 step on {ncells[0]}x{ncells[1]} ...", flush=True)
        result, field = run_step_case(ncells=ncells, final_time=final_time)
        cases.append(result)
        fields[f"{ncells[0]}x{ncells[1]}"] = field

    _figure(outdir / "forward_step_density.png", fields)

    payload = {
        "case": "REPRO-DG-CKS1998 forward-facing step",
        "profile": args.profile,
        "reference": "Cockburn & Shu, JCP 141 (1998) 199-224; Woodward & Colella, JCP 54 (1984)",
        "settings": {"mach": MACH, "rho_inf": RHO_INF, "p_inf": P_INF, "u_inf": U_INF,
                     "step": [STEP_X, STEP_Y], "domain": [LENGTH, HEIGHT],
                     "gamma": GAMMA},
        "gates": gates,
        "cases": cases,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    (outdir / "forward_step.json").write_text(json.dumps(payload, indent=2))
    print(f"  wrote {outdir/'forward_step.json'} and {outdir/'forward_step_density.png'}")
    ok = all(g["pass"] for g in gates)
    if not ok:
        print("  FAIL: at least one exact gate did not pass")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
