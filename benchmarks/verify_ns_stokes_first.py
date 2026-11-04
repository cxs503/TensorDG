"""Stokes' first problem (impulsively started plate) as a wall-flux verification.

Why this benchmark
------------------
Every verification case shipped with ``tensordg/ns_dg_2d.py`` before this
benchmark is either *periodic* (decaying shear wave, MMS) or a *steady*
wall-bounded state (Couette, conduction, standing shear wave).  What none of
them measures is the quantity engineering users actually want from a viscous
wall -- the **unsteady wall shear stress**.  ``docs/reports/tensordg-ns-dg-2d.md``
§5.6 therefore has to disclaim "不适用于边界层/壁面热流工程计算（壁面路径未验证）".
Stokes' first problem closes that gap with a classical closed-form reference
(Schlichting, *Boundary-Layer Theory*, §5; White, *Viscous Fluid Flow*, §3-4.1):

    u(y, t) = U erfc(eta),          eta = y / (2 sqrt(nu t))
    tau_w(t) = mu U / sqrt(pi nu t)                     (plate shear on fluid)

Both are exact for constant-property Navier--Stokes; with ``U = 0.02`` the Mach
number is 0.017 and the viscous heating ``mu U^2 / (2 kappa) = 4.1e-5`` is four
decades below the base temperature, so the analytic incompressible profile is
the correct reference for the compressible solver to O(M^2).

Three independent estimators
----------------------------
A. **profile** -- relative L2 error of ``u(y, t)`` against ``erfc`` on the DG
   quadrature nodes (tests the volume diffusion operator);
B. **wall flux** -- ``tau_h = mu du_h/dn`` taken from the same one-sided wall
   trace derivative the solver's Dirichlet viscous flux consumes
   (``_face_primitives(..., side="left")``), i.e. the *code path*;
C. **momentum budget** -- the moving plate is the only x-momentum source, so
   ``(1/Lx) sum_cells(rho u)_bar area = 2 mu U sqrt(t / (pi nu))``
   *integrated in time*.  This estimator never touches a wall closure, so it
   isolates "how much momentum the boundary condition actually delivered".

Estimator B is measured twice: on the *projected exact field* (pure spatial
consistency of the wall closure, cheap, any degree / grid) and on the *time
marched* solution (the full chain).

Run::

    PYTHONPATH=. python benchmarks/verify_ns_stokes_first.py            # both studies
    PYTHONPATH=. python benchmarks/verify_ns_stokes_first.py projection  # cheap only
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_dg_2d import legendre_operators

# --- physics ---------------------------------------------------------------
SPEED = 0.02          # plate speed U (Mach 0.017 at T = 1, gamma = 1.4)
MU = 0.05             # constant dynamic viscosity
PRANDTL = 0.71
PRESSURE = 1.0
TEMPERATURE = 1.0
RHO = PRESSURE / TEMPERATURE           # unit gas constant -> rho = 1
NU = MU / RHO                          # kinematic viscosity
KAPPA = MU * ns.CP / PRANDTL
FINAL_TIME = 0.05
LAYER = 2.0 * math.sqrt(NU * FINAL_TIME)   # viscous length 2 sqrt(nu t)
HEIGHT = 3.0 * LAYER                       # no-slip lid at 3 layer widths
WIDTH = 0.1                                # periodic, one cell, no x-variation
DTYPE = torch.float64
DEVICE = torch.device("cpu")
OUT_JSON = os.path.join("validation", "ns_dg_2d_stokes", "stokes_first.json")


def exact_solution(y, time=FINAL_TIME):
    """``u = U erfc(y / (2 sqrt(nu t)))``, uniform p and T."""
    eta = y / (2.0 * math.sqrt(NU * time))
    return SPEED * torch.erfc(eta)


def wall_shear_exact(time=FINAL_TIME, speed=SPEED, mu=MU, nu=NU):
    """Plate shear stress on the fluid, ``mu U / sqrt(pi nu t)``."""
    return mu * speed / math.sqrt(math.pi * nu * time)


def layer_length(time=FINAL_TIME, nu=NU):
    return 2.0 * math.sqrt(nu * time)


# --- set-up helpers --------------------------------------------------------
def make_boundary(speed=SPEED):
    return {
        "xmin": None,
        "xmax": None,
        "ymin": {"velocity": (speed, 0.0), "temperature": TEMPERATURE},
        "ymax": {"velocity": (0.0, 0.0), "temperature": TEMPERATURE},
    }


def rest_initial(x, y):
    zeros = torch.zeros_like(x * y)
    return (torch.full_like(x * y, RHO), zeros, zeros,
            torch.full_like(x * y, PRESSURE))


def exact_initial(time=FINAL_TIME):
    def initial(x, y):
        velocity = exact_solution(y, time)
        zeros = torch.zeros_like(y)
        return (
            torch.full_like(y, RHO),
            velocity,
            zeros * velocity,
            torch.full_like(y, PRESSURE),
        )

    return initial


def grid_operators(degree, ncells):
    """Nodes/basis plus the cell-centre coordinates of the DG mesh."""
    nx, ny = ncells
    order = max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, order, DEVICE, DTYPE
    )
    dx, dy = WIDTH / nx, HEIGHT / ny
    x = (torch.arange(nx, dtype=DTYPE) + 0.5) * dx
    y = (torch.arange(ny, dtype=DTYPE) + 0.5) * dy
    return dict(nodes=nodes, weights=weights, basis=basis, derivative=derivative,
                signs=signs, dx=dx, dy=dy, x=x, y=y, degree=degree, ncells=ncells)


def project(field, op):
    return ns._initial_coefficients(
        field, op["x"], op["y"], op["dx"], op["dy"], op["nodes"], op["weights"],
        op["basis"], DEVICE, DTYPE,
    )


def node_state(coeff, op):
    """Conservative state on the tensor Gauss nodes inside every cell."""
    basis = op["basis"]
    return torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)


def node_fields(coeff, op):
    state = node_state(coeff, op)
    rho = state[..., 0].clamp_min(1e-12)
    u = state[..., 1] / rho
    v = state[..., 2] / rho
    kinetic = 0.5 * (u * u + v * v)
    p = (ns.GAMMA - 1.0) * (state[..., 3] - rho * kinetic)
    return {"rho": rho, "u": u, "v": v, "p": p, "T": p / rho}


def node_coordinates(op):
    nodes, dx, dy = op["nodes"], op["dx"], op["dy"]
    nx, ny = op["ncells"]
    xc = (torch.arange(nx, dtype=DTYPE) + 0.5) * dx
    yc = (torch.arange(ny, dtype=DTYPE) + 0.5) * dy
    y = yc[:, None] + 0.5 * dy * nodes[None, :]          # [ny, q]
    return xc, y


def profile_error(coeff, op, time=FINAL_TIME):
    """Estimator A: relative L2 error of ``u`` against the analytic profile."""
    fields = node_fields(coeff, op)
    _, y_nodes = node_coordinates(op)
    exact = exact_solution(y_nodes, time)                # [ny, q]
    w = op["weights"]
    # state/dG fields are indexed [x cell, y cell, x node, y node]; the analytic
    # reference varies along y only, so broadcast on the fourth (y node) axis.
    ex = exact[None, :, None, :].expand_as(fields["u"])
    weight = w[None, None, :, None] * w[None, None, None, :]
    num = ((fields["u"] - ex) ** 2 * weight).sum()
    den = (ex ** 2 * weight).sum()
    return float(torch.sqrt(num / den))


def wall_shear(code_path=True, coeff=None, op=None):
    """Estimator B: ``mu du/dn`` from the one-sided DG wall trace derivative.

    ``_face_primitives(..., side="left")`` evaluates ``d/dxi`` at ``xi = -1``,
    the face of cell ``y = 0`` that lies on the moving plate; the physical
    derivative is that value times ``2/dy`` (``n = +y``).
    """
    face = ns._face_primitives(coeff, op["basis"], op["derivative"],
                               op["signs"], 1, "left")
    derivative = float(face["u_n"][:, 0, :].mean()) * (2.0 / op["dy"])
    # du/dy < 0 (fluid slower than the plate).  The fluid occupies y > 0, so the
    # outward normal of the fluid at the plate is -y and the x-traction the plate
    # exerts on the fluid is -mu du/dy = +mu U / sqrt(pi nu t) > 0.
    return -MU * derivative


def momentum_budget(coeff, op, time=FINAL_TIME):
    """Estimator C: total x-momentum per unit plate length vs the analytic integral."""
    dx, dy = op["dx"], op["dy"]
    total = float(coeff[..., 0, 0, 1].sum()) * dx * dy      # mode (0,0) = cell mean
    measured = total / dx
    expected = 2.0 * MU * SPEED * math.sqrt(time / (math.pi * NU))
    return measured, float(expected)


# --- study 1: projected exact field (spatial consistency, no time stepping) -
def projection_study(degrees=(1, 2, 3), counts=(8, 16, 32, 64, 128)):
    tau_exact = wall_shear_exact()
    rows = []
    for degree in degrees:
        for ny in counts:
            op = grid_operators(degree, (1, ny))
            coeff = project(exact_initial(), op)
            tau_h = wall_shear(coeff=coeff, op=op)
            rows.append({
                "degree": degree,
                "cells_y": ny,
                "cells_per_layer": layer_length() / op["dy"],
                "profile_l2": profile_error(coeff, op),
        "profile": dict(zip(("eta", "u_over_U"), _profile_samples(coeff, op))),
                "wall_shear": tau_h,
                "wall_shear_rel_error": (tau_h - tau_exact) / tau_exact,
            })
            print("  P%d ny=%3d  h/layer=%.3f  profile L2=%.3e  tau rel err=%+8.3f%%"
                  % (degree, ny, op["dy"] / layer_length(), rows[-1]["profile_l2"],
                     100.0 * rows[-1]["wall_shear_rel_error"]), flush=True)
    for degree in degrees:
        block = [r for r in rows if r["degree"] == degree]
        coarse, fine = block[0], block[-1]
        detail = math.log2(abs(coarse["wall_shear_rel_error"]
                               / fine["wall_shear_rel_error"]))
        span = math.log2(fine["cells_y"] / coarse["cells_y"])
        block_order = detail / span
        profile_order = math.log2(coarse["profile_l2"] / fine["profile_l2"]) / span
        for row in block:
            row["wall_shear_order"] = block_order
            row["profile_order"] = profile_order
    return rows


# --- study 2: time marched solution (the full chain) -----------------------
def run_case(ny, degree=1, cfl_viscous=1.0, cfl=0.3, wall_gradient="face"):
    op = grid_operators(degree, (1, ny))
    start = time.time()
    x, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
        rest_initial,
        ncells=(1, ny),
        degree=degree,
        domain=((0.0, WIDTH), (0.0, HEIGHT)),
        final_time=FINAL_TIME,
        cfl=cfl,
        cfl_viscous=cfl_viscous,
        mu=MU,
        prandtl=PRANDTL,
        periodic=(True, False),
        boundary=make_boundary(),
        wall_gradient=wall_gradient,
        device=DEVICE,
        dtype=DTYPE,
    )
    elapsed = time.time() - start
    tau_h = wall_shear(coeff=coeff, op=op)
    measured, expected = momentum_budget(coeff, op)
    case = {
        "degree": degree,
        "cells_y": ny,
        "cfl_viscous": cfl_viscous,
        "wall_gradient": wall_gradient,
        "cells_per_layer": layer_length() / op["dy"],
        "steps": meta["steps"],
        "dt": meta["dt"],
        "diffusive_steps": meta["diffusive_steps"],
        "seconds": elapsed,
        "profile_l2": profile_error(coeff, op),
        "wall_shear": tau_h,
        "wall_shear_rel_error": (tau_h - wall_shear_exact()) / wall_shear_exact(),
        "profile": dict(zip(("eta", "u_over_U"), _profile_samples(coeff, op))),
        "momentum_measured": measured,
        "momentum_expected": expected,
        "momentum_rel_error": (measured - expected) / expected,
        "final_time_reached": meta["final_time_reached"],
    }
    print("  P%d ny=%3d cv=%.1f  steps=%5d  %6.1fs  profile L2=%.3e  tau=%+8.3f%%  M=%+8.3f%%"
          % (degree, ny, cfl_viscous, meta["steps"], elapsed, case["profile_l2"],
             100.0 * case["wall_shear_rel_error"], 100.0 * case["momentum_rel_error"]),
          flush=True)
    return case


def transient_study(cases):
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(4, len(cases))) as pool:
        futures = [pool.submit(run_case, **case) for case in cases]
        return [f.result() for f in futures]


def _run_case_star(kwargs):
    torch.set_num_threads(1)
    return run_case(**kwargs)


# --- figures ---------------------------------------------------------------
ANALYTIC_COLOR = "k"


def _profile_samples(coeff, op, time=FINAL_TIME):
    """``(eta, u/U)`` on the cell Gauss nodes, flattened, for plotting."""
    fields = node_fields(coeff, op)
    _, y_nodes = node_coordinates(op)
    u = fields["u"][0, :, 0, :].reshape(-1)              # one cell in x
    y = y_nodes.reshape(-1)
    order = torch.argsort(y)
    scale = layer_length(time)
    return (y[order] / scale).tolist(), (u[order] / SPEED).tolist()


def figures(projection, transient, directory="docs/figures"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(directory, exist_ok=True)
    written = []

    # (a) similarity profiles + (b) wall-shear convergence
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    ax = axes[0]
    eta = np.linspace(0.0, 3.4, 500)
    ax.plot(eta, [math.erfc(e) for e in eta], "-", color=ANALYTIC_COLOR, lw=2.0,
            label=r"analytic $\mathrm{erfc}(\eta)$")
    colours = ["C0", "C1", "C2", "C3", "C4"]
    for colour, case in zip(colours, transient):
        sample = case.get("profile")
        if not sample:
            continue
        marker = "os^d*"[min(case["degree"], 3)]
        ax.plot(sample["eta"], sample["u_over_U"], marker, ms=4.5, mfc="none",
                color=colour,
                label="P%d $n_y=%d$ (%+.1f%%)" % (case["degree"], case["cells_y"],
                                                  100.0 * case["wall_shear_rel_error"]))
    ax.set_xlabel(r"$\eta = y/(2\sqrt{\nu t})$")
    ax.set_ylabel(r"$u/U$")
    ax.set_title("Stokes layer at $t=%.2f$ (markers = DG nodes)" % FINAL_TIME)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for index, degree in enumerate(sorted({r["degree"] for r in projection})):
        block = [r for r in projection if r["degree"] == degree]
        ax.loglog([r["cells_y"] for r in block],
                  [abs(r["wall_shear_rel_error"]) for r in block], "o-",
                  color=colours[index],
                  label="projected exact field, P%d (order %.2f)"
                  % (degree, block[0]["wall_shear_order"]))
    if transient:
        ax.loglog([c["cells_y"] for c in transient],
                  [abs(c["wall_shear_rel_error"]) for c in transient], "k*",
                  ms=10, ls="none", label="time marched P1/P2")
    ax.axhline(0.03, color="r", ls="--", lw=1.0, label="3% gate")
    ax.set_xlabel("cells across the layer height ($n_y$)")
    ax.set_ylabel(r"$|\tau_h-\tau_{\rm exact}|/\tau_{\rm exact}$")
    ax.set_title("Wall shear stress (unsteady) vs $\\mu U/\\sqrt{\\pi\\nu t}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    path = os.path.join(directory, "ns_dg_2d_stokes_first.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # (c) momentum budget of the time marched solutions
    if transient:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ny = [c["cells_y"] for c in transient]
        ax.plot(ny, [100.0 * c["momentum_rel_error"] for c in transient], "o-",
                label="marched momentum vs analytic")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axhspan(-3.0, 3.0, color="green", alpha=0.12, label="$\\pm$3% gate")
        ax.set_xscale("log")
        ax.set_xlabel("$n_y$")
        ax.set_ylabel(r"error of $\int\rho u\,dA$ [%]")
        ax.set_title("x-momentum delivered by the moving plate")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(directory, "ns_dg_2d_stokes_momentum.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main(argv):
    mode = argv[1] if len(argv) > 1 else "all"
    payload = {
        "mission": 3614,
        "benchmark": "stokes first problem (impulsively started plate)",
        "reference": "Schlichting, Boundary-Layer Theory, ch. 5; White, Viscous Fluid Flow, 3-4.1",
        "parameters": {
            "speed_U": SPEED, "mu": MU, "prandtl": PRANDTL, "rho": RHO,
            "nu": NU, "kappa": KAPPA, "final_time": FINAL_TIME,
            "layer_length": layer_length(), "height": HEIGHT, "width": WIDTH,
            "mach": SPEED / math.sqrt(ns.GAMMA * PRESSURE / RHO),
            "viscous_heating_kappaT": MU * SPEED ** 2 / (2.0 * KAPPA),
            "wall_shear_exact": wall_shear_exact(),
        },
        "torch": torch.__version__,
        "numpy_erfc_available": hasattr(math, "erfc"),
    }
    if os.path.exists(OUT_JSON):
        # studies are independent; running one mode must not delete the other, and
        # the figures are always drawn from the merged record.
        with open(OUT_JSON) as handle:
            previous = json.load(handle)
        previous.update(payload)
        payload = previous
    if mode in ("all", "projection"):
        print("projection study (spatial consistency of the wall closure)")
        payload["projection"] = projection_study()
    if mode in ("all", "transient"):
        print("time marched study")
        cases = [
            {"ny": 8, "degree": 1, "cfl_viscous": 1.0},
            {"ny": 8, "degree": 1, "cfl_viscous": 2.0},
            {"ny": 16, "degree": 1, "cfl_viscous": 2.0},
            {"ny": 32, "degree": 1, "cfl_viscous": 2.0},
            {"ny": 8, "degree": 2, "cfl_viscous": 2.0},
        ]
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=4) as pool:
            payload["transient"] = list(pool.map(_run_case_star, cases))
    payload["figures"] = figures(payload.get("projection", []),
                                 payload.get("transient", []))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))