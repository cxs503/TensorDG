"""Isothermal-wall closure of ``tensordg.ns_dg_2d``: consistency, order, stability.

Three probes, all programmatic (numbers -> ``validation/ns_heat_wall_closure/evidence.json``):

E1  exactly-representable state (T = p/rho = 1, no motion, both walls at T = 1):
    a consistent closure must give a machine-zero residual, walls included.
E2  exact pure-conduction state ``T = 1 - 0.5 y``, ``p = 1``, ``rho = p/T`` L2
    projected: wall heat flux error against ``kappa dT/dn`` for a mesh sequence.
E3  time stepping from the exact steady state and from the shipped uniform
    initial condition: how long does ``solve_navier_stokes_dg_2d`` survive, and
    does raising the Dirichlet penalty (the usual SIPG cure) help?

Run: ``PYTHONPATH=. python -m tools.verify_ns_heat_wall_closure``
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensordg import ns_dg_2d as ns
from tensordg.euler_dg_2d import _initial_coefficients, legendre_operators

MU, PR = 0.1, 0.71
KAPPA = MU * ns.CP / PR
P0 = 1.0
T_BOT, T_TOP = 1.0, 0.5
OUT = Path("validation/ns_heat_wall_closure")
EXACT_FLUX = KAPPA * (T_TOP - T_BOT)  # kappa dT/dy, constant


def _full(x, y):
    """Broadcast to the full (nx, ny, a, b) quadrature grid.

    ``_initial_coefficients`` builds its cell array from the *shape* of what the
    initial-condition callback returns, so a profile that only depends on y
    silently collapses the x axis (and vice versa).  Adding ``0 * x`` forces the
    full tensor.
    """
    return x + torch.zeros_like(y), y + torch.zeros_like(x)


def uniform_state(xq, yq):
    x, y = _full(xq, yq)
    one = torch.ones_like(x)
    return one, torch.zeros_like(x), torch.zeros_like(x), one.clone()


def steady_state(xq, yq):
    x, y = _full(xq, yq)
    T = T_BOT + (T_TOP - T_BOT) * y
    return P0 / T, torch.zeros_like(T), torch.zeros_like(T), torch.full_like(T, P0)


def operators(degree, ncells, nx=2):
    dtype = torch.float64
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, max(degree + 2, 3), "cpu", dtype)
    dx, dy = 1.0 / nx, 1.0 / ncells
    xc = (torch.arange(nx, dtype=dtype) + 0.5) * dx
    yc = (torch.arange(ncells, dtype=dtype) + 0.5) * dy
    return nodes, weights, basis, derivative, signs, xc, yc, dx, dy


def boundary_of(t_bot, t_top):
    return {"xmin": None, "xmax": None,
            "ymin": {"velocity": (0.0, 0.0), "temperature": t_bot},
            "ymax": {"velocity": (0.0, 0.0), "temperature": t_top}}


def probe_consistency(ncells=16, degree=1):
    """E1: residual of the exactly representable uniform state."""
    nodes, weights, basis, derivative, signs, xc, yc, dx, dy = operators(degree, ncells)
    coeff = _initial_coefficients(uniform_state, xc, yc, dx, dy, nodes, weights, basis,
                                  "cpu", torch.float64)
    out = {}
    for grad in ("face", "taylor"):
        resid = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy, (True, False),
                           boundary_of(1.0, 1.0), MU, KAPPA, 2.0 * (degree + 1.0) ** 2, grad)
        out[grad] = float(resid.abs().max())
    return out


def probe_flux_order(ncells, degree=1, grad="face"):
    """E2: wall heat flux produced by the closure vs the exact kappa dT/dy."""
    nodes, weights, basis, derivative, signs, xc, yc, dx, dy = operators(degree, ncells)
    coeff = _initial_coefficients(steady_state, xc, yc, dx, dy, nodes, weights, basis,
                                  "cpu", torch.float64)
    penalty = 2.0 * (degree + 1.0) ** 2
    boundary = boundary_of(T_BOT, T_TOP)
    resid = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy, (True, False),
                       boundary, MU, KAPPA, penalty, grad)
    amps = resid.abs().amax(dim=(2, 3))  # (nx, ny, 4)
    row = {
        "ncells": ncells, "degree": degree, "wall_gradient": grad,
        "energy_residual_bottom": float(amps[:, 0, 3].max()),
        "energy_residual_interior": float(amps[:, 1:-1, 3].max()) if ncells > 2 else float("nan"),
        "energy_residual_top": float(amps[:, -1, 3].max()),
        "mass_residual": float(amps[..., 0].max()),
        "momentum_residual": float(amps[..., 1:3].max()),
    }
    right = ns._face_primitives(coeff, basis, derivative, signs, 1, "right")
    left = ns._face_primitives(coeff, basis, derivative, signs, 1, "left")
    for label, side, wall, index in (("bottom", left, boundary["ymin"], 0),
                                     ("top", right, boundary["ymax"], -1)):
        flux = ns._boundary_flux(side, wall, 1, dy, MU, KAPPA, penalty, 2.0 / dy, 2.0 / dx)
        key = [slice(None)] * flux.dim()
        key[1] = index
        energy = flux[tuple(key)][..., 3]
        # the closure returns F_inv - F_visc, i.e. minus the physical kappa dT/dy
        row[label + "_flux"] = float(energy.mean())
        row[label + "_flux_err"] = float((energy + EXACT_FLUX).abs().max())
        row[label + "_t_trace_err"] = float(
            (side["temperature"][:, index] - wall["temperature"]).abs().max())
    return row


def probe_stability(ncells, degree=1, t_top=T_TOP, penalty=None, initial=steady_state,
                    final_time=0.05, cfl=0.3, grad="face", max_steps=400):
    """E3: explicit SSP-RK3 advance of the module residual; report survival time."""
    nodes, weights, basis, derivative, signs, xc, yc, dx, dy = operators(degree, ncells)
    coeff = _initial_coefficients(initial, xc, yc, dx, dy, nodes, weights, basis,
                                  "cpu", torch.float64)
    if penalty is None:
        penalty = 2.0 * (degree + 1.0) ** 2
    boundary = boundary_of(T_BOT, t_top)
    h_min = min(dx, dy)

    def rhs(c):
        return ns._rhs_ns(c, basis, derivative, weights, signs, dx, dy, (True, False),
                          boundary, MU, KAPPA, penalty, grad)

    time, steps = 0.0, 0
    status, detail = "survived", 0.0
    while time < final_time and steps < max_steps:
        at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
        rho_min = float(at_quad[..., 0].min())
        diffusivity = max((4.0 / 3.0) * MU / rho_min, KAPPA / (rho_min * ns.CV))
        rate = diffusivity * (degree + 1.0) ** 4 / (ns.VISCOUS_STABILITY * h_min ** 2)
        dt = min(cfl / rate, final_time - time)
        k1 = rhs(coeff)
        u1 = coeff + dt * k1
        k2 = rhs(u1)
        u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
        k3 = rhs(u2)
        coeff = ns._positivity_filter(coeff / 3.0 + (2.0 / 3.0) * (u2 + dt * k3), basis, signs)
        time += dt
        steps += 1
        p_avg = (ns.GAMMA - 1.0) * (
            coeff[..., 0, 0, 3] - 0.5 * (coeff[..., 0, 0, 1] ** 2 + coeff[..., 0, 0, 2] ** 2)
            / coeff[..., 0, 0, 0].clamp_min(1e-12))
        if not torch.isfinite(coeff).all() or float(p_avg.min()) <= 0.0:
            status, detail = "crashed", time
            break
    return {"ncells": (2, ncells), "degree": degree, "t_top": t_top, "penalty": penalty,
            "initial": getattr(initial, "__name__", str(initial)), "wall_gradient": grad,
            "status": status, "time": time, "steps": steps,
            "p_avg_min": float(p_avg.min()), "p_avg_max": float(p_avg.max())}


def figure(evidence, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    rows = evidence["flux_order"]
    for grad, marker in (("face", "o"), ("taylor", "s")):
        sub = [r for r in rows if r["wall_gradient"] == grad]
        h = [1.0 / r["ncells"] for r in sub]
        axes[0].loglog(h, [r["bottom_flux_err"] for r in sub], marker + "-", label=grad + " bottom")
        axes[0].loglog(h, [r["top_flux_err"] for r in sub], marker + "--", label=grad + " top")
    axes[0].set_xlabel("h = 1/ny")
    axes[0].set_ylabel("|wall heat flux - exact|")
    axes[0].set_title("E2 wall heat-flux error (P1)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    sub = [r for r in rows if r["wall_gradient"] == "face"]
    h = [1.0 / r["ncells"] for r in sub]
    axes[1].loglog(h, [r["energy_residual_bottom"] for r in sub], "o-", label="bottom wall cell")
    axes[1].loglog(h, [r["energy_residual_top"] for r in sub], "s-", label="top wall cell")
    axes[1].loglog(h, [r["energy_residual_interior"] for r in sub], "^-", label="interior cells")
    axes[1].set_xlabel("h = 1/ny")
    axes[1].set_ylabel("max |energy residual| (modal)")
    axes[1].set_title("E2 residual: wall rows do not decay")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    labels, values = [], []
    for r in evidence["stability"]:
        labels.append("ny=%d T_top=%.2f s=%.0f\n%s" % (r["ncells"][1], r["t_top"],
                                                       r["penalty"], r["initial"][:6]))
        values.append(r["time"] if r["status"] == "crashed" else -r["time"])
    colors = ["tab:red" if v > 0 else "tab:green" for v in values]
    axes[2].bar(range(len(values)), [abs(v) for v in values], color=colors)
    axes[2].set_xticks(range(len(values)))
    axes[2].set_xticklabels(labels, fontsize=6)
    axes[2].set_ylabel("time reached (red = crashed)")
    axes[2].set_title("E3 survival time")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {
        "module": "tensordg/ns_dg_2d.py",
        "mu": MU, "prandtl": PR, "kappa": KAPPA,
        "exact_wall_heat_flux_kappa_dTdy": EXACT_FLUX,
        "consistency_uniform": [
            dict(ncells=n, **probe_consistency(n)) for n in (8, 16, 32)
        ],
        "flux_order": [],
    }
    for grad in ("face", "taylor"):
        for n in (16, 32, 64):
            evidence["flux_order"].append(probe_flux_order(n, 1, grad))
    evidence["stability"] = [
        probe_stability(8, 1, T_TOP, None, steady_state, 0.06),
        probe_stability(8, 1, T_TOP, None, uniform_state, 0.06),
        probe_stability(8, 1, 0.95, None, uniform_state, 0.06),
        probe_stability(8, 1, T_TOP, 16.0, steady_state, 0.06),
        probe_stability(8, 1, T_TOP, 32.0, steady_state, 0.06),
        probe_stability(4, 1, T_TOP, None, steady_state, 0.06),
    ]
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=1))
    figure(evidence, "docs/figures/ns_heat_wall_closure.png")
    print(json.dumps(evidence, indent=1))


if __name__ == "__main__":
    main()