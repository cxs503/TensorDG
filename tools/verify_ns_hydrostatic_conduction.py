"""Representable hydrostatic-conduction equilibrium of ``tensordg.ns_dg_2d``.

Why this probe exists
---------------------
``docs/reports/tensordg-ns-heat-wall-closure.md`` (m3257) measured the
isothermal-wall closure on the textbook steady state

    u = v = 0,  p = p0 (const),  T(y) = T_bot + (T_top - T_bot) y,  rho = p / T

and found ``(a)`` the wall energy residual of that state floors at ``O(1)`` and
does not decay, and ``(b)`` the wall heat flux is only first order.  Both
measurements are taken on the **L2 projection** of that state.  Because
``rho = p/T`` is a rational function it is *not* an element of the DG space, so
the projected state is not a discrete steady state -- a projection artefact can
therefore masquerade as a closure defect.

This probe removes the ambiguity with a state that *is* in the space, so that a
consistent closure has no excuse to leave a residual:

    u = v = 0,  T(y) = 1 - y/2,
    rho(y) = 1 + a y^2,            (a = 0 -> P0, a = 0.5 -> P2)
    p(y)   = rho(y) T(y),          (P1 for a = 0, P3 for a = 0.5)
    S      = (0, 0, dp/dy, 0)      conserved-variable momentum source

With the constant source ``S`` this is an exact steady solution of the
compressible Navier--Stokes equations *and* it lies inside the
``degree >= 3`` tensor-product modal space (``degree >= 1`` for ``a = 0``).
Physically it is hydrostatic equilibrium in a constant heat flux (a stably
stratified layer): the source is the pressure-gradient support, the energy
equation degenerates to ``kappa d^2 T / dy^2 = 0`` and the wall heat flux
``kappa dT/dy = -kappa/2 != 0`` is the quantity the old report called first
order.

Probes
------
E1  residual of the representable state, split into bottom-wall / interior /
    top-wall rows, degrees 1/2/3, ``ny = 8, 16, 32`` (must be round-off);
E2  wall energy flux of the same state against the exact ``-kappa/2`` on both
    walls (must be round-off), i.e. the closure reproduces a *nonzero* heat
    flux exactly when the state is representable;
E3  the textbook state of m3257 (rational ``rho = p/T``): wall temperature
    trace defect ``|T^- - T_D|``, wall heat-flux error and residual floor for
    ``ny = 8 .. 64``, with the causality ratio ``residual / ((sigma/h) delta)``
    that identifies the residual floor as the penalty term acting on the
    projection defect.

Run ``PYTHONPATH=. python -m tools.verify_ns_hydrostatic_conduction``
-> ``validation/ns_hydrostatic/evidence.json`` + ``docs/figures/ns_hydrostatic_conduction.png``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensordg import ns_dg_2d as ns
from tensordg.euler_dg_2d import _initial_coefficients, legendre_operators

MU, PR = 0.1, 0.71
KAPPA = MU * ns.CP / PR
T_BOT, T_TOP = 1.0, 0.5
EXACT_FLUX = KAPPA * (T_TOP - T_BOT)      # kappa dT/dy < 0, constant
OUT = Path("validation/ns_hydrostatic")
FIG = Path("docs/figures/ns_hydrostatic_conduction.png")


# ---------------------------------------------------------------------------
# States, source, operators
# ---------------------------------------------------------------------------
def _full(x, y):
    """Broadcast a y-only profile to the full (nx, ny, a, b) quadrature grid."""
    return x + 0.0 * y, y + 0.0 * x


def hydrostatic_state(a):
    """Exactly representable hydrostatic state with a constant heat flux."""

    def state(xq, yq):
        x, y = _full(xq, yq)
        temperature = T_BOT + (T_TOP - T_BOT) * y
        density = 1.0 + a * y * y
        pressure = density * temperature
        return density, torch.zeros_like(x), torch.zeros_like(x), pressure

    return state


def hydrostatic_source(a):
    """Conserved-variable source ``(0, 0, dp/dy, 0)`` for ``p = (1 + a y^2) T``."""

    def source(xq, yq, t=0.0):
        x, y = _full(xq, yq)
        dpdy = (T_TOP - T_BOT) * (1.0 + a * y * y) + (1.0 - 0.5 * y) * (2.0 * a * y)
        z = torch.zeros_like(x)
        return torch.stack([z, z, dpdy, z], dim=-1)

    return source


def textbook_state(xq, yq):
    """m3257 state: ``p = p0``, ``rho = p0 / T`` (rational -> not representable)."""
    x, y = _full(xq, yq)
    temperature = T_BOT + (T_TOP - T_BOT) * y
    return (1.0 / temperature, torch.zeros_like(x), torch.zeros_like(x),
            torch.ones_like(temperature))


def operators(degree, ny, nx=2):
    dtype = torch.float64
    order = max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, order, "cpu", dtype)
    dx, dy = 1.0 / nx, 1.0 / ny
    xc = (torch.arange(nx, dtype=dtype) + 0.5) * dx
    yc = (torch.arange(ny, dtype=dtype) + 0.5) * dy
    return nodes, weights, basis, derivative, signs, xc, yc, dx, dy


def _source_quad(source, nodes, nx, ny, dx, dy, degree):
    if source is None:
        return None
    order = max(degree + 2, 3)
    dtype = torch.float64
    xi = nodes.to(dtype)
    xq = ((torch.arange(nx, dtype=dtype).view(nx, 1, 1, 1)
           + 0.5 * (xi.view(1, 1, -1, 1) + 1.0)) * dx).expand(nx, ny, order, order)
    yq = ((torch.arange(ny, dtype=dtype).view(1, ny, 1, 1)
           + 0.5 * (xi.view(1, 1, 1, -1) + 1.0)) * dy).expand(nx, ny, order, order)
    return source(xq, yq, 0.0)


def boundary_of():
    return {"xmin": None, "xmax": None,
            "ymin": {"velocity": (0.0, 0.0), "temperature": T_BOT},
            "ymax": {"velocity": (0.0, 0.0), "temperature": T_TOP}}


def build(init, degree, ny, source=None, nx=2):
    nodes, weights, basis, derivative, signs, xc, yc, dx, dy = operators(degree, ny, nx)
    coeff = _initial_coefficients(init, xc, yc, dx, dy, nodes, weights, basis,
                                  "cpu", torch.float64)
    ops = (nodes, weights, basis, derivative, signs, xc, yc, dx, dy)
    return coeff, ops, _source_quad(source, nodes, nx, ny, dx, dy, degree)


def residual(coeff, ops, degree, source_quad=None, wall_gradient="face"):
    nodes, weights, basis, derivative, signs, xc, yc, dx, dy = ops
    penalty = 2.0 * (degree + 1.0) ** 2
    return ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy, (True, False),
                      boundary_of(), MU, KAPPA, penalty, wall_gradient, None, source_quad)


def wall_energy_flux(coeff, ops, degree):
    """Return ``(bottom, top)`` energy fluxes of the wall closure (all faces)."""
    nodes, weights, basis, derivative, signs, xc, yc, dx, dy = ops
    penalty = 2.0 * (degree + 1.0) ** 2
    right = ns._face_primitives(coeff, basis, derivative, signs, 1, "right")
    left = ns._face_primitives(coeff, basis, derivative, signs, 1, "left")
    top = ns._boundary_flux(right, boundary_of()["ymax"], 1, dy, MU, KAPPA, penalty,
                            2.0 / dy, 2.0 / dx)
    bottom = ns._boundary_flux(left, boundary_of()["ymin"], 1, dy, MU, KAPPA, penalty,
                               2.0 / dy, 2.0 / dx)
    # the closure returns F_inv - F_visc, so it is -kappa dT/dy > 0 here
    return bottom[:, 0, :, 3], top[:, -1, :, 3]


def trace_defect(coeff, ops):
    """Wall temperature trace defect ``|T^- - T_D|`` (max over faces)."""
    nodes, weights, basis, derivative, signs, xc, yc, dx, dy = ops
    right = ns._face_primitives(coeff, basis, derivative, signs, 1, "right")
    left = ns._face_primitives(coeff, basis, derivative, signs, 1, "left")
    top = float((right["temperature"][:, -1] - T_TOP).abs().max())
    bottom = float((left["temperature"][:, 0] - T_BOT).abs().max())
    return bottom, top


def rows(resid):
    """Max modal residual in bottom-wall / interior / top-wall rows, by equation."""
    amps = resid.abs()
    return {
        "bottom_row": float(amps[:, -1].max()),
        "interior": float(amps[:, 1:-1].max()) if amps.shape[1] > 2 else 0.0,
        "top_row": float(amps[:, 0].max()),
        "energy_bottom_row": float(amps[:, -1, ..., 3].max()),
        "energy_top_row": float(amps[:, 0, ..., 3].max()),
        "mass": float(amps[..., 0].max()),
        "momentum": float(amps[..., 1:3].max()),
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def probe_representable(degrees=(1, 2, 3), resolutions=(8, 16, 32)):
    out = []
    for degree, a in ((1, 0.0), (2, 0.0), (3, 0.0), (3, 0.5)):
        if degree not in degrees:
            continue
        for ny in resolutions:
            if a > 0.0 and degree < 3:
                continue
            coeff, ops, sq = build(hydrostatic_state(a), degree, ny,
                                   hydrostatic_source(a))
            resid = residual(coeff, ops, degree, sq)
            bottom, top = wall_energy_flux(coeff, ops, degree)
            out.append({
                "degree": degree, "density_amplitude": a, "ny": ny,
                "state_is_representable": True,
                "residual": rows(resid),
                "wall_flux_bottom": float(bottom.mean()),
                "wall_flux_top": float(top.mean()),
                "wall_flux_error": float(max((bottom + EXACT_FLUX).abs().max(),
                                             (top + EXACT_FLUX).abs().max())),
                "wall_flux_exact": EXACT_FLUX,
            })
    return out


def probe_textbook(resolutions=(8, 16, 32, 64), degree=1):
    out = []
    for ny in resolutions:
        coeff, ops, _ = build(textbook_state, degree, ny)
        resid = residual(coeff, ops, degree)
        bottom, top = wall_energy_flux(coeff, ops, degree)
        defect_bottom, defect_top = trace_defect(coeff, ops)
        penalty = 2.0 * (degree + 1.0) ** 2
        r = rows(resid)
        sigma_over_h = penalty * ny
        out.append({
            "degree": degree, "ny": ny, "state_is_representable": False,
            "wall_flux_error_bottom": float((bottom + EXACT_FLUX).abs().max()),
            "wall_flux_error_top": float((top + EXACT_FLUX).abs().max()),
            "trace_defect_bottom": defect_bottom,
            "trace_defect_top": defect_top,
            "energy_residual_bottom_row": r["energy_bottom_row"],
            "energy_residual_top_row": r["energy_top_row"],
            # if the residual floor is the penalty term acting on the projection
            # defect this ratio is resolution independent
            "causality_ratio_bottom": r["energy_bottom_row"] / (sigma_over_h * defect_bottom),
            "causality_ratio_top": r["energy_top_row"] / (sigma_over_h * defect_top),
            # wall heat-flux error over the penalty term (sigma/h) * delta: this
            # is the decisive test -- if the first-order flux error is the
            # penalty acting on the projection defect, the ratio is constant
            "flux_ratio_bottom": (float((bottom + EXACT_FLUX).abs().max())
                                  / (sigma_over_h * defect_bottom)),
            "flux_ratio_top": (float((top + EXACT_FLUX).abs().max())
                               / (sigma_over_h * defect_top)),
            "residual_scaled_bottom": (r["energy_bottom_row"] / ny
                                       / float((bottom + EXACT_FLUX).abs().max())),
            "n_dofs": 2 * ny * (degree + 1) ** 2 * 4,
            "mass_residual": r["mass"], "momentum_residual": r["momentum"],
        })
    return out


def observed_order(values, resolutions):
    xs = [math.log(n) for n in resolutions]
    ys = [math.log(v) for v in values]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return -num / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT / "evidence.json"))
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    representable = probe_representable()
    textbook = probe_textbook()

    resolutions = [row["ny"] for row in textbook]
    summary = {
        "representable_max_residual": max(
            max(row["residual"]["bottom_row"], row["residual"]["interior"],
                row["residual"]["top_row"]) for row in representable),
        "representable_max_wall_flux_error": max(
            row["wall_flux_error"] for row in representable),
        "textbook_trace_defect_order": observed_order(
            [row["trace_defect_bottom"] for row in textbook], resolutions),
        "textbook_flux_error_order": observed_order(
            [row["wall_flux_error_bottom"] for row in textbook], resolutions),
        "textbook_residual_order": observed_order(
            [row["energy_residual_bottom_row"] for row in textbook], resolutions),
        "flux_ratio_spread": (
            max(row["flux_ratio_bottom"] for row in textbook)
            / min(row["flux_ratio_bottom"] for row in textbook)),
        "causality_ratio_spread": (
            max(row["causality_ratio_bottom"] for row in textbook)
            / min(row["causality_ratio_bottom"] for row in textbook)),
    }
    payload = {
        "probe": "ns-hydrostatic-representable-conduction",
        "parameters": {"mu": MU, "prandtl": PR, "kappa": KAPPA, "t_bottom": T_BOT,
                       "t_top": T_TOP, "exact_wall_flux": EXACT_FLUX,
                       "domain": [0.0, 1.0], "x_periodic": True},
        "representable": representable,
        "textbook": textbook,
        "summary": summary,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.no_figure:
        make_figure(representable, textbook, resolutions)


def make_figure(representable, textbook, resolutions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    ny = [row["ny"] for row in textbook]
    ax.loglog(ny, [row["trace_defect_bottom"] for row in textbook], "o-",
              label=r"wall trace defect $|T^- - T_D|$")
    ax.loglog(ny, [row["wall_flux_error_bottom"] for row in textbook], "s-",
              label=r"wall heat-flux error vs $\kappa\,dT/dy$")
    ax.loglog(ny, [row["energy_residual_bottom_row"] for row in textbook], "^-",
              label="wall energy residual floor")
    ax.set_xlabel("cells in y")
    ax.set_ylabel("magnitude")
    ax.set_title(r"textbook state $\rho = p_0/T$ (not representable)")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=7)

    ax = axes[1]
    degree_labels, residuals, flux_errors = [], [], []
    for row in representable:
        label = f"p={row['degree']}" + ("" if row["density_amplitude"] == 0 else " (rho=P2)")
        degree_labels.append(f"{label}\nny={row['ny']}")
        residuals.append(max(row["residual"]["bottom_row"], row["residual"]["top_row"]))
        flux_errors.append(row["wall_flux_error"])
    idx = range(len(degree_labels))
    ax.semilogy(list(idx), residuals, "o-", label="wall energy residual")
    ax.semilogy(list(idx), [max(v, 1e-17) for v in flux_errors], "s-",
                label="wall heat-flux error")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(degree_labels, fontsize=6, rotation=90)
    ax.set_ylabel("magnitude (round-off)")
    ax.set_title("representable hydrostatic state: closure is exact")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=7)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"figure -> {FIG}")


if __name__ == "__main__":
    main()