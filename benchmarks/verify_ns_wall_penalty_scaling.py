"""Penalty scaling law and consistency control for the Dirichlet wall of ``ns_dg_2d``.

Why this exists
---------------
Three earlier units measured a large (``O(1)``) residual of the *exact analytic*
steady Couette state on the Dirichlet wall path of ``tensordg/ns_dg_2d.py`` and
read it as an inconsistent discretisation:

* m2496 reported ``energy_residual / wall_flux = 4.07`` for the static channel;
* m2589 measured ``rhs_max = 0.3045`` on ``2 x {8,16,32}`` (P1) and blamed the
  one-sided wall gradient;
* m2786 repeated the measurement at P1/P2/P3 and reported the same *constant*;
* m2879 attributed the constant to the SIPG penalty term, but never separated
  the penalty contribution from the representation error of the projected state,
  and never ran the control that decides the question: a state that the
  polynomial space represents **exactly** must have a machine-zero residual.

This tool runs that control and the scaling law.  Two independent mechanisms
produce every number the earlier reports recorded:

``consistency``  state ``(rho, u, v, p) = (1, y, 0, 1)``.  It is an exact
                 solution of the inviscid Euler equations and, at P2/P3, the
                 conservative state is representable in the space (the energy is
                 quadratic in ``y``).  Consequence: mass / x / y momentum
                 residuals must vanish to round-off and the energy residual must
                 equal the physical dissipation ``mu U^2``.  This isolates the
                 wall closure from the projection.
``scaling``      residual of the exact analytic Couette state vs the SIPG
                 penalty: ``R(sigma) = R(0) + k sigma`` with ``k = J/h`` where
                 ``J`` is the wall trace error of the projected state.  A linear
                 law with ``R^2 ~ 1`` for every degree proves the earlier
                 "O(1) residual" is a *representation* effect amplified by the
                 penalty, not a defect of the wall operator.
``onset``        time at which the wall-bounded transient goes non-physical as a
                 function of the penalty, starting from the representable
                 ``(1, y, 0, 1)`` state.  ``t_fail ~ 1/sigma`` (m2589's
                 observation) reproduced on a start that has *no* start-up
                 discontinuity, i.e. the instability is a property of the wall
                 operator at the default closure, not of the impulsive start.

Outputs: ``validation/ns_dg_2d_wall/penalty_scaling.json`` and
``docs/figures/ns_wall_penalty_scaling.png``.  No number in the report is typed
by hand; every table is rendered from the JSON.

Usage::

    PYTHONPATH=. python -m benchmarks.verify_ns_wall_penalty_scaling
    PYTHONPATH=. python -m benchmarks.verify_ns_wall_penalty_scaling consistency scaling
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_dg_2d import _positivity_filter, _primitive, legendre_operators
from tools.diag_ns_wall import (
    KAPPA,
    MU,
    PRESSURE,
    WALL_SPEED,
    WALL_T,
    build_state,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "ns_dg_2d_wall" / "penalty_scaling.json"
FIG = ROOT / "docs" / "figures" / "ns_wall_penalty_scaling.png"

SCHEMA = "tensordg.ns_wall_penalty_scaling/1"
SIGMA_SCALES = (0.0, 0.5, 2.0, 8.0, 32.0, 128.0)
WALLS = {
    "xmin": None,
    "xmax": None,
    "ymin": {"velocity": (0.0, 0.0), "temperature": WALL_T},
    "ymax": {"velocity": (WALL_SPEED, 0.0), "temperature": WALL_T},
}
NO_WALLS = {"xmin": None, "xmax": None, "ymin": None, "ymax": None}


# ---------------------------------------------------------------------------
# Exactly representable control state (rho, u, v, p) = (1, y, 0, 1)
# ---------------------------------------------------------------------------
def representable_state(ncells=(2, 8), degree=1):
    nx, ny = ncells
    dx, dy = 1.0 / nx, 1.0 / ny
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, max(degree + 2, 3), torch.device("cpu"), torch.float64
    )

    def initial(x, y):
        u = y
        return torch.ones_like(u), u, torch.zeros_like(u), torch.ones_like(u)

    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(
        initial, x, y, dx, dy, nodes, weights, basis, torch.device("cpu"),
        torch.float64,
    )
    return dict(coeff=coeff, basis=basis, derivative=derivative, weights=weights,
                signs=signs, dx=dx, dy=dy, degree=degree, ncells=ncells)


def _residual(state, boundary, mu, kappa, penalty, periodic=(True, False),
              wall_gradient="face", wall_penalty=None):
    return ns._rhs_ns(
        state["coeff"], state["basis"], state["derivative"], state["weights"],
        state["signs"], state["dx"], state["dy"], periodic, boundary, mu, kappa,
        penalty, wall_gradient, wall_penalty,
    )


def _channels(rhs):
    values = rhs.abs().amax(dim=(0, 1, 2, 3)).tolist()
    mass, x, y, energy = values
    return dict(mass=mass, x_momentum=x, y_momentum=y, energy=energy,
                maximum=max(values))


# ---------------------------------------------------------------------------
# Section 1: consistency control
# ---------------------------------------------------------------------------
def section_consistency():
    rows = []
    for degree in (1, 2, 3):
        state = representable_state((2, 8), degree)
        penalty = 2.0 * (degree + 1.0) ** 2
        for mu, kappa, label in (
            (0.0, 0.0, "inviscid"),
            (MU, 0.0, "momentum_only"),
            (0.0, KAPPA, "conduction_only"),
            (MU, KAPPA, "full"),
        ):
            rhs = _residual(state, WALLS, mu, kappa, penalty)
            chans = _channels(rhs)
            rows.append(dict(
                degree=degree, ncells=list(state["ncells"]), viscosity=label,
                mu=mu, kappa=kappa, penalty=penalty, **chans,
                physical_dissipation=mu * WALL_SPEED ** 2,
                energy_minus_dissipation=abs(chans["energy"] - mu * WALL_SPEED ** 2),
                periodic_control=_channels(
                    _residual(state, NO_WALLS, mu, kappa, penalty, periodic=(True, True))
                )["maximum"],
            ))
    return rows


# ---------------------------------------------------------------------------
# Section 2: residual of the exact analytic Couette state vs sigma
# ---------------------------------------------------------------------------
def _linear_fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return dict(slope=slope, intercept=intercept, r2=r2)


def section_scaling(wall_gradient="face"):
    rows = []
    for degree in (1, 2, 3):
        for ny in (8, 16, 32):
            state = build_state((2, ny), degree)
            for scale in SIGMA_SCALES:
                penalty = scale * (degree + 1.0) ** 2
                rhs = _residual(state, WALLS, MU, KAPPA, penalty, wall_gradient=wall_gradient)
                chans = _channels(rhs)
                rows.append(dict(ncells=[2, ny], degree=degree, sigma_scale=scale,
                                 penalty=penalty, wall_gradient=wall_gradient,
                                 **chans))
    fits = []
    for degree in (1, 2, 3):
        for ny in (8, 16, 32):
            block = [r for r in rows if r["degree"] == degree and r["ncells"] == [2, ny]]
            fits.append(dict(
                degree=degree, ncells=[2, ny],
                **_linear_fit([r["sigma_scale"] for r in block],
                              [r["maximum"] for r in block]),
                intercept_over_slope=(
                    _linear_fit([r["sigma_scale"] for r in block],
                                [r["maximum"] for r in block])["intercept"]
                    / _linear_fit([r["sigma_scale"] for r in block],
                                  [r["maximum"] for r in block])["slope"]
                ),
            ))
    return rows, fits


# ---------------------------------------------------------------------------
# Section 3: instability onset vs penalty, from the representable state
# ---------------------------------------------------------------------------
def march_onset(degree=2, ncells=(2, 8), sigma_scale=2.0, final_time=0.2,
                max_steps=4000, wall_gradient="face"):
    """SSP-RK3 march; returns the time at which the state goes non-physical."""
    state = representable_state(ncells, degree)
    basis, derivative = state["basis"], state["derivative"]
    weights, signs = state["weights"], state["signs"]
    dx, dy = state["dx"], state["dy"]
    h = min(dx, dy)
    penalty = sigma_scale * (degree + 1.0) ** 2
    coeff = state["coeff"].clone()
    t = 0.0
    start = time.time()
    for step in range(max_steps):
        at = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
        _, ux, uy, pressure = _primitive(at)
        rho = at[..., 0].clamp_min(1e-12)
        sound = torch.sqrt(ns.GAMMA * pressure.clamp_min(1e-12) / rho)
        rate = (float(torch.max(ux.abs() + sound))
                + float(torch.max(uy.abs() + sound))) / h
        dt = 0.3 / (rate * (2 * degree + 1))
        diffusivity = max(4.0 / 3.0 * MU / float(rho.min()),
                          KAPPA / (float(rho.min()) * ns.CV))
        dt = min(dt, 1.0 / (diffusivity * (degree + 1.0) ** 4 / (0.1 * h * h)))
        k1 = _residual(state | {"coeff": coeff}, WALLS, MU, KAPPA, penalty,
                       wall_gradient=wall_gradient)
        u1 = _positivity_filter(coeff + dt * k1, basis, signs)
        k2 = _residual(state | {"coeff": u1}, WALLS, MU, KAPPA, penalty,
                       wall_gradient=wall_gradient)
        u2 = _positivity_filter(0.75 * coeff + 0.25 * (u1 + dt * k2), basis, signs)
        k3 = _residual(state | {"coeff": u2}, WALLS, MU, KAPPA, penalty,
                       wall_gradient=wall_gradient)
        coeff = _positivity_filter(coeff / 3.0 + (2.0 / 3.0) * (u2 + dt * k3),
                                   basis, signs)
        t += dt
        at = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
        _, _, uy, pressure = _primitive(at)
        nonphysical = (not bool(torch.isfinite(coeff).all())
                       or float(pressure.min()) <= 0.0
                       or float(at[..., 0].min()) <= 0.0)
        if nonphysical:
            return dict(degree=degree, ncells=list(ncells), sigma_scale=sigma_scale,
                        penalty=penalty, wall_gradient=wall_gradient,
                        survived=False, failed_time=t, steps=step, dt=dt,
                        peak_normal_velocity=float(uy.abs().max()),
                        seconds=time.time() - start)
        if t >= final_time:
            break
    return dict(degree=degree, ncells=list(ncells), sigma_scale=sigma_scale,
                penalty=penalty, wall_gradient=wall_gradient, survived=True,
                failed_time=None, steps=step, dt=dt,
                peak_normal_velocity=float(uy.abs().max()),
                seconds=time.time() - start)


def section_onset(sigmas=(2.0, 8.0, 32.0), degrees=(1, 2)):
    rows = []
    for degree in degrees:
        for scale in sigmas:
            rows.append(march_onset(degree=degree, sigma_scale=scale))
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def render_figure(payload, path=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    ax = axes[0]
    for row in payload["consistency"]:
        if row["degree"] == 1:
            continue
        ax.semilogy([row["degree"]], [max(row["y_momentum"], 1e-16)], "o",
                    color="tab:blue",
                    label="y-momentum" if row["degree"] == 2 else None)
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel(r"$|R_y|$ (representable state)")
    ax.set_title("wall closure consistency\n(must be round-off)")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    for degree, marker in zip((1, 2, 3), ("o", "s", "^")):
        block = [r for r in payload["scaling"]
                 if r["ncells"] == [2, 8] and r["degree"] == degree]
        fit = [f for f in payload["fits"]
               if f["ncells"] == [2, 8] and f["degree"] == degree][0]
        xs = [r["sigma_scale"] for r in block]
        ys = [r["maximum"] for r in block]
        ax.loglog(xs, ys, marker + "-", label=f"P{degree} data")
        ax.loglog([x for x in xs if x > 0],
                  [fit["intercept"] + fit["slope"] * x for x in xs if x > 0],
                  "--", lw=1,
                  label=f"P{degree} fit R²={fit['r2']:.4f}")
    ax.set_xlabel(r"penalty scale $\sigma/(p+1)^2$")
    ax.set_ylabel(r"$\max|R|$ at the exact Couette state")
    ax.set_title("residual is linear in the penalty")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6)

    ax = axes[2]
    for degree, marker in zip((1, 2), ("o", "s")):
        block = [r for r in payload["onset"] if r["degree"] == degree]
        xs = [r["sigma_scale"] for r in block]
        ys = [r["failed_time"] if r["failed_time"] else float("nan") for r in block]
        ax.loglog(xs, ys, marker + "-", label=f"P{degree}")
    ax.set_xlabel(r"penalty scale $\sigma/(p+1)^2$")
    ax.set_ylabel("time of first non-physical state")
    ax.set_title("transient onset, representable start")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sections = [a for a in argv if not a.startswith("-")] or [
        "consistency", "scaling", "onset"]
    previous = {}
    if OUT.exists():
        try:
            candidate = json.loads(OUT.read_text())
        except json.JSONDecodeError:          # pragma: no cover - corrupt artifact
            candidate = {}
        if candidate.get("schema") == SCHEMA:
            previous = candidate
    payload = {
        "schema": SCHEMA,
        "parameters": dict(
            mu=MU, kappa=KAPPA, prandtl=KAPPA and MU * ns.CP / KAPPA,
            wall_speed=WALL_SPEED, wall_temperature=WALL_T, pressure=PRESSURE,
            gamma=ns.GAMMA, sigma_scales=list(SIGMA_SCALES),
        ),
        "consistency": [],
        "scaling": [],
        "fits": [],
        "onset": [],
    }
    for key in ("consistency", "scaling", "fits", "onset"):
        if previous.get(key) and key.split("_")[0] not in sections:
            payload[key] = previous[key]
    if "consistency" in sections:
        payload["consistency"] = section_consistency()
        print(f"consistency: {len(payload['consistency'])} rows", flush=True)
    if "scaling" in sections:
        payload["scaling"], payload["fits"] = section_scaling()
        print(f"scaling: {len(payload['scaling'])} rows, "
              f"{len(payload['fits'])} fits", flush=True)
    if "onset" in sections:
        payload["onset"] = section_onset()
        print(f"onset: {len(payload['onset'])} rows", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}")
    if all(payload[key] for key in ("consistency", "scaling", "onset")):
        print(f"figure {render_figure(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())