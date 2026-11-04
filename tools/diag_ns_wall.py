"""Reproducible diagnostics for the wall (Dirichlet) path of ``ns_dg_2d``.

Context: the m2496 delivery added the BR1/SIPG viscous Navier-Stokes solver and
honestly marked the *wall-bounded transient* case unverified (the impulsively
started Couette run raises ``nonphysical cell average``).  That report guessed
the cause was "penalty / explicit-RK3 stiffness" and suggested shrinking the
time step.  This tool measures the four questions that decide it:

``residual``  residual of the **exact steady Couette state**.  The state is an
              (almost) exact solution, so a consistent wall operator has a
              residual that vanishes under mesh refinement.  The residual is
              reported per channel and the fraction that lives in the wall rows.
``fluxconv``  error of the *wall* viscous flux against the analytic values
              (``tau_xy``, ``u tau_xy + kappa dT/dy``) on three meshes; the
              observed slope tells whether the boundary flux is O(h) or O(h^2).
``channels``  which Dirichlet channel (u / v / T) destabilises the transient:
              the wall penalty is switched on for one channel at a time.
``penalty``   growth of the transient as a function of the SIPG penalty
              multiplier ``sigma`` (``pen = sigma (p+1)^2``).  A *larger*
              penalty making things worse is the signature of an
              anti-dissipative wall term, not of time-step stiffness.
``dtindep``   physical time and step count at which the transient goes
              non-physical for two viscous CFL numbers.  If the failure happens
              at the same *physical time* with a 4x smaller step, the instability
              is spatial, not a stability-limit artefact.

All numbers land in ``validation/ns_dg_2d_wall/summary.json`` and the figure
``docs/figures/ns_dg_2d_wall.png``; nothing in the report is hand-transcribed.

Usage::

    PYTHONPATH=. python -m tools.diag_ns_wall            # everything (~3 min)
    PYTHONPATH=. python -m tools.diag_ns_wall residual   # one section
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_2d import _positivity_filter, _primitive, legendre_operators

MU = 0.1
WALL_SPEED = 1.0
WALL_T = 1.0
PRESSURE = 1.0
PRANDTL = 0.71
KAPPA = MU * ns.CP / PRANDTL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "ns_dg_2d_wall"
FIG = ROOT / "docs" / "figures"


# ---------------------------------------------------------------------------
# Exact steady Couette state (velocity U y/H, Brinkman temperature parabola)
# ---------------------------------------------------------------------------
def _exact(y):
    s = y
    u = WALL_SPEED * s
    v = torch.zeros_like(u)
    temperature = WALL_T + MU * WALL_SPEED**2 / (2.0 * KAPPA) * s * (1.0 - s)
    return u, v, temperature


def build_state(ncells, degree, dtype=torch.float64):
    """Modal coefficients of the exact steady Couette state + the wall dict."""
    nx, ny = ncells
    dev = torch.device("cpu")
    order = max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, order, dev, dtype
    )
    dx, dy = 1.0 / nx, 1.0 / ny

    def initial(x, y):
        u, v, temperature = _exact(y)
        rho = PRESSURE / temperature
        # ``_initial_coefficients`` takes the primitive tuple (rho, u, v, p)
        return rho, u, v, torch.full_like(u, PRESSURE)

    x = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy
    coeff = ns._initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis, dev, dtype)
    boundary = {
        "xmin": None,
        "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": WALL_T},
        "ymax": {"velocity": (WALL_SPEED, 0.0), "temperature": WALL_T},
    }
    return dict(
        x=x, y=y, coeff=coeff, boundary=boundary, basis=basis, derivative=derivative,
        weights=weights, signs=signs, dx=dx, dy=dy, degree=degree,
    )


# ---------------------------------------------------------------------------
# Section 1: residual of the exact steady state
# ---------------------------------------------------------------------------
def section_residual(wall_gradient="taylor"):
    rows = []
    for ncells, degree in (((2, 8), 1), ((2, 16), 1), ((2, 32), 1), ((2, 8), 2)):
        d = build_state(ncells, degree)
        penalty = 2.0 * (degree + 1.0) ** 2
        rhs = ns._rhs_ns(d["coeff"], d["basis"], d["derivative"], d["weights"],
                         d["signs"], d["dx"], d["dy"], (True, False), d["boundary"],
                         MU, KAPPA, penalty, wall_gradient)
        magnitude = float(rhs.abs().max())
        channels = rhs.abs().amax(dim=(0, 1, 2, 3)).tolist()
        wall = float(rhs[:, [0, -1]].abs().max())
        rows.append(dict(
            ncells=list(ncells), degree=degree, wall_gradient=wall_gradient,
            rhs_max=magnitude,
            rhs_channels=channels, wall_fraction=wall / magnitude,
            interior_max=float(rhs[:, 1:-1].abs().max()),
        ))
    return rows


# ---------------------------------------------------------------------------
# Section 2: wall flux error vs the analytic traction / heat flux
# ---------------------------------------------------------------------------
ANALYTIC_MX_FLUX = -MU * WALL_SPEED                       # total flux (F_inv - F_v)_x
ANALYTIC_E_FLUX = -(WALL_SPEED * MU * WALL_SPEED
                    + KAPPA * (MU * WALL_SPEED**2 / (2.0 * KAPPA)) * (1.0 - 2.0))
ANALYTIC_E_FLUX_BOTTOM = -(KAPPA * (MU * WALL_SPEED**2 / (2.0 * KAPPA)))


def wall_fluxes(ncells, degree, penalty_scale=2.0, wall_gradient="taylor"):
    d = build_state(ncells, degree)
    penalty = penalty_scale * (degree + 1.0) ** 2
    h = d["dy"]
    left = ns._face_primitives(d["coeff"], d["basis"], d["derivative"], d["signs"], 1, "left")
    right = ns._face_primitives(d["coeff"], d["basis"], d["derivative"], d["signs"], 1, "right")
    recon = (ns._wall_normal_derivative(d["coeff"], d["basis"], d["signs"], h, 1)
             if wall_gradient == "taylor" else None)
    top = ns._boundary_flux(right, d["boundary"]["ymax"], 1, h, MU, KAPPA, penalty,
                            2.0 / h, 2.0 / d["dx"],
                            None if recon is None else recon["right"])
    bottom = ns._boundary_flux(left, d["boundary"]["ymin"], 1, h, MU, KAPPA, penalty,
                               2.0 / h, 2.0 / d["dx"],
                               None if recon is None else recon["left"])
    return dict(
        mx_top=float(top[0, -1, 0, 1]), mx_bottom=float(bottom[0, 0, 0, 1]),
        energy_top=float(top[0, -1, 0, 3]), energy_bottom=float(bottom[0, 0, 0, 3]),
        my_top=float(top[0, -1, 0, 2]), my_bottom=float(bottom[0, 0, 0, 2]),
    )


def section_fluxconv(wall_gradient="taylor"):
    rows = []
    for ny in (8, 16, 32, 64):
        f = wall_fluxes((2, ny), 1, wall_gradient=wall_gradient)
        rows.append(dict(
            ncells=[2, ny], degree=1, wall_gradient=wall_gradient, **f,
            mx_error=abs(f["mx_top"] - ANALYTIC_MX_FLUX),
            mx_error_bottom=abs(f["mx_bottom"] - ANALYTIC_MX_FLUX),
            energy_error_top=abs(f["energy_top"] - ANALYTIC_E_FLUX),
            energy_error_bottom=abs(f["energy_bottom"] - ANALYTIC_E_FLUX_BOTTOM),
        ))
    for previous, current in zip(rows, rows[1:]):
        current["slope_energy_bottom"] = (
            torch.log(torch.tensor(previous["energy_error_bottom"] / current["energy_error_bottom"]))
            / torch.log(torch.tensor(2.0))
        ).item()
        current["slope_mx_bottom"] = (
            torch.log(torch.tensor(previous["mx_error_bottom"] / current["mx_error_bottom"]))
            / torch.log(torch.tensor(2.0))
        ).item()
    return rows


# ---------------------------------------------------------------------------
# Marching helper: explicit SSP-RK3 from the exact steady state
# ---------------------------------------------------------------------------
def march(penalty_scale=2.0, channel_weights=(1.0, 1.0, 1.0), ncells=(2, 8),
          degree=1, cfl_viscous=1.0, max_steps=400, stop_on_nonphysical=True,
          record_every=0, wall_gradient="taylor"):
    """SSP-RK3 march; returns history and the failure step (if any)."""
    d = build_state(ncells, degree)
    boundary, basis, derivative, weights, signs = (
        d["boundary"], d["basis"], d["derivative"], d["weights"], d["signs"]
    )
    dx, dy = d["dx"], d["dy"]
    h = min(dx, dy)
    penalty = penalty_scale * (degree + 1.0) ** 2
    wu, wv, wt = channel_weights

    original = ns._boundary_flux

    def channel_masked(side, wall, axis, h_, mu, kappa, pen, normal_scale,
                       tangential_scale, normal_derivative=None):
        """Same flux as ``ns._boundary_flux`` with per-channel penalty weights."""
        inner_u, inner_v = side["u"], side["v"]
        wall_velocity = wall.get("velocity", (0.0, 0.0))
        face_u = torch.as_tensor(wall_velocity[0], device=inner_u.device,
                                dtype=inner_u.dtype).expand_as(inner_u)
        face_v = torch.as_tensor(wall_velocity[1], device=inner_v.device,
                                dtype=inner_v.dtype).expand_as(inner_v)
        wall_temperature = wall.get("temperature", None)
        face_T = (side["temperature"] if wall_temperature is None else
                  torch.as_tensor(wall_temperature, device=inner_u.device,
                                  dtype=inner_u.dtype).expand_as(inner_u))
        if normal_derivative is None:
            base_u = side["u_n"] * normal_scale
            base_v = side["v_n"] * normal_scale
            base_T = side["T_n"] * normal_scale
        else:
            base_u = normal_derivative["u_n"]
            base_v = normal_derivative["v_n"]
            base_T = normal_derivative["T_n"]
        normal_u = base_u - wu * (pen / h_) * (inner_u - face_u)
        normal_v = base_v - wv * (pen / h_) * (inner_v - face_v)
        normal_T = base_T - wt * (pen / h_) * (side["temperature"] - face_T)
        zeros = torch.zeros_like(inner_u)
        pressure = side["pressure"]
        if axis == 0:
            inviscid = torch.stack([zeros, pressure, zeros, zeros], dim=-1)
        else:
            inviscid = torch.stack([zeros, zeros, pressure, zeros], dim=-1)
        face = {"u": face_u, "v": face_v}
        tu = side["u_t"] * tangential_scale
        tv = side["v_t"] * tangential_scale
        tT = side["T_t"] * tangential_scale
        if axis == 0:
            face.update(u_x=normal_u, v_x=normal_v, T_x=normal_T,
                        u_y=tu, v_y=tv, T_y=tT)
        else:
            face.update(u_y=normal_u, v_y=normal_v, T_y=normal_T,
                        u_x=tu, v_x=tv, T_x=tT)
        return inviscid - ns._viscous_flux(face, axis, mu, kappa)

    ns._boundary_flux = channel_masked
    history, failure = [], None
    try:
        coeff = d["coeff"].clone()
        time = 0.0
        for step in range(max_steps):
            at = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
            _, ux, uy, pressure = _primitive(at)
            rho = at[..., 0].clamp_min(1e-12)
            sound = torch.sqrt(GAMMA * pressure.clamp_min(1e-12) / rho)
            rate = (float(torch.max(ux.abs() + sound)) + float(torch.max(uy.abs() + sound))) / h
            dt = 0.3 / (rate * (2 * degree + 1))
            diffusivity = max((4.0 / 3.0) * MU / float(rho.min()),
                              KAPPA / (float(rho.min()) * ns.CV))
            dt = min(dt, cfl_viscous / (diffusivity * (degree + 1.0) ** 4 / (0.1 * h * h)))
            k1 = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                            (True, False), boundary, MU, KAPPA, penalty,
                            wall_gradient)
            u1 = _positivity_filter(coeff + dt * k1, basis, signs)
            k2 = ns._rhs_ns(u1, basis, derivative, weights, signs, dx, dy,
                            (True, False), boundary, MU, KAPPA, penalty,
                            wall_gradient)
            u2 = _positivity_filter(0.75 * coeff + 0.25 * (u1 + dt * k2), basis, signs)
            k3 = ns._rhs_ns(u2, basis, derivative, weights, signs, dx, dy,
                            (True, False), boundary, MU, KAPPA, penalty,
                            wall_gradient)
            coeff = _positivity_filter(coeff / 3.0 + (2.0 / 3.0) * (u2 + dt * k3),
                                       basis, signs)
            time += dt
            at = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
            _, ux, uy, pressure = _primitive(at)
            finite = bool(torch.isfinite(coeff).all())
            physical = finite and float(at[..., 0].min()) > 0.0 and float(pressure.min()) > 0.0
            if record_every and (step % record_every == 0):
                history.append(dict(step=step, time=time, dt=dt,
                                    min_pressure=float(pressure.min()),
                                    min_rho=float(at[..., 0].min()),
                                    max_u=float(ux.abs().max())))
            if stop_on_nonphysical and not physical:
                failure = dict(step=step, time=time, dt=dt,
                               min_pressure=float(pressure.min()))
                break
        else:
            at = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
            _, ux, uy, pressure = _primitive(at)
            history.append(dict(step=max_steps, time=time, dt=dt,
                                min_pressure=float(pressure.min()),
                                min_rho=float(at[..., 0].min()),
                                max_u=float(ux.abs().max())))
    finally:
        ns._boundary_flux = original
    return history, failure


def section_channels():
    rows = []
    for name, weights in (("all channels", (1.0, 1.0, 1.0)),
                          ("u only", (1.0, 0.0, 0.0)),
                          ("v only", (0.0, 1.0, 0.0)),
                          ("T only", (0.0, 0.0, 1.0)),
                          ("no penalty", (0.0, 0.0, 0.0))):
        _, failure = march(channel_weights=weights, max_steps=120)
        rows.append(dict(penalised=name, survived=failure is None,
                         failure=failure))
    return rows


def section_penalty():
    rows = []
    for sigma in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        _, failure = march(penalty_scale=sigma, max_steps=120)
        rows.append(dict(sigma=sigma, penalty=sigma * 4.0, survived=failure is None,
                         failure=failure))
    return rows


def section_dtindep():
    rows = []
    for cfl_viscous in (1.0, 0.25):
        history, failure = march(cfl_viscous=cfl_viscous, max_steps=2000)
        rows.append(dict(cfl_viscous=cfl_viscous,
                         survived=failure is None, failure=failure))
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(residual, fluxconv, channels, penalty, dtindep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    ny = [row["ncells"][1] for row in residual]
    ax.semilogx(ny, [row["rhs_channels"][1] for row in residual], "o-", label="momentum x")
    ax.semilogx(ny, [row["rhs_channels"][3] for row in residual], "s-", label="energy")
    ax.semilogx(ny, [row["rhs_channels"][0] for row in residual], "^-", label="density")
    ax.axhline(1e-10, color="k", lw=0.6, ls=":")
    ax.set_xlabel("cells in y (P1, degree 1)")
    ax.set_ylabel(r"$\max|\partial_t U|$ (exact steady state)")
    ax.set_title("(a) wall residual does not vanish with h")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ny = [row["ncells"][1] for row in fluxconv]
    ax.loglog(ny, [row["energy_error_bottom"] for row in fluxconv], "o-",
              label=r"$|E$-flux err, bottom wall$|$")
    ax.loglog(ny, [row["energy_error_top"] for row in fluxconv], "s-",
              label=r"$|E$-flux err, top wall$|$")
    ax.loglog(ny, [row["mx_error_bottom"] for row in fluxconv], "^-",
              label=r"$|\tau_{xy}$ err, bottom wall$|$")
    reference = [fluxconv[0]["energy_error_bottom"] * (fluxconv[0]["ncells"][1] / n)
                 for n in ny]
    ax.loglog(ny, reference, "k--", lw=0.8, label=r"$\propto h$")
    ax.set_xlabel("cells in y")
    ax.set_ylabel("wall flux error")
    ax.set_title("(b) wall flux is first order (O(h))")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for row in channels:
        history, _ = march(channel_weights=_weights_of(row["penalised"]),
                           max_steps=120, record_every=4, stop_on_nonphysical=False)
        ax.plot([h["time"] for h in history], [h["min_pressure"] for h in history],
                label=row["penalised"])
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\min p$")
    ax.set_title("(c) the temperature penalty drives the runaway")
    ax.set_ylim(-1.0, 1.1)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    target = FIG / "ns_dg_2d_wall.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def _weights_of(name):
    return {"all channels": (1.0, 1.0, 1.0), "u only": (1.0, 0.0, 0.0),
            "v only": (0.0, 1.0, 0.0), "T only": (0.0, 0.0, 1.0),
            "no penalty": (0.0, 0.0, 0.0)}[name]


SECTIONS = {
    "residual": section_residual,
    "fluxconv": section_fluxconv,
    "channels": section_channels,
    "penalty": section_penalty,
    "dtindep": section_dtindep,
}


def main(argv):
    wanted = argv[1:] or list(SECTIONS)
    results = {}
    for name in wanted:
        print(f"[{name}] running ...", flush=True)
        results[name] = SECTIONS[name]()
        for row in results[name]:
            print("   ", json.dumps(row, default=str), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("summary.json" if len(wanted) == len(SECTIONS) else f"partial_{'_'.join(wanted)}.json")
    path.write_text(json.dumps(results, indent=2))
    print("wrote", path)
    if len(wanted) == len(SECTIONS):
        fig = make_figure(results["residual"], results["fluxconv"],
                          results["channels"], results["penalty"], results["dtindep"])
        print("wrote", fig)


if __name__ == "__main__":
    main(sys.argv)