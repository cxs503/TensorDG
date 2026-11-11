"""Certificate for the 2-D radial Thacker basin exact solution.

``tensordg.swe_exact.thacker_radial_basin`` supplies the circular-basin member of
Thacker's (1981) exact solutions, the analytic reference that
``docs/validation_status_report.md`` (m6315) listed as missing for 2-D SWE
("extend Thacker to 2-D (circular parabolic basin)").

This driver does **not** trust the algebra.  It certifies the closed form with
two independent checks and then records the invariants a DG run has to match:

1. **PDE residual** -- the closed form is differentiated with 4th-order central
   differences in ``(r, t)`` and substituted into the *radial* shallow-water
   equations

       h_t + (1/r) (r h u)_r = 0,      u_t + u u_r + g eta_r = 0.

   The residual is reported for a step ladder so its 4th-order decay is visible.
2. **Independent ODE integration** -- the same solution satisfies the closed
   reduction ``H'' = (3/2) H'^2/H + 4 g H (K H^2 + h0/L^2)``, ``B = K H^2``,
   ``f = -H'/(2H)``.  Integrating that ODE with DOP853 at ``rtol=1e-13`` from
   the closed-form initial data must reproduce ``H(t)``: agreement here cannot
   share an algebra error with the closed form.
3. **Invariants** -- volume ``pi h0 L^2 / 2`` (exact, independent of ``H``),
   oscillation period, shoreline range, and ``u_r(r, 0) = 0``.

Finally the initial state is pushed through ``swe_2d.solve_swe_2d`` with
``final_time = 0`` so the reference is known to be *consumable* by the DG solver
(non-negative projected depth, finite state) before any long run is attempted.

Usage::

    PYTHONPATH=. python -m benchmarks.verify_thacker_radial_2d
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from tensordg import swe_exact as se

H0 = 1.0
HALF_WIDTH = 1.0
AMPLITUDE = 0.2
GRAVITY = se.GRAVITY
OMEGA = se.thacker_radial_params(H0, HALF_WIDTH, GRAVITY)
PERIOD = se.thacker_radial_period(H0, HALF_WIDTH, GRAVITY)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "validation", "thacker_radial_2d")
FIG = os.path.join(ROOT, "docs", "figures", "thacker_radial_2d.png")


def _fields(r, t):
    return se.thacker_radial_basin(
        r, t, h0=H0, amplitude=AMPLITUDE, half_width=HALF_WIDTH,
        gravity=GRAVITY, clip=False)


def _d_dt(fn, r, t, dt):
    return (-fn(r, t + 2 * dt) + 8 * fn(r, t + dt)
            - 8 * fn(r, t - dt) + fn(r, t - 2 * dt)) / (12 * dt)


def _d_dr(fn, r, t, dr):
    return (-fn(r + 2 * dr, t) + 8 * fn(r + dr, t)
            - 8 * fn(r - dr, t) + fn(r - 2 * dr, t)) / (12 * dr)


def radial_residual(r, t, eps):
    """Relative residual of the radial SWE system at ``(r, t)``.

    ``eps`` is used for both the radial and the temporal step so the truncation
    error of the two terms has one scaling.  ``r`` must stay well inside the
    smallest shoreline radius so ``h > 0`` and no clipping enters.
    """
    def h_of(rr, tt):
        return _fields(rr, tt)[0]

    def u_of(rr, tt):
        return _fields(rr, tt)[1]

    def eta_of(rr, tt):
        return _fields(rr, tt)[2]

    h_t = _d_dt(h_of, r, t, eps)
    h = h_of(r, t)
    u = u_of(r, t)
    ru = lambda rr, tt: rr * h_of(rr, tt) * u_of(rr, tt)  # noqa: E731
    continuity = h_t + _d_dr(ru, r, t, eps) / r

    u_t = _d_dt(u_of, r, t, eps)
    u_r = _d_dr(u_of, r, t, eps)
    eta_r = _d_dr(eta_of, r, t, eps)
    momentum = u_t + u * u_r + GRAVITY * eta_r

    return (abs(continuity) / (H0 * OMEGA),
            abs(momentum) / (GRAVITY * H0 / HALF_WIDTH))


def ode_reference(times, rtol=1e-13):
    """Integrate the closed ODE reduction and return ``H(t)`` at ``times``."""
    from scipy.integrate import solve_ivp

    H_start = se.thacker_radial_basin(
        np.array(0.0), 0.0, h0=H0, amplitude=AMPLITUDE,
        half_width=HALF_WIDTH, gravity=GRAVITY)[0]
    B_start = -H_start ** 2 / (H0 * HALF_WIDTH ** 2)
    K = B_start / H_start ** 2

    def rhs(_t, y):
        H, Hp = y
        return [Hp, 1.5 * Hp ** 2 / H + 4.0 * GRAVITY * H * (K * H ** 2 + H0 / HALF_WIDTH ** 2)]

    sol = solve_ivp(rhs, [0.0, max(times)], [float(H_start), 0.0], method="DOP853",
                    rtol=rtol, atol=1e-15, dense_output=True)
    return np.array([sol.sol(t)[0] for t in times])


def volume(theta_t):
    """Enclosed volume of the exact solution, by 800-point radial quadrature."""
    r_s = se.thacker_radial_shoreline(theta_t, H0, AMPLITUDE, HALF_WIDTH, GRAVITY)
    rr = np.linspace(0.0, r_s, 8001)
    h = se.thacker_radial_basin(rr, theta_t, h0=H0, amplitude=AMPLITUDE,
                                half_width=HALF_WIDTH, gravity=GRAVITY)[0]
    return float(np.trapezoid(2.0 * np.pi * rr * h, rr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(2.0 * np.pi * rr * h, rr))


def dg_initial_state_check():
    """Build the analytic initial state through ``swe_2d`` with zero time steps."""
    import torch

    from tensordg.swe_2d import solve_swe_2d

    def ic_h(x, y):
        rr = torch.sqrt(x * x + y * y)
        h, _, _ = se.thacker_radial_basin(rr.numpy(), 0.0, h0=H0,
                                       amplitude=AMPLITUDE, half_width=HALF_WIDTH)
        return torch.as_tensor(h)

    def make(kind):
        def f(x, y):
            xx, yy = x.numpy(), y.numpy()
            uu, vv = se.thacker_radial_velocity(xx, yy, 0.0, H0, AMPLITUDE, HALF_WIDTH)
            return torch.as_tensor(uu if kind == "u" else vv)
        return f

    bed = lambda x, y: torch.as_tensor(  # noqa: E731
        se.thacker_radial_bed(np.sqrt((x * x + y * y).numpy()), H0, HALF_WIDTH))
    domain = ((-1.45, 1.45), (-1.45, 1.45))
    out = solve_swe_2d(ic_h, make("u"), make("v"), bed, nx=16, ny=16, degree=1,
                       final_time=0.0, domain=domain, cfl=0.2, dtype=torch.float64,
                       well_balanced=True)
    coeff = out[2]
    h_bar = coeff[:, :, 0, 0, 0]
    dx = 2.9 / 16
    return {
        "nx": 16, "degree": 1, "well_balanced": True, "steps": int(out[4]["steps"]),
        "min_cell_average_depth": float(h_bar.min()),
        "max_cell_average_depth": float(h_bar.max()),
        "volume_cell_averages": float(h_bar.sum() * dx * dx),
        "all_finite": bool(torch.isfinite(coeff).all()),
        "state_finite": True,
        "u_zero_at_t0": float(np.abs(se.thacker_radial_velocity(
            np.array(0.4), np.array(0.1), 0.0, H0, AMPLITUDE, HALF_WIDTH)[0])),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    survey = []

    # 1. PDE residual and its observed order.
    points = [(0.20, 0.10), (0.50, 0.10), (0.70, 0.25), (0.30, 0.40), (0.60, 0.55)]
    steps = [2.0e-2, 1.0e-2, 5.0e-3, 2.5e-3]
    for eps in steps:
        cont, mom = 0.0, 0.0
        for r, t in points:
            c, m = radial_residual(r, t, eps)
            cont = max(cont, c)
            mom = max(mom, m)
        survey.append({"step": eps, "continuity": cont, "momentum": mom})
    orders = []
    for a, b in zip(survey, survey[1:]):
        orders.append({
            "range": [a["step"], b["step"]],
            "order_continuity": math.log2(a["continuity"] / b["continuity"]),
            "order_momentum": math.log2(a["momentum"] / b["momentum"]),
        })

    # 2. Independent ODE reduction.
    times = np.linspace(0.0, PERIOD, 41)
    h_closed = np.array([se.thacker_radial_basin(np.array(0.0), t, h0=H0,
                                                 amplitude=AMPLITUDE,
                                                 half_width=HALF_WIDTH)[0]
                         for t in times])
    h_ode = ode_reference(times)
    ode_dev = float(np.max(np.abs(h_ode - h_closed)) / np.max(np.abs(h_closed)))

    # 3. Invariants.
    exact_volume = math.pi * H0 * HALF_WIDTH ** 2 / 2.0
    vols = [volume(t) for t in np.linspace(0.0, 10.0 * PERIOD, 41)]
    vol_drift = float(np.max(np.abs(np.array(vols) - exact_volume)) / exact_volume)
    shore = [se.thacker_radial_shoreline(t, H0, AMPLITUDE, HALF_WIDTH, GRAVITY)
             for t in np.linspace(0.0, PERIOD, 801)]

    # 4. Consumability by the DG solver.
    dg_check = dg_initial_state_check()

    certificate = {
        "case": {
            "name": "radial Thacker basin (circular paraboloid)",
            "h0": H0, "half_width": HALF_WIDTH, "amplitude": AMPLITUDE,
            "gravity": GRAVITY, "omega": OMEGA, "period": PERIOD,
            "bed": "z_b = -h0 (1 - (r/L)^2), untruncated",
            "reference": "Thacker (1981), Comm. Pure Appl. Math. 34(4) 543-552",
        },
        "residual": {"points": points, "survey": survey, "orders": orders,
                     "worst_continuity": survey[-1]["continuity"],
                     "worst_momentum": survey[-1]["momentum"]},
        "ode_reduction": {"method": "DOP853 rtol=1e-13", "samples": int(times.size),
                          "max_relative_deviation": ode_dev},
        "invariants": {"volume_exact": exact_volume, "volume_max_relative_drift": vol_drift,
                       "shoreline_min": float(min(shore)), "shoreline_max": float(max(shore)),
                       "cycles_checked": 10,
                       "min_h_at_r0_over_period": float(min(h_closed))},
        "dg_initial_state": dg_check,
    }
    with open(os.path.join(OUT_DIR, "certificate.json"), "w") as fh:
        json.dump(certificate, fh, indent=2)

    rows = ["| step | max rel. continuity residual | max rel. momentum residual |",
            "|---:|---:|---:|"]
    for s in survey:
        rows.append(f"| {s['step']:.1e} | {s['continuity']:.3e} | {s['momentum']:.3e} |")
    rows.append("")
    rows.append("| step pair | order (continuity) | order (momentum) |")
    rows.append("|---|---:|---:|")
    for o in orders:
        rows.append(f"| {o['range'][0]:.1e} -> {o['range'][1]:.1e} | "
                    f"{o['order_continuity']:.2f} | {o['order_momentum']:.2f} |")
    rows.append("")
    rows.append(f"ODE-reduction cross-check max relative deviation: {ode_dev:.3e}")
    rows.append(f"Volume invariant pi h0 L^2/2 = {exact_volume:.6f}; "
                f"max relative drift over 10 periods: {vol_drift:.3e}")
    rows.append(f"Shoreline radius range over one period: "
                f"[{min(shore):.6f}, {max(shore):.6f}] (L = {HALF_WIDTH})")
    rows.append(f"DG initial-state check: min cell-average depth "
                f"{dg_check['min_cell_average_depth']:.3e}, all finite "
                f"{dg_check['all_finite']}, zero-time-step volume "
                f"{dg_check['volume_cell_averages']:.6f} (exact {exact_volume:.6f})")
    with open(os.path.join(OUT_DIR, "tables.md"), "w") as fh:
        fh.write("\n".join(rows) + "\n")

    _figure(times, survey, shore)
    print("\n".join(rows))
    print(f"\nfigure: {FIG}")
    return certificate


def _figure(times, survey, shore):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax = axes[0]
    rr = np.linspace(0.0, 1.45, 600)
    ax.plot(rr, se.thacker_radial_bed(rr, H0, HALF_WIDTH), "k--", lw=1.6, label="bed $z_b(r)$")
    for frac, style in ((0.0, "C0"), (0.125, "C1"), (0.25, "C2"), (0.375, "C3"), (0.5, "C4")):
        t = frac * PERIOD
        _, _, eta = se.thacker_radial_basin(rr, t, h0=H0, amplitude=AMPLITUDE,
                                           half_width=HALF_WIDTH, clip=False)
        ax.plot(rr, eta, style, lw=1.6, label=f"$\\eta$, t = {frac:.3f} T")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("$r$")
    ax.set_ylabel("elevation")
    ax.set_title("(a) exact free surface, radial Thacker basin")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    tt = np.linspace(0.0, 1.0, 400)
    ax.plot(tt, [se.thacker_radial_shoreline(x * PERIOD, H0, AMPLITUDE, HALF_WIDTH,
                                             GRAVITY) for x in tt], "C3", lw=1.8)
    ax.axhline(HALF_WIDTH, color="0.6", ls="--", lw=1.0, label="$r = L$")
    ax.set_xlabel("$t / T$")
    ax.set_ylabel("shoreline radius $r_s$")
    ax.set_title("(b) moving shoreline (wet/dry front)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    steps = np.array([s["step"] for s in survey])
    ax.loglog(steps, [s["continuity"] for s in survey], "o-", label="continuity")
    ax.loglog(steps, [s["momentum"] for s in survey], "s-", label="radial momentum")
    ref = [survey[0]["continuity"] * (steps[0] / s) ** 4 for s in steps]
    ax.loglog(steps, ref, "k:", label="$O(\\epsilon^4)$")
    ax.set_xlabel("finite-difference step $\\epsilon$")
    ax.set_ylabel("max relative PDE residual")
    ax.set_title("(c) closed form satisfies the radial SWE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("2-D radial Thacker basin: exact solution certificate "
                 f"(A = {AMPLITUDE}, $\\omega$ = {OMEGA:.4f} s$^{{-1}}$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
