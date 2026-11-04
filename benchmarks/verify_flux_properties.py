"""Verification of the Euler interface-flux kernels against analytic properties.

Two distinct levels of evidence are produced:

1. **Exact analytic invariants** that any consistent, symmetric Riemann solver
   must satisfy to round-off (no tolerance beyond floating point):

   * consistency          ``F(U, U) = f(U)``,
   * mirror symmetry      ``R F(U_L, U_R) = F(M U_R, M U_L)`` with
     ``M = diag(1, -1, 1)`` on the conserved variables and
     ``R = diag(-1, 1, -1)`` on the flux (x -> -x reflection),
   * isolated-contact exactness: for ``u_L = u_R = u`` and ``p_L = p_R = p``
     the HLLC flux equals the exact physical flux of the upwind state
     (Toro 2009, sec. 10.4) whereas an HLL/Rusanov type flux adds diffusion.

2. **Agreement with the exact Riemann solution** for the standard two-state
   cases (Toro 2009, ch. 4).  The exact state *at the interface* is obtained
   from ``benchmarks/riemann_exact.sample`` (which resolves rarefaction fans,
   so the comparison is valid even when x/t = 0 lies inside a fan, and it
   picks the correct side of a contact discontinuity).  HLLC and a local
   Lax-Friedrichs (Rusanov) flux are compared against it.  Wave-speed
   estimates are Davis' (as used in ``tensordg/euler_1d.py``), not the
   Einfeldt/Roe estimates required for the flux to reproduce the star state
   exactly, so this level is a tolerance-based comparison, not an identity.

Normalisation: ``relative`` divides by ``max(|F_exact|)``; ``scaled`` divides
by the larger of that and the incoming flux scale ``max(|F_L|, |F_R|)`` so
that cases whose exact interface flux nearly vanishes (near-vacuum star
regions) do not produce meaningless ratios.

Usage::

    PYTHONPATH=. python3 benchmarks/verify_flux_properties.py \
        --json validation/euler_fluxes/flux_properties.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from benchmarks.riemann_exact import sample as sample_exact
from benchmarks.riemann_exact import star_state as sample_star
from tensordg.euler_1d import (
    GAMMA,
    hllc_flux,
    physical_flux,
    primitive_to_conservative,
    sound_speed,
)

DTYPE = torch.float64
DEVICE = torch.device("cpu")
DIAPHRAGM = 0.5
SAMPLE_TIME = 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def state(rho, velocity, pressure):
    """Single-state conserved vector of shape (1, 3)."""
    return primitive_to_conservative(
        torch.tensor([rho], dtype=DTYPE, device=DEVICE),
        torch.tensor([velocity], dtype=DTYPE, device=DEVICE),
        torch.tensor([pressure], dtype=DTYPE, device=DEVICE),
    )


def pair(left, right):
    return state(*left), state(*right)


def mirror(conserved):
    out = conserved.clone()
    out[:, 1] = -out[:, 1]
    return out


REFLECTION = torch.tensor([-1.0, 1.0, -1.0], dtype=DTYPE, device=DEVICE)


def rusanov_flux(left, right):
    """Local Lax-Friedrichs (Rusanov) interface flux, reference implementation."""
    flux_left = physical_flux(left)
    flux_right = physical_flux(right)
    speed_left = (left[:, 1] / left[:, 0]).abs() + sound_speed(left)
    speed_right = (right[:, 1] / right[:, 0]).abs() + sound_speed(right)
    alpha = torch.maximum(speed_left, speed_right)
    return 0.5 * (flux_left + flux_right) - 0.5 * alpha[:, None] * (right - left)


def max_abs(tensor):
    return float(tensor.abs().max())


def relative_error_scale(numerical, exact):
    """max_abs(numerical - exact) / max_abs(exact), guarding the zero flux."""
    denominator = max_abs(exact)
    absolute = max_abs(numerical - exact)
    return absolute / denominator if denominator > 0.0 else absolute


def error_metrics(numerical, exact, scale):
    absolute = max_abs(numerical - exact)
    return {
        "max_abs_error": absolute,
        "relative": absolute / max_abs(exact) if max_abs(exact) > 0.0 else 0.0,
        "scaled": absolute / scale if scale > 0.0 else 0.0,
    }


def exact_interface_state(left, right):
    """Exact primitive state at x/t = 0 for the two-state Riemann problem."""
    sampled = sample_exact(np.array([DIAPHRAGM]), SAMPLE_TIME, left, right)
    return {key: float(value[0]) for key, value in sampled.items()}


# Toro's standard two-state Riemann test cases (Toro 2009, sec. 4.3-4.5).
STANDARD_CASES = [
    ("sod", (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), "shock tube: rarefaction + contact + shock"),
    ("lax", (0.445, 0.698, 3.528), (0.5, 0.0, 0.571), "Lax shock tube"),
    ("123problem", (1.0, -2.0, 0.4), (1.0, 2.0, 0.4), "symmetric double rarefaction (near vacuum)"),
    ("strong_shock", (5.99924, 19.5975, 460.894), (5.99242, -6.19633, 46.095), "very strong shock"),
    ("stationary_contact", (1.0, 0.0, 1.0), (0.125, 0.0, 1.0), "pure stationary contact (u = 0)"),
    ("moving_contact", (1.0, 0.5, 1.0), (0.125, 0.5, 1.0), "pure right-moving contact (u > 0)"),
    ("left_moving_contact", (1.0, -0.5, 1.0), (0.125, -0.5, 1.0), "pure left-moving contact (u < 0)"),
    ("sonic_rarefaction", (1.0, -0.75, 1.0), (1.0, 0.75, 1.0), "interface inside a rarefaction fan"),
]


# ---------------------------------------------------------------------------
# level 1: exact invariants
# ---------------------------------------------------------------------------
def check_exact_invariants():
    rng = np.random.default_rng(20261103)
    invariant_cases = [(1.0, 0.0, 1.0), (1.0, 3.0, 0.1), (0.125, 0.0, 0.1),
                       (5.99924, 19.5975, 460.894), (1.0, -1.5, 2.5)]
    for _ in range(12):
        invariant_cases.append((
            float(rng.uniform(0.05, 5.0)),
            float(rng.uniform(-4.0, 4.0)),
            float(rng.uniform(0.02, 50.0)),
        ))

    consistency = []
    for case in invariant_cases:
        u = state(*case)
        exact = physical_flux(u)
        numerical = hllc_flux(u, u)
        consistency.append({
            "state": list(case),
            "max_abs_error": max_abs(numerical - exact),
            "relative_error": error_metrics(numerical, exact, max_abs(exact))["relative"],
        })

    symmetry = []
    for name, left, right, _ in STANDARD_CASES:
        ul, ur = pair(left, right)
        lhs = REFLECTION * hllc_flux(ul, ur)
        rhs = hllc_flux(mirror(ur), mirror(ul))
        metrics = error_metrics(rhs, lhs, max_abs(lhs))
        symmetry.append({"case": name, "flux_scale": max_abs(lhs), **metrics})

    contacts = []
    for name, left, right, _ in STANDARD_CASES:
        if not (abs(left[1] - right[1]) < 1e-15 and abs(left[2] - right[2]) < 1e-15):
            continue
        ul, ur = pair(left, right)
        upwind = ul if left[1] > 0.0 else ur  # u = 0: both give (0, p, 0)
        exact = physical_flux(upwind)
        scale = max_abs(exact)
        contacts.append({
            "case": name,
            "upwind_side": "left" if left[1] > 0.0 else "right",
            "hllc": error_metrics(hllc_flux(ul, ur), exact, scale),
            "rusanov": error_metrics(rusanov_flux(ul, ur), exact, scale),
        })

    return {"consistency": consistency, "mirror_symmetry": symmetry,
            "isolated_contact": contacts}


# ---------------------------------------------------------------------------
# level 2: agreement with the exact Riemann solution
# ---------------------------------------------------------------------------
def check_exact_riemann_agreement():
    rows = []
    for name, left, right, description in STANDARD_CASES:
        exact_primitive = exact_interface_state(left, right)
        exact_flux = physical_flux(state(exact_primitive["density"],
                                       exact_primitive["velocity"],
                                       exact_primitive["pressure"]))
        ul, ur = pair(left, right)
        hllc = hllc_flux(ul, ur)
        rusanov = rusanov_flux(ul, ur)
        scale = max(max_abs(physical_flux(ul)), max_abs(physical_flux(ur)),
                    max_abs(exact_flux))
        rows.append({
            "case": name,
            "description": description,
            "left": list(left),
            "right": list(right),
            "exact_interface_primitive": exact_primitive,
            "exact_interface_flux": [float(v) for v in exact_flux[0]],
            "hllc_flux": [float(v) for v in hllc[0]],
            "rusanov_flux": [float(v) for v in rusanov[0]],
            "flux_scale": scale,
            "hllc": error_metrics(hllc, exact_flux, scale),
            "rusanov": error_metrics(rusanov, exact_flux, scale),
            "contact_only": abs(left[1] - right[1]) < 1e-15 and abs(left[2] - right[2]) < 1e-15,
            "all_finite": bool(torch.isfinite(hllc).all() and torch.isfinite(rusanov).all()),
        })
    return rows


def check_rankine_hugoniot_shock():
    """An isolated right-going shock: the flux at x/t = 0 is the post-shock flux.

    The post-shock state is built from the Rankine-Hugoniot relations for a
    chosen shock Mach number, so no contact or rarefaction is present at the
    interface and the exact flux is ``f(U_post)``.
    """
    rows = []
    for mach in (1.5, 2.0, 3.0, 5.0):
        rho_r, u_r, p_r = 1.0, 0.0, 1.0
        ratio = (2.0 * GAMMA * mach**2 - (GAMMA - 1.0)) / (GAMMA + 1.0)
        p_l = p_r * ratio
        rho_l = rho_r * ((GAMMA + 1.0) * mach**2) / ((GAMMA - 1.0) * mach**2 + 2.0)
        shock_speed = mach * np.sqrt(GAMMA * p_r / rho_r)
        u_l = shock_speed * (1.0 - rho_r / rho_l)
        ul, ur = pair((rho_l, u_l, p_l), (rho_r, u_r, p_r))
        exact = physical_flux(ul)  # uniform post-shock state, contact at rest
        numerical = hllc_flux(ul, ur)
        scale = max_abs(exact)
        rows.append({
            "shock_mach": mach,
            "post_shock_state": [rho_l, u_l, p_l],
            "exact_flux": [float(v) for v in exact[0]],
            "hllc": error_metrics(numerical, exact, scale),
        })
    return rows


# ---------------------------------------------------------------------------
# level 3: attribution of the residual error to the wave-speed estimates
# ---------------------------------------------------------------------------
def exact_wave_speeds(left, right):
    """Exact outer wave speeds S_L, S_R of the Riemann fan (Toro 2009, ch. 4)."""
    star_p, star_u = sample_star(left, right)
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = np.sqrt(GAMMA * p_l / rho_l)
    a_r = np.sqrt(GAMMA * p_r / rho_r)
    if star_p > p_l:  # left-going shock
        ratio = star_p / p_l
        speed_l = u_l - a_l * np.sqrt((GAMMA + 1.0) / (2.0 * GAMMA) * ratio
                                      + (GAMMA - 1.0) / (2.0 * GAMMA))
    else:  # left-going rarefaction: head of the fan
        speed_l = u_l - a_l
    if star_p > p_r:  # right-going shock
        ratio = star_p / p_r
        speed_r = u_r + a_r * np.sqrt((GAMMA + 1.0) / (2.0 * GAMMA) * ratio
                                      + (GAMMA - 1.0) / (2.0 * GAMMA))
    else:  # right-going rarefaction: head of the fan
        speed_r = u_r + a_r
    return float(speed_l), float(speed_r)


def hllc_flux_with_speeds(left, right, speed_l, speed_r):
    """Identical HLLC algebra to ``tensordg.euler_1d.hllc_flux`` with S_L, S_R given.

    Used only as a diagnostic: it isolates how much of the interface-flux error
    comes from Davis' wave-speed estimates rather than from the HLLC formula.
    """
    rho_l = left[:, 0]
    rho_r = right[:, 0]
    velocity_l = left[:, 1] / rho_l
    velocity_r = right[:, 1] / rho_r
    pressure_l = (GAMMA - 1.0) * (left[:, 2] - 0.5 * left[:, 1] ** 2 / rho_l)
    pressure_r = (GAMMA - 1.0) * (right[:, 2] - 0.5 * right[:, 1] ** 2 / rho_r)
    s_l = torch.full_like(rho_l, speed_l)
    s_r = torch.full_like(rho_r, speed_r)
    denominator = rho_l * (s_l - velocity_l) - rho_r * (s_r - velocity_r)
    speed_middle = (
        pressure_r - pressure_l
        + rho_l * velocity_l * (s_l - velocity_l)
        - rho_r * velocity_r * (s_r - velocity_r)
    ) / denominator

    def star(state, rho, velocity, pressure, speed):
        factor = speed - velocity
        rho_star = rho * factor / (speed - speed_middle)
        energy_star = rho_star * (
            state[:, 2] / rho
            + (speed_middle - velocity) * (speed_middle + pressure / (rho * factor))
        )
        return torch.stack((rho_star, rho_star * speed_middle, energy_star), dim=1)

    star_l = star(left, rho_l, velocity_l, pressure_l, s_l)
    star_r = star(right, rho_r, velocity_r, pressure_r, s_r)
    flux_l = physical_flux(left)
    flux_r = physical_flux(right)
    flux = torch.where((s_l >= 0.0)[:, None], flux_l, flux_l + s_l[:, None] * (star_l - left))
    flux = torch.where((speed_middle < 0.0)[:, None],
                       flux_r + s_r[:, None] * (star_r - right), flux)
    flux = torch.where((s_r <= 0.0)[:, None], flux_r, flux)
    return flux


def check_wave_speed_attribution():
    rows = []
    for name, left, right, _ in STANDARD_CASES:
        exact_primitive = exact_interface_state(left, right)
        exact_flux = physical_flux(state(exact_primitive["density"],
                                         exact_primitive["velocity"],
                                         exact_primitive["pressure"]))
        ul, ur = pair(left, right)
        scale = max(max_abs(physical_flux(ul)), max_abs(physical_flux(ur)),
                    max_abs(exact_flux))
        speed_l, speed_r = exact_wave_speeds(left, right)
        davis = hllc_flux(ul, ur)
        exact_speeds = hllc_flux_with_speeds(ul, ur, speed_l, speed_r)
        # Davis' estimates as implemented in tensordg/euler_1d.py, for contrast.
        a_l = np.sqrt(GAMMA * left[2] / left[0])
        a_r = np.sqrt(GAMMA * right[2] / right[0])
        davis_l = min(left[1] - a_l, right[1] - a_r)
        davis_r = max(left[1] + a_l, right[1] + a_r)
        rows.append({
            "case": name,
            "exact_wave_speeds": [speed_l, speed_r],
            "davis_wave_speeds": [float(davis_l), float(davis_r)],
            "davis_speed_error": [float(abs(davis_l - speed_l)), float(abs(davis_r - speed_r))],
            "hllc_davis": error_metrics(davis, exact_flux, scale),
            "hllc_exact_speeds": error_metrics(exact_speeds, exact_flux, scale),
        })
    return rows


def summarize(exact_invariants, riemann_rows, shock_rows, attribution_rows):
    contacts = exact_invariants["isolated_contact"]
    hllc_scaled = [row["hllc"]["scaled"] for row in riemann_rows]
    rusanov_scaled = [row["rusanov"]["scaled"] for row in riemann_rows]
    return {
        "consistency_max_abs_error": max(r["max_abs_error"] for r in exact_invariants["consistency"]),
        "mirror_symmetry_max_relative_error": max(
            r["max_abs_error"] / r["flux_scale"] for r in exact_invariants["mirror_symmetry"]),
        "contact_hllc_max_abs_error": max(r["hllc"]["max_abs_error"] for r in contacts),
        "contact_rusanov_max_scaled_error": max(r["rusanov"]["scaled"] for r in contacts),
        "riemann_hllc_max_scaled_error": max(hllc_scaled),
        "riemann_rusanov_max_scaled_error": max(rusanov_scaled),
        "riemann_hllc_median_scaled_error": float(np.median(hllc_scaled)),
        "riemann_rusanov_median_scaled_error": float(np.median(rusanov_scaled)),
        "riemann_hllc_max_relative_error": max(row["hllc"]["relative"] for row in riemann_rows),
        "isolated_shock_max_scaled_error": max(r["hllc"]["scaled"] for r in shock_rows),
        "isolated_shock_max_relative_error": max(r["hllc"]["relative"] for r in shock_rows),
        "all_finite": all(row["all_finite"] for row in riemann_rows),
        "hllc_davis_max_scaled_error": max(r["hllc_davis"]["scaled"] for r in attribution_rows),
        "hllc_exact_speeds_max_scaled_error": max(r["hllc_exact_speeds"]["scaled"] for r in attribution_rows),
        "hllc_davis_median_scaled_error": float(np.median([r["hllc_davis"]["scaled"] for r in attribution_rows])),
        "hllc_exact_speeds_median_scaled_error": float(np.median([r["hllc_exact_speeds"]["scaled"] for r in attribution_rows])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="validation/euler_fluxes/flux_properties.json")
    args = parser.parse_args()

    torch.set_default_dtype(DTYPE)
    exact_invariants = check_exact_invariants()
    riemann_rows = check_exact_riemann_agreement()
    shock_rows = check_rankine_hugoniot_shock()
    attribution_rows = check_wave_speed_attribution()
    summary = summarize(exact_invariants, riemann_rows, shock_rows, attribution_rows)

    payload = {
        "timestamp": "2026-11-03",
        "dtype": "float64",
        "gamma": GAMMA,
        "exact_solver": "benchmarks/riemann_exact.py (Toro 2009 ch. 4)",
        "exact_invariants": exact_invariants,
        "exact_riemann_agreement": riemann_rows,
        "rankine_hugoniot_shock": shock_rows,
        "wave_speed_attribution": attribution_rows,
        "summary": summary,
    }
    directory = os.path.dirname(args.json)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"wrote {args.json}")
    for key, value in summary.items():
        print(f"  {key:42s} {value:.3e}" if isinstance(value, float) else f"  {key:42s} {value}")
    print("\nwave-speed attribution (scaled interface-flux error)")
    print(f"{'case':22s} {'Davis S':>12s} {'exact S':>12s}")
    for row in attribution_rows:
        print(f"{row['case']:22s} {row['hllc_davis']['scaled']:12.3e} "
              f"{row['hllc_exact_speeds']['scaled']:12.3e}")
    print("\nper-case interface-flux error vs exact Riemann state (scaled / relative)")
    print(f"{'case':22s} {'HLLC':>22s} {'Rusanov':>22s}")
    for row in riemann_rows:
        print(f"{row['case']:22s} {row['hllc']['scaled']:10.3e} / {row['hllc']['relative']:10.3e} "
              f"{row['rusanov']['scaled']:10.3e} / {row['rusanov']['relative']:10.3e}")


if __name__ == "__main__":
    main()