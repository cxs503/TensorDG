"""Verification of the HLLC outer wave-speed estimates.

Background
----------
``docs/euler_flux_kernel_verification.md`` found that ``tensordg.euler_1d.hllc_flux``
with Davis' wave speeds misses the exact Riemann interface flux by a *median
11.3%* (max 24.2%), i.e. it failed the project's 3% gate, and filed the
replacement of Davis' estimates by Einfeldt/Roe estimates (or better) as
follow-up work.  This driver closes that follow-up:

* level A -- the four selectable wave-speed models in
  ``tensordg.euler_1d.WAVE_SPEEDS`` (``davis``, ``einfeldt``, ``toro``,
  ``toro_iter``) against the exact interface flux of the standard Toro
  two-state Riemann cases, plus the *exact outer wave speeds* as an
  attribution ceiling (a three-wave flux cannot do better than that);
* level B -- accuracy of the star-pressure estimates underlying the
  pressure-based models against the exact ``p*``;
* level C -- isolated Rankine-Hugoniot shocks of increasing strength
  (the case where Davis' estimates are documented to over-diffuse);
* level D -- consistency / mirror-symmetry / isolated-contact identities for
  every model (these must hold to round-off for any admissible estimates);
* level E -- end-to-end solver sensitivity (Toro test 3 strong shock, Lax and
  Sod at P0), which is where a kernel-level gain can and cannot show up.

Everything is written to one JSON file; the markdown report and the figure are
produced by ``benchmarks/report_hllc_wave_speeds.py`` from that file only, so no
number is transcribed by hand.

Usage::

    PYTHONPATH=. python3 benchmarks/verify_hllc_wave_speeds.py \
        --json validation/euler_fluxes/wave_speed_estimates.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from benchmarks.riemann_exact import sample as sample_exact
from benchmarks.riemann_exact import star_state as sample_star
from benchmarks.sod_exact import sample as sod_sample
from benchmarks.verify_flux_properties import (
    REFLECTION,
    STANDARD_CASES,
    exact_wave_speeds,
    hllc_flux_with_speeds,
    max_abs,
    mirror,
    pair,
    physical_flux,
    state,
)
from tensordg.euler_1d import GAMMA, WAVE_SPEEDS, hllc_flux, hllc_rhs, solve_riemann

DTYPE = torch.float64
MODELS = ["davis", "einfeldt", "toro", "toro_iter"]
# Cases whose interface x/t = 0 lies inside a rarefaction fan: a three-wave
# solver cannot reproduce the exact flux there, whatever the wave speeds.
FAN_CASES = {"sod", "123problem", "sonic_rarefaction"}
DIAPHRAGM = 0.5
SAMPLE_TIME = 1.0


# ---------------------------------------------------------------------------
# level A: interface flux against the exact Riemann flux
# ---------------------------------------------------------------------------
def exact_interface_flux(left, right):
    sampled = sample_exact(np.array([DIAPHRAGM]), SAMPLE_TIME, left, right)
    exact_state = state(float(sampled["density"][0]), float(sampled["velocity"][0]),
                        float(sampled["pressure"][0]))
    return physical_flux(exact_state)


def level_a_kernel():
    rows = []
    for name, left, right, description in STANDARD_CASES:
        exact_flux = exact_interface_flux(left, right)
        ul, ur = pair(left, right)
        scale = max(max_abs(physical_flux(ul)), max_abs(physical_flux(ur)),
                    max_abs(exact_flux))
        errors = {model: max_abs(hllc_flux(ul, ur, model) - exact_flux) / scale
                  for model in MODELS}
        speed_l, speed_r = exact_wave_speeds(left, right)
        errors["exact_speeds"] = max_abs(
            hllc_flux_with_speeds(ul, ur, speed_l, speed_r) - exact_flux) / scale
        rows.append({
            "case": name,
            "description": description,
            "inside_rarefaction_fan": name in FAN_CASES,
            "exact_interface_state": {k: float(v[0]) for k, v in
                                      sample_exact(np.array([DIAPHRAGM]), SAMPLE_TIME,
                                                   left, right).items()},
            "scaled_error": errors,
        })
    return rows


def summarise_kernel(rows):
    summary = {}
    for model in MODELS + ["exact_speeds"]:
        all_values = [row["scaled_error"][model] for row in rows]
        recoverable = [row["scaled_error"][model] for row in rows
                       if not row["inside_rarefaction_fan"]]
        summary[model] = {
            "all_median": float(np.median(all_values)),
            "all_max": float(np.max(all_values)),
            "recoverable_median": float(np.median(recoverable)),
            "recoverable_max": float(np.max(recoverable)),
        }
    return summary


# ---------------------------------------------------------------------------
# level B: star-pressure estimates
# ---------------------------------------------------------------------------
def level_b_star_pressure():
    from tensordg.euler_1d import (  # local import keeps the module namespace small
        _pressure_function_and_derivative,
        _two_rarefaction_pressure,
        star_pressure,
        sound_speed,
    )

    rows = []
    for name, left, right, _ in STANDARD_CASES:
        exact_p, exact_u = sample_star(left, right)
        if exact_p <= 0.0:
            continue
        ul, ur = pair(left, right)
        rho_l, u_l, p_l = left
        rho_r, u_r, p_r = right
        a_l = float(sound_speed(ul)[0])
        a_r = float(sound_speed(ur)[0])
        trrs, pvrs = _two_rarefaction_pressure(
            torch.tensor([rho_l], dtype=DTYPE), torch.tensor([u_l], dtype=DTYPE),
            torch.tensor([p_l], dtype=DTYPE), torch.tensor([a_l], dtype=DTYPE),
            torch.tensor([rho_r], dtype=DTYPE), torch.tensor([u_r], dtype=DTYPE),
            torch.tensor([p_r], dtype=DTYPE), torch.tensor([a_r], dtype=DTYPE),
        )
        newton = float(star_pressure(ul, ur)[0])
        rows.append({
            "case": name,
            "exact_p_star": float(exact_p),
            "exact_u_star": float(exact_u),
            "trrs_p_star": float(trrs[0]),
            "trrs_relative_error": abs(float(trrs[0]) - exact_p) / exact_p,
            "pvrs_p_star": float(pvrs[0]),
            "pvrs_relative_error": abs(float(pvrs[0]) - exact_p) / exact_p,
            "newton_p_star": newton,
            "newton_relative_error": abs(newton - exact_p) / exact_p,
        })
    return rows


# ---------------------------------------------------------------------------
# level C: isolated Rankine-Hugoniot shocks
# ---------------------------------------------------------------------------
def rankine_hugoniot_post_shock(mach, rho_1=1.0, p_1=1.0):
    """Exact post-shock state of a normal shock of Mach number ``mach`` (gamma = 1.4)."""
    ratio = (GAMMA + 1.0) * mach ** 2 / ((GAMMA - 1.0) * mach ** 2 + 2.0)
    rho_2 = rho_1 * ratio
    p_2 = p_1 * (1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (mach ** 2 - 1.0))
    sound_1 = np.sqrt(GAMMA * p_1 / rho_1)
    u_2 = mach * sound_1 * (1.0 - 1.0 / ratio)
    return rho_2, u_2, p_2


def level_c_shock_sweep(machs=(1.5, 2.0, 3.0, 5.0, 10.0)):
    rows = []
    for mach in machs:
        rho_2, u_2, p_2 = rankine_hugoniot_post_shock(mach)
        # Upwind (post-shock) state is the physical exact flux; compare the
        # numerical interface flux of (pre, post) against it.
        pre = (1.0, mach * np.sqrt(GAMMA), 1.0)
        post = (float(rho_2), float(u_2), float(p_2))
        ul, ur = pair(pre, post)
        exact_flux = physical_flux(ur)
        scale = max_abs(exact_flux)
        errors = {model: max_abs(hllc_flux(ul, ur, model) - exact_flux) / scale
                  for model in MODELS}
        speed_l, speed_r = exact_wave_speeds(pre, post)
        errors["exact_speeds"] = max_abs(
            hllc_flux_with_speeds(ul, ur, speed_l, speed_r) - exact_flux) / scale
        rows.append({
            "mach": float(mach),
            "post_shock_state": {"density": float(rho_2), "velocity": float(u_2),
                                 "pressure": float(p_2)},
            "scaled_error": errors,
        })
    return rows


# ---------------------------------------------------------------------------
# level D: exact identities every admissible estimate must satisfy
# ---------------------------------------------------------------------------
def level_d_invariants():
    rng = np.random.default_rng(20261103)
    states = [(1.0, 0.0, 1.0), (1.0, 3.0, 0.1), (0.125, 0.0, 0.1),
              (5.99924, 19.5975, 460.894), (1.0, -1.5, 2.5)]
    for _ in range(12):
        states.append((float(rng.uniform(0.05, 5.0)), float(rng.uniform(-4.0, 4.0)),
                       float(rng.uniform(0.02, 50.0))))

    out = {}
    for model in MODELS:
        consistency = max(max_abs(hllc_flux(state(*s), state(*s), model)
                                  - physical_flux(state(*s))) for s in states)
        symmetry = 0.0
        contact = 0.0
        for name, left, right, _ in STANDARD_CASES:
            ul, ur = pair(left, right)
            lhs = REFLECTION * hllc_flux(ul, ur, model)
            rhs = hllc_flux(mirror(ur), mirror(ul), model)
            symmetry = max(symmetry, max_abs(rhs - lhs))
            if abs(left[1] - right[1]) < 1e-15 and abs(left[2] - right[2]) < 1e-15:
                upwind = ul if left[1] > 0.0 else ur
                exact = physical_flux(upwind)
                contact = max(contact, max_abs(hllc_flux(ul, ur, model) - exact)
                              / max_abs(exact))
        out[model] = {"consistency_max_abs": consistency,
                      "mirror_symmetry_max_abs": symmetry,
                      "isolated_contact_max_relative": contact}
    return out


# ---------------------------------------------------------------------------
# level E: end-to-end solver sensitivity
# ---------------------------------------------------------------------------
def _profile_errors(x, fields, exact):
    errors = {}
    for key in ("density", "velocity", "pressure"):
        reference = exact[key]
        errors[key] = float(np.linalg.norm(fields[key] - reference)
                            / np.linalg.norm(reference) * 100.0)
    return errors


def shock_width_cells(x, density):
    """Width of the leading shock in cells, measured as the 10%-90% rise.

    A more sensitive diffusion metric than the L2 error of a first-order
    solution: it isolates the smearing that the interface flux controls.
    """
    gradient = np.abs(np.gradient(density, x))
    index = int(np.argmax(gradient))
    window = slice(max(index - 40, 0), min(index + 41, density.size))
    local = density[window]
    high, low = float(local.max()), float(local.min())
    if high <= low:
        return None
    level_high = low + 0.9 * (high - low)
    level_low = low + 0.1 * (high - low)

    def crossing(level):
        for i in range(index, density.size - 1):
            if density[i] <= level <= density[i + 1]:
                return x[i] + (level - density[i]) / (density[i + 1] - density[i]) * (x[i + 1] - x[i])
        return None

    upper, lower = crossing(level_high), crossing(level_low)
    if upper is None or lower is None:
        return None
    return float((upper - lower) / (x[1] - x[0]))


def level_e_end_to_end(problems, grids=(200, 400), models=("davis", "einfeldt", "toro", "toro_iter")):
    rows = []
    for problem in problems:
        name = problem["name"]
        left, right = problem["left"], problem["right"]
        final_time = problem["final_time"]
        for ncells in grids:
            for model in models:
                start = time.time()
                x, fields, meta = solve_riemann(
                    left, right, ncells=ncells, final_time=final_time,
                    flux="hllc", hllc_wave_speed=model,
                )
                elapsed = time.time() - start
                exact = sod_sample(x.numpy(), final_time, 0.5) if name == "sod" else None
                if exact is None:
                    exact = sample_exact(x.numpy(), final_time, left, right)
                host_fields = {k: fields[k].numpy() for k in
                               ("density", "velocity", "pressure")}
                errors = _profile_errors(x.numpy(), host_fields, exact)
                rows.append({
                    "problem": name, "cells": ncells, "model": model,
                    "steps": meta["steps"], "seconds": elapsed,
                    "relative_l2_percent": errors,
                    "max_relative_l2_percent": max(errors.values()),
                    "shock_width_cells": shock_width_cells(x.numpy(), host_fields["density"]),
                })
                print(f"[end-to-end] {name} n={ncells} {model} "
                      f"max={max(errors.values()):.4f}% ({elapsed:.1f}s)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# conservation / admissibility of the semi-discrete operator
# ---------------------------------------------------------------------------
def conservation_check(model):
    ncells, dx = 200, 1.0 / 200
    rng = np.random.default_rng(7)
    rho = torch.tensor(rng.uniform(0.5, 1.5, ncells), dtype=DTYPE)
    velocity = torch.tensor(rng.uniform(-0.2, 0.2, ncells), dtype=DTYPE)
    pressure = torch.tensor(rng.uniform(0.8, 1.2, ncells), dtype=DTYPE)
    from tensordg.euler_1d import primitive_to_conservative
    u = primitive_to_conservative(rho, velocity, pressure)
    rhs = hllc_rhs(u, dx, wave_speed=model)
    # Transmissive boundaries: mass flux must cancel pairwise except at the ends,
    # so the interior mass balance is a pure telescoping sum.
    mass_total = float(rhs[:, 0].sum() * dx)
    return {"mass_rate_total": mass_total,
            "finite": bool(torch.isfinite(rhs).all())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="validation/euler_fluxes/wave_speed_estimates.json")
    parser.add_argument("--quick", action="store_true",
                        help="skip the end-to-end level (kernel levels only)")
    args = parser.parse_args()

    kernel = level_a_kernel()
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "gamma": GAMMA,
        "models": MODELS,
        "fan_cases": sorted(FAN_CASES),
        "kernel": kernel,
        "kernel_summary": summarise_kernel(kernel),
        "star_pressure": level_b_star_pressure(),
        "shock_sweep": level_c_shock_sweep(),
        "invariants": level_d_invariants(),
        "conservation": {model: conservation_check(model) for model in MODELS},
    }

    if not args.quick:
        problems = [
            {"name": "strong_shock", "left": (5.99924, 19.5975, 460.894),
             "right": (5.99242, -6.19633, 46.095), "final_time": 0.035},
            {"name": "lax", "left": (0.445, 0.698, 3.528),
             "right": (0.5, 0.0, 0.571), "final_time": 0.13},
            {"name": "sod", "left": (1.0, 0.0, 1.0), "right": (0.125, 0.0, 0.1),
             "final_time": 0.2},
        ]
        payload["end_to_end"] = level_e_end_to_end(problems)

    directory = os.path.dirname(args.json)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.json, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()