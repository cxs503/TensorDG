"""Transient wall heat flux of the 2-D compressible Navier--Stokes DG solver.

Verification debt this closes (reports m3614 §5 局限 4 / §6 lead 1, m2831/m2876):
the wall **heat-flux** channel of ``tensordg/ns_dg_2d.py`` had *steady* evidence
only (linear conduction field, ``wall_gradient="face"``, 32 cells < 3 %).  The
wall **shear** channel now has a literature benchmark (Stokes' first problem);
the equivalent unsteady thermal benchmark was still missing.

Physical problem
----------------
A gas at rest and at uniform temperature ``T0`` fills a channel ``0 <= y <= H``.
At ``t = 0+`` the bottom wall temperature is raised to ``T_w = T0 + dT`` and held
there; the top wall stays at ``T0`` and is placed far enough away
(``H >> 2 sqrt(alpha t)``) to be invisible.  The leading-order (constant
pressure) balance is the heat equation ``dT/dt = alpha d2T/dy2`` with
``alpha = kappa / (rho0 c_p)``, whose semi-infinite-medium solution is

    T(y, t) = T0 + dT erfc(eta),   eta = y / (2 sqrt(alpha t))
    q_w(t)  = -kappa dT/dy|_(y=0) = kappa dT / sqrt(pi alpha t)   (>0, into fluid)

``q_w`` is extracted from exactly the quantity the scheme uses at the wall: the
energy component of the DG boundary numerical flux ``_boundary_flux`` at the
``ymin`` face (the same estimator audited in
``validation/ns_dg_2d_wall/gradient_fix.json`` and m3139).

Parameter regime (a trap this benchmark fell into first, see the report)
-----------------------------------------------------------------------
The erfc solution additionally assumes the *pressure* response to be
quasi-static, i.e. the acoustic time across the thermal layer
``2 sqrt(alpha t) / c_s`` must be much smaller than ``t``, equivalently

    t  >>  4 alpha / c_s**2  =  4 mu / (gamma * p0 * Pr)

With ``mu = 0.05`` and ``t = 0.2`` that ratio is ~1: the layer is still
acoustically adjusting and the wall flux is biased **low by 5.6 %** with no mesh
or degree dependence (a parameter-regime artefact, not a discretisation error).
``mu = 0.01`` with ``t = 0.7`` puts the ratio at ~17 and the bias collapses to
sub-percent, so those are the defaults below.

Honest scope note
-----------------
The solver integrates the *compressible* system, so the erfc profile is an
exact solution only up to ``O(dT/T0)``: the isobaric heating forces a weak
acoustic/expansion velocity field of that order.  The model error is therefore
measured, not assumed: the amplitude ladder ``dT/T0 = 0.02, 0.01, 0.005`` shows
the wall-flux error falling proportionally to ``dT/T0``, and the
``dT/T0 -> 0`` extrapolation removes it.  What remains is the discretisation
error, which is what the mesh/degree ladder measures.

Usage::

    PYTHONPATH=. OMP_NUM_THREADS=1 python -m benchmarks.verify_ns_transient_wall_heat
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_dg_2d import GAMMA, legendre_operators

MU = 0.01
PRANDTL = 0.71
CP = GAMMA / (GAMMA - 1.0)
KAPPA = MU * CP / PRANDTL          # kappa = mu c_p / Pr (solver convention)
T0 = 1.0
DENSITY0 = 1.0                     # p = rho T with R = 1 and p0 = 1
ALPHA = KAPPA / (DENSITY0 * CP)    # = MU / (DENSITY0 * PRANDTL)
HEIGHT = 1.0
FINAL_TIME = 0.7
DTYPE = torch.float64
OUTPUT_DIR = os.path.join("validation", "ns_dg_2d_transient_wall")
FIGURE = os.path.join("docs", "figures", "ns_dg_2d_transient_wall_heat.png")

# reports to the caller which channels are exercised
FIELDS = ("density", "velocity_x", "velocity_y", "temperature")


# ---------------------------------------------------------------------------
# Boundary data and analytic reference
# ---------------------------------------------------------------------------

def hot_wall_spec(amplitude, time_on=0.0):
    """Time-dependent wall state: ``T = T0 + amplitude`` for ``t > time_on``.

    Returned as a callable so that ``solve_navier_stokes_dg_2d`` evaluates the
    Dirichlet data at every SSP-RK stage time (m3640 feature).
    """

    def spec(time_value):
        value = T0 + amplitude if float(time_value) > time_on else T0
        return {"velocity": (0.0, 0.0), "temperature": float(value)}

    spec.__name__ = "hot_wall_spec"
    return spec


def cold_wall_spec():
    return {"velocity": (0.0, 0.0), "temperature": T0}


def erfc_temperature(y, time_value, amplitude):
    """Semi-infinite-medium temperature profile ``T0 + dT erfc(y/2 sqrt(alpha t))``."""
    eta = np.asarray(y, dtype=float) / (2.0 * math.sqrt(ALPHA * float(time_value)))
    return T0 + float(amplitude) * np.vectorize(math.erfc)(eta)


def exact_wall_heat_flux(time_value, amplitude):
    """``q_w = kappa dT / sqrt(pi alpha t)`` — heat flux from the wall into the gas."""
    return KAPPA * float(amplitude) / math.sqrt(math.pi * ALPHA * float(time_value))


def exact_flux_selfcheck(time_value=FINAL_TIME, amplitude=0.02, delta=1.0e-6):
    """Protect the reference: differentiate the erfc profile numerically.

    ``-kappa dT/dy`` at ``y = 0`` from a one-sided second-order difference of the
    *analytic* field must reproduce :func:`exact_wall_heat_flux`.  If the closed
    form were wrong, every convergence study below would silently measure
    nothing.
    """
    samples = erfc_temperature(np.array([0.0, delta, 2.0 * delta]), time_value, amplitude)
    derivative = (-3.0 * samples[0] + 4.0 * samples[1] - samples[2]) / (2.0 * delta)
    numerical = -KAPPA * derivative
    reference = exact_wall_heat_flux(time_value, amplitude)
    relative = abs(numerical - reference) / abs(reference)
    return {
        "finite_difference": float(numerical),
        "closed_form": float(reference),
        "relative_difference": float(relative),
        "delta": delta,
        "pass": bool(relative < 1.0e-4),
    }


def top_boundary_influence(time_value, height=HEIGHT):
    """Residual erfc weight of the top wall (must be negligible for the claim)."""
    return float(math.erfc(height / (2.0 * math.sqrt(ALPHA * float(time_value)))))


# ---------------------------------------------------------------------------
# Single numerical case
# ---------------------------------------------------------------------------

def _operators(degree):
    order = max(degree + 2, 3)
    return legendre_operators(degree, order, torch.device("cpu"), DTYPE)


def wall_heat_flux(coeff, degree, ny, wall_state, ncells_x=1):
    """Energy component of the DG boundary flux at the ``ymin`` wall.

    Mirrors ``tensordg/ns_dg_2d.py``: the integrated wall heat flux of the weak
    form is exactly this face flux, so it is the quantity the scheme itself
    claims.
    """
    nodes, weights, basis, derivative, signs = _operators(degree)
    dy = HEIGHT / ny
    dx = 1.0 / ncells_x
    penalty = 2.0 * (degree + 1.0) ** 2
    left = ns._face_primitives(coeff, basis, derivative, signs, 1, "left")
    flux = ns._boundary_flux(
        left, wall_state, 1, dy, MU, KAPPA, penalty,
        2.0 / dy, 2.0 / dx, outward_sign=-1.0,
    )
    return float(flux[0, 0, 0, 3])


_CACHE = {}


def run_case(degree=1, ny=16, amplitude=0.01, final_time=FINAL_TIME, nx=1):
    """March the sudden-wall-heating problem and compare with the erfc solution."""
    key = (int(degree), int(ny), float(amplitude), float(final_time), int(nx))
    if key in _CACHE:
        return _CACHE[key]
    started = time.time()

    def initial(x, y):
        ones = torch.ones_like(x * y)
        zeros = torch.zeros_like(ones)
        return ones, zeros, zeros, torch.full_like(ones, T0)  # p = rho T = 1

    hot = hot_wall_spec(amplitude)
    boundary = {"ymin": hot, "ymax": cold_wall_spec()}
    x, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
        initial,
        ncells=(nx, ny),
        degree=degree,
        domain=((0.0, 1.0), (0.0, HEIGHT)),
        final_time=final_time,
        mu=MU,
        prandtl=PRANDTL,
        periodic=(True, False),
        boundary=boundary,
        device="cpu",
        dtype=DTYPE,
    )
    reached = float(meta["final_time_reached"])
    averaged = coeff[..., 0, 0, :]
    rho, u, v, pressure = ns._primitive(averaged)
    temperature = (pressure / rho).detach().numpy().reshape(-1)
    y_centres = (y.detach().numpy().reshape(-1))
    exact_profile = erfc_temperature(y_centres, reached, amplitude)

    denominator = float(np.linalg.norm(exact_profile))
    profile_error = float(np.linalg.norm(temperature - exact_profile) / denominator)

    flux = wall_heat_flux(coeff, degree, ny, hot(reached), ncells_x=nx)
    flux_exact = exact_wall_heat_flux(reached, amplitude)
    return {
        "degree": degree,
        "ncells": [nx, ny],
        "amplitude": amplitude,
        "final_time": reached,
        "steps": int(meta["steps"]),
        "dt": float(meta["dt"]),
        "wall_heat_flux": flux,
        "wall_heat_flux_exact": flux_exact,
        "wall_heat_flux_relative_error": abs(flux - flux_exact) / abs(flux_exact),
        "profile_relative_l2": profile_error,
        "max_abs_velocity_y": float(v.abs().max()),
        "max_abs_temperature_error": float(np.max(np.abs(temperature - exact_profile))),
        "y_centres": [float(value) for value in y_centres],
        "temperature": [float(value) for value in temperature],
        "temperature_exact": [float(value) for value in exact_profile],
        "seconds": time.time() - started,
    }
    _CACHE[key] = row
    return row


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

def amplitude_study(amplitudes=(0.02, 0.01, 0.005), degree=1, ny=16,
                    final_time=FINAL_TIME):
    """Does the wall-flux error scale like the low-Mach model error ``dT/T0``?"""
    rows = [run_case(degree=degree, ny=ny, amplitude=float(a), final_time=final_time)
            for a in amplitudes]
    for row in rows:
        row["model_error_scale"] = row["amplitude"] / T0
    # Linear extrapolation of q_w / dT to dT -> 0 (removes the O(dT/T0) model term)
    ratios = np.array([row["wall_heat_flux"] / row["amplitude"] for row in rows])
    amps = np.array([row["amplitude"] for row in rows], dtype=float)
    slope, intercept = np.polyfit(amps, ratios, 1)
    limit = float(intercept)
    target = exact_wall_heat_flux(final_time, 1.0)  # flux per unit dT
    return {
        "cases": rows,
        "flux_per_amplitude": [float(value) for value in ratios],
        "zero_amplitude_limit": limit,
        "target_flux_per_amplitude": target,
        "extrapolation_relative_error": abs(limit - target) / target,
        "raw_relative_errors": [row["wall_heat_flux_relative_error"] for row in rows],
        "error_ratio_finest_over_coarsest": (
            rows[-1]["wall_heat_flux_relative_error"] / rows[0]["wall_heat_flux_relative_error"]
        ),
        "amplitude_ratio": amplitudes[-1] / amplitudes[0],
    }


def mesh_study(degree_levels=None, amplitude=0.005, final_time=FINAL_TIME):
    """Mesh/degree ladder at fixed physical time, amplitude chosen low-Mach.

    ``degree_levels`` maps the polynomial degree to the ``ny`` ladder it is run
    on: the P2 rung is deliberately coarser because the explicit diffusive step
    scales like ``(p+1)**4 / h**2`` (see the module docstring).
    """
    if degree_levels is None:
        degree_levels = {1: (8, 16, 24), 2: (8, 12)}
    rows = [
        run_case(degree=int(degree), ny=int(ny), amplitude=amplitude,
                 final_time=final_time)
        for degree in sorted(degree_levels)
        for ny in degree_levels[degree]
    ]
    orders = {}
    for degree, levels in degree_levels.items():
        subset = [row for row in rows if row["degree"] == degree]
        errors = [row["wall_heat_flux_relative_error"] for row in subset]
        ratio = float(subset[-1]["ncells"][1] / subset[0]["ncells"][1])
        step = ratio ** (1.0 / (len(subset) - 1))
        orders[str(degree)] = float(
            np.mean([math.log(errors[i] / errors[i + 1]) / math.log(step)
                     for i in range(len(errors) - 1)])
        )
    return {"cases": rows, "observed_order": orders, "amplitude": amplitude,
            "degree_levels": {str(k): list(v) for k, v in degree_levels.items()}}


def quasi_static_ratio(final_time=FINAL_TIME):
    """Acoustic transit time across the layer divided by the physical time.

    Must be << 1 for the constant-pressure erfc reference to apply.
    """
    sound_speed = math.sqrt(GAMMA * T0 / DENSITY0)  # p0 = rho0 T0
    layer = 2.0 * math.sqrt(ALPHA * float(final_time))
    return layer / sound_speed / float(final_time)


def stability_note(final_time=FINAL_TIME):
    return {
        "alpha": ALPHA,
        "kappa": KAPPA,
        "diffusion_length": 2.0 * math.sqrt(ALPHA * final_time),
        "top_wall_residual_erfc": top_boundary_influence(final_time),
        "quasi_static_ratio": quasi_static_ratio(final_time),
        "sound_speed": math.sqrt(GAMMA * T0 / DENSITY0),
        "criterion": ("top-wall image negligible (<1e-6) and acoustic transit "
                      "across the layer << t (ratio < 0.1)"),
        "pass": bool(top_boundary_influence(final_time) < 1.0e-6
                     and quasi_static_ratio(final_time) < 0.1),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))

    ax = axes[0]
    for row in results["mesh"]["cases"]:
        ax.plot(row["temperature_exact"], row["y_centres"], "-", color="0.55",
                linewidth=1.0,
                label="exact erfc" if row is results["mesh"]["cases"][0] else None)
    for row in results["mesh"]["cases"]:
        label = "P%d %d cells (rel. L2 %.2f %%)" % (
            row["degree"], row["ncells"][1], 100.0 * row["profile_relative_l2"])
        ax.plot(row["temperature"], row["y_centres"], "o-", markersize=2.6,
                linewidth=1.0, label=label)
    ax.set_xlabel("T(y, t=%.2f)" % results["mesh"]["cases"][0]["final_time"])
    ax.set_ylabel("y")
    ax.set_title("temperature profile vs erfc (dT = %.3f)" % results["mesh"]["amplitude"])
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for degree in sorted(int(key) for key in results["mesh"]["degree_levels"]):
        subset = [row for row in results["mesh"]["cases"] if row["degree"] == degree]
        errors = [max(row["wall_heat_flux_relative_error"], 1e-16) for row in subset]
        ax.loglog([row["ncells"][1] for row in subset], errors, "o-",
                  label="P%d: EOC %.2f" % (degree, results["mesh"]["observed_order"][str(degree)]))
    ax.set_xlabel("cells in y")
    ax.set_ylabel("relative error of q_w")
    ax.set_title("wall heat flux error vs mesh (dT = %.3f)" % results["mesh"]["amplitude"])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[2]
    amplitudes = [row["amplitude"] for row in results["amplitude"]["cases"]]
    errors = [100.0 * row["wall_heat_flux_relative_error"] for row in results["amplitude"]["cases"]]
    ax.plot(amplitudes, errors, "s-", label="numerical q_w error")
    ax.plot(amplitudes, [100.0 * a / T0 for a in amplitudes], "k--",
            label="O(dT/T0) model-error guide")
    ax.set_xlabel("dT / T0")
    ax.set_ylabel("relative error of q_w  [%]")
    ax.set_title("low-Mach model error is linear in dT/T0")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    figure.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None):
    argv = list(argv or [])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "parameters": {
            "mu": MU, "prandtl": PRANDTL, "kappa": KAPPA, "alpha": ALPHA,
            "T0": T0, "density0": DENSITY0, "height": HEIGHT,
            "final_time": FINAL_TIME, "gamma": GAMMA,
        },
        "reference_selfcheck": exact_flux_selfcheck(),
        "top_wall": stability_note(),
    }
    results["amplitude"] = amplitude_study()
    results["mesh"] = mesh_study()
    created = []
    for name, payload in (("amplitude_study.json", results["amplitude"]),
                          ("mesh_study.json", results["mesh"]),
                          ("summary.json", results)):
        target = os.path.join(OUTPUT_DIR, name)
        slim = payload
        if "--slim" in argv and isinstance(payload, dict) and "cases" in payload:
            slim = dict(payload)
            slim["cases"] = [{k: v for k, v in row.items()
                              if k not in ("y_centres", "temperature", "temperature_exact")}
                             for row in payload["cases"]]
        with open(target, "w") as handle:
            json.dump(slim, handle, indent=2, default=float)
        created.append(target)
    results["figure"] = make_figure(results, FIGURE)
    created.append(FIGURE)

    print("reference self-check:", results["reference_selfcheck"]["relative_difference"])
    print("top wall residual:", results["top_wall"]["top_wall_residual_erfc"])
    for row in results["amplitude"]["cases"]:
        print("dT=%.3f  q_w=%.6f  exact=%.6f  rel=%6.3f%%  max|v|=%.2e  %.1fs"
              % (row["amplitude"], row["wall_heat_flux"], row["wall_heat_flux_exact"],
                 100.0 * row["wall_heat_flux_relative_error"], row["max_abs_velocity_y"],
                 row["seconds"]))
    print("zero-amplitude limit %.6f vs exact %.6f (%.3f%%)"
          % (results["amplitude"]["zero_amplitude_limit"],
             results["amplitude"]["target_flux_per_amplitude"],
             100.0 * results["amplitude"]["extrapolation_relative_error"]))
    for row in results["mesh"]["cases"]:
        print("P%d ny=%2d  q_w rel=%7.3f%%  profile rel-L2=%7.3f%%  steps=%d %.1fs"
              % (row["degree"], row["ncells"][1],
                 100.0 * row["wall_heat_flux_relative_error"],
                 100.0 * row["profile_relative_l2"], row["steps"], row["seconds"]))
    print("observed order:", results["mesh"]["observed_order"])
    print("written:", created)
    return results


if __name__ == "__main__":
    main()