"""``CASE-DG-DALEMBERT-ELASTIC-1D``: 1-D elastic rod, exact d'Alembert profiles.

What the case is
----------------
A homogeneous, linearly elastic rod of length ``L`` is described by the
first-order velocity--stress system solved by :mod:`tensordg.elastic_1d`::

    v_t - sigma_x / rho = 0,      sigma_t - E v_x = 0,

i.e. a symmetric linear hyperbolic system with characteristic speeds
``+-c``, ``c = sqrt(E/rho)``.  Two scenarios are shipped:

``pulse``
    Periodic bar, initial data ``v(x,0) = exp(-((x-L/2)/w)^2)``,
    ``sigma(x,0) = 0``.  d'Alembert splits the pulse into two counter-
    propagating waves of half amplitude, so the exact solution is

        v(x,t)     = (v0(x-ct) + v0(x+ct)) / 2
        sigma(x,t) = rho*c*(v0(x+ct) - v0(x-ct)) / 2 .

    Because that closed form is a pure travelling-wave superposition the case
    contains **no physical dispersion**; any deviation of the numerical profile
    is numerical dispersion/dissipation.  This is the headline metric
    ``sigma(x,t)`` against the closed form with a ``<2%`` gate.

``standing``
    Bar clamped at ``x=0``, free at ``x=L``, released from the first
    longitudinal eigenmode ``u = A sin(kx) cos(omega t)`` with
    ``k = pi/(2L)``, ``omega = c k``.  Used to check reflection rules and the
    exact eigenfrequency ``f1 = c/(4L)``.

Reference
---------
K. F. Graff, *Wave Motion in Elastic Solids* (Ohio State University Press,
1975; Dover reprint 1991), Chapter 1, longitudinal waves in thin rods:
d'Alembert's one-dimensional solution ``f(x-ct) + g(x+ct)``, the bar speed
``c = sqrt(E/rho)`` and the reflection rules at free (``sigma = 0``) and
fixed (``v = 0``) ends.

**Honest attribution.**  The task anchor is Graff 1975 (secondary DOI
``10.1177/058310247901100605``).  No copy of the book is reachable from this
execution environment, therefore *no digit in this module is transcribed from
Graff's tables*: every reference number is computed here from the closed-form
d'Alembert/eigenmode expressions, and ``tests/test_elastic_1d_dalembert.py``
verifies those expressions independently (PDE residual, characteristic
invariance).  Graff is cited for the formulation and the reflection rules
only.  The one independent literature cross-check is the bar speed of steel,
``c = sqrt(E/rho) = 5172.2 m/s`` for ``E = 210 GPa, rho = 7850 kg/m^3``, which
must fall inside the 5100--5200 m/s band quoted for steel in handbooks; it is
computed from ``MATERIALS`` below, not copied.

Usage
-----
>>> from tensordg.cases.elastic_rod_1d import solve_case, parameters
>>> centers, coeff, meta = solve_case("pulse", ncells=100, degree=2)
>>> round(meta["c"], 6)
1.0
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..elastic_1d import (
    dalembert_pulse,
    elastic_wave_speed,
    initial_pulse_stress,
    initial_pulse_velocity,
    initial_standing_stress,
    initial_standing_velocity,
    solve_elastic_1d,
    standing_wave,
    stress_profile,
)

CASE_KEY = "dalembert-elastic-rod-1d"

#: non-dimensional rod (unit length, unit density, unit Young modulus -> c = 1)
LENGTH = 1.0
RHO = 1.0
YOUNG = 1.0
C = elastic_wave_speed(RHO, YOUNG)          # == 1.0

#: Gaussian velocity pulse of unit amplitude
PULSE_CENTER = 0.5
PULSE_WIDTH = 0.05
PULSE_AMPLITUDE = 1.0

#: headline times: the two pulses stay well inside [0, L] for pulse_time
PULSE_TIME = 0.2
STANDING_TIME = 0.5

#: ``reflection`` scenario: a right-going stress pulse started at x = 0.2
#: reaches the fixed end x = L at ``REFLECTION_TIME``
REFLECTION_START = 0.2
REFLECTION_TIME = (LENGTH - REFLECTION_START) / C      # == 0.8
REFLECTION_AMPLITUDE = 1.0
DEFAULT_DEGREE = 2
DEFAULT_CFL = 0.3
MESHES = (50, 100, 200, 400)
STANDING_MESHES = (40, 80, 160)
SCENARIOS = ("pulse", "standing", "reflection")

#: dimensional materials for the literature cross-check of c = sqrt(E/rho)
MATERIALS = {
    "steel": {"E": 210e9, "rho": 7850.0},       # c = 5172.19 m/s
    "aluminium": {"E": 70e9, "rho": 2700.0},    # c = 5091.75 m/s
}


# ---------------------------------------------------------------------------
# exact (reference) solutions -- computed, never transcribed
# ---------------------------------------------------------------------------
def exact_pulse(x, t: float = PULSE_TIME):
    """d'Alembert two-wave solution of the ``pulse`` scenario -> ``(v, sigma)``."""
    x = torch.as_tensor(x, dtype=torch.float64) if not torch.is_tensor(x) else x
    return dalembert_pulse(x, t, rho=RHO, young=YOUNG,
                           center=PULSE_CENTER, width=PULSE_WIDTH)


def exact_standing(x, t: float = STANDING_TIME):
    """First clamped-free eigenmode -> ``(v, sigma, u)``."""
    x = torch.as_tensor(x, dtype=torch.float64) if not torch.is_tensor(x) else x
    return standing_wave(x, t, length=LENGTH, rho=RHO, young=YOUNG,
                         amplitude=PULSE_AMPLITUDE)


def peak_stress_amplitude(amplitude: float = PULSE_AMPLITUDE) -> float:
    """Exact peak stress of one travelling pulse, ``rho*c*A/2``."""
    return 0.5 * RHO * C * amplitude


def front_positions(t: float, center: float = PULSE_CENTER):
    """Exact ``(x_left, x_right)`` of the two d'Alembert wave packets."""
    shift = C * t
    return center - shift, center + shift


def probe_arrival_time(x_probe: float, center: float = PULSE_CENTER) -> float:
    """Exact time at which the pulse *peak* passes ``x_probe``."""
    return abs(x_probe - center) / C


def clamped_free_frequency(mode: int = 1) -> float:
    """Exact eigenfrequency ``f_n = (2n-1) c / (4 L)`` of a clamped-free bar."""
    return (2 * mode - 1) * C / (4.0 * LENGTH)


def reflection_pulse(x: torch.Tensor) -> torch.Tensor:
    """Gaussian stress pulse of the ``reflection`` scenario."""
    x = torch.as_tensor(x, dtype=torch.float64) if not torch.is_tensor(x) else x
    return REFLECTION_AMPLITUDE * torch.exp(
        -((x - REFLECTION_START) / PULSE_WIDTH) ** 2)


def reflection_velocity(x: torch.Tensor) -> torch.Tensor:
    """Matching velocity ``v = -sigma/(rho c)`` -> one purely right-going wave."""
    return -reflection_pulse(x) / (RHO * C)


def initial_state(scenario: str):
    """Initial ``(velocity, stress)`` callables of a scenario."""
    if scenario == "pulse":
        return initial_pulse_velocity, initial_pulse_stress
    if scenario == "reflection":
        return reflection_velocity, reflection_pulse
    if scenario == "standing":
        return (
            initial_standing_velocity,
            lambda x: initial_standing_stress(
                x, length=LENGTH, young=YOUNG, amplitude=PULSE_AMPLITUDE),
        )
    raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")


def boundary_condition(scenario: str) -> str:
    """DG boundary condition tag of a scenario."""
    if scenario == "pulse":
        return "periodic"
    if scenario == "standing":
        return "fixed-free"
    if scenario == "reflection":
        return "free-fixed"          # free at x=0, fixed (v=0) at x=L
    raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")


def solve_case(scenario: str, ncells: int, degree: int = DEFAULT_DEGREE,
               final_time: float | None = None, cfl: float = DEFAULT_CFL):
    """Run one scenario through :func:`tensordg.elastic_1d.solve_elastic_1d`.

    Returns ``(centers, coeff, meta)`` exactly as the solver does; the case
    only fixes the initial data and the boundary condition so that benchmarks,
    tests and downstream callers share one definition.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    if final_time is None:
        final_time = {"pulse": PULSE_TIME, "standing": STANDING_TIME,
                      "reflection": REFLECTION_TIME}[scenario]
    vel, sig = initial_state(scenario)
    return solve_elastic_1d(
        ncells=ncells, degree=degree, final_time=final_time, length=LENGTH,
        rho=RHO, young=YOUNG, cfl=cfl, bc=boundary_condition(scenario),
        initial_velocity=vel, initial_stress=sig,
    )


def sample(scenario: str, ncells: int, degree: int = DEFAULT_DEGREE,
           final_time: float | None = None, cfl: float = DEFAULT_CFL,
           nsample: int | None = None):
    """Convenience wrapper -> ``(x, v, sigma, meta)`` sampled per cell."""
    centers, coeff, meta = solve_case(scenario, ncells, degree, final_time, cfl)
    if nsample is None:
        nsample = max(2 * degree + 3, 6)
    x, v, s = stress_profile(centers, coeff, nsample=nsample)
    return x, v, s, meta


def boundary_values(coeff: torch.Tensor):
    """``(left, right)`` traces of the modal state, each ``tensor([v, sigma])``.

    The traces are what a boundary flux sees, so they are the right quantity
    for reflection checks (the per-cell sampling never lands exactly on x=0/L).
    """
    nmodes = coeff.shape[1]
    signs = torch.tensor([(-1.0) ** k for k in range(nmodes)],
                         dtype=coeff.dtype, device=coeff.device)
    left = torch.einsum("k,kv->v", signs, coeff[0])
    right = torch.einsum("k,kv->v", torch.ones(nmodes, dtype=coeff.dtype,
                                              device=coeff.device), coeff[-1])
    return left, right


def parameters() -> dict:
    """Full non-dimensional parameter table of the case (report section 3)."""
    return {
        "case": CASE_KEY,
        "length": LENGTH, "rho": RHO, "young": YOUNG, "c": C,
        "pulse_amplitude": PULSE_AMPLITUDE,
        "pulse_center": PULSE_CENTER, "pulse_width": PULSE_WIDTH,
        "pulse_time": PULSE_TIME, "standing_time": STANDING_TIME,
        "degree": DEFAULT_DEGREE, "cfl": DEFAULT_CFL,
        "meshes": list(MESHES), "standing_meshes": list(STANDING_MESHES),
        "nondimensional_groups": {
            "pulse_width_over_length": PULSE_WIDTH / LENGTH,
            "travel_distance_over_length": C * PULSE_TIME / LENGTH,
            "pulse_amplitude_over_c": PULSE_AMPLITUDE / C,
            "samples_per_pulse_width": PULSE_WIDTH
            / (LENGTH / MESHES[-1]) / (DEFAULT_DEGREE + 1),
        },
        "material_cross_check": {
            k: {"E": v["E"], "rho": v["rho"],
                "c": elastic_wave_speed(v["rho"], v["E"])}
            for k, v in MATERIALS.items()
        },
        "reference": ("Graff 1975 Ch. 1 (d'Alembert bar waves, c = sqrt(E/rho), "
                      "reflection rules); exact forms recomputed here"),
    }


def anchor_table(t: float = PULSE_TIME) -> list[dict]:
    """Exact anchor values (``ref``) that a numerical run must reproduce."""
    x_l, x_r = front_positions(t)
    return [
        {"name": "bar wave speed c", "unit": "-", "ref": C},
        {"name": "right-going packet position", "unit": "L", "ref": x_r},
        {"name": "left-going packet position", "unit": "L", "ref": x_l},
        {"name": "peak stress of one packet sigma_max", "unit": "-",
         "ref": peak_stress_amplitude()},
        {"name": "clamped-free first eigenfrequency f1", "unit": "1/T",
         "ref": clamped_free_frequency(1)},
        {"name": "wall stress amplification (fixed end)",
         "unit": "-", "ref": 2.0 * REFLECTION_AMPLITUDE},
        {"name": "stress at free end (standing antinode)", "unit": "-", "ref": 0.0},
        {"name": "elastic energy at t (periodic, conserved)", "unit": "-",
         "ref": 0.5 * PULSE_WIDTH * math.sqrt(math.pi / 2.0)},
    ]