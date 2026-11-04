"""Regression tests for the HLLC outer wave-speed estimates.

``docs/euler_flux_kernel_verification.md`` measured a median 11.3% (max 24.2%)
error of ``hllc_flux`` against the exact Riemann interface flux when Davis'
wave speeds are used, i.e. outside the project's 3% gate, and filed improved
estimates as follow-up work.  These tests pin the four selectable models in
``tensordg.euler_1d.WAVE_SPEEDS``:

* the exact identities (consistency, mirror symmetry, isolated-contact
  exactness) hold to round-off for *every* model,
* the Newton-refined star pressure reproduces the exact ``p*``,
* on the cases where a three-wave solver can reproduce the flux at all
  (everything except the rarefaction-fan cases) ``toro_iter`` is inside 3%,
  while Davis' estimates are not -- the historical defect stays pinned,
* the ordering ``davis > einfeldt > toro_iter`` holds on the strong shock,
* a bad model name is rejected by both the kernel and the solver.

Evidence: ``docs/hllc_wave_speed_verification.md``,
``benchmarks/verify_hllc_wave_speeds.py``,
``validation/euler_fluxes/wave_speed_estimates.json``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.riemann_exact import sample as sample_exact
from benchmarks.riemann_exact import star_state as sample_star
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
from tensordg.euler_1d import (
    WAVE_SPEEDS,
    hllc_flux,
    hllc_rhs,
    primitive_to_conservative,
    solve_riemann,
    star_pressure,
)

DTYPE = torch.float64
MODELS = ["davis", "einfeldt", "toro", "toro_iter"]
FAN_CASES = {"sod", "123problem", "sonic_rarefaction"}
CASES = {name: (left, right) for name, left, right, _ in STANDARD_CASES}


def exact_interface_flux(left, right):
    sampled = sample_exact(np.array([0.5]), 1.0, left, right)
    conserved = state(float(sampled["density"][0]), float(sampled["velocity"][0]),
                      float(sampled["pressure"][0]))
    return physical_flux(conserved)


def scaled_kernel_error(name, model):
    left, right = CASES[name]
    ul, ur = pair(left, right)
    exact = exact_interface_flux(left, right)
    scale = max(max_abs(physical_flux(ul)), max_abs(physical_flux(ur)), max_abs(exact))
    return max_abs(hllc_flux(ul, ur, model) - exact) / scale


# ---------------------------------------------------------------------------
# exact identities, every model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("case", [(1.0, 0.0, 1.0), (0.125, 0.0, 0.1),
                                  (5.99924, 19.5975, 460.894), (1.0, -1.5, 2.5)])
def test_consistency_holds_for_every_model(model, case):
    u = state(*case)
    assert max_abs(hllc_flux(u, u, model) - physical_flux(u)) < 1e-12


@pytest.mark.parametrize("model", MODELS)
def test_mirror_symmetry_holds_for_every_model(model):
    for left, right in CASES.values():
        ul, ur = pair(left, right)
        lhs = REFLECTION * hllc_flux(ul, ur, model)
        rhs = hllc_flux(mirror(ur), mirror(ul), model)
        assert max_abs(rhs - lhs) < 1e-12


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("name", ["stationary_contact", "moving_contact",
                                  "left_moving_contact"])
def test_isolated_contact_is_exact_for_every_model(model, name):
    left, right = CASES[name]
    ul, ur = pair(left, right)
    upwind = ul if left[1] > 0.0 else ur
    exact = physical_flux(upwind)
    assert max_abs(hllc_flux(ul, ur, model) - exact) / max_abs(exact) < 1e-12


# ---------------------------------------------------------------------------
# star pressure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["lax", "strong_shock"])
def test_newton_star_pressure_matches_exact(name):
    left, right = CASES[name]
    exact_p, _ = sample_star(left, right)
    refined = float(star_pressure(*pair(left, right))[0])
    assert abs(refined - exact_p) / exact_p < 1e-6


def test_two_rarefaction_pressure_is_off_for_strong_shock():
    """TRRS is +37% on Toro's test 3 -- documented, and why Newton exists."""
    from tensordg.euler_1d import _two_rarefaction_pressure, sound_speed

    left, right = CASES["strong_shock"]
    ul, ur = pair(left, right)
    estimate = float(_two_rarefaction_pressure(
        torch.tensor([left[0]], dtype=DTYPE), torch.tensor([left[1]], dtype=DTYPE),
        torch.tensor([left[2]], dtype=DTYPE), sound_speed(ul),
        torch.tensor([right[0]], dtype=DTYPE), torch.tensor([right[1]], dtype=DTYPE),
        torch.tensor([right[2]], dtype=DTYPE), sound_speed(ur),
    )[0][0])
    exact_p, _ = sample_star(left, right)
    assert abs(estimate - exact_p) / exact_p > 0.1


# ---------------------------------------------------------------------------
# the 3% gate on the cases a three-wave solver can resolve
# ---------------------------------------------------------------------------
def test_davis_wave_speeds_fail_the_three_percent_gate():
    """The historical defect must stay visible: >3% on the strong shock."""
    assert scaled_kernel_error("strong_shock", "davis") > 0.03


def test_newton_refined_wave_speeds_pass_the_three_percent_gate():
    for name in CASES:
        if name in FAN_CASES:
            continue
        assert scaled_kernel_error(name, "toro_iter") < 0.03, name


@pytest.mark.parametrize("name", ["lax", "strong_shock"])
def test_error_ordering_improves_with_better_estimates(name):
    error = {model: scaled_kernel_error(name, model) for model in MODELS}
    assert error["toro_iter"] <= error["einfeldt"] <= error["davis"]
    assert error["toro_iter"] < 0.25 * error["davis"] + 1e-12


@pytest.mark.parametrize("name", ["lax", "strong_shock", "stationary_contact"])
def test_newton_wave_speeds_reach_the_exact_speed_ceiling(name):
    """A three-wave flux cannot beat the exact-speed flux; it must match it."""
    left, right = CASES[name]
    ul, ur = pair(left, right)
    exact = exact_interface_flux(left, right)
    scale = max(max_abs(physical_flux(ul)), max_abs(physical_flux(ur)), max_abs(exact))
    speed_l, speed_r = exact_wave_speeds(left, right)
    ceiling = max_abs(hllc_flux_with_speeds(ul, ur, speed_l, speed_r) - exact) / scale
    assert scaled_kernel_error(name, "toro_iter") <= ceiling * 1.05 + 1e-12


# ---------------------------------------------------------------------------
# plumbing: validation, mass balance, end-to-end run
# ---------------------------------------------------------------------------
def test_unknown_wave_speed_is_rejected():
    with pytest.raises(ValueError):
        hllc_flux(state(1.0, 0.0, 1.0), state(0.125, 0.0, 0.1), "roe_typo")
    with pytest.raises(ValueError):
        solve_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=64,
                      final_time=0.01, hllc_wave_speed="roe_typo")


@pytest.mark.parametrize("model", MODELS)
def test_uniform_state_has_zero_rhs(model):
    ncells, dx = 64, 1.0 / 64
    u = primitive_to_conservative(torch.full((ncells,), 1.0, dtype=DTYPE),
                                  torch.zeros(ncells, dtype=DTYPE),
                                  torch.full((ncells,), 1.0, dtype=DTYPE))
    rhs = hllc_rhs(u, dx, wave_speed=model)
    assert max_abs(rhs) < 1e-12


@pytest.mark.parametrize("model", MODELS)
def test_solver_runs_and_is_finite_for_every_model(model):
    x, fields, meta = solve_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1),
                                    ncells=200, final_time=0.05,
                                    hllc_wave_speed=model)
    assert meta["hllc_wave_speed"] == model
    assert torch.isfinite(fields["conservative"]).all()
    assert float(fields["density"].min()) > 0.0
    # Total mass is conserved by the transmissive-boundary P0 operator.
    mass = float(fields["conservative"][:, 0].sum() * (x[1] - x[0]))
    assert abs(mass - (0.5 * 1.0 + 0.5 * 0.125)) / (0.5 * 1.125) < 1e-12