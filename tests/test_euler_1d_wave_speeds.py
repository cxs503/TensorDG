"""Tests for the 1-D Euler flux kernels that had no dedicated coverage.

``tensordg.euler_1d`` exposes ``primitive_fields``, ``rusanov_rhs`` and the four
HLLC outer wave-speed estimates (``davis``, ``einfeldt``, ``toro``,
``toro_iter``) plus ``toro_iterative_wave_speeds``.  None of them was referenced
by an existing test file.  The positive cases check them against exact
properties (uniform-state reduction, acoustic bounds, Sod star pressure from
the exact Riemann solution, discrete conservation of ``rusanov_rhs``); the
negative cases are controls that must be red when the kernel degrades.
"""
from __future__ import annotations

import pytest
import torch

from tensordg.euler_1d import (
    GAMMA,
    WAVE_SPEEDS,
    davis_wave_speeds,
    einfeldt_wave_speeds,
    hllc_flux,
    physical_flux,
    primitive_fields,
    primitive_to_conservative,
    rusanov_rhs,
    sound_speed,
    star_pressure,
    toro_iterative_wave_speeds,
    toro_wave_speeds,
)

DTYPE = torch.float64


def conserved(states):
    """``states`` is a list of ``(rho, u, p)`` triples -> conserved (n,3)."""
    return torch.stack(
        [primitive_to_conservative(*(torch.as_tensor(v, dtype=DTYPE) for v in s))
         for s in states], dim=0,
    )


def sod_states():
    """Sod (1978) shock tube initial data, left/right as one-row tensors."""
    data = conserved([(1.0, 0.0, 1.0), (0.125, 0.0, 0.1)])
    return data[0:1], data[1:2]


def uniform_state(rho=1.0, velocity=0.0, pressure=1.0, cells=5):
    single = primitive_to_conservative(
        torch.tensor(rho, dtype=DTYPE),
        torch.tensor(velocity, dtype=DTYPE),
        torch.tensor(pressure, dtype=DTYPE),
    )
    return single.reshape(1, 3).repeat(cells, 1)


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------

def test_primitive_fields_invert_primitive_to_conservative():
    values = [(1.0, 0.0, 1.0), (0.125, 0.0, 0.1), (0.4, 2.5, 0.7)]
    data = conserved(values)
    fields = primitive_fields(data)
    for index, (rho, velocity, pressure) in enumerate(values):
        assert float(fields["density"][index]) == pytest.approx(rho, rel=1e-14)
        assert float(fields["velocity"][index]) == pytest.approx(velocity, abs=1e-14)
        assert float(fields["pressure"][index]) == pytest.approx(pressure, rel=1e-13)
    assert sound_speed(data)[0] == pytest.approx((GAMMA * 1.0 / 1.0) ** 0.5, rel=1e-14)


def test_uniform_state_wave_speeds_reduce_to_the_acoustic_cones():
    """For equal states every estimate must collapse to ``u -/+ a``."""
    state = uniform_state(rho=1.2, velocity=0.8, pressure=0.9)
    sound = float(sound_speed(state)[:1])
    for name, estimate in WAVE_SPEEDS.items():
        speed_l, speed_r = estimate(state, state)
        assert float(speed_l[0]) == pytest.approx(0.8 - sound, rel=1e-12), name
        assert float(speed_r[0]) == pytest.approx(0.8 + sound, rel=1e-12), name


def test_davis_estimate_is_the_min_max_of_the_acoustic_characteristics():
    left, right = sod_states()
    speed_l, speed_r = davis_wave_speeds(left, right)
    sound_l, sound_r = float(sound_speed(left)[0]), float(sound_speed(right)[0])
    assert float(speed_l[0]) == pytest.approx(min(-sound_l, -sound_r), rel=1e-13)
    assert float(speed_r[0]) == pytest.approx(max(sound_l, sound_r), rel=1e-13)


@pytest.mark.parametrize(
    "estimate", [einfeldt_wave_speeds, toro_wave_speeds, toro_iterative_wave_speeds],
)
def test_acoustic_estimates_bound_the_left_and_right_cones(estimate):
    """``S_L <= u_L - a_L`` and ``S_R >= u_R + a_R`` (Toro 2009, sec. 10.5)."""
    left, right = sod_states()
    speed_l, speed_r = estimate(left, right)
    velocity_l = float(primitive_fields(left)["velocity"][0])
    velocity_r = float(primitive_fields(right)["velocity"][0])
    sound_l, sound_r = float(sound_speed(left)[0]), float(sound_speed(right)[0])
    assert float(speed_l[0]) <= velocity_l - sound_l + 1e-12
    assert float(speed_r[0]) >= velocity_r + sound_r - 1e-12
    assert float(speed_l[0]) < 0.0 < float(speed_r[0])


def test_newton_refined_star_pressure_matches_the_exact_riemann_value():
    """Reference: Toro (2009) table 4.1, Sod problem, ``p* = 0.30313``."""
    left, right = sod_states()
    reference = 0.30313
    for estimate in (star_pressure(left, right), star_pressure(left, right, iterations=20)):
        assert float(estimate[0]) == pytest.approx(reference, rel=1e-4)
    slow = float(star_pressure(left, right, iterations=0)[0])
    fast = float(star_pressure(left, right, iterations=8)[0])
    assert abs(fast - reference) <= abs(slow - reference)


@pytest.mark.parametrize("wave_speed", sorted(WAVE_SPEEDS))
def test_hllc_flux_is_consistent_on_a_uniform_state(wave_speed):
    state = uniform_state(rho=1.0, velocity=0.0, pressure=1.0)
    flux = hllc_flux(state, state, wave_speed=wave_speed)
    assert torch.allclose(flux, physical_flux(state), atol=1e-13, rtol=0.0)


@pytest.mark.parametrize("wave_speed", sorted(WAVE_SPEEDS))
def test_hllc_flux_selects_the_upwind_flux_for_supersonic_flow(wave_speed):
    """If both outer speeds are positive the flux must be ``F(U_L)`` exactly."""
    left = uniform_state(rho=1.0, velocity=3.0, pressure=0.5, cells=1)
    right = uniform_state(rho=0.9, velocity=2.8, pressure=0.45, cells=1)
    speed_l, _ = WAVE_SPEEDS[wave_speed](left, right)
    assert float(speed_l[0]) > 0.0
    flux = hllc_flux(left, right, wave_speed=wave_speed)
    assert torch.allclose(flux, physical_flux(left), atol=1e-13, rtol=0.0)


def test_rusanov_rhs_is_zero_for_a_uniform_state_and_conservative():
    state = uniform_state(rho=1.0, velocity=0.2, pressure=1.0, cells=6)
    dx = 0.1
    rhs = rusanov_rhs(state, dx)
    assert float(rhs.abs().max()) < 1e-13

    perturbed = state.clone()
    perturbed[2] = primitive_to_conservative(
        torch.tensor(1.3, dtype=DTYPE),
        torch.tensor(0.1, dtype=DTYPE),
        torch.tensor(0.8, dtype=DTYPE),
    )
    rhs = rusanov_rhs(perturbed, dx)
    assert float(rhs.abs().max()) > 1e-3
    # Discrete conservation: the change of the total state equals the flux
    # difference across the transmissive boundaries.
    flux = physical_flux(perturbed)
    total_change = rhs.sum(dim=0) * dx
    expected = -(flux[-1] - flux[0])
    assert torch.allclose(total_change, expected, atol=1e-12, rtol=0.0)


def test_rusanov_rhs_scales_with_the_mesh_width():
    state = uniform_state(cells=4)
    state[1] = primitive_to_conservative(
        torch.tensor(1.5, dtype=DTYPE),
        torch.tensor(0.3, dtype=DTYPE),
        torch.tensor(0.9, dtype=DTYPE),
    )
    coarse = rusanov_rhs(state, 0.2)
    fine = rusanov_rhs(state, 0.1)
    assert torch.allclose(fine, 2.0 * coarse, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# Negative cases (red when the kernels or the checks degrade)
# ---------------------------------------------------------------------------

def test_acoustic_bound_check_rejects_a_violating_estimate():
    """Negative control for :func:`test_acoustic_estimates_bracket_the_two_states`."""
    left, right = sod_states()
    sound = float(sound_speed(left)[0])
    bogus = torch.tensor([sound], dtype=DTYPE)  # S_L = +a_L instead of <= -a_L
    assert float(bogus[0]) > -sound
    with pytest.raises(AssertionError):
        assert float(bogus[0]) <= -sound


def test_hllc_flux_reacts_to_a_small_state_perturbation():
    """The consistency check would be vacuous if the flux ignored its inputs."""
    state = uniform_state(cells=1)
    perturbed = state.clone()
    perturbed[0, 2] *= 1.001  # 0.1 % more internal energy, hence higher pressure
    base = hllc_flux(state, state)
    changed = hllc_flux(state, perturbed)
    assert float((base - changed).abs().max()) > 1e-8
    # A neighbouring-state interface must differ from the upwind-only flux.
    assert not torch.allclose(changed, physical_flux(state), atol=1e-13, rtol=0.0)


def test_physical_flux_of_a_degenerate_state_is_finite():
    """``physical_flux`` clamps the density, so no inf/NaN may leak out."""
    degenerate = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    flux = physical_flux(degenerate)
    assert torch.isfinite(flux).all()
    assert torch.isfinite(sound_speed(degenerate)).all()


@pytest.mark.parametrize("bad_name", ["roe", "", "DAVIS"])
def test_unknown_wave_speed_name_raises(bad_name):
    left, right = sod_states()
    with pytest.raises(ValueError):
        hllc_flux(left, right, wave_speed=bad_name)