"""Tests for the 3-D tensor-product DG advection module.

The reference is the closed-form translated separable field of
:mod:`tensordg.advection_3d`, evaluated by Gauss quadrature -- no stored
numeric profile is used anywhere, so a regression that only rescales the
solution cannot pass.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg.advection_3d import (
    error_3d, exact_solution_3d, mass_3d, solve_advection_3d,
)

VELOCITY = (1.0, 0.5, 0.25)


def _run(ncells: int, degree: int, final_time: float,
         velocity=VELOCITY) -> tuple[dict[str, float], float, dict]:
    x, y, z, coeff, meta = solve_advection_3d(
        ncells=ncells, degree=degree, final_time=final_time,
        velocity=velocity,
    )
    metrics = error_3d(x, y, z, coeff, final_time, velocity=velocity)
    drift = abs(mass_3d(coeff) - 1.0)
    assert torch.all(torch.isfinite(coeff))
    return metrics, drift, meta


def _eoc(coarse: float, fine: float, ratio: float) -> float:
    return math.log(coarse / fine) / math.log(ratio)


def test_mass_is_conserved_to_round_off():
    """Upwind fluxes telescope; the analytic integral of the field is 1."""
    for velocity in (VELOCITY, (-1.0, -0.5, 0.25), (1.0, 1.0, 1.0)):
        _, drift, meta = _run(4, 1, 0.1, velocity=velocity)
        assert meta["steps"] > 0
        assert drift < 1e-12, drift


def test_p1_converges_at_second_order():
    coarse, _, _ = _run(4, 1, 0.1)
    fine, _, _ = _run(8, 1, 0.1)
    order = _eoc(coarse["relative_l2"], fine["relative_l2"], 2.0)
    assert 1.7 < order < 2.3, order


def test_p2_converges_at_third_order():
    coarse, _, _ = _run(4, 2, 0.25)
    fine, _, _ = _run(8, 2, 0.25)
    order = _eoc(coarse["relative_l2"], fine["relative_l2"], 2.0)
    assert 2.55 < order < 3.45, order


def test_initial_projection_alone_has_design_order():
    """With ``final_time = 0`` the only error is the L2 projection."""
    coarse, _, meta = _run(4, 2, 0.0)
    fine, _, _ = _run(8, 2, 0.0)
    assert meta["steps"] == 0
    order = _eoc(coarse["relative_l2"], fine["relative_l2"], 2.0)
    assert 2.7 < order < 3.3, order


def test_documented_case_is_below_three_percent():
    p1, _, _ = _run(8, 1, 0.25)
    p2, _, _ = _run(8, 2, 0.25)
    assert 100.0 * p1["relative_l2"] < 3.0, p1
    assert 100.0 * p2["relative_l2"] < 3.0, p2
    # and the higher degree must actually be more accurate on the same mesh
    assert p2["relative_l2"] < 0.2 * p1["relative_l2"]


def test_negative_velocity_components_conserve_and_converge():
    coarse, drift_c, _ = _run(4, 1, 0.1, velocity=(-1.0, -0.5, -0.25))
    fine, drift_f, _ = _run(8, 1, 0.1, velocity=(-1.0, -0.5, -0.25))
    assert max(drift_c, drift_f) < 1e-12
    assert _eoc(coarse["relative_l2"], fine["relative_l2"], 2.0) > 1.7


def test_zero_velocity_leaves_the_projection_untouched():
    _, _, meta = _run(4, 2, 1.0, velocity=(0.0, 0.0, 0.0))
    assert meta["steps"] == 0 and meta["dt"] == 0.0


def test_exact_solution_has_unit_mean_and_is_a_translation():
    nodes = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)[:-1]
    x = nodes[:, None, None]
    y = nodes[None, :, None]
    z = nodes[None, None, :]
    value = exact_solution_3d(x, y, z, 0.3, VELOCITY)
    assert float(value.mean().item()) == pytest.approx(1.0, abs=2e-3)
    shifted = exact_solution_3d(
        x - VELOCITY[0] * 0.3, y - VELOCITY[1] * 0.3, z - VELOCITY[2] * 0.3,
        0.0, VELOCITY,
    )
    assert torch.allclose(value, shifted, atol=1e-12)


def test_anisotropic_mesh_and_degree_zero_are_supported():
    """P0 on a stretched hexahedral mesh: conservative and first order."""
    coarse, drift_c, meta = _run((6, 4, 3), 0, 0.05)
    fine, drift_f, _ = _run((12, 8, 6), 0, 0.05)
    assert meta["ncells"] == (6, 4, 3)
    assert max(drift_c, drift_f) < 1e-12
    order = _eoc(coarse["relative_l2"], fine["relative_l2"], 2.0)
    # P0 upwind on this coarse stretched ladder is still pre-asymptotic.
    assert 0.6 < order < 1.4, order


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ncells": 1, "degree": 1},
        {"ncells": 4, "degree": 9},
        {"ncells": 4, "degree": 1, "final_time": -1.0},
        {"ncells": 4, "degree": 1, "cfl": 0.0},
        {"ncells": 4, "degree": 1, "velocity": (1.0, 1.0)},
    ],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_advection_3d(**kwargs)

def test_oblique_custom_wave_on_cuboid(record_property):
    def wave(x, y, z):
        return .4 + torch.cos(2*math.pi*(x-y+z))

    errors = []
    velocity = (-.4, .25, .6)
    for grid in ((4, 5, 6), (8, 10, 12), (16, 20, 24)):
        x, y, z, coeff, _ = solve_advection_3d(grid, 2, .04, velocity, initial=wave)
        error = error_3d(x, y, z, coeff, .04, velocity, initial=wave)["absolute_l2"]
        errors.append(error)
        record_property(f"grid_{grid}_L2", error)
        assert abs(mass_3d(coeff) - .4) < 1e-13
    assert all(math.log2(a/b) > 2.7 for a,b in zip(errors, errors[1:]))


def test_periodic_trilinear_translation_and_zero_reference():
    def profile(x, y, z):
        assert all(torch.all((v >= 0) & (v < 1)) for v in (x, y, z))
        return x*y*z

    x, y, z, coeff, _ = solve_advection_3d((4, 8, 12), 1, 0., initial=profile)
    velocity = (1., -1., 1.)
    shifted = coeff.roll((1, -2, 3), dims=(0, 1, 2))
    time = 2**40 + .25
    assert error_3d(x, y, z, shifted, time, velocity, initial=profile)["absolute_l2"] < 1e-14
    sample = exact_solution_3d(x, y[0], z[0], time, velocity, initial=profile)
    torch.testing.assert_close(sample, ((x-.25)%1)*((y[0]+.25)%1)*((z[0]-.25)%1))
    assert error_3d(x, y, z, coeff, 0., initial=lambda x,y,z: 0.)["relative_l2"] == math.inf


def test_physical_cuboid_transport_against_characteristics(record_property):
    lengths = (2., 0.75, 3.)
    volume = math.prod(lengths)
    velocity = (0.5, -0.375, 0.75)
    time = 0.08
    def wave(x, y, z):
        return 1 + torch.cos(2 * math.pi * (x/2 - y/0.75 + z/3))

    def transported(x, y, z):
        return wave(x - velocity[0]*time, y - velocity[1]*time, z - velocity[2]*time)

    errors = []
    for n in (4, 8, 16):
        x, y, z, c, meta = solve_advection_3d(
            (n, n+1, n+2), 2, time, velocity, initial=wave, lengths=lengths)
        metrics = error_3d(x, y, z, c, 0., initial=transported)
        measured = error_3d(x, y, z, c, time, velocity, initial=wave, lengths=lengths)
        assert measured["absolute_l2"] == pytest.approx(metrics["absolute_l2"], abs=2e-14)
        assert measured["exact_l2"] == pytest.approx(math.sqrt(1.5 * volume), rel=1e-12)
        assert mass_3d(c, lengths=lengths) == pytest.approx(volume, abs=1e-13)
        assert meta["lengths"] == lengths
        errors.append(metrics["absolute_l2"])
        record_property(f"absolute_l2_n{n}", errors[-1])
    assert min(math.log2(a/b) for a,b in zip(errors, errors[1:])) > 2.0


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_physical_axis_profile_wraps_after_many_box_crossings(axis, record_property):
    lengths = (2., .5, 4.)
    velocity = (-2., .5, 4.)
    cells = (8, 4, 12)
    def profile(x, y, z):
        for q, side in zip((x, y, z), lengths):
            assert torch.all((q >= 0) & (q < side))
        return (x, y, z)[axis]

    x, y, z, c, _ = solve_advection_3d(cells, 1, 0, velocity, initial=profile, lengths=lengths)
    shifted = c.roll((-2, 1, 3), dims=(0, 1, 2))
    error = error_3d(x, y, z, shifted, 2**40 + .25, velocity, initial=profile, lengths=lengths)
    record_property("absolute_l2", error["absolute_l2"])
    assert error["absolute_l2"] < 2e-14
    assert mass_3d(c, lengths=lengths) == pytest.approx(math.prod(lengths)*lengths[axis]/2)
    point = exact_solution_3d(x[0], y[0], z[0], 2**40 + .25, velocity, initial=profile, lengths=lengths)
    assert point.item() == pytest.approx(((x[0], y[0], z[0])[axis].item() - .25*velocity[axis]) % lengths[axis])



def test_long_translation_retains_fractional_turn_before_product_rounding(record_property):
    sides = (1., 3., 2.)
    velocity = (1.5, -1.5, 1.)
    def field(x, y, z):
        return x + y - z
    x, y, z, coeff, _ = solve_advection_3d((4, 6, 4), 1, 0., lengths=sides, initial=field)
    # t=2^52+1 translates by +1/2, -1/2 and +1/2 turns respectively.
    shifted = coeff.roll((2, -3, 2), dims=(0, 1, 2))
    time = float(2**52 + 1)
    metrics = error_3d(x, y, z, shifted, time, velocity, initial=field, lengths=sides)
    record_property("absolute_l2", metrics["absolute_l2"])
    assert metrics["absolute_l2"] < 2e-14
    # 2^2046 has residue 1 modulo 3, although the unreduced product overflows float64.
    huge = float(2**1023)
    sample = torch.tensor([0.25], dtype=torch.float64)
    result = exact_solution_3d(sample, sample, sample, huge, (0., huge, 0.),
                               initial=lambda x,y,z: y, lengths=sides)
    torch.testing.assert_close(result, torch.tensor([2.25], dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("degree", [4, 5, 6, 7, 8])
def test_high_order_modes_transport_a_translated_wave(degree, record_property):
    velocity = (-0.5, 0.25, 0.375)
    errors = []
    for n in ((3, 6, 12) if degree == 4 else (3, 6)):
        x, y, z, coeff, _ = solve_advection_3d(n, degree, 0.03, velocity)
        metrics = error_3d(x, y, z, coeff, 0.03, velocity)
        errors.append(metrics["absolute_l2"])
        drift = abs(mass_3d(coeff) - 1.0)
        record_property(f"absolute_l2_{n}", errors[-1])
        record_property(f"mass_error_{n}", drift)
        assert drift < 2e-14
    assert min(math.log2(a/b) for a, b in zip(errors, errors[1:])) > degree + 0.5



@pytest.mark.parametrize(("method", "order", "stages"), [
    ("euler", 1, 1), ("rk2", 2, 2), ("rk3", 3, 3), ("ssp_rk3", 3, 3),
    ("rk4", 4, 4), ("ssp_rk54", 4, 5), ("rk5", 5, 6),
])
def test_temporal_order_against_exact_semidiscrete_mode(method, order, stages, record_property):
    import cmath

    # At P0 the Cartesian upwind operator acts diagonally on a Fourier mode.
    # Use its exponential to isolate time error at fixed spatial resolution.
    velocity = (0.25, -0.125, 0.125)
    phases = (math.pi/2, -math.pi/2, math.pi/2)
    eigenvalue = sum(4 * abs(a) * (cmath.exp(-1j * math.copysign(1, a) * phase) - 1)
                     for a, phase in zip(velocity, phases))
    def wave(x, y, z):
        return 1 + torch.cos(2 * math.pi * (x - y + z))

    _, _, _, initial, _ = solve_advection_3d(4, 0, 0., velocity, initial=wave)
    cosine = initial - 1
    sine = cosine.roll(1, dims=0)
    factor = cmath.exp(eigenvalue)
    exact = 1 + factor.real * cosine - factor.imag * sine
    errors = []
    for steps in (8, 16, 32, 64):
        _, _, _, coeff, meta = solve_advection_3d(
            4, 0, 1., velocity, cfl=2/steps, initial=wave, time_integrator=method)
        assert (meta["steps"], meta["rk_order"], meta["rk_stages"]) == (steps, order, stages)
        errors.append(float((coeff - exact).square().mean().sqrt()))
        record_property(f"rms_error_{steps}", errors[-1])
        assert abs(mass_3d(coeff) - 1) < 2e-14
    measured = math.log2(errors[-2] / errors[-1])
    record_property("final_observed_order", measured)
    assert abs(measured - order) < 0.2
