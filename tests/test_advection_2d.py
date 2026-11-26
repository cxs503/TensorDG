import math

import torch

from tensordg import error_2d, solve_advection_2d


def test_mass_conservation_and_negative_velocity():
    x, y, coeff, meta = solve_advection_2d(
        ncells=12, degree=1, final_time=0.1, velocity=(-0.7, -0.4),
    )
    assert torch.isfinite(coeff).all()
    assert abs(coeff[:, :, 0, 0].mean().item() - 1.0) < 1e-12
    assert error_2d(x, y, coeff, 0.1, (-0.7, -0.4))["relative_l2"] < 0.03
    assert meta["steps"] > 0


def test_degree_one_converges_at_second_order():
    errors = []
    for cells in (8, 16):
        x, y, coeff, _ = solve_advection_2d(
            ncells=cells, degree=1, final_time=0.2,
        )
        errors.append(error_2d(x, y, coeff, 0.2)["relative_l2"])
    rate = math.log2(errors[0] / errors[1])
    assert rate > 1.8, (errors, rate)


def test_documented_two_dimensional_case_is_below_three_percent():
    x, y, coeff, _ = solve_advection_2d(
        ncells=24, degree=1, final_time=1.0,
    )
    metrics = error_2d(x, y, coeff, 1.0)
    assert metrics["relative_l2"] < 0.03, metrics


def test_oblique_custom_wave_on_rectangular_mesh(record_property):
    def wave(x, y):
        return .4 + torch.cos(2 * math.pi * (x + 2*y))

    errors = []
    for nx, ny in ((6, 10), (12, 20), (24, 40)):
        vel = (-.8, .3)
        x, y, coefficients, _ = solve_advection_2d((nx, ny), 2, .08, vel, initial=wave)
        error = error_2d(x, y, coefficients, .08, vel, initial=wave)["absolute_l2"]
        errors.append(error)
        record_property(f"mesh_{nx}x{ny}_absolute_l2", error)
        assert abs(coefficients[:, :, 0, 0].mean().item() - .4) < 1e-13
    assert all(math.log2(coarse/fine) > 2.7 for coarse, fine in zip(errors, errors[1:]))


def test_long_periodic_translation_preserves_projected_piecewise_bilinear_wave():
    def profile(x, y):
        assert torch.all((x >= 0) & (x < 1) & (y >= 0) & (y < 1))
        return x*y

    x, y, coeff, _ = solve_advection_2d((8, 12), 1, 0., initial=profile)
    shifted = coeff.roll((2, -3), dims=(0, 1))
    assert error_2d(x, y, shifted, 2**40+.25, (1., -1.), initial=profile)["absolute_l2"] < 1e-14


def test_scalar_zero_profile_and_relative_norm():
    zero = lambda x, y: 0.
    x, y, coeff, _ = solve_advection_2d((4, 6), 1, .1, initial=zero)
    assert error_2d(x, y, coeff, .1, initial=zero)["relative_l2"] == 0.
    coeff[:, :, 0, 0] = 1.
    assert math.isinf(error_2d(x, y, coeff, .1, initial=zero)["relative_l2"])


def test_translation_on_physical_rectangle(record_property):
    lengths, velocity, t = (3., .5), (-.75, .25), .125

    def profile(x, y):
        return 1 + torch.sin(2*math.pi*x/3)*torch.sin(4*math.pi*y)

    errors = []
    for n in (6, 12, 24):
        x, y, coeff, meta = solve_advection_2d((n, n+2), 2, t, velocity, lengths=lengths)
        assert meta["lengths"] == lengths
        assert abs(x[-1].item() - (3.-1.5/n)) < 1e-14
        assert abs(y[0].item() - .25/(n+2)) < 1e-14
        mass_error = abs(coeff[..., 0, 0].mean().item()*1.5 - 1.5)
        assert mass_error < 1e-13
        metrics = error_2d(x, y, coeff, t, velocity, initial=profile, lengths=lengths)
        assert abs(metrics["exact_l2"] - math.sqrt(1.5*1.25)) < 1e-13
        errors.append(metrics["absolute_l2"])
        record_property(f"n{n}_absolute_l2", errors[-1])
        record_property(f"n{n}_mass_error", mass_error)
    assert all(math.log2(a/b) > 2.5 for a,b in zip(errors, errors[1:]))


def test_rectangular_exact_translation_crosses_both_seams():
    def profile(x, y):
        assert torch.all((x >= 0) & (x < 2) & (y >= 0) & (y < .5))
        return x*y

    lengths = (2., .5)
    x, y, coeff, _ = solve_advection_2d((8, 8), 1, 0., lengths=lengths, initial=profile)
    rolled = coeff.roll((1, -2), dims=(0, 1))
    metrics = error_2d(x, y, rolled, 2**40+.25, (1., -.5), lengths=lengths, initial=profile)
    assert metrics["absolute_l2"] < 1e-14
