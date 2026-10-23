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
