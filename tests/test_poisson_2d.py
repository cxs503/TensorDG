import math

import torch

from tensordg import poisson_error_2d, solve_poisson_2d


def test_sipg_degree_one_converges_and_meets_three_percent_tolerance():
    errors = []
    for cells in (8, 16):
        x, y, coeff, meta = solve_poisson_2d(ncells=cells, degree=1)
        assert torch.isfinite(coeff).all()
        assert meta["matrix_symmetry_error"] < 1e-11
        errors.append(poisson_error_2d(x, y, coeff)["relative_l2"])
    rate = math.log2(errors[0] / errors[1])
    assert rate > 1.8, (errors, rate)
    assert errors[-1] < 0.03, errors


def test_sipg_degree_two_has_third_order_accuracy():
    errors = []
    for cells in (4, 8):
        x, y, coeff, _ = solve_poisson_2d(ncells=cells, degree=2)
        errors.append(poisson_error_2d(x, y, coeff)["relative_l2"])
    rate = math.log2(errors[0] / errors[1])
    assert rate > 2.7, (errors, rate)
