import math

import torch

from tensordg.heat_1d import heat_error, solve_heat_1d


def test_heat_sipg_p1_converges_at_second_order():
    errors = []
    for cells in (8, 16):
        x, coeff, meta = solve_heat_1d(ncells=cells, degree=1, final_time=0.1)
        assert meta["diffusion_symmetry_error"] < 1e-11
        errors.append(heat_error(x, coeff, 0.1)["relative_l2"])
    rate = math.log2(errors[0] / errors[1])
    assert rate > 1.8, (errors, rate)


def test_documented_heat_case_is_below_three_percent():
    x, coeff, _ = solve_heat_1d(ncells=16, degree=2, final_time=0.1)
    assert torch.isfinite(coeff).all()
    metrics = heat_error(x, coeff, 0.1)
    assert metrics["relative_l2"] < 0.03, metrics
