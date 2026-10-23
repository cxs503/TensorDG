import math

import torch

from tensordg.burgers_1d import burgers_error, solve_burgers


def test_burgers_conserves_mass_and_matches_characteristics():
    x, coeff, meta = solve_burgers(ncells=32, degree=2, final_time=0.1)
    assert torch.isfinite(coeff).all()
    assert abs(coeff[:, 0].mean().item() - 0.2) < 1e-12
    assert burgers_error(x, coeff, 0.1)["relative_l2"] < 0.001
    assert meta["steps"] > 0


def test_burgers_p2_has_near_third_order_convergence():
    errors = []
    for cells in (16, 32, 64):
        x, coeff, _ = solve_burgers(ncells=cells, degree=2, final_time=0.1)
        errors.append(burgers_error(x, coeff, 0.1)["relative_l2"])
    rates = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(rates) > 2.7, (errors, rates)
    assert errors[-1] < 0.03
