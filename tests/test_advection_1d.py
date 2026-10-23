import math
import torch
from tensordg import solve_advection, l2_error


def test_solution_is_finite_and_mass_conservative():
    centers, coeff, meta = solve_advection(ncells=32, degree=1, final_time=0.2)
    assert torch.isfinite(coeff).all()
    # Integral over the periodic domain is h * sum(cell-average coefficients).
    mass = coeff[:, 0].sum().item() / 32
    assert abs(mass) < 1e-10
    assert meta['steps'] > 0


def test_degree_one_converges_at_expected_rate():
    errors = []
    for n in (20, 40):
        centers, coeff, _ = solve_advection(ncells=n, degree=1, final_time=0.25)
        errors.append(l2_error(centers, coeff, final_time=0.25))
    rate = math.log(errors[0] / errors[1], 2)
    assert rate > 1.7, (errors, rate)
    # The exact sine wave has L2 norm 1/sqrt(2) on the unit interval.
    assert errors[-1] * math.sqrt(2.0) < 0.03


def test_zero_final_time_returns_initial_projection():
    centers, coeff, meta = solve_advection(ncells=16, degree=2, final_time=0.0)
    assert meta['steps'] == 0
    assert torch.isfinite(coeff).all()
