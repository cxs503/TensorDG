import torch

from benchmarks.lax_1d import LEFT, RIGHT
from benchmarks.riemann_dg_error import riemann_dg_errors
from benchmarks.riemann_exact import star_state
from tensordg.euler_dg_1d import solve_euler_dg_1d


def test_lax_star_state_and_short_time_positivity():
    pressure, velocity = star_state(LEFT, RIGHT)
    assert abs(pressure - 2.4660979192073564) < 1e-12
    assert abs(velocity - 1.528723026632884) < 1e-12
    _, coeff, _ = solve_euler_dg_1d(LEFT, RIGHT, ncells=40, degree=2, final_time=0.005)
    assert torch.isfinite(coeff).all()
    average = coeff[:, 0]
    pressure_avg = 0.4 * (average[:, 2] - 0.5 * average[:, 1] ** 2 / average[:, 0])
    assert (average[:, 0] > 0).all()
    assert (pressure_avg > 0).all()


def test_documented_lax_case_matches_exact_riemann_within_three_percent():
    x, coeff, _ = solve_euler_dg_1d(
        LEFT, RIGHT, ncells=400, degree=3, final_time=0.13, cfl=0.35,
    )
    errors = riemann_dg_errors(x, coeff, 0.13, LEFT, RIGHT)
    for key, metrics in errors.items():
        assert metrics["relative_l2"] < 0.03, (key, metrics)
