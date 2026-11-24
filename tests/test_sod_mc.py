"""Sod shock tube using TVB detection and monotonized central reconstruction."""
import numpy as np

from benchmarks.shock_reference_quadrature import cell_averages
from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_1d import solve_euler_dg_1d


def test_mc_sod_matches_wave_split_riemann_reference():
    left, right = (1., 0., 1.), (.125, 0., .1)
    cells, time = 400, .2
    _, coeff, _ = solve_euler_dg_1d(
        left, right, ncells=cells, degree=2, final_time=time, cfl=.12,
        limiter="tvb", tvb_constant=1., limiter_stage="final",
        limiter_kwargs={"reconstruction": "mc"},
    )
    rho, momentum, energy = coeff[:, 0].numpy().T
    velocity = momentum / rho
    pressure = (GAMMA - 1.) * (energy - .5 * momentum * velocity)
    _, reference = cell_averages(cells, time, left, right)
    for field, value in zip(("density", "velocity", "pressure"), (rho, velocity, pressure)):
        exact = reference[field]
        error = np.linalg.norm(value - exact) / np.linalg.norm(exact)
        assert error < .03, (field, error)
