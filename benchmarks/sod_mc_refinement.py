"""Run the Sod P2/MC refinement study with python -m benchmarks.sod_mc_refinement."""
import json

import numpy as np

from benchmarks.shock_reference_quadrature import cell_averages, relative_l2
from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_1d import solve_euler_dg_1d


def measure(cells):
    """Return relative primitive errors and absolute conserved-integral residuals."""
    left, right = (1., 0., 1.), (.125, 0., .1)
    time = .2
    _, coeff, meta = solve_euler_dg_1d(
        left, right, ncells=cells, degree=2, final_time=time, cfl=.12,
        limiter="tvb", tvb_constant=1., limiter_stage="final",
        limiter_kwargs={"reconstruction": "mc"},
    )
    averages = coeff[:, 0].numpy()
    rho, momentum, energy = averages.T
    velocity = momentum / rho
    pressure = (GAMMA - 1.) * (energy - .5 * momentum * velocity)
    _, reference = cell_averages(cells, time, left, right)
    fields = dict(density=rho, velocity=velocity, pressure=pressure)
    # Waves remain inside [0, 1]. End states have zero mass and energy flux,
    # while the momentum integral gains t * (p_left - p_right).
    expected = np.array([
        .5 * (left[0] + right[0]), time * (left[2] - right[2]),
        .5 * (left[2] + right[2]) / (GAMMA - 1.),
    ])
    residual = np.abs(averages.mean(axis=0) - expected)
    return {
        "cells": cells, "steps": meta["steps"],
        "relative_l2": {k: relative_l2(value, reference[k]) for k, value in fields.items()},
        "balance_residual": dict(zip(("mass", "momentum", "energy"), residual)),
    }


if __name__ == "__main__":
    print(json.dumps([measure(n) for n in (100, 200, 400)], indent=2))
