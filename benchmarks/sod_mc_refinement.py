"""Run the Sod P2/MC refinement study with python -m benchmarks.sod_mc_refinement."""
import argparse
import json

import numpy as np

from benchmarks.shock_reference_quadrature import cell_averages, relative_l2, wave_positions
from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_1d import solve_euler_dg_1d


def measure(cells, reconstruction="mc"):
    """Return relative primitive errors and absolute conserved-integral residuals."""
    left, right = (1., 0., 1.), (.125, 0., .1)
    time = .2
    _, coeff, meta = solve_euler_dg_1d(
        left, right, ncells=cells, degree=2, final_time=time, cfl=.12,
        limiter="tvb", tvb_constant=1., limiter_stage="final",
        limiter_kwargs={"reconstruction": reconstruction},
    )
    averages = coeff[:, 0].numpy()
    rho, momentum, energy = averages.T
    velocity = momentum / rho
    pressure = (GAMMA - 1.) * (energy - .5 * momentum * velocity)
    centers, reference = cell_averages(cells, time, left, right)
    fields = dict(density=rho, velocity=velocity, pressure=pressure)
    # Waves remain inside [0, 1]. End states have zero mass and energy flux,
    # while the momentum integral gains t * (p_left - p_right).
    expected = np.array([
        .5 * (left[0] + right[0]), time * (left[2] - right[2]),
        .5 * (left[2] + right[2]) / (GAMMA - 1.),
    ])
    residual = np.abs(averages.mean(axis=0) - expected)
    _, conserved = cell_averages(cells, time, left, right, conserved=True)
    ref_velocity = conserved["momentum"] / conserved["density"]
    ref_pressure = (GAMMA - 1.) * (conserved["energy"] - .5 * conserved["momentum"] * ref_velocity)
    matched = dict(density=conserved["density"], velocity=ref_velocity, pressure=ref_pressure)
    shock = wave_positions(left, right, time)["front"]
    near_shock = np.abs(centers - shock) <= 3. / cells
    velocity_sq_error = (velocity - ref_velocity)**2
    return {
        "cells": cells, "steps": meta["steps"],
        "relative_l2": {k: relative_l2(value, reference[k]) for k, value in fields.items()},
        "reconstruction": reconstruction,
        "matched_relative_l2": {k: relative_l2(value, matched[k]) for k, value in fields.items()},
        "shock_velocity_squared_error_fraction": float(velocity_sq_error[near_shock].sum() / velocity_sq_error.sum()),
        "balance_residual": dict(zip(("mass", "momentum", "energy"), residual)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--reconstruction", nargs="+", choices=["minmod", "mc"], default=["mc"])
    args = parser.parse_args()
    print(json.dumps([measure(n, r) for n in args.cells for r in args.reconstruction], indent=2))
