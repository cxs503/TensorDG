"""Probe: does a recovered wall gradient remove the first-order Dirichlet wall defect?

Measures, for both wall closures (``one_sided`` = historical mirror ghost,
``recovered`` = quadratic extrapolation from the cell centre to the wall):

1. the strong-form residual of the *exact* steady Couette state (should vanish
   as h -> 0 for a consistent wall operator; today it is O(1) and constant);
2. the order of the wall energy flux against the analytic ``-kappa dT/dy``;
3. whether the impulsively started Couette case marches to ``t = 0.1``.

Run: PYTHONPATH=. python tools/probe_wall_closure.py
"""
import json
import math
import sys

import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_dg_2d import legendre_operators

MU = 0.1
KAPPA = MU * ns.CP / 0.71
DTYPE = torch.float64
DEVICE = torch.device("cpu")


def operators(degree, ncells):
    nx, ny = ncells
    order = max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, order, DEVICE, DTYPE
    )
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, device=DEVICE, dtype=DTYPE) + 0.5) * dx
    y = (torch.arange(ny, device=DEVICE, dtype=DTYPE) + 0.5) * dy
    return dict(nodes=nodes, weights=weights, basis=basis, derivative=derivative,
                signs=signs, dx=dx, dy=dy, x=x, y=y)


def couette_state(ncells=(2, 8), degree=1, wall_speed=1.0):
    op = operators(degree, ncells)
    amplitude = MU * wall_speed**2 / (2.0 * KAPPA)

    def initial(x, y):
        u = wall_speed * y
        zeros = torch.zeros_like(u)
        temperature = 1.0 + amplitude * y * (1.0 - y)
        return 1.0 / temperature, u, zeros, torch.ones_like(u)

    coeff = ns._initial_coefficients(initial, op["x"], op["y"], op["dx"], op["dy"],
                                     op["nodes"], op["weights"], op["basis"],
                                     DEVICE, DTYPE)
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
                "ymax": {"velocity": (wall_speed, 0.0), "temperature": 1.0}}
    return op, coeff, boundary


def residual(ny, degree=1, closure="one_sided", wall_penalty=None):
    op, coeff, boundary = couette_state((2, ny), degree)
    penalty = 2.0 * (degree + 1.0) ** 2
    rhs = ns._rhs_ns(coeff, op["basis"], op["derivative"], op["weights"], op["signs"],
                     op["dx"], op["dy"], (True, False), boundary, MU, KAPPA,
                     penalty, closure, wall_penalty)
    return float(rhs.abs().max())


def wall_fluxes(ny, degree=1, closure="one_sided", wall_penalty=None):
    """Wall heat flux on the bottom (y=0) wall, as the operator reports it."""
    op, coeff, boundary = couette_state((2, ny), degree)
    penalty = 2.0 * (degree + 1.0) ** 2
    wall_sigma = penalty if wall_penalty is None else wall_penalty
    h, normal_scale = op["dy"], 2.0 / op["dy"]
    side = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1,
                               "left")
    recovered = None
    if closure == "recovered":
        centre = ns._face_primitives(coeff, op["basis"], op["derivative"],
                                     op["signs"], 1, "centre")
        # average gradient on the interior face of the wall cell
        right = ns._face_primitives(coeff, op["basis"], op["derivative"],
                                    op["signs"], 1, "right")
        left = ns._face_primitives(coeff, op["basis"], op["derivative"],
                                   op["signs"], 1, "left")
        outer = {k: torch.roll(v, shifts=-1, dims=1) for k, v in left.items()}
        names = {"u_n": "u", "v_n": "v", "T_n": "temperature"}
        averaged = {}
        for key, name in names.items():
            averaged[key] = (0.5 * (right[key] + outer[key]) * normal_scale
                             - (penalty / h) * (right[name] - outer[name]))
        # sign: the +axis normal is +y, so the wall cell's interior face is '+'.
        recovered = ns._recover_wall_gradient(centre, averaged, normal_scale, 1,
                                              "left")
    flux = ns._boundary_flux(side, boundary["ymin"], 1, h, MU, KAPPA, wall_sigma,
                             normal_scale, 2.0 / op["dx"], recovered=recovered)
    return float(flux[:, 0, :, 3].mean())


def transient(ny=8, degree=1, closure="one_sided", final_time=0.1,
              wall_penalty=None):
    try:
        ns.solve_couette_dg_2d(ncells=(2, ny), degree=degree, mu=MU,
                               final_time=final_time, device=DEVICE, dtype=DTYPE,
                               wall_gradient=closure, wall_penalty=wall_penalty)
        return "reached"
    except FloatingPointError as error:
        return "FloatingPointError: %s" % error


def main():
    out = {}
    analytic = -KAPPA * (MU * 1.0**2 / (2.0 * KAPPA))
    cases = [("one_sided", 1, None), ("recovered", 1, None), ("recovered", 1, 1.0),
             ("recovered", 1, 0.0)]
    for closure, degree, wp in cases:
        block = {"degree": degree, "wall_penalty": wp}
        block["residual"] = {ny: residual(ny, degree, closure, wp)
                             for ny in (8, 16, 32)}
        errors = {ny: abs(wall_fluxes(ny, degree, closure, wp) - analytic)
                  for ny in (8, 16, 32)}
        block["wall_flux_error"] = errors
        block["wall_flux_order"] = math.log2(errors[8] / errors[32]) / 2.0
        block["transient"] = transient(degree=degree, closure=closure, wall_penalty=wp)
        out["%s|p%d|sigma_w=%s" % (closure, degree, wp)] = block
    payload = {"analytic_wall_flux": analytic, "closures": out,
               "mu": MU, "kappa": KAPPA, "prandtl": 0.71,
               "wall_speed": 1.0, "penalty_rule": "2*(p+1)^2"}
    with open("validation/wall_closure_probe.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(main())