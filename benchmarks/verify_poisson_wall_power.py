"""SIPG field and conservative wall power for a mixed-boundary exponential solution."""
import json
import math

import torch

from tensordg.poisson_2d_bc import l2_error_2d, poisson_face_powers_2d, solve_poisson_2d_bc


def measure(n, degree):
    def exact(x, y):
        return torch.exp(x+.5*y)

    data = dict(bc={"top":"neumann"}, neumann_flux={"top":lambda x,y: .5*exact(x,y)},
                dirichlet_values={side:exact for side in ("left","right","bottom")})
    x, y, coeff, _ = solve_poisson_2d_bc(n, degree, forcing=lambda x,y: -1.25*exact(x,y), **data)
    qx, qy = poisson_face_powers_2d(coeff, **data)
    field = l2_error_2d(x, y, coeff, exact)["relative_l2"]
    faces = torch.arange(n+1, dtype=x.dtype)/n
    # Analytic integrals, with q = -grad(T) and both orientations positive along the axis.
    ix = torch.exp(x)*(2*math.sinh(.5/n))
    iy = torch.exp(.5*y)*(4*math.sinh(.25/n))
    target_x = -torch.exp(faces[:,None])*iy
    target_y = -.5*ix[:,None]*torch.exp(.5*faces)
    wall_error = n*max((qx[[0,-1]]-target_x[[0,-1]]).abs().max(),
                       (qy[:,[0,-1]]-target_y[:,[0,-1]]).abs().max())
    residual = qx.diff(dim=0)+qy.diff(dim=1)+1.25*ix[:,None]*iy
    return field, float(wall_error), float(residual.abs().max())


if __name__ == "__main__":
    runs = []
    for degree in (1,2,3):
        previous = None
        for n in (3,6,12):
            field, wall, balance = measure(n, degree)
            errors = (field, wall)
            rates = [math.log2(a/b) for a,b in zip(previous, errors)] if previous else []
            ok = balance < 1e-10 and all(math.isfinite(e) for e in errors)
            if rates:
                ok &= rates[0] > degree+.8 and rates[1] > degree-.2
            runs.append(dict(degree=degree, cells_per_side=n, relative_L2_temperature=field,
                             max_wall_mean_flux_error=wall, max_cell_balance=balance, orders=rates, passed=ok))
            previous = errors
    print(json.dumps(runs, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
