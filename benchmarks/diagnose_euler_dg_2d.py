"""One-shot diagnosis of the 2-D DG face-flux projection defect.

Run with ``python -m benchmarks.diagnose_euler_dg_2d``.

The Sod tube is extruded to two dimensions (one cell thick, y-invariant data),
so the 2-D residual must reproduce the independently written 1-D residual cell
by cell.  The script prints, for the discontinuity face:

* the HLLC interface mass flux returned by ``_hllc``,
* the same quantity after projection onto the face test functions inside
  ``_axis_fluxes``,
* the reference 1-D density residual of the neighbouring cells.

Measured on 2026-11-02 (mission m2143): interface flux 0.430260, projected
flux 0.0, 1-D reference 43.026 (= 0.430260 / dx).  The Riemann solver is
correct; the projection step is not.
"""
from __future__ import annotations

import torch

from tensordg.euler_dg_2d import (
    _axis_fluxes,
    _hllc,
    _rhs,
    _traces,
    legendre_operators,
)
from tensordg.euler_dg_1d import _operators, _rhs as rhs_1d
from tensordg.euler_1d import GAMMA


def main() -> dict:
    degree, nx, dx, dy = 1, 100, 0.01, 0.25
    device, dtype = torch.device("cpu"), torch.float64
    _, weights, basis, derivative, signs = legendre_operators(degree, 3, device, dtype)

    state = torch.zeros(nx, 1, degree + 1, degree + 1, 4, dtype=dtype)
    state[:50, :, 0, 0, :] = torch.tensor([1.0, 0.0, 0.0, 1.0 / (GAMMA - 1.0)], dtype=dtype)
    state[50:, :, 0, 0, :] = torch.tensor([0.125, 0.0, 0.0, 0.1 / (GAMMA - 1.0)], dtype=dtype)

    trace_right, trace_left = _traces(state, basis, signs, 0)
    neighbour = torch.roll(trace_left, shifts=-1, dims=0)
    neighbour[-1] = trace_right[-1]
    interface = _hllc(trace_right, neighbour, 0)
    projected, _ = _axis_fluxes(state, basis, signs, weights, False, 0)
    residual_2d = _rhs(state, basis, derivative, weights, signs, dx, dy, (False, True))

    nodes, weights_1d, phi, dphi, phi_left, phi_right = _operators(degree, device, dtype)
    state_1d = torch.zeros(nx, degree + 1, 3, dtype=dtype)
    state_1d[:50, 0, :] = torch.tensor([1.0, 0.0, 1.0 / (GAMMA - 1.0)], dtype=dtype)
    state_1d[50:, 0, :] = torch.tensor([0.125, 0.0, 0.1 / (GAMMA - 1.0)], dtype=dtype)
    residual_1d = rhs_1d(state_1d, dx, weights_1d, phi, dphi, phi_left, phi_right)

    face = 49
    report = {
        "interface_mass_flux": float(interface[face, 0, 0, 0]),
        "projected_mass_flux": float(projected[face, 0, 0, 0]),
        "residual_1d_density_cell49": float(residual_1d[face, 0, 0]),
        "residual_2d_density_cell49": float(residual_2d[face, 0, 0, 0, 0]),
        "residual_1d_density_max": float(residual_1d[:, 0, 0].abs().max()),
        "residual_2d_density_max": float(residual_2d[..., 0, 0, 0].abs().max()),
        "residual_1d_energy_max": float(residual_1d[:, 0, 2].abs().max()),
        "residual_2d_energy_max": float(residual_2d[..., 0, 0, 3].abs().max()),
    }
    report["projection_lost_fraction"] = 1.0 - (
        report["projected_mass_flux"] / report["interface_mass_flux"]
    )
    for key, value in report.items():
        print(f"{key:34s} {value: .8e}")
    return report


if __name__ == "__main__":
    main()
