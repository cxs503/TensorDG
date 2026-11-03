"""Verification tests for the Dirichlet (wall) path of :mod:`tensordg.ns_dg_2d`.

The m2496 delivery shipped the BR1/SIPG viscous solver with the wall-bounded
*transient* case marked unverified.  The audit behind
``docs/reports/tensordg-ns-wall-verification.md`` (and ``tools/diag_ns_wall.py``)
localised the reason; the tests here keep the evidence reproducible:

* :func:`test_wall_flux_is_exact_for_polynomial_data` -- **green**: for a state
  that the DG space represents exactly the wall traction and heat flux are
  reproduced to machine precision, so the wall *operator* is algebraically
  sound.  Any future sign/factor slip flips this test red.
* the three ``xfail(strict=True)`` tests record the *open* defect: the wall flux
  is only first order for curved data, which leaves an O(1) residual in the
  strong form and makes the wall-bounded transient diverge.  ``strict=True``
  means they turn into failures as soon as the defect is fixed, forcing the
  limitation to be retired rather than silently kept.
"""
import math

import pytest
import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_dg_2d import legendre_operators

MU = 0.1
KAPPA = MU * ns.CP / 0.71
DTYPE = torch.float64
DEVICE = torch.device("cpu")

XFAIL_REASON = (
    "open defect: the Dirichlet wall flux uses the one-sided cell gradient, so it "
    "is only O(h) for curved data (see docs/reports/tensordg-ns-wall-verification.md)"
)


def _operators(degree, ncells):
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


def _couette_state(ncells=(2, 8), degree=1, wall_speed=1.0):
    """Exact steady Couette state: u = U y/H, Brinkman T parabola, p = 1."""
    op = _operators(degree, ncells)
    amplitude = MU * wall_speed**2 / (2.0 * KAPPA)

    def initial(x, y):
        u = wall_speed * y
        zeros = torch.zeros_like(u)
        temperature = 1.0 + amplitude * y * (1.0 - y)
        return 1.0 / temperature, u, zeros, torch.ones_like(u)

    coeff = ns._initial_coefficients(initial, op["x"], op["y"], op["dx"], op["dy"],
                                     op["nodes"], op["weights"], op["basis"],
                                     DEVICE, DTYPE)
    boundary = {
        "xmin": None, "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
        "ymax": {"velocity": (wall_speed, 0.0), "temperature": 1.0},
    }
    return op, coeff, boundary


def _wall_fluxes(ncells=(2, 8), degree=1, penalty_scale=2.0):
    op, coeff, boundary = _couette_state(ncells, degree)
    penalty = penalty_scale * (degree + 1.0) ** 2
    h = op["dy"]
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "right")
    top = ns._boundary_flux(right, boundary["ymax"], 1, h, MU, KAPPA, penalty,
                            2.0 / h, 2.0 / op["dx"])
    bottom = ns._boundary_flux(left, boundary["ymin"], 1, h, MU, KAPPA, penalty,
                               2.0 / h, 2.0 / op["dx"])
    return {
        "mx_top": float(top[0, -1, 0, 1]), "mx_bottom": float(bottom[0, 0, 0, 1]),
        "energy_top": float(top[0, -1, 0, 3]),
        "energy_bottom": float(bottom[0, 0, 0, 3]),
    }


# ---------------------------------------------------------------------------
# Green: the wall operator is exact for data the DG space resolves
# ---------------------------------------------------------------------------
def test_wall_flux_is_exact_for_polynomial_data():
    """rho = 1, u = v = 0, p linear in y -> T = p is linear and exactly resolved.

    The wall flux must then equal the analytic traction and Fourier flux:
    ``mx = 0``, ``my = p``, ``E = -kappa dT/dy``.
    """
    slope = 0.4
    degree, ncells = 1, (2, 4)
    op = _operators(degree, ncells)

    def initial(x, y):
        zeros = torch.zeros_like(x + y)
        return torch.ones_like(x + y), zeros, zeros, 1.0 + slope * y

    coeff = ns._initial_coefficients(initial, op["x"], op["y"], op["dx"], op["dy"],
                                     op["nodes"], op["weights"], op["basis"],
                                     DEVICE, DTYPE)
    boundary = {
        "xmin": None, "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
        "ymax": {"velocity": (0.0, 0.0), "temperature": 1.0 + slope},
    }
    penalty = 2.0 * (degree + 1.0) ** 2
    h = op["dy"]
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "right")
    top = ns._boundary_flux(right, boundary["ymax"], 1, h, MU, KAPPA, penalty,
                            2.0 / h, 2.0 / op["dx"])
    bottom = ns._boundary_flux(left, boundary["ymin"], 1, h, MU, KAPPA, penalty,
                               2.0 / h, 2.0 / op["dx"])

    conduction = -KAPPA * slope
    assert top[0, -1, 0, 1].abs() < 1e-12
    assert bottom[0, 0, 0, 1].abs() < 1e-12
    assert abs(float(top[0, -1, 0, 2]) - (1.0 + slope)) < 1e-12
    assert abs(float(bottom[0, 0, 0, 2]) - 1.0) < 1e-12
    assert abs(float(top[0, -1, 0, 3]) - conduction) < 1e-12
    assert abs(float(bottom[0, 0, 0, 3]) - conduction) < 1e-12


# ---------------------------------------------------------------------------
# Open defect (strict xfail): first-order boundary flux in a strong-form scheme
# ---------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason=XFAIL_REASON)
def test_exact_steady_couette_state_has_no_wall_residual():
    """A consistent strong-form wall operator leaves no residual at the exact
    steady state, at any mesh.  Currently the residual is O(1) and lives
    entirely in the wall rows (0.30 at both 8 and 32 cells)."""
    residuals = {}
    for ny in (8, 32):
        op, coeff, boundary = _couette_state((2, ny), 1)
        penalty = 2.0 * 2.0**2
        rhs = ns._rhs_ns(coeff, op["basis"], op["derivative"], op["weights"],
                         op["signs"], op["dx"], op["dy"], (True, False), boundary,
                         MU, KAPPA, penalty)
        residuals[ny] = float(rhs.abs().max())
    assert max(residuals.values()) < 1e-3, residuals


@pytest.mark.xfail(strict=True, reason=XFAIL_REASON)
def test_wall_energy_flux_converges_at_second_order():
    """The wall heat flux of the steady Couette state must converge to
    ``-kappa dT/dy`` at O(h) *after* the boundary gradient correction
    (m2496 report, section 6 item 2).  Measured slope today: 1.00."""
    errors = []
    for ny in (8, 16, 32):
        flux = _wall_fluxes((2, ny), 1)
        analytic = -KAPPA * (MU * 1.0**2 / (2.0 * KAPPA))
        errors.append(abs(flux["energy_bottom"] - analytic))
    # two dyadic refinements between 8 and 32 cells
    slope = math.log2(errors[0] / errors[-1]) / 2.0
    assert slope > 1.8, (slope, errors)


@pytest.mark.xfail(strict=True, reason=XFAIL_REASON)
def test_impulsively_started_couette_reaches_steady_state():
    """Impulsively started Couette must march to t = 0.1 without going
    non-physical.  Currently it raises ``FloatingPointError`` near t = 0.033,
    independently of the time step (the instability is spatial)."""
    ns.solve_couette_dg_2d(ncells=(2, 8), degree=1, mu=MU, final_time=0.1,
                           device=DEVICE, dtype=DTYPE)