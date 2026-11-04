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


def _wall_fluxes(ncells=(2, 8), degree=1, penalty_scale=2.0, wall_gradient="face"):
    op, coeff, boundary = _couette_state(ncells, degree)
    penalty = penalty_scale * (degree + 1.0) ** 2
    h = op["dy"]
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "right")
    recon = (ns._wall_normal_derivative(coeff, op["basis"], op["signs"], h, 1)
             if wall_gradient == "taylor" else None)
    top = ns._boundary_flux(right, boundary["ymax"], 1, h, MU, KAPPA, penalty,
                            2.0 / h, 2.0 / op["dx"],
                            None if recon is None else recon["right"])
    bottom = ns._boundary_flux(left, boundary["ymin"], 1, h, MU, KAPPA, penalty,
                               2.0 / h, 2.0 / op["dx"],
                               None if recon is None else recon["left"])
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

# ---------------------------------------------------------------------------
# Second-order one-sided wall gradient (``wall_gradient="taylor"``)
# ---------------------------------------------------------------------------
def test_taylor_wall_gradient_is_exact_for_polynomial_data():
    """For data the DG space resolves, the reconstruction must not disturb the
    algebraically exact wall flux (the ``1.5 q_0 - 0.5 q_1`` stencil is exact
    for a linear field)."""
    slope, degree, ncells = 0.4, 1, (2, 4)
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
    penalty, h = 2.0 * (degree + 1.0) ** 2, op["dy"]
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "right")
    recon = ns._wall_normal_derivative(coeff, op["basis"], op["signs"], h, 1)
    pairs = {
        "bottom": (left, boundary["ymin"], recon["left"]),
        "top": (right, boundary["ymax"], recon["right"]),
    }
    for name, (side, wall, derivative) in pairs.items():
        legacy = ns._boundary_flux(side, wall, 1, h, MU, KAPPA, penalty,
                                   2.0 / h, 2.0 / op["dx"])
        improved = ns._boundary_flux(side, wall, 1, h, MU, KAPPA, penalty,
                                     2.0 / h, 2.0 / op["dx"], derivative)
        # conduction = -kappa * slope is the exact wall heat flux
        assert abs(float(legacy[0, -1 if name == "top" else 0, 0, 3])
                   - (-KAPPA * slope)) < 1e-12
        assert abs(float(improved[0, -1 if name == "top" else 0, 0, 3])
                   - (-KAPPA * slope)) < 1e-12
        row = -1 if name == "top" else 0
        assert torch.abs(improved[:, row, 0, :] - legacy[:, row, 0, :]).max() < 1e-12


def test_taylor_wall_gradient_reduces_the_wall_flux_error():
    """The one-sided Taylor gradient halves the *rest-wall* wall heat-flux error
    of the analytic Brinkman/Couette state (measured ratio 0.573 at 16 cells).

    Bottom-wall statement only (m3139): the derivative part of the P1 error is
    removed, the Dirichlet-penalty part is not, the observed order is unchanged
    (1.005 vs 1.001) and the moving wall and every degree ``p >= 2`` get worse.
    That is why the stencil is opt-in; see ``test_taylor_wall_gradient_*`` here
    and the design-order guards in ``test_ns_dg_2d_wall_scope.py``.
    """
    analytic = -KAPPA * (MU * 1.0**2 / (2.0 * KAPPA))
    errors = {}
    for mode in ("face", "taylor"):
        flux = _wall_fluxes((2, 16), 1, wall_gradient=mode)
        errors[mode] = abs(flux["energy_bottom"] - analytic)
    assert errors["taylor"] < 0.65 * errors["face"], errors


def test_default_wall_gradient_is_the_one_sided_derivative():
    """Re-anchored by m3139: the default is ``"face"``, not ``"taylor"`` (m2920).

    PR #36 made the extrapolated stencil the default on the strength of one
    measurement (P1, bottom wall).  The audit in
    ``validation/ns_dg_2d_wall/bisect_wall_gradient.json`` shows the stencil is
    ``O(h^2)`` at *every* degree, so as a default it caps the wall heat flux at
    second order: P3 observed 2.026 against design order 3 and an error up to
    97x the one-sided DG derivative, P2 1.88 against 2.42, and at P1 it degrades
    the *moving* wall (4.4e-3 vs 1.1e-3) even after halving the rest wall.  It
    therefore stays available as an explicit opt-in for the m2920 spectral
    study, which is degree-1 and keeps its recorded ``"taylor"`` results.
    """
    import inspect

    signature = inspect.signature(ns.solve_navier_stokes_dg_2d)
    assert signature.parameters["wall_gradient"].default == "face"
    with pytest.raises(ValueError):
        ns.solve_navier_stokes_dg_2d(
            lambda x, y: (torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
                          torch.ones_like(x)),
            ncells=(2, 4), degree=1, final_time=0.0, wall_gradient="bogus",
            device=DEVICE, dtype=DTYPE,
        )


# ---------------------------------------------------------------------------
# Spectral diagnosis: what actually destabilises the wall operator
# ---------------------------------------------------------------------------
def _wall_spectrum(variant="default"):
    from tools.diag_ns_wall import build_state
    from tools.spectrum_ns_wall import _spectrum

    state = build_state((2, 8), 1)
    return _spectrum(state["coeff"], state, state["boundary"], (True, False), 8.0,
                     "taylor", variant)


def test_periodic_diffusion_operator_is_marginally_stable():
    """Without walls the operator has no growing mode (max Re = 0): the
    instability is created by the Dirichlet walls, not by the interior scheme."""
    from tools.diag_ns_wall import build_state
    from tools.spectrum_ns_wall import _spectrum

    state = build_state((2, 8), 1)
    free = {"xmin": None, "xmax": None, "ymin": None, "ymax": None}
    result = _spectrum(state["coeff"], state, free, (True, True), 8.0, "taylor")
    assert result["max_real"] <= 1e-9, result


def test_adiabatic_wall_operator_is_stable():
    """Dropping the diffusive wall flux (free-slip/adiabatic) removes the
    growing mode: the *gradient* term, not the inviscid wall pressure, is what
    makes the Dirichlet wall unstable."""
    result = _wall_spectrum("inviscid")
    assert result["max_real"] <= 1e-9, result


@pytest.mark.xfail(strict=True, reason=(
    "open defect: the Dirichlet wall lacks the adjoint (Nitsche) boundary term, so "
    "the semi-discrete wall operator has a positive eigenvalue whose growth rate "
    "scales with the SIPG penalty; see docs/reports/tensordg-ns-wall-spectrum.md"))
def test_wall_operator_has_no_growing_eigenmode():
    result = _wall_spectrum()
    assert result["max_real"] <= 0.0, result


# ----------------------------------------------------------------------
# Preserved from mission m2838 (wall closure probe): the legacy one-sided
# closure (strict xfail, retired defect) and the *recovered* wall closure
# (``_recover_wall_gradient``) exposed via ``wall_gradient="recovered"``
# and the optional ``wall_penalty``.  Public solver defaults unchanged.
# ----------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Legacy closure (strict xfail): the retired first-order wall flux
# ---------------------------------------------------------------------------
# These three keep the historical defect reproducible: they run the default
# ``wall_gradient="one_sided"`` closure, where the mirror ghost
# ``grad|+ = grad|-`` reproduces the one-sided face gradient (O(h**p) for
# curved data).  Divided by ``h`` in the strong form it leaves an O(1) wall
# source (0.3045 at 8, 16 and 32 cells) that makes the wall-bounded transient
# diverge.  ``strict=True`` retires the limitation instead of letting it rot:
# they turn red the day the default is fixed as well.
@pytest.mark.xfail(strict=True, reason=XFAIL_REASON)
def test_legacy_closure_leaves_a_constant_wall_residual():
    """The one-sided (mirror-ghost) wall closure is inconsistent for curved data."""
    residuals = {}
    for ny in (8, 32):
        op, coeff, boundary = _couette_state((2, ny), 1)
        penalty = 2.0 * 2.0**2
        rhs = ns._rhs_ns(coeff, op["basis"], op["derivative"], op["weights"],
                         op["signs"], op["dx"], op["dy"], (True, False), boundary,
                         MU, KAPPA, penalty, wall_gradient="one_sided")
        residuals[ny] = float(rhs.abs().max())
    assert max(residuals.values()) < 1e-3, residuals


@pytest.mark.xfail(strict=True, reason=XFAIL_REASON)
def test_legacy_closure_wall_energy_flux_is_first_order():
    """The legacy wall heat flux converges at order 1, not at design order."""
    errors = []
    analytic = -KAPPA * (MU * 1.0**2 / (2.0 * KAPPA))
    for ny in (8, 16, 32):
        errors.append(abs(_wall_fluxes((2, ny), 1)["energy_bottom"] - analytic))
    slope = math.log2(errors[0] / errors[-1]) / 2.0
    assert slope > 1.8, (slope, errors)


@pytest.mark.xfail(strict=True, reason=XFAIL_REASON)
def test_legacy_closure_transient_diverges():
    """The impulsively started wall-bounded case never reaches ``t = 0.1``."""
    ns.solve_couette_dg_2d(ncells=(2, 8), degree=1, mu=MU, final_time=0.1,
                           device=DEVICE, dtype=DTYPE,
                           wall_gradient="one_sided")


# ---------------------------------------------------------------------------
# Recovered closure: consistent wall-normal gradient, no wall penalty
# ---------------------------------------------------------------------------
# ``wall_gradient="recovered"`` rebuilds the missing ghost side from the
# element-centre gradient and the (already averaged) gradient on the opposite
# interior face of the same cell,
#
#     grad_n|_wall = 2 grad_n(centre) - grad_n(opposite face),
#
# which is one order more accurate than the one-sided value.  With the SIPG
# wall penalty switched off (``wall_penalty=0``) nothing re-injects an O(h)
# bias, and both the wall heat flux and the wall-bounded transient recover.
RECOVERED = dict(wall_gradient="recovered", wall_penalty=0.0)
ANALYTIC_ENERGY_FLUX = -KAPPA * (MU * 1.0**2 / (2.0 * KAPPA))


def _recovered_wall_energy_flux(ny, degree=1):
    """Wall heat flux on the bottom wall with the recovered closure.

    Mirrors :func:`_wall_fluxes` but builds the recovered gradient exactly the
    way ``_axis_fluxes_ns`` does (centre gradient plus the BR1-averaged
    gradient on the interior face of the wall cell).
    """
    op, coeff, boundary = _couette_state((2, ny), degree)
    penalty = 2.0 * (degree + 1.0) ** 2
    h, normal_scale = op["dy"], 2.0 / op["dy"]
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1,
                               "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1,
                                "right")
    centre = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1,
                                 "centre")
    outer = {key: torch.roll(value, shifts=-1, dims=1) for key, value in left.items()}
    names = {"u_n": "u", "v_n": "v", "T_n": "temperature"}
    averaged = {
        key: 0.5 * (right[key] + outer[key]) * normal_scale
        - (penalty / h) * (right[name] - outer[name])
        for key, name in names.items()
    }
    recovered = ns._recover_wall_gradient(centre, averaged, normal_scale, 1, "left")
    flux = ns._boundary_flux(left, boundary["ymin"], 1, h, MU, KAPPA, 0.0,
                             normal_scale, 2.0 / op["dx"], recovered=recovered)
    return float(flux[:, 0, :, 3].mean())


def test_recovered_closure_removes_the_constant_wall_residual():
    """The wall residual decays with h instead of being pinned at O(1)."""
    residuals = {}
    for ny in (8, 16, 32):
        op, coeff, boundary = _couette_state((2, ny), 1)
        penalty = 2.0 * 2.0**2
        rhs = ns._rhs_ns(coeff, op["basis"], op["derivative"], op["weights"],
                         op["signs"], op["dx"], op["dy"], (True, False), boundary,
                         MU, KAPPA, penalty, **RECOVERED)
        residuals[ny] = float(rhs.abs().max())
    # legacy closure: 0.3045 at every mesh; recovered: 0.0247 -> 0.0062
    assert residuals[32] < 0.05, residuals
    assert residuals[8] / residuals[32] > 2.0, residuals
    # the wall rows are no longer the dominant error source
    assert residuals[32] < 0.1 * 0.3045, residuals


def test_recovered_closure_wall_energy_flux_is_second_order():
    """The wall heat flux must converge at (near) design order: slope > 1.8."""
    errors = [abs(_recovered_wall_energy_flux(ny) - ANALYTIC_ENERGY_FLUX)
              for ny in (8, 16, 32)]
    slope = math.log2(errors[0] / errors[-1]) / 2.0
    assert slope > 1.8, (slope, errors)
    assert errors[-1] < 1e-5, errors


def test_recovered_closure_impulsively_started_couette_reaches_steady_state():
    """The wall-bounded transient now marches to t = 0.1 (legacy: t ~ 0.032)."""
    _, _, coeff, meta = ns.solve_couette_dg_2d(
        ncells=(2, 8), degree=1, mu=MU, final_time=0.1, device=DEVICE, dtype=DTYPE,
        **RECOVERED
    )
    assert meta["steps"] > 0
    assert torch.isfinite(coeff).all()


def test_recovered_closure_keeps_the_wall_operator_consistent():
    """On data the DG space resolves the recovered closure still returns the
    exact traction and Fourier flux (no regression of the algebraic identity)."""
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
    h = op["dy"]
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1,
                                "right")
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1,
                               "left")
    centre = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1,
                                 "centre")
    outer = {key: torch.roll(value, shifts=-1, dims=1) for key, value in left.items()}
    averaged = {key: 0.5 * (right[key] + outer[key]) * (2.0 / h)
                for key in ("u_n", "v_n", "T_n")}
    recovered = ns._recover_wall_gradient(centre, averaged, 2.0 / h, 1, "left")
    flux = ns._boundary_flux(left, boundary["ymin"], 1, h, MU, KAPPA, 0.0, 2.0 / h,
                             2.0 / op["dx"], recovered=recovered)
    assert abs(float(flux[0, 0, 0, 3]) + KAPPA * slope) < 1e-12
    assert flux[0, 0, 0, 1].abs() < 1e-12
