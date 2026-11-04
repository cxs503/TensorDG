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
import numpy as np

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
                               None if recon is None else recon["left"],
                               side_name="left", outward_sign=-1.0)
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


# retired by the m2782 outward_sign fix (the taylor-default transient now marches to steady state)
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


def test_taylor_wall_gradient_does_not_reduce_the_oriented_wall_flux_error():
    """The one-sided Taylor gradient does **not** reduce the *rest-wall* wall
    heat-flux error once the Dirichlet penalty is oriented along the outward
    normal on both faces (``outward_sign``; m2782/m2876).

    History of the claim, kept for the record: measured against the pre-fix
    bottom-wall diagnostic (penalty sign slipped on the ``min`` face) the
    stencil *halved* the rest-wall error (ratio 0.573 at 16 cells), which is
    how the opt-in ``wall_gradient="taylor"`` mode was justified.  With the
    oriented helper the face-mode bottom error drops to 9.2e-4 and the Taylor
    stencil is *worse* (3.6e-3, ratio 3.9 at 16 cells): the earlier advantage
    was an artifact of the sign slip, not a property of the stencil.  The
    moving-wall behaviour and every degree ``p >= 2`` were already known to
    get worse; that is why the stencil stays opt-in.  Design-order guards
    live in ``test_ns_dg_2d_wall_scope.py``.
    """
    analytic = -KAPPA * (MU * 1.0**2 / (2.0 * KAPPA))
    errors = {}
    for mode in ("face", "taylor"):
        flux = _wall_fluxes((2, 16), 1, wall_gradient=mode)
        errors[mode] = abs(flux["energy_bottom"] - analytic)
    assert errors["taylor"] > errors["face"], errors


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


def test_wall_operator_has_no_growing_eigenmode():
    """Retired limitation (m2782 outward_sign fix): the anti-dissipative
    eigenvalue of docs/reports/tensordg-ns-wall-spectrum.md was the SIPG
    penalty jump entering with the wrong sign at the low-side wall.  With the
    jump oriented by the outward normal the wall operator has no growing
    eigenmode, so the strict xfail retires itself per the retirement
    contract.

    The zero-growth property is asserted up to the double-precision floor
    of the eigensolve rather than at ``<= 0`` exactly: the measured max
    real part is roundoff-level and sign-unstable across runs (8.2e-14
    static state on the merge box, -1.1e-13 couette, +1.6e-13 on the CI
    runner, workflow run 41783449770) while the spectrum magnitude is
    ~1.3e3, so the excursion is ~1e-16 *relative* -- machine epsilon.  A
    true growth mode is O(1) or larger (the pre-fix operator measured
    > 50)."""
    result = _wall_spectrum()
    assert result["max_real"] <= 1e-12, result


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


# retired by the m2782 outward_sign fix (the one_sided transient now survives; see wall_closure_probe.json)
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


# ----------------------------------------------------------------------
# Preserved from mission m2831 (fd2 wall stencil): the second-order
# one-sided wall closure ``wall_gradient='fd2'`` (opt-in; the public
# default stays 'taylor'), the retired-penalty companion constant
# ``WALL_PENALTY`` and the m2589 criteria it made green.  The default
# flip test below is kept verbatim and expected to fail under the
# m2920 'taylor' default (superseded semantics, owner to adjudicate).
# ----------------------------------------------------------------------

BRINKMAN_AMPLITUDE = MU / (2.0 * KAPPA)
ANALYTIC_MOMENTUM_FLUX = -MU


def _couette_series(y, time, speed=1.0, nu=MU, terms=600):
    """Closed-form impulsively started Couette flow (Fourier series)."""
    profile = speed * y
    for n in range(1, terms + 1):
        profile = profile + (2.0 * speed / math.pi) * (-1.0) ** n / n * torch.sin(
            n * math.pi * y
        ) * math.exp(-nu * n * n * math.pi**2 * time)
    return profile


def _fd2_wall_fluxes(ncells=(2, 8), degree=1, wall_gradient="fd2"):
    op, coeff, boundary = _couette_state(ncells, degree)
    penalty = 2.0 * (degree + 1.0) ** 2
    h = op["dy"]
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "right")
    centre = None
    wall_penalty = penalty
    if wall_gradient == "fd2":
        centre_u, centre_v, centre_T = ns._cell_centre_primitives(coeff)
        centre = {"u": centre_u, "v": centre_v, "T": centre_T}
        wall_penalty = ns.WALL_PENALTY
    top = ns._boundary_flux(right, boundary["ymax"], 1, h, MU, KAPPA, wall_penalty,
                            2.0 / h, 2.0 / op["dx"], centre=centre, side_name="right")
    bottom = ns._boundary_flux(left, boundary["ymin"], 1, h, MU, KAPPA, wall_penalty,
                               2.0 / h, 2.0 / op["dx"], centre=centre, side_name="left")
    return {
        "mx_top": float(top[0, -1, 0, 1]), "mx_bottom": float(bottom[0, 0, 0, 1]),
        "energy_top": float(top[0, -1, 0, 3]),
        "energy_bottom": float(bottom[0, 0, 0, 3]),
    }


# ---------------------------------------------------------------------------
# Green: the wall operator is exact for data the DG space resolves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wall_gradient", ["br1", "fd2"])
def test_fd2_wall_flux_is_exact_for_polynomial_data(wall_gradient):
    """rho = 1, u = v = 0, p linear in y -> T = p is linear and exactly resolved.

    The wall flux must then equal the analytic traction and Fourier flux:
    ``mx = 0``, ``my = p``, ``E = -kappa dT/dy`` -- for *both* wall gradients,
    so the new stencil is a strict replacement, not a different operator.
    """
    slope = 0.4
    degree, ncells = 1, (2, 4)
    op = _operators(degree, ncells)

    def initial(x, y):
        fill = torch.zeros_like(x + y)
        return torch.ones_like(x + y), fill.clone(), fill.clone(), 1.0 + slope * y + fill

    coeff = ns._initial_coefficients(initial, op["x"], op["y"], op["dx"], op["dy"],
                                     op["nodes"], op["weights"], op["basis"],
                                     DEVICE, DTYPE)
    boundary = {
        "xmin": None, "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
        "ymax": {"velocity": (0.0, 0.0), "temperature": 1.0 + slope},
    }
    penalty = 2.0 * (degree + 1.0) ** 2
    centre = None
    wall_penalty = penalty
    if wall_gradient == "fd2":
        centre_u, centre_v, centre_T = ns._cell_centre_primitives(coeff)
        centre = {"u": centre_u, "v": centre_v, "T": centre_T}
        wall_penalty = ns.WALL_PENALTY
    h = op["dy"]
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "right")
    top = ns._boundary_flux(right, boundary["ymax"], 1, h, MU, KAPPA, wall_penalty,
                            2.0 / h, 2.0 / op["dx"], centre=centre, side_name="right")
    bottom = ns._boundary_flux(left, boundary["ymin"], 1, h, MU, KAPPA, wall_penalty,
                               2.0 / h, 2.0 / op["dx"], centre=centre, side_name="left")

    conduction = -KAPPA * slope
    assert top[0, -1, 0, 1].abs() < 1e-12
    assert bottom[0, 0, 0, 1].abs() < 1e-12
    assert abs(float(top[0, -1, 0, 2]) - (1.0 + slope)) < 1e-12
    assert abs(float(bottom[0, 0, 0, 2]) - 1.0) < 1e-12
    assert abs(float(top[0, -1, 0, 3]) - conduction) < 1e-12
    assert abs(float(bottom[0, 0, 0, 3]) - conduction) < 1e-12


# ---------------------------------------------------------------------------
# Retired defect (m2589 xfail -> green): wall flux at the design order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("degree,meshes,minimum_order", [(1, (8, 16, 32), 0.9),
                                                         (2, (4, 8, 16), 1.8)])
def test_wall_energy_flux_converges_at_design_order(degree, meshes, minimum_order):
    """The wall heat flux must converge at the *design* order ``p``.

    m2589 measured 1.00 at P1 and 2.50 at P2 for the legacy interior-only
    gradient (the P1 wall flux was 1.27e-2 at 8 cells, i.e. 25 % of the analytic
    ``-0.05``).  With the one-sided stencil the errors are 1.24e-3 / 1.02e-6 and
    the order matches ``p`` (or better, since the stencil is exact for the
    quadratic Brinkman profile).
    """
    errors = [
        abs(_fd2_wall_fluxes((2, ny), degree)["energy_bottom"] - ANALYTIC_ENERGY_FLUX)
        for ny in meshes
    ]
    order = math.log2(errors[0] / errors[-1]) / math.log2(meshes[-1] / meshes[0])
    assert order > minimum_order, (order, errors)
    assert errors[-1] < 0.03 * abs(ANALYTIC_ENERGY_FLUX), errors


def test_wall_momentum_flux_below_three_percent_on_the_coarsest_mesh():
    """Even ``2 x 8`` P1 cells give a wall traction within 3 % of ``-mu U/H``."""
    flux = _fd2_wall_fluxes((2, 8), 1)
    assert abs(flux["mx_top"] - ANALYTIC_MOMENTUM_FLUX) < 0.03 * abs(ANALYTIC_MOMENTUM_FLUX)


# ---------------------------------------------------------------------------
# Retired defect (m2589 xfail -> green): wall-bounded transient
# ---------------------------------------------------------------------------
def test_impulsively_started_couette_matches_fourier_series():
    """Impulsively started Couette vs its closed-form Fourier series at t = 0.1.

    m2589 recorded a ``FloatingPointError`` ("nonphysical cell average") at
    ``t ~ 0.033`` for ``ncells=(2, 8), degree=1``, independent of the time step.
    The stencil fix makes the case run to ``t = 0.1``; the velocity profile is
    then compared with ``u = U y/H + (2U/pi) sum (-1)^n/n sin(n pi y/H)
    exp(-nu n^2 pi^2 t / H^2)``.

    At ``2 x 16`` the agreement is 0.84 % (``validation/ns_dg_2d_wall_gradient``);
    the coarser ``2 x 8`` mesh is 3.34 %, i.e. it misses the 3 % gate -- reported
    honestly in the m2831 report rather than hidden by a laxer tolerance.
    """
    _, y, coeff, meta = ns.solve_couette_dg_2d(
        ncells=(2, 16), degree=1, mu=MU, final_time=0.1, device=DEVICE, dtype=DTYPE,
        wall_gradient="fd2"
    )
    centre_u, _, centre_T = ns._cell_centre_primitives(coeff)
    density = coeff[..., 0, 0, 0]
    exact = _couette_series(y.view(1, -1), 0.1, nu=MU / float(density.mean()))
    relative = float(
        torch.linalg.norm(centre_u - exact) / torch.linalg.norm(exact)
    )
    assert meta["steps"] > 0
    assert relative < 0.03, relative
    assert float(centre_T.min()) > 0.5           # viscous heating stays bounded
    assert float(centre_u.abs().max()) <= 1.0 + 1e-6   # no wall-speed overshoot


# ---------------------------------------------------------------------------
# Backward compatibility: the legacy wall gradient is still selectable
# ---------------------------------------------------------------------------
def test_legacy_wall_gradient_reproduces_m2589_numbers():
    """``wall_gradient="br1"`` must keep the numbers of the m2589 report."""
    flux = _fd2_wall_fluxes((2, 8), 1, wall_gradient="br1")
    assert flux["energy_top"] == pytest.approx(-0.05220103, abs=1e-8)
    assert flux["energy_bottom"] == pytest.approx(-0.06270412, abs=1e-8)
    assert flux["mx_top"] == pytest.approx(-0.10033390, abs=1e-8)


def test_wall_gradient_argument_is_validated():
    with pytest.raises(ValueError):
        ns.solve_navier_stokes_dg_2d(lambda x, y: (x * 0 + 1, x * 0, x * 0, x * 0 + 1),
                                     wall_gradient="nope")


def test_wall_gradient_selection_is_echoed_and_fd2_is_opt_in():
    """m2831 re-anchored on the m3139 baseline: the public default is the
    one-sided DG derivative (``"face"``, the guard in the scope audit), the
    second-order fd2 wall stencil stays available opt-in, and the solver
    metadata echoes the path that actually ran."""
    _, _, _, meta = ns.solve_couette_dg_2d(ncells=(2, 4), degree=1, mu=MU,
                                           final_time=2e-3)
    assert meta["wall_gradient"] == ns.WALL_GRADIENT == "face"
    _, _, _, meta_fd2 = ns.solve_couette_dg_2d(ncells=(2, 4), degree=1, mu=MU,
                                               final_time=2e-3,
                                               wall_gradient="fd2")
    assert meta_fd2["wall_gradient"] == "fd2"

# ----------------------------------------------------------------------
# Preserved from mission m2782 (wall penalty orientation): the penalty
# jump at the low-side wall flips sign (``outward_sign``), retiring the
# anti-dissipative boundary term, plus the analytic shear-mode gate and
# ``final_time_reached`` metadata.  The three transient tests pin
# ``wall_gradient='one_sided'`` — the closure this mission measured
# (the union default remains 'taylor').
# ----------------------------------------------------------------------

def _project(op, initial, ncells):
    return ns._initial_coefficients(initial, op["x"], op["y"], op["dx"], op["dy"],
                                    op["nodes"], op["weights"], op["basis"],
                                    DEVICE, DTYPE)


def _cell_centres(coeff, degree):
    modes = torch.as_tensor(
        np.polynomial.legendre.legvander(np.array([0.0]), degree)[0], dtype=coeff.dtype
    )
    state = torch.einsum("xyklv,k,l->xyv", coeff, modes, modes)
    rho = state[..., 0]
    velocity_x = state[..., 1] / rho
    pressure = (ns.GAMMA - 1.0) * (state[..., 3]
                                   - 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / rho)
    return {"velocity_x": velocity_x, "temperature": pressure / rho}


def _signfix_wall_fluxes(ncells=(2, 8), degree=1, wall_temperature_bottom=1.0,
                 wall_temperature_top=1.0, penalty_scale=2.0):
    """Wall fluxes of ``rho = 1, u = v = 0, p = 1 + 0.5 y (1 - y)`` (T parabola)."""
    op = _operators(degree, ncells)
    amplitude = 0.5

    def initial(x, y):
        ones = torch.ones_like(x * y)
        zeros = torch.zeros_like(ones)
        return ones, zeros, zeros, 1.0 + amplitude * y * (1.0 - y)

    coeff = _project(op, initial, ncells)
    penalty = penalty_scale * (degree + 1.0) ** 2
    h, dx = op["dy"], op["dx"]
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "right")
    top = ns._boundary_flux(right, {"velocity": (0.0, 0.0),
                                    "temperature": wall_temperature_top},
                            1, h, MU, KAPPA, penalty, 2.0 / h, 2.0 / dx, outward_sign=1.0)
    bottom = ns._boundary_flux(left, {"velocity": (0.0, 0.0),
                                      "temperature": wall_temperature_bottom},
                               1, h, MU, KAPPA, penalty, 2.0 / h, 2.0 / dx,
                               outward_sign=-1.0)
    return op, top, bottom


def test_signfix_wall_flux_is_exact_for_polynomial_data():
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

    coeff = _project(op, initial, ncells)
    penalty = 2.0 * (degree + 1.0) ** 2
    h = op["dy"]
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "right")
    top = ns._boundary_flux(right, {"velocity": (0.0, 0.0), "temperature": 1.0 + slope},
                            1, h, MU, KAPPA, penalty, 2.0 / h, 2.0 / op["dx"],
                            outward_sign=1.0)
    bottom = ns._boundary_flux(left, {"velocity": (0.0, 0.0), "temperature": 1.0},
                               1, h, MU, KAPPA, penalty, 2.0 / h, 2.0 / op["dx"],
                               outward_sign=-1.0)

    conduction = -KAPPA * slope
    assert top[0, -1, 0, 1].abs() < 1e-12
    assert bottom[0, 0, 0, 1].abs() < 1e-12
    assert abs(float(top[0, -1, 0, 2]) - (1.0 + slope)) < 1e-12
    assert abs(float(bottom[0, 0, 0, 2]) - 1.0) < 1e-12
    assert abs(float(top[0, -1, 0, 3]) - conduction) < 1e-12
    assert abs(float(bottom[0, 0, 0, 3]) - conduction) < 1e-12


def test_wall_penalty_orientation_is_antisymmetric():
    """Root cause of the m2589 blow-up.

    A cell whose wall trace is hotter than the (colder) wall must lose energy at
    *both* walls: the numerical energy flux in the ``+y`` direction is negative
    at the bottom wall and positive at the top one.  The pre-fix code applied the
    jump ``q|-_face - q_wall`` with the same sign to both walls, so the bottom
    wall *heated* the cell -- an anti-dissipative boundary term whose growth rate
    scales with the penalty, exactly as the m2589 diagnosis measured.
    """
    _op, top, bottom = _signfix_wall_fluxes((2, 8), 1, wall_temperature_bottom=0.9,
                                    wall_temperature_top=0.9)
    flux_bottom = float(bottom[0, 0, 0, 3])
    flux_top = float(top[0, -1, 0, 3])
    assert flux_bottom < 0.0 < flux_top, (flux_bottom, flux_top)
    # the two walls must be symmetric for a mirror-symmetric state
    assert abs(flux_bottom + flux_top) < 1e-12


def test_wall_heat_flux_is_exact_for_representable_parabola():
    """Thermal wall path: at P2 the Brinkman parabola is resolved exactly.

    The interior trace equals the wall temperature, so the penalty contributes
    nothing and the Fourier flux must match ``-kappa dT/dy`` to machine
    precision.  This pins the constitutive law *and* the wall orientation.
    """
    _op, top, bottom = _signfix_wall_fluxes((2, 8), 2)
    # T = 1 + 0.5 y (1 - y): dT/dy = 0.5 (1 - 2y), i.e. +0.5 at the bottom wall
    # and -0.5 at the top one, so the Fourier flux -kappa dT/dy flips sign.
    assert abs(float(bottom[0, 0, 0, 3]) - (-KAPPA * 0.5)) < 1e-11
    assert abs(float(top[0, -1, 0, 3]) - (+KAPPA * 0.5)) < 1e-11


def test_wall_heat_flux_is_first_order_at_p1_for_curved_data():
    """Documented limitation (corrected from the m2589 criterion).

    The m2589 audit asked for second order.  With the BR1 mirror ghost the wall
    normal derivative is ``q_y|-_cell - (penalty/h)(q|- - q_wall)``: for curved
    data the trace error is ``O(h^2)``, so the flux is ``O(h)`` at P1 no matter
    how the penalty is oriented.  The *solution* is still second order (see the
    shear-mode test); only the pointwise wall flux is first order, which is the
    property to keep -- a jump to 1.8 would mean the ghost treatment changed.
    """
    errors = []
    for ny in (8, 16, 32):
        _op, _top, bottom = _signfix_wall_fluxes((2, ny), 1)
        analytic = -KAPPA * 0.5
        errors.append(abs(float(bottom[0, 0, 0, 3]) - analytic))
    slope = math.log2(errors[0] / errors[-1]) / 2.0
    assert 0.9 < slope < 1.1, (slope, errors)


def test_exact_steady_couette_state_has_no_residual_beyond_the_projection_floor():
    """Residual of the strong form at the analytic steady Couette state.

    m2589 measured 0.3045 at 8, 16 *and* 32 cells (no convergence) plus the wrong
    sign; after the fix it is ~4.5-5.1e-02 and no longer grows with the penalty.
    A ``< 1e-3`` gate is *not* a valid criterion here: at P1 the Brinkman
    parabola ``T = 1 + A s(1-s)`` is outside the DG space and ``rho = p/T`` is
    rational, so the projected state is not a discrete steady state; the residual
    is ``O(1)`` only because the discrete operator is stiff (its inverse scales
    as ``h^2``).  The verdict is therefore carried by the exact-solution tests
    above and below; this one guards against a regression to the pre-fix state.
    """
    amplitude = MU / (2.0 * KAPPA)  # wall speed U = 1
    residuals = {}
    for degree, ny in ((1, 8), (1, 32)):
        ncells = (2, ny)
        op = _operators(degree, ncells)

        def initial(x, y):
            zeros = torch.zeros_like(y)
            temperature = 1.0 + amplitude * y * (1.0 - y)
            return 1.0 / temperature, torch.ones_like(y) * y, zeros, torch.ones_like(y)

        coeff = _project(op, initial, ncells)
        boundary = {"xmin": None, "xmax": None,
                    "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
                    "ymax": {"velocity": (1.0, 0.0), "temperature": 1.0}}
        penalty = 2.0 * (degree + 1.0) ** 2
        rhs = ns._rhs_ns(coeff, op["basis"], op["derivative"], op["weights"],
                         op["signs"], op["dx"], op["dy"], (True, False), boundary,
                         MU, KAPPA, penalty, wall_gradient="one_sided")
        residuals[ny] = float(rhs.abs().max())
    assert max(residuals.values()) < 0.06, residuals


def test_impulsively_started_couette_marches_to_steady_state():
    """The configuration that used to raise ``FloatingPointError`` at t ~ 0.05.

    Before the fix the failure time was independent of ``dt`` and halved when the
    penalty doubled (m2589, ``tools/diag_ns_wall.py``).  The march must now reach
    ``t = 0.1`` with a physical velocity field.
    """
    x, y, coeff, meta = ns.solve_couette_dg_2d(ncells=(2, 8), degree=1, mu=MU,
                                               final_time=0.1, device=DEVICE, dtype=DTYPE,
                                               wall_gradient="one_sided")
    assert meta["final_time_reached"] >= 0.1 - 1e-12
    fields = _cell_centres(coeff, 1)
    assert torch.isfinite(coeff).all()
    assert float(fields["velocity_x"].abs().max()) <= 1.0 + 1e-9
    assert float(fields["temperature"].min()) > 0.0


def test_dirichlet_shear_mode_matches_exact_solution_below_three_percent():
    """Quantitative wall-path gate against an exact solution.

    ``u(y,0) = A sin(pi y)`` between two no-slip isothermal walls decays as
    ``u = A sin(pi y) exp(-nu pi^2 t)``; with ``A = 0.05`` the neglected viscous
    heating is ``O(A^2)`` (< 0.2 %).  Measured at 8 cells: 0.67 % (2.00 % with the
    old sign-blind penalty), second order in ``h``.
    """
    amplitude = 0.05
    ncells = (1, 8)

    def initial(x, y):
        ones = torch.ones_like(x * y)
        return ones, amplitude * torch.sin(math.pi * y), torch.zeros_like(ones), ones

    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
                "ymax": {"velocity": (0.0, 0.0), "temperature": 1.0}}
    x, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
        initial, ncells=ncells, degree=1, domain=((0.0, 1.0), (0.0, 1.0)),
        final_time=0.02, mu=MU, periodic=(True, False), boundary=boundary,
        device=DEVICE, dtype=DTYPE, wall_gradient="one_sided",
    )
    time = meta["final_time_reached"]
    exact = amplitude * torch.sin(math.pi * y) * math.exp(-MU * math.pi ** 2 * time)
    numerical = _cell_centres(coeff, 1)["velocity_x"][0, :]
    error = float(np.sqrt(torch.mean((numerical - exact) ** 2))
                  / np.sqrt(torch.mean(exact ** 2)))
    assert error < 0.03, (error, time)


# ---------------------------------------------------------------------------
# m2972: wall-bounded standing shear wave (smooth Dirichlet data) -- the
# accuracy judge for the wall path that the impulsive-start transient cannot
# provide.  Re-anchored note: the impulsive-start stability defect this
# mission also fixed was already retired on this chain by the m2782
# outward_sign correction; the symmetric (Nitsche adjoint) wall term below is
# the second, structural half of that fix and is additive to it.
# ---------------------------------------------------------------------------
def test_wall_bounded_standing_shear_wave_matches_exact_decay():
    """Wall-bounded standing viscous shear wave against its exact solution.

    Re-anchored: judged with the symmetric wall term explicitly on
    (``wall_adjoint=True``); on this chain the term is opt-in, so the default
    operator keeps the face + outward_sign semantics the earlier wall
    missions settled.

    ``u(y, 0) = A sin(k y)`` with ``k = 2 pi`` vanishes on both no-slip walls, so
    the Dirichlet boundary data is smooth (no impulsive start).  For small ``A``
    the compressible equations reduce to the linear diffusion problem, whose
    exact solution is ``u = A sin(k y) exp(-nu k**2 t)`` with ``nu = mu / rho``.
    Judge: relative L2 error of the cell-average velocity < 3%.
    """
    amplitude, mu, final_time = 0.01, 0.02, 0.05
    k = 2.0 * math.pi
    boundary = {
        "xmin": None, "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
        "ymax": {"velocity": (0.0, 0.0), "temperature": 1.0},
    }

    def initial(x, y):
        u = amplitude * torch.sin(k * y)
        return torch.ones_like(u), u, torch.zeros_like(u), torch.ones_like(u)

    x, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
        initial, ncells=(2, 16), degree=1, domain=((0.0, 1.0), (0.0, 1.0)),
        final_time=final_time, mu=mu, periodic=(True, False), boundary=boundary,
        wall_adjoint=True, device=DEVICE, dtype=DTYPE,
    )
    decay = math.exp(-(mu / 1.0) * k * k * final_time)
    numerical = coeff[:, :, 0, 0, 1] / coeff[:, :, 0, 0, 0]
    exact = amplitude * torch.sin(k * y).unsqueeze(0).expand_as(numerical) * decay
    relative = float(((numerical - exact) ** 2).mean().sqrt() / (amplitude * decay))
    assert relative < 0.03, (relative, meta["steps"])


# ---------------------------------------------------------------------------
# Wall-penalty orientation (m2876, re-anchored): the SIPG penalty must point
# along the outward normal on both faces.  In this chain the mechanism is the
# ``outward_sign`` keyword of ``_boundary_flux`` (m2782); this mission found
# the same root cause independently and contributes the finite-difference
# Jacobian guard and the both-walls heat-flux gate.
# ---------------------------------------------------------------------------
def test_wall_penalty_has_no_unstable_mode():
    """No eigenvalue of the semi-discrete wall-bounded operator may have a
    positive real part.

    Before the fix the (2, 4) P1 Jacobian had 13 eigenvalues with Re > 0 and
    max Re = +86.7; it now has none.  This is the cheapest possible guard for
    the orientation of the Dirichlet penalty: a sign slip on one face
    reappears here immediately.  Re-anchored on the m3139 ``face`` default
    (the wall-gradient and wall-adjoint options stay off).
    """
    import numpy as np
    op, coeff, boundary = _couette_state((2, 4), 1)
    penalty = 2.0 * 2.0**2
    args = (op["basis"], op["derivative"], op["weights"], op["signs"], op["dx"],
            op["dy"], (True, False), boundary, MU, KAPPA, penalty)
    flat = coeff.reshape(-1).clone()
    n = flat.numel()
    jac = np.zeros((n, n))
    eps = 1e-6
    base = ns._rhs_ns(coeff, *args).reshape(-1).numpy().astype(float)
    for i in range(n):
        probe = flat.clone()
        probe[i] += eps
        jac[:, i] = (
            ns._rhs_ns(probe.reshape(coeff.shape), *args).reshape(-1).numpy().astype(float)
            - base
        ) / eps
    real = np.linalg.eigvals(jac).real
    assert real.max() < 1e-8, float(real.max())


def test_wall_heat_flux_is_below_three_percent_on_both_walls():
    """The Dirichlet heat flux must meet the 3% documentation gate on *both*
    walls once the mesh resolves the Brinkman parabola.

    Before the orientation fix the bottom wall (the min face) carried an
    error of 3.17% even at 64 cells while the top wall was at 0.56% -- the
    asymmetry was the fingerprint of the flipped penalty orientation.  Both
    walls now agree.
    """
    exact = -KAPPA * (MU * 1.0**2 / (2.0 * KAPPA))
    for ny in (32, 64):
        flux = _wall_fluxes((2, ny), 1)
        bottom = abs(flux["energy_bottom"] - exact) / abs(exact)
        top = abs(flux["energy_top"] - exact) / abs(exact)
        assert bottom < 0.03, (ny, bottom)
        assert top < 0.03, (ny, top)
