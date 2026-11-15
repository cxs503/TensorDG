"""Regression tests for the no-MCMP two-phase (phase-field) LBM.

Each test binds one *claim* of ``docs/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY.md`` to a
number.  The tolerances used here are the ones pre-registered in
``benchmarks/phasefield_laplace_xval.py`` (5 % for the free-energy leg, 15 % for
the Laplace leg, 3 % target), not values chosen after looking at the output.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tensorlbm.phasefield as pf  # noqa: E402


SIGMA = 0.05
ELL = 4.0
MOBILITY = 0.10


def make_solver(n=64, ell=ELL, sigma=SIGMA, mobility=MOBILITY, **kw):
    beta = 3.0 * sigma / (8.0 * ell)
    kappa = 2.0 * beta * ell ** 2
    return pf.PhaseFieldTwoPhase(nx=n, ny=n, beta=beta, kappa=kappa,
                                 mobility=mobility, **kw)


# --------------------------------------------------------------- operators
def test_isotropic_grad_recovers_unit_slope():
    x = torch.arange(8, dtype=torch.float64).view(1, 8).repeat(8, 1)
    gx, gy = pf.isotropic_grad(x)
    assert float(gx[2:-2, 2:-2].mean()) == pytest.approx(1.0, abs=1e-12)
    assert float(gy[2:-2, 2:-2].abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_isotropic_laplacian_recovers_quadratic():
    x = torch.arange(8, dtype=torch.float64).view(1, 8).repeat(8, 1)
    lap = pf.isotropic_laplacian(x ** 2)
    assert float(lap[2:-2, 2:-2].mean()) == pytest.approx(2.0, abs=1e-12)


# ----------------------------------------------------------- model algebra
def test_analytic_surface_tension_closed_form():
    solver = make_solver()
    beta, kappa = solver.beta, solver.kappa
    assert solver.interface_width == pytest.approx(math.sqrt(kappa / (2 * beta)))
    # sigma = (8/3) beta ell = (4 sqrt2 / 3) sqrt(beta kappa)
    assert solver.sigma_analytic == pytest.approx((8.0 / 3.0) * beta * solver.interface_width)
    assert solver.sigma_analytic == pytest.approx(
        (4.0 * math.sqrt(2.0) / 3.0) * math.sqrt(beta * kappa))
    # the factory in the benchmark must hit both targets
    assert solver.interface_width == pytest.approx(ELL)
    assert solver.sigma_analytic == pytest.approx(SIGMA)


def test_chemical_potential_decays_to_discrete_equilibrium():
    """mu = psi' - kappa lap(phi) must decay towards 0 (the stationarity
    condition of the free energy).  It does not reach machine zero because the
    discrete 9-point Laplacian does not reproduce the continuum tanh exactly --
    that residual is the origin of the ~1 % sigma_profile error, so it is
    asserted at 1 % of the dominating term instead of an invented tiny value."""
    solver = make_solver(n=96)
    solver.setup_planar_interface()
    x0 = solver.nx // 2
    band = torch.zeros(solver.nx, dtype=torch.bool)
    band[max(0, x0 - 20):x0 + 20] = True
    mu0 = float(solver.chemical_potential()[:, band].abs().max())
    solver._lbm_step = lambda fx, fy: None
    solver.run(1000)
    mu = solver.chemical_potential()
    scale = float((4 * solver.beta * (solver.phi ** 3 - solver.phi))[:, band].abs().max())
    mu1 = float(mu[:, band].abs().max())
    assert mu1 < mu0, "chemical potential must decay towards equilibrium"
    assert mu1 < 1e-2 * scale, f"mu={mu1:.3e} vs term scale {scale:.3e}"


def test_stability_guard_rejects_unstable_mobility():
    solver = make_solver(mobility=0.5)          # M*kappa = 0.075 > 2/64
    assert solver.stability_margin() > 1.0
    solver.setup_planar_interface()
    with pytest.raises(ValueError, match="unstable"):
        solver.run(10)


def test_bad_configuration_rejected():
    with pytest.raises(ValueError):
        make_solver(force_form="not_a_form")
    with pytest.raises(ValueError):
        make_solver(advection="not_a_scheme")


# ------------------------------------------------------- forcing correctness
def test_guo_forcing_gives_u_equals_F_times_t():
    """Uniform body force on a periodic single-phase box: u = F*t exactly.
    This pins the Guo source term (a sign/factor slip here is what makes a
    static droplet drift instead of staying put)."""
    n = 16
    solver = pf.PhaseFieldTwoPhase(nx=n, ny=n, beta=1e-9, kappa=1e-9, mobility=0.0)
    solver.set_phi(-torch.ones(n, n, dtype=torch.float64))
    F = torch.full((n, n), 1e-4, dtype=torch.float64)
    zero = torch.zeros_like(F)
    for _ in range(100):
        solver._lbm_step(F, zero)
    assert float(solver.ux.mean()) == pytest.approx(1e-4 * 100, rel=1e-6)
    assert float(solver.uy.abs().max()) < 1e-15


# ------------------------------------------------------------- the physics
def test_planar_interface_surface_tension_matches_analytic():
    """Leg A: sigma measured from the field == sigma predicted by (beta, kappa)."""
    solver = make_solver(n=64)
    solver.setup_planar_interface()
    solver._lbm_step = lambda fx, fy: None       # free-energy leg only
    solver.run(1000)
    sigma_profile = solver.sigma_from_profile("x")
    rel = abs(sigma_profile - solver.sigma_analytic) / solver.sigma_analytic
    assert rel <= 0.05, f"sigma_profile={sigma_profile} vs {solver.sigma_analytic} ({rel:.3%})"


def test_droplet_laplace_law_and_interface_survival():
    """Leg B at CI scale, plus the regression that actually bit during
    development: a dissipative (upwind) order-parameter advection dissolves a
    4-cell interface within ~1e3 steps, so phi must stay saturated."""
    solver = make_solver(n=64)
    solver.setup_droplet(14.0)
    solver.run(1000)
    assert float(solver.phi.max()) > 0.9, "interface dissolved (phi_max collapsed)"
    assert float(solver.phi.min()) < -0.9, "interface dissolved (phi_min collapsed)"
    assert solver.spurious_current() < 1e-2
    # time-average over 3 x 100 steps: the initial condition leaves an acoustic
    # ring-down that would otherwise be read as Laplace-law error
    samples = []
    for _ in range(3):
        solver.step(100)
        samples.append(solver.pressure_drop(14.0)["dp"])
    dp = sum(samples) / len(samples)
    sigma_over_R = solver.sigma_analytic / 14.0
    assert dp > 0.0
    rel = abs(dp - sigma_over_R) / sigma_over_R
    assert rel <= 0.15, f"dp={m['dp']:.4e} vs sigma/R={sigma_over_R:.4e} ({rel:.2%})"


def test_phi_is_conserved_by_central_advection():
    """The central div(phi u) has zero net flux on a periodic box, so the
    Allen-Cahn mass must not drift (upwinding would leak at the interface)."""
    solver = make_solver(n=48)
    solver.setup_droplet(10.0)
    m0 = solver.total_phi()
    solver.run(300)
    assert abs(solver.total_phi() - m0) / abs(m0) < 1e-10


def test_pressure_drop_rejects_impossible_geometry():
    solver = make_solver(n=48)
    solver.setup_droplet(4.0)                     # R < 2*band -> empty sample
    with pytest.raises(ValueError, match="empty sampling region"):
        solver.pressure_drop(4.0)


# ------------------------------------------------------- "no MCMP" property
def test_no_mcmp_dependency_single_component_state():
    """The anchor is about the *no-MCMP* path: one distribution function, no
    per-component density, no pseudopotential cross-force."""
    import inspect
    src = inspect.getsource(pf)
    imports = [ln for ln in src.splitlines()
               if ln.strip().startswith(("import ", "from "))]
    assert not any("mcmp" in ln or "cross_force" in ln for ln in imports), imports
    assert not any("mcmp" in getattr(v, "__name__", "") for v in vars(pf).values())
    solver = make_solver(n=32)
    assert solver.f.shape[0] == 9                       # one D2Q9 population set
    assert not hasattr(solver, "n_components")
    assert not hasattr(solver, "G_matrix")
    # and the second thermodynamic field is an order parameter, not a density
    solver.setup_droplet(8.0)
    solver.run(50)
    assert float(solver.phi.min()) < -0.9 and float(solver.phi.max()) > 0.9
    # matched-density model: one density field, uniform to O(dp) = O(sigma/R),
    # unlike MCMP where each component carries its own rho field
    assert float(solver.rho.min()) == pytest.approx(float(solver.rho.max()), rel=5e-2)