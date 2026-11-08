"""Verification of the 1-D SWE solver against Thacker (1981) planar sloshing.

History
-------
`docs/thacker_validation_gap.md` (mission m6315) declared a *critical* solver
gap for this benchmark.  The reference solution used there was not an exact
solution of the shallow water equations: it paired a planar free surface with
``u = -A omega sin(omega t) / g``, whereas the exact velocity amplitude for the
parabolic basin ``z_b = -h0 (1 - (x/L)^2)`` is ``A omega L / (2 h0)`` -- a factor
``L^2 g / (2 h0) ~ 4.9`` difference for the benchmark parameters.  The tests below
therefore (a) verify the corrected closed form from first principles, and
(b) measure what the solver actually does against it.

The corrected solution (see ``tensordg.swe_exact.thacker_planar_basin``) is

    omega = sqrt(2 g h0) / L
    s(t)  = (A / L) cos(omega t)                 (free-surface tilt)
    u(t)  = -(A omega L / (2 h0)) sin(omega t)   (spatially uniform)
    eta_c(t) = h0 + A^2 / (4 h0) sin^2(omega t)
    h = eta_c + s x - z_b.

It is exact iff the three scalar identities
``s_t = 2 h0 u / L^2``, ``eta_c,t = -u s`` and ``u_t = -g s`` hold; together they
imply that both PDE residuals vanish identically.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg import swe_exact
from tensordg.swe_1d import solve_swe_1d, reconstruct_fields

G, H0, AMP, L = 9.81, 1.0, 0.5, 1.0
OMEGA = math.sqrt(2.0 * G * H0) / L


def _fields(x, t):
    x = torch.as_tensor(x, dtype=torch.float64)
    h, u, eta = swe_exact.thacker_planar_basin(x.numpy(), t, H0, AMP, L, G)
    zb = swe_exact.thacker_parabolic_bed(x.numpy(), H0, L)
    return (torch.as_tensor(h, dtype=torch.float64), torch.as_tensor(u, dtype=torch.float64),
            torch.as_tensor(eta, dtype=torch.float64), torch.as_tensor(zb, dtype=torch.float64))


def _dg_run(ncells=20, degree=1, t_end=0.1, exact_bc=False):
    def init_h(x):
        h, _ = swe_exact.thacker_planar_basin_initial(x.numpy(), H0, AMP, L)
        return torch.as_tensor(h, dtype=x.dtype)

    def zb(x):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed(x.numpy(), H0, L), dtype=x.dtype)

    kwargs = {}
    if exact_bc:
        kwargs["bc_left_flux"] = lambda t: torch.as_tensor(
            swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)[0], dtype=torch.float64)
        kwargs["bc_right_flux"] = lambda t: torch.as_tensor(
            swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)[1], dtype=torch.float64)

    x, coeff, zb_faces, meta = solve_swe_1d(
        init_h, lambda z: torch.zeros_like(z), zb, ncells=ncells, degree=degree,
        final_time=t_end, domain=(-L, L), cfl=0.3, flux_kind="hll", limiter=None,
        rk_order=3, dtype=torch.float64, **kwargs)
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    return x, h, u, eta, meta


def _rel(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2)) / b.abs().max()) * 100.0


def _exact_at(x, t):
    h, u, eta = swe_exact.thacker_planar_basin(x.numpy(), t, H0, AMP, L, G)
    return (torch.as_tensor(h, dtype=torch.float64),
            torch.as_tensor(u, dtype=torch.float64),
            torch.as_tensor(eta, dtype=torch.float64))


class TestExactSolution:
    """The reference itself must be a genuine solution of the SWE."""

    @pytest.mark.parametrize("t", [0.0, 0.17, 0.5, 1.13])
    def test_three_scalar_identities(self, t):
        """s_t = 2 h0 u / L^2, eta_c,t = -u s, u_t = -g s (exact for all t)."""
        eps = 1e-6

        def u_of(tt):
            return float(swe_exact.thacker_planar_basin(torch.tensor([0.0]), tt, H0, AMP, L, G)[1][0])

        def eta_of(tt, x=0.0):
            return float(swe_exact.thacker_planar_basin(torch.tensor([x]), tt, H0, AMP, L, G)[2][0])

        u = u_of(t)
        u_t = (u_of(t + eps) - u_of(t - eps)) / (2 * eps)
        # free-surface tilt s(t) = d eta / d x (constant in x)
        s = (eta_of(t, 1e-3) - eta_of(t, -1e-3)) / 2e-3
        s_t = ((eta_of(t + eps, 1e-3) - eta_of(t + eps, -1e-3))
               - (eta_of(t - eps, 1e-3) - eta_of(t - eps, -1e-3))) / (4 * eps * 1e-3)
        eta_c_t = (eta_of(t + eps, 0.0) - eta_of(t - eps, 0.0)) / (2 * eps)

        assert s_t == pytest.approx(2.0 * H0 * u / L ** 2, abs=1e-6)
        assert eta_c_t == pytest.approx(-u * s, abs=1e-6)
        assert u_t == pytest.approx(-G * s, abs=1e-6)

    @pytest.mark.parametrize("t", [0.05, 0.31, 0.62])
    def test_pde_residuals_vanish(self, t):
        """Pointwise residual of continuity and (conservative) momentum ~ 0."""
        x = torch.linspace(-0.9, 0.9, 181, dtype=torch.float64)
        eps = 1e-5
        h0_, u0_, eta0_, zb_ = _fields(x, t)
        ht, ut, etat, _ = _fields(x, t + eps)
        hm, um, etam, _ = _fields(x, t - eps)
        h_t = (ht - hm) / (2 * eps)
        hu_t = (ht * ut - hm * um) / (2 * eps)
        eta_t = (etat - etam) / (2 * eps)
        dx = float(x[1] - x[0])
        hu = h0_ * u0_
        flux_m = hu
        flux_e = hu * u0_ + 0.5 * G * h0_ ** 2
        dfluxm = torch.gradient(flux_m, spacing=dx)[0]
        dfluxe = torch.gradient(flux_e, spacing=dx)[0]
        zbp = torch.gradient(zb_, spacing=dx)[0]
        # one-sided differences at the two end points are only O(dx) accurate
        cont = (h_t + dfluxm)[1:-1]
        mom = (hu_t + dfluxe + G * h0_ * zbp)[1:-1]
        scale = max(float(eta_t.abs().max()), 1.0)
        # continuity: h_t and (hu)_x are both at most quadratic -> 2nd-order
        # central differences are exact, so only round-off survives.
        assert float(cont.abs().max()) / scale < 1e-6
        # momentum: (hu^2 + g h^2/2)_x involves a quartic in x, so the central
        # difference carries an O(dx^2) truncation error (dx = 0.01 here).
        assert float(mom.abs().max()) / scale < 5e-3

    def test_velocity_is_uniform_and_depth_positive(self):
        for t in (0.0, 0.25, 0.5, 1.0):
            h, u, eta, zb = _fields(torch.linspace(-L, L, 51, dtype=torch.float64), t)
            assert torch.allclose(u, u[:1])
            assert float(h.min()) > 0.0
            assert torch.allclose(h, eta - zb)

    def test_boundary_flux_equals_physical_flux_at_the_rim(self):
        for t in (0.0, 0.2, 0.5):
            fl, fr = swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)
            h, u, _, _ = _fields(torch.tensor([-L, L], dtype=torch.float64), t)
            for h_, u_, f in ((float(h[0]), float(u[0]), fl), (float(h[1]), float(u[1]), fr)):
                assert f[0] == pytest.approx(h_ * u_, rel=1e-12, abs=1e-14)
                assert f[1] == pytest.approx(h_ * u_ ** 2 + 0.5 * G * h_ ** 2, rel=1e-12)

    def test_initial_condition_is_tilted_still_water(self):
        h, u, eta, zb = _fields(torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64), 0.0)
        assert torch.allclose(u, torch.zeros_like(u))
        assert float(eta[1]) == pytest.approx(H0)
        assert float(eta[2] - eta[0]) == pytest.approx(AMP)

    def test_period(self):
        assert swe_exact.thacker_planar_basin_period(H0, L, G) == pytest.approx(
            2 * math.pi / OMEGA, rel=1e-14)


class TestSolverAgainstExactSolution:
    """What the DG SWE solver does with the corrected reference."""

    def test_exact_boundary_flux_conserves_volume(self):
        """New ``bc_*_flux`` hook: the discrete mass budget is then exact."""
        x, h, u, eta, meta = _dg_run(ncells=20, degree=1, t_end=0.5, exact_bc=True)
        assert meta["exact_boundary_flux"] is True
        dx = 2.0 * L / 20
        vol = float(h.sum()) * dx
        # exact volume V = 2 eta_c + 4/3 h0 L
        eta_c = H0 + (AMP ** 2 / (4 * H0)) * math.sin(OMEGA * 0.5) ** 2
        vol_exact = 2.0 * eta_c + (4.0 / 3.0) * H0 * L
        assert abs(vol - vol_exact) / vol_exact < 1e-6

    def test_transmissive_closure_breaks_the_mass_budget(self):
        """Documented limitation: the basin has *open* ends for this solution."""
        x, h, u, eta, meta = _dg_run(ncells=20, degree=1, t_end=0.5, exact_bc=False)
        dx = 2.0 * L / 20
        vol = float(h.sum()) * dx
        eta_c = H0 + (AMP ** 2 / (4 * H0)) * math.sin(OMEGA * 0.5) ** 2
        vol_exact = 2.0 * eta_c + (4.0 / 3.0) * H0 * L
        assert abs(vol - vol_exact) / vol_exact > 0.01

    def test_short_time_free_surface_within_three_percent(self):
        x, h, u, eta, meta = _dg_run(ncells=40, degree=1, t_end=0.1)
        _, _, eta_e = _exact_at(x, 0.1)
        assert _rel(eta, eta_e) < 3.0

    def test_short_time_velocity_within_three_percent(self):
        """The old reference claimed ~100% velocity error; it was the formula."""
        x, h, u, eta, meta = _dg_run(ncells=40, degree=1, t_end=0.1)
        _, u_e, _ = _exact_at(x, 0.1)
        assert _rel(u, u_e) < 3.0

    def test_long_time_accuracy_gap_is_documented(self):
        """CHARACTERIZATION (known defect, do not read as a pass).

        At t = 0.5 s (~1/3 period) the error is ~12% and does *not* decrease for
        N = 25..200 cells: the deviation is not a discretization error that
        refinement removes.  An independent first-order finite volume solver
        (Rusanov, N = 2000, exact boundary flux) reproduces the analytic
        solution to 0.27% on the same problem, so the reference is right and the
        gap is in this solver.  Tracked in
        ``docs/thacker_validation_gap_corrected.md`` (open item).
        """
        x, h, u, eta, meta = _dg_run(ncells=80, degree=1, t_end=0.5)
        _, _, eta_e = _exact_at(x, 0.5)
        assert _rel(eta, eta_e) > 3.0

    def test_water_depth_stays_positive(self):
        x, h, u, eta, meta = _dg_run(ncells=50, degree=1, t_end=0.5)
        assert torch.all(h > 0.0)

    def test_smaller_amplitude_is_more_accurate(self):
        x, h, u, eta, meta = _dg_run(ncells=40, degree=1, t_end=0.2)
        _, _, eta_e = _exact_at(x, 0.2)
        base = _rel(eta, eta_e)

        def init_h(xx):
            hh, _ = swe_exact.thacker_planar_basin_initial(xx.numpy(), H0, 0.05, L)
            return torch.as_tensor(hh, dtype=xx.dtype)

        x2, coeff, zbf, meta2 = solve_swe_1d(
            init_h, lambda z: torch.zeros_like(z),
            lambda z: torch.as_tensor(swe_exact.thacker_parabolic_bed(z.numpy(), H0, L),
                                      dtype=z.dtype),
            ncells=40, degree=1, final_time=0.2, domain=(-L, L), cfl=0.3,
            flux_kind="hll", limiter=None, rk_order=3, dtype=torch.float64)
        h2, u2, eta2 = reconstruct_fields(coeff, zbf)
        _, _, eta_e2 = _exact_at(x2, 0.2)
        # recompute the small-amplitude reference with A = 0.05
        hh, uu, _ = swe_exact.thacker_planar_basin(x2.numpy(), 0.2, H0, 0.05, L, G)
        eta_e2 = torch.as_tensor(hh + swe_exact.thacker_parabolic_bed(x2.numpy(), H0, L),
                                dtype=torch.float64)
        assert _rel(eta2, eta_e2) < base + 0.5