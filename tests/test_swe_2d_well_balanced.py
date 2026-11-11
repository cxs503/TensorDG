"""Hydrostatic well-balancedness of the 2-D SWE solver.

A lake at rest (``η = h + z_b = const``, ``u = v = 0``) must be a *discrete
fixed point* of the scheme, at machine precision, on a curved bed and across a
bed jump.  These tests pin down the ``well_balanced=True`` path of
``tensordg.swe_2d.solve_swe_2d`` and, in ``test_curved_bed_rest_vs_default``,
quantify what it buys over the default cell-constant-slope path.

Reference for the reconstruction: Audusse et al. (2004), *SIAM J. Sci.
Comput.* 25(6), 2050-2065.  The 1-D counterpart of this suite lives in
``tests/test_swe_wellbalanced_hydrostatic.py``.
"""
from __future__ import annotations

import pytest
import torch

from tensordg.swe_2d import (
    DRY_DEPTH,
    check_hydrostatic_equilibrium_2d,
    reconstruct_fields_2d,
    solve_swe_2d,
)

G = 9.81
ETA0 = 2.0
DOMAIN = ((0.0, 1.0), (0.0, 1.0))

# Tolerances: the well-balanced path is exact to round-off (measured residuals
# are ~1e-13 for P1 and ~1e-11 for P2 in double precision); the thresholds below
# leave two orders of margin while still being far below any physical signal.
TOL_ETA = 1e-8
TOL_VEL = 1e-8


def _zeros(x, y):
    return torch.zeros_like(x)


def _rest_run(zb, h, nx=10, ny=4, degree=1, final_time=0.05, **kwargs):
    """Still lake on bed ``zb`` with depth ``h = ETA0 - zb`` (WB by default)."""
    kwargs.setdefault("well_balanced", True)
    return solve_swe_2d(
        h, _zeros, _zeros, zb,
        nx=nx, ny=ny, degree=degree, final_time=final_time,
        domain=DOMAIN, cfl=0.3, **kwargs,
    )


class TestWellBalancedRest:

    @pytest.mark.parametrize("degree", [1, 2])
    def test_flat_bed_rest(self, degree):
        x, y, coeff, zb_faces, meta = _rest_run(
            lambda xx, yy: torch.zeros_like(xx),
            lambda xx, yy: torch.full_like(xx, ETA0),
            degree=degree,
        )
        check = check_hydrostatic_equilibrium_2d(
            coeff, meta["zb_modes"], eta_expected=ETA0, tolerance=TOL_VEL)
        assert check["is_hydrostatic"], check["message"]
        assert check["u_max"] < TOL_VEL and check["v_max"] < TOL_VEL
        assert check["eta_error_max"] < TOL_ETA
        assert meta["well_balanced"] is True

    @pytest.mark.parametrize("degree", [1, 2])
    def test_curved_bed_rest(self, degree):
        """Quadratic bed: the default cell-constant slope is *not* exact here."""
        def zb(xx, yy):
            return 0.2 * xx + 0.05 * xx ** 2

        def h(xx, yy):
            return ETA0 - zb(xx, yy)

        x, y, coeff, zb_faces, meta = _rest_run(zb, h, nx=12, ny=4, degree=degree)
        check = check_hydrostatic_equilibrium_2d(
            coeff, meta["zb_modes"], eta_expected=ETA0, tolerance=TOL_VEL)
        assert check["is_hydrostatic"], check["message"]
        assert check["u_max"] < TOL_VEL
        assert check["eta_error_max"] < TOL_ETA

    def test_curved_bed_rest_vs_default(self):
        """Baseline contrast: the same state drifts on the default path.

        The cell-constant bed slope of the default path is only the *average*
        of dz_b/dx, so the O(dx * z_b'') momentum residual that it leaves
        accumulates into a spurious current.  This is the defect the
        well-balanced path removes; the assertion is one-sided (the default
        path must be clearly worse) so that it documents the gap rather than
        freezing a particular number.
        """
        def zb(xx, yy):
            return 0.2 * xx + 0.05 * xx ** 2

        def h(xx, yy):
            return ETA0 - zb(xx, yy)

        _, _, c_wb, _, meta_wb = _rest_run(zb, h, nx=12, ny=4, degree=2)
        _, _, c_def, _, meta_def = _rest_run(zb, h, nx=12, ny=4, degree=2,
                                             well_balanced=False)
        wb = check_hydrostatic_equilibrium_2d(
            c_wb, meta_wb["zb_modes"], eta_expected=ETA0)
        default = check_hydrostatic_equilibrium_2d(
            c_def, meta_def["zb_modes"], eta_expected=ETA0)
        assert meta_def["bed_representation"] == "cell_constant_slope"
        assert default["u_max"] > 1e-6, default["message"]
        assert wb["u_max"] < 1e-2 * default["u_max"]

    @pytest.mark.parametrize("degree", [1, 2])
    def test_bed_jump_rest_with_dry_bank(self, degree):
        """A bed jump at a cell face + a dry bank is a discrete fixed point.

        Cell-local nodal data is required to express the jump: a single
        callable is evaluated at the shared face coordinate and would smear it
        over the two neighbouring cells.
        """
        nx, ny, nmodes = 12, 4, degree + 1
        zb = torch.zeros(1, 1, nmodes, nmodes).repeat(nx, ny, 1, 1)
        zb[6:] = 2.5                      # dry bank, shoreline on the face x = 0.5
        h = (ETA0 - zb).clamp_min(0.0)
        u = torch.zeros_like(zb)
        v = torch.zeros_like(zb)

        x, y, coeff, zb_faces, meta = solve_swe_2d(
            h, u, v, zb, nx=nx, ny=ny, degree=degree, final_time=0.05,
            domain=DOMAIN, cfl=0.3, well_balanced=True)

        check = check_hydrostatic_equilibrium_2d(
            coeff, meta["zb_modes"], eta_expected=ETA0, tolerance=TOL_VEL)
        assert check["is_hydrostatic"], check["message"]
        assert h[:6].min() > 1.0          # the wet side is still wet
        assert h[6:].max() == 0.0         # and the dry side still dry
        assert float(coeff[6:, :, :, :, 0].abs().max()) < TOL_ETA

    @pytest.mark.parametrize("degree", [1, 2])
    def test_periodic_bed_rest(self, degree):
        """Periodic closure must not inject momentum at the seam."""
        def zb(xx, yy):
            return 0.1 * torch.sin(2.0 * torch.pi * xx)

        def h(xx, yy):
            return ETA0 - zb(xx, yy)

        x, y, coeff, zb_faces, meta = _rest_run(
            zb, h, nx=12, ny=4, degree=degree, periodic=(True, False))
        check = check_hydrostatic_equilibrium_2d(
            coeff, meta["zb_modes"], eta_expected=ETA0, tolerance=TOL_VEL)
        assert check["is_hydrostatic"], check["message"]
        assert check["u_max"] < TOL_VEL


class TestWellBalancedUnsteady:

    def test_dam_break_over_hump_stays_physical(self):
        """Unsteady run over a curved bed: stable, positive depth, mass tight.

        Periodic in both directions so that mass is conserved *exactly* by the
        (0,0)-mode update and any drift would be a real loss of conservation,
        not a boundary flux.
        """
        def zb(xx, yy):
            return 0.15 * (1.0 - torch.cos(2.0 * torch.pi * xx)) * \
                (1.0 - torch.cos(2.0 * torch.pi * yy))

        # Cell-local nodal depth: the dam break must sit exactly on the face
        # x = 0.5, otherwise the sampled jump is smeared and the discrete
        # initial mass is not 0.7.
        nx, ny = 12, 6
        h = torch.full((nx, ny, 2, 2), 0.4, dtype=torch.float64)
        h[: nx // 2] = 1.0
        zero = torch.zeros_like(h)

        x, y, coeff, zb_faces, meta = solve_swe_2d(
            h, zero, zero, zb, nx=nx, ny=ny, degree=1, final_time=0.05,
            domain=DOMAIN, cfl=0.3, periodic=(True, True), well_balanced=True)

        assert torch.isfinite(coeff).all()
        h_avg = coeff[:, :, 0, 0, 0]
        assert float(h_avg.min()) > 0.0
        dx, dy = 1.0 / nx, 1.0 / ny
        mass = float(h_avg.sum()) * dx * dy
        mass0 = 0.7  # (1.0 * 0.5 + 0.4 * 0.5) * 1.0, jump on the face x = 0.5
        assert abs(mass - mass0) < 1e-12

    def test_default_path_unchanged_by_new_branch(self):
        """well_balanced=False keeps the legacy representation and closure."""
        x, y, coeff, zb_faces, meta = solve_swe_2d(
            lambda xx, yy: torch.full_like(xx, 1.0), _zeros, _zeros,
            lambda xx, yy: 0.1 * xx, nx=8, ny=4, degree=1, final_time=0.01,
            domain=DOMAIN, cfl=0.3)
        assert meta["well_balanced"] is False
        assert meta["bed_representation"] == "cell_constant_slope"
        assert meta["zb_modes"] is None

    @pytest.mark.parametrize("degree", [1, 2])
    def test_nodal_tensor_requires_well_balanced(self, degree):
        nmodes = degree + 1
        zb = torch.zeros(1, 1, nmodes, nmodes).repeat(8, 4, 1, 1)
        h = torch.full_like(zb, 1.0)
        with pytest.raises(ValueError, match="well_balanced"):
            solve_swe_2d(h, torch.zeros_like(zb), torch.zeros_like(zb), zb,
                         nx=8, ny=4, degree=degree, final_time=0.01,
                         domain=DOMAIN, cfl=0.3, well_balanced=False)

    def test_nodal_tensor_shape_is_checked(self):
        with pytest.raises(ValueError, match="cell-local LGL nodal values"):
            solve_swe_2d(
                torch.full((8, 4, 3), ETA0), _zeros, _zeros,
                torch.zeros(8, 4, 3), nx=8, ny=4, degree=1, final_time=0.01,
                domain=DOMAIN, cfl=0.3, well_balanced=True)


class TestHelpers:
    def test_reconstruct_fields_2d_returns_free_surface(self):
        _, _, coeff, _, meta = _rest_run(
            lambda xx, yy: 0.1 * xx, lambda xx, yy: ETA0 - 0.1 * xx,
            nx=8, ny=4, degree=1)
        h, u, v, eta = reconstruct_fields_2d(coeff, meta["zb_modes"])
        assert h.shape == (8, 4)
        assert eta is not None and torch.allclose(
            eta, torch.full_like(eta, ETA0), atol=1e-10)
        assert float(u.abs().max()) < TOL_VEL

    def test_checker_flags_a_perturbed_surface(self):
        _, _, coeff, _, meta = _rest_run(
            lambda xx, yy: 0.1 * xx, lambda xx, yy: ETA0 - 0.1 * xx,
            nx=8, ny=4, degree=1)
        perturbed = coeff.clone()
        perturbed[0, 0, 0, 0, 0] += 0.05
        check = check_hydrostatic_equilibrium_2d(
            perturbed, meta["zb_modes"], eta_expected=ETA0, tolerance=1e-8)
        assert not check["is_hydrostatic"]
        assert check["eta_error_max"] > 1e-3

    def test_dry_cells_are_excluded_from_the_eta_test(self):
        """A dry cell's η is the bed, not the water level: it must be ignored."""
        nx, ny = 12, 4
        zb = torch.zeros(1, 1, 2, 2).repeat(nx, ny, 1, 1)
        zb[6:] = 2.5
        h = (ETA0 - zb).clamp_min(0.0)
        _, _, coeff, _, meta = solve_swe_2d(
            h, torch.zeros_like(zb), torch.zeros_like(zb), zb,
            nx=nx, ny=ny, degree=1, final_time=0.05, domain=DOMAIN,
            cfl=0.3, well_balanced=True)
        check = check_hydrostatic_equilibrium_2d(
            coeff, meta["zb_modes"], eta_expected=ETA0, tolerance=1e-8)
        assert check["is_hydrostatic"], check["message"]
        assert DRY_DEPTH > 0.0