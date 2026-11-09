"""Exact lake-at-rest on a *curved* bed: the ``well_balanced=True`` path.

Background
----------
``tests/test_swe_wellbalanced.py`` pins the bed-slope source factor and records
an **open defect** at P2: a still lake on the smooth bed
``z_b = 0.4 sin(pi x / 2)`` is not a discrete fixed point of the default
(L2-projected, piecewise-linear-bed) discretisation.  At ``N = 64``, ``t = 1 s``
the free surface drifts by 0.645 m = 32 % of the 2 m still-water depth.  The
documented reason was "no hydrostatic reconstruction at the interfaces".

The diagnosis behind this file is different, and it is *testable*: the rest
state fails because the discrete representation of ``(h, z_b)`` is not
compatible, not because the interface flux lacks a hydrostatic correction.
The default path L2-projects ``h`` on Gauss nodes but keeps the bed only as
per-cell face values (piecewise linear), so the discrete functions do not
satisfy ``h_h + z_h = const`` as polynomials; the interface traces of ``h`` of
the two adjacent cells then disagree by the projection error and the jump term
of the numerical flux injects a spurious momentum every step.

The fix implemented in ``solve_swe_1d(..., well_balanced=True)`` is to
represent the bed *and* the initial data by their degree-``p`` interpolants at
the same Gauss-Lobatto-Legendre nodes.  Interpolation is linear, so a sampled
``h + z_b = eta_0`` gives ``h_h + z_h = eta_0`` exactly (as polynomials), both
cell traces of ``h`` coincide at every interface, and the momentum source uses
the exact derivative of the polynomial bed.  The still lake is then a discrete
fixed point to round-off at P1 *and* P2.

There is a second, post-processing trap that these tests pin separately: for a
curved bed the trapezoidal mean ``(z_L + z_R)/2`` used by
:func:`reconstruct_fields` is *not* the cell average of the bed, it differs by
``O(dx^2)``.  Reading eta with it would report an ``O(dx^2)`` "drift" that the
solver does not have; ``metadata["zb_modes"][:, 0]`` is the consistent choice.

Everything here is 1-D.  The 2-D solver (``tensordg.swe_2d``) still uses the
L2-projected path and is expected to show the same class of drift; extending
the nodal representation to 2-D is *not* part of this change and is declared
as a limitation in ``docs/swe_well_balanced_exact.md``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg import swe_exact
from tensordg.swe_1d import reconstruct_fields, solve_swe_1d

ETA0 = 2.0
SLOPE = 0.4


def _bed(x):
    return SLOPE * torch.sin(0.5 * math.pi * x)


def _lake_at_rest(ncells=64, degree=1, final_time=1.0, well_balanced=True,
                  use_poly_bed=True):
    x, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: ETA0 - _bed(xx),
        lambda xx: torch.zeros_like(xx),
        _bed,
        ncells=ncells, degree=degree, final_time=final_time,
        domain=(-2.0, 2.0), cfl=0.3, limiter=None, rk_order=3,
        well_balanced=well_balanced)
    assert meta["reached_final_time"], meta["time_reached"]
    zb_center = meta["zb_modes"][:, 0] if (meta["well_balanced"] and use_poly_bed) else None
    h, u, eta = reconstruct_fields(coeff, zb_faces, zb_center=zb_center)
    return x, coeff, zb_faces, meta, h, u, eta


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("ncells", [32, 64])
def test_curved_bed_lake_at_rest_is_machine_precise(degree, ncells):
    """``h + z_b = eta_0`` must survive 1 s of RK3 at round-off level.

    Measured on this tree: P1 3.5e-13 m (N=32) / 1.3e-12 m (N=64),
    P2 2.7e-11 m (N=32) / 8.5e-11 m (N=64).  The gate is 1e-9 m = 5e-10 of the
    still-water depth, i.e. four orders of magnitude tighter than the 5e-4
    relative gate of the default P1 path and ten orders tighter than the 32 %
    P2 defect it replaces.
    """
    x, coeff, _, meta, h, u, eta = _lake_at_rest(ncells=ncells, degree=degree)
    drift = float((eta - ETA0).abs().max())
    momentum = float(coeff[:, :, 1].abs().max())
    assert drift < 1e-9, f"eta drift {drift:.3e} m"
    assert momentum < 1e-8, f"spurious momentum {momentum:.3e} m^2/s"
    assert float(u.abs().max()) < 1e-8
    assert meta["bed_representation"] == "nodal_lgl_polynomial"


@pytest.mark.parametrize("degree", [1, 2])
def test_trapezoidal_bed_average_is_the_only_O_dx2_residual(degree):
    """The ``O(dx^2)`` part of the P2 reading is a *post-processing* error.

    Reading the free surface with the piecewise-linear (trapezoidal) bed
    average leaves a residual that shrinks by ~4x per refinement, whereas the
    solver state read with the polynomial bed average is at round-off and does
    not care about the mesh.  This separates the two effects instead of
    attributing the whole number to the solver.
    """
    _, _, _, _, _, _, eta_poly = _lake_at_rest(ncells=32, degree=degree)
    _, _, _, _, _, _, eta_trap = _lake_at_rest(
        ncells=32, degree=degree, use_poly_bed=False)
    drift_poly = float((eta_poly - ETA0).abs().max())
    drift_trap = float((eta_trap - ETA0).abs().max())
    if degree == 1:
        # A linear bed has no trapezoidal error: nothing to separate.
        assert drift_trap < 1e-9
    else:
        assert drift_trap > 1e3 * drift_poly, (drift_trap, drift_poly)
        # 1.278e-3 m measured at N = 32 for the 0.4 sin bed; the trapezoid
        # error is (dx^2/12) z'' ~ 1.3e-3, i.e. it is fully explained by the
        # bed average, not by a leftover solver imbalance.
        assert abs(drift_trap - (0.4 * math.pi ** 2 / 4) * (0.125 ** 2) / 12) < 2e-4


def test_well_balanced_state_is_a_fixed_point_of_the_rhs():
    """Zero drift after one RK3 step is not enough: check the residual itself."""
    import tensordg.swe_1d as swe

    degree, ncells, dx = 2, 16, 4.0 / 16
    x = -2.0 + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    vinv, lgl = swe._lgl_transform(degree, torch.device("cpu"), torch.float64)
    nodes, weights, phi, dphi, phi_left, phi_right = swe._operators(
        degree, torch.device("cpu"), torch.float64)
    node_x_lgl = x[:, None] + 0.5 * dx * lgl[None, :]
    cz = _bed(node_x_lgl) @ vinv.T
    zb_slope_quad = torch.einsum("cm,qm->cq", cz, dphi) * (2.0 / dx)
    coeff = torch.zeros(ncells, degree + 1, 2, dtype=torch.float64)
    coeff[:, :, 0] = (ETA0 - _bed(node_x_lgl)) @ vinv.T
    faces = -2.0 + torch.arange(ncells + 1, dtype=torch.float64) * dx
    zb_faces = _bed(faces)
    rhs = swe._rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right,
                   zb_faces[1:] - zb_faces[:-1], "hll", False, 0.0, None, None,
                   zb_slope_quad)
    assert float(rhs[:, :, 1].abs().max()) < 1e-11, float(rhs[:, :, 1].abs().max())
    assert float(rhs[:, :, 0].abs().max()) < 1e-12


def test_default_path_still_shows_the_32_percent_p2_defect():
    """The opt-in path is a real capability delta, not a re-reading of the same run.

    Default (L2 projection) at N = 64, t = 1 s: 0.645 m = 32 % of the 2 m
    depth.  ``well_balanced=True`` on the same mesh: 8.5e-11 m.  The ratio is
    therefore > 1e9; asserting it keeps the comparison honest (if someone
    later makes the default well balanced, this test says so instead of
    silently keeping a stale claim).
    """
    _, _, _, _, _, _, eta_default = _lake_at_rest(
        ncells=64, degree=2, well_balanced=False)
    _, _, _, _, _, _, eta_wb = _lake_at_rest(ncells=64, degree=2)
    drift_default = float((eta_default - ETA0).abs().max())
    drift_wb = float((eta_wb - ETA0).abs().max())
    assert drift_default > 0.1, drift_default  # documented 0.645 m
    assert drift_default > 1e6 * drift_wb, (drift_default, drift_wb)


@pytest.mark.parametrize("degree", [1, 2])
def test_thacker_accuracy_is_not_degraded(degree):
    """A still lake is a degenerate test; a moving one must not get worse.

    Thacker (1981) planar basin, ``t = 0.1 s``, 20 cells.  Measured with the
    nodal path: P1 eta 0.062 % / u 0.523 %, P2 eta 0.116 % / u < 1e-8 % (the
    exact velocity is spatially uniform, so P2 represents it exactly).  The
    default path gives P2 eta 1.16 % / u 9.72 %.  Gates are 1 % on eta and on
    u, i.e. both are inside the repository's 3 % documentation threshold.
    """
    g, h0, amp, length = 9.81, 1.0, 0.5, 1.0

    def init_h(xx):
        h, _ = swe_exact.thacker_planar_basin_initial(xx.numpy(), h0, amp, length)
        return torch.as_tensor(h, dtype=xx.dtype)

    def bed(xx):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed(xx.numpy(), h0, length),
                               dtype=xx.dtype)

    x, coeff, zb_faces, meta = solve_swe_1d(
        init_h, lambda z: torch.zeros_like(z), bed, ncells=20, degree=degree,
        final_time=0.1, domain=(-length, length), cfl=0.3, limiter=None,
        rk_order=3, well_balanced=True)
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    h_ex, u_ex, eta_ex = swe_exact.thacker_planar_basin(
        x.numpy(), 0.1, h0, amp, length, g)
    h_ex = torch.as_tensor(h_ex)
    u_ex = torch.as_tensor(u_ex)
    eta_ex = torch.as_tensor(eta_ex)
    rel = lambda a, b: float(torch.sqrt(torch.mean((a - b) ** 2)) / b.abs().max()) * 100.0
    assert rel(eta, eta_ex) < 1.0, rel(eta, eta_ex)
    assert rel(u, u_ex) < 1.0, rel(u, u_ex)
    assert rel(h, h_ex) < 1.0, rel(h, h_ex)


def test_flat_bed_rest_state_is_exact_too():
    _, coeff, _, meta, h, u, eta = _lake_at_rest(ncells=32, degree=2)
    flat = solve_swe_1d(
        lambda xx: torch.full_like(xx, ETA0),
        lambda xx: torch.zeros_like(xx),
        lambda xx: torch.zeros_like(xx),
        ncells=32, degree=2, final_time=1.0, domain=(-2.0, 2.0), cfl=0.3,
        well_balanced=True)
    x2, coeff2, zb2, meta2 = flat
    h2, u2, eta2 = reconstruct_fields(coeff2, zb2, zb_center=meta2["zb_modes"][:, 0])
    assert float((eta2 - ETA0).abs().max()) < 1e-10
    assert float(coeff2[:, :, 1].abs().max()) < 1e-10


def test_well_balanced_does_not_change_the_dry_state_contract():
    """``well_balanced=True`` is *not* a wet/dry reconstruction.

    A dam break with a dry right state still relies on the ``DRY_DEPTH`` guard
    and still reaches the requested final time; the initial mass, however, is
    now the integral of the LGL interpolant, so a discontinuity that falls on a
    cell face is sampled at the face node (measure-zero tie-break).  For the
    Ritter IC at 200 cells that shifts the initial mass by half a cell (0.5 %),
    and the run conserves that discrete mass.
    """
    h0 = 1.0
    x, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: torch.where(xx < 0.0, torch.full_like(xx, h0),
                               torch.full_like(xx, 1e-8)),
        lambda xx: torch.zeros_like(xx),
        lambda xx: torch.zeros_like(xx),
        ncells=200, degree=1, final_time=0.5, domain=(-10.0, 10.0), cfl=0.3,
        rk_order=3, well_balanced=True)
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    assert meta["reached_final_time"]
    mass = float(h.sum() * 0.1)
    assert 9.9 < mass < 10.0, mass  # 9.95: half a wet cell at the tie-break
    assert torch.isfinite(h).all() and torch.isfinite(u).all()
    # The run conserves its *discrete* initial mass: compare with the state the
    # solver started from (t = 0 re-run) rather than with the analytic 10.0.
    _, coeff0, zb0, meta0 = solve_swe_1d(
        lambda xx: torch.where(xx < 0.0, torch.full_like(xx, h0),
                               torch.full_like(xx, 1e-8)),
        lambda xx: torch.zeros_like(xx),
        lambda xx: torch.zeros_like(xx),
        ncells=200, degree=1, final_time=0.0, domain=(-10.0, 10.0),
        well_balanced=True)
    h0d, _, _ = reconstruct_fields(coeff0, zb0)
    assert abs(mass - float(h0d.sum() * 0.1)) < 1e-9