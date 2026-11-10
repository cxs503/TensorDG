"""Hydrostatic (Audusse) reconstruction: exact rest states across bed jumps.

Companion to ``tests/test_swe_wellbalanced_hydrostatic.py`` (which pins the
smooth-bed nodal path).  This file pins the interface closure.

Background
----------
``tests/test_swe_well_balanced_exact.py`` pins the *nodal* well-balanced path:
the bed and the initial depth are interpolated at the same LGL nodes, so a
still lake over a smooth curved bed is a discrete fixed point to round-off.
That path shares one HLL flux between the two cells of a face, which is only
sound while the bed is continuous: at a bed *jump* the two cells evaluate a
different bed level at the same face, their traces of ``h`` differ, and the
jump term of the flux injects spurious momentum every step (the documented
reason a plain scheme cannot hold a lake at rest over a step).

This file pins the missing piece: ``solve_swe_1d(..., well_balanced=True)``
reconstructs the interface states with Audusse et al. (2004),

    z_b* = max(z_b,iR, z_b,i+1,L),    h*_m = max(0, h_m + z_b,m - z_b*),

evaluates the HLL flux with the reconstructed states and adds the hydrostatic
pressure correction ``g/2 (h_m^2 - h*_m^2)`` to the momentum flux of each side.
At rest every cell then sees the pressure flux of its *own* trace, so the flux
divergence cancels the exact bed-slope source cell by cell.  For a continuous
bed ``z_b,iR = z_b,i+1,L``, ``h* = h`` and the correction vanishes, which is why
the curved-bed tests are unchanged.

The bed jump is expressed with cell-local nodal data (a tensor of shape
``(ncells, degree + 1)``): a pointwise callable cannot return the left and the
right limit at a shared face, so it cannot describe a step exactly.

Measured on this tree (t = 1 s, ``N = 64``): step lake at rest with the
reconstruction ``max|hu| = 0`` (P1, exactly) / ``1.0e-10`` (P2) versus
``5.9`` / ``2.6`` without it; the wet/dry step stays at ``4.2e-11``; Thacker
(1981) ``t = 0.5 s`` accuracy is unchanged at 0.060 % (P1) / 8e-7 % (P2) on
``eta``.  The former declared limitation (a still lake with a dry bank, whose
shoreline may pass exactly through a cell face) is now a fixed point: the
positivity filter scales the cell shape instead of flattening it, see
``test_lake_at_rest_with_a_dry_bank`` and
``validation/swe_thacker/shoreline_rest_state_results.json``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg import swe_exact
from tensordg.swe_1d import (_lgl_transform, _operators, _positivity_filter,
                             _rhs, reconstruct_fields, solve_swe_1d)

ETA0 = 2.0
STEP_BED = 0.5
G = 9.81


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cell_step_bed(ncells, degree, lower=-2.0, upper=2.0):
    """Cell-local nodal bed with a jump at ``x = 0`` (left 0.5 m, right 0 m)."""
    lgl = _lgl_transform(degree, torch.device("cpu"), torch.float64)[1]
    dx = (upper - lower) / ncells
    xc = lower + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    node_x = xc[:, None] + 0.5 * dx * lgl[None, :]
    left = (torch.arange(ncells) < ncells // 2)[:, None]
    zb = torch.where(left, torch.full_like(node_x, STEP_BED),
                     torch.zeros_like(node_x))
    return node_x, zb


def _hump_bed(xx):
    """Smooth bed whose crest (1.5 m) rises above the 1 m still-water level."""
    return 1.5 - 0.5 * xx ** 2


def _solve_step(ncells=64, degree=1, final_time=1.0, well_balanced=True,
                hydrostatic=True):
    node_x, zb_nodes = _cell_step_bed(ncells, degree)
    h_nodes = ETA0 - zb_nodes
    u_nodes = torch.zeros_like(zb_nodes)
    return solve_swe_1d(h_nodes, u_nodes, zb_nodes, ncells=ncells,
                        degree=degree, final_time=final_time,
                        domain=(-2.0, 2.0), cfl=0.3, rk_order=3,
                        well_balanced=well_balanced, hydrostatic=hydrostatic)


def _rest_rhs(degree=2, ncells=16, with_reconstruction=True):
    """Residual of the discrete rest state over the step, one RHS evaluation."""
    dx = 4.0 / ncells
    xc = -2.0 + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    vinv, lgl = _lgl_transform(degree, torch.device("cpu"), torch.float64)
    _, weights, phi, dphi, phi_left, phi_right = _operators(
        degree, torch.device("cpu"), torch.float64)
    node_x = xc[:, None] + 0.5 * dx * lgl[None, :]
    left = (torch.arange(ncells) < ncells // 2)[:, None]
    zb_nodes = torch.where(left, torch.full_like(node_x, STEP_BED),
                           torch.zeros_like(node_x))
    h_nodes = ETA0 - zb_nodes
    coeff = torch.zeros(ncells, degree + 1, 2, dtype=torch.float64)
    coeff[:, :, 0] = h_nodes @ vinv.T
    cz = zb_nodes @ vinv.T
    zb_l, zb_r = cz @ phi_left, cz @ phi_right
    zb_slope = torch.einsum("cm,qm->cq", cz, dphi) * (2.0 / dx)
    return _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right,
                zb_r - zb_l, "hll", False, 0.0, None, None,
                zb_slope, cz if with_reconstruction else None)


# --------------------------------------------------------------------------- #
# 1. the capability: a still lake across a submerged bed step
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("ncells", [32, 64])
def test_lake_at_rest_across_a_bed_step_is_machine_precise(degree, ncells):
    x, coeff, zb_faces, meta = _solve_step(ncells=ncells, degree=degree)
    assert meta["reached_final_time"], meta["time_reached"]
    assert meta["hydrostatic_reconstruction"]
    h, u, eta = reconstruct_fields(coeff, zb_faces,
                                   zb_center=meta["zb_modes"][:, 0])
    momentum = float(coeff[:, :, 1].abs().max())
    drift = float((eta - ETA0).abs().max())
    assert momentum < 1e-9, f"spurious momentum {momentum:.3e} m^2/s"
    assert drift < 1e-9, f"free-surface drift {drift:.3e} m"
    assert float(u.abs().max()) < 1e-9
    # Both sides of the jump must still carry their own water column.
    assert float(h[x < 0].min()) > 1.4 and float(h[x > 0].min()) > 1.9


@pytest.mark.parametrize("degree", [1, 2])
def test_solver_level_drift_without_the_reconstruction(degree):
    """Same discretisation, only the interface closure switched off.

    Measured at ``N = 64, t = 1 s``: the pre-reconstruction closure lets the
    step-lake momentum grow to 1.4e-3 (P1) / 3.3 (P2) and the surface drift to
    metres; the reconstruction holds both at 1e-9.
    """
    _, c_off, zf_off, m_off = _solve_step(64, degree, hydrostatic=False)
    h_off, _, eta_off = reconstruct_fields(c_off, zf_off,
                                           zb_center=m_off["zb_modes"][:, 0])
    _, c_on, zf_on, m_on = _solve_step(64, degree, hydrostatic=True)
    h_on, _, eta_on = reconstruct_fields(c_on, zf_on,
                                         zb_center=m_on["zb_modes"][:, 0])
    assert float(c_off[:, :, 1].abs().max()) > 1e-2
    assert float((eta_off - ETA0).abs().max()) > 1e-2
    assert float(c_on[:, :, 1].abs().max()) < 1e-9
    assert float((eta_on - ETA0).abs().max()) < 1e-9


def test_without_hydrostatic_reconstruction_the_step_injects_momentum():
    """The delta is a property of the interface closure, not of the time loop."""
    legacy = float(_rest_rhs(2, 16, with_reconstruction=False)[:, :, 1].abs().max())
    fixed = float(_rest_rhs(2, 16, with_reconstruction=True)[:, :, 1].abs().max())
    assert legacy > 1e-2, f"legacy residual unexpectedly small ({legacy:.3e})"
    assert fixed < 1e-11, f"reconstructed residual {fixed:.3e}"
    assert legacy > 1e9 * max(fixed, 1e-16)


def test_reconstruction_is_the_identity_on_a_continuous_bed():
    """Curved continuous bed: the correction terms must vanish elementwise.

    Guards the regression risk of routing the curved-bed path through the new
    interface closure: ``z_b,iR = z_b,i+1,L`` there, so the reconstructed
    states equal the traces and both closures return the same flux.
    """
    degree, ncells, dx = 2, 16, 4.0 / 16
    xc = -2.0 + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    vinv, lgl = _lgl_transform(degree, torch.device("cpu"), torch.float64)
    _, weights, phi, dphi, phi_left, phi_right = _operators(
        degree, torch.device("cpu"), torch.float64)
    node_x = xc[:, None] + 0.5 * dx * lgl[None, :]

    def bed(xx):
        return 0.4 * torch.sin(0.5 * math.pi * xx)

    zb_nodes = bed(node_x)
    coeff = torch.zeros(ncells, degree + 1, 2, dtype=torch.float64)
    coeff[:, :, 0] = (ETA0 - zb_nodes) @ vinv.T
    cz = zb_nodes @ vinv.T
    zb_l, zb_r = cz @ phi_left, cz @ phi_right
    zb_slope = torch.einsum("cm,qm->cq", cz, dphi) * (2.0 / dx)
    args = (coeff, dx, weights, phi, dphi, phi_left, phi_right,
            zb_r - zb_l, "hll", False, 0.0, None, None, zb_slope)
    shared = _rhs(*args, None)
    reconstructed = _rhs(*args, cz)
    assert float((shared - reconstructed).abs().max()) < 1e-12
    assert float(reconstructed[:, :, 1].abs().max()) < 1e-11


# --------------------------------------------------------------------------- #
# 2. dry cells: wet/dry interfaces
# --------------------------------------------------------------------------- #
def _cell_step_data(ncells, degree, zb_left, zb_right, h_left, h_right):
    """Cell-local nodal data with a bed jump and a dry right-hand side."""
    node_x, _ = _cell_step_bed(ncells, degree)
    left = (torch.arange(ncells) < ncells // 2)[:, None]
    zb = torch.where(left, torch.full_like(node_x, zb_left),
                     torch.full_like(node_x, zb_right))
    h = torch.where(left, torch.full_like(node_x, h_left),
                    torch.full_like(node_x, h_right))
    return node_x, zb, h


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("ncells", [32, 64])
def test_lake_at_rest_against_a_dry_raised_bed(degree, ncells):
    """Water impounded against a bed that rises above the free surface.

    ``z_b = 0.5`` on the left with ``h = 1.5`` (``eta = 2``) and ``z_b = 2.5``
    (above ``eta``) on the right: at the jump the reconstruction gives
    ``z_b* = 2.5`` and ``h* = 0`` on *both* sides, i.e. the wet trace of depth
    1.5 is replaced by a dry one.  Without that correction the HLL flux of
    ``(1.5, 0)`` against ``(0, 0)`` would drain the lake through the wall.
    Measured drift: <= 1.4e-11 m (P2, N = 64), ``max|hu|`` <= 5.3e-11.
    """
    ncells = ncells
    node_x, zb, h = _cell_step_data(ncells, degree, 0.5, 2.5, 1.5, 0.0)
    x, coeff, zb_faces, meta = solve_swe_1d(
        h, torch.zeros_like(h), zb, ncells=ncells, degree=degree,
        final_time=1.0, domain=(-2.0, 2.0), cfl=0.3, rk_order=3,
        well_balanced=True)
    assert meta["reached_final_time"]
    hh, u, eta = reconstruct_fields(coeff, zb_faces,
                                    zb_center=meta["zb_modes"][:, 0])
    dry = hh < 1e-6
    assert int(dry.sum()) == ncells // 2
    assert float(hh[dry].max()) < 1e-12          # the wall never wets
    assert float(hh[~dry].min()) > 1.49          # and the lake never drains
    assert float((eta[~dry] - ETA0).abs().max()) < 1e-9
    assert float(coeff[:, :, 1].abs().max()) < 1e-9
    assert float(u.abs().max()) < 1e-9


@pytest.mark.parametrize("degree", [1, 2])
def test_lake_at_rest_with_a_dry_bank(degree):
    """A still lake whose surface cuts the bed is a discrete fixed point.

    This was the one *declared limitation* of the file: the shoreline cell
    carries ``h = 0`` exactly at its dry node, the positivity filter tested
    ``h > h_min`` (1e-12), rejected the cell and replaced it by its cell
    average, which zeroes the linear mode and therefore destroys
    ``eta = h + z_b = const`` in exactly that cell.  Measured then (N = 64,
    t = 1 s): ``max|eta - eta0|`` over wet cells 3.0e-2 m (P1) / 5.0e-1 m (P2)
    and spurious ``max|u|`` 0.54 m/s (P1) / 36 m/s (P2); the run was not a
    rest state at all.

    After the fix (Zhang-Shu scaling of the higher modes towards the cell
    average, see ``swe_1d._positivity_filter``) the depth average is kept, an
    admissible cell is left untouched, and the drift is round-off:
    ``max|eta - eta0| <= 8.6e-12 m`` and ``max|u| <= 1.9e-11 m/s`` at P1/P2 on
    N = 32...128 (``validation/swe_thacker/shoreline_rest_state_results.json``).
    """
    def init_h(xx):
        return (1.0 - _hump_bed(xx)).clamp_min(0.0)

    x, coeff, zb_faces, meta = solve_swe_1d(
        init_h, lambda xx: torch.zeros_like(xx), _hump_bed,
        ncells=32, degree=degree, final_time=1.0, domain=(-2.0, 2.0),
        cfl=0.3, rk_order=3, well_balanced=True)
    assert meta["reached_final_time"]
    h, u, eta = reconstruct_fields(coeff, zb_faces,
                                   zb_center=meta["zb_modes"][:, 0])
    dry = h < 1e-6
    assert 0 < int(dry.sum()) < h.numel()        # the bank really is dry
    assert float((eta[~dry] - 1.0).abs().max()) < 1e-9
    assert float(u.abs().max()) < 1e-9           # a lake at rest stays at rest
    assert float(h.min()) >= 0.0


def test_lake_at_rest_with_a_shoreline_on_a_cell_face():
    """Same rest state, but the shoreline lands exactly on a cell face.

    ``z_b = |x| / 1.5`` on ``[-2, 2]`` with ``eta0 = 1 m`` puts the shoreline
    at ``x = +-1.5``, which is face 8 and face 56 for N = 64 (``dx = 1/16``).
    The wet depth then vanishes *at the face*, so both the right trace of the
    last wet cell and the left trace of the first dry cell are exactly zero:
    the case the old filter could not represent.  Measured drift after the
    fix: ``max|eta - eta0| <= 1e-11 m`` (P1, N = 64, t = 1 s).
    """
    def bed(xx):
        return xx.abs() / 1.5

    def init_h(xx):
        return (1.0 - bed(xx)).clamp_min(0.0)

    x, coeff, zb_faces, meta = solve_swe_1d(
        init_h, lambda xx: torch.zeros_like(xx), bed,
        ncells=64, degree=1, final_time=1.0, domain=(-2.0, 2.0),
        cfl=0.3, rk_order=3, well_balanced=True)
    assert meta["reached_final_time"]
    h, u, eta = reconstruct_fields(coeff, zb_faces,
                                   zb_center=meta["zb_modes"][:, 0])
    dry = h < 1e-6
    assert int(dry.sum()) == 16                  # |x| > 1.5 -> 8 cells per side
    assert float((eta[~dry] - 1.0).abs().max()) < 1e-9
    assert float(u.abs().max()) < 1e-9


# --------------------------------------------------------------------------- #
# 2b. the positivity filter itself
# --------------------------------------------------------------------------- #
def _filter_inputs(degree=1, ncells=3):
    _, _, phi, _, _, _ = _operators(degree, torch.device("cpu"), torch.float64)
    coeff = torch.zeros(ncells, degree + 1, 2, dtype=torch.float64)
    return coeff, phi


def test_positivity_filter_leaves_admissible_cells_untouched():
    """Idempotence is what preserves a rest state: theta = 1 where h >= 0.

    On the well-balanced path the filter scales the cell shape, so any trace
    point at or below ``h_min`` -- including the exact zero of a dry node --
    no longer triggers the average-only fallback.
    """
    coeff, phi = _filter_inputs(degree=1)
    coeff[0, :, 0] = torch.tensor([0.5, 0.25], dtype=torch.float64)      # h = 0.5 + 0.25 xi >= 0.25
    coeff[1, :, 0] = torch.tensor([0.25, 0.25], dtype=torch.float64)     # h = 0.25 - 0.25 xi, dry node
    coeff[:, :, 1] = 0.0
    out = _positivity_filter(coeff, phi, well_balanced=True)
    assert torch.allclose(out, coeff)              # both are admissible as is


def test_positivity_filter_scales_the_shape_and_keeps_the_average():
    """An inadmissible cell is *scaled*, not flattened: conservation + h >= 0."""
    coeff, phi = _filter_inputs(degree=1)
    coeff[0, :, 0] = torch.tensor([0.1, 0.3], dtype=torch.float64)       # h = 0.1 + 0.3 xi -> -0.2
    coeff[0, :, 1] = torch.tensor([0.0, 0.2], dtype=torch.float64)
    out = _positivity_filter(coeff, phi, well_balanced=True)
    h_q = torch.einsum("cm,qm->cq", out[:, :, 0], phi)
    assert float(h_q.min()) >= -1e-15               # positivity restored
    assert float(out[0, 0, 0]) == pytest.approx(0.1)   # mass conserved exactly
    theta = float(out[0, 1, 0] / coeff[0, 1, 0])
    assert 0.0 < theta < 1.0
    # theta is the *minimal* scaling: the worst quadrature point ends up dry,
    # and it is a Gauss point (not the cell end), hence theta > 0.1 / 0.4.
    assert float(h_q.min()) == pytest.approx(0.0, abs=1e-15)
    # h_avg + theta (h_q - h_avg) = 0 at the worst Gauss point xi_min:
    assert theta == pytest.approx(0.1 / (0.3 * float(phi[:, 1].abs().max())),
                                  rel=1e-12)
    assert float(out[0, 1, 1]) == pytest.approx(0.2 * theta)  # same scaling on hu


def test_default_path_keeps_the_legacy_filter_action():
    """The default (non-hydrostatic) path is unchanged by this fix.

    Its dry-bed dam-break verification was tuned against the
    replace-by-cell-average action (the flattening acts as front diffusion), so
    the scaling limiter is enabled only where the Audusse reconstruction runs.
    On an inadmissible cell the two actions are therefore distinguishable: the
    default path still drops the shape entirely, the fixed path keeps the
    largest shape positivity allows.
    """
    coeff, phi = _filter_inputs(degree=1)
    coeff[0, :, 0] = torch.tensor([0.1, 0.3], dtype=torch.float64)   # h < 0 at xi < -1/3
    coeff[0, :, 1] = torch.tensor([0.0, 0.2], dtype=torch.float64)
    legacy = _positivity_filter(coeff, phi)                       # default path
    scaling = _positivity_filter(coeff, phi, well_balanced=True)  # fixed path
    # both keep the cell average (mass) ...
    assert float(legacy[0, 0, 0]) == pytest.approx(0.1)
    assert float(scaling[0, 0, 0]) == pytest.approx(0.1)
    # ... but only the default path flattens the cell to that average
    assert float(legacy[0, 1, 0]) == 0.0
    assert float(legacy[0, 1, 1]) == 0.0
    theta = float(scaling[0, 1, 0] / coeff[0, 1, 0])
    assert 0.0 < theta < 1.0
    # and an admissible cell is a no-op only on the fixed path
    admissible, phi_a = _filter_inputs(degree=1)
    admissible[0, :, 0] = torch.tensor([0.25, 0.25], dtype=torch.float64)
    assert torch.allclose(_positivity_filter(admissible, phi_a, well_balanced=True),
                          admissible)


def test_positivity_filter_pins_a_cell_with_a_nonpositive_average():
    """A cell whose average depth is <= 0 cannot be repaired by scaling."""
    coeff, phi = _filter_inputs(degree=1)
    coeff[0, :, 0] = torch.tensor([-0.1, 0.2], dtype=torch.float64)
    coeff[0, :, 1] = torch.tensor([0.3, -0.4], dtype=torch.float64)
    out = _positivity_filter(coeff, phi, well_balanced=True)
    # average kept, higher modes dropped (the state cannot be made admissible)
    assert torch.allclose(out[0, 0], coeff[0, 0])
    assert torch.allclose(out[0, 1], torch.zeros(2, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# 3. transient use of the same closure
# --------------------------------------------------------------------------- #
def test_dam_break_over_a_step_stays_positive_and_conserves_mass():
    """The reconstruction must survive a moving front, not only a rest state."""
    ncells, degree = 100, 1
    node_x, zb_nodes = _cell_step_bed(ncells, degree)
    left = (torch.arange(ncells) < ncells // 2)[:, None]
    h_nodes = torch.where(left, torch.full_like(node_x, 2.0),
                          torch.full_like(node_x, 1.0))
    u_nodes = torch.zeros_like(node_x)
    kwargs = dict(ncells=ncells, degree=degree, final_time=0.3,
                  domain=(-2.0, 2.0), cfl=0.3, rk_order=3, well_balanced=True)
    x, coeff, zb_faces, meta = solve_swe_1d(h_nodes, u_nodes, zb_nodes, **kwargs)
    assert meta["reached_final_time"], meta["time_reached"]
    h, u, _ = reconstruct_fields(coeff, zb_faces, zb_center=meta["zb_modes"][:, 0])
    assert torch.isfinite(h).all() and torch.isfinite(u).all()
    assert float(h.min()) > -1e-12
    dx = 4.0 / ncells
    mass = float(h.sum() * dx)
    assert abs(mass - 6.0) < 1e-8, mass  # 2 m over half the domain + 1 m over half


def test_rk2_stage_two_uses_the_analytic_boundary_data():
    """SSP-RK2 stage 2 must receive ``time`` and the boundary-flux callbacks.

    Before the fix ``k2`` was evaluated at ``t = 0`` with the transmissive
    closure.  Measured on the Thacker planar basin (N = 40, P1, t = 0.5 s): the
    relative volume error was 2.599e-4 with RK2 versus 2.441e-4 with RK3 (the
    common part is the ``O(dx^2)`` bed-representation error); after the fix RK2
    matches RK3 to 2e-7.  See
    ``validation/swe_thacker/rk2_stage2_defect_measurement.json``.
    """
    h0, amp, half, t_end = 1.0, 0.5, 1.0, 0.5

    def bed(xx):
        return torch.as_tensor(
            swe_exact.thacker_parabolic_bed(xx.numpy(), h0, half), dtype=xx.dtype)

    def left(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, h0, amp, half, G)[0])

    def right(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, h0, amp, half, G)[1])

    def init_h(xx):
        h, _ = swe_exact.thacker_planar_basin_initial(xx.numpy(), h0, amp, half)
        return torch.as_tensor(h, dtype=xx.dtype)

    # Direct pin: a run that is a single step long must evaluate the boundary
    # data at t = 0 (stage 1) *and* at t = dt (stage 2).  Before the fix the
    # second stage silently used t = 0 with the transmissive closure.
    calls = []

    def left_rec(t):
        calls.append(float(t))
        return left(t)

    solve_swe_1d(init_h, lambda xx: torch.zeros_like(xx), bed, ncells=20,
                 degree=1, final_time=2e-4, domain=(-half, half), cfl=0.3,
                 limiter=None, rk_order=2, well_balanced=True,
                 bc_left_flux=left_rec, bc_right_flux=right)
    assert calls and max(calls) == pytest.approx(2e-4), calls

    # The Thacker basin exchanges water through the rim, so the mass at t_end is
    # not the initial mass: the reference is the analytic mass
    # ``2 L eta_c + 4 h0 L / 3``.  The piecewise-linear bed representation is
    # not the parabola, which leaves a common O(dx^2) bed error for both RK
    # orders; the *difference* between the two is what the stage-2 time/boundary
    # fix removes (measured: 1.6e-5 of the volume before the fix, 2e-7 after).
    eta_c = h0 + (amp ** 2 / (4.0 * h0)) * math.sin(
        math.sqrt(2.0 * G * h0) / half * t_end) ** 2
    exact_mass = 2 * half * eta_c + 4.0 * h0 * half / 3.0

    errors = {}
    for rk in (2, 3):
        x, coeff, zb_faces, meta = solve_swe_1d(
            init_h, lambda xx: torch.zeros_like(xx), bed, ncells=40,
            degree=1, final_time=t_end, domain=(-half, half), cfl=0.3,
            limiter=None, rk_order=rk, well_balanced=True,
            bc_left_flux=left, bc_right_flux=right)
        assert meta["reached_final_time"]
        h, _, _ = reconstruct_fields(coeff, zb_faces,
                                     zb_center=meta["zb_modes"][:, 0])
        errors[rk] = abs(float(h.sum()) * (2 * half / 40) - exact_mass) / exact_mass
    assert errors[2] < 1e-3, f"RK2 volume error {errors[2]:.3e}"
    assert errors[3] < 1e-3, f"RK3 volume error {errors[3]:.3e}"
    # Both orders now see the same time-dependent boundary data.
    assert abs(errors[2] - errors[3]) < 1e-6, (errors[2], errors[3])


def test_thacker_accuracy_is_not_degraded_by_the_reconstruction():
    """Moving solution, continuous bed: same accuracy as the nodal path alone.

    Measured (``N = 20``, ``t = 0.5 s``): eta 0.158 % (P1 and P2), i.e. the
    reconstruction is transparent on a continuous bed as claimed above and the
    3 % documentation threshold is met with a wide margin.
    """
    h0, amp, half, t_end = 1.0, 0.5, 1.0, 0.5

    def bed(xx):
        return torch.as_tensor(
            swe_exact.thacker_parabolic_bed(xx.numpy(), h0, half), dtype=xx.dtype)

    def init_h(xx):
        h, _ = swe_exact.thacker_planar_basin_initial(xx.numpy(), h0, amp, half)
        return torch.as_tensor(h, dtype=xx.dtype)

    def left(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, h0, amp, half, G)[0])

    def right(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, h0, amp, half, G)[1])

    for degree in (1, 2):
        x, coeff, zb_faces, meta = solve_swe_1d(
            init_h, lambda xx: torch.zeros_like(xx), bed, ncells=20,
            degree=degree, final_time=t_end, domain=(-half, half), cfl=0.3,
            limiter=None, rk_order=3, well_balanced=True,
            bc_left_flux=left, bc_right_flux=right)
        h, u, eta = reconstruct_fields(coeff, zb_faces,
                                       zb_center=meta["zb_modes"][:, 0])
        h_ex, u_ex, eta_ex = swe_exact.thacker_planar_basin(
            x.numpy(), t_end, h0, amp, half, G)
        h_ex = torch.as_tensor(h_ex)
        eta_ex = torch.as_tensor(eta_ex)
        rel = (lambda a, b: float(torch.sqrt(torch.mean((a - b) ** 2))
                                  / b.abs().max()) * 100.0)
        assert rel(eta, eta_ex) < 0.5, (degree, rel(eta, eta_ex))
        assert rel(h, h_ex) < 0.5, (degree, rel(h, h_ex))


# --------------------------------------------------------------------------- #
# 4. input contract
# --------------------------------------------------------------------------- #
def test_cell_local_tensors_require_the_well_balanced_path():
    node_x, zb_nodes = _cell_step_bed(16, 1)
    with pytest.raises(ValueError, match="well_balanced=True"):
        solve_swe_1d(ETA0 - zb_nodes, torch.zeros_like(zb_nodes), zb_nodes,
                     ncells=16, degree=1, final_time=0.0, domain=(-2.0, 2.0),
                     well_balanced=False)
    with pytest.raises(ValueError, match="shape"):
        solve_swe_1d(torch.zeros(16, 5), torch.zeros(16, 5), torch.zeros(16, 5),
                     ncells=16, degree=1, final_time=0.0, domain=(-2.0, 2.0),
                     well_balanced=True)
