"""The exact solution of the *periodic* vortex problem, and what the metric sees.

``CASE-DG-EULER-VORTEX-ERR`` compares two periodic solvers against
:func:`tensordg.euler_2d.vortex_primitive`.  Until round 4 that comparison used the
free-space form of the vortex, which is not the solution of the periodic problem
once the centre has been advected across the box wall: at ``t = 5`` with
``u_inf = (1, 1)`` on ``[-5, 5]^2`` the free-space field leaves the far corner at
the undisturbed state while the periodic solution must carry a core there.  The
resulting "error" is a property of the reference, not of the scheme.

These tests pin the invariant at the *metric* level, with a construction that can
fail: the exact solution itself is fed to the error metric.  The metric is a
quadrature of an analytic field, so

* fed the exact **periodic** state and compared with the periodic reference it must
  return zero up to quadrature error (measured below at ``<= 1e-4`` relative);
* fed the exact **free-space** state it must return the reference defect
  (:func:`benchmarks.case_euler_vortex_errconserve.reference_defect_sweep`), which
  is ``> 2x`` the case tolerance at ``t = 5``.

Both statements are false for a metric with a wrong normalisation, a wrong
quadrature, or a reference evaluated on the wrong branch of the min-image map --
which is what makes them useful rather than decorative.
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarks import case_euler_vortex_errconserve as ec
from benchmarks.verify_euler_dg_2d import _vortex_error as dg_vortex_error
from tensordg import euler_dg_2d as dg2
from tensordg.euler_2d import (
    _cell_averages,
    _wrap_shift,
    solve_isentropic_vortex,
    vortex_error,
    vortex_primitive,
)

FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
T_CORNER = 5.0           # centre exactly on the box corner (5, 5)
CORE_DENSITY = 0.4938    # min density of the beta=5 vortex, see corner probe
P0_CELLS = 64
DG_CELLS = 32
DG_DEGREE = 2


def _grid(nx, ny=None):
    ny = nx if ny is None else ny
    (xmin, xmax), (ymin, ymax) = ec.DOMAIN
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    return x, y, dx, dy


def _exact(time, periodic: bool):
    def initial(xx, yy):
        return vortex_primitive(xx, yy, time, ec.STRENGTH, ec.BACKGROUND,
                                period=ec.PERIOD if periodic else None)
    return initial


def _p0_metric(state, time, periodic: bool):
    x, y, dx, dy = _grid(P0_CELLS)
    raw = vortex_error(x, y, state, time, ec.STRENGTH, ec.BACKGROUND,
                       period=ec.PERIOD if periodic else None)
    return {name: raw[name]["relative_l2"] for name in FIELDS}


def _dg_metric(coeff, basis, nodes, time, periodic: bool):
    raw = dg_vortex_error(coeff, basis, nodes, ec.DOMAIN, time, ec.STRENGTH,
                          ec.BACKGROUND,
                          period=ec.PERIOD if periodic else None)
    return {name: float(raw[name]) for name in FIELDS}


def _exact_p0_state(time, periodic: bool):
    x, y, dx, dy = _grid(P0_CELLS)
    return _cell_averages(_exact(time, periodic), x, y, dx, dy, ec.P0_QUADRATURE,
                          torch.device("cpu"), torch.float64)


def _exact_dg_state(time, periodic: bool):
    nodes, weights, basis, _signs, _ = ec._operators(DG_DEGREE)
    x, y, dx, dy = _grid(DG_CELLS)
    coeff = dg2._initial_coefficients(_exact(time, periodic), x, y, dx, dy,
                                      nodes, weights, basis,
                                      torch.device("cpu"), torch.float64)
    return coeff, basis, nodes


# ---------------------------------------------------------------------------
# structural invariants of the periodic reference
# ---------------------------------------------------------------------------

def test_periodic_reference_is_invariant_under_a_time_period_translation():
    """``L / |u_inf| = 10``: the periodic solution repeats exactly in time."""
    x, y, _dx, _dy = _grid(16)  # cells not on the wrap seam
    for time in (0.0, 1.5, 5.0):
        now = vortex_primitive(x[:, None], y[None, :], time, ec.STRENGTH,
                               ec.BACKGROUND, period=ec.PERIOD)
        later = vortex_primitive(x[:, None], y[None, :], time + 10.0, ec.STRENGTH,
                                 ec.BACKGROUND, period=ec.PERIOD)
        for name, a, b in zip(FIELDS, now, later):
            assert torch.allclose(a, b, atol=1e-12), f"{name} not time periodic"


def test_min_image_wrap_folds_into_half_the_box():
    shift = torch.linspace(-30.0, 30.0, 4001, dtype=torch.float64)
    folded = _wrap_shift(shift, 10.0)
    assert float(folded.min()) >= -5.0 and float(folded.max()) <= 5.0
    # exact periodicity of the fold ...
    assert torch.allclose(_wrap_shift(shift + 10.0, 10.0), folded, atol=1e-12)
    # ... and that it *is* the nearest image: the removed part is a multiple of L
    removed = (shift - folded) / 10.0
    assert torch.allclose(removed, removed.round(), atol=1e-12)
    assert float(removed.abs().max()) == 3.0  # |s| <= 30 = 3 L


def test_free_space_reference_agrees_with_the_periodic_one_before_the_wrap():
    """Up to ``t = 1`` the two references coincide, so round-3 t=1 numbers stand."""
    x, y, _dx, _dy = _grid(16)
    for periodic in (False, True):
        field = vortex_primitive(x[:, None], y[None, :], 0.0, ec.STRENGTH,
                                 ec.BACKGROUND,
                                 period=ec.PERIOD if periodic else None)
        if periodic:
            base = field
    free = vortex_primitive(x[:, None], y[None, :], 0.0, ec.STRENGTH, ec.BACKGROUND)
    for name, a, b in zip(FIELDS, free, base):
        assert torch.allclose(a, b, atol=1e-14), f"t=0 mismatch in {name}"
    sweep = ec.reference_defect_sweep([1.0])
    assert max(sweep["1"].values()) < 0.02, "t=1 reference defect is not negligible"


def test_reference_defect_at_the_first_corner_pass_exceeds_case_tolerance():
    """At t=5 the reference defect alone is > 2x the 6 % case tolerance."""
    sweep = ec.reference_defect_sweep([T_CORNER])
    velocity = sweep[f"{T_CORNER:g}"]["velocity_x"]
    density = sweep[f"{T_CORNER:g}"]["density"]
    assert velocity > 2.0 * 100.0 * ec.TOLERANCE, velocity
    assert 13.0 < velocity < 15.5, velocity       # 14.175 % measured
    assert 5.5 < density < 6.8, density           # 6.175 % measured


def test_reference_defect_saturates_after_the_box_crossing():
    """A wrapped field can only see images at distance <= L, so the defect saturates."""
    sweep = ec.reference_defect_sweep([7.5, 10.0])
    late, full = sweep["7.5"]["velocity_x"], sweep["10"]["velocity_x"]
    assert abs(full - late) / full < 5e-3, (late, full)


def test_the_wrapped_corner_carries_the_core_that_free_space_denies():
    """Corner ``(-5,-5)`` at ``t=5``: free space reads ambient, periodic reads a core."""
    corner_x = torch.tensor([-5.0], dtype=torch.float64)
    corner_y = torch.tensor([-5.0], dtype=torch.float64)
    free = vortex_primitive(corner_x, corner_y, T_CORNER, ec.STRENGTH, ec.BACKGROUND)
    wrapped = vortex_primitive(corner_x, corner_y, T_CORNER, ec.STRENGTH,
                               ec.BACKGROUND, period=ec.PERIOD)
    assert abs(float(free[0]) - 1.0) < 1e-6, "free-space far corner is not ambient"
    assert abs(float(wrapped[0]) - CORE_DENSITY) < 5e-3, float(wrapped[0])
    assert abs(float(free[0]) - float(wrapped[0])) > 0.45


# ---------------------------------------------------------------------------
# metric exactness: the exact solution must be reported as error 0 / as the defect
# ---------------------------------------------------------------------------

def test_p0_metric_reports_zero_on_the_exact_periodic_state():
    residual = _p0_metric(_exact_p0_state(T_CORNER, True), T_CORNER, True)
    for name, value in residual.items():
        assert value < 2e-4, (name, value)


def test_dg_metric_reports_zero_on_the_exact_periodic_state():
    coeff, basis, nodes = _exact_dg_state(T_CORNER, True)
    residual = _dg_metric(coeff, basis, nodes, T_CORNER, True)
    for name, value in residual.items():
        assert value < 2e-4, (name, value)


def test_metrics_read_the_reference_defect_when_fed_the_free_space_solution():
    """The mismatched branch returns the defect, not a small number."""
    defect = ec.reference_defect_sweep([T_CORNER])[f"{T_CORNER:g}"]
    p0 = _p0_metric(_exact_p0_state(T_CORNER, False), T_CORNER, True)
    coeff, basis, nodes = _exact_dg_state(T_CORNER, False)
    dg = _dg_metric(coeff, basis, nodes, T_CORNER, True)
    for metric in (p0, dg):
        for name in ("density", "velocity_x", "pressure"):
            expected = 0.01 * defect[name]
            assert metric[name] == pytest.approx(expected, rel=0.05), (name, metric[name])


def test_vortex_error_defaults_to_the_legacy_free_space_reference():
    """Backward compatibility: omitting ``period`` must not change the old readings."""
    state = _exact_p0_state(T_CORNER, False)
    x, y, _dx, _dy = _grid(P0_CELLS)
    default = vortex_error(x, y, state, T_CORNER, ec.STRENGTH, ec.BACKGROUND)
    explicit = vortex_error(x, y, state, T_CORNER, ec.STRENGTH, ec.BACKGROUND,
                            period=None)
    for name in FIELDS:
        assert default[name]["relative_l2"] == explicit[name]["relative_l2"]


# ---------------------------------------------------------------------------
# end-to-end: the same P0 solution, two references
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("time,cells,cfl", [(1.0, 64, 0.4), (5.0, 64, 0.4)])
def test_p0_reading_switches_with_the_reference_only_after_the_wrap(time, cells, cfl):
    """t=1: the reading is reference-independent; t=5: it is not (35 % apart, measured)."""
    x, y, fields, _meta = solve_isentropic_vortex(
        ncells=cells, final_time=time, cfl=cfl, domain=ec.DOMAIN)
    state = fields["conservative"]
    periodic = _p0_metric(state, time, True)
    free = _p0_metric(state, time, False)
    worst_periodic = max(periodic.values())
    worst_free = max(free.values())
    if time <= 1.0:
        # free-space reference is still the solution: both readings coincide
        assert abs(worst_periodic - worst_free) / worst_free < 1e-4
        assert worst_periodic < ec.TOLERANCE, worst_periodic
    else:
        # after the wrap they differ by ~1/3 relative -- the reference must be stated
        assert abs(worst_periodic - worst_free) / worst_free > 0.2
        assert math.isfinite(worst_periodic)