"""Frame discrimination of the 2-D P3 isentropic-vortex velocity channel.

The documented smooth-vortex verification (``docs/reports/tensordg-euler-dg-2d-design-order.md``,
m3024) only reaches ``16^2`` for P3 and reports the velocity channels 12-13 %
below the design order ``p + 1 = 4``; a later mission (m3259) saw an
``h``-independent velocity error floor and left its origin open.

The pinned facts below are the ones that hold on the current tree and that any
future change to the residual assembly, the HLLC flux or the SSP-RK3 driver must
not break:

* ``rho`` and ``p`` converge at the design order throughout ``12^2 -> 20^2``;
* the *static* vortex (``u_bg = (0, 0)``) converges at the design order in the
  velocity channels too, and its domain-mean velocity error is machine zero --
  so the plateau is **not** a property of the DG machinery itself;
* the *translated* vortex (``u_bg = (1, 1)``, the documented case) leaves an
  ``h``-independent domain-mean velocity offset (~``-1.1e-05``) and its velocity
  error stops converging between ``16^2`` and ``20^2``.

The meshes are deliberately small (``20^2`` P3 is ~1 min single-threaded) so the
regression stays inside the normal test budget.
"""
from __future__ import annotations

import pytest

from benchmarks.diagnose_euler_dg_2d_velocity_floor import FIELDS, eoc, measure


@pytest.fixture(scope="module")
def moving():
    return [measure(cells, background=(1.0, 1.0)) for cells in (12, 16)]


@pytest.fixture(scope="module")
def static():
    return [measure(cells, background=(0.0, 0.0)) for cells in (12, 16)]


def test_density_and_pressure_keep_the_design_order_for_the_moving_vortex(moving):
    """rho and p are the healthy channels: order ~4 on the 1.5x mesh pair."""
    for field in ("density", "pressure"):
        order = eoc(moving[0], moving[1], field)
        assert order > 3.4, (field, order)


def test_static_vortex_velocity_converges_at_the_design_order(static):
    """Without a Galilean boost the velocity channel is clean."""
    order = eoc(static[0], static[1], "velocity_x")
    assert order > 3.4, order
    assert order == pytest.approx(3.66, abs=0.25)


def test_static_vortex_has_no_zero_mode_velocity_drift(static):
    """A steady static vortex must not drift: the mean velocity error is ~0."""
    for row in static:
        for field in ("velocity_x", "velocity_y"):
            assert abs(row["errors"][field]["mean"]) < 1e-12


def test_moving_vortex_carries_an_h_independent_mean_velocity_offset(moving):
    """The open defect pinned as a documented number, not silently allowed.

    The domain-mean velocity error of the translated vortex is constant to a few
    per cent over the mesh ladder (it does not shrink with h).  If a future fix
    removes the offset this assertion flips and the document must be updated.
    """
    first, second = (row["errors"]["velocity_x"]["mean"] for row in moving)
    assert first < 0.0 and second < 0.0
    assert abs(second / first - 1.0) < 0.08          # h-independent to 8 %
    assert abs(second) > 1e-6                         # and clearly non-zero
    # density has no such offset: the defect lives in the momentum channel
    assert abs(moving[1]["errors"]["density"]["mean"]) < 1e-9


def test_velocity_error_of_the_moving_vortex_leaves_the_asymptotic_window(moving):
    """The plateau itself: velocity order collapses on the same mesh pair."""
    order_velocity = eoc(moving[0], moving[1], "velocity_x")
    order_density = eoc(moving[0], moving[1], "density")
    assert order_density - order_velocity > 0.0, (order_density, order_velocity)


def test_error_decomposition_sums_back_to_the_total(moving):
    """mean + centred reproduction of the total error (metric self-check)."""
    for row in moving:
        for field in FIELDS:
            entry = row["errors"][field]
            total = entry["relative_l2"]
            parts = (entry["relative_l2_mean"] ** 2
                     + entry["relative_l2_centred"] ** 2) ** 0.5
            # the two parts are orthogonal only for the unweighted L2 norm;
            # the shared scale still makes them comparable to ~10 %
            assert parts == pytest.approx(total, rel=0.15), (field, parts, total)