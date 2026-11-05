"""Regression tests for the ``limiter_stage`` parameter of ``solve_euler_dg_2d``.

The 1-D solver gained ``limiter_stage`` in mission m2500; the 2-D solver was
missing it, so it always limited after every SSP-RK3 sub-step (``"every"``).
PR #94 added the option to :func:`solve_euler_dg_2d` but not to the vortex
driver, and its order test compared the two policies on a configuration where
the limiter is provably inert; both defects are fixed here.

Measured behaviour of the P2 isentropic vortex (density relative L2 error,
``N`` cells per direction, ``T`` final time; full data in
``artifacts/pr94_fix/sweep.json``):

=========================================  =================  ==========
configuration                              errors             EOC
=========================================  =================  ==========
c = 10, ``"every"``, N = 8 -> 16, T = 1    3.275e-3 / 3.976e-4   3.042
c = 10, ``"final"``, N = 8 -> 16, T = 1    3.275e-3 / 3.976e-4   3.042
c = 0,  ``"every"``, N = 8 -> 16, T = 1    4.267e-2 / 2.027e-2   1.074
c = 0,  ``"final"``, N = 8 -> 16, T = 1    4.266e-2 / 2.039e-2   1.065
c = 0,  ``"every"``, N = 8, T = 0.05       3.1607e-2             --
c = 0,  ``"final"``, N = 8, T = 0.05       3.1430e-2             --
=========================================  =================  ==========

Two consequences drive this file:

* with a *scale-consistent* TVB constant (c = 10) the detector never flags this
  smooth flow, so the two policies are bit-identical and the design order
  ``p + 1 = 3`` survives -- comparing EOC("every") < EOC("final") there is
  therefore unphysical (both are 3.042), and the accuracy claim to guard is
  that the ``"final"`` policy does *not* damage the order;
* with a *strict* TVB constant (c = 0) every curved cell is limited, the two
  policies genuinely differ (relative difference ~5.6e-3 at ``T = 0.05``), and
  the observed order collapses to ~1 for both -- that is the configuration
  that proves the flag really reaches the sub-steps.

Tests:

* ``test_limiter_stage_is_validated`` -- invalid values are rejected;
* ``test_limiter_stage_inert_without_limiter`` -- ``"every"`` and ``"final"``
  produce identical results when no limiter is active;
* ``test_limiter_stage_preserves_design_order_on_vortex`` -- with a
  scale-consistent TVB constant ``"final"`` keeps EOC > 2.7 and is
  indistinguishable from ``"every"`` (the limiter never fires);
* ``test_limiter_stage_reaches_the_rk_substeps`` -- with an active limiter the
  two policies give different end-of-step states;
* ``test_limiter_stage_recorded_in_metadata`` -- the value is echoed back.
"""
import math

import pytest
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))


def _vortex(ncells, degree, final_time=1.0, limiter=None, limiter_stage="every",
            tvb_constant=1.0):
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells, degree=degree, final_time=final_time,
        limiter=limiter, limiter_stage=limiter_stage, tvb_constant=tvb_constant,
    )
    return _vortex_error(coeff, basis, nodes, DOMAIN, final_time), metadata


def test_limiter_stage_is_validated():
    """Invalid limiter_stage values must raise ValueError."""
    with pytest.raises(ValueError, match="limiter_stage"):
        dg2.solve_euler_dg_2d(
            lambda x, y: (torch.ones_like(x), torch.zeros_like(x),
                          torch.zeros_like(x), torch.ones_like(x)),
            ncells=4, degree=1, final_time=0.01, limiter_stage="sometimes",
        )


def test_limiter_stage_inert_without_limiter():
    """When limiter=None, 'every' and 'final' must produce identical results."""
    for degree in (1, 2):
        _, _, coeff_every, _ = dg2.solve_euler_dg_2d_vortex(
            ncells=6, degree=degree, final_time=0.02, limiter=None,
            limiter_stage="every",
        )
        _, _, coeff_final, _ = dg2.solve_euler_dg_2d_vortex(
            ncells=6, degree=degree, final_time=0.02, limiter=None,
            limiter_stage="final",
        )
        assert torch.equal(coeff_every, coeff_final), (
            f"degree {degree}: every vs final differ without limiter"
        )


def test_limiter_stage_preserves_design_order_on_vortex():
    """The ``"final"`` policy must keep the P2 design order on smooth data.

    ``tvb_constant=10`` is scale consistent with the vortex: the detector never
    flags a smooth cell, so the limiter is inert and the P2 run has to converge
    at (nearly) ``p + 1 = 3`` -- measured 3.042.  The same configuration also
    makes the two policies bit-equal, which is why the accuracy gate is stated
    per policy and not as an ``"every" < "final"`` comparison.
    """
    errors = {}
    for stage in ("final", "every"):
        collected = []
        for cells in (8, 16):
            err, meta = _vortex(cells, 2, final_time=1.0, limiter="tvb",
                                limiter_stage=stage, tvb_constant=10.0)
            collected.append(err["density"])
            assert meta["steps"] > 0
            assert meta["limiter_stage"] == stage
        errors[stage] = collected
        eoc = math.log(collected[0] / collected[1]) / math.log(2.0)
        assert eoc > 2.7, (
            f"P2 vortex with limiter_stage={stage!r}: EOC={eoc:.3f} < 2.7"
        )

    for stage in ("final", "every"):
        assert math.isclose(
            errors["final"][0], errors[stage][0], rel_tol=1e-12
        ), "with a scale-consistent TVB constant the policies must coincide"


def test_limiter_stage_reaches_the_rk_substeps():
    """An active limiter must make the two policies produce different states.

    ``tvb_constant=0`` is strict TVD: every cell carrying curvature is limited,
    so the sub-step policy is actually felt (measured relative difference
    5.6e-3 at ``N = 8``, ``T = 0.05``).  If the flag were ignored -- the defect
    this file guards -- both runs would be bit-identical.
    """
    errs = {}
    for stage in ("every", "final"):
        err, meta = _vortex(8, 2, final_time=0.05, limiter="tvb",
                            limiter_stage=stage, tvb_constant=0.0)
        assert meta["limiter_stage"] == stage
        errs[stage] = err["density"]

    assert not math.isclose(errs["every"], errs["final"], rel_tol=1e-6), (
        "limiter_stage did not reach the SSP-RK3 sub-steps: "
        f"every={errs['every']:.6e} final={errs['final']:.6e}"
    )


def test_limiter_stage_recorded_in_metadata():
    """The metadata must record the limiter_stage value."""
    for stage in ("every", "final"):
        _, _, _, meta = dg2.solve_euler_dg_2d_vortex(
            ncells=4, degree=1, final_time=0.01, limiter="tvb",
            limiter_stage=stage,
        )
        assert meta["limiter_stage"] == stage