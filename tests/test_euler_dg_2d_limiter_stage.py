"""Regression tests for the ``limiter_stage`` parameter of ``solve_euler_dg_2d``.

The 1-D solver gained ``limiter_stage`` in mission m2500; the 2-D solver was
missing it, so it always used ``"every"`` (limiting at every RK sub-step), which
the limiter-policy study proved destroys design order on smooth data.  This file
guards the fix:

* ``test_limiter_stage_is_validated`` -- invalid values are rejected;
* ``test_limiter_stage_inert_without_limiter`` -- ``"every"`` and ``"final"``
  produce identical results when no limiter is active;
* ``test_limiter_stage_preserves_design_order_on_vortex`` -- with
  ``limiter_stage="final"`` the P2 vortex converges at order > 2.7, while
  ``"every"`` degrades it (the historic defect).
"""
import math

import numpy as np
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
    """With limiter_stage='final' the P2 vortex converges at order > 2.7.

    The historic 'every' policy degrades the order because it re-limits smooth
    cells at every RK sub-step.  This test verifies that the new 'final' option
    restores the design order.
    """
    errors_final = []
    for cells in (8, 16):
        err, meta = _vortex(cells, 2, final_time=1.0, limiter="tvb",
                            limiter_stage="final", tvb_constant=10.0)
        errors_final.append(err["density"])
        assert meta["steps"] > 0
        assert meta["limiter_stage"] == "final"

    eoc_final = math.log(errors_final[0] / errors_final[1]) / math.log(2.0)
    assert eoc_final > 2.7, (
        f"P2 vortex with limiter_stage='final': EOC={eoc_final:.3f} < 2.7"
    )

    # Verify that 'every' still degrades the order (the historic defect)
    errors_every = []
    for cells in (8, 16):
        err, _ = _vortex(cells, 2, final_time=1.0, limiter="tvb",
                         limiter_stage="every", tvb_constant=10.0)
        errors_every.append(err["density"])
    eoc_every = math.log(errors_every[0] / errors_every[1]) / math.log(2.0)
    # 'every' should have a lower EOC than 'final' on this smooth problem
    assert eoc_every < eoc_final, (
        f"'every' EOC ({eoc_every:.3f}) should be < 'final' EOC ({eoc_final:.3f})"
    )


def test_limiter_stage_recorded_in_metadata():
    """The metadata must record the limiter_stage value."""
    for stage in ("every", "final"):
        _, _, _, meta = dg2.solve_euler_dg_2d_vortex(
            ncells=4, degree=1, final_time=0.01, limiter="tvb",
            limiter_stage=stage,
        )
        assert meta["limiter_stage"] == stage
