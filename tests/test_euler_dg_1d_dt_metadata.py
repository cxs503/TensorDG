"""Regression guard on the time-step metadata of ``solve_euler_dg_1d``.

Defect this pins down (found while re-adjudicating ``CASE-LAX-DG``): the
metadata key ``dt`` was the **last** step of the SSP-RK3 loop, i.e. the step
truncated by ``final_time - time``.  That remnant is neither the step size of
the run nor monotone in ``cfl``: in the committed Lax artifact it *grew*
(1.48e-5 -> 3.54e-5) while the CFL divisor was doubled, which is exactly the
opposite of what a dt-refinement study needs.  Any doc that prints
``meta["dt"]`` as "the step" is therefore quoting a numerical artefact.

The fix is additive -- ``dt`` keeps its old value, ``dt_last`` repeats it under
an honest name, and ``dt_mean = final_time / steps`` is the representative
step -- so this test asserts both the new invariants and the fact that the old
field is *not* a substitute for them.
"""
from __future__ import annotations

import pytest

from tensordg.euler_dg_1d import solve_euler_dg_1d

LEFT = (0.445, 0.698, 3.528)
RIGHT = (0.5, 0.0, 0.571)
FINAL_TIME = 0.05


def _run(cfl=0.35, ncells=20, degree=1, final_time=FINAL_TIME):
    _, _, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=ncells, degree=degree,
                                   final_time=final_time, cfl=cfl)
    return meta


@pytest.mark.parametrize("cfl", [0.35, 0.175])
def test_dt_mean_is_the_actual_mean_step(cfl):
    meta = _run(cfl=cfl)
    assert meta["steps"] > 0
    assert meta["dt_mean"] == pytest.approx(FINAL_TIME / meta["steps"], rel=1e-12)
    # the truncated remnant can only be shorter than (or equal to) the mean step
    assert 0.0 < meta["dt_last"] <= meta["dt_mean"]
    assert meta["dt"] == meta["dt_last"], "dt must stay a backwards-compatible alias"


def test_dt_mean_is_monotone_in_cfl_while_dt_is_not():
    """The invariant the dt-refinement study relies on, and the defect it hid."""
    coarse, fine = _run(cfl=0.35), _run(cfl=0.175)
    # doubling the CFL divisor must shrink the representative step (~2x)
    assert fine["dt_mean"] < coarse["dt_mean"]
    assert coarse["dt_mean"] / fine["dt_mean"] == pytest.approx(2.0, rel=0.1)
    # ... and the run really does take about twice as many steps
    assert 1.8 < fine["steps"] / coarse["steps"] < 2.2
    # the old field cannot serve as a proxy: it is strictly shorter than the
    # mean step in at least one of the two runs (time truncation happened)
    assert min(coarse["dt"], fine["dt"]) < 0.9 * min(coarse["dt_mean"],
                                                    fine["dt_mean"])


def test_zero_final_time_is_representable():
    meta = _run(final_time=0.0)
    assert meta["steps"] == 0
    assert meta["dt_mean"] == 0.0 and meta["dt_last"] == 0.0 and meta["dt"] == 0.0


def test_dt_mean_scales_with_the_mesh_and_degree():
    """Sanity: dt ~ cfl * dx / ((2p+1) * a), so refining the mesh shrinks it."""
    fine_mesh = _run(cfl=0.35, ncells=40, degree=1)
    coarse_mesh = _run(cfl=0.35, ncells=20, degree=1)
    assert fine_mesh["dt_mean"] == pytest.approx(0.5 * coarse_mesh["dt_mean"],
                                                 rel=0.15)
    p3 = _run(cfl=0.35, ncells=20, degree=3)
    assert p3["dt_mean"] < coarse_mesh["dt_mean"]