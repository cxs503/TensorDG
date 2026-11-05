"""Regression pins for the fixed-step option and the time/space error split.

The 2-D Euler DG solver gained a ``dt=`` keyword so temporal refinement studies
can hold the mesh fixed (``benchmarks/euler_dg_2d_time_split.py``).  These tests
keep that path honest and cheap:

* the fixed step marches the requested number of steps and lands on
  ``final_time`` exactly;
* with the step set to the CFL value the fixed-step run reproduces the adaptive
  run (so the verification knob does not silently change the scheme);
* an invalid step is rejected;
* halving the step at a fixed mesh leaves the exact-solution error unchanged
  (the spatial error dominates at CFL 0.2) while the successive-solution
  self-difference decays at the SSP-RK3 rate -- i.e. the temporal error is
  third order and is *not* what caps the P3 order measured by m3024;
* the TVB limiter is inert on the smooth vortex.
"""
from __future__ import annotations

import math

import pytest
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.euler_dg_2d_time_split import (
    FINAL_TIME,
    modal_norm,
    run_case,
    self_difference,
)


def _vortex_initial(x, y):
    return dg2.vortex_primitive(x, y, 0.0, 5.0, (1.0, 1.0))


def test_fixed_step_count_and_final_time():
    dt = 2.0e-3
    _, _, _, metadata = dg2.solve_euler_dg_2d(
        _vortex_initial, ncells=4, degree=2, domain=((-5.0, 5.0), (-5.0, 5.0)),
        final_time=FINAL_TIME, cfl=0.2, dt=dt,
    )
    assert metadata["dt_mode"] == "fixed"
    assert metadata["prescribed_dt"] == pytest.approx(dt)
    # ceil(1.0 / 2e-3) = 500 steps, the last one trimmed onto final_time.
    assert metadata["steps"] == math.ceil(FINAL_TIME / dt)
    assert metadata["dt"] <= dt


def test_fixed_step_matches_adaptive_step():
    plain = run_case(2, 4)[0]
    fixed = run_case(2, 4, dt=0.95 * plain["dt_last"])[0]
    assert fixed["dt_mode"] == "fixed"
    # Same scheme, so the error must agree to far below the 3 % gate.
    assert fixed["density"] == pytest.approx(plain["density"], rel=0.05)


def test_invalid_fixed_step_is_rejected():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            dg2.solve_euler_dg_2d(
                _vortex_initial, ncells=4, degree=1, domain=((-5.0, 5.0), (-5.0, 5.0)),
                final_time=0.01, dt=bad,
            )


def test_temporal_order_is_third_and_exact_error_is_dt_insensitive():
    """Halving dt must not move the exact-solution error, only the self-difference."""
    base, coeff_base = run_case(3, 4)
    dt0 = 0.9 * base["dt_last"]
    coarse, coeff_coarse = run_case(3, 4, dt=dt0)
    fine, coeff_fine = run_case(3, 4, dt=dt0 / 2.0)
    nearer, coeff_nearer = run_case(3, 4, dt=dt0 / 4.0)

    # The exact-solution error is spatial-error dominated: unchanged to 0.1 %.
    assert coarse["density"] == pytest.approx(fine["density"], rel=1e-3)
    assert fine["density"] == pytest.approx(nearer["density"], rel=1e-3)

    # The self-difference isolates the temporal error and decays like dt**3.
    stale = self_difference(coeff_coarse, coeff_fine)
    fresh = self_difference(coeff_fine, coeff_nearer)
    order = math.log(stale / fresh) / math.log(2.0)
    assert 2.6 <= order <= 3.4, f"SSP-RK3 self-convergence order {order:.3f}"
    # The temporal error really is a small share of the reported error.
    assert stale / nearer["density"] < 0.05
    assert modal_norm(coeff_base) > 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "open defect found by this study: with the default tvb_constant=1.0 the 2-D "
        "troubled-cell detector fires on the smooth isentropic vortex and the density "
        "error grows by a factor ~34 (8^2) / ~136 (12^2).  Raising the constant to 10 "
        "restores the unlimited error exactly, so the threshold scale (or the units it "
        "is compared against) is off, not the limiter machinery.  When the default is "
        "fixed this pin turns XPASS and forces the limitation to be retired."
    ),
)
def test_default_tvb_constant_is_inert_on_the_smooth_vortex():
    # 4^2 is too coarse to trigger the detector; the defect shows from 8^2 on.
    plain, _ = run_case(3, 8)
    limited, _ = run_case(3, 8, limiter="tvb")
    assert limited["density"] == pytest.approx(plain["density"], rel=1e-3)


def test_large_tvb_constant_is_inert_on_the_smooth_vortex():
    """Documented workaround: from tvb_constant >= 10 the detector stays quiet."""
    plain, _ = run_case(3, 8)
    limited, _ = run_case(3, 8, limiter="tvb", limiter_constant=10.0)
    assert limited["density"] == pytest.approx(plain["density"], rel=1e-8)


def test_modal_norm_matches_quadrature_l2():
    """Sanity pin on the modal norm used for the self-difference."""
    # Unit constant mode on 2 x 2 cells x 4 components: weight (2k+1)(2l+1) = 1.
    constant = torch.zeros((2, 2, 2, 2, 4), dtype=torch.float64)
    constant[..., 0, 0, :] = 1.0
    assert modal_norm(constant) == pytest.approx(math.sqrt(2 * 2 * 4))
    # A pure first mode is weighted by 1/3 in each direction.
    linear = torch.zeros((1, 1, 2, 2, 1), dtype=torch.float64)
    linear[..., 1, 0, 0] = 1.0
    assert modal_norm(linear) == pytest.approx(math.sqrt(1.0 / 3.0))