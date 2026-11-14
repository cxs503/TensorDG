"""Contract tests for CASE-DG-BURGERS-STEEPEN (steepening gradient peak).

The case exists to measure a quantity no existing gate looked at -- the *peak*
of ``|du/dx|`` near breaking -- so the tests pin the three things that make the
comparison meaningful:

1. the analytic reference (2)/(3) is right and is not a tautology;
2. the reference inverts characteristics where the production 40-step Newton
   does not (documented limitation, not a silent replacement);
3. the DG run reproduces the peak inside the declared 5 % band at the declared
   level, and the band *can* reject a coarser run (non-vacuity).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordg.cases import burgers_steepening as case


TIMES = (0.0, 0.05, 0.10, 0.15, 0.19)


def test_breaking_time_and_initial_peak_come_from_the_data():
    assert case.SLOPE_MAX == pytest.approx(2.0 * math.pi * case.AMPLITUDE, rel=1e-15)
    assert case.BREAKING_TIME == pytest.approx(1.0 / case.SLOPE_MAX, rel=1e-15)
    # G(0) is the initial gradient peak by definition
    assert case.exact_peak_gradient(0.0) == pytest.approx(case.SLOPE_MAX, rel=1e-15)


@pytest.mark.parametrize("t", TIMES)
def test_closed_form_agrees_with_numerically_maximised_gradient(t):
    """(3) vs a dense maximisation of (2): two independent code paths."""
    closed = case.SLOPE_MAX / (1.0 - t * case.SLOPE_MAX) if t > 0 else case.SLOPE_MAX
    sampled, x_peak = case.sampled_peak_gradient(t, samples=40001)
    assert sampled == pytest.approx(closed, rel=2e-4)
    # the peak of the negative-slope branch sits where u_0'(xi) is most negative:
    # the characteristic foot is a half-integer (xi = 1/2 mod 1)
    if t > 0.0:  # at t = 0 both slope branches are maximal
        foot = float(case.characteristic_foot(np.asarray([x_peak]), t)[0])
        distance = abs((foot - 0.5) - round(foot - 0.5))
        assert distance < 3e-3


@pytest.mark.parametrize("t", TIMES[1:])
def test_characteristic_foot_inverts_the_implicit_relation(t):
    x = np.linspace(0.0, 1.0, 97)
    xi = case.characteristic_foot(x, t)
    residual = xi - x + t * case.initial_profile_np(xi)
    assert np.max(np.abs(residual)) < 1e-12


def test_reference_stays_valid_where_the_production_newton_gives_up():
    """``burgers_1d.exact_profile`` caps at 40 Newton steps and raises near breaking.

    ``t = 0.195`` is still smooth (``t / t_b = 0.980``), yet the capped iteration
    aborts.  The bracketed reference must still satisfy the characteristic
    relation to machine precision there.
    """
    t = 0.195
    assert t < case.BREAKING_TIME
    # x in [0.5025, 0.517] is where the capped iteration stalls at this time
    x = np.linspace(0.45, 0.55, 101)
    xi = case.characteristic_foot(x, t)
    assert np.all(np.isfinite(xi))
    residual = xi - x + t * case.initial_profile_np(xi)
    assert np.max(np.abs(residual)) < 1e-12
    # and the closed form keeps growing monotonically towards t_b
    assert case.exact_peak_gradient(t) > case.exact_peak_gradient(case.FINAL_TIME)
    with pytest.raises(RuntimeError):
        import torch

        from tensordg.burgers_1d import exact_profile

        exact_profile(torch.as_tensor(x), t)


def test_reference_and_production_helper_agree_in_the_smooth_regime():
    import torch

    from tensordg.burgers_1d import exact_profile

    x = np.linspace(0.0, 1.0, 257)
    mine = case.exact_solution(x, 0.10)
    theirs = exact_profile(torch.as_tensor(x), 0.10).numpy()
    assert np.max(np.abs(mine - theirs)) < 1e-12


def test_gradient_field_matches_a_finite_difference_of_the_exact_solution():
    t = 0.15
    x = np.linspace(0.1, 0.9, 401)
    dx = 1e-6
    fd = (case.exact_solution(x + dx, t) - case.exact_solution(x - dx, t)) / (2.0 * dx)
    analytic = case.exact_gradient(x, t)
    assert np.max(np.abs(fd - analytic)) < 1e-6 * np.max(np.abs(analytic))


def test_beyond_breaking_the_reference_refuses_to_answer():
    with pytest.raises(ValueError):
        case.characteristic_foot(np.asarray([0.5]), case.BREAKING_TIME)
    with pytest.raises(ValueError):
        case.exact_peak_gradient(case.BREAKING_TIME)


def test_documented_run_reproduces_the_peak_within_the_five_percent_band():
    """Acceptance: the documented level N=256, p=3 at T=0.19 is inside 5 %."""
    row = case.solve_case(cells=256, degree=3, final_time=case.FINAL_TIME)
    assert abs(row["relative_peak_error"]) < case.TOLERANCE, row
    # the L2 error of the whole field is far smaller -- that is the point of the case
    assert row["relative_l2_error"] < 0.1 * case.TOLERANCE
    # peak located where the analytic solution puts it (sub-cell agreement)
    assert abs(row["peak_x_numeric_polynomial"] - row["peak_x_reference"]) < 0.01
    assert abs(row["mean_conservation_error"]) < 1e-12


def test_the_band_rejects_a_coarse_run_so_the_gate_is_not_vacuous():
    row = case.solve_case(cells=64, degree=3, final_time=case.FINAL_TIME)
    assert abs(row["relative_peak_error"]) > case.TOLERANCE, row


def test_peak_measurement_protocol_matters_and_is_documented():
    """The naive quadrature-point reading is biased on the *reference* field.

    Recording the bias keeps a later reader from mistaking a protocol artefact
    for scheme error; both readings are stored in the artifact.
    """
    row = case.solve_case(cells=64, degree=3, final_time=case.FINAL_TIME)
    bias = row["peak_reference_at_quadrature"] / row["peak_reference_closed"] - 1.0
    assert bias < -0.05  # a point-sample max under-reads a sharp extremum
    assert row["relative_peak_error"] > row["relative_peak_error_quadrature_protocol"] - 1e-12


def test_steepening_law_is_monotone_and_diverges_at_breaking():
    peaks = [case.exact_peak_gradient(t) for t in case.TIME_LADDER]
    assert all(b > a for a, b in zip(peaks, peaks[1:]))
    near = case.exact_peak_gradient(case.BREAKING_TIME * (1.0 - 1e-6))
    assert near > 1e5