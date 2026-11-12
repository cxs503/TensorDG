"""Gates for the sensor-gated modal artificial viscosity (SVV) filter.

Every gate is built so that it can fail: the sensor law is compared to its
theoretical exponent ``2p``, the inertness gate is *bitwise*, the conservation
gate compares mode 0 bit-for-bit against the input, and the Sod gate requires a
measurable oscillation reduction relative to the unlimited scheme (which is run
in the same test).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.artificial_viscosity_verification import project, sensor_law
from tensordg.artificial_viscosity import (
    apply_av_filter, av_sensor, av_strength, exponential_filter_factor, modal_smoothness,
    sensor_reference,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.limiters import apply_limiter


def sine_coeff(ncells=40, degree=2):
    return project(lambda x: np.sin(2 * np.pi * x), ncells, degree)


def jump_coeff(ncells=40, degree=2):
    return project(lambda x: (x > 0.5 + 0.37 / ncells).astype(float), ncells, degree)


def test_mode_zero_is_conserved_bitwise_under_full_strength():
    coeff = sine_coeff()
    filtered = apply_av_filter(coeff, 1.0 / 40, s0=-50.0, kappa=1.0, alpha_max=200.0)
    assert torch.equal(filtered[:, 0], coeff[:, 0])
    assert not torch.equal(filtered[:, 1:], coeff[:, 1:])


def test_filter_is_bitwise_inert_below_threshold():
    coeff = sine_coeff()
    level = sensor_reference(coeff, smooth=0)
    filtered = apply_av_filter(coeff, 1.0 / 40, s0=level + 0.5, kappa=0.5)
    assert torch.equal(filtered, coeff)


def test_sensor_separates_smooth_data_from_a_jump():
    smooth = av_sensor(sine_coeff(), smooth=0).max().item()
    jump = av_sensor(jump_coeff(), smooth=0).max().item()
    assert jump - smooth > 2.0, (smooth, jump)


def test_sensor_scaling_law_is_reproduced():
    """Measured law vs theory, per normalisation (see docs/CASE-DG-AV-SHOCK-CAPTURE.md).

    Smooth branch must decay at ``2p`` per decade.  The jump branch is asserted
    on the *range* of the sensor (measured -1.51 at p = 2), not on its slope:
    the slope of that branch is measured and reported by the benchmark
    (``metrics.json -> sensor_law``) rather than gated, because a jump
    deliberately placed inside one element is quadrature sensitive.
    """
    for normalisation in ("fraction", "global"):
        law = sensor_law(2, cells=(20, 40, 80), normalisation=normalisation)
        assert abs(law["smooth_slope"] - 4.0) < 1.5, (normalisation, law["smooth"])
        assert all(-3.0 < value < 0.0 for value in law["jump"]), (normalisation, law["jump"])
    # p = 1: the local (fraction) form is dominated by the element at the zero
    # crossing; the global form recovers the theoretical 2.
    local = sensor_law(1, cells=(20, 40, 80), normalisation="fraction")
    global_ = sensor_law(1, cells=(20, 40, 80), normalisation="global")
    assert abs(global_["smooth_slope"] - 2.0) < 1.0, global_["smooth"]
    assert abs(local["smooth_slope"]) < abs(global_["smooth_slope"]), (local["smooth"], global_["smooth"])


def test_sensor_does_not_fire_on_a_variable_that_is_numerically_zero():
    # rho = const, u = 0: momentum is roundoff only -> the sensor must stay quiet
    coeff = torch.zeros(8, 3, 3, dtype=torch.float64)
    coeff[:, 0, 0] = 1.0
    coeff[:, 0, 2] = 2.5
    coeff[:, 2, 1] = 1e-17
    assert av_sensor(coeff, smooth=0).max().item() < -10.0


def test_strength_ramp_is_monotone_and_saturates():
    s = torch.linspace(-8.0, 0.0, 41, dtype=torch.float64)
    s0, kappa = -2.0, 0.5
    alpha = av_strength(s, s0=s0, kappa=kappa, alpha_max=20.0)
    assert alpha.min().item() == 0.0 and alpha.max().item() == 20.0
    assert torch.all(alpha[1:] >= alpha[:-1])
    assert alpha[(s < s0 - kappa)].max().item() == 0.0
    assert alpha[(s > s0 + kappa)].min().item() == 20.0


def test_filter_factor_damps_only_higher_modes():
    alpha = torch.tensor([5.0, 0.0], dtype=torch.float64)
    sigma = exponential_filter_factor(3, alpha, order=2.0)
    assert sigma[0, 0].item() == 1.0 and sigma[1, 0].item() == 1.0
    assert sigma[0, 2].item() < sigma[0, 1].item() < 1.0


def test_dispatch_through_apply_limiter_matches_the_module():
    coeff = jump_coeff()
    a = apply_limiter(coeff, 0.025, kind="av", s0=-2.0, kappa=0.5, alpha_max=20.0)
    b = apply_av_filter(coeff, 0.025, s0=-2.0, kappa=0.5, alpha_max=20.0)
    assert torch.equal(a, b)


def test_p0_state_is_a_no_op():
    coeff = torch.zeros(6, 1, 3, dtype=torch.float64)
    coeff[:, 0] = torch.tensor([1.0, 0.0, 2.5])
    assert torch.equal(apply_av_filter(coeff, 1.0 / 6), coeff)
    with pytest.raises(ValueError):
        modal_smoothness(coeff)


def test_diagnostics_report_the_flagged_cells():
    coeff = jump_coeff(40)
    _, info = apply_av_filter(coeff, 0.025, s0=-2.0, kappa=0.5, diagnostics=True)
    assert info["flagged"] >= 1
    assert info["sensor_max"] > -2.5


def test_sod_av_reduces_total_variation_relative_to_unlimited():
    """Same solver, same mesh: the AV closure must cut the oscillation budget."""
    common = dict(ncells=60, degree=2, final_time=0.2, cfl=0.12)

    def metrics(limiter, kwargs=None):
        x, coeff, meta = solve_euler_dg_1d(
            (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), limiter=limiter, limiter_kwargs=kwargs, **common,
        )
        avg = coeff[:, 0].numpy()
        rho = avg[:, 0]
        pressure = 0.4 * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / rho)
        return float(np.sum(np.abs(np.diff(rho)))), float(pressure.min()), meta

    tv_none, p_none, _ = metrics(None)
    tv_av, p_av, meta = metrics("av", {"s0": -2.0, "kappa": 0.5, "alpha_max": 20.0})
    assert p_av > 0.0 and p_none > 0.0
    assert tv_av < tv_none, (tv_av, tv_none)
    assert meta["limiter_kwargs"]["s0"] == -2.0
