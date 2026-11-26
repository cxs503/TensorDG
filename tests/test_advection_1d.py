import math
import pytest
import torch
from tensordg import solve_advection, l2_error


def test_solution_is_finite_and_mass_conservative():
    centers, coeff, meta = solve_advection(ncells=32, degree=1, final_time=0.2)
    assert torch.isfinite(coeff).all()
    # Integral over the periodic domain is h * sum(cell-average coefficients).
    mass = coeff[:, 0].sum().item() / 32
    assert abs(mass) < 1e-10
    assert meta['steps'] > 0


def test_degree_one_converges_at_expected_rate():
    errors = []
    for n in (20, 40):
        centers, coeff, _ = solve_advection(ncells=n, degree=1, final_time=0.25)
        errors.append(l2_error(centers, coeff, final_time=0.25))
    rate = math.log(errors[0] / errors[1], 2)
    assert rate > 1.7, (errors, rate)
    # The exact sine wave has L2 norm 1/sqrt(2) on the unit interval.
    assert errors[-1] * math.sqrt(2.0) < 0.03


def test_zero_final_time_returns_initial_projection():
    centers, coeff, meta = solve_advection(ncells=16, degree=2, final_time=0.0)
    assert meta['steps'] == 0
    assert torch.isfinite(coeff).all()

@pytest.mark.parametrize("speed", [-1., 1.])
@pytest.mark.parametrize("final_time", [0.25, 2.25, 2**40 + 0.25])
def test_translated_sawtooth_error_norms(speed, final_time):
    from tensordg.advection_1d import l1_error, linf_error

    # u0(x) = x on [0, 1) has a jump at the periodic seam. This quarter
    # turn shifts the jump exactly two cells, preserving the modal slope.
    x = (torch.arange(8, dtype=torch.float64) + 0.5) / 8
    modes = torch.stack((x, torch.full_like(x, 1/16)), dim=1).roll(int(2*speed), 0)

    def profile(foot):
        assert torch.all((foot >= 0) & (foot < 1))
        return foot

    for norm in (l1_error, l2_error, linf_error):
        assert norm(x, modes, final_time, speed, exact=profile) < 1e-14
        shifted = modes.clone()
        shifted[:, 0] += 0.125
        assert norm(x, shifted, final_time, speed, exact=profile) == pytest.approx(0.125)


@pytest.mark.parametrize("method,limiter", [("rk4", None), ("ssp_rk3", "tvb"), ("rk5", None)])
def test_restart_modal_transport_on_absolute_clock(method, limiter, record_property):
    opts = dict(ncells=64, degree=2, speed=-1., cfl=.15625, time_integrator=method, limiter=limiter)
    start = 2.**34
    span = 1/32
    def profile(x):
        return 1. + torch.sin(2*math.pi*x)
    x, full, info = solve_advection(initial=profile, start_time=start, final_time=start+span, **opts)
    _, middle, first = solve_advection(initial=profile, start_time=start, final_time=start+span/2, **opts)
    saved = middle.clone()
    _, result, second = solve_advection(initial=middle, start_time=first["final_time"],
                                         final_time=start+span, **opts)
    assert info["steps"] == first["steps"] + second["steps"] == 64
    assert first["dt"] == second["dt"] == info["dt"]
    assert torch.equal(middle, saved)
    difference = float((result-full).abs().max())
    error = l2_error(x, result, span, speed=-1., exact=profile)
    mass_error = abs(float(result[:,0].mean())-1.)
    record_property("max_modal_difference", difference)
    record_property("absolute_l2_vs_translation", error)
    record_property("mass_error", mass_error)
    assert difference == 0.
    assert error < .005
    assert mass_error < 1e-14


def test_zero_duration_modal_restart_preserves_all_coefficients():
    initial = torch.arange(24., dtype=torch.float64).reshape(3, 8).T
    _, output, meta = solve_advection(8, 2, 4., initial=initial, start_time=4., limiter="tvb")
    assert meta["steps"] == 0
    assert torch.equal(output, initial)
    assert output.data_ptr() != initial.data_ptr()
