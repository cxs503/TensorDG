import math

import pytest
import torch

from tensordg import error_2d, solve_advection_2d


def test_mass_conservation_and_negative_velocity():
    x, y, coeff, meta = solve_advection_2d(
        ncells=12, degree=1, final_time=0.1, velocity=(-0.7, -0.4),
    )
    assert torch.isfinite(coeff).all()
    assert abs(coeff[:, :, 0, 0].mean().item() - 1.0) < 1e-12
    assert error_2d(x, y, coeff, 0.1, (-0.7, -0.4))["relative_l2"] < 0.03
    assert meta["steps"] > 0


def test_degree_one_converges_at_second_order():
    errors = []
    for cells in (8, 16):
        x, y, coeff, _ = solve_advection_2d(
            ncells=cells, degree=1, final_time=0.2,
        )
        errors.append(error_2d(x, y, coeff, 0.2)["relative_l2"])
    rate = math.log2(errors[0] / errors[1])
    assert rate > 1.8, (errors, rate)


def test_documented_two_dimensional_case_is_below_three_percent():
    x, y, coeff, _ = solve_advection_2d(
        ncells=24, degree=1, final_time=1.0,
    )
    metrics = error_2d(x, y, coeff, 1.0)
    assert metrics["relative_l2"] < 0.03, metrics


def test_oblique_custom_wave_on_rectangular_mesh(record_property):
    def wave(x, y):
        return .4 + torch.cos(2 * math.pi * (x + 2*y))

    errors = []
    for nx, ny in ((6, 10), (12, 20), (24, 40)):
        vel = (-.8, .3)
        x, y, coefficients, _ = solve_advection_2d((nx, ny), 2, .08, vel, initial=wave)
        error = error_2d(x, y, coefficients, .08, vel, initial=wave)["absolute_l2"]
        errors.append(error)
        record_property(f"mesh_{nx}x{ny}_absolute_l2", error)
        assert abs(coefficients[:, :, 0, 0].mean().item() - .4) < 1e-13
    assert all(math.log2(coarse/fine) > 2.7 for coarse, fine in zip(errors, errors[1:]))


def test_long_periodic_translation_preserves_projected_piecewise_bilinear_wave():
    def profile(x, y):
        assert torch.all((x >= 0) & (x < 1) & (y >= 0) & (y < 1))
        return x*y

    x, y, coeff, _ = solve_advection_2d((8, 12), 1, 0., initial=profile)
    shifted = coeff.roll((2, -3), dims=(0, 1))
    assert error_2d(x, y, shifted, 2**40+.25, (1., -1.), initial=profile)["absolute_l2"] < 1e-14


def test_scalar_zero_profile_and_relative_norm():
    zero = lambda x, y: 0.
    x, y, coeff, _ = solve_advection_2d((4, 6), 1, .1, initial=zero)
    assert error_2d(x, y, coeff, .1, initial=zero)["relative_l2"] == 0.
    coeff[:, :, 0, 0] = 1.
    assert math.isinf(error_2d(x, y, coeff, .1, initial=zero)["relative_l2"])


def test_translation_on_physical_rectangle(record_property):
    lengths, velocity, t = (3., .5), (-.75, .25), .125

    def profile(x, y):
        return 1 + torch.sin(2*math.pi*x/3)*torch.sin(4*math.pi*y)

    errors = []
    for n in (6, 12, 24):
        x, y, coeff, meta = solve_advection_2d((n, n+2), 2, t, velocity, lengths=lengths)
        assert meta["lengths"] == lengths
        assert abs(x[-1].item() - (3.-1.5/n)) < 1e-14
        assert abs(y[0].item() - .25/(n+2)) < 1e-14
        mass_error = abs(coeff[..., 0, 0].mean().item()*1.5 - 1.5)
        assert mass_error < 1e-13
        metrics = error_2d(x, y, coeff, t, velocity, initial=profile, lengths=lengths)
        assert abs(metrics["exact_l2"] - math.sqrt(1.5*1.25)) < 1e-13
        errors.append(metrics["absolute_l2"])
        record_property(f"n{n}_absolute_l2", errors[-1])
        record_property(f"n{n}_mass_error", mass_error)
    assert all(math.log2(a/b) > 2.5 for a,b in zip(errors, errors[1:]))


def test_rectangular_exact_translation_crosses_both_seams():
    def profile(x, y):
        assert torch.all((x >= 0) & (x < 2) & (y >= 0) & (y < .5))
        return x*y

    lengths = (2., .5)
    x, y, coeff, _ = solve_advection_2d((8, 8), 1, 0., lengths=lengths, initial=profile)
    rolled = coeff.roll((1, -2), dims=(0, 1))
    metrics = error_2d(x, y, rolled, 2**40+.25, (1., -.5), lengths=lengths, initial=profile)
    assert metrics["absolute_l2"] < 1e-14


def test_modal_restart_matches_uninterrupted_translation(record_property):
    opts = dict(ncells=(8, 12), lengths=(2., 3.), degree=2, velocity=(-.75, .25), cfl=.15625)
    t0 = 2.**30
    x, y, full, info = solve_advection_2d(start_time=t0, final_time=t0+.125, **opts)
    _, _, first, early = solve_advection_2d(start_time=t0, final_time=t0+.0625, **opts)
    saved = first.clone()
    _, _, restarted, late = solve_advection_2d(start_time=early["final_time"], final_time=t0+.125,
                                                initial=first, **opts)
    assert info["steps"] == early["steps"] + late["steps"] == 16
    assert late["dt"] == info["dt"] == 1/128
    assert torch.equal(first, saved)
    mismatch = float((restarted-full).abs().max())
    mass_error = abs(float(restarted[..., 0, 0].mean())*6.-6.)
    l2 = error_2d(x, y, restarted, .125, opts["velocity"], lengths=opts["lengths"])["relative_l2"]
    record_property("max_modal_difference", mismatch)
    record_property("mass_error", mass_error)
    record_property("relative_l2_vs_translation", l2)
    assert mismatch == 0.
    assert mass_error < 1e-13
    assert l2 < .005


def test_zero_duration_restart_copies_and_converts_modes():
    initial = torch.arange(24., dtype=torch.float32).reshape(3, 2, 2, 2).transpose(0, 1)
    _, _, result, info = solve_advection_2d((2, 3), 1, 4., start_time=4., initial=initial)
    assert info["steps"] == 0
    assert result.dtype == torch.float64
    assert torch.equal(result, initial.double())
    _, _, copied, _ = solve_advection_2d((2, 3), 1, 4., start_time=4., initial=result)
    assert copied.data_ptr() != result.data_ptr()

@pytest.mark.parametrize("method", ["euler", "rk2", "rk3", "ssp_rk3", "rk4", "ssp_rk54", "rk5"])
def test_rectangular_rk_fourier_amplification(method, record_property):
    import cmath
    from tensordg.rk_stability import stability_function_dense
    from tensordg.runge_kutta import get_tableau

    nx, ny = 8, 12
    lengths, velocity = (2., 3.), (-1., .5)
    tx, ty = 2*math.pi/nx, 4*math.pi/ny
    phase = tx*torch.arange(nx, dtype=torch.float64)[:, None] + ty*torch.arange(ny, dtype=torch.float64)
    initial = (1 + phase.cos())[:, :, None, None]
    dt = .0625
    _, _, result, meta = solve_advection_2d((nx, ny), 0, dt, velocity, cfl=.5,
            initial=initial, lengths=lengths, time_integrator=method)
    # For P0 the upwind stencil samples i+1 and j-1 on this velocity field.
    eigenvalue = (nx/lengths[0])*(cmath.exp(1j*tx)-1) + (.5*ny/lengths[1])*(cmath.exp(-1j*ty)-1)
    tableau = get_tableau(method)
    gain = stability_function_dense(tableau, dt*eigenvalue)[0]
    expected = 1 + gain.real*phase.cos() - gain.imag*phase.sin()
    error = float((result[..., 0, 0]-expected).abs().max())
    mass_error = abs(float(result.mean())*6.-6.)
    record_property("max_cell_average_error", error)
    record_property("mass_error", mass_error)
    assert meta["steps"] == 1 and meta["time_integrator"] == method
    assert (meta["rk_order"], meta["rk_stages"]) == (tableau.order, tableau.stages)
    assert error < 2e-15
    assert mass_error < 2e-15


def test_time_varying_velocity_against_integrated_fourier_mode(record_property):
    import cmath

    nx, ny = 8, 12
    tx, ty = 2*math.pi/nx, 4*math.pi/ny
    phase = tx*torch.arange(nx, dtype=torch.float64)[:, None] + ty*torch.arange(ny, dtype=torch.float64)
    initial = (1 + phase.cos())[:, :, None, None]
    start, span = 2., .25

    def velocity(t):
        return -(1 + (t-start)), .5 + (t-start)**2

    # The two P0 upwind operators commute. Integrate each signed speed
    # exactly before exponentiating the combined Fourier eigenvalue.
    distance_x = span + span**2/2
    distance_y = .5*span + span**3/3
    gain = cmath.exp(4*distance_x*(cmath.exp(1j*tx)-1) + 4*distance_y*(cmath.exp(-1j*ty)-1))
    exact = 1 + gain.real*phase.cos() - gain.imag*phase.sin()
    errors = []
    for steps in (4, 8, 16):
        _, _, coeff, info = solve_advection_2d((nx, ny), 0, start+span, velocity,
                lengths=(2., 3.), initial=initial, start_time=start, cfl=1., max_dt=span/steps)
        assert info["steps"] == steps
        errors.append(float((coeff[..., 0, 0]-exact).abs().max()))
        mass_error = abs(float(coeff.mean())*6.-6.)
        record_property(f"max_cell_error_{steps}", errors[-1])
        record_property(f"mass_error_{steps}", mass_error)
        assert mass_error < 2e-15
    assert min(math.log2(a/b) for a,b in zip(errors, errors[1:])) > 3.8


def test_velocity_switch_matches_two_constant_translations(record_property):
    start, switch, end = 3., 3.0625, 3.1875
    first, second = (.5, -.25), (-.25, .5)
    opts = dict(ncells=(8, 12), degree=2, lengths=(2., 3.), max_speed=(.5, .5), max_dt=1/64, cfl=.5)
    x, y, switched, _ = solve_advection_2d(final_time=end, start_time=start,
        velocity=lambda t: first if t < switch else second, velocity_jump_times=(switch,), **opts)
    _, _, checkpoint, _ = solve_advection_2d(start_time=start, final_time=switch, velocity=first, **opts)
    _, _, restarted, _ = solve_advection_2d(start_time=switch, final_time=end,
                                            velocity=second, initial=checkpoint, **opts)
    difference = float((switched-restarted).abs().max())
    displacement = tuple((switch-start)*a+(end-switch)*b for a,b in zip(first, second))
    relative = error_2d(x, y, switched, 1., displacement, lengths=opts["lengths"])["relative_l2"]
    record_property("max_split_modal_difference", difference)
    record_property("relative_l2_vs_translation", relative)
    assert difference < 5e-15
    assert relative < .005


@pytest.mark.parametrize("velocity", [(-.75, .5), (.75, -.5), (0., -.5)])
def test_modal_face_rates_close_each_cell(velocity, record_property):
    import numpy as np
    from numpy.polynomial.legendre import leggauss, legval2d
    from tensordg import advection_face_fluxes_2d

    nx, ny, degree = 5, 8, 3
    lengths = (2.5, 2.)
    hx, hy = lengths[0]/nx, lengths[1]/ny
    modes = torch.randn(nx, ny, degree+1, degree+1, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(42))
    fx, fy = advection_face_fluxes_2d(modes, velocity, lengths=lengths)
    nodes, weights = leggauss(5)
    ax, ay = velocity
    end_x = np.full_like(nodes, 1. if ax >= 0 else -1.)
    end_y = np.full_like(nodes, 1. if ay >= 0 else -1.)
    expected_x, expected_y = np.empty((nx, ny)), np.empty((nx, ny))
    for i, j in np.ndindex(nx, ny):
        donor_x = modes[(i + (ax < 0)) % nx, j].numpy()
        donor_y = modes[i, (j + (ay < 0)) % ny].numpy()
        expected_x[i,j] = .5*hy*ax*(weights @ legval2d(end_x, nodes, donor_x))
        expected_y[i,j] = .5*hx*ay*(weights @ legval2d(nodes, end_y, donor_y))
    face_error = max(np.max(abs(fx.numpy()-expected_x)), np.max(abs(fy.numpy()-expected_y)))

    dt = 1/512
    _, _, advanced, info = solve_advection_2d((nx, ny), degree, dt, velocity, initial=modes,
                                               lengths=lengths, time_integrator="euler")
    assert info["steps"] == 1
    outward = fx-fx.roll(1, 0) + fy-fy.roll(1, 1)
    residual = (advanced[...,0,0]-modes[...,0,0])*hx*hy + dt*outward
    balance_error = float(residual.abs().max())
    record_property("max_face_quadrature_error", face_error)
    record_property("max_cell_mass_balance_error", balance_error)
    assert face_error < 3e-15
    assert balance_error < 1e-16


@pytest.mark.parametrize("method", ["euler", "rk2", "rk3", "ssp_rk3", "rk4", "ssp_rk54", "rk5"])
def test_integrated_fluxes_close_switched_cell_budget(method, record_property):
    nx, ny = 5, 6
    lengths = (2.5, 1.5)
    hx, hy = lengths[0]/nx, lengths[1]/ny
    end, switch = .125, .04
    velocity = lambda t: (.75, -.25) if t < switch else (-.5, 1.)
    opts = dict(ncells=(nx, ny), degree=2, final_time=end, lengths=lengths, velocity=velocity,
                velocity_jump_times=(switch,), time_integrator=method, return_integrated_fluxes=True)
    initial = torch.randn(nx, ny, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(17))
    _, _, final, _, (fx, fy) = solve_advection_2d(initial=initial, **opts)
    change = (final[..., 0, 0] - initial[..., 0, 0])*hx*hy
    residual = change + fx-fx.roll(1, 0) + fy-fy.roll(1, 1)
    error = float(residual.abs().max())
    uniform = torch.zeros_like(initial)
    uniform[..., 0, 0] = 2.
    _, _, _, _, (fx, fy) = solve_advection_2d(initial=uniform, **opts)
    expected_x = 2*hy*(.75*switch-.5*(end-switch))
    expected_y = 2*hx*(-.25*switch+(end-switch))
    uniform_error = max(float((fx-expected_x).abs().max()), float((fy-expected_y).abs().max()))
    record_property("max_cell_balance_error", error)
    record_property("max_uniform_transport_error", uniform_error)
    assert error < 3e-15
    assert uniform_error < 3e-15


def test_zero_duration_face_transports():
    _, _, _, meta, fluxes = solve_advection_2d((3, 4), 1, 2., start_time=2., return_integrated_fluxes=True)
    assert meta["steps"] == 0
    assert all(q.shape == (3, 4) and torch.count_nonzero(q) == 0 for q in fluxes)


def test_fixed_velocity_rk4_face_transports_add_across_restart(record_property):
    opts = dict(ncells=(4, 6), degree=1, lengths=(2., 3.), velocity=(-.5, .75), max_dt=1/128,
                return_integrated_fluxes=True)
    _, _, full, _, totals = solve_advection_2d(final_time=.125, **opts)
    _, _, middle, _, first = solve_advection_2d(final_time=.0625, **opts)
    _, _, final, _, second = solve_advection_2d(start_time=.0625, final_time=.125, initial=middle, **opts)
    error = max(float((q-(a+b)).abs().max()) for q,a,b in zip(totals, first, second))
    record_property("max_restart_transport_difference", error)
    assert torch.equal(full, final)
    assert error < 2e-15


@pytest.mark.parametrize("velocity", [(-.75, .5), (.75, 0.), (0., -.5)])
def test_modal_energy_and_upwind_power(velocity, record_property):
    import numpy as np
    from numpy.polynomial.legendre import leggauss, legval2d
    from tensordg import energy_2d, upwind_dissipation_2d
    from tensordg.advection_2d import _operators, _rhs

    nx, ny, degree = 5, 6, 3
    lengths = (2.5, 1.5)
    hx, hy = lengths[0]/nx, lengths[1]/ny
    field = torch.randn(nx, ny, 4, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(29))
    _, _, _, volume, left, right = _operators(degree, field.device, field.dtype)
    rhs = _rhs(field, hx, hy, velocity, volume, left, right)
    nodes, weights = leggauss(5)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    energy = power = 0.
    for i, j in np.ndindex(nx, ny):
        u = legval2d(x, y, field[i,j].numpy())
        ut = legval2d(x, y, rhs[i,j].numpy())
        energy += hx*hy/8*(weights @ (u*u) @ weights)
        power += hx*hy/4*(weights @ (u*ut) @ weights)
    computed = energy_2d(field, lengths=lengths)
    rates = upwind_dissipation_2d(field, velocity, lengths=lengths)
    energy_error = abs(computed/energy-1)
    balance_error = abs(1+sum(rates)/power)
    record_property("relative_energy_error", energy_error)
    record_property("relative_power_balance_error", balance_error)
    assert energy_error < 4e-15
    assert balance_error < 1e-14
    assert all(rate == 0. if speed == 0. else rate > 0. for rate, speed in zip(rates, velocity))


def test_upwind_loss_resolves_modes_on_large_uniform_background():
    from tensordg import upwind_dissipation_2d

    field = torch.zeros(3, 4, 2, 2, dtype=torch.float64)
    field[..., 1, 0] = 1.
    field[..., 0, 1] = 2.
    field[..., 0, 0] = 2.**60
    assert upwind_dissipation_2d(field, (1., -1.)) == (6., 32.)
