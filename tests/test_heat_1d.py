import math

import torch

from tensordg.heat_1d import heat_error, solve_heat_1d


def test_heat_sipg_p1_converges_at_second_order():
    errors = []
    for cells in (8, 16):
        x, coeff, meta = solve_heat_1d(ncells=cells, degree=1, final_time=0.1)
        assert meta["diffusion_symmetry_error"] < 1e-11
        errors.append(heat_error(x, coeff, 0.1)["relative_l2"])
    rate = math.log2(errors[0] / errors[1])
    assert rate > 1.8, (errors, rate)


def test_documented_heat_case_is_below_three_percent():
    x, coeff, _ = solve_heat_1d(ncells=16, degree=2, final_time=0.1)
    assert torch.isfinite(coeff).all()
    metrics = heat_error(x, coeff, 0.1)
    assert metrics["relative_l2"] < 0.03, metrics


def test_face_flux_divergence_matches_each_sipg_cell_balance():
    from tensordg.heat_1d import assemble_sipg_1d, sipg_heat_face_fluxes

    assembly = assemble_sipg_1d(7, 3, penalty=13.)
    coefficients = torch.arange(28, dtype=torch.float64).cos().reshape(7, 4)
    flux = sipg_heat_face_fluxes(coefficients, penalty=13.)
    # Testing with the constant Legendre mode gives the integrated cell dT/dt.
    integrated_rate = -(assembly["diffusion"] @ coefficients.flatten()).reshape(7, 4)[:, 0]
    torch.testing.assert_close(flux[:-1]-flux[1:], integrated_rate, atol=2e-11, rtol=0.)


def test_heat_face_fluxes_converge_to_fourier_flux():
    from tensordg.heat_1d import solve_heat_1d_exact_time, sipg_heat_face_fluxes

    errors = []
    for n in (8, 16, 32):
        _, coeff, _ = solve_heat_1d_exact_time(ncells=n, degree=2, final_time=.02)
        x = torch.linspace(0., 1., n+1, dtype=coeff.dtype)
        exact = -torch.pi * torch.cos(torch.pi*x) * math.exp(-torch.pi**2*.02)
        errors.append(float((sipg_heat_face_fluxes(coeff)-exact).abs().max()))
    assert all(fine < coarse/3.5 for coarse, fine in zip(errors, errors[1:]))


def test_accumulated_flux_balances_all_cell_stores_and_matches_sine_decay():
    from tensordg.heat_1d import assemble_sipg_1d

    n, end = 16, .02
    initial = assemble_sipg_1d(n, 2)["initial_coefficients"]
    faces = torch.linspace(0., 1., n+1, dtype=initial.dtype)
    exact = math.expm1(-math.pi**2*end) * torch.cos(math.pi*faces) / math.pi
    for backend in ("dense", "sparse", "matrix-free"):
        _, final, info = solve_heat_1d(n, 2, end, 48, backend=backend, integrate_face_fluxes=True)
        flux = initial.new_tensor(info["integrated_face_fluxes"])
        torch.testing.assert_close((final[:, 0]-initial[:, 0])/n, flux[:-1]-flux[1:],
                                   atol=1e-11, rtol=0.)
        torch.testing.assert_close(flux, exact, atol=1e-6, rtol=0.)
        _, _, zero = solve_heat_1d(n, 2, 0., backend=backend, integrate_face_fluxes=True)
        assert zero["integrated_face_fluxes"] == [0.]*(n+1)


def test_asymmetric_modal_heat_transfer_with_custom_profile():
    from tensordg.heat_1d import solve_heat_1d_exact_time

    n, end = 16, .03

    def profile(x):
        return torch.sin(math.pi*x) + .8*torch.sin(2*math.pi*x)

    modes = torch.tensor([math.pi, 2*math.pi], dtype=torch.float64)
    amplitudes = modes.new_tensor([1., .8])
    faces = torch.linspace(0., 1., n+1, dtype=modes.dtype)
    exact = (amplitudes * torch.expm1(-modes**2*end) / modes) @ torch.cos(modes[:, None]*faces)
    _, reference, _ = solve_heat_1d_exact_time(n, 2, end, initial_temperature=profile)
    for backend in ("dense", "sparse", "matrix-free"):
        _, coeff, info = solve_heat_1d(n, 2, end, 192, backend=backend,
                                      integrate_face_fluxes=True, initial_temperature=profile)
        flux = modes.new_tensor(info["integrated_face_fluxes"])
        torch.testing.assert_close(flux, exact, atol=3e-6, rtol=0.)
        torch.testing.assert_close(coeff[:, 0], reference[:, 0], atol=6e-6, rtol=0.)
