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


def test_stationary_quartic_source_profile_preserves_equilibrium_and_wall_flux():
    from tensordg.heat_1d import solve_heat_1d_exact_time

    # u=x-x⁴ has -u''=12x², homogeneous endpoints, and nonzero wall fluxes.
    initial = lambda x: x-x**4
    opts = dict(ncells=5, degree=4, final_time=.013, source=lambda x: 12*x*x,
                initial_temperature=initial)
    for backend in ("dense", "sparse", "matrix-free", "exact-time"):
        if backend == "exact-time":
            centers, coeff, meta = solve_heat_1d_exact_time(**opts, integrate_face_fluxes=True)
        else:
            centers, coeff, meta = solve_heat_1d(**opts, backend=backend, time_steps=5,
                                                 integrate_face_fluxes=True)
        faces = torch.linspace(0.,1.,6,dtype=coeff.dtype)
        heat = torch.tensor(meta["integrated_face_fluxes"],dtype=coeff.dtype)
        torch.testing.assert_close(heat,.013*(4*faces**3-1),atol=5e-10,rtol=0.)
        assert heat_error(centers,coeff,.013,reference=lambda x,t: initial(x))["absolute_l2"] < 2e-11


def test_unequal_walls_and_volume_heating_all_backends():
    from tensordg.heat_1d import solve_heat_1d_exact_time, sipg_heat_face_fluxes

    # T=2-3x+x(1-x), -T''=2, rightward flux=2+2x.
    def profile(x):
        return 2-3*x+x*(1-x)
    walls = (2.,-1.)
    duration = .025
    options = dict(ncells=6,degree=2,final_time=duration,wall_temperatures=walls,
                   initial_temperature=profile,source=2.,integrate_face_fluxes=True)
    for backend in ("dense","sparse","matrix-free","exact-time"):
        if backend == "exact-time":
            centers, coeff, meta = solve_heat_1d_exact_time(**options)
        else:
            centers, coeff, meta = solve_heat_1d(**options,backend=backend,time_steps=8)
        target = 2+2*torch.linspace(0.,1.,7,dtype=coeff.dtype)
        instantaneous = sipg_heat_face_fluxes(coeff,wall_temperatures=walls)
        cumulative = coeff.new_tensor(meta["integrated_face_fluxes"])
        torch.testing.assert_close(instantaneous,target,atol=5e-8,rtol=0.)
        torch.testing.assert_close(cumulative,duration*target,atol=5e-9,rtol=0.)
        assert heat_error(centers,coeff,duration,reference=lambda x,t: profile(x))["absolute_l2"] < 2e-10


def test_ramped_walls_recover_cubic_field_and_integrated_flux():
    from tensordg import solve_heat_1d_ramped_walls

    # u=1-2x+t*(2-3x)+x²-x³/2-x/2 satisfies u_t=u_xx.
    def initial(x):
        return 1-2*x+x*x-.5*x**3-.5*x
    t = .07
    for backend in ("dense","sparse","matrix-free","exact-time"):
        centers,coeff,meta = solve_heat_1d_ramped_walls(5,3,t,wall_temperatures=(1.,-1.),
            wall_rates=(2.,-1.),initial_temperature=initial,backend=backend,
            time_steps=None if backend=="exact-time" else 8,integrate_face_fluxes=True)
        x = torch.linspace(0.,1.,6,dtype=coeff.dtype)
        exact_heat = t*(2.5-2*x+1.5*x**2)+1.5*t*t
        torch.testing.assert_close(coeff.new_tensor(meta["integrated_face_fluxes"]),exact_heat,atol=5e-9,rtol=0.)
        assert heat_error(centers,coeff,t,reference=lambda x,t: initial(x)+t*(2-3*x))["absolute_l2"] < 5e-10


def test_piecewise_wall_history_restarts_modes_and_conserves_interval_heat():
    from tensordg import solve_heat_1d_wall_history

    times = [.2,.23,.24,.28]
    walls = [[.1,.4],[.3,.2],[.3,.2],[.15,.25]]
    for backend in ("dense","sparse","matrix-free","exact-time"):
        x, states, meta = solve_heat_1d_wall_history(times,walls,5,2,backend=backend,
            steps_per_interval=None if backend=="exact-time" else 4,source=lambda x: 1+x,
            initial_temperature=lambda x: .1+.3*x,integrate_face_fluxes=True)
        heat = states.new_tensor(meta["interval_face_heat"])
        dt = states.new_tensor(times).diff()
        residual = states[:,:,0].diff(dim=0)/5+heat.diff(dim=1)-dt[:,None]*(1+x)/5
        assert residual.abs().max() < 5e-10
        torch.testing.assert_close(heat.sum(dim=0),states.new_tensor(meta["integrated_face_fluxes"]))
        torch.testing.assert_close(states[0,:,0],.1+.3*x)
        _,tail,_ = solve_heat_1d_wall_history(times[1:],walls[1:],5,2,backend=backend,
            steps_per_interval=None if backend=="exact-time" else 4,source=lambda x: 1+x,initial_temperature=states[1])
        torch.testing.assert_close(tail,states[1:],atol=5e-11,rtol=0.)


def test_switched_volume_heating_cell_energy_for_every_backend():
    from tensordg import solve_heat_1d_wall_history

    times = [0., .01, .037, .05]
    walls = [[0.,0.], [.01,-.01], [.01,-.01], [.02,.015]]
    for backend in ("dense","sparse","matrix-free","exact-time"):
        x,states,meta = solve_heat_1d_wall_history(times,walls,4,2,backend=backend,
            initial_temperature=torch.zeros_like, interval_sources=[2.,None,lambda x: -3*x],
            steps_per_interval=None if backend=="exact-time" else 6,integrate_face_fluxes=True)
        heat = states.new_tensor(meta["interval_face_heat"])
        deposited = torch.stack([2*torch.ones_like(x),torch.zeros_like(x),-3*x])
        deposited = deposited*states.new_tensor(times).diff()[:,None]/4
        residual = states[:,:,0].diff(dim=0)/4+heat.diff(dim=1)-deposited
        assert residual.abs().max() < 5e-10


def test_harmonic_cubic_field_for_independent_wall_types():
    from itertools import product
    from tensordg import solve_harmonic_heat_1d

    k, c, omega = 2.1, .7, 4.
    def temperature(x):
        return 1+.2j + (.7-.4j)*x + (.2+.6j)*x**3
    def flux(x):
        return -k*((.7-.4j)+3*(.2+.6j)*x*x)
    def source(x):
        return 1j*omega*c*temperature(x)-6*k*(.2+.6j)*x
    for types in product(("dirichlet","neumann"),repeat=2):
        values = [temperature(i) if kind=="dirichlet" else (1-2*i)*flux(i) for i,kind in enumerate(types)]
        x,u,q = solve_harmonic_heat_1d(omega,ncells=6,degree=3,conductivity=k,capacity=c,
                                       source=source,wall_types=types,wall_values=values)
        faces = torch.linspace(0.,1.,7,dtype=x.dtype)
        expected_mean = temperature(x)+(.2+.6j)*x/(4*6**2)
        torch.testing.assert_close(u[:,0],expected_mean,atol=1e-11,rtol=0.)
        torch.testing.assert_close(q,flux(faces),atol=1e-10,rtol=0.)


def test_slow_two_flux_walls_separate_bulk_storage_from_local_flux():
    from tensordg import solve_harmonic_heat_1d

    omega, c, k = 1e-12, 1.5, 2.
    # The spatial mean grows like 1/omega; the zero-mean quadratic keeps
    # heat travelling to the right with a finite slope-independent limit.
    bulk = (.6+.3j)/(1j*omega*c)
    def local(x):
        return (.3-.5j)*(x**2-1/3)+(.2+.4j)*(x-.5)
    def power(x):
        return -k*(2*(.3-.5j)*x+.2+.4j)
    def source(x):
        return (.6+.3j)+1j*omega*c*local(x)-2*k*(.3-.5j)
    x,u,q = solve_harmonic_heat_1d(omega,ncells=16,degree=2,capacity=c,conductivity=k,
        source=source,wall_types=("neumann","neumann"),wall_values=(power(0),-power(1)))
    torch.testing.assert_close(u[:,0].mean()/bulk,u.new_tensor(1.),atol=5e-15,rtol=0.)
    faces = torch.linspace(0.,1.,17,dtype=x.dtype)
    torch.testing.assert_close(q,power(faces),atol=1e-11,rtol=0.)


def test_harmonic_storing_wall_material_flux_and_trace():
    from itertools import product
    from tensordg import solve_harmonic_heat_1d

    omega,c,k = 3.,1.4,2.1
    def temp(x):
        return (.5+.7j)+(1.-.2j)*x**2
    def flux(x):
        return -2*k*(1.-.2j)*x
    def source(x):
        return 1j*omega*c*temp(x)-2*k*(1.-.2j)
    for kinds in product(("dirichlet","neumann","robin"), repeat=2):
        data,films,cw = [],[],[]
        for i,kind in enumerate(kinds):
            cap = 0. if kind=="dirichlet" else (.3,.8)[i]
            film = (2.,5.)[i] if kind=="robin" else 0.
            external = (1-2*i)*flux(i)+1j*omega*cap*temp(i)
            data.append(temp(i)+external/film if film else external if kind=="neumann" else temp(i))
            films.append(film)
            cw.append(cap)
        x,u,q,surface = solve_harmonic_heat_1d(omega,ncells=7,degree=2,capacity=c,conductivity=k,
            wall_types=kinds,wall_values=data,film_coefficients=films,wall_areal_heat_capacity=cw,
            source=source,return_wall_temperatures=True)
        faces = torch.linspace(0.,1.,8,dtype=x.dtype)
        torch.testing.assert_close(surface,temp(faces[[0,-1]]),rtol=0.,atol=1e-11)
        torch.testing.assert_close(q,flux(faces),rtol=0.,atol=1e-10)


def test_slow_flux_driven_walls_include_surface_storage_in_mean_constraint():
    from tensordg import solve_harmonic_heat_1d

    omega,c,k = 1e-12,1.5,2.
    cw = torch.tensor([.3,.9],dtype=torch.float64)
    ends = torch.tensor([0.,1.],dtype=torch.float64)
    rate = .8-.6j
    def local(x):
        return (.5+.3j)*x*x
    def flux(x):
        return -2*k*(.5+.3j)*x
    input_flux = flux(ends)*torch.tensor([1.,-1.])+cw*(rate+1j*omega*local(ends))
    def source(x):
        return c*(rate+1j*omega*local(x))-2*k*(.5+.3j)
    x,u,q,surface = solve_harmonic_heat_1d(omega,ncells=12,degree=2,capacity=c,conductivity=k,
        source=source,wall_types=("neumann","neumann"),wall_values=input_flux,
        wall_areal_heat_capacity=cw,return_wall_temperatures=True)
    torch.testing.assert_close(q,flux(torch.linspace(0.,1.,13,dtype=x.dtype)),atol=2e-11,rtol=0.)
    torch.testing.assert_close(1j*omega*surface,rate+1j*omega*local(ends),atol=1e-14,rtol=0.)
