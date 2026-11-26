import math

import torch
import pytest

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


@pytest.mark.parametrize("wall_penalty", [None, 40.])
def test_piecewise_wall_history_restarts_modes_and_conserves_interval_heat(wall_penalty):
    from tensordg import solve_heat_1d_wall_history

    times = [.2,.23,.24,.28]
    walls = [[.1,.4],[.3,.2],[.3,.2],[.15,.25]]
    for backend in ("dense","sparse","matrix-free","exact-time"):
        x, states, meta = solve_heat_1d_wall_history(times,walls,5,2,backend=backend,wall_penalty=wall_penalty,
            steps_per_interval=None if backend=="exact-time" else 4,source=lambda x: 1+x,
            initial_temperature=lambda x: .1+.3*x,integrate_face_fluxes=True)
        heat = states.new_tensor(meta["interval_face_heat"])
        dt = states.new_tensor(times).diff()
        residual = states[:,:,0].diff(dim=0)/5+heat.diff(dim=1)-dt[:,None]*(1+x)/5
        assert residual.abs().max() < 5e-10
        torch.testing.assert_close(heat.sum(dim=0),states.new_tensor(meta["integrated_face_fluxes"]))
        torch.testing.assert_close(states[0,:,0],.1+.3*x)
        _,tail,_ = solve_heat_1d_wall_history(times[1:],walls[1:],5,2,backend=backend,wall_penalty=wall_penalty,
            steps_per_interval=None if backend=="exact-time" else 4,source=lambda x: 1+x,initial_temperature=states[1])
        torch.testing.assert_close(tail,states[1:],atol=5e-11,rtol=0.)


def test_wall_history_jump_between_independent_ramps():
    from tensordg import solve_heat_1d_wall_history

    times = [0., .017, .043]
    walls = [[[0., 1.], [.3, .7]], [[-.4, .8], [.2, .1]]]
    opts = dict(ncells=5, degree=2, backend="exact-time", integrate_face_fluxes=True)
    _, history, meta = solve_heat_1d_wall_history(times, walls, initial_temperature=lambda x: x, **opts)
    for i, endpoints in enumerate(walls):
        _, segment, single = solve_heat_1d_wall_history(
            times[i:i+2], endpoints, initial_temperature=history[i], **opts)
        torch.testing.assert_close(segment[-1], history[i+1])
        torch.testing.assert_close(torch.tensor(single["interval_face_heat"][0], dtype=history.dtype),
                                   torch.tensor(meta["interval_face_heat"][i], dtype=history.dtype))


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


def test_uniform_harmonic_field_with_two_storing_films_has_zero_material_flux():
    from tensordg import solve_harmonic_heat_1d

    temperature = .8+.4j
    capacity = 1.7
    films = (2.,5.)
    surface_capacity = (.35,1.1)
    ambient = [temperature*(1+1j*c/h) for c,h in zip(surface_capacity,films)]
    _,u,q,wall = solve_harmonic_heat_1d(1.,ncells=128,degree=2,conductivity=2.3,capacity=capacity,
        source=1j*capacity*temperature,wall_types=("robin","robin"),wall_values=ambient,
        film_coefficients=films,wall_areal_heat_capacity=surface_capacity,return_wall_temperatures=True)
    assert (wall-temperature).abs().max()<5e-13
    assert (u[:,0]-temperature).abs().max()<5e-13
    assert q[[0,-1]].abs().max()<1e-12


def test_graded_bulk_exchange_complex_ambient_and_flux_walls():
    from tensordg import solve_harmonic_heat_1d

    n,k,c,omega = 12,1.4,2.,.8
    def temp(x):
        return .3+.6j+(.7-.4j)*x**2
    def ambient(x):
        return 1+.1j-(.5+.2j)*x
    def loss(x):
        return 1+x*x
    def source(x):
        return 1j*omega*c*temp(x)-2*k*(.7-.4j)+loss(x)*(temp(x)-ambient(x))
    x,u,q = solve_harmonic_heat_1d(omega,ncells=n,degree=2,conductivity=k,capacity=c,
        wall_types=("neumann","neumann"),wall_values=(0.,2*k*(.7-.4j)),
        heat_loss_coefficient=loss,bulk_ambient_temperature=ambient,source=source)
    faces = torch.linspace(0.,1.,n+1,dtype=x.dtype)
    torch.testing.assert_close(u[:,0],temp(x)+(.7-.4j)/(12*n*n),rtol=0.,atol=2e-12)
    torch.testing.assert_close(q,-2*k*(.7-.4j)*faces,rtol=0.,atol=2e-11)


def test_layerwise_bulk_reservoir_pins_uniform_insulated_slab():
    from tensordg import solve_harmonic_heat_1d

    n,omega = 8,1e-12
    h = torch.linspace(.4,2.,n,dtype=torch.float64)
    target = .4-.8j
    ambient = target*(1+1j*omega/h)
    x,u,q = solve_harmonic_heat_1d(omega,ncells=n,degree=2,wall_types=("neumann","neumann"),
        heat_loss_coefficient=h,bulk_ambient_temperature=ambient)
    torch.testing.assert_close(u[:,0],torch.full_like(u[:,0],target),rtol=0.,atol=2e-12)
    assert q.abs().max()<2e-11


def test_bulk_history_preserves_linear_warming_and_cell_heat_with_graded_exchange():
    from tensordg import solve_heat_1d_bulk_history

    times = [0.,.07,.18]
    speeds = [1.3,-.6]
    loss = lambda x: 1+x*x
    start = .4
    pairs,expected = [],[start]
    for dt,speed in zip((.07,.11),speeds):
        finish = start+dt*speed
        pairs.append(tuple(lambda x,v=v,b=speed: v+2*b/loss(x) for v in (start,finish)))
        start = finish
        expected.append(finish)
    x,u,q,b = solve_heat_1d_bulk_history(times,pairs,initial_temperature=lambda x: .4,
        ncells=8,degree=2,conductivity=.7,capacity=2.,heat_loss_coefficient=loss,steps_per_interval=3)
    torch.testing.assert_close(u[:,:,0],x.new_tensor(expected)[:,None].expand(-1,8),rtol=0.,atol=1e-12)
    assert q.abs().max()<1e-12
    torch.testing.assert_close(b,2*u[:,:,0].diff(dim=0)/8,rtol=0.,atol=1e-12)


def test_bulk_history_with_time_varying_end_powers_has_exact_quadratic_field():
    from tensordg import solve_heat_1d_bulk_history

    c,k,h = 1.3,2.,3.5
    times = [.1,.15,.28]
    def field(x,t):
        return (1+t)*x*x+(.4-t)*x
    def reservoir(x,t):
        return field(x,t)+(c*(x*x-x)-2*k*(1+t))/h
    def flux(x,t):
        return -k*(2*(1+t)*x+.4-t)
    pairs = [[lambda x,t=t: reservoir(x,t) for t in ends] for ends in zip(times,times[1:])]
    powers = [[[flux(0,t),-flux(1,t)] for t in ends] for ends in zip(times,times[1:])]
    x,u,q,b = solve_heat_1d_bulk_history(times,pairs,initial_temperature=lambda x: field(x,times[0]),
        ncells=7,degree=2,capacity=c,conductivity=k,heat_loss_coefficient=h,inward_flux_intervals=powers)
    faces = torch.linspace(0.,1.,8,dtype=x.dtype)
    for i,(a,z) in enumerate(zip(times,times[1:])):
        torch.testing.assert_close(q[i],(z-a)*flux(faces,(a+z)/2),atol=2e-12,rtol=0.)
        torch.testing.assert_close(u[i+1,:,0],field(x,z)+(1+z)/(12*7**2),atol=1e-12,rtol=0.)
    assert ((c/7)*u[:,:,0].diff(dim=0)+q.diff(dim=1)-b).abs().max()<2e-12


def test_switched_films_and_wall_storage_match_time_linear_quadratic_solution():
    from tensordg import solve_heat_1d_bulk_history

    c,k,h = 1.4,.8,2.
    film = torch.tensor([1.7,3.4],dtype=torch.float64)
    cw = torch.tensor([.35,.8],dtype=film.dtype)
    ends = film.new_tensor([0.,1.])
    times = [0.,.08,.19]
    initial = lambda x: .3+x-.2*x*x
    mode = lambda x: .4+x*x
    amplitude = 0.
    reservoirs,air,powers,targets,heats = [],[],[],[],[]
    for a,b,speed in zip(times,times[1:],[1.2,-.6]):
        dt = b-a
        pair,wall_pair,power_pair = [],[],[]
        for elapsed in (0.,dt):
            amp = amplitude+elapsed*speed
            pair.append(lambda x,v=amp,s=speed: initial(x)+v*mode(x)+(c*s*mode(x)-k*(-.4+2*v))/h)
            surface = initial(ends)+amp*mode(ends)
            material = -k*(1-.4*ends+2*amp*ends)*film.new_tensor([1.,-1.])
            applied = (.1+elapsed)*film.new_tensor([1.,-2.])
            wall_pair.append((surface+(material+cw*speed*mode(ends)-applied)/film).tolist())
            power_pair.append(applied.tolist())
        reservoirs.append(pair)
        air.append(wall_pair)
        powers.append(power_pair)
        heats.append(dt*(-k*(1-.4*ends+2*(amplitude+.5*dt*speed)*ends)*film.new_tensor([1.,-1.])+cw*speed*mode(ends)))
        amplitude += dt*speed
        targets.append(initial(ends)+amplitude*mode(ends))
    _,u,q,bulk,wall,means = solve_heat_1d_bulk_history(times,reservoirs,ncells=7,degree=2,initial_temperature=initial,
        capacity=c,conductivity=k,heat_loss_coefficient=h,steps_per_interval=4,film_coefficients=film,
        wall_areal_heat_capacity=cw,wall_ambient_intervals=air,inward_flux_intervals=powers,return_wall_history=True,
        return_interval_modes=True)
    # Each exact quadratic modal profile varies linearly inside a loading interval.
    torch.testing.assert_close(means,.5*(u[:-1]+u[1:]),atol=2e-12,rtol=0.)
    torch.testing.assert_close(wall["temperature"][1:],torch.stack(targets),atol=2e-12,rtol=0.)
    exact_knots = torch.stack([initial(ends), *targets])
    torch.testing.assert_close(wall["mean_temperature"],.5*(exact_knots[:-1]+exact_knots[1:]),atol=2e-12,rtol=0.)
    torch.testing.assert_close(wall["external_heat"],torch.stack(heats),atol=2e-12,rtol=0.)
    assert (c*u[:,:,0].diff(dim=0)/7+q.diff(dim=1)-bulk).abs().max()<2e-12


def test_wall_admittance_and_cell_means_against_hyperbolic_slab(record_property):
    from tensordg import harmonic_heat_wall_admittance_1d, solve_harmonic_heat_1d

    k, c, loss, omega = 1.7, 2.1, 0.8, 4.
    wave = torch.tensor(((loss+1j*omega*c)/k)**0.5, dtype=torch.complex128)
    # U is the sum of the two normalized sinh profiles. Fourier's law at
    # either wall gives Y_ii=k*b*coth(b) and Y_ij=-k*b/sinh(b).
    exact_y = (k*wave/wave.sinh()) * torch.stack((
        torch.stack((wave.cosh(), -torch.ones_like(wave))),
        torch.stack((-torch.ones_like(wave), wave.cosh()))))
    drive = torch.tensor([1.+.3j, -.2+.7j], dtype=wave.dtype)
    errors = []
    for n in (8, 16, 32):
        opts = dict(ncells=n, degree=2, conductivity=k, capacity=c, heat_loss_coefficient=loss)
        admittance = harmonic_heat_wall_admittance_1d(omega, **opts)
        _, coeff, flux = solve_harmonic_heat_1d(omega, wall_values=drive, **opts)
        inward = torch.stack((flux[0], -flux[-1]))
        torch.testing.assert_close(admittance @ drive, inward, atol=2e-11, rtol=0)
        x = torch.linspace(0, 1, n+1, dtype=torch.float64)
        primitive = (-drive[0]*(wave*(1-x)).cosh()+drive[1]*(wave*x).cosh())/(wave*wave.sinh())
        exact_means = n*primitive.diff()
        row = [(admittance-exact_y).abs().max().item(), (coeff[:, 0]-exact_means).abs().max().item()]
        record_property(f"max_admittance_error_{n}_W_per_m2_K", row[0])
        record_property(f"max_cell_mean_error_{n}_K", row[1])
        errors.append(row)
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 3.5 for a, b in zip(coarse, fine))


def test_physical_thickness_admittance_against_two_layer_transfer_matrix(record_property):
    import cmath
    import numpy as np
    import pytest
    from tensordg import harmonic_heat_wall_admittance_1d

    k, c, omega = 1.6, 2.2, 1.7
    for length in (0.25, 2.5):
        # Transfer state [T, -k*dT/dx] from left to right through both layers.
        transfer = np.eye(2)
        for loss in (0.3, 3.4):
            b = cmath.sqrt((loss+1j*omega*c)/k)
            cosh, sinh = cmath.cosh(b*length/2), cmath.sinh(b*length/2)
            transfer = np.array([[cosh, -sinh/(k*b)], [-k*b*sinh, cosh]]) @ transfer
        a, off = transfer[0]
        exact = torch.from_numpy(np.array([[-a, 1.], [1., -transfer[1, 1]]])/off)
        errors = []
        for n in (8, 16, 32):
            opts = dict(thickness=length, ncells=n, degree=2, conductivity=k, capacity=c)
            def physical_loss(x):
                return 0.3 + 3.1*(x >= length/2).to(x.dtype)
            y = harmonic_heat_wall_admittance_1d(omega, heat_loss_coefficient=physical_loss, **opts)
            per_cell = [0.3]*(n//2) + [3.4]*(n//2)
            sampled = harmonic_heat_wall_admittance_1d(omega, heat_loss_coefficient=per_cell, **opts)
            torch.testing.assert_close(y, sampled, atol=2e-12, rtol=1e-12)
            error = (y-exact).abs().max().item()
            errors.append(error)
            record_property(f"max_admittance_error_L{length}_{n}_W_per_m2_K", error)
        assert all(a/b > 8 for a, b in zip(errors, errors[1:]))
    for invalid in (0., -1., float("inf"), float("nan")):
        with pytest.raises(ValueError, match="thickness"):
            harmonic_heat_wall_admittance_1d(omega, thickness=invalid)


def test_composite_harmonic_admittance_against_contact_transfer(record_property):
    import cmath
    import numpy as np
    from tensordg import compose_heat_wall_admittances_1d, harmonic_heat_wall_admittance_1d

    omega = 3.2
    layers = [(0.35, 1.3, 2.1), (0.2, 3.7, 1.4), (0.6, 0.9, 1.2)]
    contacts = [0., 0.27]
    transfer = np.eye(2)
    for i, (length, k, c) in enumerate(layers):
        if i:
            transfer = np.array([[1., -contacts[i-1]], [0., 1.]]) @ transfer
        wave = cmath.sqrt(1j*omega*c/k)
        a, b = cmath.cosh(wave*length), cmath.sinh(wave*length)
        transfer = np.array([[a, -b/(k*wave)], [-k*wave*b, a]]) @ transfer
    exact = torch.from_numpy(np.array([[-transfer[0, 0], 1.], [1., -transfer[1, 1]]])/transfer[0, 1])
    errors = []
    for n in (4, 8, 16):
        matrices = [harmonic_heat_wall_admittance_1d(
            omega, ncells=n, thickness=length, conductivity=k, capacity=c) for length, k, c in layers]
        combined = compose_heat_wall_admittances_1d(matrices, contacts)
        reverse = compose_heat_wall_admittances_1d([a.flip((0, 1)) for a in reversed(matrices)], contacts[::-1])
        torch.testing.assert_close(reverse.flip((0, 1)), combined, atol=1e-13, rtol=0)
        torch.testing.assert_close(compose_heat_wall_admittances_1d(matrices[:1]), matrices[0])
        errors.append((combined-exact).abs().max().item())
        record_property(f"max_stack_admittance_error_{n}_W_per_m2_K", errors[-1])
    assert all(a/b > 8 for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize("film_exchange", [False, True])
@pytest.mark.parametrize("heated", [False, True])
@pytest.mark.parametrize("storing", [False, True])
def test_harmonic_stack_internal_states_against_propagated_traces(heated, storing, film_exchange, record_property):
    import cmath
    import numpy as np
    from tensordg import harmonic_heat_wall_admittance_1d, compose_heat_wall_admittances_1d

    omega = 2.7
    layers = [(0.5, 1.4, 2.1), (0.2, 2.6, 1.2), (0.3, 1.8, 1.4)]
    resistances = [0.23, 0.]
    sources = [0.43+0.18j, -0.27j] if heated else [0., 0.]
    surface_capacity = [0.7, 0.19] if storing else [0., 0.]
    films = [0.6, 1.1] if film_exchange else [0., 0.]
    ambient = [0.31-0.45j, -0.23+0.9j]
    temperature, q = 1.2+0.4j, 0.8-0.15j
    reference_t, reference_q = [], []
    for i, (length, k, capacity) in enumerate(layers):
        wave = cmath.sqrt(1j*omega*capacity/k)
        co, si = cmath.cosh(wave*length), cmath.sinh(wave*length)
        right_t, right_q = co*temperature-si*q/(k*wave), -k*wave*si*temperature+co*q
        reference_t.append((temperature, right_t))
        reference_q.append((q, -right_q))
        if i < len(resistances):
            temperature = right_t-resistances[i]*right_q
            exchange = films[i]*(ambient[i]-temperature)
            q = right_q+sources[i]+exchange-1j*omega*surface_capacity[i]*temperature
    drive = [reference_t[0][0], reference_t[-1][1]]
    errors = []
    for n in (4, 8, 16):
        matrices = [harmonic_heat_wall_admittance_1d(
            omega, ncells=n, thickness=length, conductivity=k, capacity=c) for length, k, c in layers]
        opts = dict(interface_heat_sources=sources) if heated else {}
        if film_exchange:
            opts.update(interface_film_coefficients=films, interface_ambient_temperatures=ambient)
        if storing:
            opts.update(interface_areal_heat_capacity=surface_capacity, angular_frequency=omega)
        y, t, flux = compose_heat_wall_admittances_1d(matrices, resistances, wall_temperatures=drive, **opts)
        _, _, base_flux = compose_heat_wall_admittances_1d(matrices, resistances, wall_temperatures=(0., 0.), **opts)
        if storing and not heated:
            reduced = compose_heat_wall_admittances_1d(matrices, resistances, **opts)
            torch.testing.assert_close(reduced, y)
        response = flux - base_flux
        torch.testing.assert_close(y @ t.new_tensor(drive), torch.stack((response[0, 0], response[-1, 1])))
        stored_power = 1j*omega*t.new_tensor(surface_capacity)*t[1:, 0]
        film_power = t.new_tensor(films)*(t.new_tensor(ambient)-t[1:, 0])
        torch.testing.assert_close(flux[:-1, 1]+flux[1:, 0]+stored_power, t.new_tensor(sources)+film_power,
                                   atol=1e-12, rtol=0)
        torch.testing.assert_close(t[:-1, 1]-t[1:, 0], -t.new_tensor(resistances)*flux[:-1, 1],
                                   atol=1e-12, rtol=0)
        row = [np.max(np.abs(t.numpy()-reference_t)), np.max(np.abs(flux.numpy()-reference_q))]
        record_property(f"max_trace_error_{n}_K", float(row[0]))
        record_property(f"max_flux_error_{n}_W_per_m2", float(row[1]))
        errors.append(row)
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 8 for a, b in zip(coarse, fine))

def test_stack_rejects_undefined_interface_storage():
    from tensordg import compose_heat_wall_admittances_1d

    layers = [torch.eye(2)]*2
    for capacity, omega in ((1., None), (-1., 2.), (float("nan"), 2.), ([1., 2.], 2.), (1j, 2.)):
        with pytest.raises(ValueError):
            compose_heat_wall_admittances_1d(layers, interface_areal_heat_capacity=capacity,
                                             angular_frequency=omega)


def test_stack_with_affine_volume_generation_and_contact_heater(record_property):
    import cmath
    from tensordg import harmonic_heat_wall_admittance_1d, compose_heat_wall_admittances_1d

    omega = 2.3
    # L, k, C, constant and slope of heat generation in each layer's local x.
    layers = [(0.35, 1.3, 1.7, 1.2-0.3j, 2j), (0.6, 2.1, 0.8, -0.5j, -1.3+0.2j)]
    resistance, heater = 0.21, 0.7-0.4j
    t, q = 0.8+0.3j, 0.4-0.2j
    exact_t, exact_q = [], []
    for length, k, c, base, slope in layers:
        wave = cmath.sqrt(1j*omega*c/k)
        co, si = cmath.cosh(wave*length), cmath.sinh(wave*length)
        # Subtract the exact affine particular solution before propagation.
        particular, derivative = base/(1j*omega*c), slope/(1j*omega*c)
        u, f = t-particular, q+k*derivative
        end_t = particular+length*derivative + co*u-si*f/(k*wave)
        end_q = -k*derivative-k*wave*si*u+co*f
        exact_t.append((t, end_t))
        exact_q.append((q, -end_q))
        t, q = end_t-resistance*end_q, end_q+heater
    temperatures = [exact_t[0][0], exact_t[-1][1]]
    errors = []
    for n in (4, 8, 16):
        matrices, offsets = zip(*(harmonic_heat_wall_admittance_1d(
            omega, ncells=n, thickness=length, conductivity=k, capacity=c,
            source=lambda x: base+slope*x) for length, k, c, base, slope in layers))
        _, surface, flux = compose_heat_wall_admittances_1d(
            matrices, resistance, layer_flux_offsets=offsets, wall_temperatures=temperatures,
            interface_heat_sources=heater)
        row = [(surface-surface.new_tensor(exact_t)).abs().max().item(),
               (flux-flux.new_tensor(exact_q)).abs().max().item()]
        record_property(f"max_temperature_error_{n}_K", row[0])
        record_property(f"max_flux_error_{n}_W_per_m2", row[1])
        torch.testing.assert_close(flux[0, 1]+flux[1, 0], flux.new_tensor(heater), rtol=0, atol=1e-12)
        errors.append(row)
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 8 for a, b in zip(coarse, fine))


def test_loaded_wall_flux_offset_against_augmented_transfer(record_property):
    import numpy as np
    from scipy.linalg import expm
    from tensordg import harmonic_heat_wall_admittance_1d, compose_heat_wall_admittances_1d

    omega = 2.1
    layers = [(.31, 1.5, 2., .8j), (.4, .9, 1.1, -.3+.1j), (.2, 2., 1.3, .7)]
    resistances, heaters = [.12, .25], [.4-.3j, .1j]
    # Exact propagation of [T, rightward flux, 1] with constant generation.
    transfer = np.eye(3)
    for index, (length, k, c, source) in enumerate(layers):
        generator = np.array([[0, -1/k, 0], [-1j*omega*c, 0, source], [0, 0, 0]])
        transfer = expm(length*generator) @ transfer
        if index < len(heaters):
            transfer = np.array([[1, -resistances[index], 0], [0, 1, heaters[index]], [0, 0, 1]]) @ transfer

    def exact_flux(drive):
        q = (drive[1]-transfer[0, 0]*drive[0]-transfer[0, 2])/transfer[0, 1]
        return np.array([q, -(transfer[1] @ [drive[0], q, 1])])

    errors = []
    for n in (4, 8, 16):
        matrices, offsets = zip(*(harmonic_heat_wall_admittance_1d(
            omega, thickness=length, conductivity=k, capacity=c, source=source, ncells=n)
            for length, k, c, source in layers))
        opts = dict(layer_flux_offsets=offsets, interface_heat_sources=heaters, return_flux_offset=True)
        y, offset = compose_heat_wall_admittances_1d(matrices, resistances, **opts)
        error = np.max(abs(offset.numpy()-exact_flux((0., 0.))))
        record_property(f"zero_drive_flux_error_{n}_W_per_m2", float(error))
        for drive in ([.4+.3j, -.1j], [-.3, .7+.1j]):
            _, _, flux, baseline = compose_heat_wall_admittances_1d(
                matrices, resistances, wall_temperatures=drive, **opts)
            prediction = y @ y.new_tensor(drive) + offset
            torch.testing.assert_close(prediction, torch.stack((flux[0, 0], flux[-1, 1])), rtol=0, atol=1e-12)
            torch.testing.assert_close(baseline, offset)
            error = max(error, np.max(abs(prediction.numpy()-exact_flux(drive))))
        record_property(f"max_wall_flux_error_{n}_W_per_m2", float(error))
        errors.append(error)
    assert all(a/b > 8 for a, b in zip(errors, errors[1:]))

@pytest.mark.parametrize("storing", [False, True])
def test_loaded_stack_mixed_wall_conditions_against_transfer_solution(storing, record_property):
    from itertools import product
    from scipy.linalg import expm
    import numpy as np
    from tensordg import (harmonic_heat_wall_admittance_1d, compose_heat_wall_admittances_1d,
                          solve_heat_wall_admittance_1d)

    omega, resistance = 2.7, .18
    layers = [(.6, 1.5, 1.2, .5-.2j), (.35, .9, 1.9, -.3j)]
    initial_t, initial_q = 1.+.2j, .4-.3j
    state = np.array([initial_t, initial_q, 1.])
    for i, (length, k, c, source) in enumerate(layers):
        if i:
            state[0] -= resistance*state[1]
        state = expm(length*np.array([[0, -1/k, 0], [-1j*omega*c, 0, source], [0, 0, 0]])) @ state
    exact_t = np.array([initial_t, state[0]])
    exact_q = np.array([initial_q, -state[1]])
    capacities = np.array([.37, .81]) if storing else np.zeros(2)
    exact_external = exact_q + 1j*omega*capacities*exact_t
    errors = []
    for n in (4, 8, 16):
        matrices, offsets = zip(*(harmonic_heat_wall_admittance_1d(
            omega, thickness=L, conductivity=k, capacity=c, source=s, ncells=n) for L, k, c, s in layers))
        y, offset = compose_heat_wall_admittances_1d(matrices, resistance, layer_flux_offsets=offsets,
                                                    return_flux_offset=True)
        worst_t = worst_q = worst_external = 0.
        for kinds in product(("dirichlet", "neumann", "robin"), repeat=2):
            films, values, heaters = [], [], []
            for i, kind in enumerate(kinds):
                h = (1.8, 3.5)[i] if kind == "robin" else 0.
                heater = (1.3+.7j, -.8+.5j)[i] if kind != "dirichlet" else 0.
                heaters.append(heater)
                films.append(h)
                if kind == "robin":
                    values.append(exact_t[i]+(exact_external[i]-heater)/h)
                elif kind == "neumann":
                    values.append(exact_external[i]-heater)
                else:
                    values.append(exact_t[i])
            t, q, external = solve_heat_wall_admittance_1d(
                y, kinds, values, flux_offset=offset, film_coefficients=films,
                wall_areal_heat_capacity=capacities, angular_frequency=omega,
                return_external_flux=True, wall_applied_flux=heaters)
            worst_external = max(worst_external, np.max(abs(external.numpy()-exact_external)))
            worst_t = max(worst_t, np.max(abs(t.numpy()-exact_t)))
            worst_q = max(worst_q, np.max(abs(q.numpy()-exact_q)))
        errors.append((worst_t, worst_q, worst_external))
        record_property(f"max_external_supply_error_{n}_W_per_m2", float(worst_external))
        record_property(f"max_wall_temperature_error_{n}_K", float(worst_t))
        record_property(f"max_wall_flux_error_{n}_W_per_m2", float(worst_q))
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 8 for a,b in zip(coarse, fine))


def test_wall_admittance_storage_requires_valid_capacities_and_frequency():
    from tensordg import solve_heat_wall_admittance_1d

    args = (torch.eye(2), ("neumann", "dirichlet"), (1., 2.))
    temperature, flux = solve_heat_wall_admittance_1d(*args)
    torch.testing.assert_close(temperature, flux.new_tensor([1., 2.]))
    torch.testing.assert_close(temperature, flux)
    for cap, omega in (([1., 0.], None), ([1., 0.], -1.), ([1., -1.], 2.),
                       ([1.j, 0.], 2.), ([float("inf"), 0.], 2.), ([1.], 2.)):
        with pytest.raises(ValueError):
            solve_heat_wall_admittance_1d(*args, wall_areal_heat_capacity=cap, angular_frequency=omega)


def test_two_trace_sources_balance_each_side_of_resistive_and_perfect_contacts():
    from tensordg import compose_heat_wall_admittances_1d

    layer = torch.tensor([[2.+.6j, -.9], [-.9, 1.3+.2j]], dtype=torch.complex128)
    left = layer.new_tensor([.34-.5j, -.12+.3j])
    right = layer.new_tensor([.15+.1j, .32-.25j])
    resistance = left.real.new_tensor([.6, 0.])
    drive = layer.new_tensor([1.-.3j, .55j])
    opts = dict(interface_heat_sources=right, interface_left_heat_sources=left)
    y, offset = compose_heat_wall_admittances_1d([layer]*3, resistance, return_flux_offset=True, **opts)
    _, t, flux = compose_heat_wall_admittances_1d([layer]*3, resistance, wall_temperatures=drive, **opts)
    torch.testing.assert_close(y @ drive+offset, torch.stack((flux[0, 0], flux[-1, 1])))
    torch.testing.assert_close(flux[:-1, 1]+flux[1:, 0], left+right)
    torch.testing.assert_close(t[:-1, 1]-t[1:, 0], resistance*(left-flux[:-1, 1]))


@pytest.mark.parametrize("resistance", [0., .35])
def test_independent_contact_storage_rates(resistance):
    from tensordg import compose_heat_wall_admittances_1d

    layer = torch.tensor([[2.+.6j, -.9], [-.9, 1.3+.2j]], dtype=torch.complex128)
    omega, cl, cr = 2.3, .28, .51
    opts = dict(interface_left_areal_heat_capacity=cl, interface_areal_heat_capacity=cr,
                interface_left_heat_sources=.4+.1j, interface_heat_sources=-.2j)
    with pytest.raises(ValueError, match="angular frequency"):
        compose_heat_wall_admittances_1d([layer]*2, resistance, return_flux_offset=True, **opts)
    _, t, flux = compose_heat_wall_admittances_1d([layer]*2, resistance, wall_temperatures=(1., .3j),
                                                  angular_frequency=omega, **opts)
    through = .4+.1j-flux[0, 1]-1j*omega*cl*t[0, 1]
    torch.testing.assert_close(t[0, 1]-t[1, 0], resistance*through)
    torch.testing.assert_close(flux[1, 0], through-.2j-1j*omega*cr*t[1, 0])


@pytest.mark.parametrize("r", [0., .35])
def test_left_film_equals_reflected_right_film(r):
    from tensordg import compose_heat_wall_admittances_1d

    layer = torch.tensor([[2.+.6j, -.9], [-.9, 1.3+.2j]], dtype=torch.complex128)
    layers = [layer, .7*layer]
    reverse = [m.flip((0, 1)) for m in reversed(layers)]
    left = dict(interface_left_film_coefficients=2.1, interface_left_ambient_temperatures=.32-.46j)
    right = dict(interface_film_coefficients=2.1, interface_ambient_temperatures=.32-.46j)
    y, offset = compose_heat_wall_admittances_1d(layers, r, return_flux_offset=True, **left)
    yr, br = compose_heat_wall_admittances_1d(reverse, r, return_flux_offset=True, **right)
    torch.testing.assert_close(y, yr.flip((0, 1)))
    torch.testing.assert_close(offset, br.flip((0,)))


@pytest.mark.parametrize("backend", ["dense", "sparse", "matrix-free", "exact-time"])
def test_ramped_wall_interval_temperature_modes(backend):
    from tensordg.heat_1d import solve_heat_1d_wall_history

    times = torch.tensor([0., .03, .1], dtype=torch.float64)
    walls = torch.stack((times, 3*times), dim=1)
    centers, _, meta = solve_heat_1d_wall_history(
        times, walls, 6, 2, backend=backend, source=lambda x: 1+2*x,
        initial_temperature=torch.zeros_like, return_interval_temperatures=True)
    means = centers.new_tensor(meta["interval_temperature_modes"])
    midpoint = (times[1:]+times[:-1])/2
    expected = torch.zeros_like(means)
    expected[:, :, 0] = midpoint[:, None]*(1+2*centers)
    expected[:, :, 1] = midpoint[:, None]/6
    torch.testing.assert_close(means, expected, atol=1e-10, rtol=0.)


def test_independent_wall_penalty_changes_only_two_trace_blocks():
    from tensordg.heat_1d import assemble_sipg_1d

    n = 5
    base = assemble_sipg_1d(n)
    tuned = assemble_sipg_1d(n, wall_penalty=40.)
    delta = torch.zeros_like(base["diffusion"])
    left = delta.new_tensor([1, -1, 1])
    right = delta.new_tensor([1, 1, 1])
    delta[:3, :3] = 20*9*n*torch.outer(left, left)
    delta[-3:, -3:] = 20*9*n*torch.outer(right, right)
    torch.testing.assert_close(tuned["diffusion"] - base["diffusion"], delta)
    for value in (0., -1., math.nan, math.inf):
        with pytest.raises(ValueError, match="wall_penalty"):
            assemble_sipg_1d(n, wall_penalty=value)


@pytest.mark.parametrize("omega", [1e-14, 1e-8, 1.])
def test_separated_wall_and_cell_temperature_fluctuations(omega, record_property):
    from tensordg import solve_harmonic_heat_1d

    n, c, k = 12, 1.5, 2.
    walls = torch.tensor([.3, .9], dtype=torch.float64)
    ends = torch.tensor([0., 1.], dtype=torch.float64)
    a, b, rate = .5+.3j, -.2+.6j, .8-.6j
    # Shift the quadratic to zero integrated storage, including the two walls.
    shift = (c*(a/3+b/2)+walls[1]*(a+b))/(c+walls.sum())

    def local(x):
        return a*x**2+b*x-shift

    def flux(x):
        return -k*(2*a*x+b)

    def source(x):
        return c*(rate+1j*omega*local(x))-2*k*a

    loads = flux(ends)*ends.new_tensor([1., -1.])+walls*(rate+1j*omega*local(ends))
    x, u, q, surface, mean = solve_harmonic_heat_1d(
        omega, ncells=n, degree=2, capacity=c, conductivity=k, source=source,
        wall_types=("neumann", "neumann"), wall_values=loads,
        wall_areal_heat_capacity=walls, return_wall_temperatures=True, separate_mean=True)
    for name, actual, expected in (
        ("cell_mean_K", u[:, 0], local(x)+a/(12*n**2)),
        ("wall_trace_K", surface, local(ends)),
        ("face_flux_W_m2", q, flux(torch.linspace(0., 1., n+1, dtype=x.dtype))),
        ("mean_rate_K_s", 1j*omega*mean, mean.new_tensor(rate)),
    ):
        record_property(name, (actual-expected).abs().max().item())
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=0.)


def test_separate_harmonic_mean_requires_unanchored_slab():
    from tensordg import solve_harmonic_heat_1d

    with pytest.raises(ValueError, match="without temperature or exchange anchors"):
        solve_harmonic_heat_1d(2., separate_mean=True)


@pytest.mark.parametrize("omega,rate", [(0., 0.), (0., .75), (0., -.5), (.3, .75)])
def test_constant_load_heating_rate_and_relative_profile(omega, rate, record_property):
    from tensordg import solve_harmonic_heat_1d

    n, c, k = 8, 2., 3.
    wall_c = torch.tensor([.25, .75], dtype=torch.float64)
    ends = torch.tensor([0., 1.], dtype=torch.float64)
    # Relative profile x²-x/2 shifted to have zero bulk + wall stored energy.
    offset = (c/12+wall_c[1]/2)/(c+wall_c.sum())

    def local(x):
        return x**2-x/2-offset

    def source(x):
        return c*(rate+1j*omega*local(x))-2*k

    def flux(x):
        return -k*(2*x-.5)

    external = torch.tensor([1., -1.])*flux(ends)+wall_c*(rate+1j*omega*local(ends))
    x, u, q, surface, measured = solve_harmonic_heat_1d(
        omega, ncells=n, degree=2, capacity=c, conductivity=k, source=source,
        wall_types=("neumann", "neumann"), wall_values=external,
        wall_areal_heat_capacity=wall_c, return_wall_temperatures=True, return_mean_rate=True)
    for name, err in (
        ("cell_mean_K", u[:, 0]-local(x)-1/(12*n**2)),
        ("wall_trace_K", surface-local(ends)),
        ("face_flux_W_m2", q-flux(torch.linspace(0., 1., n+1, dtype=x.dtype))),
        ("mean_rate_K_s", measured-rate),
        ("relative_stored_energy_J_m2", c*u[:, 0].mean()+(wall_c*surface).sum()),
    ):
        maximum = err.abs().max().item()
        record_property(name, maximum)
        assert maximum < 2e-11


def test_drifting_heat_stack_single_layer_linear_profile():
    from tensordg import solve_drifting_heat_stack_1d

    x, modes, flux, walls, rate = solve_drifting_heat_stack_1d(
        [2.], 3., 1.5, (.6, -.6), source=.75, ncells=5, degree=1)
    torch.testing.assert_close(modes[:, :, 0], -.2*(x-1.), atol=1e-12, rtol=0.)
    torch.testing.assert_close(flux, torch.full_like(flux, .6), atol=1e-12, rtol=0.)
    torch.testing.assert_close(walls, walls.new_tensor([[.2, -.2]]), atol=1e-12, rtol=0.)
    assert rate == .5


def test_drifting_stack_uses_crossing_flux_for_storing_contact_jump():
    from tensordg import solve_drifting_heat_stack_1d

    _, u, q, traces, rate = solve_drifting_heat_stack_1d(
        [1., 1.], 1., 1., (4., 0.), contact_resistance=.25, ncells=4,
        interface_left_areal_heat_capacity=.5, interface_areal_heat_capacity=1.5)
    assert rate == 1.
    # Four units of incoming heat supply two bulk units and two surface units.
    torch.testing.assert_close(q[:, [0, -1]], q.new_tensor([[4., 3.], [1., 0.]]))
    crossing = (traces[0, 1]-traces[1, 0])/.25
    torch.testing.assert_close(crossing, q.new_tensor(2.5), atol=1e-12, rtol=0.)
    energy = u[:, :, 0].mean(1).sum()+.5*traces[0, 1]+1.5*traces[1, 0]
    assert energy.abs() < 1e-12


def test_balanced_contact_heater_and_sink_drive_an_internal_temperature_jump():
    from tensordg import solve_drifting_heat_stack_1d

    _, u, q, walls, rate = solve_drifting_heat_stack_1d(
        [1., 1.], 1., 1., (0., 0.), contact_resistance=.4, ncells=3,
        interface_left_heat_sources=1.5, interface_heat_sources=-1.5)
    assert rate == 0.
    torch.testing.assert_close(q, torch.zeros_like(q), atol=1e-12, rtol=0.)
    expected = walls.new_tensor([.3, -.3])[:, None]
    torch.testing.assert_close(walls, expected.expand_as(walls), atol=1e-12, rtol=0.)
    torch.testing.assert_close(u[:, :, 0], expected.expand_as(u[:, :, 0]), atol=1e-12, rtol=0.)


def test_stepped_drifting_stack_conserves_power_across_an_insulated_shoulder():
    from tensordg import solve_drifting_heat_stack_1d

    _, _, flux, trace, rate = solve_drifting_heat_stack_1d(
        [1., 2.], 3., 2., (1., -.25), area=[2., 8.], contact_resistance=.6)
    assert rate == 0.
    torch.testing.assert_close(flux, flux.new_tensor([[1.], [.25]]).expand_as(flux))
    torch.testing.assert_close(trace[0, 1]-trace[1, 0], flux.new_tensor(.6))


@pytest.mark.parametrize("reverse", [False, True])
def test_drifting_stack_shoulder_storage_follows_large_section(reverse):
    from tensordg import solve_drifting_heat_stack_1d

    area = [3., 1.] if reverse else [1., 3.]
    ends = (0., 6.) if reverse else (6., 0.)
    _, _, q, trace, rate = solve_drifting_heat_stack_1d(
        [1., 1.], 1., 1., ends, area=area, shoulder_areal_heat_capacity=1.,
        contact_resistance=.2, ncells=4)
    assert rate == 1.
    # Heating the smaller layer consumes 1 W. The remaining 5 W crosses
    # the contact and supplies 3 W to the larger bulk and 2 W to the shoulder.
    crossing = (trace[0, 1]-trace[1, 0])/.2
    torch.testing.assert_close(crossing, q.new_tensor(-5. if reverse else 5.))
    stored = area[0]*q[0, -1]-area[1]*q[1, 0]
    torch.testing.assert_close(stored, q.new_tensor(2.))


def test_drifting_stack_storing_ends_share_the_bulk_heating_rate():
    from tensordg import solve_drifting_heat_stack_1d

    # An 8 W input across 2 m² heats the bulk and both external end masses.
    _, u, q, t, rate = solve_drifting_heat_stack_1d(
        [1.], 2., 1., (4., 0.), area=2., end_areal_heat_capacity=(1., 2.), ncells=4)
    assert rate == 1.
    torch.testing.assert_close(q[0, [0, -1]], q.new_tensor([3., 2.]))
    energy = 2*u[0, :, 0].mean()+2*t[0, 0]+4*t[0, 1]
    assert energy.abs() < 1e-12


@pytest.mark.parametrize("degree,n", [(2, 4), (2, 8), (2, 16), (3, 6)])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_drifting_stack_affine_generation(degree, n, sign, record_property):
    from tensordg import (reconstruct_heat_stack_flux, reconstruct_heat_stack_temperature,
                          sample_heat_stack_fields, solve_drifting_heat_stack_1d)

    length = [1.5, .5]
    area = [1., 2.]
    k = [2., .8]
    capacity = [1., 1.5]
    gradients = [sign*2., -sign*.75]
    # Uniform external through-power plus zero-integral generation gradients.
    power, resistance, target_rate = .3, .4, .5
    _, u, q, walls, rate = solve_drifting_heat_stack_1d(
        length, k, capacity, (power, -power/2), area=area, contact_resistance=resistance,
        source=[c*target_rate for c in capacity], source_gradient=gradients, ncells=n, degree=degree)
    expected_means, expected_fluxes, expected_walls = [], [], []
    fractions = [.13, .41, .78]
    sampled_t, sampled_q = sample_heat_stack_fields(u, length, k, fractions)
    recovered = reconstruct_heat_stack_flux(q, length, [0., *fractions, 1.], source_gradient=gradients)
    torch.testing.assert_close(recovered[:, :, 0], q[:, :-1], atol=0., rtol=0.)
    torch.testing.assert_close(recovered[:, :, -1], q[:, 1:], atol=0., rtol=0.)
    reconstructed_t = reconstruct_heat_stack_temperature(
        u[:, :, 0], q, length, k, [0., *fractions, 1.], source_gradient=gradients)
    target_t, target_q = [], []
    temperature = 0.
    for h, section, cond, gradient in zip(length, area, k, gradients):
        x = torch.linspace(0., h, n+1, dtype=u.dtype)
        # Integrate dq/dx = G*(x-h/2), then Fourier's law and the cell mean.
        points = x[:-1, None]+(h/n)*u.new_tensor(fractions)
        target_q.append(power/section+gradient*points*(points-h)/2)
        target_t.append(temperature-power*points/(section*cond)
                        -gradient*(points**3/6-h*points**2/4)/cond)
        expected_fluxes.append(power/section + gradient*x*(x-h)/2)
        primitive = (temperature*x-power*x**2/(2*section*cond)
                     -gradient*(x**4/24-h*x**3/12)/cond)
        expected_means.append(primitive.diff()/(h/n))
        east = temperature-power*h/(section*cond)+gradient*h**3/(12*cond)
        expected_walls.append([temperature, east])
        temperature = east-resistance*power
    means = torch.stack(expected_means)
    masses = u.new_tensor([h*a*c for h, a, c in zip(length, area, capacity)])
    shift = masses@means.mean(1)/masses.sum()
    errors = dict(cell_mean_K=(u[:, :, 0]-means+shift).abs().max().item(),
                  trace_K=(walls-u.new_tensor(expected_walls)+shift).abs().max().item(),
                  face_flux_W_m2=(q-torch.stack(expected_fluxes)).abs().max().item(),
                  rate_K_s=abs(rate-target_rate))
    errors["reconstructed_interior_K"] = (reconstructed_t[:, :, 1:-1]-torch.stack(target_t)+shift).abs().max().item()
    jump = reconstructed_t[0, -1, -1]-reconstructed_t[1, 0, 0]
    errors["reconstructed_contact_jump_K"] = abs(jump.item()-resistance*power)
    assert max(errors["reconstructed_interior_K"], errors["reconstructed_contact_jump_K"]) < 2e-11
    recovered_error = (recovered[:, :, 1:-1]-torch.stack(target_q)).abs().max().item()
    record_property("conservative_interior_flux_W_m2", recovered_error)
    assert recovered_error < 2e-11
    interior_t = (sampled_t-torch.stack(target_t)+shift).abs().max().item()
    interior_q = (sampled_q-torch.stack(target_q)).abs().max().item()
    record_property("interior_temperature_K", interior_t)
    record_property("interior_flux_W_m2", interior_q)
    assert interior_t < (2e-10 if degree == 3 else .1/n**3)
    assert interior_q < (2e-10 if degree == 3 else .5/n**2)
    for name, error in errors.items():
        record_property(name, error)
    assert errors["rate_K_s"] < 1e-13
    assert max(errors.values()) < (2e-10 if degree == 3 else .1/n**2)


def test_heat_stack_sampling_retains_separate_endpoint_traces():
    from tensordg import sample_heat_stack_fields

    coeff = torch.tensor([[[1+2j, 2.], [-3., 1j]]], dtype=torch.complex128)
    t, q = sample_heat_stack_fields(coeff, [2.], 3., [0., 1.])
    torch.testing.assert_close(t, coeff.new_tensor([[[-1+2j, 3+2j], [-3-1j, -3+1j]]]))
    torch.testing.assert_close(q, coeff.new_tensor([[[-12., -12.], [-6j, -6j]]]))

def test_stack_temperature_reconstruction_preserves_arbitrary_cell_means():
    from tensordg import reconstruct_heat_stack_temperature

    means = torch.tensor([[2., -1.], [.4, 1.3]], dtype=torch.float64)
    flux = means.new_tensor([[4., -3., 2.], [1., 7., -.2]])
    # Two-point Gauss quadrature integrates each reconstructed cubic exactly.
    offset = .5/math.sqrt(3)
    t = reconstruct_heat_stack_temperature(means, flux, [1., 2.], [1., 4.],
                                           [.5-offset, .5+offset], source_gradient=[.5, -2.])
    torch.testing.assert_close(t.mean(-1), means, rtol=0., atol=1e-14)


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_quadratic_generation_in_drifting_stack(degree, record_property):
    from tensordg import (solve_drifting_heat_stack_1d, reconstruct_heat_stack_flux,
                          reconstruct_heat_stack_temperature, sample_heat_stack_fields)

    # T = a*x^4+b*x^3+offset, q = -k*T'. Start x at each layer's
    # midpoint. Prescribe the source from energy balance, enforcing a common
    # interface flux via the linear term of each temperature polynomial.
    import numpy as np
    from numpy.polynomial import Polynomial
    lengths, ks, capacities = [1.5, .75], [1.2, 2.5], [.8, 1.7]
    drift, resistance, incoming = .35, .2, .6
    walls, fields, fluxes, means, gradients, curvatures = [], [], [], [], [], []
    left_temp = 0.
    for length, k, c, a, b in zip(lengths, ks, capacities, [.12, -.28], [-.07, .13]):
        left, right = -length/2, length/2
        t = Polynomial([0., -incoming/k-4*a*left**3-3*b*left**2, 0., b, a])
        t += left_temp-t(left)
        q = -k*t.deriv()
        source = q.deriv()+drift*c
        means.append((source.integ()(right)-source.integ()(left))/length)
        gradients.append(source.deriv()(0.))
        curvatures.append(source.deriv(2)(0.))
        walls.append([t(left), t(right)])
        fields.append(t)
        fluxes.append(q)
        incoming = q(right)
        left_temp = t(right)-incoming*resistance
    params = dict(source_gradient=gradients, source_curvature=curvatures)
    fractions = np.array([0., .21, .72, 1.])
    offset = sum(c*(t.integ()(h/2)-t.integ()(-h/2)) for c,t,h in zip(capacities,fields,lengths))/np.dot(capacities,lengths)
    previous = None
    for n in ((6,) if degree == 4 else (4, 8, 16, 32)):
        _, modes, faces, traces, rate = solve_drifting_heat_stack_1d(
            lengths, ks, capacities, (.6, -incoming), ncells=n, degree=degree,
            source=means, contact_resistance=resistance, **params)
        heat = reconstruct_heat_stack_flux(faces, lengths, fractions, **params)
        temp = reconstruct_heat_stack_temperature(modes[:, :, 0], faces, lengths, ks, fractions, **params)
        raw_t, raw_q = sample_heat_stack_fields(modes, lengths, ks, fractions)
        points = [(np.arange(n)[:, None]+fractions)*h/n-h/2 for h in lengths]
        exact_t = np.stack([t(x) for t,x in zip(fields,points)])-offset
        exact_q = np.stack([q(x) for q,x in zip(fluxes,points)])
        errors = dict(rate_K_s=abs(rate-drift), trace_K=np.max(abs(traces.numpy()-walls+offset)),
                      temperature_K=np.max(abs(temp.numpy()-exact_t)),
                      flux_W_m2=np.max(abs(heat.numpy()-exact_q)),
                      modal_temperature_K=np.max(abs(raw_t.numpy()-exact_t)),
                      derivative_flux_W_m2=np.max(abs(raw_q.numpy()-exact_q)))
        for key, error in errors.items():
            record_property(f"n{n}_{key}", float(error))
        assert errors["rate_K_s"] < 1e-13 and errors["flux_W_m2"] < 2e-10
        assert max(errors["temperature_K"], errors["trace_K"]) < 2e-10
        if degree == 4:
            assert max(errors.values()) < 2e-10
        if previous is not None:
            for key, expected_order in (("modal_temperature_K", degree+1), ("derivative_flux_W_m2", degree)):
                order = math.log2(previous[key]/errors[key])
                record_property(f"n{n}_{key}_order", order)
                assert order > expected_order-.25
        previous = errors


@pytest.mark.parametrize("kinds", [("robin", "robin"), ("dirichlet", "neumann"), ("neumann", "dirichlet")])
def test_harmonic_cell_powers_against_polynomial_slab(kinds, record_property):
    import numpy as np
    from numpy.polynomial import Polynomial
    from tensordg import solve_harmonic_heat_1d

    n, k, c, omega = 9, 1.7, 2.1, 3.
    field = Polynomial([.4+.2j, -.7+.5j, .9-.4j])
    ambient = Polynomial([-.3j, .8+.2j])
    loss = Polynomial([.5, 1.1])
    bulk = loss*(ambient-field)
    storage = 1j*omega*c*field
    source = storage-k*field.deriv(2)-bulk
    ends = np.array([0., 1.])
    wall_mass = np.array([.3, .7])
    film = np.array([3., 4.])
    inward = np.array([-1, 1])*k*field.deriv()(ends)
    wall_mass[np.array(kinds) == "dirichlet"] = 0.
    stored = 1j*omega*wall_mass*field(ends)
    inputs = inward+stored
    heater = np.array([.45+.3j, -.25+.6j])
    heater[np.array(kinds) == "dirichlet"] = 0.
    primary = inputs-heater
    film_reference = np.where(np.array(kinds) == "robin", primary, 0.)
    values = [dict(robin=field(x)+primary[i]/film[i], dirichlet=field(x), neumann=primary[i])[kind]
              for i, (kind, x) in enumerate(zip(kinds, ends))]
    film[np.array(kinds) != "robin"] = 0.

    def profile(poly):
        return lambda x: sum(a*x**i for i, a in enumerate(poly.coef))

    _, _, flux, surface, powers, wall = solve_harmonic_heat_1d(
        omega, ncells=n, degree=2, conductivity=k, capacity=c, source=profile(source),
        heat_loss_coefficient=profile(loss), bulk_ambient_temperature=profile(ambient),
        wall_types=kinds, wall_values=values, film_coefficients=film, wall_areal_heat_capacity=wall_mass,
        wall_applied_flux=heater,
        return_wall_temperatures=True, return_cell_power=True, return_wall_power=True)
    for name, reference in (("external", inputs), ("storage", stored), ("material", inward),
                            ("applied", heater), ("film", film_reference)):
        error = np.max(abs(wall[name].numpy()-reference))
        record_property("wall_"+name+"_error_W_m2", float(error))
        assert error < 1e-11
    residual = (wall["external"]-wall["material"]-wall["storage"]).abs().max().item()
    record_property("wall_balance_W_m2", residual)
    assert residual < 1e-13
    faces = np.linspace(0., 1., n+1)
    for channel, polynomial in (("bulk", bulk), ("source", source), ("storage", storage)):
        reference = np.diff(polynomial.integ()(faces))
        error = np.max(abs(powers[channel].numpy()-reference))
        record_property(channel+"_max_error_W_m2", float(error))
        assert error < 1e-12
    balance = powers["source"]+powers["bulk"]-flux.diff()-powers["storage"]
    record_property("max_balance_error_W_m2", balance.abs().max().item())
    assert balance.abs().max() < 1e-11
    np.testing.assert_allclose(surface.numpy(), field(ends), atol=1e-12, rtol=0)


@pytest.mark.parametrize("omega", [0., 1e-16])
def test_floating_cell_power_retains_common_heating_rate(omega):
    from tensordg import solve_harmonic_heat_1d

    _, _, flux, powers, wall, rate = solve_harmonic_heat_1d(
        omega, ncells=6, capacity=2., source=2., wall_types=("neumann", "neumann"),
        wall_values=(.2, .5), wall_areal_heat_capacity=(.2, .5),
        return_cell_power=True, return_wall_power=True, return_mean_rate=True)
    torch.testing.assert_close(wall["storage"], rate.new_tensor([.2, .5]))
    torch.testing.assert_close(wall["external"], wall["storage"])
    torch.testing.assert_close(wall["material"], torch.zeros_like(wall["material"]), atol=1e-13, rtol=0)
    torch.testing.assert_close(rate, rate.new_tensor(1.))
    torch.testing.assert_close(powers["storage"], powers["source"], atol=1e-14, rtol=0)
    torch.testing.assert_close(powers["bulk"], torch.zeros_like(powers["bulk"]))
    torch.testing.assert_close(flux, torch.zeros_like(flux), atol=1e-13, rtol=0)


def test_surface_heater_with_zero_film_and_floating_slab():
    from tensordg import solve_harmonic_heat_1d

    _, _, flux, wall, rate = solve_harmonic_heat_1d(
        0., ncells=4, wall_types=("robin", "neumann"), wall_values=(7., .25),
        wall_applied_flux=(1., -.25), return_mean_rate=True, return_wall_power=True)
    torch.testing.assert_close(rate, rate.new_tensor(1.))
    torch.testing.assert_close(wall["film"], rate.new_zeros(2))
    torch.testing.assert_close(wall["material"], rate.new_tensor([1., 0.]))
    torch.testing.assert_close(flux.diff(), rate.new_full((4,), -.25))
    with pytest.raises(ValueError, match="Neumann or Robin"):
        solve_harmonic_heat_1d(1., wall_applied_flux=(1., 0.))


def test_loaded_wall_admittance_recovers_direct_polynomial_solution(record_property):
    from tensordg import (harmonic_heat_wall_admittance_1d, solve_heat_wall_admittance_1d,
                          solve_harmonic_heat_1d)

    omega, cap, k = 2.5, 1.4, .8
    def field(x):
        return .7+.2j+(.4-.3j)*x+(.1+.5j)*x**2
    def source(x):
        return 1j*omega*cap*field(x)-2*k*(.1+.5j)
    ends = torch.tensor([0., 1.], dtype=torch.float64)
    exact = field(ends)
    material = k*(.4-.3j+2*(.1+.5j)*ends)*ends.new_tensor([-1., 1.])
    wall_mass = ends.new_tensor([.3, .7])
    supply = material+1j*omega*wall_mass*exact
    heater = exact.new_tensor([.6j, 1.-.2j])
    values = (exact[0]+(supply[0]-heater[0])/2., supply[1]-heater[1])
    options = dict(wall_types=("robin","neumann"), wall_values=values,
                   film_coefficients=(2.,0.), wall_areal_heat_capacity=wall_mass, wall_applied_flux=heater)
    slab = dict(ncells=8, conductivity=k, capacity=cap, source=source)
    _, _, _, direct, powers = solve_harmonic_heat_1d(
        omega, **slab, **options, return_wall_temperatures=True, return_wall_power=True)
    y, offset = harmonic_heat_wall_admittance_1d(omega, **slab)
    surface, flux, external = solve_heat_wall_admittance_1d(
        y, **options, flux_offset=offset, angular_frequency=omega, return_external_flux=True)
    for name, reduced, full, reference in (("temperature_K",surface,direct,exact),
                                            ("material_W_m2",flux,powers["material"],material),
                                            ("external_W_m2",external,powers["external"],supply)):
        for comparison, target in (("direct",full), ("analytic",reference)):
            error = (reduced-target).abs().max().item()
            record_property(comparison+"_"+name,error)
            assert error < 1e-11


def test_admittance_stack_with_independent_bulk_reservoir_profiles(record_property):
    import numpy as np
    from numpy.polynomial import Polynomial
    from tensordg import harmonic_heat_wall_admittance_1d, compose_heat_wall_admittances_1d

    from tensordg import solve_harmonic_heat_layer_1d

    layer_cases = []
    omega, resistance = 2.3, .24
    left, flowing = .75+.3j, -.4+.2j
    matrices, offsets, walls, powers = [], [], [], []
    for length, k, c, loss, curvature, heater in (
            (.43, 1.3, .9, 2., .32+.16j, None), (1.25, 2., 1.6, .7, -.41j, .45-.23j)):
        # In local physical coordinates k*T'' + S + H*(Ta-T) = i*omega*C*T.
        # Continue T with a finite contact jump and q with continuous flux.
        field = Polynomial([left, -flowing/k, curvature])
        ambient = field + (1j*omega*c*field-k*field.deriv(2)-(heater or 0.))/loss
        def reservoir(x, ambient=ambient):
            return sum(a*x**i for i, a in enumerate(ambient.coef))
        options = dict(ncells=7, degree=2, thickness=length, conductivity=k,
                       capacity=c, heat_loss_coefficient=loss)
        layer_cases.append((dict(**options, source=heater, bulk_ambient_temperature=reservoir),
                            {"source": Polynomial([heater or 0.]), "bulk": loss*(ambient-field),
                             "storage": 1j*omega*c*field}))
        y, b = harmonic_heat_wall_admittance_1d(
            omega, **options, source=heater, bulk_ambient_temperature=reservoir)
        torch.testing.assert_close(y, harmonic_heat_wall_admittance_1d(omega, **options), rtol=0, atol=0)
        matrices.append(y)
        offsets.append(b)
        right, leaving = field(length), -k*field.deriv()(length)
        walls.append([left, right])
        powers.append([flowing, -leaving])
        left, flowing = right-resistance*leaving, leaving
    _, temperature, flux = compose_heat_wall_admittances_1d(
        matrices, resistance, layer_flux_offsets=offsets, wall_temperatures=(walls[0][0], walls[-1][1]))
    for name, result, reference in (("temperature_K",temperature,walls), ("inward_flux_W_m2",flux,powers)):
        error = (result-result.new_tensor(reference)).abs().max().item()
        record_property("max_"+name, error)
        assert error < 1e-11
    net_volume_power = 0j
    for i, (params, polynomials) in enumerate(layer_cases):
        _, _, face_flux, cell = solve_harmonic_heat_layer_1d(
            omega, temperature[i], **params, return_cell_power=True)
        faces = np.linspace(0., params["thickness"], params["ncells"]+1)
        for channel, poly in polynomials.items():
            exact = np.diff(poly.integ()(faces))
            error = np.max(abs(cell[channel].numpy()-exact))
            record_property(f"layer{i+1}_{channel}_error_W_m2", float(error))
            assert error < 1e-12
        net = cell["storage"]-cell["source"]-cell["bulk"]
        residual = (net+face_flux.diff()).abs().max().item()
        record_property(f"layer{i+1}_cell_balance_W_m2", residual)
        assert residual < 1e-11
        net_volume_power += net.sum().item()
    # Internal contact powers cancel, so only the two exterior faces remain.
    residual = abs(net_volume_power - flux[0, 0].item() - flux[-1, 1].item())
    record_property("stack_balance_W_m2", residual)
    assert residual < 1e-11


def test_affine_bulk_reservoir_stack_contact_refinement(record_property):
    import numpy as np
    from scipy.integrate import quad
    from scipy.linalg import expm
    from tensordg import compose_heat_wall_admittances_1d, harmonic_heat_wall_admittance_1d

    from tensordg import integrate_heat_stack_energy, sample_heat_stack_fields, solve_harmonic_heat_layer_1d

    from tensordg import harmonic_cell_pairing, harmonic_cell_conduction_form, reconstruct_modal_heat_flux, reconstruct_modal_heat_temperature

    references = []
    omega, resistance = 3.2, .27
    layers = [(.65, 1.7, 1.3, 1.1, .6-.3j, -.35+.4j),
              (.85, 1.2, .8, 2.2, -.2+.7j, .55-.1j)]
    left = .4+.6j
    temperature, rightward = left, -.35+.1j
    # Propagate [T, q, 1, x] exactly for an affine reservoir Ta=a+b*x
    # using q=-k*T' and q'=H*Ta-(H+i*omega*C)*T. Restart x at each layer.
    for i, (length, k, c, h, a, b) in enumerate(layers):
        matrix = np.array([[0., -1/k, 0., 0.], [-(h+1j*omega*c), 0., h*a, h*b],
                           [0., 0., 0., 0.], [0., 0., 1., 0.]])
        initial = [temperature, rightward, 1., 0.]
        references.append((matrix, initial))
        temperature, rightward, _, _ = expm(length*matrix) @ initial
        if i == 0:
            contact_rate = rightward
            traces = np.array([temperature, temperature-resistance*rightward])
            temperature = traces[1]
    # The control volume includes the contact and cuts both materials off-grid.
    portions = [(layers[0][0]/3, layers[0][0]), (0., 2*layers[1][0]/3)]
    exact_energy = 0j
    ambient_power = []
    for layer, (west, east), (matrix, initial) in zip(layers, portions, references):
        _, _, c, h, a, b = layer
        ambient = h*(a*(east-west) + b*(east**2-west**2)/2)
        ambient_power.append(ambient)
        # Analytically integrate q' = H*Ta - (H+iωC)*T.
        q_change = ((expm(east*matrix)-expm(west*matrix)) @ initial)[1]
        exact_energy += c*(ambient-q_change)/(h+1j*omega*c)
    exact_forms = []
    for (length, k, *_), (matrix, initial) in zip(layers, references):
        exact_forms.append(quad(lambda x: abs((expm(x*matrix) @ initial)[1])**2/(2*k), 0., length)[0])
    exact_contact_form = resistance*abs(contact_rate)**2/2
    previous = None
    for n in (4, 8, 16, 32):
        matrices, offsets = [], []
        for length, k, c, h, a, b in layers:
            y, offset = harmonic_heat_wall_admittance_1d(
                omega, ncells=n, degree=2, thickness=length, conductivity=k, capacity=c,
                heat_loss_coefficient=h, bulk_ambient_temperature=lambda x: a+b*x)
            matrices.append(y)
            offsets.append(offset)
        _, surfaces, inward = compose_heat_wall_admittances_1d(
            matrices, resistance, layer_flux_offsets=offsets, wall_temperatures=(left, temperature))
        contact = np.array([surfaces[0, 1].item(), surfaces[1, 0].item()])
        transfer = np.array([-inward[0, 1].item(), inward[1, 0].item()])
        errors = {"temperature_K": max(abs(contact-traces)), "flux_W_m2": max(abs(transfer-contact_rate))}
        contact_form = ((contact[0]-contact[1])*transfer[0].conjugate()).real/2
        errors["contact_form_W_K_m2"] = abs(contact_form-exact_contact_form)
        bulk_forms = []
        reservoir_input, reactive_storage = 0j, 0j
        fractions = np.array([.12, .37, .84])
        interior_errors = np.zeros(2)
        layer_modes, cut_fluxes, conservative_cuts = [], [], []
        exchange = 0j
        for i, (length, k, c, h, a, b) in enumerate(layers):
            centers, modes, faces = solve_harmonic_heat_layer_1d(
                omega, surfaces[i], ncells=n, degree=2, thickness=length, conductivity=k,
                capacity=c, heat_loss_coefficient=h, bulk_ambient_temperature=lambda x: a+b*x)
            torch.testing.assert_close(faces[[0, -1]], inward[i]*faces.new_tensor([1, -1]), atol=1e-10, rtol=0)
            field, flux = sample_heat_stack_fields(modes[None], [length], k, fractions)
            layer_modes.append(modes)
            west, east = portions[i]
            energy = integrate_heat_stack_energy(modes[None], [length], c, west, east).item()
            exchange += ambient_power[i] - h*energy/c
            cut_position = west if i == 0 else east
            coordinate = n*cut_position/length
            cell = math.floor(coordinate)
            _, cut_flux = sample_heat_stack_fields(modes[None], [length], k, [coordinate-cell])
            cut_fluxes.append(cut_flux[0, cell, 0].item())
            width = length/n
            # The affine ambient field has only constant and linear Legendre modes.
            net_power = -(h+1j*omega*c)*modes
            net_power[:, 0] += h*(a+b*centers)
            net_power[:, 1] += h*b*width/2
            ambient_modes = torch.stack((a+b*centers, torch.full_like(centers, width/2)*b), dim=1)
            temperature_norm = harmonic_cell_pairing(modes, modes, width).sum().item()
            ambient_pairing = harmonic_cell_pairing(modes, ambient_modes, width).sum().item()
            reservoir_input += h*(ambient_pairing-temperature_norm)
            reactive_storage += 1j*omega*c*temperature_norm
            cell_forms = harmonic_cell_conduction_form(faces, net_power, width, k)
            assert (cell_forms >= 0).all()
            bulk_forms.append(cell_forms.sum().item())
            errors[f"layer{i}_form_W_K_m2"] = abs(bulk_forms[-1]-exact_forms[i])
            local_flux = reconstruct_modal_heat_flux(faces, net_power, width, [0., coordinate-cell, 1.])
            torch.testing.assert_close(local_flux[:, 0], faces[:-1], rtol=0, atol=1e-13)
            torch.testing.assert_close(local_flux[:, -1], faces[1:], rtol=0, atol=1e-13)
            conservative_cuts.append(local_flux[cell, 1].item())
            cut_matrix, cut_initial = references[i]
            exact_cut = (expm(cut_position*cut_matrix) @ cut_initial)[1]
            errors[f"layer{i}_reconstructed_flux_W_m2"] = abs(conservative_cuts[-1]-exact_cut)
            positions = (np.arange(n)[:, None]+fractions)*length/n
            np.testing.assert_allclose(centers.numpy(), (np.arange(n)+.5)*length/n)
            matrix, initial = references[i]
            exact = np.array([(expm(x*matrix) @ initial)[:2] for x in positions.ravel()])
            recovered = reconstruct_modal_heat_temperature(modes[:, 0], faces, net_power, width, k,
                                                           np.r_[fractions, 0., 1.])
            errors[f"layer{i}_recovered_temperature_K"] = abs(recovered[:, :-2].numpy().ravel()-exact[:, 0]).max()
            interface_value = recovered[-1, -1] if i == 0 else recovered[0, -2]
            errors[f"layer{i}_recovered_trace_K"] = abs(interface_value.item()-traces[i])
            observed = np.column_stack((field.numpy().ravel(), flux.numpy().ravel()))
            interior_errors = np.maximum(interior_errors, abs(observed-exact).max(axis=0))
        wall_input = (left.conjugate()*inward[0, 0] + temperature.conjugate()*inward[-1, 1]).item()/2
        # Integration by parts for q' = H(Ta-T)-iωCT with the two exterior
        # inward fluxes and the real contact contribution R|q_contact|²/2.
        form_residual = wall_input + reservoir_input - sum(bulk_forms) - contact_form - reactive_storage
        record_property(f"n{n}_form_balance_imag_W_K_m2", abs(form_residual.imag))
        assert abs(form_residual.imag) < 1e-10
        errors["form_balance_real_W_K_m2"] = abs(form_residual.real)
        errors["total_form_W_K_m2"] = abs(sum(bulk_forms)+contact_form-sum(exact_forms)-exact_contact_form)
        errors.update(interior_temperature_K=interior_errors[0], interior_flux_W_m2=interior_errors[1])
        stored = integrate_heat_stack_energy(torch.stack(layer_modes), [row[0] for row in layers],
                                             [row[2] for row in layers], portions[0][0],
                                             layers[0][0]+portions[1][1]).item()
        energy_error = abs(stored-exact_energy)
        power_error = abs(cut_fluxes[0]-cut_fluxes[1]+exchange-1j*omega*stored)
        residual = abs(conservative_cuts[0]-conservative_cuts[1]+exchange-1j*omega*stored)
        record_property(f"n{n}_reconstructed_balance_W_m2", residual)
        assert residual < 1e-10
        record_property(f"n{n}_interval_energy_error_J_m2", energy_error)
        record_property(f"n{n}_interval_power_residual_W_m2", power_error)
        if previous is not None:
            record_property(f"n{n}_interval_energy_order", math.log2(previous_energy/energy_error))
            record_property(f"n{n}_interval_power_order", math.log2(previous_power/power_error))
            assert energy_error < previous_energy/2 and power_error < previous_power/2
        previous_energy, previous_power = energy_error, power_error
        balance = abs(transfer[0]-transfer[1])
        jump = abs(contact[0]-contact[1]-resistance*transfer[0])
        record_property(f"n{n}_flux_balance_W_m2", float(balance))
        record_property(f"n{n}_jump_residual_K", float(jump))
        assert balance < 1e-11 and jump < 1e-11
        for name, error in errors.items():
            record_property(f"n{n}_{name}", float(error))
            if previous is not None:
                order = math.log2(previous[name]/error)
                record_property(f"n{n}_{name}_order", order)
                assert order > (1.7 if name == "interior_flux_W_m2" else 2.8)
        previous = errors


def test_interface_power_channels_for_loaded_three_layer_stack(record_property):
    import numpy as np
    from tensordg import compose_heat_wall_admittances_1d, harmonic_heat_wall_admittance_1d

    from tensordg import solve_harmonic_heat_layer_1d, solve_heat_wall_admittance_1d

    layer_options = []
    omega = 3.2
    lengths = [.45, .7, 1.1]
    conductivities = [1.3, .8, 2.1]
    # Linear temperature in each material gives a known constant Fourier flux.
    rates = np.array([.4+.3j, -.2+.5j, .7-.1j])
    through = np.array([.65+.2j, -.4+.15j])
    resistances = np.array([0., .24])
    left = .7-.3j
    walls, matrices, offsets = [], [], []
    for i, (length, k, q) in enumerate(zip(lengths, conductivities, rates)):
        right = left-length*q/k
        walls.append([left, right])
        # Constant bulk ambient with H=1: choose generation so source+exchange
        # equals the harmonic storage pointwise in this linear field.
        options = dict(thickness=length, conductivity=k, capacity=1., ncells=6, degree=2,
                       heat_loss_coefficient=1., bulk_ambient_temperature=.3+.2j,
                       source=lambda x, t=left, slope=q/k: (1+1j*omega)*(t-x*slope)-(.3+.2j))
        layer_options.append(options)
        matrix, offset = harmonic_heat_wall_admittance_1d(omega, **options)
        matrices.append(matrix)
        offsets.append(offset)
        if i < len(through):
            left = right-resistances[i]*through[i]
    walls = np.array(walls)
    traces = np.column_stack((walls[:-1, 1], walls[1:, 0]))
    film = np.array([[.85, .4], [.6, 1.2]])
    ambient = np.array([[.4j, .6], [-.3j, .8-.5j]])
    capacity = np.array([[.45, .6], [.3, .7]])
    exchange = film*(ambient-traces)
    stored = 1j*omega*capacity*traces
    source = stored-exchange+np.column_stack((through-rates[:-1], rates[1:]-through))
    contact_options = dict(angular_frequency=omega,
        layer_flux_offsets=offsets, interface_left_heat_sources=source[:,0], interface_heat_sources=source[:,1],
        interface_left_areal_heat_capacity=capacity[:,0], interface_areal_heat_capacity=capacity[:,1],
        interface_left_film_coefficients=film[:,0], interface_film_coefficients=film[:,1],
        interface_left_ambient_temperatures=ambient[:,0], interface_ambient_temperatures=ambient[:,1],
        return_flux_offset=True)
    y, offset = compose_heat_wall_admittances_1d(matrices, resistances, **contact_options)
    outer_t = walls[[0,-1],[0,1]]
    outer_q = rates[[0,-1]]*np.array([1., -1.])
    outer_mass = np.array([.42, .57])
    outer_film = np.array([1.3, .85])
    heaters = np.array([.2+.3j, -.4j])
    outer_stored = 1j*omega*outer_mass*outer_t
    air = outer_t+(outer_q+outer_stored-heaters)/outer_film
    ends, _, wall_power = solve_heat_wall_admittance_1d(
        y, ("robin", "robin"), air, flux_offset=offset, angular_frequency=omega,
        wall_areal_heat_capacity=outer_mass, film_coefficients=outer_film,
        wall_applied_flux=heaters, return_wall_power=True)
    _, solved_t, inward, _, power = compose_heat_wall_admittances_1d(
        matrices, resistances, wall_temperatures=ends, **contact_options, return_interface_power=True)
    for name, reference in (("storage",outer_stored), ("film",outer_film*(air-outer_t)),
                            ("material",outer_q), ("external",outer_q+outer_stored)):
        error = np.max(abs(wall_power[name].numpy()-reference))
        record_property("outer_"+name+"_error_W_m2", float(error))
        assert error < 1e-11
    for name, reference in (("source",source), ("film",exchange), ("storage",stored), ("through",through)):
        error = np.max(abs(power[name].numpy()-reference))
        record_property(name+"_max_error_W_m2", float(error))
        assert error < 1e-11
    residual = (power["source"]+power["film"]-power["storage"]).sum(1)-inward[:-1,1]-inward[1:,0]
    record_property("interface_balance_W_m2", residual.abs().max().item())
    assert residual.abs().max() < 1e-12

    supplied = (wall_power["applied"]+wall_power["film"]).sum() + (power["source"]+power["film"]).sum()
    accumulated = wall_power["storage"].sum() + power["storage"].sum()
    for i, options in enumerate(layer_options):
        _, _, _, cell = solve_harmonic_heat_layer_1d(
            omega, solved_t[i], **options, return_cell_power=True)
        supplied += (cell["source"]+cell["bulk"]).sum()
        accumulated += cell["storage"].sum()
    global_balance = abs(supplied-accumulated).item()
    record_property("whole_stack_balance_W_m2", global_balance)
    assert global_balance < 1e-11


def test_off_grid_energy_across_discontinuous_layers():
    from tensordg import integrate_heat_stack_energy

    # One P2 cell per layer, independent local Legendre coefficients.
    u = torch.tensor([[[1., 2., .5]], [[-3., 1., -.8]]], dtype=torch.float64)
    lengths, capacities = [.4, .7], [2., 4.]
    def primitive(xi, a, b, c):
        return a*xi + b*xi**2/2 + c*(xi**3-xi)/2
    exact = (2*.2*(primitive(1, 1, 2, .5)-primitive(-.5, 1, 2, .5))
             + 4*.35*(primitive(0, -3, 1, -.8)-primitive(-1, -3, 1, -.8)))
    value = integrate_heat_stack_energy(u, lengths, capacities, .1, .75)
    assert abs(value.item()-exact) < 1e-14
    assert integrate_heat_stack_energy(u, lengths, capacities, .4, .4).item() == 0.


def test_harmonic_off_grid_energy_refinement(record_property):
    import cmath
    from tensordg import integrate_heat_stack_energy, solve_harmonic_heat_layer_1d

    k, capacity, omega = .6, 2.3, 4.
    a, b = .173, .717
    wave = cmath.sqrt(1j*omega*capacity/k)
    # Dirichlet data chosen from T=exp(-wave*x).
    exact = capacity*(cmath.exp(-wave*a)-cmath.exp(-wave*b))/wave
    previous = None
    for cells in (8, 16, 32, 64):
        _, modes, _ = solve_harmonic_heat_layer_1d(
            omega, (1., cmath.exp(-wave)), ncells=cells, degree=2, conductivity=k, capacity=capacity)
        energy = integrate_heat_stack_energy(modes[None], [1.], capacity, a, b)
        error = abs(energy.item()-exact)
        record_property(f"cells{cells}_error_J_m2", error)
        if previous is not None:
            record_property(f"cells{cells}_order", math.log2(previous/error))
            assert error < previous/4
        previous = error


def test_modal_flux_reconstruction_uses_faces_for_mean_power():
    import numpy as np
    from tensordg import reconstruct_modal_heat_flux

    power = np.array([[8., 2., -1., .2, .4], [-3., .2, .7, -.1, -.5]])
    faces = torch.tensor([2., -.4, .9], dtype=torch.float64)
    width = .37
    fractions = np.array([0., .33, .85, 1.])
    flux = reconstruct_modal_heat_flux(faces, power, width, fractions)
    for cell, modes in enumerate(power.copy()):
        modes[0] = (faces[cell+1]-faces[cell]).item()/width
        primitive = np.polynomial.legendre.legint(modes, lbnd=-1)
        exact = faces[cell].item() + width/2*np.polynomial.legendre.legval(2*fractions-1, primitive)
        np.testing.assert_allclose(flux[cell].numpy(), exact, rtol=0, atol=1e-14)


def test_modal_temperature_preserves_complex_means_and_fourier_law():
    import numpy as np
    from tensordg import reconstruct_modal_heat_temperature

    means = torch.tensor([2.+.3j, -1.+.9j], dtype=torch.complex128)
    faces = torch.tensor([1.+.2j, .3-.5j, -.4+.8j], dtype=means.dtype)
    power = torch.tensor([[0., .2, -.4, .7], [0., -.6, .8, .3]], dtype=means.dtype)
    k = [1.5, .8]
    nodes, weights = np.polynomial.legendre.leggauss(5)
    f = (1+nodes)/2
    recovered = reconstruct_modal_heat_temperature(means, faces, power, .6, k, f)
    torch.testing.assert_close(recovered @ means.new_tensor(weights/2), means, atol=1e-14, rtol=0)
    # Differentiate the recovered polynomial independently at both boundaries.
    # Degree-five temperature needs six samples for exact differentiation.
    points = np.linspace(0., 1., 6)
    u = reconstruct_modal_heat_temperature(means, faces, power, .6, k, points)
    for cell in range(len(means)):
        poly = np.polynomial.Polynomial.fit(points, u[cell].numpy(), 5)
        np.testing.assert_allclose(-k[cell]/.6*poly.deriv()(np.array([0.,1.])), faces[cell:cell+2].numpy(), atol=2e-12)


def test_harmonic_cell_conduction_form_for_linear_flux():
    from tensordg import harmonic_cell_conduction_form

    faces = torch.tensor([2+1j, -.3+.5j, 1.-.7j], dtype=torch.complex128)
    width = .37
    k = torch.tensor([1.4, .8], dtype=torch.float64)
    west, east = faces[:-1], faces[1:]
    exact = width/(6*k)*(west.abs()**2 + east.abs()**2 + (west*east.conj()).real)
    for phase in (1., 1j):
        observed = harmonic_cell_conduction_form(phase*faces, torch.zeros((2, 1)), width, k)
        torch.testing.assert_close(observed, exact, rtol=1e-14, atol=1e-15)


def test_harmonic_cell_pairing_against_physical_quadrature():
    import numpy as np
    from tensordg import harmonic_cell_pairing

    test = torch.tensor([[1+.7j, -.2j, .5+.3j], [-.5, 1j, .1]], dtype=torch.complex128)
    load = torch.tensor([[.6j, 1.3], [.4+.8j, 1.2j]], dtype=torch.complex128)
    width = .23
    xi, weights = np.polynomial.legendre.leggauss(4)
    exact = []
    for a, b in zip(test.numpy(), load.numpy()):
        product = np.polynomial.legendre.legval(xi, a).conj()*np.polynomial.legendre.legval(xi, b)
        exact.append(width/4*np.dot(weights, product))
    actual = harmonic_cell_pairing(test, load, width)
    np.testing.assert_allclose(actual.numpy(), exact, atol=1e-15, rtol=0)
    torch.testing.assert_close(harmonic_cell_pairing(load, test, width), actual.conj())


def test_wall_and_contact_heater_reciprocity(record_property):
    """Exchange wall drive and observation ports in a storing, lossy contact."""
    import numpy as np
    from scipy.linalg import expm
    from tensordg import compose_heat_wall_admittances_1d, harmonic_heat_wall_admittance_1d

    omega, resistance = 2.7, .23
    layers = [(.41, 1.3, 2.1, .8), (.67, 2.2, 1.1, .3)]
    film = (.6, 1.2)
    storage = (.35, .11)
    sheets = [np.array([[1., 0.], [-h-1j*omega*c, 1.]]) for h, c in zip(film, storage)]
    bulk = [expm(length*np.array([[0., -1/k], [-h-1j*omega*c, 0.]])) for length, k, c, h in layers]
    jump = np.array([[1., -resistance], [0., 1.]])
    transfer = bulk[1] @ sheets[1] @ jump @ sheets[0] @ bulk[0]
    # Each transfer has determinant one, so both cross admittances equal 1/B.
    reference = 1/transfer[0, 1]
    heaters = (.45+.2j, -.16+.5j)
    previous = None
    for n in (4, 8, 16, 32):
        matrices = [harmonic_heat_wall_admittance_1d(
            omega, ncells=n, degree=2, thickness=length, conductivity=k,
            capacity=c, heat_loss_coefficient=h) for length, k, c, h in layers]
        options = dict(angular_frequency=omega, interface_left_areal_heat_capacity=storage[0],
                       interface_areal_heat_capacity=storage[1], interface_left_film_coefficients=film[0],
                       interface_film_coefficients=film[1])
        _, _, heated = compose_heat_wall_admittances_1d(
            matrices, resistance, wall_temperatures=(0., 0.), interface_left_heat_sources=heaters[0],
            interface_heat_sources=heaters[1], **options)
        cross = []
        heater_errors = []
        for side, drive in enumerate(((1., 0.), (0., 1.))):
            _, traces, inward = compose_heat_wall_admittances_1d(
                matrices, resistance, wall_temperatures=drive, **options)
            cross.append(inward[1-side, 1-side].item())
            # Unconjugated thermal reciprocity maps both contact heaters to the
            # grounded wall flux via the traces from a unit drive at that wall.
            heater_errors.append(abs((heated[side, side] + heaters[0]*traces[0, 1]
                                      + heaters[1]*traces[1, 0]).item()))
        error = max(abs(value-reference) for value in cross)
        mismatch = abs(cross[0]-cross[1])
        record_property(f"n{n}_cross_error_W_m2_K", error)
        record_property(f"n{n}_reciprocity_W_m2_K", mismatch)
        record_property(f"n{n}_heater_reciprocity_W_m2", max(heater_errors))
        assert mismatch < 1e-11 and max(heater_errors) < 1e-11
        if previous is not None:
            order = math.log2(previous/error)
            record_property(f"n{n}_cross_order", order)
            assert order > 3.5
        previous = error


def test_affine_bulk_source_to_wall_reciprocity(record_property):
    import numpy as np
    from scipy.linalg import expm
    from tensordg import (compose_heat_wall_admittances_1d, harmonic_heat_wall_admittance_1d,
                          reciprocal_bulk_wall_flux, solve_harmonic_heat_layer_1d)

    omega, resistance = 2.7, .23
    layers = [(.41, 1.3, 2.1, .8, .45+.2j, -.3+.7j),
              (.67, 2.2, 1.1, .3, -.16+.5j, .8-.2j)]
    # Propagate [T,q,1,x] in each layer. Reset local x after the resistive jump.
    transfer = np.eye(4)
    for i, (length, k, c, h, a, b) in enumerate(layers):
        generator = np.array([[0, -1/k, 0, 0], [-h-1j*omega*c, 0, a, b],
                              [0, 0, 0, 0], [0, 0, 1, 0]])
        transfer = expm(length*generator) @ transfer
        if i == 0:
            interface = np.eye(4)
            interface[0, 1] = -resistance
            interface[3] = 0
            transfer = interface @ transfer
    left = -transfer[0, 2]/transfer[0, 1]
    exact = np.array([left, -(transfer @ [0, left, 1, 0])[1]])
    previous = None
    for n in (4, 8, 16, 32):
        options = [dict(ncells=n, degree=2, thickness=length, conductivity=k, capacity=c,
                        heat_loss_coefficient=h) for length, k, c, h, _, _ in layers]
        matrices, offsets = zip(*(harmonic_heat_wall_admittance_1d(
            omega, source=lambda x, a=a, b=b: a+b*x, **opt)
            for opt, (_, _, _, _, a, b) in zip(options, layers)))
        _, _, direct = compose_heat_wall_admittances_1d(
            matrices, resistance, layer_flux_offsets=offsets, wall_temperatures=(0., 0.))
        predicted = []
        for drive in ((1., 0.), (0., 1.)):
            _, traces, _ = compose_heat_wall_admittances_1d(matrices, resistance, wall_temperatures=drive)
            flux = 0j
            for opt, walls, (length, _, _, _, a, b) in zip(options, traces, layers):
                centers, modes, _ = solve_harmonic_heat_layer_1d(omega, walls, **opt)
                width = length/n
                source = torch.stack((a+b*centers, b*torch.ones_like(centers)*width/2), dim=1)
                flux += reciprocal_bulk_wall_flux(modes, source, width).item()
            predicted.append(flux)
        mismatch = np.max(abs(predicted-direct.diagonal().numpy()))
        error = np.max(abs(predicted-exact))
        record_property(f"n{n}_direct_mismatch_W_m2", float(mismatch))
        record_property(f"n{n}_reference_error_W_m2", float(error))
        assert mismatch < 1e-11
        if previous is not None:
            order = math.log2(previous/error)
            record_property(f"n{n}_order", order)
            assert order > 3.5
        previous = error
