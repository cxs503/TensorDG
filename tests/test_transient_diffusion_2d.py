import numpy as np
import pytest
import torch

from tensordg import poisson_face_powers_2d, solve_transient_diffusion_2d


INSULATED = dict.fromkeys(("left","right","bottom","top"),"neumann")


@pytest.mark.parametrize("theta",[.5,1.])
def test_layer_storage_face_power_and_restart(theta):
    times = np.array([0.,.01,.025,.08])
    opts = dict(ncells=(4,3),degree=2,capacity=2.3,theta=theta,bc=INSULATED,
                reaction=lambda x,y:3.*(x>.5),neumann_flux={"right":.4})
    x,y,states = solve_transient_diffusion_2d(lambda x,y: 1+x*y,times,**opts)
    _,_,resumed = solve_transient_diffusion_2d(states[1],times[1:],**opts)
    torch.testing.assert_close(states[1:],resumed)
    for i,dt in enumerate(np.diff(times)):
        avg = (1-theta)*states[i]+theta*states[i+1]
        qx,qy = poisson_face_powers_2d(avg,bc=INSULATED,neumann_flux=opts["neumann_flux"])
        loss = (3.*(x>.5))[:,None]*avg[:,:,0,0]/(len(x)*len(y))
        storage = 2.3*(states[i+1]-states[i])[:,:,0,0]/(len(x)*len(y))
        torch.testing.assert_close(storage, -dt*(loss+qx.diff(dim=0)+qy.diff(dim=1)),atol=2e-13,rtol=0.)


def test_mixed_boundary_equilibrium_is_stationary():
    exact = lambda x,y: 1+x+y*y
    opts = dict(ncells=(3,4),degree=2,reaction=2.,forcing=lambda x,y:2*exact(x,y)-2,
                bc=dict(left="dirichlet",right="robin",top="neumann",bottom="neumann"),
                dirichlet_values={"left":exact},robin_films={"right":(2.,lambda x,y:exact(x,y)+.5)},
                neumann_flux={"top":2.})
    _,_,states = solve_transient_diffusion_2d(exact,[.1,.13,.5],**opts)
    torch.testing.assert_close(states,states[0].expand_as(states),rtol=0.,atol=3e-13)


def test_ramped_load_heats_unanchored_mean_on_irregular_grid():
    times = np.array([.1,.17,.2,.5])
    _,_,states = solve_transient_diffusion_2d(1.,times,ncells=(2,3),bc=INSULATED,capacity=2.,
                      forcing=lambda x,y:1+0*x+0*y,neumann_flux={"left":.5},
                      load_amplitude=lambda t:2*t)
    means = states[:,:,:,0,0].mean(dim=(1,2)).numpy()
    np.testing.assert_allclose(means,1+.75*(times**2-times[0]**2),atol=3e-13,rtol=0.)


def test_smooth_capacity_reproduces_linear_temperature_ramp():
    capacity = lambda x,y:1.+x+2*y+x*y
    initial = lambda x,y:x*x+y
    x,y,history = solve_transient_diffusion_2d(initial,[0.,.02,.09],ncells=(3,4),degree=2,
                    capacity=capacity,bc=INSULATED,neumann_flux=dict(right=2.,bottom=-1.,top=1.),
                    forcing=lambda x,y:capacity(x,y)-2)
    change = torch.zeros_like(history[0])
    change[:,:,0,0] = .09
    torch.testing.assert_close(history[-1]-history[0],change,rtol=0.,atol=5e-13)


@pytest.mark.parametrize("invalid",[0.,lambda x,y:x-1.,torch.ones(4)])
def test_invalid_capacity(invalid):
    with pytest.raises(ValueError,match="capacity"):
        solve_transient_diffusion_2d(0.,[0.,.1],capacity=invalid,ncells=(2,3))


def test_weighted_initial_projection_preserves_total_heat():
    # For C=1+xy and T=x²y, integral(C*T) = 1/6+1/12.
    capacity = lambda x,y:1+x*y
    x,y,states = solve_transient_diffusion_2d(lambda x,y:x.square()*y,[0.,.01,.04],
                                    ncells=(3,4),degree=1,capacity=capacity,bc=INSULATED)
    # Integrate each product with the four modal basis functions analytically.
    hx,hy = 1/len(x),1/len(y)
    moments = torch.stack(torch.broadcast_tensors(1+x[:,None]*y,x[:,None]*hy/6,
                          hx*y[None,:]/6,torch.tensor(hx*hy/36,dtype=x.dtype)),dim=-1)
    heat = (states.flatten(-2)*moments).sum(dim=(1,2,3))*hx*hy
    torch.testing.assert_close(heat,torch.full_like(heat,.25),atol=2e-13,rtol=0.)


def test_rannacher_matches_split_solve_and_stage_means():
    opts = dict(ncells=(3,2),degree=1,load_amplitude=lambda t:1+t*t,
                dirichlet_values={"left":1.},conductivity=(2.,1.,.2))
    initial = lambda x,y:x*y+.2
    _,_,fields,means = solve_transient_diffusion_2d(initial,[.1,.16,.25],rannacher=True,
                                                       return_interval_means=True,**opts)
    _,_,be = solve_transient_diffusion_2d(initial,[.1,.13,.16],theta=1.,**opts)
    _,_,cn = solve_transient_diffusion_2d(be[-1],[.16,.25],**opts)
    torch.testing.assert_close(fields[1:],cn)
    torch.testing.assert_close(means[0],be[1:].mean(dim=0))
    torch.testing.assert_close(means[1],cn.mean(dim=0))


@pytest.mark.parametrize("damped",[False,True])
def test_multiple_off_grid_load_jumps_match_independent_restarts(damped):
    jump_times = [.015,.025,.09]
    levels = [.3,1.4,-.5,.2]
    opts = dict(ncells=(3,2),degree=1,dirichlet_values={"left":1.},
                rannacher=damped,return_interval_means=True,return_load_means=True)
    amplitude = lambda t:levels[np.searchsorted(jump_times,t,side="right")]
    time = np.array([0.,.06,.09,.17])
    *_,fields,means,loads = solve_transient_diffusion_2d(
        .3,time,load_amplitude=amplitude,load_jump_times=jump_times,**opts)
    init = fields[0]
    stores,stages,weights = [],[],[]
    for interval,value in zip(([0.,.015],[.015,.025],[.025,.06,.09],[.09,.17]),levels):
        _,_,sol,stage,load = solve_transient_diffusion_2d(init,interval,load_amplitude=lambda t:value,**opts)
        init = sol[-1]
        stores.extend(sol[1:])
        stages.extend(stage)
        weights.extend(load)
    torch.testing.assert_close(fields[1:],torch.stack(stores)[[2,3,4]])
    torch.testing.assert_close(means[0],(.015*stages[0]+.010*stages[1]+.035*stages[2])/.06)
    torch.testing.assert_close(means[1:],torch.stack(stages)[3:])
    np.testing.assert_allclose(loads,[(.015*.3+.01*1.4-.035*.5)/.06,-.5,.2],atol=1e-14)


@pytest.mark.parametrize("jumps",[[0.],[.5,.5],[1.],[float("nan")]])
def test_bad_jump_schedule(jumps):
    with pytest.raises(ValueError,match="jump"):
        solve_transient_diffusion_2d(0.,[0.,1.],load_jump_times=jumps)


@pytest.mark.parametrize("damped", [False, True])
def test_independent_uniform_source_jump_conserves_heat(damped):
    jump = .033
    times = np.array([0., .05, .1])
    _,_,states = solve_transient_diffusion_2d(
        .4,times,ncells=(3,2),capacity=2.,bc=INSULATED,forcing=lambda x,y:1+x*0+y*0,
        load_amplitude=lambda t:.5,source_history=lambda t,x,y:3. if t<jump else -1.,
        load_jump_times=[jump],rannacher=damped)
    heat = .8+.5*times+3*np.minimum(times,jump)-np.maximum(times-jump,0.)
    np.testing.assert_allclose(2*states[:,:,:,0,0].mean(dim=(1,2)).numpy(),heat,atol=1e-13,rtol=0.)


@pytest.mark.parametrize("value",[1j, float("nan"), torch.ones(17)])
def test_invalid_source_history(value):
    with pytest.raises(ValueError,match="source_history"):
        solve_transient_diffusion_2d(0.,[0.,.1],ncells=2,source_history=lambda t,x,y:value)


def test_low_frequency_linear_capacity_energy_at_both_orders():
    from tensordg import solve_low_frequency_diffusion_2d

    # U0=x-a, C=1+x. The constant a enforces integral C*U0=0.
    a = 5/9
    x,y,u,v = solve_low_frequency_diffusion_2d(ncells=(4,3),degree=2,bc=INSULATED,
                   capacity=lambda x,y:1+x,neumann_flux={"left":-1.,"right":1.})
    torch.testing.assert_close(u[:,:,0,0],(x[:,None]-a).expand(-1,len(y)),atol=1e-12,rtol=0.)
    for field in (u,v):
        moments = (1+x[:,None])*field[:,:,0,0]+field[:,:,1,0]/(6*len(x))
        assert abs(moments.mean())<1e-13
    with pytest.raises(ValueError,match="zero net load"):
        solve_low_frequency_diffusion_2d(ncells=3,bc=INSULATED,neumann_flux={"left":1.})


def test_harmonic_affine_field_with_complex_mixed_boundary_data():
    from tensordg import solve_harmonic_diffusion_2d

    omega = 3.
    a,b = .3+.2j,-.4j
    def exact(x,y):
        return 1j+a*x+b*y
    # Constant anisotropic conductivity gives constant exact Fourier flux.
    k = (2.,1.4,.3)
    gx,gy = k[0]*a+k[2]*b,k[2]*a+k[1]*b
    data = dict(bc=dict(left="dirichlet",right="robin",bottom="neumann",top="neumann"),
                dirichlet_values={"left":exact},neumann_flux={"top":gy,"bottom":-gy},
                robin_films={"right":(3.,lambda x,y:exact(x,y)+gx/3)},conductivity=k)
    x,y,u = solve_harmonic_diffusion_2d(omega,ncells=(4,3),degree=2,capacity=lambda x,y:1+x,
                  reaction=.5,forcing=lambda x,y:(.5+1j*omega*(1+x))*exact(x,y),**data)
    torch.testing.assert_close(u[:,:,0,0],exact(x[:,None],y),rtol=0.,atol=1e-11)
    qx,qy = poisson_face_powers_2d(u,**data)
    assert abs(qx*len(y)+gx).max()<1e-10
    assert abs(qy*len(x)+gy).max()<1e-10


def test_insulated_harmonic_uniform_unbalanced_source():
    from tensordg import solve_harmonic_diffusion_2d

    _,_,field = solve_harmonic_diffusion_2d(2.,ncells=3,bc=INSULATED,capacity=3.,
                                             forcing=lambda x,y:1.+.5j)
    assert abs(field[:,:,0,0]-(1.+.5j)/6j).max()<1e-12
    with pytest.raises(ValueError,match="angular frequency"):
        solve_harmonic_diffusion_2d(0.)


def test_harmonic_wall_storage_all_sides_affine_exact():
    from tensordg import solve_harmonic_diffusion_2d

    omega = 2.
    gx,gy = .3+.4j,-.7j
    def exact(x,y):
        return 1.+.2j+gx*x+gy*y
    bc = dict.fromkeys(INSULATED,"robin")
    surface = dict.fromkeys(bc,lambda x,y:.4+x+.5*y)
    outward = dict(left=gx,right=-gx,bottom=gy,top=-gy)
    films = {s:(3.,lambda x,y,s=s: exact(x,y)+(1j*omega*surface[s](x,y)*exact(x,y)-outward[s])/3)
             for s in bc}
    opts = dict(bc=bc,robin_films=films,wall_areal_heat_capacity=surface)
    x,y,u = solve_harmonic_diffusion_2d(omega,ncells=(4,3),degree=2,
                                        forcing=lambda x,y:1j*omega*exact(x,y),**opts)
    torch.testing.assert_close(u[:,:,0,0],exact(x[:,None],y),rtol=0.,atol=1e-12)
    qx,qy = poisson_face_powers_2d(u,temperature_rate=1j*omega*u,**opts)
    assert abs(qx*len(y)+gx).max()<1e-11
    assert abs(qy*len(x)+gy).max()<1e-11


def test_closed_surface_capacity_sets_dc_mean_and_first_order_flux():
    from tensordg import solve_low_frequency_diffusion_2d

    bc = dict(INSULATED,right="robin")
    opts = dict(bc=bc,robin_films={"right":(0.,0.)},wall_areal_heat_capacity={"right":2.})
    # Source and left flux balance. U0 has curvature -1 and zero right derivative.
    x,y,u,v = solve_low_frequency_diffusion_2d(ncells=(6,3),degree=3,forcing=lambda x,y:1.,
                                               neumann_flux={"left":-1.},**opts)
    for field in (u,v):
        right_trace = field[-1,:,:,0].sum(1).mean()
        assert abs(field[:,:,0,0].mean()+2*right_trace)<1e-12
    fx,fy = poisson_face_powers_2d(v,temperature_rate=u,**opts)
    assert abs(fx.diff(dim=0)+fy.diff(dim=1)+u[:,:,0,0]/(len(x)*len(y))).max()<1e-11


def test_storage_flux_preserves_imaginary_rate_for_real_trace():
    coeff = torch.zeros((2,3,2,2),dtype=torch.float64)
    coeff[:,:,0,0] = 1.
    qx,_ = poisson_face_powers_2d(coeff,bc={"right":"robin"},robin_films={"right":(1.,1.)},
                                  wall_areal_heat_capacity={"right":2.},temperature_rate=3j*coeff)
    assert abs(qx[-1]-2j).max()<1e-14


def test_transient_surface_projection_conserves_initial_heat_and_restart():
    # The continuum projection uses both the volume and true wall values.
    # For T=x³, C=1+x and wall capacity 2+y at x=1 the total is 2.95.
    opts = dict(ncells=(4,3),degree=1,capacity=lambda x,y:1+x,
                bc=dict(INSULATED,right="robin"),robin_films={"right":(0.,0.)},
                wall_areal_heat_capacity={"right":lambda x,y:2+y})
    x,y,states = solve_transient_diffusion_2d(lambda x,y:x**3,[0.,.04,.09],**opts)
    for u in states:
        bulk = ((1+x[:,None])*u[:,:,0,0]+u[:,:,1,0]/(6*len(x))).mean()
        trace = u[-1].sum(1)
        wall = ((2+y)*trace[:,0]+trace[:,1]/(6*len(y))).mean()
        assert abs(bulk+wall-2.95)<1e-12
    _,_,restart = solve_transient_diffusion_2d(states[1],[.04,.09],**opts)
    torch.testing.assert_close(restart[-1],states[-1],rtol=0.,atol=1e-12)


@pytest.mark.parametrize("axis",["x","y"])
def test_storing_contact_low_frequency_limit_from_rod_admittance(axis):
    from tensordg import solve_low_frequency_diffusion_2d, solve_harmonic_diffusion_2d

    n = 6
    face = np.zeros((n-1,n))
    face[n//2-1] = 1.
    if axis=="y":
        face = face.T
    r = .35
    surface = np.array([.5,.2])
    drive = np.array([1.,-.3])
    options = dict(ncells=n,degree=3,bc=dict.fromkeys(("top","bottom") if axis=="x" else ("left","right"),"neumann"),
                   contact_resistance={axis:r*face},trace_power={axis:(drive[0]*face,drive[1]*face)},
                   trace_heat_capacity={axis:(surface[0]*face,surface[1]*face)})
    _,_,steady,lag = solve_low_frequency_diffusion_2d(**options)
    # k*A*w*coth(w*L) = k*A/L + iω*C*A*L/3 + O(ω²).
    matrix = 2*np.eye(2)+np.array([[1.,-1.],[-1.,1.]])/r
    expected = np.linalg.solve(matrix,drive)
    derivative = np.linalg.solve(matrix,-(surface+1/6)*expected)
    for field,reference in ((steady,expected),(lag,derivative)):
        oriented = field if axis=="x" else field.permute(1,0,3,2)
        modes = oriented[:,:,:,0].mean(1).numpy()
        traces = [modes[n//2-1].sum(),modes[n//2]@np.array([1.,-1.,1.,-1.])]
        np.testing.assert_allclose(traces,reference,atol=1e-11,rtol=0)
    with pytest.raises(ValueError,match="nonnegative"):
        solve_harmonic_diffusion_2d(1.,**dict(options,trace_heat_capacity={axis:(-face,face)}))


@pytest.mark.parametrize("axis",["x","y"])
def test_initial_contact_traces_and_restart_conserve_stored_heat(axis):
    n = 4
    face = torch.zeros((n-1,n),dtype=torch.float64)
    face[n//2-1] = 1.
    if axis=="y":
        face = face.T
    def initial(x,y):
        t = x if axis=="x" else y
        return torch.where(t<.5,t**2,2+t**2)
    opts = dict(ncells=n,degree=1,bc=INSULATED,contact_resistance={axis:.3*face},
                trace_heat_capacity={axis:(.7*face,.4*face)})
    _,_,fields = solve_transient_diffusion_2d(initial,[0.,.02,.09],**opts)
    for field in fields:
        u = field if axis=="x" else field.permute(1,0,3,2)
        modes = u[:,:,:,0].mean(1)
        energy = modes[:,0].mean()+.7*modes[1].sum()+.4*(modes[2,0]-modes[2,1])
        assert abs(energy-(1/3+1+.7*.25+.4*2.25))<1e-12
    _,_,restart = solve_transient_diffusion_2d(fields[1],[.02,.09],**opts)
    torch.testing.assert_close(restart[-1],fields[-1],atol=1e-12,rtol=0)


@pytest.mark.parametrize("damped",[False,True])
def test_independent_trace_histories_at_off_grid_jumps(damped):
    # A manufactured piecewise-linear temperature on each half slab. Bulk
    # sources carry the local ramp and each trace supplies its own storage
    # plus its signed heat loss to the other side.
    n = 4
    face = np.zeros((n-1,n))
    face[n//2-1] = 1.
    def field(t):
        return np.array([.25+t-1.5*max(0.,t-.04),-.2+2*max(0.,t-.07)])
    def slope(t):
        return np.array([1.-1.5*(t>=.04),2.*(t>=.07)])
    def heaters(t):
        transfer = (field(t)[0]-field(t)[1])/.3
        left,right = [.7,.4]*slope(t)+[transfer,-transfer]
        return {"x":(left*face,right*face)}
    times = [0.,.03,.06,.1,.12]
    x,_,u,heat = solve_transient_diffusion_2d(lambda x,y:torch.where(x<.5,.25+0*x,-.2+0*x),times,
        ncells=n,degree=2,bc=INSULATED,rannacher=damped,contact_resistance={"x":.3*face},
        trace_heat_capacity={"x":(.7*face,.4*face)},trace_power_history=heaters,
        source_history=lambda t,x,y:torch.where(x<.5,slope(t)[0]+0*x,slope(t)[1]+0*x),
        load_jump_times=(.04,.07),return_history_heat=True)
    for j,(start,stop) in enumerate(zip(times,times[1:])):
        # Internal transfer cancels over the domain. Both bulk and trace
        # histories contribute the change in total bulk plus surface energy.
        assert abs(heat[j].sum()-np.dot([1.2,.9],field(stop)-field(start)))<1e-13
    for t,state in zip(times,u):
        expected = torch.where(x[:,None]<.5,field(t)[0]+0*x[:,None],field(t)[1]+0*x[:,None])
        assert abs(state[:,:,0,0]-expected).max()<1e-12
        assert abs(state[:,:,1:,:]).max()<1e-12
        assert abs(state[:,:,:,1:]).max()<1e-12


@pytest.mark.parametrize("reaction",[2.,1e5])
def test_sdirk2_uniform_decay_amplification_and_stage_means(reaction):
    dt = .2
    gamma = 1-1/np.sqrt(2.)
    z = dt*reaction
    first = 1/(1+gamma*z)
    final = (1-(1-2*gamma)*z)/(1+gamma*z)**2
    _,_,u,mean = solve_transient_diffusion_2d(1.,[0.,dt],ncells=3,bc=INSULATED,
                    reaction=reaction,method="sdirk2",return_interval_means=True)
    assert abs(u[-1,:,:,0,0]-final).max()<2e-13
    assert abs(mean[0,:,:,0,0]-((1-gamma)*first+gamma*final)).max()<2e-13
    assert abs(u[-1,:,:,0,0]-1+z*mean[0,:,:,0,0]).max()<1e-12


def test_sdirk2_counts_short_pulse_and_linear_load_inside_one_output_interval():
    # The entire pulse occurs between outputs. Linear time dependence is
    # integrated exactly by this second-order scheme, even at its cutoff.
    def pulse(t,x,y):
        return 4.+3*t if .031<=t<.042 else 0.
    _,_,u,amplitude,heat = solve_transient_diffusion_2d(0.,[0.,.1],ncells=(3,4),bc=INSULATED,
             forcing=lambda x,y:torch.ones_like(x+y),load_amplitude=lambda t:2*t,source_history=pulse,
             method="sdirk2",load_jump_times=(.031,.042),return_load_means=True,return_history_heat=True)
    exact = .011*(4+3*(.031+.042)/2)
    assert abs(heat.sum()-exact)<1e-13
    assert abs(amplitude.item()-.1)<1e-13
    assert abs(u[-1,:,:,0,0]-(exact+.01)).max()<1e-12
