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
