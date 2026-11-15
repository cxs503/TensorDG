import numpy as np, torch
from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh, euler_flux, hllc_normal
from tensordg.euler_2d import vortex_primitive
def exact(t):
    def fn(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(x), torch.as_tensor(y), time=t, strength=5.0,
                                   background_velocity=(1.0,1.0), period=10.0)
        return r.numpy(), u.numpy(), v.numpy(), p.numpy()
    return fn
nx=ny=8; p=1
mesh = straight_box_mesh(nx,ny,1,length=10.0); s = CurvilinearEuler(mesh, degree=p)
U = s.project(exact(0.0))
traces = s.traces(U)
inner, outer = s._face_states(traces, 0, -1)
print("inner shape", inner.shape, "max|inner-outer|", np.abs(inner-outer).max())
g = s.face[(0,-1)]
num = g["magnitude"][...,None]*hllc_normal(inner, outer, g["unit_x"], g["unit_y"])
fx, fy = euler_flux(inner)
phys = g["metric"][...,0,None]*fx + g["metric"][...,1,None]*fy
print("max|num|", np.abs(num).max(), "max|phys|", np.abs(phys).max(), "max|corr|", np.abs(num-phys).max())
# volume magnitude
basis, w = s.basis, s.w
sampled = np.einsum("qk,nklc,rl->nqrc", basis, U, basis)
fxs, fys = euler_flux(sampled)
cxi = s.metric_xi[...,0,None]*fxs + s.metric_xi[...,1,None]*fys
ceta = s.metric_eta[...,0,None]*fxs + s.metric_eta[...,1,None]*fys
div = s.differentiate(cxi,0)+s.differentiate(ceta,1)
print("max|div|", np.abs(div).max())
print("metric_xi", s.metric_xi[0,0,0], "J", s.J[0,0,0], "faces mag", g["magnitude"][0,0])
print("neighbours (0,-1)", mesh.neighbours[(0,-1)][:4], "(0,1)", mesh.neighbours[(0,1)][:4])
