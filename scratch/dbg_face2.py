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
basis, weights = s.basis, s.w
traces = s.traces(U)
face = np.zeros_like(U)
for axis in (0,1):
    for side in (-1,1):
        g = s.face[(axis,side)]
        inner, outer = s._face_states(traces, axis, side)
        num = g["magnitude"][...,None]*hllc_normal(inner, outer, g["unit_x"], g["unit_y"], s.gamma)
        fx, fy = euler_flux(inner)
        phys = g["metric"][...,0,None]*fx + g["metric"][...,1,None]*fy
        corr = num - phys
        eb = np.ones(s.K) if side>0 else s.signs
        if axis==0:
            contrib = np.einsum("k,rl,nrc,r->nklc", eb, basis, corr, weights)
        else:
            contrib = np.einsum("l,qk,nqc,q->nklc", eb, basis, corr, weights)
        print(axis, side, "max|corr|", np.abs(corr).max(), "max|contrib|", np.abs(contrib).max())
        face += contrib
print("max|face|", np.abs(face).max())
r = s.rhs(U)
print("max|rhs|", np.abs(r).max())
