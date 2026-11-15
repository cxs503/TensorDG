import numpy as np, torch, types
from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh, euler_flux, hllc_normal
from tensordg.euler_2d import vortex_primitive
from tensordg import euler_dg_2d as E

def exact(t):
    def fn(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(x), torch.as_tensor(y), time=t, strength=5.0,
                                   background_velocity=(1.0,1.0), period=10.0)
        return r.numpy(), u.numpy(), v.numpy(), p.numpy()
    return fn

def rhs_variant(self, state, sign=1.0):
    basis, weights = self.basis, self.w
    sampled = np.einsum("qk,nklc,rl->nqrc", basis, state, basis)
    fx, fy = euler_flux(sampled, self.gamma)
    cxi = self.metric_xi[...,0,None]*fx + self.metric_xi[...,1,None]*fy
    ceta = self.metric_eta[...,0,None]*fx + self.metric_eta[...,1,None]*fy
    div = self.differentiate(cxi,0) + self.differentiate(ceta,1)
    volume = np.einsum("qk,rl,nqrc,q,r->nklc", basis, basis, div, weights, weights)
    traces = self.traces(state); face = np.zeros_like(state)
    for axis in (0,1):
        for side in (-1,1):
            g = self.face[(axis,side)]
            inner, outer = self._face_states(traces, axis, side)
            num = g["magnitude"][...,None]*hllc_normal(inner, outer, g["unit_x"], g["unit_y"], self.gamma)
            fin_x, fin_y = euler_flux(inner, self.gamma)
            phys = g["metric"][...,0,None]*fin_x + g["metric"][...,1,None]*fin_y
            corr = num - phys
            eb = np.ones(self.K) if side>0 else self.signs
            if axis==0: face += np.einsum("k,rl,nrc,r->nklc", eb, basis, corr, weights)
            else: face += np.einsum("l,qk,nqc,q->nklc", eb, basis, corr, weights)
    return self.apply_mass_inverse(-volume + sign*face)

nx=ny=16; p=1; L=10.0
mesh = straight_box_mesh(nx,ny,1,length=L); s = CurvilinearEuler(mesh, degree=p)
U = s.project(exact(0.0))
Ut = torch.as_tensor(U.reshape(nx,ny,p+1,p+1,4))
nodes, weights, basis, derivative, signs = E.legendre_operators(p, max(p+2,3))
fref = E._rhs(Ut, basis, derivative, weights, signs, L/nx, L/ny, (True,True), None, 0.0).numpy().reshape(nx*ny,p+1,p+1,4)
scale = float(np.abs(fref).max())
for sign in (+1.0, -1.0):
    r = rhs_variant(s, U, sign)
    print("sign", sign, "max|r|", np.abs(r).max(), "rel diff vs ref", float(np.abs(r-fref).max())/scale)
print("ref max", scale)
