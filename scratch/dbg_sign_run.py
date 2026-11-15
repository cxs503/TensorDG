import numpy as np, torch, math
from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh, euler_flux, hllc_normal
from tensordg.euler_2d import vortex_primitive
from tensordg import euler_dg_2d as E

def prim(t):
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

n=16; p=1
for sign in (+1.0,-1.0):
    mesh = straight_box_mesh(n,n,1,length=10.0); s = CurvilinearEuler(mesh, degree=p, cfl=0.2)
    U = s.project(prim(0.0))
    dt = s.stable_dt(U)
    ok=True
    for k in range(int(0.5/dt)+1):
        U = s.ssp_rk3_step(U, min(dt, 0.5-k*dt) if k*dt<0.5 else 0.0)
        if not np.all(np.isfinite(U)) or np.abs(U).max()>1e3: ok=False; break
    print("sign",sign,"stable",ok, "err" , s.relative_errors(U, prim(0.5)) if ok else None)

# reference Cartesian solver, same case
def initial(x, y):
    r,u,v,p = vortex_primitive(x, y, time=0.0, strength=5.0, background_velocity=(1.0,1.0), period=10.0)
    return r,u,v,p
res = E.solve_euler_dg_2d(initial, ncells=n, degree=p, final_time=0.5, domain=((0.0,10.0),(0.0,10.0)),
                          cfl=0.2, periodic=(True,True), limiter=None)
print("reference keys", list(res.keys())[:12])
coeff = res["coeff"]; nodes, weights, basis, derivative, signs = E.legendre_operators(p, max(p+2,3))
at = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
rho, u, v, pr = E._primitive(at)
re_, ue, ve, pe = vortex_primitive(nodes[:,None], nodes[None,:], time=0.5, strength=5.0,
                                   background_velocity=(1.0,1.0), period=10.0)
rt = torch.as_tensor(re_); 
# broadcast exact over cells
def rel(g, e):
    return float(torch.sqrt(torch.sum((g-e)**2))/torch.sqrt(torch.sum(e**2)))
print("ref err rho", rel(rho, re_.expand_as(rho) if False else torch.broadcast_to(torch.as_tensor(re_), rho.shape)),
      "u", rel(u, torch.broadcast_to(torch.as_tensor(ue), u.shape)))
