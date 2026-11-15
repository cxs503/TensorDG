import numpy as np, warnings
warnings.filterwarnings("ignore")
from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh, euler_flux, hllc_normal

def make_rhs(mode):
    def f(self, state):
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
                left, right = (inner, outer) if side>0 else (outer, inner)
                num = g["magnitude"][...,None]*hllc_normal(left, right, g["unit_x"], g["unit_y"], self.gamma)
                finx, finy = euler_flux(inner, self.gamma)
                phys = g["metric"][...,0,None]*finx + g["metric"][...,1,None]*finy
                corr = num - phys
                eb = np.ones(self.K) if side>0 else self.signs
                if axis==0: face += np.einsum("k,rl,nrc,r->nklc", eb, basis, corr, weights)
                else: face += np.einsum("l,qk,nqc,q->nklc", eb, basis, corr, weights)
        if mode=="full": return self.apply_mass_inverse(-volume + face)
        if mode=="face": return self.apply_mass_inverse(face)
        return self.apply_mass_inverse(-volume)
    return f

for mode in ("full","face"):
  for p in (0,1,2):
    m = straight_box_mesh(8,8,1,length=10.0); s = CurvilinearEuler(m, degree=p, cfl=0.2)
    s.rhs = make_rhs(mode).__get__(s)
    st = s.project(lambda x,y: (1.0,0.3,-0.2,1.0))
    U0 = np.zeros_like(st); U0[:,0,0,:] = st[:,0,0,:]
    rng = np.random.default_rng(0)
    U = U0 + 1e-8*rng.standard_normal(U0.shape)
    dt = 1e-3
    n=0
    for k in range(400):
        U = s.ssp_rk3_step(U, dt); n+=1
        if not np.all(np.isfinite(U)): break
    amp = np.abs(U-U0).max()
    print(f"{mode} p={p} steps={n} t={n*dt:.2f} amp={amp:.3e} ratio={amp/1e-8:.2e}")
