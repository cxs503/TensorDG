"""Diagnostic: does the mcmp.py velocity-shift scheme diverge because of the
scheme itself, or because of the model?  Compare it against the standard
mixture-velocity + Guo (2002) formulation on the *same* model and IC."""
import sys, time, torch
sys.path.insert(0, 'TensorLBM/src')
from tensorlbm.mcmp import MCMPSolver
from tensorlbm.cross_force import pseudopotential_rho

CS2 = 1.0/3.0
E = torch.tensor([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=torch.float64)
W = torch.tensor([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36], dtype=torch.float64)


def mac(f):
    rho = f.sum(0)
    ux = (f * E[:,0].view(-1,1,1)).sum(0) / rho
    uy = (f * E[:,1].view(-1,1,1)).sum(0) / rho
    return rho, ux, uy


def feq(rho, ux, uy):
    out = torch.empty((9,) + rho.shape, dtype=rho.dtype)
    eu = E[:,0].view(-1,1,1)*ux + E[:,1].view(-1,1,1)*uy
    u2 = ux*ux + uy*uy
    for i in range(9):
        out[i] = W[i]*rho*(1.0 + eu[i]/CS2 + eu[i]**2/(2*CS2**2) - u2/(2*CS2))
    return out


def stream(f):
    return torch.stack([torch.roll(f[i], shifts=(int(E[i,1]), int(E[i,0])), dims=(0,1)) for i in range(9)])


def sc_force(rho1, rho2, G):
    def grad(p):
        gx = torch.zeros_like(p); gy = torch.zeros_like(p)
        for i in range(9):
            s = torch.roll(p, shifts=(int(E[i,1]), int(E[i,0])), dims=(0,1))
            gx += W[i]*s*E[i,0]; gy += W[i]*s*E[i,1]
        return gx, gy
    g1x, g1y = grad(rho2); g2x, g2y = grad(rho1)
    return G*rho1*g1x, G*rho1*g1y, G*rho2*g2x, G*rho2*g2y


def guo_term(rho, ux, uy, fx, fy, tau=1.0):
    pref = (1.0 - 0.5/tau)
    eu = E[:,0].view(-1,1,1)*ux + E[:,1].view(-1,1,1)*uy
    ef = E[:,0].view(-1,1,1)*fx + E[:,1].view(-1,1,1)*fy
    out = torch.empty((9,) + rho.shape, dtype=rho.dtype)
    for i in range(9):
        out[i] = pref*W[i]*(((E[i,0]-ux)*fx + (E[i,1]-uy)*fy)/CS2 + eu[i]*ef[i]/CS2**2)
    return out


def run_guo(N, G, steps, width=2.0, every=1000, tag=""):
    y = torch.arange(N, dtype=torch.float64).view(N,1).expand(N,N)
    phi = torch.tanh((N/2.0-y)/width)
    r1 = 0.5+0.35*phi; r2 = 1.0-r1
    f1 = feq(r1, torch.zeros_like(r1), torch.zeros_like(r1))
    f2 = feq(r2, torch.zeros_like(r2), torch.zeros_like(r2))
    t0 = time.time()
    for k in range(1, steps+1):
        r1, ux1, uy1 = mac(f1); r2, ux2, uy2 = mac(f2)
        Fx1, Fy1, Fx2, Fy2 = sc_force(r1, r2, G)
        rt = (r1 + r2)
        umx = (r1*ux1 + 0.5*Fx1 + r2*ux2 + 0.5*Fx2)/rt
        umy = (r1*uy1 + 0.5*Fy1 + r2*uy2 + 0.5*Fy2)/rt
        f1 = f1 - (f1 - feq(r1, umx, umy)) + guo_term(r1, umx, umy, Fx1, Fy1)
        f2 = f2 - (f2 - feq(r2, umx, umy)) + guo_term(r2, umx, umy, Fx2, Fy2)
        f1 = stream(f1); f2 = stream(f2)
        if k % every == 0 or k == steps:
            r1, _, _ = mac(f1); r2, _, _ = mac(f2)
            rt = r1+r2
            p = CS2*(r1+r2) - G*CS2*r1*r2
            prof = r1.mean(dim=1)
            um = max(float(((r1*ux1).abs()).max()), 1e-30)
            fin = bool(torch.isfinite(f1).all() and torch.isfinite(f2).all())
            print(f"  {tag} G={G:+.2f} k={k:5d} hi={prof[:4].mean():.4f} lo={prof[-4:].mean():.4f} "
                  f"dip={prof.min():.4f} ptot_spread={(p.max()-p.min()):.2e} fin={fin} t={time.time()-t0:.1f}s")
            if not fin: return None
    return dict(r1=r1, r2=r2)


if __name__ == "__main__":
    torch.set_num_threads(4)
    N = 48
    for G in (-3.0,):
        run_guo(N, G, 4000, every=500, tag="guo")
