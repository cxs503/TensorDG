import sys, torch
sys.path.insert(0,'TensorLBM/scratch')
from probe_schemes import *

def psi_rho(r): return r
def psi_exp(r): return 1.0 - torch.exp(-r)

def sc_force_psi(rho1, rho2, G, psi):
    def grad(p):
        gx = torch.zeros_like(p); gy = torch.zeros_like(p)
        for i in range(9):
            s = torch.roll(p, shifts=(int(E[i,1]), int(E[i,0])), dims=(0,1))
            gx += W[i]*s*E[i,0]; gy += W[i]*s*E[i,1]
        return gx, gy
    p1, p2 = psi(rho1), psi(rho2)
    g1x,g1y = grad(p2); g2x,g2y = grad(p1)
    return G*p1*g1x, G*p1*g1y, G*p2*g2x, G*p2*g2y

def trace(N=48, G=-3.0, steps=120, psi=psi_exp, tau=1.0, width=2.0):
    y = torch.arange(N, dtype=torch.float64).view(N,1).expand(N,N)
    phi = torch.tanh((N/2.0-y)/width)
    r1 = 0.5+0.35*phi; r2 = 1.0-r1
    f1 = feq(r1, torch.zeros_like(r1), torch.zeros_like(r1))
    f2 = feq(r2, torch.zeros_like(r2), torch.zeros_like(r2))
    for k in range(1, steps+1):
        r1, ux1, uy1 = mac(f1); r2, ux2, uy2 = mac(f2)
        Fx1,Fy1,Fx2,Fy2 = sc_force_psi(r1,r2,G,psi)
        rt = r1+r2
        umx = (r1*ux1+0.5*Fx1+r2*ux2+0.5*Fx2)/rt
        umy = (r1*uy1+0.5*Fy1+r2*uy2+0.5*Fy2)/rt
        f1 = f1 - (f1-feq(r1,umx,umy))/tau + guo_term(r1,umx,umy,Fx1,Fy1,tau)
        f2 = f2 - (f2-feq(r2,umx,umy))/tau + guo_term(r2,umx,umy,Fx2,Fy2,tau)
        f1 = stream(f1); f2 = stream(f2)
        if k <= 20 or k % 10 == 0:
            r1n,ux1n,uy1n = mac(f1); r2n,_,_ = mac(f2)
            print(f"k={k:4d} r1min={r1n.min():.4f} r1max={r1n.max():.4f} r2min={r2n.min():.4f} "
                  f"umax={max(umx.abs().max().item(),umy.abs().max().item()):.3e} "
                  f"Fmax={max(Fx1.abs().max().item(),Fy1.abs().max().item()):.3e} "
                  f"rtmin={r1n.min().item()+r2n.min().item():.4f}")
            if not torch.isfinite(r1n).all():
                print("DIVERGED"); return
torch.set_num_threads(4)
trace(psi=psi_exp, G=-3.0, steps=200)
