import sys, torch, time
sys.path.insert(0,'TensorLBM/scratch')
from probe_trace import *
from probe_schemes import *
import probe_schemes as PS

torch.set_num_threads(2)
N=48
def run_guo(N, G, steps, psi, tau=1.0, width=2.0, every=1000, tag=""):
    import torch
    y = torch.arange(N, dtype=torch.float64).view(N,1).expand(N,N)
    phi = torch.tanh((N/2.0-y)/width)
    r1 = 0.5+0.35*phi; r2 = 1.0-r1
    f1 = feq(r1, torch.zeros_like(r1), torch.zeros_like(r1)); f2 = feq(r2, torch.zeros_like(r2), torch.zeros_like(r2))
    t0=time.time()
    for k in range(1, steps+1):
        r1, ux1, uy1 = mac(f1); r2, ux2, uy2 = mac(f2)
        Fx1,Fy1,Fx2,Fy2 = sc_force_psi(r1,r2,G,psi)
        rt = r1+r2
        umx = (r1*ux1+0.5*Fx1+r2*ux2+0.5*Fx2)/rt
        umy = (r1*uy1+0.5*Fy1+r2*uy2+0.5*Fy2)/rt
        f1 = f1 - (f1-feq(r1,umx,umy))/tau + guo_term(r1,umx,umy,Fx1,Fy1,tau)
        f2 = f2 - (f2-feq(r2,umx,umy))/tau + guo_term(r2,umx,umy,Fx2,Fy2,tau)
        f1 = stream(f1); f2 = stream(f2)
        if not torch.isfinite(f1).all():
            print(f"  [{tag}] G={G} DIVERGED at k={k}"); return
        if k % every == 0 or k==steps:
            r1n,_,_ = mac(f1); r2n,_,_ = mac(f2)
            prof = r1n.mean(dim=1)
            print(f"  [{tag}] G={G:+.1f} k={k:5d} hi={prof[:4].mean():.4f} lo={prof[-4:].mean():.4f} "
                  f"rt_min={(r1n+r2n).min():.3f} fin=True t={time.time()-t0:.0f}s")
run_guo(N, -6.0, 4000, psi_exp, tag="guo-exp")
