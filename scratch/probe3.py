import math, torch
from tensordg.elastic_2d import (ElasticMedium, plane_wave, solve_elastic_2d,
                                 relative_l2_error, elastic_energy, modal_to_quadrature)
med = ElasticMedium()
k = 2*math.pi
names = ["vx","vy","sxx","syy","sxy"]
for mode in ("P","S"):
    ex = plane_wave(mode, k, k, 1.0, med)
    for n in (8,16,32):
        for T in (0.0, 0.005, 0.02):
            res = solve_elastic_2d(lambda x, y: ex(x,y,0.0), nx=n, ny=n, degree=2,
                                   final_time=max(T,1e-12), cfl=0.1)
            xg,yg = res.quadrature_grid()
            num = res.fields(); exf = ex(xg,yg,T)
            w = res.ops["weights"]; w2 = torch.outer(w,w).reshape(1,1,w.numel(),w.numel(),1)
            parts=[]
            for c in range(5):
                num_c = (num[...,c:c+1]**2*w2).sum().sqrt(); 
                d = ((num[...,c:c+1]-exf[...,c:c+1])**2*w2).sum().sqrt()
                parts.append((d/num_c).item())
            Emax = (num[...,0:2]**2*w2).sum().sqrt().item()
            E0n = (exf[...,0:2]**2*w2).sum().sqrt().item()
            print(f"{mode} n={n} T={T:.3f} " + " ".join(f"{nm}={e:.2e}" for nm,e in zip(names,parts)) + f"  |v|ratio={Emax/max(E0n,1e-30):.4f}")
