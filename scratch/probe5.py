import math, torch
from tensordg.elastic_2d import (ElasticMedium, plane_wave, legendre_operators_2d,
                                 quadrature_grid, _rhs, modal_to_quadrature)
med = ElasticMedium(); k=2*math.pi; deg=2; n=16
ops = legendre_operators_2d(deg)
nx, ny, K = n, 1, deg+1
hx, hy = 1.0/nx, 1.0/ny
xg, yg = quadrature_grid(nx, ny, hx, hy, ops["nodes"])
ex = plane_wave("P", k, 0.0, 1.0, med)
# derivative of exact field wrt time at t=0, via analytic omega
omega = ex.omega
s = torch.sin(k*xg); c = torch.cos(k*xg)
dt_u = torch.empty(*s.shape, 5, dtype=torch.float64)
# exact: u(t) = f(x) sin(kx - wt) -> du/dt = -w f cos(kx - wt) at t=0 = -w f cos(kx)
u0 = ex(xg, yg, 0.0)
dt_exact = -omega * ex(xg, yg, 0.0 + math.pi/(2*omega))   # cos = sin(.. + pi/2)
u0c = torch.einsum("ka,lb,ijabm->ijklm", ops["proj"], ops["proj"], u0)
r = _rhs(u0c, ops, med, hx, hy, "periodic", "periodic")
rc = torch.einsum("ka,lb,ijabm->ijklm", ops["proj"], ops["proj"], dt_exact)
w = ops["weights"]; w2 = torch.outer(w,w).reshape(1,1,w.numel(),w.numel(),1)
w5 = w2.expand(nx,ny,K,K,5)
print("rhs/expected per component ratio:")
for comp, nm in enumerate(["vx","vy","sxx","syy","sxy"]):
    num = (((r[...,comp:comp+1]-rc[...,comp:comp+1])**2*w2).sum()).sqrt().item()
    den = ((rc[...,comp:comp+1]**2*w2).sum()).sqrt().item()
    print(f"  {nm}: rel diff={num/max(den,1e-30):.3e}   rhs_rms={((r[...,comp]**2*w2).sum()/w2.sum()).sqrt().item():.4f} exact_rms={((rc[...,comp]**2*w2).sum()/w2.sum()).sqrt().item():.4f}")
