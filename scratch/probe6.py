import math, torch
from tensordg.elastic_2d import (ElasticMedium, plane_wave, solve_elastic_2d, legendre_operators_2d,
                                 quadrature_grid, _rhs, modal_to_quadrature)
med = ElasticMedium(); k=2*math.pi; deg=2; n=16
ops = legendre_operators_2d(deg); nx,ny,K=n,1,deg+1
hx,hy=1.0/nx,1.0/ny
xg,yg=quadrature_grid(nx,ny,hx,hy,ops["nodes"])
ex = plane_wave("P", k, 0.0, 1.0, med)
u0 = ex(xg,yg,0.0)
c0 = torch.einsum("ka,lb,ijabm->ijklm", ops["proj"], ops["proj"], u0)
w=ops["weights"]; w2=torch.outer(w,w).reshape(1,1,w.numel(),w.numel(),1)
def err(a,b):
    return (((a-b)**2*w2).sum()/((b**2*w2).sum())).sqrt().item()
# exact derivative check (corrected)
dt_true = -ex.omega*ex(xg,yg,0.0+math.pi/(2*ex.omega))  # = -w f cos
r = _rhs(c0, ops, med, hx, hy, "periodic","periodic")
rq = modal_to_quadrature(r, ops["phi"])
print("rhs vs true dt rel:", err(rq, dt_true), " ratio of rms:", (rq**2*w2).sum().sqrt().item()/((dt_true**2*w2).sum().sqrt().item()))
# one tiny step
for T in (1e-6, 1e-4, 1e-3):
    res = solve_elastic_2d(lambda x,y: ex(x,y,0.0), nx=nx, ny=ny, degree=deg, final_time=T, cfl=0.04)
    xg2,yg2 = res.quadrature_grid()
    print("T",T,"err", err(res.fields(), ex(xg2,yg2,T)), "dt,n", res.times)
