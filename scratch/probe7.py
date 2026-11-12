import math, torch
from tensordg.elastic_2d import (ElasticMedium, plane_wave, legendre_operators_2d,
                                 quadrature_grid, modal_to_quadrature, flux_x, flux_y)
med=ElasticMedium(); k=2*math.pi; deg=2; n=16
ops=legendre_operators_2d(deg); nx,ny,K=n,1,deg+1; hx,hy=1.0/nx,1.0/ny
# derivative operator check
c = torch.zeros(K, dtype=torch.float64); c[1]=1.0
print("dmat @ coeff(xi) ->", (ops["dmat"]@c).tolist(), " expected [1,0,0]")
print("proj@phi  ->\n", (ops["proj"]@ops["phi"]))
xg,yg=quadrature_grid(nx,ny,hx,hy,ops["nodes"])
ex=plane_wave("P",k,0.0,1.0,med)
u0=ex(xg,yg,0.0)
c0=torch.einsum("ka,lb,ijabm->ijklm",ops["proj"],ops["proj"],u0)
fx=flux_x(u0,med)
cfx=torch.einsum("ka,lb,ijabm->ijklm",ops["proj"],ops["proj"],fx)
vol=-(2.0/hx)*torch.einsum("kn,ijnlm->ijklm",ops["dmat"],cfx)
volq=modal_to_quadrature(vol,ops["phi"])
# analytic dF/dx at quad points: fx = -sigma/rho etc; sigma ~ sin(kx) -> d/dx = k cos
fx_dx = k*flux_x(ex(xg,yg,0.0+math.pi/(2*ex.omega)),med)  # cos -> use shift trick: d/dx sin = k cos = k*sin(..+pi/2)
w=ops["weights"]; w2=torch.outer(w,w).reshape(1,1,w.numel(),w.numel(),1)
def rel(a,b): return (((a-b)**2*w2).sum()/((b**2*w2).sum())).sqrt().item()
print("volume(-dF/dx) vs analytic -dF/dx rel:", rel(volq, -fx_dx))
for comp,nm in enumerate(["vx","vy","sxx","syy","sxy"]):
    print(f"  {nm}: vol_rms={((volq[...,comp]**2*w2).sum()/w2.sum()).sqrt().item():.4f} ana_rms={((fx_dx[...,comp]**2*w2).sum()/w2.sum()).sqrt().item():.4f}")
