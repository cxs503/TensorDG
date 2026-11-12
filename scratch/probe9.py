import math, torch
from tensordg.elastic_2d import (ElasticMedium, plane_wave, legendre_operators_2d,
                                 quadrature_grid, modal_to_quadrature, _rhs)
med=ElasticMedium(); k=2*math.pi; deg=2; n=8
ops=legendre_operators_2d(deg); nx,ny,K=n,1,deg+1; hx,hy=1.0/nx,1.0/ny
xg,yg=quadrature_grid(nx,ny,hx,hy,ops["nodes"])
ex=plane_wave("P",k,0.0,1.0,med)
c0=torch.einsum("ka,lb,ijabm->ijklm",ops["proj"],ops["proj"],ex(xg,yg,0.0))
r=modal_to_quadrature(_rhs(c0,ops,med,hx,hy,"periodic","periodic"),ops["phi"])
dt=1e-6
fd=(ex(xg,yg,dt)-ex(xg,yg,-dt))/(2*dt)
i,j,a,b=1,0,0,0
for comp,nm in enumerate(["vx","vy","sxx","syy","sxy"]):
    print(f"{nm}: rhs={r[i,j,a,b,comp].item():+.6f}  fd={fd[i,j,a,b,comp].item():+.6f}")
print("x=",xg[i,j,a,b].item())
