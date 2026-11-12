import torch
from tensordg.elastic_2d import (ElasticMedium, legendre_operators_2d,
                                 quadrature_grid, modal_to_quadrature, flux_x)
med=ElasticMedium(); deg=2; n=4
ops=legendre_operators_2d(deg); nx,ny,K=n,1,deg+1; hx,hy=1.0/nx,1.0/ny
xg,yg=quadrature_grid(nx,ny,hx,hy,ops["nodes"])
def u_fn(x,y):
    out=torch.zeros(*x.shape,5,dtype=torch.float64)
    out[...,2]=x**2   # sxx = x^2
    return out
u0=u_fn(xg,yg)
c0=torch.einsum("ka,lb,ijabm->ijklm",ops["proj"],ops["proj"],u0)
fx=flux_x(u0,med)
cfx=torch.einsum("ka,lb,ijabm->ijklm",ops["proj"],ops["proj"],fx)
vol=-(2.0/hx)*torch.einsum("kn,ijnlm->ijklm",ops["dmat"],cfx)
volq=modal_to_quadrature(vol,ops["phi"])
print("vol[...,0] at nodes vs 2x/rho:", volq[0,0,:,0,0].tolist(), (2*xg[0,0,:,0]/med.rho).tolist())
