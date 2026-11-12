import math, torch
from tensordg.elastic_2d import ElasticMedium, plane_wave, solve_elastic_2d
med=ElasticMedium()
def err(kx,ky,nx,ny,deg,T,cfl):
    ex=plane_wave("P",kx,ky,1.0,med)
    res=solve_elastic_2d(lambda x,y: ex(x,y,0.0), nx=nx, ny=ny, degree=deg, final_time=T, cfl=cfl)
    xg,yg=res.quadrature_grid(); num=res.fields(); exf=ex(xg,yg,T)
    w=res.ops["weights"]; w2=torch.outer(w,w).reshape(1,1,w.numel(),w.numel(),1)
    return ((((num-exf)**2*w2).sum()/((exf**2*w2).sum())).sqrt().item(), res.times)
for cfl in (0.4,0.2,0.1,0.05,0.025):
    print("cfl",cfl, "n=64 p=2", err(2*math.pi,0.0,64,1,2,0.02,cfl))
for T in (0.005,0.01,0.02,0.04):
    print("T",T,"n=64 p=2 cfl=.1", err(2*math.pi,0.0,64,1,2,T,0.1))
