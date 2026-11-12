import math, torch
from tensordg.elastic_2d import (ElasticMedium, plane_wave, solve_elastic_2d,
                                 modal_to_quadrature)
med = ElasticMedium()
def run(kx, ky, nx, ny, degree, T, mode="P"):
    ex = plane_wave(mode, kx, ky, 1.0, med)
    res = solve_elastic_2d(lambda x,y: ex(x,y,0.0), nx=nx, ny=ny, degree=degree,
                           final_time=T, cfl=0.1)
    xg,yg = res.quadrature_grid(); num = res.fields(); exf = ex(xg,yg,T)
    w = res.ops["weights"]; w2 = torch.outer(w,w).reshape(1,1,w.numel(),w.numel(),1)
    err = (((num-exf)**2*w2).sum()/((exf**2*w2).sum())).sqrt().item()
    return err
print("== x-only (ky=0), ny=1, P ==")
for deg in (1,2,3):
    prev=None
    for n in (8,16,32,64):
        e = run(2*math.pi, 0.0, n, 1, deg, 0.02)
        print(f"deg={deg} n={n} err={e:.3e} eoc={'' if prev is None else round(math.log(prev/e)/math.log(2),2)}")
        prev=e
print("== y-only (kx=0), nx=1, P ==")
for deg in (1,2,3):
    prev=None
    for n in (8,16,32,64):
        e = run(0.0, 2*math.pi, 1, n, deg, 0.02)
        print(f"deg={deg} n={n} err={e:.3e} eoc={'' if prev is None else round(math.log(prev/e)/math.log(2),2)}")
        prev=e
