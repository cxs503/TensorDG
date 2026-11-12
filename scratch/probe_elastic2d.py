import math, torch
from tensordg.elastic_2d import (ElasticMedium, plane_wave, solve_elastic_2d,
                                 relative_l2_error, elastic_energy)

med = ElasticMedium(rho=1.0, lam=1.0, mu=1.0)
print("cp", med.cp, "cs", med.cs, "zp", med.zp, "zs", med.zs)
L = 1.0
for mode in ("P", "S"):
    k = 2*math.pi*1
    ex = plane_wave(mode, k, k, 1.0, med)
    T = 0.02
    prev = None
    for n in (8, 16, 32):
        res = solve_elastic_2d(lambda x, y: ex(x, y, 0.0), nx=n, ny=n, degree=2,
                               final_time=T, cfl=0.2)
        xg, yg = res.quadrature_grid()
        num = res.fields()
        exf = ex(xg, yg, T)
        err = relative_l2_error(num, exf, res.ops["weights"]).item()
        eoc = None if prev is None else math.log(prev/err)/math.log(2)
        print(f"{mode} n={n} err={err:.3e} eoc={eoc}")
        prev = err
