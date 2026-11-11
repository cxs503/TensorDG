import math, time, torch
import tensordg.euler_dg_2d as dg2
from benchmarks.verify_anisotropic_2d import _operators, _vortex_error, DOMAIN, FINAL_TIME, CFL, TVB, FIELDS
import benchmarks.verify_euler_dg_2d as base

torch.set_default_dtype(torch.float64)
for (nx,ny) in ((8,16),(16,32)):
    for lim in (None,"tvb","weno"):
        t0=time.time()
        nodes, weights, basis, _, _ = _operators(2)
        _, _, coeff, meta = dg2.solve_euler_dg_2d_vortex(ncells=(nx,ny), degree=2,
            final_time=FINAL_TIME, cfl=CFL, limiter=lim, tvb_constant=TVB, domain=DOMAIN)
        errs=_vortex_error(coeff, basis, nodes, DOMAIN, FINAL_TIME)
        print(f"{nx}x{ny} P2 lim={lim}: steps={meta['steps']} rho={errs['density']:.4e} | {time.time()-t0:.1f}s")
