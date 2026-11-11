import json, time, torch, math
import tensordg.euler_dg_2d as dg2
from benchmarks.verify_anisotropic_2d import _operators, _vortex_error, DOMAIN, FINAL_TIME, CFL, TVB
torch.set_default_dtype(torch.float64); torch.set_num_threads(4)
out={}
nodes, weights, basis, der, signs = _operators(2)
for (nx,ny) in ((8,16),(16,32)):
    t0=time.time()
    _,_,c0,m0 = dg2.solve_euler_dg_2d_vortex(ncells=(nx,ny), degree=2, final_time=FINAL_TIME,
        cfl=CFL, limiter=None, tvb_constant=TVB, domain=DOMAIN)
    e0=_vortex_error(c0,basis,nodes,DOMAIN,FINAL_TIME)
    dx=(DOMAIN[0][1]-DOMAIN[0][0])/nx; dy=(DOMAIN[1][1]-DOMAIN[1][0])/ny
    cw = dg2.apply_weno_limiter_2d(c0, dx, dy, TVB, (True,True))
    ew = _vortex_error(cw,basis,nodes,DOMAIN,FINAL_TIME)
    ct = dg2.apply_tvb_limiter_2d(c0, dx, dy, TVB, (True,True))
    et = _vortex_error(ct,basis,nodes,DOMAIN,FINAL_TIME)
    # how many cells modified by weno
    diff = (cw-c0).abs().sum(dim=(-1,-2,-3))
    nmod = int((diff>1e-14).sum())
    # per-stage every
    _,_,ce,me = dg2.solve_euler_dg_2d_vortex(ncells=(nx,ny), degree=2, final_time=FINAL_TIME,
        cfl=CFL, limiter="weno", tvb_constant=TVB, domain=DOMAIN)
    ee=_vortex_error(ce,basis,nodes,DOMAIN,FINAL_TIME)
    # final-only on the unlimited run
    _,_,cf,mf = dg2.solve_euler_dg_2d_vortex(ncells=(nx,ny), degree=2, final_time=FINAL_TIME,
        cfl=CFL, limiter="weno", tvb_constant=TVB, limiter_stage="final", domain=DOMAIN)
    ef=_vortex_error(cf,basis,nodes,DOMAIN,FINAL_TIME)
    out[f"{nx}x{ny}"]=dict(unlimited=e0, single_shot_weno=ew, single_shot_tvb=et,
        every_stage=ee, final_stage=ef, modified_cells=nmod, cells=nx*ny,
        steps=[m0['steps'],me['steps'],mf['steps']], wall=time.time()-t0)
    print(json.dumps(out[f"{nx}x{ny}"], indent=1), flush=True)
json.dump(out, open('scratch/probe3.json','w'), indent=1)
