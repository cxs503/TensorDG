import json, time, torch
import tensordg.euler_dg_2d as dg2
from benchmarks.verify_anisotropic_2d import _operators, _vortex_error, DOMAIN, FINAL_TIME, CFL, TVB
torch.set_default_dtype(torch.float64); torch.set_num_threads(4)
nodes, weights, basis, der, signs = _operators(2)
out={}
for (nx,ny) in ((8,16),(16,32)):
    dx=(DOMAIN[0][1]-DOMAIN[0][0])/nx; dy=(DOMAIN[1][1]-DOMAIN[1][0])/ny
    _,_,c0,m0 = dg2.solve_euler_dg_2d_vortex(ncells=(nx,ny), degree=2, final_time=FINAL_TIME,
        cfl=CFL, limiter=None, tvb_constant=TVB, domain=DOMAIN)
    e0=_vortex_error(c0,basis,nodes,DOMAIN,FINAL_TIME)
    cw=dg2.apply_weno_limiter_2d(c0,dx,dy,TVB,(True,True))
    ew=_vortex_error(cw,basis,nodes,DOMAIN,FINAL_TIME)
    _,_,ce,me = dg2.solve_euler_dg_2d_vortex(ncells=(nx,ny), degree=2, final_time=FINAL_TIME,
        cfl=CFL, limiter="weno", tvb_constant=TVB, domain=DOMAIN)
    ee=_vortex_error(ce,basis,nodes,DOMAIN,FINAL_TIME)
    _,_,cf,mf = dg2.solve_euler_dg_2d_vortex(ncells=(nx,ny), degree=2, final_time=FINAL_TIME,
        cfl=CFL, limiter="weno", tvb_constant=TVB, limiter_stage="final", domain=DOMAIN)
    ef=_vortex_error(cf,basis,nodes,DOMAIN,FINAL_TIME)
    out[f"{nx}x{ny}"]=dict(unlimited=e0,single_shot=ew,every=ee,final_only=ef,steps=[m0['steps'],me['steps'],mf['steps']])
    print(nx,ny, json.dumps({k:{'rho':round(v['density'],6),'ux':round(v['velocity_x'],6)} for k,v in out[f'{nx}x{ny}'].items() if isinstance(v,dict)}), flush=True)
json.dump(out, open('scratch/probe4.json','w'), indent=1)
