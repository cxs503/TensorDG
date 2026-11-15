import time, torch
from tensordg.euler_dg_2d import solve_euler_dg_2d
def init(x, y):
    return (torch.ones_like(x), 0.7*torch.ones_like(x), torch.zeros_like(x), torch.ones_like(x))
for nx, p, t in ((60,1,0.005),(60,1,0.02),(60,1,0.05)):
    t0=time.time()
    x,y,coeff,meta = solve_euler_dg_2d(init, ncells=(nx,1), degree=p, final_time=t,
        cfl=0.12, domain=((0.0,1.2),(0.0,0.1)), periodic=(False,False),
        bcs={"left":"outflow","right":"wall","bottom":"wall","top":"wall"})
    print(f"nx={nx} p={p} t={t} steps={meta['steps']} wall={time.time()-t0:.1f}s", flush=True)
