import time, torch
from benchmarks.euler_dg_2d_wall_bc import WallReflection
from tensordg.euler_dg_2d import solve_euler_dg_2d
wr = WallReflection(x_shock=0.2, x_wall=0.6, incident_mach=2.0)
nx, ny = 40, 1; xmax, ymax = 0.7, 0.1
pre_rho, pre_u, pre_p = wr.pre; post_rho, post_u, post_p = wr.post
def initial(x, y):
    ahead = x < wr.x_zero
    return (torch.where(ahead, torch.full_like(x, pre_rho), torch.full_like(x, post_rho)),
            torch.where(ahead, torch.full_like(x, pre_u), torch.full_like(x, post_u)),
            torch.zeros_like(x),
            torch.where(ahead, torch.full_like(x, pre_p), torch.full_like(x, post_p)))
dx = xmax/nx
xc = (torch.arange(nx)+0.5)*dx
solid = torch.zeros(nx, ny, dtype=torch.bool); solid[xc >= 0.6+0.5*dx-1e-12] = True
for t in (0.02, 0.05, 0.10):
    t0=time.time()
    x,y,coeff,meta = solve_euler_dg_2d(initial, ncells=(nx,ny), degree=1, final_time=t,
        cfl=0.12, domain=((0.0,xmax),(0.0,ymax)), periodic=(False,False), solid=solid,
        limiter=None, tvb_constant=1.0, limiter_stage="final", track_boundary_flux=True,
        bcs={"left":"outflow","right":"wall","bottom":"wall","top":"wall"})
    print(f"t={t} steps={meta['steps']} wall={time.time()-t0:.1f}s", flush=True)
