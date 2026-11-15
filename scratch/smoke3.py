import numpy as np, torch, time
from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh, warped_box_mesh, hllc_normal, euler_flux, conserved_from_primitive
from tensordg.euler_2d import vortex_primitive
U = conserved_from_primitive(np.array([1.0]), np.array([0.3]), np.array([-0.2]), np.array([1.0]))
nx, ny = np.array([0.6]), np.array([0.8])
print("hllc consistency", hllc_normal(U,U,nx,ny) - (euler_flux(U)[0]*nx[:,None]+euler_flux(U)[1]*ny[:,None]))
def exact(t, period=10.0, strength=5.0, bg=(1.0,1.0)):
    def fn(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(x), torch.as_tensor(y), time=t, strength=strength, background_velocity=bg, period=period)
        return r.numpy(), u.numpy(), v.numpy(), p.numpy()
    return fn
m = straight_box_mesh(8,8,1)
s = CurvilinearEuler(m, degree=1)
U0 = s.project(exact(0.0))
print("dt", s.stable_dt(U0), "speed", s.max_wave_speed(U0))
t0=time.time(); r = s.rhs(U0); print("rhs time", time.time()-t0, "max", np.abs(r).max())
for nst in (1,5,10):
    U = s.ssp_rk3_step(U0, s.stable_dt(U0))
    print("err after 1 step", s.relative_errors(U, exact(s.stable_dt(U0)))['worst'])
    break
U1, dt, steps = s.evolve(U0, 0.2)
print("steps", steps, "dt", dt, s.relative_errors(U1, exact(0.2)))
