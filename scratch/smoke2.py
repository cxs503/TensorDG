import numpy as np, torch
from tensordg.euler_curved import (CurvilinearEuler, warped_box_mesh, straight_box_mesh,
                                   hllc_normal, euler_flux, conserved_from_primitive)
from tensordg.euler_2d import vortex_primitive

U = conserved_from_primitive(np.array([1.0]), np.array([0.3]), np.array([-0.2]), np.array([1.0]))
nx, ny = np.array([0.6]), np.array([0.8])
F = hllc_normal(U, U, nx, ny)
Fx, Fy = euler_flux(U)
print("hllc consistency", F - (Fx*nx[:,None] + Fy*ny[:,None]))
print("flux U", F)

def exact(t, period=10.0, strength=5.0, bg=(1.0,1.0)):
    def fn(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(np.ascontiguousarray(x)),
                                   torch.as_tensor(np.ascontiguousarray(y)),
                                   time=t, strength=strength, background_velocity=bg, period=period)
        return r.numpy(), u.numpy(), v.numpy(), p.numpy()
    return fn

# straight mesh test
for p in (1,2):
  for n in (8,16,32):
    m = straight_box_mesh(n,n,1)
    s = CurvilinearEuler(m, degree=p)
    U0 = s.project(exact(0.0))
    try:
        U1,dt,steps = s.evolve(U0, 2.0)
        print(f"straight p={p} n={n} steps={steps} dt={dt:.4f}", s.relative_errors(U1, exact(2.0)))
    except RuntimeError as e:
        print("FAILED", p, n, e, "dt", s.stable_dt(U0))
