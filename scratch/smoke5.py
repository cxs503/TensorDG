import numpy as np, torch
from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh, warped_box_mesh
from tensordg.euler_2d import vortex_primitive
def exact(t, period=10.0, strength=5.0, bg=(1.0,1.0)):
    def fn(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(x), torch.as_tensor(y), time=t, strength=strength, background_velocity=bg, period=period)
        return r.numpy(), u.numpy(), v.numpy(), p.numpy()
    return fn
for name, mk in (("warped", lambda n: warped_box_mesh(n,n,3)), ("straight", lambda n: straight_box_mesh(n,n,1))):
  for n in (8,16):
    m = mk(n); s = CurvilinearEuler(m, degree=1, cfl=0.2)
    U = s.project(exact(0.0))
    for T in (0.5, 2.0):
        try:
            U1, dt, steps = s.evolve(U, T)
            e = s.relative_errors(U1, exact(T))
            rho,u,v,p = [None]*4
            print(f"{name} n={n} T={T} steps={steps} dt={dt:.5f} err={e['worst']:.4e} rho={e['rho']:.3e} p={e['p']:.3e}")
        except RuntimeError as ex:
            print(f"{name} n={n} T={T} BLOWUP {ex}")
            break
