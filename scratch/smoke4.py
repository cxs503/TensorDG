import numpy as np, torch, time
from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh, warped_box_mesh
from tensordg.euler_2d import vortex_primitive
def exact(t, period=10.0, strength=5.0, bg=(1.0,1.0)):
    def fn(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(x), torch.as_tensor(y), time=t, strength=strength, background_velocity=bg, period=period)
        return r.numpy(), u.numpy(), v.numpy(), p.numpy()
    return fn
for mesh_name, mk in (("warped", lambda n: warped_box_mesh(n,n,3)), ("straight", lambda n: straight_box_mesh(n,n,1))):
  for p in (1,2):
    for cfl in (0.3, 0.1):
        m = mk(8); s = CurvilinearEuler(m, degree=p, cfl=cfl)
        U0 = s.project(exact(0.0))
        try:
            U1, dt, steps = s.evolve(U0, 0.2)
            print(f"{mesh_name} p={p} cfl={cfl} steps={steps} dt={dt:.5f} err={s.relative_errors(U1, exact(0.2))['worst']:.4e}")
        except RuntimeError as e:
            print(f"{mesh_name} p={p} cfl={cfl} BLOWUP dt={s.stable_dt(U0):.5f}")
