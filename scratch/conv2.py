import numpy as np, torch, time
from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh, warped_box_mesh
from tensordg.euler_2d import vortex_primitive
def prim(t, period=10.0):
    def fn(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(x), torch.as_tensor(y), time=t, strength=5.0,
                                   background_velocity=(1.0,1.0), period=period)
        return r.numpy(), u.numpy(), v.numpy(), p.numpy()
    return fn
t0=time.time()
for name, mk in (("straight", lambda n: straight_box_mesh(n,n,1)), ("warped", lambda n: warped_box_mesh(n,n,3))):
  for p in (1,2):
    errs=[]
    for n in (8,16):
        s = CurvilinearEuler(mk(n), degree=p, cfl=0.2)
        U = s.project(prim(0.0)); U,dt,st = s.evolve(U, 0.1)
        e = s.relative_errors(U, prim(0.1))['worst']; errs.append(e)
        print(f"{name} p={p} n={n} steps={st} err={e:.4e} t={time.time()-t0:.1f}")
    print(f"  order {np.log(errs[0]/errs[1])/np.log(2):.3f}")
