import numpy as np, torch, time
from tensordg.euler_curved import (CurvilinearEuler, warped_box_mesh, straight_box_mesh)
from tensordg.euler_2d import vortex_primitive

def exact(t, period=10.0, strength=5.0, bg=(1.0,1.0)):
    def fn(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(np.ascontiguousarray(x)),
                                   torch.as_tensor(np.ascontiguousarray(y)),
                                   time=t, strength=strength, background_velocity=bg,
                                   period=period)
        return r.numpy(), u.numpy(), v.numpy(), p.numpy()
    return fn

t0=time.time()
mesh = warped_box_mesh(4,4,degree=3)
solver = CurvilinearEuler(mesh, degree=1)
print("J min/max", solver.J.min(), solver.J.max())
fs = solver.free_stream_defect(steps=20)
print("FSP", fs)
neg = CurvilinearEuler(mesh, degree=1, volume_derivative_degree=1)
print("FSP neg", neg.free_stream_defect(steps=20))
print("elapsed", time.time()-t0)

# order check P1/P2 on warped mesh
for p in (1,2):
    errs=[]
    for n in (8,16):
        m = warped_box_mesh(n,n,degree=3)
        s = CurvilinearEuler(m, degree=p)
        U = s.project(exact(0.0))
        U,dt,steps = s.evolve(U, 2.0)
        e = s.relative_errors(U, exact(2.0))
        errs.append(e['worst'])
        print(f"p={p} n={n} steps={steps} dt={dt:.4f} err={e}")
    print("order", np.log(errs[0]/errs[1])/np.log(2))
print("total elapsed", time.time()-t0)
