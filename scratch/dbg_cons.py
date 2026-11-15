import numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from tensordg.euler_curved import CurvilinearEuler, warped_box_mesh, hllc_normal, conserved_from_primitive, euler_flux
from tensordg.euler_2d import vortex_primitive
def prim(t):
    def f(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(x), torch.as_tensor(y), time=t, strength=5.0,
                                   background_velocity=(1.0,1.0), period=10.0)
        return r.numpy(),u.numpy(),v.numpy(),p.numpy()
    return f
s = CurvilinearEuler(warped_box_mesh(8,8,3,length=10.0,amplitude=0.06), degree=1, cfl=0.2)
U = s.project(prim(0.0)); t0 = s.totals(U)
U,dt,steps = s.evolve(U, 0.5)
t1 = s.totals(U)
print("steps", steps, "dt", dt)
print("totals", t0, t1)
print("relative drift", np.max(np.abs(t1-t0))/np.max(np.abs(t0)))
st = conserved_from_primitive(np.array([1.0]), np.array([0.3]), np.array([-0.2]), np.array([1.0]))
f = hllc_normal(st, st, np.array([0.6]), np.array([0.8]))
fx, fy = euler_flux(st)
print("consistency", np.abs(f-(fx*0.6+fy*0.8)).max())
