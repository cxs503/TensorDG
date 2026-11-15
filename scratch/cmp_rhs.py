import numpy as np, torch
from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh
from tensordg.euler_2d import vortex_primitive
from tensordg import euler_dg_2d as E
from tensordg.euler_curved import conserved_from_primitive

def exact(t, period=10.0, strength=5.0, bg=(1.0,1.0)):
    def fn(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(x), torch.as_tensor(y), time=t, strength=strength, background_velocity=bg, period=period)
        return r.numpy(), u.numpy(), v.numpy(), p.numpy()
    return fn

nx=ny=16; p=1; L=10.0
mesh = straight_box_mesh(nx,ny,1,length=L)
print("index(1,0)", mesh.index(1,0) if hasattr(mesh,'index') else 'na')
s = CurvilinearEuler(mesh, degree=p)
U = s.project(exact(0.0))
print("my U shape", U.shape)
# convert to theirs (nx, ny, K, K, 4)
Ut = torch.as_tensor(U.reshape(nx, ny, p+1, p+1, 4))
order = max(p+2,3)
nodes, weights, basis, derivative, signs = E.legendre_operators(p, order)
dx = dy = L/nx
f0 = E._rhs(Ut, basis, derivative, weights, signs, dx, dy, (True,True), None, 0.0)
print("diff", float((torch.as_tensor(s.rhs(U).reshape(nx,ny,p+1,p+1,4))-f0).abs().max())/float(f0.abs().max()))
# my residual with their quadrature? mine has nq=4 gauss; theirs order=3 -> compare
r_mine = s.rhs(U).reshape(nx,ny,p+1,p+1,4)
print("mine max", np.abs(r_mine).max(), "theirs max", float(f.abs().max()))
print("rel diff", float((torch.as_tensor(r_mine)-f).abs().max())/float(f.abs().max()))
