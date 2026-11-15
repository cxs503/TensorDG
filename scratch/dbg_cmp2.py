import numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from tensordg import euler_dg_2d as E
from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh
n=8; p=1; L=10.0
mesh = straight_box_mesh(n,n,1,length=L); s = CurvilinearEuler(mesh, degree=p)
rng = np.random.default_rng(1)
U = np.zeros((n*n,p+1,p+1,4)); U[:,0,0,:] = np.array([1.0,0.3,-0.2,2.565])
U += 1e-3*rng.standard_normal(U.shape)
Ut = torch.as_tensor(U.reshape(n,n,p+1,p+1,4))
nodes, weights, basis, derivative, signs = E.legendre_operators(p, max(p+2,3))
fref = E._rhs(Ut, basis, derivative, weights, signs, L/n, L/n, (True,True), None, 0.0).numpy().reshape(n*n,p+1,p+1,4)
mine = s.rhs(U)
print("max|mine|", np.abs(mine).max(), "max|ref|", np.abs(fref).max())
print("max|diff|", np.abs(mine-fref).max(), "rel", np.abs(mine-fref).max()/np.abs(fref).max())
print("mode 00 diff", np.round((mine-fref)[:,0,0,0][:6],6))
print("mode 00 mine", np.round(mine[:,0,0,0][:6],6), "ref", np.round(fref[:,0,0,0][:6],6))
# stability of the reference operator itself
Ut2 = Ut.clone(); dt=1e-3
for k in range(200):
    k1 = E._rhs(Ut2, basis, derivative, weights, signs, L/n, L/n, (True,True), None, 0.0)
    a = Ut2 + dt*k1; k2 = E._rhs(a, basis, derivative, weights, signs, L/n, L/n, (True,True), None, 0.0)
    bb = 0.75*Ut2 + 0.25*(a + dt*k2); k3 = E._rhs(bb, basis, derivative, weights, signs, L/n, L/n, (True,True), None, 0.0)
    Ut2 = Ut2/3 + (2/3)*(bb + dt*k3)
print("reference amp", float((Ut2-torch.as_tensor(U.reshape(n,n,p+1,p+1,4))).abs().max())/1e-3)
