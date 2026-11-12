import math, torch, numpy as np
from tensordg.elastic_2d import ElasticMedium, legendre_operators_2d, _rhs, plane_wave, solve_elastic_2d
# (a) eigenvalues of the semi-discrete operator for the smallest interesting case, ny=1
med=ElasticMedium(); deg=2; nx,ny=2,1; K=deg+1
ops=legendre_operators_2d(deg); hx,hy=1.0/nx,1.0/ny
N=nx*ny*K*K*5; A=np.zeros((N,N))
for j in range(N):
    c=torch.zeros(nx,ny,K,K,5,dtype=torch.float64).reshape(-1); c[j]=1.0
    A[:,j]=_rhs(c.reshape(nx,ny,K,K,5),ops,med,hx,hy,"periodic","periodic").reshape(-1).numpy()
ev=np.linalg.eigvals(A)
print("deg2 nx=2 ny=1: max Re =", ev.real.max(), " max|Im| =", np.abs(ev.imag).max())
# (b) energy of the plane wave over time (is it conservative?)
ex=plane_wave("P",2*math.pi,0.0,1.0,med)
res=solve_elastic_2d(lambda x,y: ex(x,y,0.0), nx=32, ny=1, degree=2, final_time=0.08, cfl=0.1,
                     snapshot_times=(0.02,0.04,0.06))
E0=res.energy(res.snapshots[0.0]).item() if 0.0 in res.snapshots else float('nan')
print("energy at snapshots (rel to E(0.02)):")
base=None
for t,c in sorted(res.snapshots.items()):
    e=res.energy(c).item()
    base = e if base is None else base
    print(f"  t={t:.3f} E={e:.6e} E/E0={e/base:.6f}")
