import torch, numpy as np
from tensordg.elastic_2d import ElasticMedium, legendre_operators_2d, _rhs
med=ElasticMedium(); deg=1; nx,ny=4,1; K=deg+1
ops=legendre_operators_2d(deg); hx,hy=1.0/nx,1.0/ny
N=nx*ny*K*K*5
A=torch.zeros(N,N,dtype=torch.float64)
for j in range(N):
    c=torch.zeros(nx,ny,K,K,5,dtype=torch.float64).reshape(-1); c[j]=1.0
    A[:,j]=_rhs(c.reshape(nx,ny,K,K,5),ops,med,hx,hy,"periodic","periodic").reshape(-1)
ev=torch.linalg.eigvals(A)
ev=np.linalg.eigvals(A.numpy())
print("max Re(eig):", ev.real.max(), " at", ev.real.argmax())
print("top 5 Re:", np.sort(ev.real)[-5:])
print("max |Im|:", np.abs(ev.imag).max())
# also for periodic y? ny=1 periodic y gives zero flux contributions
