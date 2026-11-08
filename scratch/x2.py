import torch, time
from tensordg.swe_2d import solve_swe_2d, legendre_operators_2d
def h2d(x,y): return torch.where(x<0.5, torch.ones_like(x), 0.2*torch.ones_like(x))
z=lambda x,y: torch.zeros_like(x)
for deg in (1,2):
    for n in (32,64):
        t0=time.time()
        x,y,c,zf,m=solve_swe_2d(h2d,z,z,z,nx=n,ny=8,degree=deg,final_time=0.1,cfl=0.15,periodic=(False,False))
        print(deg,n,'steps',m['steps'],'dt',m['dt'],'wall %.2fs'%(time.time()-t0), flush=True)
