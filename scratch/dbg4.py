import numpy as np, math
E,nu,rho=30e6,0.3,2000.0
a=E/((1+nu)*(1-2*nu)); Dxx,Dyy,Dxy,G=a*(1-nu),a*(1-nu),a*nu,a*(1-2*nu)/2
C=np.array([[Dxx,Dxy,0],[Dxy,Dyy,0],[0,0,G]])
nx,ny=3,4; W,H=1.0,2.0; dx,dy=W/nx,H/ny; nnx,nny=nx+1,ny+1
gg=1/math.sqrt(3); gps=[(-gg,-gg),(gg,-gg),(gg,gg),(-gg,gg)]; corners=[(0,0),(1,0),(1,1),(0,1)]
wt=dx*dy/4
# dense K
ndof=2*nnx*nny; K=np.zeros((ndof,ndof))
for e_i in range(nx):
  for e_j in range(ny):
    dofs=[]
    for (di,dj) in corners:
        node=(e_i+di)*nny+(e_j+dj); dofs += [2*node, 2*node+1]
    ke=np.zeros((8,8))
    for k,(Xi,Eta) in enumerate(gps):
        dNdxi=np.array([-(1-Eta),(1-Eta),(1+Eta),-(1+Eta)])/4; dNdeta=np.array([-(1-Xi),-(1+Xi),(1+Xi),(1-Xi)])/4
        dNx=dNdxi*2/dx; dNy=dNdeta*2/dy
        B=np.zeros((3,8))
        for i in range(4):
            B[0,2*i]=dNx[i]; B[1,2*i+1]=dNy[i]; B[2,2*i]=dNy[i]; B[2,2*i+1]=dNx[i]
        ke+=wt*B.T@C@B
    for p in range(8):
        for q in range(8): K[dofs[p],dofs[q]]+=ke[p,q]
rng=np.random.default_rng(0)
v=rng.standard_normal(ndof)
f_ref=-K@v   # internal force = -K v for elastic
# my assembly
vx=v[0::2].reshape(nnx,nny); vy=v[1::2].reshape(nnx,nny)
sxx=[np.zeros((nx,ny)) for _ in range(4)]; syy=[np.zeros((nx,ny)) for _ in range(4)]; sxy=[np.zeros((nx,ny)) for _ in range(4)]
for k in range(4):
    dNdxi=np.array([-(1-gps[k][1]),(1-gps[k][1]),(1+gps[k][1]),-(1+gps[k][1])])/4
    dNdeta=np.array([-(1-gps[k][0]),-(1+gps[k][0]),(1+gps[k][0]),(1-gps[k][0])])/4
    dNx=dNdxi*2/dx; dNy=dNdeta*2/dy
    vxe=[vx[di:di+nx,dj:dj+ny] for di,dj in corners]; vye=[vy[di:di+nx,dj:dj+ny] for di,dj in corners]
    exx=sum(dNx[i]*vxe[i] for i in range(4)); eyy=sum(dNy[i]*vye[i] for i in range(4)); gxy=sum(dNy[i]*vxe[i]+dNx[i]*vye[i] for i in range(4))
    sxx[k]=Dxx*exx+Dxy*eyy; syy[k]=Dxy*exx+Dyy*eyy; sxy[k]=G*gxy
fx=np.zeros((nnx,nny)); fy=np.zeros_like(fx)
for k in range(4):
    dNdxi=np.array([-(1-gps[k][1]),(1-gps[k][1]),(1+gps[k][1]),-(1+gps[k][1])])/4
    dNdeta=np.array([-(1-gps[k][0]),-(1-gps[k][0]),(1+gps[k][0]),(1-gps[k][0])])/4
    dNx=dNdxi*2/dx; dNy=dNdeta*2/dy
    for i,(di,dj) in enumerate(corners):
        fx[di:di+nx,dj:dj+ny]+=wt*(dNx[i]*sxx[k]+dNy[i]*sxy[k])
        fy[di:di+nx,dj:dj+ny]+=wt*(dNy[i]*syy[k]+dNx[i]*sxy[k])
f_mine=np.zeros(ndof); f_mine[0::2]=fx.ravel(); f_mine[1::2]=fy.ravel()
print('max rel diff', np.abs(f_mine-f_ref).max()/np.abs(f_ref).max())
print('f_mine max', np.abs(f_mine).max(), 'f_ref max', np.abs(f_ref).max())
