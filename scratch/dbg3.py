import numpy as np, math
E,nu,rho=30e6,0.3,2000.0
a=E/((1+nu)*(1-2*nu)); Dxx,Dyy,Dxy,G=a*(1-nu),a*(1-nu),a*nu,a*(1-2*nu)/2
C=np.array([[Dxx,Dxy,0],[Dxy,Dyy,0],[0,0,G]])
nx,ny=2,2; W,H=1.0,1.0; dx,dy=W/nx,H/ny
nnx,nny=nx+1,ny+1
ndof=2*nnx*nny
K=np.zeros((ndof,ndof))
gg=1/math.sqrt(3); gps=[(-gg,-gg),(gg,-gg),(gg,gg),(-gg,gg)]
corners=[(0,0),(1,0),(1,1),(0,1)]
wt=dx*dy/4
for e_i in range(nx):
  for e_j in range(ny):
    dofs=[]
    for (di,dj) in corners:
        node=(e_i+di)*(nny)+(e_j+dj)
        dofs += [2*node, 2*node+1]
    ke=np.zeros((8,8))
    for k,(Xi,Eta) in enumerate(gps):
        dNdxi=np.array([-(1-Eta),(1-Eta),(1+Eta),-(1+Eta)])/4
        dNdeta=np.array([-(1-Xi),-(1+Xi),(1+Xi),(1-Xi)])/4
        dNx=dNdxi*2/dx; dNy=dNdeta*2/dy
        B=np.zeros((3,8))
        for i in range(4):
            B[0,2*i]=dNx[i]; B[1,2*i+1]=dNy[i]; B[2,2*i]=dNy[i]; B[2,2*i+1]=dNx[i]
        ke += wt*B.T@C@B
    for p in range(8):
        for q in range(8):
            K[dofs[p],dofs[q]]+=ke[p,q]
trib=np.zeros((nnx,nny))
for di in (0,1):
    for dj in (0,1): trib[di:di+nx,dj:dj+ny]+=0.25*dx*dy
m=np.repeat((rho*trib).ravel(),2)
Minv=np.diag(1/m)
A=-Minv@K
Ms=np.diag(m)
w=np.sort(np.linalg.eigvals(np.linalg.solve(Ms,K)))
print('dt crit (2/omega)', 2/math.sqrt(w.max()))
cp=math.sqrt(Dxx/rho)
print('h/cp', min(dx,dy)/cp, 'ratio dtcrit/(h/cp)', (2/math.sqrt(w.max()))/(min(dx,dy)/cp))
