import numpy as np, math
src = open('scratch/proto4.py').read().split('dec = np.full')[0]
exec(src)
print('dt', dt, 'cpe', cpe, 'cp', cp, 'dx', dx, 'dy', dy)
# elastic-only test (huge cohesion), uniform compression, no imperfection
nx,ny = 10, 20
dx,dy = W/nx, H/ny
Msc=1e4; cpe=cp/math.sqrt(Msc); dt=0.35*min(dx,dy)/cpe
gg=1.0/math.sqrt(3.0)
gps=[(-gg,-gg),(gg,-gg),(gg,gg),(-gg,gg)]
dNdx=[];dNdy=[]
for Xi,Eta in gps:
    dNdxi=np.array([-(1-Eta),(1-Eta),(1+Eta),-(1+Eta)])/4.0
    dNdeta=np.array([-(1-Xi),-(1+Xi),(1+Xi),(1-Xi)])/4.0
    dNdx.append(dNdxi*2.0/dx); dNdy.append(dNdeta*2.0/dy)
wt=dx*dy/4.0; corners=[(0,0),(1,0),(1,1),(0,1)]
dec=np.full((nx,ny),-sig3); sxx=[dec.copy() for _ in range(4)]; syy=[dec.copy() for _ in range(4)]
sxy=[np.zeros((nx,ny)) for _ in range(4)]
vx=np.zeros((nx+1,ny+1)); vy=np.zeros((nx+1,ny+1))
trib=np.zeros((nx+1,ny+1))
for di in (0,1):
    for dj in (0,1): trib[di:di+nx,dj:dj+ny]+=0.25*dx*dy
mass=rho*Msc*trib
wn=np.ones(ny+1); wn[0]=wn[-1]=0.5
v0=5.7e-4; damp=0.05
for n in range(80):
    for k in range(4):
        vxe=[vx[di:di+nx,dj:dj+ny] for di,dj in corners]
        vye=[vy[di:di+nx,dj:dj+ny] for di,dj in corners]
        exx=sum(dNdx[k][i]*vxe[i] for i in range(4)); eyy=sum(dNdy[k][i]*vye[i] for i in range(4))
        gxy=sum(dNdy[k][i]*vxe[i]+dNdx[k][i]*vye[i] for i in range(4))
        sxx[k]=sxx[k]+dt*(Dxx*exx+Dxy*eyy); syy[k]=syy[k]+dt*(Dxy*exx+Dyy*eyy); sxy[k]=sxy[k]+dt*G*gxy
    fx=np.zeros((nx+1,ny+1)); fy=np.zeros_like(fx)
    for k in range(4):
        for i,(di,dj) in enumerate(corners):
            fx[di:di+nx,dj:dj+ny]-=wt*(dNdx[k][i]*sxx[k]+dNdy[k][i]*sxy[k])
            fy[di:di+nx,dj:dj+ny]-=wt*(dNdy[k][i]*syy[k]+dNdx[k][i]*sxy[k])
    fx[0,1:-1]+=sig3*dy*wn[1:-1]; fx[-1,1:-1]-=sig3*dy*wn[1:-1]
    vx+=(1-damp)*dt*fx/mass; vy+=(1-damp)*dt*fy/mass
    vy[0,:]=v0; vy[-1,:]=-v0
    if n<8 or n%10==0:
        print(f"{n:3d} |vx|={abs(vx).max():.3e} |vy|={abs(vy).max():.3e} sxx[min,max]=({sxx[0].min():.4e},{sxx[0].max():.4e}) sxy={abs(sxy[0]).max():.3e} fx={abs(fx).max():.3e}")