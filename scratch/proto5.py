"""Scratch prototype v5: plane-strain biaxial compression, Q4 + 2x2 Gauss,
nonlocal softening Mohr-Coulomb, central difference + linear damping."""
import numpy as np, math, time, sys

E,nu,rho = 30e6, 0.3, 2000.0
phi,psi = math.radians(40.0), math.radians(0.0)
c0, cres, gp_ref = 350e3, 25e3, 2.0e-2
sig3 = 50e3
a = E/((1+nu)*(1-2*nu))
Dxx, Dyy, Dxy, G = a*(1-nu), a*(1-nu), a*nu, a*(1-2*nu)/2
sp, cpn = math.sin(phi), math.cos(phi)
cp = math.sqrt(Dxx/rho)

def mq(sxx,syy,sxy):
    return 0.5*(sxx+syy), np.sqrt(0.25*(sxx-syy)**2+sxy**2)

def Cdot(q):
    return np.stack([Dxx*q[0]+Dxy*q[1], Dxy*q[0]+Dyy*q[1], G*q[2]])

def flowgrad(sxx,syy,sxy,ang):
    s = 0.25*(sxx-syy); q = np.maximum(np.sqrt(s*s+sxy*sxy),1e-30); sa = math.sin(ang)
    return np.stack([0.5*s/q+0.5*sa, -0.5*s/q+0.5*sa, sxy/q])

def return_map(sxx,syy,sxy,ceff):
    m, q = mq(sxx,syy,sxy)
    F = q + m*sp - ceff*cpn
    act = F > 0
    dgp = np.zeros_like(sxx)
    if not act.any():
        return sxx, syy, sxy, dgp
    m_t = ceff*cpn/sp
    sx = sxx.copy(); sy = syy.copy(); sxyo = sxy.copy()
    apex = act & (m >= m_t)
    if apex.any():
        sx = np.where(apex, m_t, sx); sy = np.where(apex, m_t, sy); sxyo = np.where(apex, 0.0, sxyo)
        dgp = np.where(apex, np.abs(m_t-m)/a, dgp)
    cone = act & (~apex)
    if cone.any():
        Qv = flowgrad(np.where(cone,sxx,0.0), np.where(cone,syy,0.0), np.where(cone,sxy,0.0), psi)
        CQ = Cdot(Qv)
        den = Qv[0]*CQ[0]+Qv[1]*CQ[1]+2.0*Qv[2]*CQ[2]
        den = np.where(np.abs(den) > 1e-12*Dxx, den, Dxx)
        dlam = np.zeros_like(sxx)
        cap = np.where(cone, 100.0*np.maximum(F,0.0)/np.abs(den), 0.0)
        for it in range(60):
            m2,q2 = mq(sxx-dlam*CQ[0], syy-dlam*CQ[1], sxy-dlam*CQ[2])
            F2 = q2 + m2*sp - ceff*cpn
            step = np.where(cone, F2/np.abs(den), 0.0)
            dlam = np.clip(dlam+step, 0.0, cap)
            if np.max(np.abs(np.where(cone,step,0.0))) < 1e-6*float(ceff.max()):
                break
        dg = dlam*np.sqrt((Qv[0]**2+Qv[1]**2+2.0*Qv[2]**2)*2.0/3.0)
        sx = np.where(cone, sxx-dlam*CQ[0], sx); sy = np.where(cone, syy-dlam*CQ[1], sy)
        sxyo = np.where(cone, sxy-dlam*CQ[2], sxyo)
        dgp = np.where(cone, dg, dgp)
    return sx, sy, sxyo, dgp

def blur(f, r):
    if r < 1: return f
    f = f.copy()
    for ax in (0,1):
        acc = np.zeros_like(f)
        for s in range(-r, r+1): acc += np.roll(f, s, axis=ax)
        f = acc/(2*r+1)
    return f

nx = int(sys.argv[1]) if len(sys.argv)>1 else 60
ny = int(sys.argv[2]) if len(sys.argv)>2 else 120
nsteps = int(sys.argv[3]) if len(sys.argv)>3 else 30000
v0 = float(sys.argv[4]) if len(sys.argv)>4 else 1.0e-3
W,H = 1.0,2.0
dx,dy = W/nx, H/ny
Msc = 1e4
cpe = cp/math.sqrt(Msc)
dt = 0.35*min(dx,dy)/cpe
nr = max(1, int(round(0.05/dx)))
gg = 1.0/math.sqrt(3.0)
gps = [(-gg,-gg),(gg,-gg),(gg,gg),(-gg,gg)]
dNdx=[]; dNdy=[]
for Xi,Eta in gps:
    dNdxi = np.array([-(1-Eta),(1-Eta),(1+Eta),-(1+Eta)])/4.0
    dNdeta = np.array([-(1-Xi),-(1+Xi),(1+Xi),(1-Xi)])/4.0
    dNdx.append(dNdxi*2.0/dx); dNdy.append(dNdeta*2.0/dy)
wt = dx*dy/4.0
corners = [(0,0),(1,0),(1,1),(0,1)]

dec = np.full((nx,ny),-sig3)
sxx=[dec.copy() for _ in range(4)]; syy=[dec.copy() for _ in range(4)]; sxy=[np.zeros((nx,ny)) for _ in range(4)]
gpc = np.zeros((nx,ny))
vx=np.zeros((nx+1,ny+1)); vy=np.zeros((nx+1,ny+1))
trib=np.zeros((nx+1,ny+1))
for di in (0,1):
    for dj in (0,1): trib[di:di+nx,dj:dj+ny]+=0.25*dx*dy
mass = rho*Msc*trib
Xc=(np.arange(nx)+0.5)*dx; Yc=(np.arange(ny)+0.5)*dy
# weak disc imperfection used to seed ONE localised band
imp = ((Xc[:,None]-0.0*W)**2+(Yc[None,:]-0.5*H)**2) < (0.12*W)**2
Yn=(np.arange(ny+1)*dy)[:,None].T.ravel()
wn=np.ones(ny+1); wn[0]=wn[-1]=0.5
damp=float(sys.argv[5]) if len(sys.argv)>5 else 0.2
aff=-2.0*v0/H
t0=time.time(); hist=[]
for n in range(nsteps):
    ceff=np.clip(c0+(cres-c0)*(gpc/gp_ref),cres,c0)
    ceff=np.where(imp,0.5*ceff,ceff)
    dgp_sum=np.zeros((nx,ny))
    for k in range(4):
        vxe=[vx[di:di+nx,dj:dj+ny] for di,dj in corners]
        vye=[vy[di:di+nx,dj:dj+ny] for di,dj in corners]
        exx=sum(dNdx[k][i]*vxe[i] for i in range(4))
        eyy=aff+sum(dNdy[k][i]*vye[i] for i in range(4))
        gxy=sum(dNdy[k][i]*vxe[i]+dNdx[k][i]*vye[i] for i in range(4))
        sxx[k]=sxx[k]+dt*(Dxx*exx+Dxy*eyy)
        syy[k]=syy[k]+dt*(Dxy*exx+Dyy*eyy)
        sxy[k]=sxy[k]+dt*G*gxy
        sxx[k],syy[k],sxy[k],dgp=return_map(sxx[k],syy[k],sxy[k],ceff)
        dgp_sum+=dgp
    gpc=blur(gpc+0.25*dgp_sum,nr)
    fx=np.zeros((nx+1,ny+1)); fy=np.zeros_like(fx)
    for k in range(4):
        for i,(di,dj) in enumerate(corners):
            fx[di:di+nx,dj:dj+ny]-=wt*(dNdx[k][i]*sxx[k]+dNdy[k][i]*sxy[k])
            fy[di:di+nx,dj:dj+ny]-=wt*(dNdy[k][i]*syy[k]+dNdx[k][i]*sxy[k])
    fx[0,1:-1]+=sig3*dy*wn[1:-1]; fx[-1,1:-1]-=sig3*dy*wn[1:-1]
    vx+=(1-damp)*dt*fx/mass; vy+=dt*fy/mass
    # damp only the fluctuation about the affine (strain-rate controlled) field
    vx*=(1-damp); vy*=(1-damp)
    vy[0,:]=0.0; vy[-1,:]=0.0
    if n%500==0:
        mx=sum(0.25*mq(sxx[k],syy[k],sxy[k])[0] for k in range(4))
        qx=sum(0.25*mq(sxx[k],syy[k],sxy[k])[1] for k in range(4))
        ke=0.5*(mass*((vx**2)+(vy**2))).sum()
        hist.append((n*dt*2*v0/H, float(qx.mean()), float(qx.max()), float(gpc.max()), float(gpc.mean()), float(ke)))
        if n%2000==0:
            Fq=qx+mx*sp-ceff*cpn
            print(f"n={n:5d} eps={2*v0*n*dt/H:.4f} m_mean={mx.mean():9.1f} q_mean={qx.mean():8.1f} q_max={qx.max():8.1f} c_min={ceff.min():8.1f} F>0 frac={(Fq>0).mean():.2f} gp_max={gpc.max():.2e} KE={ke:.1e}",flush=True)
print('wall',time.time()-t0)
h=np.array(hist)
mx=sum(0.25*mq(sxx[k],syy[k],sxy[k])[0] for k in range(4))
qx=sum(0.25*mq(sxx[k],syy[k],sxy[k])[1] for k in range(4))
g=gpc; thr=0.5*g.max(); msk=g>thr
X=np.broadcast_to(Xc[:,None],(nx,ny))[msk]; Y=np.broadcast_to(Yc[None,:],(nx,ny))[msk]; w=g[msk]
Xm=(X*w).sum()/w.sum(); Ym=(Y*w).sum()/w.sum()
C=np.array([[((X-Xm)**2*w).sum(),((X-Xm)*(Y-Ym)*w).sum()],[((X-Xm)*(Y-Ym)*w).sum(),((Y-Ym)**2*w).sum()]])/w.sum()
ev,evec=np.linalg.eigh(C); d=evec[:,np.argmax(ev)]
ang=math.degrees(math.atan2(abs(d[1]),abs(d[0])))
print(f'gp_max={g.max():.3e} contrast={g.max()/g.mean():.2f} npts={msk.sum()} angle_from_horizontal={ang:.2f} deg ; Rice 45+(phi+psi)/4={45+(math.degrees(phi)+math.degrees(psi))/4:.2f} ; Coulomb 45+phi/2={45+math.degrees(phi)/2:.2f}')
np.savez('scratch/biax_v5.npz',gpc=gpc,q=qx,m=mx,x=Xc,y=Yc,hist=h,ang=ang)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(9,6))
im=ax[0].imshow(gpc.T,origin='lower',extent=[0,W,0,H],cmap='inferno'); ax[0].set_title('equivalent plastic strain'); plt.colorbar(im,ax=ax[0],fraction=0.046)
ax[0].plot([0,W],[0,W*math.tan(math.radians(ang))],'c--')
ax[1].plot(h[:,0],h[:,1]); ax[1].set_xlabel('axial strain'); ax[1].set_ylabel('q mean')
plt.tight_layout(); plt.savefig('scratch/biax_v5.png',dpi=110)