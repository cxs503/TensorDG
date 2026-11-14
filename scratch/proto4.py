"""Scratch prototype: plane-strain biaxial compression of a Mohr-Coulomb softening
solid on Q4 bilinear elements with 2x2 Gauss integration, lumped mass, central
difference (dynamic relaxation).  Goal: measure the shear-band inclination."""
import numpy as np, math, time, sys

E,nu,rho = 30e6, 0.3, 2000.0
phi,psi = math.radians(40.0), math.radians(15.0)
c0, cres, gp_ref = 25e3, 2.5e3, 0.02
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
    """Implicit radial return to the plane-strain Mohr-Coulomb cone, with an
    apex/tension cap at m = c*cot(phi).  ceff is the (softened) cohesion."""
    m, q = mq(sxx,syy,sxy)
    F = q + m*sp - ceff*cpn
    act = F > 0
    dgp = np.zeros_like(sxx)
    if not act.any():
        return sxx, syy, sxy, dgp
    m_t = ceff*cpn/sp            # apex mean stress (tension positive)
    sx = sxx.copy(); sy = syy.copy(); sxyo = sxy.copy()
    # --- apex branch: state at or beyond the cone tip -> project onto the tip
    apex = act & (m >= m_t)
    if apex.any():
        sx = np.where(apex, m_t, sx); sy = np.where(apex, m_t, sy)
        sxyo = np.where(apex, 0.0, sxyo)
        dgp = np.where(apex, np.abs(m_t - m)/a, dgp)
    # --- cone branch: capped Newton on the plastic multiplier
    cone = act & (~apex)
    if cone.any():
        Qv = flowgrad(np.where(cone, sxx, 0.0), np.where(cone, syy, 0.0), np.where(cone, sxy, 0.0), psi)
        CQ = Cdot(Qv)
        den = Qv[0]*CQ[0] + Qv[1]*CQ[1] + 2.0*Qv[2]*CQ[2]
        den = np.where(np.abs(den) > 1e-12*float(ceff.max())*Dxx, den, Dxx)
        dlam = np.zeros_like(sxx)
        cap = np.where(cone, 100.0*np.maximum(F, 0.0)/np.abs(den), 0.0)
        for it in range(60):
            m2, q2 = mq(sxx-dlam*CQ[0], syy-dlam*CQ[1], sxy-dlam*CQ[2])
            F2 = q2 + m2*sp - ceff*cpn
            step = np.where(cone, F2/np.abs(den), 0.0)
            dlam = np.clip(dlam+step, 0.0, cap)
            if np.max(np.abs(np.where(cone, step, 0.0))) < 1e-6*float(ceff.max()):
                break
        dg = dlam*np.sqrt((Qv[0]**2+Qv[1]**2+2.0*Qv[2]**2)*2.0/3.0)
        sx = np.where(cone, sxx-dlam*CQ[0], sx); sy = np.where(cone, syy-dlam*CQ[1], sy)
        sxyo = np.where(cone, sxy-dlam*CQ[2], sxyo)
        dgp = np.where(cone, dg, dgp)
    return sx, sy, sxyo, dgp

def blur(f, r=2):
    for ax in (0,1):
        acc = np.zeros_like(f)
        for s in range(-r, r+1):
            acc += np.roll(f, s, axis=ax)
        f = acc/(2*r+1)
    return f

nx, ny = int(sys.argv[1]) if len(sys.argv)>1 else 40, int(sys.argv[2]) if len(sys.argv)>2 else 80
W, H = 1.0, 2.0
dx, dy = W/nx, H/ny
Msc = 1e4
cpe = cp/math.sqrt(Msc)
dt = 0.35*min(dx,dy)/cpe
gg = 1.0/math.sqrt(3.0)
gps = [(-gg,-gg),(gg,-gg),(gg,gg),(-gg,gg)]
dNdx = []; dNdy = []
for Xi,Eta in gps:
    dNdxi = np.array([-(1-Eta),(1-Eta),(1+Eta),-(1+Eta)])/4.0
    dNdeta = np.array([-(1-Xi),-(1+Xi),(1+Xi),(1-Xi)])/4.0
    dNdx.append(dNdxi*2.0/dx); dNdy.append(dNdeta*2.0/dy)
wt = dx*dy/4.0
corners = [(0,0),(1,0),(1,1),(0,1)]

dec = np.full((nx,ny),-sig3); sxx=[dec.copy() for _ in range(4)]; syy=[dec.copy() for _ in range(4)]
sxy=[np.zeros((nx,ny)) for _ in range(4)]
gpc = np.zeros((nx,ny))
vx=np.zeros((nx+1,ny+1)); vy=np.zeros((nx+1,ny+1))
trib=np.zeros((nx+1,ny+1))
for di in (0,1):
    for dj in (0,1): trib[di:di+nx,dj:dj+ny]+=0.25*dx*dy
mass = rho*Msc*trib
Xc=(np.arange(nx)+0.5)*dx; Yc=(np.arange(ny)+0.5)*dy
imp=((Xc[:,None]-0.5*W)**2+(Yc[None,:]-0.5*H)**2)<(0.06*W)**2
wn=np.ones(ny+1); wn[0]=wn[-1]=0.5
v0=5.7e-4; damp=0.05
nsteps=20000
t0=time.time()
peak_q=0.0
for n in range(nsteps):
    ceff=np.clip(c0+(cres-c0)*(gpc/gp_ref),cres,c0)
    ceff=np.where(imp,0.8*ceff,ceff)
    dgp_sum=np.zeros((nx,ny))
    for k in range(4):
        vxe=[vx[di:di+nx,dj:dj+ny] for di,dj in corners]
        vye=[vy[di:di+nx,dj:dj+ny] for di,dj in corners]
        exx=sum(dNdx[k][i]*vxe[i] for i in range(4))
        eyy=sum(dNdy[k][i]*vye[i] for i in range(4))
        gxy=sum(dNdy[k][i]*vxe[i]+dNdx[k][i]*vye[i] for i in range(4))
        sxx[k]=sxx[k]+dt*(Dxx*exx+Dxy*eyy)
        syy[k]=syy[k]+dt*(Dxy*exx+Dyy*eyy)
        sxy[k]=sxy[k]+dt*G*gxy
        sxx[k],syy[k],sxy[k],dgp=return_map(sxx[k],syy[k],sxy[k],ceff)
        dgp_sum+=dgp
    gpc=blur(gpc+0.25*dgp_sum,2)
    fx=np.zeros((nx+1,ny+1)); fy=np.zeros_like(fx)
    for k in range(4):
        for i,(di,dj) in enumerate(corners):
            fx[di:di+nx,dj:dj+ny]-=wt*(dNdx[k][i]*sxx[k]+dNdy[k][i]*sxy[k])
            fy[di:di+nx,dj:dj+ny]-=wt*(dNdy[k][i]*syy[k]+dNdx[k][i]*sxy[k])
    fx[0,1:-1]+=sig3*dy*wn[1:-1]; fx[-1,1:-1]-=sig3*dy*wn[1:-1]
    vx+=(1-damp)*dt*fx/mass; vy+=(1-damp)*dt*fy/mass
    vy[0,:]=v0; vy[-1,:]=-v0
    if n%1000==0:
        mx=sum(0.25*mq(sxx[k],syy[k],sxy[k])[0] for k in range(4))
        qx=sum(0.25*mq(sxx[k],syy[k],sxy[k])[1] for k in range(4))
        peak_q=max(peak_q,qx.max())
        ke=0.5*(mass*((vx**2)+(vy**2))).sum()
        print(f"n={n:6d} eps={2*v0*n*dt/H:.4f} m={mx.mean():10.1f} q_max={qx.max():10.1f} q_mean={qx.mean():10.1f} gp_max={gpc.max():.4f} KE={ke:.3e} dt={dt:.3e}",flush=True)
print('wall',time.time()-t0)
mx=sum(0.25*mq(sxx[k],syy[k],sxy[k])[0] for k in range(4))
qx=sum(0.25*mq(sxx[k],syy[k],sxy[k])[1] for k in range(4))
g=gpc; thr=0.3*g.max(); msk=g>thr
X=np.broadcast_to(Xc[:,None],(nx,ny))[msk]; Y=np.broadcast_to(Yc[None,:],(nx,ny))[msk]; w=g[msk]
Xm=(X*w).sum()/w.sum(); Ym=(Y*w).sum()/w.sum()
C=np.array([[((X-Xm)**2*w).sum(),((X-Xm)*(Y-Ym)*w).sum()],[((X-Xm)*(Y-Ym)*w).sum(),((Y-Ym)**2*w).sum()]])/w.sum()
ev,evec=np.linalg.eigh(C)
d=evec[:,np.argmax(ev)]
ang=math.degrees(math.atan2(abs(d[1]),abs(d[0])))
print(f'gp_max={g.max():.4f} npts={msk.sum()} angle_from_horizontal={ang:.2f} deg ; Rice 45+(phi+psi)/4={45+(math.degrees(phi)+math.degrees(psi))/4:.2f} ; Coulomb 45+phi/2={45+math.degrees(phi)/2:.2f}')
np.savez('scratch/biax_last.npz',gpc=gpc,q=qx,m=mx,vx=vx,vy=vy,x=Xc,y=Yc)