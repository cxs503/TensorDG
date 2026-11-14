import numpy as np, math, time

# ---------------- material ----------------
E,nu,rho = 30e6,0.3,2000.0
phi,psi = math.radians(40.0), math.radians(15.0)
c0,cres,gp_ref = 25e3, 2.5e3, 0.02
sig3 = 50e3
a = E/((1+nu)*(1-2*nu))
Dxx,Dyy,Dxy,G = a*(1-nu), a*(1-nu), a*nu, a*(1-2*nu)/2
sp,cpn = math.sin(phi), math.cos(phi); sps=math.sin(psi)
cp = math.sqrt(Dxx/rho)

def mq(sxx,syy,sxy):
    return 0.5*(sxx+syy), np.sqrt(0.25*(sxx-syy)**2+sxy**2)

def Cdot(q):
    return np.stack([Dxx*q[0]+Dxy*q[1], Dxy*q[0]+Dyy*q[1], G*q[2]])

def flowgrad(sxx,syy,sxy,ang):
    s=0.25*(sxx-syy); q=np.maximum(np.sqrt(s*s+sxy*sxy),1e-30); sa=math.sin(ang)
    return np.stack([0.5*s/q+0.5*sa, -0.5*s/q+0.5*sa, sxy/q])

def return_map(sxx,syy,sxy,ceff):
    m,q = mq(sxx,syy,sxy)
    F = q + m*sp - ceff*cpn
    act = F > 0
    dgp = np.zeros_like(sxx)
    if not act.any():
        return sxx,syy,sxy,dgp
    Qv = flowgrad(sxx,syy,sxy,psi); CQ = Cdot(Qv)
    den = Qv[0]*CQ[0]+Qv[1]*CQ[1]+2.0*Qv[2]*CQ[2]
    dlam = np.zeros_like(sxx)
    for it in range(50):
        m2,q2 = mq(sxx-dlam*CQ[0], syy-dlam*CQ[1], sxy-dlam*CQ[2])
        F2 = q2 + m2*sp - ceff*cpn
        step = np.where(act, F2/den, 0.0)
        dlam = np.maximum(dlam+step, 0.0)
        if np.max(np.abs(np.where(act,step,0.0))) < 1e-8*float(ceff.max()):
            break
    dgp = dlam*np.sqrt((Qv[0]**2+Qv[1]**2+2*Qv[2]**2)*2.0/3.0)
    return sxx-dlam*CQ[0], syy-dlam*CQ[1], sxy-dlam*CQ[2], dgp

# ---------------- Q4 explicit dynamic relaxation ----------------
nx,ny = 40,80
W,H = 1.0,2.0
dx,dy = W/nx, H/ny
Msc = 1e4
cpe = cp/math.sqrt(Msc)
dt = 0.35*min(dx,dy)/cpe
g = 1.0/math.sqrt(3.0)
# node ordering: 1=SW 2=SE 3=NE 4=NW ; dN/dxi, dN/deta at 4 gauss points
gp_list=[(-g,-g),(g,-g),(g,g),(-g,g)]
dNdxi=[]; dNdeta=[]
for Xi,Eta in gp_list:
    dNdxi.append(np.array([-(1-Eta), (1-Eta), (1+Eta), -(1+Eta)])/4.0)
    dNdeta.append(np.array([-(1-Xi), -(1+Xi), (1+Xi), (1-Xi)])/4.0)
w = dx*dy/4.0
nodes4=[(0,0),(1,0),(1,1),(0,1)]  # (di,dj)

def run(nsteps=20000, v0=5.7e-4, damp=0.05, imp_strength=0.8, monitor=2000, np_=None):
    nx_,ny_=nx,ny
    sxx=[np.full((nx_,ny_),-sig3) for _ in range(4)]; syy=[np.full((nx_,ny_),-sig3) for _ in range(4)]
    sxy=[np.zeros((nx_,ny_)) for _ in range(4)]
    gp_=np.zeros((nx_,ny_))
    vx=np.zeros((nx_+1,ny_+1)); vy=np.zeros((nx_+1,ny_+1))
    trib=np.zeros((nx_+1,ny_+1))
    for di in (0,1):
        for dj in (0,1): trib[di:di+nx_,dj:dj+ny_]+=0.25*dx*dy
    mass=rho*Msc*trib
    Xc=(np.arange(nx_)+0.5)*dx; Yc=(np.arange(ny_)+0.5)*dy
    imp=((Xc[:,None]-0.5*W)**2+(Yc[None,:]-0.5*H)**2)<(0.06*W)**2
    wn=np.ones(ny_+1); wn[0]=wn[-1]=0.5
    hist=[]
    for n in range(nsteps):
        ceff=np.clip(c0+(cres-c0)*(gp_/gp_ref),cres,c0)
        ceff=np.where(imp,imp_strength*ceff,ceff)
        fx=np.zeros((nx_+1,ny_+1)); fy=np.zeros_like(fx)
        for k,(Xi,Eta) in enumerate(gp_list):
            # element nodal velocity gather
            vxe=[vx[di:di+nx_, dj:dj+ny_] for di,dj in nodes4]
            vye=[vy[di:di+nx_, dj:dj+ny_] for di,dj in nodes4]
            dNx=[dNdxi[k][i]*2.0/dx for i in range(4)]
            dNy=[dNdeta[k][i]*2.0/dy for i in range(4)]
            exx=sum(dNx[i]*vxe[i] for i in range(4))
            eyy=sum(dNy[i]*vye[i] for i in range(4))
            gxy=sum(dNy[i]*vxe[i]+dNx[i]*vye[i] for i in range(4))
            sxx[k]=sxx[k]+dt*(Dxx*exx+Dxy*eyy)
            syy[k]=syy[k]+dt*(Dxy*exx+Dyy*eyy)
            sxy[k]=sxy[k]+dt*G*gxy
            sxx[k],syy[k],sxy[k],dgp=return_map(sxx[k],syy[k],sxy[k],ceff)
        # internal nodal forces
        fxe=[np.zeros((nx_,ny_)) for _ in range(4)]; fye=[np.zeros((nx_,ny_)) for _ in range(4)]
        for k in range(4):
            dNx=[dNdxi[k][i]*2.0/dx for i in range(4)]
            dNy=[dNdeta[k][i]*2.0/dy for i in range(4)]
            for i in range(4):
                fxe[i]+=w*(dNx[i]*sxx[k]+dNy[i]*sxy[k])
                fye[i]+=w*(dNy[i]*syy[k]+dNx[i]*sxy[k])
        for i,(di,dj) in enumerate(nodes4):
            fx[di:di+nx_, dj:dj+ny_]+=fxe[i]
            fy[di:di+nx_, dj:dj+ny_]+=fye[i]
        # plastic strain: average gauss points to cell centre for the nonlocal softening field
        gp_=sum([return_map(sxx[k],syy[k],sxy[k],ceff)[3]*0 for k in range(4)])  # placeholder
        fx[0,1:-1]+=sig3*dy*wn[1:-1]; fx[-1,1:-1]-=sig3*dy*wn[1:-1]
        vx+=(1-damp)*dt*fx/mass; vy+=(1-damp)*dt*fy/mass
        vy[0,:]=v0; vy[-1,:]=-v0
        sxy[:]=[np.zeros((nx_,ny_)) for _ in range(4)]
    return None

print('this file is a scratch prototype; see proto4.py')