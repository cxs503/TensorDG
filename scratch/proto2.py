import numpy as np, math, time
E,nu,rho = 30e6,0.3,2000.0
phi,psi = math.radians(40.0), math.radians(15.0)
c0,cres,gp_ref = 20e3, 2e3, 0.02
sig3 = 50e3
a = E/((1+nu)*(1-2*nu))
Dxx,Dyy,Dxy,Ds = a*(1-nu), a*(1-nu), a*nu, a*(1-2*nu)/2
nx,ny = 40,80
W,H = 1.0,2.0
dx,dy = W/nx, H/ny
cp = math.sqrt(Dxx/rho)
Msc = 1e4
dt = 0.4*min(dx,dy)/cp*math.sqrt(Msc)
sp,cpn = math.sin(phi), math.cos(phi)

def p_q(sxx,syy,sxy):
    # tension-positive mean stress m and deviatoric radius q
    return 0.5*(sxx+syy), np.sqrt(0.25*(sxx-syy)**2+sxy**2)

def Cdot(q):
    return np.stack([Dxx*q[0]+Dxy*q[1], Dxy*q[0]+Dyy*q[1], Ds*q[2]])

def flow(sxx,syy,sxy,ang):
    s = 0.25*(sxx-syy); q = np.maximum(np.sqrt(s**2+sxy**2),1e-30)
    return np.stack([0.5*s/q + 0.5*math.sin(ang), -0.5*s/q + 0.5*math.sin(ang), sxy/q])

def return_map(sxx,syy,sxy,ceff):
    p,q=p_q(sxx,syy,sxy)
    F=q+p*sp-ceff*cpn
    act=F>0
    dgp=np.zeros_like(sxx)
    if not act.any():
        return sxx,syy,sxy,dgp
    Qv=flow(sxx,syy,sxy,psi); CQ=Cdot(Qv)
    den=Qv[0]*CQ[0]+Qv[1]*CQ[1]+2.0*Qv[2]*CQ[2]
    dlam=np.zeros_like(sxx)
    for it in range(30):
        p2,q2=p_q(sxx-dlam*CQ[0],syy-dlam*CQ[1],sxy-dlam*CQ[2])
        F2=q2+p2*sp-ceff*cpn
        step=np.where(act, F2/den, 0.0)
        dlam=np.maximum(dlam+step, 0.0)
        if np.max(np.abs(np.where(act,step,0.0)))<1e-6*float(ceff.max()): break
    dgp=dlam*np.sqrt((Qv[0]**2+Qv[1]**2+2*Qv[2]**2)*2.0/3.0)
    return sxx-dlam*CQ[0], syy-dlam*CQ[1], sxy-dlam*CQ[2], dgp

def blur(f,r=2):
    for ax in (0,1):
        acc=np.zeros_like(f)
        for s in range(-r,r+1): acc+=np.roll(f,s,axis=ax)
        f=acc/(2*r+1)
    return f

sxx=np.full((nx,ny),-sig3); syy=np.full((nx,ny),-sig3); sxy=np.zeros((nx,ny))
gp=np.zeros((nx,ny))
Xc=(np.arange(nx)+0.5)*dx; Yc=(np.arange(ny)+0.5)*dy
imp=((Xc[:,None]-0.5*W)**2+(Yc[None,:]-0.5*H)**2)<(0.06*W)**2
vx=np.zeros((nx+1,ny+1)); vy=np.zeros((nx+1,ny+1))
trib=np.zeros((nx+1,ny+1))
for di in (0,1):
    for dj in (0,1):
        trib[di:di+nx,dj:dj+ny]+=0.25*dx*dy
mass=rho*Msc*trib
wn=np.ones(ny+1); wn[0]=wn[-1]=0.5
v0=5.7e-4; damp=0.05
nsteps=20000
t0=time.time()
for n in range(nsteps):
    vx_sw,vx_se,vx_nw,vx_ne = vx[:-1,:-1],vx[1:,:-1],vx[:-1,1:],vx[1:,1:]
    vy_sw,vy_se,vy_nw,vy_ne = vy[:-1,:-1],vy[1:,:-1],vy[:-1,1:],vy[1:,1:]
    exx=((vx_se+vx_ne)-(vx_sw+vx_nw))/(2*dx)
    eyy=((vy_nw+vy_ne)-(vy_sw+vy_se))/(2*dy)
    exy=0.5*(((vx_nw+vx_ne)-(vx_sw+vx_se))/(2*dy)+((vy_se+vy_ne)-(vy_sw+vy_nw))/(2*dx))
    sxx=sxx+dt*(Dxx*exx+Dxy*eyy); syy=syy+dt*(Dxy*exx+Dyy*eyy); sxy=sxy+dt*Ds*exy
    pv=0.1*rho*cp*min(dx,dy)*np.maximum(0.0,-(exx+eyy)); sxx-=pv; syy-=pv
    ceff=np.clip(c0+(cres-c0)*(gp/gp_ref),cres,c0)
    ceff=np.where(imp,0.8*ceff,ceff)
    sxx,syy,sxy,dgp=return_map(sxx,syy,sxy,ceff)
    gp=blur(gp+dgp,2)
    fx=np.zeros((nx+1,ny+1)); fy=np.zeros_like(fx)
    fx[:-1,:-1]+=-sxx*dy/2-sxy*dx/2; fx[:-1,1:]+=-sxx*dy/2+sxy*dx/2
    fx[1:,:-1]+= sxx*dy/2-sxy*dx/2; fx[1:,1:]+= sxx*dy/2+sxy*dx/2
    fy[:-1,:-1]+=-sxy*dy/2-syy*dx/2; fy[:-1,1:]+=-sxy*dy/2+syy*dx/2
    fy[1:,:-1]+= sxy*dy/2-syy*dx/2; fy[1:,1:]+= sxy*dy/2+syy*dx/2
    fx[0,1:-1]+=sig3*dy*wn[1:-1]; fx[-1,1:-1]-=sig3*dy*wn[1:-1]
    vx+=(1-damp)*dt*fx/mass; vy+=(1-damp)*dt*fy/mass
    vy[0,:]=v0; vy[-1,:]=-v0
    vx[0,:]=np.maximum(vx[0,:],0.0); vx[-1,:]=np.minimum(vx[-1,:],0.0)
    sxy[:,0]=0.0; sxy[:,-1]=0.0
    if n%500==0 and n<3000 or n%2000==0:
        p,q=p_q(sxx,syy,sxy)
        print(f"n={n:6d} eps_ax={2*v0*n*dt/H:.4f} q_mean={q.mean():9.1f} p_mean={p.mean():9.1f} gp_max={gp.max():.4f} gp_mean={gp.mean():.5f} KE={0.5*(mass*((vx**2)+(vy**2))).sum():.3e}", flush=True)
print('wall', time.time()-t0, flush=True)
p,q=p_q(sxx,syy,sxy)
g=gp; thr=0.3*g.max(); m=g>thr
X=np.broadcast_to(Xc[:,None],(nx,ny))[m]; Y=np.broadcast_to(Yc[None,:],(nx,ny))[m]; w=g[m]
Xm=(X*w).sum()/w.sum(); Ym=(Y*w).sum()/w.sum()
C=np.array([[((X-Xm)**2*w).sum(), ((X-Xm)*(Y-Ym)*w).sum()],[((X-Xm)*(Y-Ym)*w).sum(), ((Y-Ym)**2*w).sum()]])/w.sum()
ev,evec=np.linalg.eigh(C)
d=evec[:,np.argmax(ev)]
ang=math.degrees(math.atan2(abs(d[1]),abs(d[0])))
print(f'gp_max={g.max():.4f} npts={m.sum()} band axis=({d[0]:+.3f},{d[1]:+.3f}) angle_from_horizontal={ang:.2f} deg ; Rice 45+(phi+psi)/4={45+(math.degrees(phi)+math.degrees(psi))/4:.2f}')
np.save('scratch/biax_gp.npy',gp); np.save('scratch/biax_q.npy',q); np.save('scratch/biax_p.npy',p); np.save('scratch/biax_s.npy',np.stack([sxx,syy,sxy]))