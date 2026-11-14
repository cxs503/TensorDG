import numpy as np, math, time
# cell-centred stress / vertex velocity staggered explicit scheme, 2D plane strain MC
E,nu,rho = 30e6,0.3,2000.0
phi,psi = math.radians(40), math.radians(15)
c0,cres = 20e3, 2e3
gp_ref = 0.02
sig3 = 50e3
a=E/((1+nu)*(1-2*nu))
Dv=np.array([a*(1-nu),a*nu,a*(1-2*nu)/2])  # C: for (exx,eyy,gxy) sigma = Dxx*exx+Dxy*eyy, ...
Dxx,Dyy,Dxy,Ds = a*(1-nu), a*(1-nu), a*nu, a*(1-2*nu)/2

nx,ny = 40,80
W,H = 1.0,2.0
dx,dy = W/nx, H/ny
cp = math.sqrt((a*(1-nu))/rho)
dt = 0.4*min(dx,dy)/cp
Msc = 1e4
def hstep(c,gp): return max(0.0, 0.0)  # placeholder

def p_q(sxx,syy,sxy):
    p = -0.5*(sxx+syy)
    q = np.sqrt(0.25*(sxx-syy)**2 + sxy**2)
    return p,q

def flow_dir(sxx,syy,sxy,ang):
    p,q = p_q(sxx,syy,sxy)
    s = 0.25*(sxx-syy)
    qq = np.maximum(q,1e-30)
    dq = np.stack([np.ones_like(s)*s/qq*0.5, -s/qq*0.5, sxy/qq],axis=0)
    dp = np.stack([-0.5*np.ones_like(s), -0.5*np.ones_like(s), 0*s],axis=0)
    return dp + math.sin(ang)*dq

def Cdot(Qv):  # apply elastic operator C (2D plane strain) to voigt vector stack
    qxx,qyy,qxy = Qv
    return np.stack([Dxx*qxx+Dxy*qyy, Dxy*qxx+Dyy*qyy, Ds*qxy],axis=0)

def return_map(sig, ceff, dlam_scale=1.0):
    sxx,syy,sxy = sig
    c = ceff
    out=[sxx.copy(),syy.copy(),sxy.copy()]
    dgp = np.zeros_like(sxx)
    active = np.ones_like(sxx,bool)
    for it in range(30):
        p,q = p_q(out[0],out[1],out[2])
        F = q + p*math.sin(phi) - c*math.cos(phi)
        act = (F>0)&active
        if not act.any(): break
        Qv = flow_dir(out[0],out[1],out[2],psi)
        CQ = Cdot(Qv)
        # dF along -dlam*CQ
        p2,q2 = p_q(out[0]-CQ[0], out[1]-CQ[1], out[2]-CQ[2])
        F2 = q2 + p2*math.sin(phi) - c*math.cos(phi)
        den = (F-F2)
        dl = np.where(np.abs(den)>1e-12, F/den, 0.0)
        dl = np.where(act, np.maximum(dl,0.0), 0.0)
        out[0]=np.where(act, out[0]-dl*CQ[0], out[0])
        out[1]=np.where(act, out[1]-dl*CQ[1], out[1])
        out[2]=np.where(act, out[2]-dl*CQ[2], out[2])
        dgp = dgp + dl*np.sqrt((Qv[0]**2+Qv[1]**2+2*Qv[2]**2)*2/3)
        # keep active set for the next iteration only where F>0 (recheck)
        active = act
    return out[0],out[1],out[2],dgp

def smooth(gp,r=3):
    if r<=0: return gp
    out=gp.copy()
    tmp=np.zeros_like(gp)
    for _ in range(2):
        tmp[...] = out
        for i in range(-r,r+1):
            tmp += np.roll(out,i,axis=1)*0.0  # noop
    # gaussian-ish via repeated box blur
    k=r
    f=out.copy()
    for ax in (0,1):
        acc=np.zeros_like(f); cnt=0
        for s in range(-k,k+1):
            acc += np.roll(f,s,axis=ax); cnt+=1
        f=acc/cnt
    return f

sxx=np.zeros((nx,ny)); syy=np.zeros((nx,ny)); sxy=np.zeros((nx,ny))
sxx-=sig3; syy-=sig3
gp=np.zeros((nx,ny))
vx=np.zeros((nx+1,ny+1)); vy=np.zeros((nx+1,ny+1))
mass = rho*Msc*dx*dy/4.0
damp = 0.05
v0 = 0.0143
def c_eff(gp): return np.clip(c0 + (cres-c0)*(gp/gp_ref), cres, c0)
t=0.0
nsteps=30000
t0=time.time()
for n in range(nsteps):
    # strain rate at cells
    exx = ((vx[1:, :]+vx[1:, 1:]) - (vx[:-1,:]+vx[:-1,1:]))/(2*dx)
    eyy = ((vy[:, 1:]+vy[1:, 1:]) - (vy[:, :-1]+vy[1:,:-1]))/(2*dy)
    exy = 0.5*(((vx[:,1:]+vx[1:,1:])-(vx[:,:-1]+vx[1:,:-1]))/(2*dy) + ((vy[1:,:]+vy[1:,1:])-(vy[:-1,:]+vy[:-1,1:]))/(2*dx))
    sxx = sxx + dt*(Dxx*exx+Dxy*eyy); syy = syy + dt*(Dxy*exx+Dyy*eyy); sxy = sxy + dt*Ds*exy
    sxx,syy,sxy,dgp = return_map((sxx,syy,sxy), c_eff(gp))
    gp = gp + dgp
    gp = smooth(gp,3)
    # nodal forces
    Sxx=sxx; Syy=syy; Sxy=sxy
    fx = np.zeros((nx+1,ny+1)); fy=np.zeros((nx+1,ny+1))
    # x-traction on vertical faces: right/left
    fr = np.zeros((nx+1,ny+1)); fl=fr.copy(); ft=fr.copy(); fb=fr.copy()
    # assemble: for cell (i,j), node (i,j) gets -sxx*dy/2 etc. use accumulate
    for (di,dj,sx_,sy_,sxxf,syyf,sxyf) in ((0,0,-1,-1,1,1,1),):
        pass
    contribx = np.zeros((nx+1,ny+1)); contriby=np.zeros_like(contribx)
    # node (i,j) from cell (i,j) corner (lower-left): -sxx*dy/2 (left face), -sxy*dx/2 (bottom face)
    contribx[:-1,:-1] += -Sxx*dy/2 - 0*0
    contribx[:-1,:-1] += 0
    # do properly with 4 contributions
    contribx = np.zeros((nx+1,ny+1)); contriby=np.zeros((nx+1,ny+1))
    contribx[:-1,:-1] += (-Sxx*dy/2 - Sxy*dx/2)
    contribx[:-1,1:]  += (-Sxx*dy/2 + Sxy*dx/2)
    contribx[1:,:-1]  += ( Sxx*dy/2 - Sxy*dx/2)
    contribx[1:,1:]   += ( Sxx*dy/2 + Sxy*dx/2)
    contriby[:-1,:-1] += (-Sxy*dy/2 - Syy*dx/2)
    contriby[:-1,1:]  += (-Sxy*dy/2 + Syy*dx/2)
    contriby[1:,:-1]  += ( Sxy*dy/2 - Syy*dx/2)
    contriby[1:,1:]   += ( Sxy*dy/2 + Syy*dx/2)
    vx += dt*contribx/mass; vy += dt*contriby/mass
    vx *= (1-damp); vy *= (1-damp)
    # BCs: platens
    vy[0,:] = +v0; vy[-1,:] = -v0
    # confining pressure on lateral faces
    vx[0,:] += dt*(-sig3*dy/2)/mass*2*0  # handled below
    # apply lateral pressure as nodal force
    vx[0,1:-1] += dt*(-sig3*dy)/mass*(mass/mass)  # approximate
    vx[-1,1:-1] += dt*(+sig3*dy)/mass
    # frictionless platens: zero shear in top/bottom cell rows
    sxy[:,0]=0.0; sxy[:,-1]=0.0
    vx[0,:]=vx[0,:]; 
    if n%3000==0:
        p,q=p_q(sxx,syy,sxy)
        print(f"n={n} t={n*dt:.4f} q_mean={q.mean():.1f} q_peak={q.max():.1f} p_mean={p.mean():.1f} gp_max={gp.max():.4f} exx_mean={exx.mean():.4e} KE={0.5*mass*((vx**2).sum()+(vy**2).sum()):.3e}")
print('steps/s', nsteps/(time.time()-t0))
p,q=p_q(sxx,syy,sxy)
print('final axial strain', (v0*nsteps*dt*2)/H, 'q_peak', q.max(), 'p', p.mean())
g=gp
thr=0.3*g.max()
yy,xx=np.nonzero(g>thr)
w=g[g>thr]
X=xx*dx; Y=yy*dy
Xm=(X*w).sum()/w.sum(); Ym=(Y*w).sum()/w.sum()
Cxx=((X-Xm)**2*w).sum()/w.sum(); Cyy=((Y-Ym)**2*w).sum()/w.sum(); Cxy=((X-Xm)*(Y-Ym)*w).sum()/w.sum()
ev,evec=np.linalg.eigh(np.array([[Cxx,Cxy],[Cxy,Cyy]]))
d=evec[:,np.argmax(ev)]
print('band axis direction', d, 'angle from horizontal deg', math.degrees(math.atan2(abs(d[1]),abs(d[0]))))
