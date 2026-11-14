import numpy as np, math
exec(open('scratch/proto2.py').read().split('sxx=np.full')[0])

# (a) return-map unit check on a biaxial trial state
def trial_check():
    sxx=np.full((2,2),-50000.0); syy=sxx.copy(); sxy=np.zeros((2,2))
    sxx = sxx - 200e3  # axial compression increment
    ceff=np.full((2,2),20e3)
    o=return_map(sxx,syy,sxy,ceff)
    m,q=p_q(*o[:3]); F=q+m*sp-ceff*cpn
    print('return map: F residual max', np.abs(F).max(), 'm', m.ravel()[0], 'q', q.ravel()[0])
    # consistency: check that the increment is non-negative volumetric (dilation)
    tr=(o[0]-sxx)+(o[1]-syy)
    print('vol plastic strain (should be >0 dilation):', tr.ravel()[0])
trial_check()

# (b) 60-step march with diagnostics
nx,ny=40,80; W,H=1.0,2.0; dx,dy=W/nx,H/ny
sxx=np.full((nx,ny),-50e3); syy=sxx.copy(); sxy=np.zeros((nx,ny)); gp=np.zeros((nx,ny))
vx=np.zeros((nx+1,ny+1)); vy=np.zeros_like(vx)
trib=np.zeros((nx+1,ny+1))
for di in (0,1):
    for dj in (0,1): trib[di:di+nx,dj:dj+ny]+=0.25*dx*dy
mass=rho*1e4*trib
wn=np.ones(ny+1); wn[0]=wn[-1]=0.5
v0=5.7e-4; damp=0.05
print('dt',dt,'cp',cp)
for n in range(60):
    vx_sw,vx_se,vx_nw,vx_ne = vx[:-1,:-1],vx[1:,:-1],vx[:-1,1:],vx[1:,1:]
    vy_sw,vy_se,vy_nw,vy_ne = vy[:-1,:-1],vy[1:,:-1],vy[:-1,1:],vy[1:,1:]
    exx=((vx_se+vx_ne)-(vx_sw+vx_nw))/(2*dx); eyy=((vy_nw+vy_ne)-(vy_sw+vy_se))/(2*dy)
    exy=0.5*(((vx_nw+vx_ne)-(vx_sw+vx_se))/(2*dy)+((vy_se+vy_ne)-(vy_sw+vy_nw))/(2*dx))
    sxx=sxx+dt*(Dxx*exx+Dxy*eyy); syy=syy+dt*(Dxy*exx+Dyy*eyy); sxy=sxy+dt*Ds*exy
    ceff=np.full((nx,ny),20e3)
    sxx,syy,sxy,dgp=return_map(sxx,syy,sxy,ceff)
    gp=blur(gp+dgp,2)
    fx=np.zeros((nx+1,ny+1)); fy=np.zeros_like(fx)
    fx[:-1,:-1]+=-sxx*dy/2-sxy*dx/2; fx[:-1,1:]+=-sxx*dy/2+sxy*dx/2
    fx[1:,:-1]+= sxx*dy/2-sxy*dx/2; fx[1:,1:]+= sxx*dy/2+sxy*dx/2
    fy[:-1,:-1]+=-sxy*dy/2-syy*dx/2; fy[:-1,1:]+=-sxy*dy/2+syy*dx/2
    fy[1:,:-1]+= sxy*dy/2-syy*dx/2; fy[1:,1:]+= sxy*dy/2+syy*dx/2
    fx[0,1:-1]+=50e3*dy*wn[1:-1]; fx[-1,1:-1]-=50e3*dy*wn[1:-1]
    vx+=(1-damp)*dt*fx/mass; vy+=(1-damp)*dt*fy/mass
    vy[0,:]=v0; vy[-1,:]=-v0
    vx[0,:]=np.maximum(vx[0,:],0.0); vx[-1,:]=np.minimum(vx[-1,:],0.0)
    sxy[:,0]=0.0; sxy[:,-1]=0.0
    if n<12 or n%10==0:
        m,q=p_q(sxx,syy,sxy)
        print(f"{n:3d} |v|max={max(abs(vx).max(),abs(vy).max()):.3e} |s|max={max(abs(sxx).max(),abs(syy).max(),abs(sxy).max()):.3e} qmax={q.max():.3e} gp={gp.max():.3e} fin={np.isfinite(sxx).all()}")