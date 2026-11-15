import numpy as np
from tensordg.euler_curved import CurvilinearEuler, warped_box_mesh
for p in (1,2,3):
    m = warped_box_mesh(4,4,3,length=10.0,amplitude=0.06)
    s = CurvilinearEuler(m, degree=p)
    st = s.project(lambda x,y: (1.0,0.3,-0.2,1.0))
    c = np.zeros_like(st); c[:,0,0,:] = st[:,0,0,:]
    r = s.rhs(c)
    print("p",p,"max|rhs|",np.abs(r).max(),"max|U|",np.abs(c).max(),"dt",s.stable_dt(c))
    U = c.copy(); dt = s.stable_dt(c)
    for k in range(200):
        U = s.ssp_rk3_step(U, dt)
        if not np.all(np.isfinite(U)):
            print("  NaN at step", k, "max|U|", np.abs(U).max()); break
    else:
        print("  drift", np.abs(U-c).max())
