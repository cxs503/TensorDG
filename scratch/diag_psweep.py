import numpy as np
from tensordg.curved_geometry import CurvedMesh, IsoparametricMap
from tensordg.euler_curved import CurvilinearEuler

def periodic_channel(nx, ny, lx=4.0, ly=1.0, degree=1):
    maps = [[IsoparametricMap.affine(lx*i/nx, lx*(i+1)/nx, ly*j/ny, ly*(j+1)/ny, degree)
             for j in range(ny)] for i in range(nx)]
    return CurvedMesh(maps, periodic=(True, True), walls=(False,)*4)

def rest(x, y):
    return (np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.ones_like(x))

for p in (0, 1, 2):
    solver = CurvilinearEuler(periodic_channel(8, 6), p)
    s0 = solver.project(rest)
    dt = solver.stable_dt(s0)
    rng = np.random.default_rng(5)
    st = s0 + rng.standard_normal(s0.shape)*1e-9
    # physical L2 norm of perturbation: sqrt(mean of (dU)^2 over volume quadrature)
    def phys(state):
        d = state - s0
        sampled = np.einsum("qk,nklc,rl->nqrc", solver.basis, d, solver.basis)
        w = solver.w
        vol = np.einsum("q,r,nqrc,q,r->", w, w, sampled**2, w, w)
        tot = np.einsum("q,r,q,r->", w, w, w, w)*sampled.shape[0]
        return np.sqrt(vol/tot)
    n0 = phys(st)
    if not np.isfinite(n0) or n0 == 0:
        print(f"p={p}: zero perturbation norm (K={solver.K})"); continue
    hist = []
    blew = False
    for n in range(400):
        st = solver.ssp_rk3_step(st, dt)
        cur = phys(st)
        if not np.isfinite(cur) or cur > 1e3*n0:
            blew = True; break
        hist.append(cur/n0)
    print(f"p={p} K={solver.K} dt={dt:.5f} blew={blew} steps={len(hist)} "
          f"rel_L2 after 20/100/400 steps: "
          f"{hist[19]:.3e} {hist[min(99,len(hist)-1)]:.3e} {hist[-1]:.3e}")
