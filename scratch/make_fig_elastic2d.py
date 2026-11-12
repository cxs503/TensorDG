import math, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensordg.elastic_2d import (ElasticMedium, plane_wave, solve_elastic_2d,
                                 legendre_operators_2d, _rhs)
med = ElasticMedium(); k = 2*math.pi
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ns = (8, 16, 32, 64)
for deg in (1, 2, 3):
    ex = plane_wave("P", k, 0.0, 1.0, med); errs = []
    for n in ns:
        res = solve_elastic_2d(lambda x, y: ex(x, y, 0.0), nx=n, ny=1, degree=deg,
                               final_time=0.02, cfl=0.1)
        xg, yg = res.quadrature_grid(); num, exf = res.fields(), ex(xg, yg, 0.02)
        w = res.ops["weights"]; w2 = torch.outer(w, w).reshape(1,1,w.numel(),w.numel(),1)
        errs.append(((((num-exf)**2*w2).sum()/((exf**2*w2).sum())).sqrt().item()))
    ax.loglog(ns, errs, "o-", label=f"p={deg}  (measured EOC {math.log(errs[0]/errs[-1])/math.log(8):.2f}, theory {deg+1})")
    h = np.array(ns, float)
    ax.loglog(h, errs[0]*(h/h[0])**-(deg+1), ":", lw=1, alpha=0.6)
ax.set_xlabel("cells per direction n (T=0.02, CFL=0.1)"); ax.set_ylabel("relative L2 error (all 5 fields)")
ax.set_title("P plane wave: error saturates instead of O(h^(p+1))")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
deg, nx, ny = 2, 2, 1
ops = legendre_operators_2d(deg); K = deg+1; hx, hy = 1.0/nx, 1.0/ny
N = nx*ny*K*K*5; A = np.zeros((N, N))
for j in range(N):
    c = torch.zeros(nx, ny, K, K, 5, dtype=torch.float64).reshape(-1); c[j] = 1.0
    A[:, j] = _rhs(c.reshape(nx, ny, K, K, 5), ops, med, hx, hy, "periodic", "periodic").reshape(-1).numpy()
ev = np.linalg.eigvals(A)
ax.scatter(ev.real, ev.imag, s=12, alpha=0.7)
ax.axvline(0.0, color="k", lw=0.8)
ax.set_xlabel("Re(lambda) of the semi-discrete operator"); ax.set_ylabel("Im(lambda)")
ax.set_title(f"2x1 cells, p=2, periodic: max Re = {ev.real.max():.3f} > 0 (unstable)")
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig("docs/figures/elastic_2d_psv_status.png", dpi=150)
print("saved", ev.real.max(), np.abs(ev.imag).max())
