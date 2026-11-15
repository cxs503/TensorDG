"""Effect of the two-sided TVB indicator on accuracy (square wave L1/L2, mesh ladder)."""
import sys, torch, numpy as np
sys.path.insert(0, "/srv/tensor/agent/work/m21413")
from tensordg.advection_1d import solve_advection
from tensordg import tvb as tvbmod

def square_wave(x):
    return ((x >= 0.25) & (x <= 0.75)).to(x.dtype)

def run(degree, ncells, M):
    c, coeff, meta = solve_advection(ncells=ncells, degree=degree, final_time=1.0,
                                     cfl=0.2, initial=square_wave, limiter="tvb",
                                     tvb_constant=M, time_integrator="ssp_rk3")
    # L1 error of the cell average against the exact (advected) square wave
    exact = np.where(((c.numpy() - 1.0) % 1.0 >= 0.25) & ((c.numpy() - 1.0) % 1.0 <= 0.75), 1.0, 0.0)
    l1 = float(np.abs(coeff[:, 0].numpy() - exact).mean())
    xi = torch.linspace(-1, 1, 21, dtype=coeff.dtype)
    basis = torch.stack([torch.as_tensor(np.polynomial.legendre.legval(xi.numpy(), np.eye(coeff.shape[1])[k])) for k in range(coeff.shape[1])], 1)
    v = coeff @ basis.T
    over = 100 * max(0.0, float(v.max()) - 1.0, -float(v.min()))
    return l1, over

print(f"{'deg':>3} {'N':>5} {'M':>5} {'L1':>10} {'pt_over%':>9}")
for degree in (1, 2, 3):
    Ns = [50, 100, 200]
    l1s = []
    for ncells in Ns:
        l1, over = run(degree, ncells, 0.0)
        l1s.append(l1)
        print(f"{degree:3d} {ncells:5d} {0.0:5.1f} {l1:10.6f} {over:9.4f}")
    orders = [np.log2(l1s[i]/l1s[i+1]) for i in range(len(l1s)-1)]
    print(f"    P{degree} h-refinement L1 orders (N=50->100->200): "
          + ", ".join(f"{o:.4f}" for o in orders))