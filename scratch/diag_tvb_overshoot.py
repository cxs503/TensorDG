"""Diagnose the E3 square-wave pointwise overshoot: where and why."""
import numpy as np, torch, sys
sys.path.insert(0, "/root/tp-agent/work/m21413")
from tensordg.advection_1d import solve_advection

def square_wave(x):
    return ((x >= 0.25) & (x <= 0.75)).to(x.dtype)

def nodes(coeff, npts=41):
    ncells, nmodes = coeff.shape
    xi = torch.linspace(-1.0, 1.0, npts, dtype=coeff.dtype)
    basis = torch.stack([
        torch.as_tensor(np.polynomial.legendre.legval(xi.numpy(), np.eye(nmodes)[k]))
        for k in range(nmodes)
    ], dim=1)
    return coeff @ basis.T

def state(ncells, degree, tvb, policy="tvb", t=1.0):
    c, coeff, meta = solve_advection(ncells=ncells, degree=degree, final_time=t,
                                     cfl=0.2, initial=square_wave, limiter="tvb",
                                     tvb_constant=tvb, time_integrator="ssp_rk3",
                                     troubled_policy=policy)
    return c, coeff, meta

print("=== pointwise overshoot vs N (P1..P3, M=0) ===")
for degree in (1, 2):
    for ncells in (50, 100, 200, 400):
        c, coeff, meta = state(ncells, degree, 0.0)
        v = nodes(coeff)
        avg = coeff[:, 0]
        over_pt = max(0.0, float(v.max()) - 1.0, -float(v.min()))
        over_avg = max(0.0, float(avg.max()) - 1.0, -float(avg.min()))
        print(f"P{degree} N={ncells:4d} pt_max={float(v.max()):.6f} pt_min={float(v.min()):.6f} "
              f"pt_over={100*over_pt:.3f}%  avg_over={100*over_avg:.3f}%")

print()
print("=== per-cell detail at P1 N=100, M=0 (final state) ===")
for ncells, degree in ((100, 1),):
    c, coeff, meta = state(ncells, degree, 0.0)
    v = nodes(coeff)
    avg = coeff[:, 0].numpy()
    d = np.diff(avg, append=avg[0])          # d_right (periodic)
    dl = avg - np.roll(avg, 1)               # d_left
    mm = np.where(np.sign(dl) == np.sign(d), np.sign(dl)*np.minimum(abs(dl), abs(d)), 0.0)
    dev = np.abs(dl - mm)
    bad = np.where((v.max(dim=1).values.numpy() > 1.0 + 1e-12) |
                   (v.min(dim=1).values.numpy() < -1e-12))[0]
    print("cells with pointwise excursion:", bad[:20], "count", len(bad))
    for j in list(bad[:12]):
        print(f"  j={j:3d} x={float(c[j]):.4f} avg={avg[j]:.6f} dl={dl[j]:+.3e} dr={d[j]:+.3e} "
              f"mm={mm[j]:+.3e} dev={dev[j]:.3e} "
              f"ptmax={float(v[j].max()):.6f} ptmin={float(v[j].min()):.6f}")
    print("  coeff rows near excurs:", [(int(j), coeff[j].tolist()) for j in bad[:5]])

print()
print("=== initial projection (t=0) ===")
for ncells, degree in ((100, 1),):
    c, coeff, meta = state(ncells, degree, 0.0, t=0.0)
    v = nodes(coeff)
    print("t=0 pt_max", float(v.max()), "pt_min", float(v.min()))

print()
print("=== M sweep at P1 N=100 (pointwise overshoot) ===")
for M in (0.0, 0.1, 1.0, 10.0):
    c, coeff, meta = state(100, 1, M)
    v = nodes(coeff)
    print(f"M={M:5.1f} pt_over={100*max(0.0,float(v.max())-1.0,-float(v.min())):.3f}%")

print()
print("=== strict policy baseline ===")
for degree in (1, 2, 3):
    c, coeff, meta = state(100, degree, 0.0, policy="all")
    v = nodes(coeff)
    print(f"P{degree} strict pt_over={100*max(0.0,float(v.max())-1.0,-float(v.min())):.4f}%")