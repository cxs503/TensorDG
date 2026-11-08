"""Scratch diagnostic: solver vs corrected exact planar (Thacker) solution."""
import math
import torch

from tensordg.swe_1d import solve_swe_1d, reconstruct_fields

g, h0, A, L = 9.81, 1.0, 0.5, 1.0
omega = math.sqrt(2 * g * h0) / L


def exact(x, t):
    s = (A / L) * math.cos(omega * t)
    u = -(A * omega * L / (2 * h0)) * math.sin(omega * t)
    eta = h0 + (A * A / (4 * h0)) * math.sin(omega * t) ** 2 + s * x
    zb = -h0 * (1 - (x / L) ** 2)
    return eta, torch.full_like(x, u), eta - zb


def run(ncells, deg, T, cfl=0.3, limiter=None):
    def ih(x):
        return (h0 + (A / L) * x) - (-h0 * (1 - (x / L) ** 2))

    def iu(x):
        return torch.zeros_like(x)

    def zb(x):
        return -h0 * (1 - (x / L) ** 2)

    x, coeff, zbf, meta = solve_swe_1d(
        ih, iu, zb, ncells=ncells, degree=deg, final_time=T, domain=(-L, L),
        cfl=cfl, flux_kind="hll", limiter=limiter, rk_order=3, dtype=torch.float64)
    h, u, eta = reconstruct_fields(coeff, zbf)
    ee, ue, he = exact(x, T)

    def rel(a, b):
        return float(torch.sqrt(torch.mean((a - b) ** 2)) / torch.max(torch.abs(b))) * 100

    return dict(n=ncells, deg=deg, T=T, eta=rel(eta, ee), u=rel(u, ue), h=rel(h, he),
                steps=meta["steps"], u_min=float(u.min()), u_max=float(u.max()))


if __name__ == "__main__":
    for n in (25, 50, 100):
        for deg in (1, 2):
            r = run(n, deg, 0.5)
            print(f"n={r['n']:4d} P{r['deg']} T={r['T']} eta={r['eta']:.4f}% "
                  f"u={r['u']:.4f}% h={r['h']:.4f}% steps={r['steps']} "
                  f"u_range=[{r['u_min']:.4f},{r['u_max']:.4f}]", flush=True)