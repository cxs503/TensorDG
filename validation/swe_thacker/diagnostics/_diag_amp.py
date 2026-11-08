"""Scratch diagnostic 2: amplitude / time scan + volume conservation check."""
import math
import torch

from tensordg.swe_1d import solve_swe_1d, reconstruct_fields

g, h0, L = 9.81, 1.0, 1.0


def make_exact(A):
    omega = math.sqrt(2 * g * h0) / L

    def exact(x, t):
        s = (A / L) * math.cos(omega * t)
        u = -(A * omega * L / (2 * h0)) * math.sin(omega * t)
        eta = h0 + (A * A / (4 * h0)) * math.sin(omega * t) ** 2 + s * x
        zb = -h0 * (1 - (x / L) ** 2)
        return eta, torch.full_like(x, u), eta - zb

    return exact


def run(ncells, deg, T, A, cfl=0.3, flux="hll"):
    exact = make_exact(A)

    def ih(x):
        return (h0 + (A / L) * x) - (-h0 * (1 - (x / L) ** 2))

    def iu(x):
        return torch.zeros_like(x)

    def zb(x):
        return -h0 * (1 - (x / L) ** 2)

    x, coeff, zbf, meta = solve_swe_1d(
        ih, iu, zb, ncells=ncells, degree=deg, final_time=T, domain=(-L, L),
        cfl=cfl, flux_kind=flux, limiter=None, rk_order=3, dtype=torch.float64)
    h, u, eta = reconstruct_fields(coeff, zbf)
    ee, ue, he = exact(x, T)

    def rel(a, b):
        return float(torch.sqrt(torch.mean((a - b) ** 2)) / torch.max(torch.abs(b))) * 100

    return dict(n=ncells, deg=deg, T=T, A=A, eta=rel(eta, ee), u=rel(u, ue),
                h=rel(h, he), steps=meta["steps"]) 


if __name__ == "__main__":
    print("--- amplitude scan (N=50, T=0.5) ---", flush=True)
    for A in (0.01, 0.1, 0.5):
        for deg in (1, 2):
            r = run(50, deg, 0.5, A)
            print(f"A={r['A']:<5} P{r['deg']} eta={r['eta']:8.4f}% u={r['u']:8.4f}% h={r['h']:8.4f}%", flush=True)
    print("--- short-time scan (N=50, A=0.5) ---", flush=True)
    for T in (0.02, 0.1, 0.5):
        for deg in (1, 2):
            r = run(50, deg, T, 0.5)
            print(f"T={r['T']:<5} P{r['deg']} eta={r['eta']:8.4f}% u={r['u']:8.4f}% steps={r['steps']}", flush=True)