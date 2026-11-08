"""Scratch diagnostic 3: solution profiles / stability statistics."""
import math
import torch

from tensordg.swe_1d import solve_swe_1d, reconstruct_fields

g, h0, L = 9.81, 1.0, 1.0
omega = math.sqrt(2 * g * h0) / L


def exact(x, t, A):
    s = (A / L) * math.cos(omega * t)
    u = -(A * omega * L / (2 * h0)) * math.sin(omega * t)
    eta = h0 + (A * A / (4 * h0)) * math.sin(omega * t) ** 2 + s * x
    zb = -h0 * (1 - (x / L) ** 2)
    return eta, torch.full_like(x, u), eta - zb


def run(ncells, deg, T, A=0.5, cfl=0.3, flux="hll"):
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
    return x, h, u, eta, meta


if __name__ == "__main__":
    torch.set_printoptions(precision=4, linewidth=160)
    for deg in (1, 2):
        for T in (0.1, 0.5):
            x, h, u, eta, meta = run(20, deg, T)
            ee, ue, he = exact(x, T, 0.5)
            print(f"== P{deg} N=20 T={T} steps={meta['steps']} "
                  f"u_exact={float(ue[0]):.4f} h[min,max]=[{float(h.min()):.4f},{float(h.max()):.4f}]", flush=True)
            print("x   :", x)
            print("u   :", u)
            print("u_ex:", ue)
            print("eta :", eta)
            print("etaE:", ee, flush=True)