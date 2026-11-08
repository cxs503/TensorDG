"""Scratch diagnostic 4: interior-only error vs N, and mass budget."""
import math
import torch

from tensordg.swe_1d import solve_swe_1d, reconstruct_fields

g, h0, L, A = 9.81, 1.0, 1.0, 0.5
omega = math.sqrt(2 * g * h0) / L


def exact(x, t):
    s = (A / L) * math.cos(omega * t)
    u = -(A * omega * L / (2 * h0)) * math.sin(omega * t)
    eta = h0 + (A * A / (4 * h0)) * math.sin(omega * t) ** 2 + s * x
    zb = -h0 * (1 - (x / L) ** 2)
    return eta, torch.full_like(x, u), eta - zb


def run(ncells, deg, T, cfl=0.3):
    def ih(x):
        return (h0 + (A / L) * x) - (-h0 * (1 - (x / L) ** 2))

    x, coeff, zbf, meta = solve_swe_1d(
        ih, lambda z: torch.zeros_like(z), lambda z: -h0 * (1 - (z / L) ** 2),
        ncells=ncells, degree=deg, final_time=T, domain=(-L, L), cfl=cfl,
        flux_kind="hll", limiter=None, rk_order=3, dtype=torch.float64)
    h, u, eta = reconstruct_fields(coeff, zbf)
    return x, h, u, eta, meta


if __name__ == "__main__":
    T = 0.5
    for deg in (1, 2):
        for n in (25, 50, 100, 200):
            x, h, u, eta, meta = run(n, deg, T)
            ee, ue, he = exact(x, T)
            dx = 2.0 / n
            m = int(round(0.15 / dx))  # interior mask: drop 0.15 length at each end
            sl = slice(m, n - m)
            voll = float(h.sum()) * dx
            ei = float(torch.sqrt(torch.mean((eta[sl] - ee[sl]) ** 2))) / float(ee.abs().max()) * 100
            ui = float(torch.sqrt(torch.mean((u[sl] - ue[sl]) ** 2))) / float(ue.abs().max()) * 100
            ea = float(torch.sqrt(torch.mean((eta - ee) ** 2))) / float(ee.abs().max()) * 100
            print(f"P{deg} N={n:4d} steps={meta['steps']:5d} eta_all={ea:7.3f}% "
                  f"eta_int={ei:7.3f}% u_int={ui:7.3f}% V={voll:.5f}", flush=True)
    x = torch.linspace(-L, L, 200001)
    ee, ue, he = exact(x, T)
    print("V_exact =", float(torch.trapz(he, x)), " V_exact(t=0) =",
          float(torch.trapz(exact(x, 0.0)[2], x)))