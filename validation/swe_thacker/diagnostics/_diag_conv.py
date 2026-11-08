"""Scratch diagnostic 11: short-time convergence (T=0.2) and volume budget."""
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


def run(n, deg, T):
    x, coeff, zbf, meta = solve_swe_1d(
        lambda z: (h0 + (A / L) * z) - (-h0 * (1 - (z / L) ** 2)),
        lambda z: torch.zeros_like(z), lambda z: -h0 * (1 - (z / L) ** 2),
        ncells=n, degree=deg, final_time=T, domain=(-L, L), cfl=0.3,
        flux_kind="hll", limiter=None, rk_order=3, dtype=torch.float64)
    h, u, eta = reconstruct_fields(coeff, zbf)
    return x, h, u, eta, meta


if __name__ == "__main__":
    for T in (0.2, 0.35):
        print(f"=== T={T} ===", flush=True)
        for deg in (1, 2):
            prev = None
            for n in (25, 50, 100, 200):
                if deg == 2 and n == 200:
                    continue
                x, h, u, eta, meta = run(n, deg, T)
                ee, ue, he = exact(x, T)
                ea = float(torch.sqrt(torch.mean((eta - ee) ** 2)) / ee.abs().max()) * 100
                ua = float(torch.sqrt(torch.mean((u - ue) ** 2)) / ue.abs().max()) * 100
                dx = 2.0 / n
                vol = float(h.sum()) * dx
                ex_t = torch.linspace(-L, L, 200001)
                v_ex = float(torch.trapz(exact(ex_t, T)[2], ex_t))
                rate = "" if prev is None else f" rate={math.log2(prev / ea):5.2f}"
                print(f"  P{deg} N={n:4d} eta={ea:8.3f}% u={ua:8.2f}% V={vol:.4f} (exact {v_ex:.4f}){rate}", flush=True)
                prev = ea