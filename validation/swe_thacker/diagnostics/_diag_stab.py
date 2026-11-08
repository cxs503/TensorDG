"""Scratch diagnostic 10: time-step sensitivity + perturbation growth (stability probe)."""
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


def run(n, deg, T, cfl=0.3, seed=None, amp=0.0):
    gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def ih(x):
        h = (h0 + (A / L) * x) - (-h0 * (1 - (x / L) ** 2))
        if gen is not None:
            h = h + amp * torch.randn(x.shape, generator=gen, dtype=x.dtype)
        return h

    x, coeff, zbf, meta = solve_swe_1d(
        ih, lambda z: torch.zeros_like(z), lambda z: -h0 * (1 - (z / L) ** 2),
        ncells=n, degree=deg, final_time=T, domain=(-L, L), cfl=cfl,
        flux_kind="hll", limiter=None, rk_order=3, dtype=torch.float64)
    h, u, eta = reconstruct_fields(coeff, zbf)
    return x, h, u, eta, meta


if __name__ == "__main__":
    x, h1, u1, e1, m1 = run(50, 1, 0.5, cfl=0.3)
    _, h2, u2, e2, m2 = run(50, 1, 0.5, cfl=0.05)
    print(f"cfl 0.3 (steps {m1['steps']}) vs cfl 0.05 (steps {m2['steps']}): "
          f"max|dh|={float((h1-h2).abs().max()):.3e} max|du|={float((u1-u2).abs().max()):.3e}",
          flush=True)
    _, h3, u3, e3, m3 = run(50, 1, 0.5, cfl=0.3, seed=1234, amp=1e-9)
    print(f"perturbation growth 1e-9: max|dh|={float((h1-h3).abs().max()):.3e} "
          f"max|du|={float((u1-u3).abs().max()):.3e}", flush=True)
    ee, ue, he = exact(x, 0.5)
    print(f"clean run error: max|eta-eta_exact|={float((e1-ee).abs().max()):.4f} "
          f"max|u-u_exact|={float((u1-ue).abs().max()):.4f}", flush=True)
    # energy-like diagnostic at several times
    for T in (0.1, 0.2, 0.3, 0.4, 0.5):
        _, hh, uu, ee_, _ = run(50, 1, T)
        eex, uex, _ = exact(torch.linspace(-L, L, 1001), T)
        # total energy of exact solution (potential+gauge independent part is tricky); report amplitude proxies
        amp_num = float((ee_[-1] - ee_[0]))
        amp_ex = float((eex[-1] - eex[0]))
        print(f"T={T}: eta_tilt_num={amp_num:+.4f} eta_tilt_exact={amp_ex:+.4f} "
              f"u_center_num={float(uu[len(uu)//2]):+.4f} u_exact={float(uex[0]):+.4f}", flush=True)