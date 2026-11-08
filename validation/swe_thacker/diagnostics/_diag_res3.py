"""Scratch diagnostic 9: per-mode semi-discrete residual convergence (projected exact state)."""
import math
import torch

from tensordg.swe_1d import _operators, _rhs, GRAVITY

g, h0, L, A = GRAVITY, 1.0, 1.0, 0.5
omega = math.sqrt(2 * g * h0) / L


def proj(vals, weights, phi, degree):
    out = torch.zeros(vals.shape[0], degree + 1, dtype=vals.dtype)
    for k in range(degree + 1):
        out[:, k] = ((2 * k + 1) / 2.0) * torch.einsum("cq,q,q->c", vals, weights, phi[:, k])
    return out


def errs(ncells, degree, t0):
    dx = 2 * L / ncells
    x = -L + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    nodes, weights, phi, dphi, phi_left, phi_right = _operators(degree, torch.device("cpu"), torch.float64)
    nx = x[:, None] + 0.5 * dx * nodes[None, :]

    s = (A / L) * math.cos(omega * t0)
    ds = -(A / L) * omega * math.sin(omega * t0)
    dds = -(A / L) * omega ** 2 * math.cos(omega * t0)
    u = ds * L ** 2 / (2 * h0)
    du = dds * L ** 2 / (2 * h0)
    eta = h0 + (A * A / (4 * h0)) * math.sin(omega * t0) ** 2 + s * nx
    eta_t = (A * A / (4 * h0)) * omega * math.sin(2 * omega * t0) + ds * nx
    zb = -h0 * (1 - (nx / L) ** 2)
    h = eta - zb
    ht, hut = eta_t, eta_t * u + h * du

    coeff = torch.stack([proj(h, weights, phi, degree), proj(h * u, weights, phi, degree)], dim=-1)
    ref = torch.stack([proj(ht, weights, phi, degree), proj(hut, weights, phi, degree)], dim=-1)
    zb_l = -h0 * (1 - ((-L + torch.arange(ncells, dtype=torch.float64) * dx) / L) ** 2)
    zb_r = -h0 * (1 - ((-L + torch.arange(ncells, dtype=torch.float64) * dx + dx) / L) ** 2)
    res = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right, zb_r - zb_l, "hll", False)
    sl = slice(2, ncells - 2)
    return [(res[sl, k, c] - ref[sl, k, c]).abs().max().item()
            for k in range(degree + 1) for c in range(2)]


if __name__ == "__main__":
    for t0 in (0.0, 0.3):
        print(f"t0={t0}: columns = (mass k), (mom k) absolute max residual error")
        for degree in (1, 2):
            prev = None
            for n in (20, 40, 80, 160):
                e = errs(n, degree, t0)
                rates = []
                if prev is not None:
                    rates = [math.log2(prev[i] / e[i]) if e[i] > 0 else float("nan")
                             for i in range(len(e))]
                print(f"  P{degree} N={n:4d} " + " ".join(f"{v:.3e}" for v in e)
                      + ("  rate " + " ".join(f"{r:5.2f}" for r in rates) if rates else ""), flush=True)
                prev = e