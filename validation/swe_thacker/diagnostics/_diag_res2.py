"""Scratch diagnostic 6: per-cell residual components at t0=0.3 (P1, N=20)."""
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


def main(ncells=20, degree=1, t0=0.3):
    dx = 2 * L / ncells
    x = -L + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    nodes, weights, phi, dphi, phi_left, phi_right = _operators(degree, torch.device("cpu"), torch.float64)
    nx = x[:, None] + 0.5 * dx * nodes[None, :]

    s = (A / L) * math.cos(omega * t0)
    ds = -(A / L) * omega * math.sin(omega * t0)
    u = ds * L ** 2 / (2 * h0)
    du = (-(A / L) * omega ** 2 * math.cos(omega * t0)) * L ** 2 / (2 * h0)
    eta_t = (A * A / (4 * h0)) * omega * math.sin(2 * omega * t0) + ds * nx
    zb = -h0 * (1 - (nx / L) ** 2)
    h = eta_t * 0 + (h0 + (A * A / (4 * h0)) * math.sin(omega * t0) ** 2 + s * nx) - zb
    hu = h * u
    ht = eta_t
    hut = ht * u + h * du

    coeff = torch.stack([proj(h, weights, phi, degree), proj(hu, weights, phi, degree)], dim=-1)
    ref = torch.stack([proj(ht, weights, phi, degree), proj(hut, weights, phi, degree)], dim=-1)

    zb_l = -h0 * (1 - ((-L + torch.arange(ncells, dtype=torch.float64) * dx) / L) ** 2)
    zb_r = -h0 * (1 - ((-L + torch.arange(ncells, dtype=torch.float64) * dx + dx) / L) ** 2)
    res = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right, zb_r - zb_l, "hll", False)

    print(f"N={ncells} P{degree} t0={t0} u={u:.4f} du={du:.4f} s={s:.4f}")
    print("mass  res:", res[:, 0, 0][:6], "...")
    print("mass  ref:", ref[:, 0, 0][:6], "...")
    print("mom   res:", res[:, 0, 1][:6], "...")
    print("mom   ref:", ref[:, 0, 1][:6], "...")
    print("h-avg nonzero (projection error) mass mode0 max:", float(coeff[:, 0, 0].abs().max()))
    print("mass res/ref ratio (interior):",
          (res[2:-2, 0, 0] / ref[2:-2, 0, 0].clamp_min(1e-30))[:6])
    print("mass mode0 res:", res[2:-2, 0, 0])
    print("mass mode0 ref:", ref[2:-2, 0, 0])
    print("mass mode1 res:", res[2:-2, 1, 0])
    print("mass mode1 ref:", ref[2:-2, 1, 0])
    print("mom mode0 res-ref:", (res[2:-2, 0, 1] - ref[2:-2, 0, 1]))
    print("mom mode1 res-ref:", (res[2:-2, 1, 1] - ref[2:-2, 1, 1]))


if __name__ == "__main__":
    main()