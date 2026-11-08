"""Scratch diagnostic 7: interface flux bookkeeping at t0=0 (still surface, sloped eta)."""
import math
import torch

from tensordg.swe_1d import _operators, _rhs, hll_flux, GRAVITY

g, h0, L, A = GRAVITY, 1.0, 1.0, 0.5
omega = math.sqrt(2 * g * h0) / L
dx, ncells, degree = 0.1, 20, 1
t0 = 0.0
x = -L + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
nodes, weights, phi, dphi, phi_left, phi_right = _operators(degree, torch.device("cpu"), torch.float64)
nx = x[:, None] + 0.5 * dx * nodes[None, :]
s = (A / L) * math.cos(omega * t0)
eta = h0 + (A * A / (4 * h0)) * math.sin(omega * t0) ** 2 + s * nx
zb = -h0 * (1 - (nx / L) ** 2)
h = eta - zb
coeff = torch.zeros((ncells, degree + 1, 2), dtype=torch.float64)
for k in range(degree + 1):
    f = (2 * k + 1) / 2.0
    coeff[:, k, 0] = f * torch.einsum("cq,q,q->c", h, weights, phi[:, k])
print("mode0 h:", coeff[:4, 0, 0])
print("mode1 h:", coeff[:4, 1, 0])
at = torch.einsum("cmv,qm->cqv", coeff, phi)
print("trace h left/right:", at[:4, 0, 0], at[:4, -1, 0])
lt = torch.einsum("cmv,m->cv", coeff, phi_left)
rt = torch.einsum("cmv,m->cv", coeff, phi_right)
rn = torch.roll(lt, -1, 0)
rn[-1] = rt[-1]
f = hll_flux(rt, rn, "hll")
print("HLL mass flux per interface:", f[:, 0])
print("physical mass flux hu:", (rt[:, 0] * rt[:, 1] / rt[:, 0]))
zb_l = -h0 * (1 - ((-L + torch.arange(ncells, dtype=torch.float64) * dx) / L) ** 2)
zb_r = -h0 * (1 - ((-L + torch.arange(ncells, dtype=torch.float64) * dx + dx) / L) ** 2)
res = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right, zb_r - zb_l, "hll", False)
print("mass residual mode0:", res[:5, 0, 0])
print("mass residual mode1:", res[:5, 1, 0])