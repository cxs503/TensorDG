"""Quick probe: is `vortex_primitive` an exact solution of the 2D Euler equations?"""
import math
import torch
from tensordg.euler_2d import vortex_primitive, GAMMA

GAMMA_V = GAMMA


def conservative(rho, u, v, p):
    E = p / (GAMMA_V - 1.0) + 0.5 * rho * (u * u + v * v)
    return rho, rho * u, rho * v, E


def fluxes(U):
    rho, mx, my, E = U
    u = mx / rho
    v = my / rho
    p = (GAMMA_V - 1.0) * (E - 0.5 * (mx * mx + my * my) / rho)
    F = (mx, mx * u + p, my * u, u * (E + p))
    G = (my, mx * v, my * v + p, v * (E + p))
    return F, G, p, u, v


def residual_at(points, time=0.0, strength=5.0, background=(1.0, 1.0), period=None):
    x = points[:, 0].clone().requires_grad_(True)
    y = points[:, 1].clone().requires_grad_(True)
    t = torch.tensor(float(time), dtype=torch.float64, requires_grad=True)
    rho, u, v, p = vortex_primitive(x, y, t, strength, background, period=period)
    U = conservative(rho, u, v, p)
    F, G, _, _, _ = fluxes(U)
    out = {}
    for name, comp in zip(("mass", "mx", "my", "energy"), U):
        dUdt = torch.autograd.grad(comp.sum(), t, create_graph=False, retain_graph=True)[0]
        dFdx = torch.autograd.grad(F[("mass", "mx", "my", "energy").index(name)].sum(), x,
                                   create_graph=False, retain_graph=True)[0]
        dGdy = torch.autograd.grad(G[("mass", "mx", "my", "energy").index(name)].sum(), y,
                                   create_graph=False, retain_graph=True)[0]
        out[name] = (dUdt + dFdx + dGdy)
    return out, (rho, u, v, p)


torch.manual_seed(0)
pts = torch.rand(400, 2, dtype=torch.float64) * 10.0 - 5.0
for label, kw in (("free t=0", {}), ("free t=2.5", {"time": 2.5}),
                  ("free t=10", {"time": 10.0})):
    res, prim = residual_at(pts, **kw)
    scales = {
        "mass": float((prim[0]).abs().max()),
        "mx": float((prim[0] * prim[1]).abs().max()),
        "my": float((prim[0] * prim[2]).abs().max()),
        "energy": float((prim[3] / (GAMMA_V - 1) + 0.5 * prim[0] * (prim[1]**2 + prim[2]**2)).abs().max()),
    }
    print(label, {k: (float(v.abs().max()), float(v.abs().max()) / scales[k]) for k, v in res.items()})