"""Probe: real-state norm-quadrature sensitivity (t=1 checkpoints)."""
import time
import numpy as np
import torch
from tensordg import euler_dg_2d as dg2
from tensordg.euler_2d import solve_isentropic_vortex, vortex_primitive, GAMMA, _primitive
from benchmarks.case_euler_vortex_errconserve import (
    DOMAIN, STRENGTH, BACKGROUND, _initial_p0_state, _operators, _initial_dg_state,
)
from benchmarks.verify_euler_dg_2d import _vortex_error


def p0_error(conservative, x, y, time, nq):
    nx, ny = conservative.shape[:2]
    dx = float((x[-1] - x[0]).item()) / (nx - 1)
    dy = float((y[-1] - y[0]).item()) / (ny - 1)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, dtype=conservative.dtype)
    weights = torch.as_tensor(weights_np, dtype=conservative.dtype)
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    exact_q = vortex_primitive(xq, yq, time, STRENGTH, BACKGROUND, period=10.0)
    ref = tuple(0.25 * torch.sum(f * weights[None, None, :, None] * weights[None, None, None, :],
                                 dim=(2, 3)) for f in exact_q)
    num = _primitive(conservative)
    out = {}
    for key, c, e in zip(("density", "velocity_x", "velocity_y", "pressure"), num, ref):
        a = torch.sqrt(torch.mean((c - e) ** 2)).item()
        b = torch.sqrt(torch.mean(e ** 2)).item()
        out[key] = a / b
    return out


t0 = time.time()
x, y, fields, meta = solve_isentropic_vortex(ncells=64, final_time=1.0, cfl=0.4,
                                            domain=DOMAIN, strength=STRENGTH,
                                            background_velocity=BACKGROUND)
cons = fields["conservative"] if isinstance(fields, dict) else fields
print("p0 n64 t1 steps", meta["steps"], f"{time.time()-t0:.1f}s")
for nq in (2, 3, 4, 6, 8, 12):
    e = p0_error(cons, x, y, 1.0, nq)
    print(" nq", nq, {k: 100.0 * v for k, v in e.items()}, "max%", 100 * max(e.values()))

t0 = time.time()
_x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(ncells=32, degree=2, final_time=1.0,
                                                   cfl=0.2, strength=STRENGTH,
                                                   background_velocity=BACKGROUND,
                                                   domain=DOMAIN)
print("dg p2 n32 t1 steps", meta["steps"], f"{time.time()-t0:.1f}s")
for mult in (1, 2, 3, 4):
    nq = 4 * mult
    n_q, w_q, b_q, _, _ = dg2.legendre_operators(2, nq)
    e = _vortex_error(coeff, b_q, n_q, DOMAIN, 1.0, STRENGTH, BACKGROUND, period=10.0)
    print(" dg nq", nq, {k: 100 * e[k] for k in ("density", "velocity_x", "velocity_y", "pressure")},
          "max%", 100 * max(e[k] for k in ("density", "velocity_x", "velocity_y", "pressure")))