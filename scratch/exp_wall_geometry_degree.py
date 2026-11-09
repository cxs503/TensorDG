"""Diagnostic: does the curved-wall order need superparametric geometry (g > p)?

Runs the annulus rigid-rotation problem for a few (p, geometry degree) pairs and
reports the observed order, to isolate the odd-degree order loss seen in the
J3 gate (p=3: 2.66, p=5: 5.03).  Scratch script, not part of the package.
"""
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensordg.curved_geometry import annulus_mesh, CurvedAdvection  # noqa: E402

OMEGA, MODE, R_MID, WIDTH = 1.0, 2, 1.5, 0.35


def vel(x, y):
    return -OMEGA * y, OMEGA * x


def exact(x, y, t):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    return np.exp(-((r - R_MID) / WIDTH) ** 2) * np.cos(MODE * (th - OMEGA * t))


def run(p, gdeg, nr, nt, T=0.3):
    mesh = annulus_mesh(nr, nt, 1.0, 2.0, gdeg)
    dg = CurvedAdvection(mesh, p, vel, cfl=0.2)
    U = dg.project(lambda x, y: exact(x, y, 0.0))
    U2, dt, ns = dg.evolve(U, T, dg.stable_dt())
    rel = dg.l2_error(U2, lambda x, y: exact(x, y, T))[2]
    h = float(np.sqrt(np.sum(np.einsum("nqr,q,r->n", dg.Jg, dg.w, dg.w)) / mesh.ncells))
    return rel, h, ns


CASES = [
    (2, 3, [(8, 16), (12, 24), (16, 32)]),
    (4, 5, [(12, 24), (16, 32), (24, 48)]),
    (5, 6, [(24, 48), (32, 64)]),
    (5, 7, [(24, 48), (32, 64)]),
]

for p, gdeg, levels in CASES:
    errs, hs, steps = [], [], []
    for nr, nt in levels:
        e, h, ns = run(p, gdeg, nr, nt)
        errs.append(e)
        hs.append(h)
        steps.append(ns)
        print(f"  p={p} g={gdeg} {nr}x{nt}: rel={e:.4e} h={h:.4f} steps={ns}", flush=True)
    slope = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    pw = [math.log(errs[i] / errs[i + 1]) / math.log(hs[i] / hs[i + 1])
          for i in range(len(errs) - 1)]
    print(f"p={p} g={gdeg} order={slope:.3f} (theory {p + 1}) pairwise="
          f"{['%.2f' % x for x in pw]}", flush=True)