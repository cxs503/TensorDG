"""Probe: quadrature convergence of (a) the invariant reference, (b) the error norm."""
import math, time
import torch
import numpy as np
from tensordg import euler_dg_2d as dg2
from benchmarks.case_euler_vortex_errconserve import (
    exact_invariants, _operators, _initial_dg_state, DOMAIN, STRENGTH, BACKGROUND,
)
from benchmarks.verify_euler_dg_2d import _vortex_error, _reconstruct

print("=== R3: invariant reference vs quadrature points ===")
ref = {p: exact_invariants(p) for p in (320, 480, 640, 960, 1280)}
base = ref[1280]
for p, d in ref.items():
    print(p, {k: (v - base[k]) / abs(base[k]) for k, v in d.items()})

print("=== R4: DG error norm, projection state (no time stepping) ===")
for degree, cells in ((1, 32), (2, 32), (3, 32)):
    cfg = {"task": "dg", "degree": degree, "cells": cells, "final_time": 0.0,
           "cfl": 0.2}
    t0 = time.time()
    coeff, basis, nodes, weights = _initial_dg_state(cfg)
    readings = {}
    for mult in (1, 2, 3, 4):
        nq = (degree + 2) * mult
        n_q, w_q, b_q, _, _ = dg2.legendre_operators(degree, nq)
        err = _vortex_error(coeff, b_q, n_q, DOMAIN, 0.0, STRENGTH, BACKGROUND,
                            period=10.0)
        readings[mult] = max(err[k] for k in ("density", "velocity_x", "velocity_y", "pressure"))
    fine = readings[4]
    print(f"p={degree} n={cells} gate={readings[1]:.6e} "
          f"rel_shift(m1 vs m4)={100*(readings[1]-fine)/fine:+.4f}% "
          f"m2={100*(readings[2]-fine)/fine:+.4f}% m3={100*(readings[3]-fine)/fine:+.4f}% "
          f"({time.time()-t0:.1f}s)")