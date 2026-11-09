"""Independent verification of :mod:`tensordg.advection_2d_var` reference cases.

Runs three families of checks and writes ``validation/advection_2d_var/results.json``:

A. reentrant deformational vortex -- smooth, divergence-free, periodic velocity;
   the exact flow map over [0, T] is the identity, so ``u(T) == u(0)`` exactly.
   This is the only *globally valid* design-order gate for this module, because
   its velocity field is smooth on the torus.
B. solid-body rotation -- rigid rotation about the box centre.  The velocity has
   a jump of ``omega`` across the periodic seam, so the "rotated profile" is not
   a solution of the PDE there; the error is measured on a disk of radius R that
   excludes the seam layer (the flow is circular, so seam noise stays at its own
   radius).
C. diagnostic: the *global* norm for B, to show why it is not a valid gate.

Usage:  PYTHONPATH=. python validation/verify_advection_2d_var.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

from tensordg.advection_2d_var import (
    DGAdvection2D,
    gaussian_bump,
    reentrant_vortex,
    rotated_profile,
    solid_body_rotation,
    trig_profile,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "advection_2d_var")


def eoc(errors):
    return [math.log(errors[i] / errors[i + 1]) / math.log(2.0) for i in range(len(errors) - 1)]


def vortex_family(p, meshes, final_time=1.0, cfl=0.3):
    rows = []
    for n in meshes:
        t0 = time.perf_counter()
        solver = DGAdvection2D(degree=p, ncells=n)
        coeff = solver.project(trig_profile)
        vel = reentrant_vortex(period=final_time)
        res = solver.solve(coeff, vel, final_time=final_time, cfl=cfl)
        err = solver.l2_error(res["coeff"], trig_profile, t=final_time)["relative_l2"]
        rows.append({"n": n, "relative_l2": err,
                     "mass_drift_rel": res["meta"]["mass_drift_rel"],
                     "wall_s": round(time.perf_counter() - t0, 2)})
        print(f"  vortex p={p} n={n:3d} relL2={err:.6e}", flush=True)
    return {"degree": p, "meshes": list(meshes), "rows": rows,
            "eoc": eoc([r["relative_l2"] for r in rows])}


def rotation_family(p, meshes, radius, final_time=0.25, cfl=0.3, omega=2.0 * math.pi):
    rows = []
    vel = solid_body_rotation(omega=omega)
    exact = rotated_profile(gaussian_bump, omega)
    for n in meshes:
        t0 = time.perf_counter()
        solver = DGAdvection2D(degree=p, ncells=n)
        coeff = solver.project(gaussian_bump)
        res = solver.solve(coeff, vel, final_time=final_time, cfl=cfl)
        disk = solver.l2_error(res["coeff"], exact, t=final_time, radius=radius)
        glob = solver.l2_error(res["coeff"], exact, t=final_time)
        rows.append({"n": n, "disk_relative_l2": disk["relative_l2"],
                     "global_relative_l2": glob["relative_l2"],
                     "mass_drift_rel": res["meta"]["mass_drift_rel"],
                     "wall_s": round(time.perf_counter() - t0, 2)})
        print(f"  rotation p={p} n={n:3d} disk={disk['relative_l2']:.6e} "
              f"global={glob['relative_l2']:.6e}", flush=True)
    return {"degree": p, "meshes": list(meshes), "disk_radius": radius,
            "final_time": final_time, "rows": rows,
            "eoc_disk": eoc([r["disk_relative_l2"] for r in rows]),
            "eoc_global": eoc([r["global_relative_l2"] for r in rows])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small mesh ladder")
    args = ap.parse_args()
    meshes = (8, 16, 32) if args.quick else (8, 16, 32, 64)
    rot_meshes = (16, 32) if args.quick else (16, 32, 64)

    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "torch": torch.__version__, "vortex": [], "rotation": []}
    for p in (1, 2, 3):
        payload["vortex"].append(vortex_family(p, meshes))
    for p in (1, 2):
        payload["rotation"].append(rotation_family(p, rot_meshes, radius=0.25))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=1)

    print("\nsummary")
    for fam in payload["vortex"]:
        print("  vortex   p=%d  eoc=%s" % (fam["degree"], ["%.2f" % e for e in fam["eoc"]]))
    for fam in payload["rotation"]:
        print("  rotation p=%d  disk eoc=%s  global eoc=%s"
              % (fam["degree"], ["%.2f" % e for e in fam["eoc_disk"]],
                 ["%.2f" % e for e in fam["eoc_global"]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())