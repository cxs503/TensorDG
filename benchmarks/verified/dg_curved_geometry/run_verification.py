#!/usr/bin/env python3
"""Verification runner for TensorDG curved geometry campaign node P127W2-N01.

Produces:
- result.json: full verification results
- gates_verified.json: gate pass/fail summary
- figures/: convergence plots
"""
import sys
import os
import json
import time
import numpy as np
import math

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from tensordg.curved_geometry import (
    annulus_mesh,
    cartesian_periodic_mesh,
    warped_periodic_mesh,
    CurvedAdvection,
    gll_nodes_weights,
    metric_identity_report,
)

# Problem setup
OMEGA = 1.0
AZIMUTH = 2
R_MID, R_WIDTH = 1.5, 0.35
R_INNER, R_OUTER = 1.0, 2.0
VEL = (0.6, 0.8)

def const_vel(x, y):
    return np.full_like(x, VEL[0]), np.full_like(y, VEL[1])

def rotation_vel(x, y):
    return -OMEGA * y, OMEGA * x

def rotation_exact(x, y, t):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    envelope = np.exp(-((r - R_MID) / R_WIDTH) ** 2)
    return envelope * np.cos(AZIMUTH * (th - OMEGA * t))

def wave_exact(x, y, t):
    ax, ay = VEL
    return np.sin(2 * np.pi * (x - ax * t)) * np.cos(2 * np.pi * (y - ay * t))

def observed_order(errs, hs):
    """Least-squares log-log slope."""
    if len(errs) < 2:
        return float("nan")
    log_h = np.log(np.array(hs))
    log_e = np.log(np.array(errs))
    A = np.vstack([log_h, np.ones(len(log_h))]).T
    result = np.linalg.lstsq(A, log_e, rcond=None)
    return float(result[0][0])

def annulus_convergence(p, gdeg, nr_list, nt_list, T=0.15, cfl=0.25):
    """Run annulus advection at multiple resolutions."""
    errs, hs = [], []
    for nr, nt in zip(nr_list, nt_list):
        mesh = annulus_mesh(nr, nt, R_INNER, R_OUTER, gdeg)
        dg = CurvedAdvection(mesh, p, rotation_vel, cfl=cfl)
        U = dg.project(lambda x, y: rotation_exact(x, y, 0.0))
        U2, dt, _ = dg.evolve(U, T, dg.stable_dt())
        num, den, rel = dg.l2_error(U2, lambda x, y: rotation_exact(x, y, T))
        h = float(np.sqrt(np.sum(np.einsum("nqr,q,r->n", dg.Jg, dg.w, dg.w)) / mesh.ncells))
        errs.append(rel)
        hs.append(h)
    return errs, hs

def main():
    t_start = time.time()
    outdir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    
    results = {
        "campaign_node": "P127W2-N01",
        "title": "TensorDG Curved Isoparametric Geometry Verification",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gates": {},
    }
    
    print("=" * 70)
    print("TensorDG Curved Geometry Verification - Campaign P127W2-N01")
    print("=" * 70)
    
    # J1: Free-stream preservation
    print("\n[J1] Free-stream preservation on curved meshes...")
    j1_pass = True
    j1_details = {}
    for p in [1, 2, 3]:
        mesh = warped_periodic_mesh(5, 5, p, 0.08)
        dg = CurvedAdvection(mesh, p, const_vel, cfl=0.2)
        fs = dg.free_stream_defect(1.0, steps=40)
        j1_details[f"p{p}"] = {
            "u_t_exact": fs["u_t_exact"],
            "drift": fs["drift"],
        }
        if fs["u_t_exact"] > 1e-12 or fs["drift"] > 1e-12:
            j1_pass = False
    results["gates"]["J1_free_stream"] = {
        "pass": j1_pass,
        "threshold": "≤1e-12",
        "details": j1_details,
    }
    print(f"  J1: {'PASS' if j1_pass else 'FAIL'}")
    
    # J2: Metric identity
    print("\n[J2] Metric identity residual...")
    j2_pass = True
    j2_details = {}
    for p in [2, 3, 4]:
        mesh = warped_periodic_mesh(4, 4, p, 0.08)
        for m in mesh.flat_maps():
            rep = metric_identity_report(m, *VEL)
            if rep["piola_coeff"] > 1e-10:
                j2_pass = False
            j2_details[f"p{p}"] = rep
    results["gates"]["J2_metric_identity"] = {
        "pass": j2_pass,
        "threshold": "≤1e-10 for p≥2",
        "details": j2_details,
    }
    print(f"  J2: {'PASS' if j2_pass else 'FAIL'}")
    
    # J3: Curved-wall analytic order p=1..4
    print("\n[J3] Curved-wall analytic order (p=1..4)...")
    j3_pass = True
    j3_details = {}
    for p in [1, 2, 3, 4]:
        gdeg = p + 1  # superparametric
        if p <= 1:
            nr_list = [6, 10, 16]
            nt_list = [12, 20, 32]
        elif p == 2:
            nr_list = [4, 8, 12]
            nt_list = [8, 16, 24]
        else:
            nr_list = [4, 6]
            nt_list = [8, 12]
        errs, hs = annulus_convergence(p, gdeg, nr_list, nt_list, T=0.15)
        obs = observed_order(errs, hs)
        theory = p + 1
        rel_dev = abs(obs - theory) / theory
        j3_details[f"p{p}"] = {
            "observed_order": obs,
            "theory_order": theory,
            "rel_deviation": rel_dev,
            "errors": errs,
            "hs": hs,
        }
        if rel_dev > 0.10:
            j3_pass = False
        print(f"  p={p}: observed={obs:.3f}, theory={theory}, deviation={rel_dev*100:.1f}%")
    results["gates"]["J3_curved_wall_order"] = {
        "pass": j3_pass,
        "threshold": "|Ô-(p+1)|/(p+1) ≤ 10%",
        "details": j3_details,
    }
    print(f"  J3: {'PASS' if j3_pass else 'FAIL'}")
    
    # J4: Straight-edge order drop
    print("\n[J4] Straight-edge (g=1) order drop...")
    p = 2
    gdeg = 1
    nr_list = [6, 10, 16]
    nt_list = [12, 20, 32]
    errs, hs = annulus_convergence(p, gdeg, nr_list, nt_list, T=0.15)
    obs = observed_order(errs, hs)
    j4_pass = obs <= 2.2
    results["gates"]["J4_straight_edge"] = {
        "pass": j4_pass,
        "threshold": "Ô ≤ 2.2",
        "observed_order": obs,
    }
    print(f"  p=2, g=1: observed={obs:.3f}, threshold ≤2.2: {'PASS' if j4_pass else 'FAIL'}")
    
    # J5: Curvature error law
    print("\n[J5] Curvature error law...")
    j5_pass = True
    j5_details = {}
    for gdeg in [1, 2, 3]:
        errs = []
        hs = []
        for nr in [4, 8, 16]:
            nt = 2 * nr
            mesh = annulus_mesh(nr, nt, R_INNER, R_OUTER, gdeg)
            nodes, _ = gll_nodes_weights(gdeg)
            max_err = 0.0
            for j in range(min(nt, 8)):
                m = mesh.maps[0][j]
                eta = (nodes + 1.0) * np.pi / nt
                x, y = m.eval(np.array([-1.0]), eta)
                r = np.hypot(x, y)
                max_err = max(max_err, np.max(np.abs(r - R_INNER)))
            errs.append(max_err)
            h = 2.0 * np.pi / nt
            hs.append(h)
        obs_exp = observed_order(errs, hs)
        theory_exp = gdeg + 1
        rel_err = abs(obs_exp - theory_exp) / theory_exp
        j5_details[f"g{gdeg}"] = {
            "observed_exponent": obs_exp,
            "theory_exponent": theory_exp,
            "rel_error": rel_err,
        }
        # Accept superconvergence
        if obs_exp < theory_exp and rel_err > 0.15:
            j5_pass = False
        print(f"  g={gdeg}: observed_exp={obs_exp:.3f}, theory={theory_exp}")
    results["gates"]["J5_curvature_law"] = {
        "pass": j5_pass,
        "threshold": "exponent error ≤ 15% (superconvergence OK)",
        "details": j5_details,
    }
    print(f"  J5: {'PASS' if j5_pass else 'FAIL'}")
    
    # J6: Negative controls
    print("\n[J6] Negative controls...")
    j6_pass = True
    
    # J6-N1: Unit Jacobian
    mesh = warped_periodic_mesh(5, 5, 2, 0.08)
    dg = CurvedAdvection(mesh, 2, const_vel, cfl=0.2)
    dg.Jg = np.ones_like(dg.Jg)
    try:
        fs = dg.free_stream_defect(1.0, steps=10)
        j6_n1_pass = fs["drift"] > 1e-6
    except RuntimeError:
        j6_n1_pass = True  # Instability is a valid failure mode
    j6_pass = j6_pass and j6_n1_pass
    
    # J6-N2: Perturbed metrics
    mesh = warped_periodic_mesh(5, 5, 3, 0.08)
    dg = CurvedAdvection(mesh, 3, const_vel, cfl=0.2)
    rng = np.random.default_rng(42)
    dg.a1g = dg.a1g + 1e-4 * rng.standard_normal(dg.a1g.shape)
    dg.a2g = dg.a2g + 1e-4 * rng.standard_normal(dg.a2g.shape)
    fs = dg.free_stream_defect(1.0, steps=10)
    j6_n2_pass = fs["drift"] > 1e-6
    j6_pass = j6_pass and j6_n2_pass
    
    # J6-N3: Wrong geometry degree
    p = 2
    errs_g1, hs_g1 = annulus_convergence(p, 1, [4, 8], [8, 16], T=0.15)
    obs_g1 = observed_order(errs_g1, hs_g1)
    errs_gp, hs_gp = annulus_convergence(p, p+1, [4, 8], [8, 16], T=0.15)
    obs_gp = observed_order(errs_gp, hs_gp)
    j6_n3_pass = obs_g1 < obs_gp - 0.5
    j6_pass = j6_pass and j6_n3_pass
    
    results["gates"]["J6_negative_controls"] = {
        "pass": j6_pass,
        "threshold": "All three negative controls must fail (gate is RED)",
        "N1_unit_J": j6_n1_pass,
        "N2_perturbed_metrics": j6_n2_pass,
        "N3_wrong_geometry": j6_n3_pass,
    }
    print(f"  J6-N1 (unit J): {'RED' if j6_n1_pass else 'GREEN (BAD)'}")
    print(f"  J6-N2 (perturbed): {'RED' if j6_n2_pass else 'GREEN (BAD)'}")
    print(f"  J6-N3 (wrong g): {'RED' if j6_n3_pass else 'GREEN (BAD)'}")
    print(f"  J6: {'PASS' if j6_pass else 'FAIL'}")
    
    # J7: GCI convergence
    print("\n[J7] Grid convergence...")
    p = 2
    gdeg = 3
    nr_list = [6, 10, 16]
    nt_list = [12, 20, 32]
    errs, hs = annulus_convergence(p, gdeg, nr_list, nt_list, T=0.15)
    r = hs[1] / hs[2]
    p_obs = abs(np.log(errs[1] / errs[2]) / np.log(r))
    theory = p + 1
    rel_dev = abs(p_obs - theory) / theory
    j7_pass = rel_dev <= 0.15
    results["gates"]["J7_grid_convergence"] = {
        "pass": j7_pass,
        "threshold": "observed order within 15% of theory",
        "observed_order": p_obs,
        "theory_order": theory,
        "rel_deviation": rel_dev,
    }
    print(f"  p=2: observed={p_obs:.3f}, theory={theory}, deviation={rel_dev*100:.1f}%")
    print(f"  J7: {'PASS' if j7_pass else 'FAIL'}")
    
    # Summary
    all_pass = all(g["pass"] for g in results["gates"].values())
    results["summary"] = {
        "all_gates_pass": all_pass,
        "total_time": time.time() - t_start,
    }
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'}")
    print("=" * 70)
    
    # Write outputs
    with open(os.path.join(outdir, "result.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    gates_verified = {
        "campaign_node": "P127W2-N01",
        "generated": results["generated"],
        "gates": {k: {"pass": v["pass"], "threshold": v.get("threshold", "")} 
                  for k, v in results["gates"].items()},
        "all_pass": all_pass,
    }
    with open(os.path.join(outdir, "gates_verified.json"), "w") as f:
        json.dump(gates_verified, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nOutputs written to {outdir}")
    print(f"  - result.json")
    print(f"  - gates_verified.json")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
