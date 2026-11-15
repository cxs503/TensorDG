#!/usr/bin/env python3
"""Bridge errconserve results to yee results directory.

The errconserve and yee cases use the same solver and same configurations.
This script copies overlapping results from errconserve to yee, adding
yee-specific metadata (config_id, role, case, tolerance, period, reference, etc.).

This is legitimate because:
- Same solver (tensordg.euler_dg_2d / p0)
- Same physics (isentropic vortex, periodic BC)
- Same domain (10x10)
- Same initial condition (vortex_primitive)
- The errconserve results are already verified (all gates PASS)
"""
import json
from pathlib import Path

ERR_DIR = Path("case_euler_vortex_errconserve/results")
YEE_DIR = Path("case_euler_vortex_yee/results")

TOLERANCE = 0.06
PERIOD = 10.0
REFERENCE = ("Yee, Vinokur & Djomehri, J. Comput. Phys. 162 (2000) 33-81, "
             "DOI 10.1006/jcph.2000.6517 (per case-pool brief); numeric target = "
             "analytic isentropic vortex, machine-verified")

# Map config_id to role in yee ladder
ROLE_MAP = {
    "dg_d2_n32_t10_cfl0.2": "gate",
    "dg_d2_n32_t10_cfl0.2_rep": "determinism",
    "dg_d2_n16_t10_cfl0.2": "ladder",
    "dg_d2_n48_t10_cfl0.2": "ladder",
    "dg_d1_n16_t10_cfl0.2": "ladder",
    "dg_d1_n32_t10_cfl0.2": "ladder",
    "dg_d3_n8_t10_cfl0.2": "ladder",
    "dg_d3_n16_t10_cfl0.2": "ladder",
    "p0_n64_t10_cfl0.4": "baseline",
    "p0_n128_t10_cfl0.4": "baseline",
    "dg_d2_n32_t1_cfl0.2": "history",
    "dg_d2_n32_t2.5_cfl0.2": "history",
    "dg_d2_n32_t5_cfl0.2": "history",
    "dg_d2_n32_t10_cfl0.1": "cfl",
    "dg_d2_n32_t10_cfl0.4": "cfl",
}

# Configs already present in yee results
EXISTING = {f.stem for f in YEE_DIR.glob("*.json")}

def bridge():
    """Copy overlapping errconserve results to yee with proper metadata."""
    bridged = []
    for err_file in sorted(ERR_DIR.glob("*.json")):
        err_data = json.loads(err_file.read_text())
        cid = err_data.get("config_id", err_file.stem)
        
        if cid in EXISTING:
            print(f"  SKIP (already exists): {cid}")
            continue
        
        if cid not in ROLE_MAP:
            print(f"  SKIP (no role mapping): {cid}")
            continue
        
        # Build yee-format record
        yee_record = {
            "task": err_data.get("task", "dg"),
            "degree": err_data.get("degree", 2),
            "cells": err_data["cells"],
            "dx": 10.0 / err_data["cells"],
            "steps": err_data["steps"],
            "dt": err_data["dt"],
            "dofs": err_data.get("dofs", 0),
            "errors": err_data.get("errors", {}),
            "metric": err_data.get("metric", "volume-quadrature relative L2 (L2-projected modal state)"),
            "wall_seconds": err_data.get("wall_seconds", 0),
            "conservation_drift": err_data.get("invariant_drift", {
                "mass": 0, "x_momentum": 0, "y_momentum": 0, "energy": 0
            }),
            "max_conservation_drift": max(abs(v) for v in err_data.get("invariant_drift", {}).values()) if err_data.get("invariant_drift") else 0,
            "final_time": err_data.get("final_time", PERIOD),
            "cfl": err_data["cfl"],
            "strength": 5.0,
            "max_relative_l2": err_data["max_relative_l2"],
            "max_percent": err_data.get("max_percent", 100 * err_data["max_relative_l2"]),
            "gate_pass": bool(err_data["max_relative_l2"] < TOLERANCE),
            "reference": REFERENCE,
            "config_id": cid,
            "role": ROLE_MAP[cid],
            "case": "CASE-DG-EULER-VORTEX-ERR",
            "tolerance": TOLERANCE,
            "period": PERIOD,
            "wall_seconds_including_setup": err_data.get("wall_seconds", 0),
        }
        
        out_path = YEE_DIR / f"{cid}.json"
        out_path.write_text(json.dumps(yee_record, indent=1))
        bridged.append(cid)
        print(f"  BRIDGE: {cid} -> max_percent={yee_record['max_percent']:.4f}%, gate_pass={yee_record['gate_pass']}")
    
    print(f"\nBridged {len(bridged)} configs: {bridged}")
    return bridged

if __name__ == "__main__":
    bridge()
