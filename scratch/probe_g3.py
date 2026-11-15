"""Probe G3 (embedded-wall shock reflection) in both arms at a coarse ladder.

Read-only driver around ``benchmarks.forward_facing_step.gate_wall_reflection``.
Writes one JSON block per configuration to stdout; nothing is patched.
"""
import json
import sys
import time

from benchmarks.forward_facing_step import gate_wall_reflection

CASES = [
    dict(ncells=(120, 1), degree=1),
    dict(ncells=(240, 1), degree=1),
    dict(ncells=(240, 1), degree=2),
]

out = {}
for case in CASES:
    for mode in ("boundary", "solid"):
        t0 = time.time()
        res = gate_wall_reflection(wall_mode=mode, x_wall=1.0, x_shock=0.4,
                                   xmax=1.2, extra_time=0.08, **case)
        key = f"{case['ncells'][0]}x{case['ncells'][1]}_p{case['degree']}_{mode}"
        out[key] = {
            "status": res["status"], "pass": res["pass"],
            "plateau_cells": res["plateau_cells"],
            "worst_pct": res["worst_relative_error_percent"],
            "mean_rho_pct": res["plateau_mean_rho_error_percent"],
            "mean_p_pct": res["plateau_mean_p_error_percent"],
            "l1_rho": res["l1_relative"]["density"], "linf_rho": res["linf_relative"]["density"],
            "l1_p": res["l1_relative"]["pressure"], "linf_p": res["linf_relative"]["pressure"],
            "steps": res["steps"], "wall_time": res["wall_time"],
            "exact_rho5": res["reflected_rho_exact"], "exact_p5": res["reflected_p_exact"],
            "mass_res": res["mass_relative_residual"],
            "energy_res": res["energy_relative_residual"],
            "wall_s": round(time.time() - t0, 1),
        }
        print(key, json.dumps(out[key]), flush=True)

print("PROBE_JSON", json.dumps(out), flush=True)