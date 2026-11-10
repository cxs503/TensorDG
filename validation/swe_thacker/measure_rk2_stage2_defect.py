#!/usr/bin/env python3
"""Measure the SSP-RK2 stage-2 defect (time argument + boundary data).

The SSP-RK2 branch of ``solve_swe_1d`` used to evaluate its second stage without
the time argument and without ``bc_left_flux`` / ``bc_right_flux``, so a run
with analytic boundary data silently used the transmissive closure at
``t = 0`` in that stage.  This script measures the effect by loading a
*baseline* module tree (extracted from a git ref) and the current tree side by
side and running the same Thacker planar-basin case, which exchanges water
through a time-dependent rim.

Usage
-----
    # fixed tree only
    python validation/swe_thacker/measure_rk2_stage2_defect.py

    # fixed tree + a pristine baseline extracted from a git ref
    python validation/swe_thacker/measure_rk2_stage2_defect.py --baseline-ref a644b46

Writes ``validation/swe_thacker/rk2_stage2_defect_measurement.json``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "rk2_stage2_defect_measurement.json"
MODULE_FILES = ("swe_1d.py", "limiters.py", "euler_1d.py", "tvb.py",
                "swe_exact.py")


def _purge() -> None:
    for name in [m for m in sys.modules if m == "tensordg" or m.startswith("tensordg.")]:
        del sys.modules[name]


def load_package(root: Path):
    """Import ``tensordg.swe_1d`` / ``tensordg.swe_exact`` from ``root``."""
    _purge()
    sys.path.insert(0, str(root))
    try:
        swe = importlib.import_module("tensordg.swe_1d")
        exact = importlib.import_module("tensordg.swe_exact")
    finally:
        sys.path.remove(str(root))
        _purge()
    return swe, exact


def extract_baseline(ref: str, dest: Path) -> Path:
    """Materialise the ``tensordg`` package of ``ref`` into ``dest``."""
    pkg = dest / "tensordg"
    pkg.mkdir(parents=True, exist_ok=True)
    # Empty package marker: only the modules used by the measurement are
    # extracted, so the full ``tensordg/__init__.py`` (which imports the whole
    # library) cannot be used here.
    (pkg / "__init__.py").write_text("")
    for name in MODULE_FILES:
        blob = subprocess.run(
            ["git", "show", f"{ref}:tensordg/{name}"], cwd=REPO,
            capture_output=True, text=True, check=True).stdout
        (pkg / name).write_text(blob)
    return dest


def volume_error(swe, exact, rk_order: int, ncells: int = 40,
                 t_end: float = 0.5) -> dict:
    h0, amp, half, g = 1.0, 0.5, 1.0, 9.81

    def bed(xx):
        return torch.as_tensor(exact.thacker_parabolic_bed(xx.numpy(), h0, half),
                               dtype=xx.dtype)

    def init_h(xx):
        h, _ = exact.thacker_planar_basin_initial(xx.numpy(), h0, amp, half)
        return torch.as_tensor(h, dtype=xx.dtype)

    def left(t):
        return torch.as_tensor(exact.thacker_planar_basin_boundary_flux(
            t, h0, amp, half, g)[0])

    def right(t):
        return torch.as_tensor(exact.thacker_planar_basin_boundary_flux(
            t, h0, amp, half, g)[1])

    calls: list[float] = []

    def left_rec(t):
        calls.append(float(t))
        return left(t)

    x, coeff, zb_faces, meta = swe.solve_swe_1d(
        init_h, lambda xx: torch.zeros_like(xx), bed, ncells=ncells, degree=1,
        final_time=t_end, domain=(-half, half), cfl=0.3, limiter=None,
        rk_order=rk_order, well_balanced=True, bc_left_flux=left_rec,
        bc_right_flux=right)
    h, _, _ = swe.reconstruct_fields(coeff, zb_faces,
                                     zb_center=meta["zb_modes"][:, 0])
    eta_c = h0 + (amp ** 2 / (4.0 * h0)) * math.sin(
        math.sqrt(2.0 * g * h0) / half * t_end) ** 2
    exact_mass = 2 * half * eta_c + 4.0 * h0 * half / 3.0
    mass = float(h.sum()) * (2 * half / ncells)
    return {
        "rk_order": rk_order,
        "volume_rel_error": abs(mass - exact_mass) / exact_mass,
        "bc_call_count": len(calls),
        "bc_time_max": max(calls),
        "bc_time_max_expected": t_end,
        "steps": meta["steps"],
    }


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ref", default=None,
                    help="git ref of the pre-fix tree (e.g. the commit before "
                         "the RK2 stage-2 fix)")
    args = ap.parse_args()

    results: dict = {"case": {
        "description": "Thacker (1981) planar basin, analytic boundary fluxes "
                       "(mass exchanged through the rim, so the boundary data "
                       "must be time dependent)",
        "ncells": 40, "degree": 1, "final_time": 0.5, "cfl": 0.3,
    }}
    swe, exact = load_package(REPO)
    results["fixed_tree"] = {f"rk{rk}": volume_error(swe, exact, rk) for rk in (2, 3)}

    if args.baseline_ref:
        with tempfile.TemporaryDirectory() as tmp:
            root = extract_baseline(args.baseline_ref, Path(tmp))
            swe_b, exact_b = load_package(root)
            results[f"baseline_{args.baseline_ref}"] = {
                f"rk{rk}": volume_error(swe_b, exact_b, rk) for rk in (2, 3)}

    OUT.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
