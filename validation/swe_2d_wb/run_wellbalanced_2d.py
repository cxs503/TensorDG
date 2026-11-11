"""Validation of the 2-D hydrostatic well-balanced path of ``tensordg.swe_2d``.

Three experiments, all of them machine-checkable:

1. **Lake at rest on a curved bed** -- a still lake (``η = h + z_b = const``,
   ``u = v = 0``) on the quadratic bed ``z_b = 0.2 x + 0.05 x^2`` is advanced in
   time and the spurious current is measured.  The well-balanced path is
   compared against the default (cell-constant bed slope) path on a sequence of
   meshes, which shows the O(dx) drift of the default path and the round-off
   level of the well-balanced one.

2. **Lake at rest across a bed jump with a dry bank** -- the bed is given as
   cell-local nodal data so that the 2.5 m step sits exactly on a cell face.
   The wet cell then faces a dry neighbour; the Audusse hydrostatic
   reconstruction has to make this a discrete fixed point as well.

3. **Unsteady dam break over a curved hump** (periodic) -- checks that the
   well-balanced path stays stable and conserves mass to round-off, i.e. that
   the exactness above is not bought with a loss of robustness.

The JSON artifact carries a ``provenance`` block (UTC time, command line, git
revision and dirty flag, versions, host) so a stale or hand-edited file is
detectable; see ``validation/swe_thacker/run_hydrostatic_wb.py`` for the same
convention in 1-D.

Run with::

    PYTHONPATH=. python validation/swe_2d_wb/run_wellbalanced_2d.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from tensordg.swe_2d import (
    DRY_DEPTH,
    check_hydrostatic_equilibrium_2d,
    solve_swe_2d,
)

ETA0 = 2.0          # still-water level
DOMAIN = ((0.0, 1.0), (0.0, 1.0))
ETA_TOL = 1e-8      # m
VEL_TOL = 1e-8      # m/s


def collect_provenance(repo_root: Path | None = None) -> dict:
    """Reproducibility metadata for the JSON artifact (same keys as in 1-D)."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:  # pragma: no cover - git absent / not a repo
            return "unknown"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "generator": "validation/swe_2d_wb/run_wellbalanced_2d.py",
    }


def _zeros(x, y):
    return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# 1. still lake on a curved bed
# ---------------------------------------------------------------------------

def curved_bed(x, y):
    """Quadratic bed: the cell-constant slope of the default path is not exact."""
    return 0.2 * x + 0.05 * x ** 2


def run_curved_rest(nx: int, ny: int, degree: int, well_balanced: bool,
                    final_time: float = 0.1) -> dict:
    bed = curved_bed
    h = lambda x, y: ETA0 - bed(x, y)          # noqa: E731
    t0 = time.perf_counter()
    x, y, coeff, zb_faces, meta = solve_swe_2d(
        h, _zeros, _zeros, bed, nx=nx, ny=ny, degree=degree,
        final_time=final_time, domain=DOMAIN, cfl=0.3,
        well_balanced=well_balanced)
    wall = time.perf_counter() - t0

    zb_modes = meta["zb_modes"]
    if zb_modes is None:
        # default path: cell-average bed of the quadratic used for the metric
        dx, dy = 1.0 / nx, 1.0 / ny
        zb_modes = torch.zeros(nx, ny, degree + 1, degree + 1, dtype=coeff.dtype)
        zb_modes[:, :, 0, 0] = (0.2 * x[:, None] + 0.05
                                * (x[:, None] ** 2 + dx ** 2 / 12.0))
    check = check_hydrostatic_equilibrium_2d(
        coeff, zb_modes, eta_expected=ETA0, tolerance=VEL_TOL)
    return {
        "nx": nx, "ny": ny, "degree": degree,
        "well_balanced": bool(well_balanced),
        "final_time": final_time,
        "steps": meta["steps"],
        "dt": meta["dt"],
        "eta_error_max": check["eta_error_max"],
        "u_max": check["u_max"],
        "v_max": check["v_max"],
        "is_hydrostatic": check["is_hydrostatic"],
        "wall_time_s": wall,
    }


# ---------------------------------------------------------------------------
# 2. still lake across a bed jump with a dry bank
# ---------------------------------------------------------------------------

def run_bed_jump_rest(nx: int, ny: int, degree: int,
                      final_time: float = 0.05) -> dict:
    nmodes = degree + 1
    zb = torch.zeros(1, 1, nmodes, nmodes).repeat(nx, ny, 1, 1)
    jump_cell = nx // 2
    zb[jump_cell:] = 2.5                     # dry bank, shoreline on a cell face
    h = (ETA0 - zb).clamp_min(0.0)
    zero = torch.zeros_like(zb)

    t0 = time.perf_counter()
    x, y, coeff, zb_faces, meta = solve_swe_2d(
        h, zero, zero, zb, nx=nx, ny=ny, degree=degree, final_time=final_time,
        domain=DOMAIN, cfl=0.3, well_balanced=True)
    wall = time.perf_counter() - t0

    check = check_hydrostatic_equilibrium_2d(
        coeff, meta["zb_modes"], eta_expected=ETA0, tolerance=VEL_TOL)
    return {
        "nx": nx, "ny": ny, "degree": degree,
        "jump_cell": jump_cell, "jump_height_m": 2.5,
        "steps": meta["steps"],
        "eta_error_max": check["eta_error_max"],
        "u_max": check["u_max"],
        "v_max": check["v_max"],
        "is_hydrostatic": check["is_hydrostatic"],
        "dry_cells_remain_dry": bool(float(coeff[jump_cell:, :, :, :, 0].abs().max()) < ETA_TOL),
        "wall_time_s": wall,
    }


# ---------------------------------------------------------------------------
# 3. unsteady dam break over a periodic hump
# ---------------------------------------------------------------------------

def run_dam_break(nx: int = 16, ny: int = 8, degree: int = 1,
                  final_time: float = 0.08) -> dict:
    def bed(x, y):
        return 0.15 * (1.0 - torch.cos(2.0 * torch.pi * x)) * \
            (1.0 - torch.cos(2.0 * torch.pi * y))

    # Cell-local nodal depth so that the dam break sits exactly on the face
    # x = 0.5; a callable would be sampled once per shared node and smear the
    # jump, which changes the discrete initial mass.
    nmodes = degree + 1
    h = torch.full((nx, ny, nmodes, nmodes), 0.4)
    h[: nx // 2] = 1.0
    zero = torch.zeros_like(h)

    t0 = time.perf_counter()
    x, y, coeff, zb_faces, meta = solve_swe_2d(
        h, zero, zero, bed, nx=nx, ny=ny, degree=degree,
        final_time=final_time, domain=DOMAIN, cfl=0.3,
        periodic=(True, True), well_balanced=True)
    wall = time.perf_counter() - t0

    dx, dy = 1.0 / nx, 1.0 / ny
    h_avg = coeff[:, :, 0, 0, 0]
    mass = float(h_avg.sum()) * dx * dy
    mass0 = 0.7            # 0.5*1.0 + 0.5*0.4; the jump sits on the face
    return {
        "nx": nx, "ny": ny, "degree": degree,
        "final_time": final_time, "steps": meta["steps"],
        "mass_initial": mass0,
        "mass_final": mass,
        "mass_drift": abs(mass - mass0),
        "h_min": float(h_avg.min()),
        "finite": bool(torch.isfinite(coeff).all()),
        "wall_time_s": wall,
    }


def main() -> dict:
    torch.set_default_dtype(torch.float64)
    print("=" * 72)
    print("TensorDG 2-D SWE hydrostatic well-balanced validation")
    print("=" * 72)

    print("\n--- 1. still lake on a quadratic bed ---")
    curved = []
    for degree in (1, 2):
        for nx in (8, 16, 32):
            for wb in (False, True):
                r = run_curved_rest(nx=nx, ny=4, degree=degree, well_balanced=wb)
                curved.append(r)
                print(f"  P{degree} nx={nx:3d} wb={int(wb)}  "
                      f"η_err={r['eta_error_max']:.3e} m  "
                      f"|u|={r['u_max']:.3e} m/s  {r['wall_time_s']:.1f}s")

    print("\n--- 2. still lake across a bed jump (dry bank) ---")
    jumps = []
    for degree in (1, 2):
        r = run_bed_jump_rest(nx=16, ny=6, degree=degree)
        jumps.append(r)
        print(f"  P{degree}  η_err={r['eta_error_max']:.3e} m  "
              f"|u|={r['u_max']:.3e} m/s  dry={r['dry_cells_remain_dry']}  "
              f"{r['wall_time_s']:.1f}s")

    print("\n--- 3. unsteady dam break over a periodic hump ---")
    dynamic = run_dam_break()
    print(f"  steps={dynamic['steps']}  mass={dynamic['mass_final']:.12f} "
          f"(drift {dynamic['mass_drift']:.2e})  h_min={dynamic['h_min']:.4f}  "
          f"{dynamic['wall_time_s']:.1f}s")

    curved_wb = [r for r in curved if r["well_balanced"]]
    curved_default = [r for r in curved if not r["well_balanced"]]
    pass_curved = all(r["eta_error_max"] < ETA_TOL and r["u_max"] < VEL_TOL
                      for r in curved_wb)
    pass_jump = all(r["eta_error_max"] < ETA_TOL and r["u_max"] < VEL_TOL
                    and r["dry_cells_remain_dry"] for r in jumps)
    pass_dynamic = (dynamic["finite"] and dynamic["mass_drift"] < 1e-10
                    and dynamic["h_min"] > 0.0)

    results = {
        "provenance": collect_provenance(),
        "setup": {
            "still_water_level_m": ETA0,
            "domain": DOMAIN,
            "cfl": 0.3,
            "dtype": "float64",
            "curved_bed": "z_b = 0.2 x + 0.05 x^2",
            "bed_jump_height_m": 2.5,
            "dry_depth_m": DRY_DEPTH,
        },
        "curved_bed_rest": curved,
        "bed_jump_rest": jumps,
        "dam_break": dynamic,
        "summary": {
            "test_description": (
                "2-D hydrostatic well-balancedness: lake at rest on a curved bed "
                "and across a bed jump, plus an unsteady conservation check"
            ),
            "pass_criteria": {
                "well_balanced_eta_error_max": "< 1e-8 m",
                "well_balanced_velocity_max": "< 1e-8 m/s",
                "dam_break_mass_drift": "< 1e-10 (absolute, reference 0.7 m^3)",
            },
            "baseline_contrast": {
                "max_u_max_default_path": max(r["u_max"] for r in curved_default),
                "max_u_max_well_balanced": max(r["u_max"] for r in curved_wb),
                "note": (
                    "Same initial state, same grids: the default path's "
                    "cell-constant bed slope leaves a spurious current that the "
                    "well-balanced path removes."
                ),
            },
            "pass": bool(pass_curved and pass_jump and pass_dynamic),
        },
    }

    out_dir = Path(__file__).parent
    out_file = out_dir / "wellbalanced_2d_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ results written to {out_file}")

    print("\n" + "=" * 72)
    print(f"curved-bed rest (WB):  {'PASS' if pass_curved else 'FAIL'}")
    print(f"bed-jump rest     :    {'PASS' if pass_jump else 'FAIL'}")
    print(f"dam break         :    {'PASS' if pass_dynamic else 'FAIL'}")
    print(f"overall           :    {'PASS' if results['summary']['pass'] else 'FAIL'}")
    print(f"spurious |u|: default path {results['summary']['baseline_contrast']['max_u_max_default_path']:.3e} m/s"
          f" -> well balanced {results['summary']['baseline_contrast']['max_u_max_well_balanced']:.3e} m/s")
    print("=" * 72)
    return results


if __name__ == "__main__":
    main()