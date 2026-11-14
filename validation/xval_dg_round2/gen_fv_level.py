"""Generate one FV-WENO5 reference level of the Shu--Osher ladder (ladder extension).

Why this exists
---------------
``CASE-SHU-OSHER-round2.md`` §6 records one honest unfinished item: the reference
ladder stopped at three levels ``[800, 1600, 3200]`` because the fourth level
(``N = 6400``) was submitted to the cluster and never dispatched.  A three-level
ladder fixes the observed order from a *single* successive difference and the
Richardson band from that one order estimate; a fourth level supplies a second
successive difference, i.e. the only in-ladder evidence that the asymptotic
range has actually been reached.  ``benchmarks/xval_dg_round2_ladder_extension``
turns that into a predeclared test (does the three-level band cover the shift of
the certified profile when the finer level arrives?).

This driver only *produces the raw profile* -- it makes no verification claim.
It is the exact call path that produced the committed ``fv_cfl0p3_n800/1600/3200``
files (``benchmarks.shu_osher_case_pool.run_fv`` -> ``solve_fv_weno5`` with
``CFL = 0.3``, ``t = 1.8``, domain ``[-5, 5]``), so the new level is
ladder-consistent by construction; the reproduction check against the committed
levels is what makes that claim testable rather than asserted.

Usage::

    python validation/xval_dg_round2/gen_fv_level.py --ncells 6400 \
        --out validation/repro_shu_osher/artifacts/raw/fv_cfl0p3_n6400.npz
    python validation/xval_dg_round2/gen_fv_level.py --ncells 800 --out /tmp/repro.npz \
        --verify-committed      # bit-for-bit check against the shipped profile
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "validation" / "repro_shu_osher" / "artifacts" / "raw"


def solve_level(ncells: int, cfl: float = 0.3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the reference solver exactly as the committed ladder was run."""
    from benchmarks.fv_weno5_1d import solve_fv_weno5
    from benchmarks.shu_osher_case_pool import DOMAIN, FINAL_TIME, initial_state

    t0 = time.perf_counter()
    x, u, meta = solve_fv_weno5(initial_state, ncells=int(ncells), cfl=float(cfl),
                                final_time=FINAL_TIME, domain=DOMAIN)
    wall = time.perf_counter() - t0
    return np.asarray(x, dtype=float), np.asarray(u, dtype=float), {
        "ncells": int(ncells), "cfl": float(cfl), "final_time": float(FINAL_TIME),
        "domain": list(DOMAIN), "steps": int(meta.get("steps", -1)),
        "wall_time_s": round(wall, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ncells", type=int, default=6400)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-committed", action="store_true",
                    help="compare against the shipped profile of the same size, bit for bit")
    args = ap.parse_args(argv)

    x, u, meta = solve_level(args.ncells, args.cfl)
    rho, rho_u, rho_e = u[:, 0], u[:, 1], u[:, 2]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x=x, rho=rho, rho_u=rho_u, rho_e=rho_e, meta=json.dumps(meta))

    summary = {"out": str(out), "meta": meta,
               "rho_min": float(rho.min()), "rho_max": float(rho.max()),
               "admissible": bool(np.all(np.isfinite(rho)) and rho.min() > 0)}
    if args.verify_committed:
        shipped = RAW / f"fv_cfl0p3_n{args.ncells}.npz"
        if shipped.exists():
            blob = np.load(shipped)
            summary["verify_committed"] = {
                "path": str(shipped),
                "max_abs_drho": float(np.abs(rho - blob["rho"]).max()),
                "x_identical": bool(np.array_equal(x, blob["x"])),
                "max_abs_dmom": float(np.abs(rho_u - blob["rho_u"]).max()),
                "max_abs_denergy": float(np.abs(rho_e - blob["rho_e"]).max()),
            }
        else:
            summary["verify_committed"] = {"path": str(shipped), "status": "missing"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())