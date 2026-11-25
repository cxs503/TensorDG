"""Verification gate for the SWE solitary-wave benchmark (TensorDG#1001).

Loads ``artifacts/swe_solitary_wave_metrics.json`` and asserts every key
quantity is inside its < 3 % gate against the Boussinesq analytical solution.
Run with::

    python docs/benchmarks/verify_swe_solitary_wave.py

Exit code 0 = all gates PASS; 1 = any gate FAIL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
GATE = 0.03  # 3 %


def _check(label: str, value: float, gate: float = GATE) -> bool:
    ok = value < gate
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {value*100:.4f}% "
          f"(gate {gate*100:.1f}%)")
    return ok


def main() -> int:
    jpath = _REPO / "artifacts" / "swe_solitary_wave_metrics.json"
    if not jpath.exists():
        print(f"ERROR: {jpath} not found -- run the benchmark first")
        return 1
    d = json.loads(jpath.read_text())

    print("== SWE solitary-wave verification gates ==")
    ok = True

    # main case (a=0.02, 2 widths, N=400, P1)
    m = d["main_case"]
    ok &= _check("main: peak height", m["peak"]["peak_height_rel_err"])
    ok &= _check("main: wave speed", m["peak"]["speed_rel_err"])
    ok &= _check("main: profile L2", m["l2"]["l2_rel"])
    ok &= _check("main: mass drift", m["mass_rel_drift"], gate=1e-6)

    # long case (5 widths)
    l = d["long_case"]
    ok &= _check("long: peak height", l["peak"]["peak_height_rel_err"])
    ok &= _check("long: wave speed", l["peak"]["speed_rel_err"])
    ok &= _check("long: profile L2", l["l2"]["l2_rel"])

    # convergence (all resolutions)
    for r in d["convergence"]:
        ok &= _check(f"conv N={r['ncells']}: peak height",
                     r["peak"]["peak_height_rel_err"])
        ok &= _check(f"conv N={r['ncells']}: wave speed",
                     r["peak"]["speed_rel_err"])

    # time series (all snapshots)
    for r in d["timeseries"]:
        ok &= _check(f"ts t={r['t']:.2f}: peak height",
                     r["peak_height_rel_err"])
        ok &= _check(f"ts t={r['t']:.2f}: wave speed", r["speed_rel_err"])

    # P2
    p2 = d["p2_case"]
    ok &= _check("P2: peak height", p2["peak"]["peak_height_rel_err"])
    ok &= _check("P2: wave speed", p2["peak"]["speed_rel_err"])

    # figures present
    for name, path in d["figures"].items():
        f = _REPO / path
        present = f.exists()
        print(f"  [{'PASS' if present else 'FAIL'}] figure {name}: {f}")
        ok &= present

    print("== " + ("ALL GATES PASS" if ok else "GATE FAILURE") + " ==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
