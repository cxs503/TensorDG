"""Dual-instrument measurement of the two 3 %-gate rows the census left ``not_run``.

Why this module exists
----------------------
``validation/repro_cks1998/instrument_census.py`` (m22969) established that the
Sod 3 %-gate family is scored against a reference instrument, and that the
legacy instrument (``benchmarks.riemann_exact.cell_averages``: a *fixed* 8-point
Gauss rule per cell) does not converge inside cells that contain a wave.  It
switched the anchor's own three measurement sites to the exact instrument
(``benchmarks.shock_reference_quadrature.cell_averages``: breakpoint-split,
converges to machine precision) and *registered* the remaining 18 gating sites.

It then had to leave one question open, in its own words (§5-2, §6):

    DG P2/N=400 and extruded 2D P0/N=3200 -- the **dual-instrument measurement**
    was not run; the verdict-flip question is answered by the *bound*
    (margin 0.63 pp > bias 0.38 pp), not by a re-run.  This item is **not
    judged**: no number.

That is this module's whole scope: **measure those two rows with both
instruments on the same numerical solution** and report, per row, the per-field
relative L2 under each instrument, the displacement in percentage points, the
two verdicts, and whether the verdict flips.  Nothing here changes a tolerance,
a published number, or a gate.

Rows (the actual production configurations, read off their own modules):

===========================  ==================================================
row                          configuration
===========================  ==================================================
``E2_dg_p2_n400``            ``tensordg.euler_dg_1d.solve_euler_dg_1d``,
                             ``tests/test_repro_cks1998.py``: N=400, P2, CFL 0.12,
                             TVB limiter (M=1, final stage), t=0.2, gate 3 %
``extruded2d_p0_n3200``      ``benchmarks.verify_euler_2d_p0.run_case``,
                             ``tests/test_euler_2d_p0_verification.py``:
                             nx=3200 x ny=4 extruded Sod, CFL 0.4, t=0.2, gate 3 %
===========================  ==================================================

The numerical solution is computed **once per row**; both reference instruments
are then applied to the *same* array.  So the measured quantity is exactly the
instrument's contribution to the gate, with the solver held fixed.

Usage
-----
    OMP_NUM_THREADS=1 PYTHONPATH=. python -m validation.repro_cks1998.gate_row_swap
    OMP_NUM_THREADS=1 PYTHONPATH=. python -m validation.repro_cks1998.gate_row_swap \\
        --rows E2_dg_p2_n400
    ... --json validation/cks1998_gate_row_swap/rows.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from validation.repro_cks1998.station_reference import (
    SOD_T,
    exact_cell_averages,
    fixed_rule_cell_averages,
)

#: The gate these rows are scored against, unchanged and owned by the rows' own
#: modules (``tests/test_repro_cks1998.py`` TOLERANCE, ``verify_euler_2d_p0.GATE``).
GATE = 0.03

#: The two rows the census left ``not_run``.  Field names are per row because the
#: 1-D DG row reads ``velocity`` while the extruded-2D row reads ``velocity_x``.
ROWS: Dict[str, Dict[str, Any]] = {
    "E2_dg_p2_n400": {
        "kind": "dg1d",
        "source": "tests/test_repro_cks1998.py::"
                  "test_sod_via_dg_p2_matches_exact_riemann_solution_within_three_percent",
        "ncells": 400,
        "degree": 2,
        "cfl": 0.12,
        "limiter": "tvb",
        "tvb_constant": 1.0,
        "limiter_stage": "final",
        "final_time": SOD_T,
        "fields": ("density", "velocity", "pressure"),
        "recorded_max_percent": 2.368,  # published (evidence.md, E2_worst_rel_l2 p2_n400)
    },
    "extruded2d_p0_n3200": {
        "kind": "extruded2d_p0",
        "source": "tests/test_euler_2d_p0_verification.py (gate row); "
                  "benchmarks/verify_euler_2d_p0.check_a_extruded_sod finest row",
        "nx": 3200,
        "ny": 4,
        "cfl": 0.4,
        "periodic": (False, True),
        "final_time": SOD_T,
        "fields": ("density", "velocity_x", "pressure"),
        "recorded_max_percent": None,
    },
}


# ---------------------------------------------------------------------------
# solvers (the rows' own production entry points; nothing is re-implemented)
# ---------------------------------------------------------------------------
def _solve_dg1d(spec: Mapping[str, Any]) -> Dict[str, Any]:
    from tensordg.euler_1d import GAMMA
    from tensordg.euler_dg_1d import solve_euler_dg_1d

    _x, coeff, meta = solve_euler_dg_1d(
        (1.0, 0.0, 1.0),
        (0.125, 0.0, 0.1),
        ncells=spec["ncells"],
        degree=spec["degree"],
        final_time=spec["final_time"],
        cfl=spec["cfl"],
        limiter=spec["limiter"],
        tvb_constant=spec["tvb_constant"],
        limiter_stage=spec["limiter_stage"],
    )
    avg = coeff[:, 0]
    fields = {
        "density": avg[:, 0].numpy(),
        "velocity": (avg[:, 1] / avg[:, 0]).numpy(),
        "pressure": ((GAMMA - 1.0) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / avg[:, 0])).numpy(),
    }
    return {"fields": fields, "meta": _scalars(meta)}


def _solve_extruded2d_p0(spec: Mapping[str, Any]) -> Dict[str, Any]:
    from benchmarks.verify_euler_2d_p0 import run_case

    _x, _y, raw, meta, wall = run_case(
        spec["nx"], spec["ny"], spec["final_time"], spec["cfl"], tuple(spec["periodic"])
    )
    fields = {
        "density": raw["density"][:, 0].cpu().numpy(),
        "velocity_x": raw["velocity_x"][:, 0].cpu().numpy(),
        "pressure": raw["pressure"][:, 0].cpu().numpy(),
    }
    info = _scalars(meta)
    info["wall_seconds_solver_stage"] = wall
    return {"fields": fields, "meta": info}


def _scalars(meta: Mapping[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in dict(meta).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------
def relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    """Same norm as every Sod gate in the repository."""
    absolute = float(np.sqrt(np.mean((numerical - reference) ** 2)))
    norm = float(np.sqrt(np.mean(reference ** 2)))
    if norm == 0.0:
        raise ValueError("reference norm is zero")
    return absolute / norm


def measure_row(key: str, spec: Mapping[str, Any] | None = None,
                profile_dir: Path | None = None) -> Dict[str, Any]:
    """Solve once, score twice: exact instrument vs legacy fixed-rule instrument."""
    spec = ROWS[key] if spec is None else spec
    ncells = spec.get("ncells", spec.get("nx"))
    start = time.perf_counter()
    solved = _solve_dg1d(spec) if spec["kind"] == "dg1d" else _solve_extruded2d_p0(spec)
    wall = time.perf_counter() - start

    # Both helpers return ``(centers, {field: values})`` -- the cell-average dict is
    # the second element; the abscissae are not needed here.
    _c_exact, exact = exact_cell_averages(ncells, spec["final_time"])
    _c_legacy, legacy = fixed_rule_cell_averages(ncells, spec["final_time"])

    rows: Dict[str, Dict[str, float]] = {}
    for field in spec["fields"]:
        got = np.asarray(solved["fields"][field], dtype=np.float64)
        rel_exact = relative_l2(got, np.asarray(exact[field], dtype=np.float64))
        rel_legacy = relative_l2(got, np.asarray(legacy[field], dtype=np.float64))
        rows[field] = {
            "rel_l2_exact": rel_exact,
            "rel_l2_legacy": rel_legacy,
            "percent_exact": 100.0 * rel_exact,
            "percent_legacy": 100.0 * rel_legacy,
            "delta_pp": 100.0 * (rel_exact - rel_legacy),
            "pass_exact": rel_exact < GATE,
            "pass_legacy": rel_legacy < GATE,
        }

    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        payload = {f"numerical_{field}": np.asarray(solved["fields"][field], dtype=np.float64)
                   for field in spec["fields"]}
        for field in spec["fields"]:
            payload[f"exact_{field}"] = np.asarray(exact[field], dtype=np.float64)
            payload[f"legacy_{field}"] = np.asarray(legacy[field], dtype=np.float64)
        dx = 1.0 / float(ncells)
        payload["x"] = (np.arange(ncells) + 0.5) * dx
        np.savez(profile_dir / f"{key}.npz", **payload)

    worst_exact = max(rows[f]["rel_l2_exact"] for f in rows)
    worst_legacy = max(rows[f]["rel_l2_legacy"] for f in rows)
    max_delta_pp = max(abs(rows[f]["delta_pp"]) for f in rows)
    return {
        "row": key,
        "source": spec["source"],
        "ncells": ncells,
        "final_time": spec["final_time"],
        "gate_percent": 100.0 * GATE,
        "solver_meta": solved["meta"],
        "wall_seconds_total": wall,
        "fields": rows,
        "max_percent_exact": 100.0 * worst_exact,
        "max_percent_legacy": 100.0 * worst_legacy,
        "verdict_exact": "PASS" if worst_exact < GATE else "FAIL",
        "verdict_legacy": "PASS" if worst_legacy < GATE else "FAIL",
        "verdict_flip": (worst_exact < GATE) != (worst_legacy < GATE),
        "margin_pp_exact": 100.0 * (GATE - worst_exact),
        "margin_pp_legacy": 100.0 * (GATE - worst_legacy),
        "instrument_displacement_pp": max_delta_pp,
        "recorded_max_percent": spec.get("recorded_max_percent"),
        "reproduces_recorded": (
            None
            if spec.get("recorded_max_percent") is None
            else _close(100.0 * worst_exact, float(spec["recorded_max_percent"]))
        ),
    }


def _close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def build_payload(keys: Sequence[str] | None = None,
                  profile_dir: Path | None = None) -> Dict[str, Any]:
    keys = list(ROWS) if keys is None else list(keys)
    rows = [measure_row(key, profile_dir=profile_dir) for key in keys]
    flips = [row["row"] for row in rows if row["verdict_flip"]]
    return {
        "generated_by": "validation/repro_cks1998/gate_row_swap.py",
        "instrument_policy": {
            "exact": "benchmarks.shock_reference_quadrature.cell_averages "
                     "(breakpoint-split, converges to machine precision)",
            "legacy": "benchmarks.riemann_exact.cell_averages "
                      "via benchmarks.sod_exact (fixed 8-point Gauss rule)",
        },
        "gate_percent": 100.0 * GATE,
        "rows": rows,
        "verdict_flips": flips,
        "all_rows_pass_under_exact_instrument": all(
            row["verdict_exact"] == "PASS" for row in rows
        ),
        "max_instrument_displacement_pp": max(
            (row["instrument_displacement_pp"] for row in rows), default=math.nan
        ),
    }


def render(payload: Mapping[str, Any]) -> str:
    lines = [
        "| row | field | exact % | legacy % | delta pp | verdict exact / legacy |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        for field, stats in row["fields"].items():
            lines.append(
                f"| {row['row']} | {field} | {stats['percent_exact']:.4f} | "
                f"{stats['percent_legacy']:.4f} | {stats['delta_pp']:+.4f} | "
                f"{'PASS' if stats['pass_exact'] else 'FAIL'} / "
                f"{'PASS' if stats['pass_legacy'] else 'FAIL'} |"
            )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", choices=sorted(ROWS), default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--profiles", type=Path, default=None,
                        help="dump numerical/exact/legacy profiles as npz per row")
    args = parser.parse_args(argv)

    payload = build_payload(args.rows, profile_dir=args.profiles)
    print(render(payload))
    print()
    for row in payload["rows"]:
        print(
            f"{row['row']}: max exact {row['max_percent_exact']:.4f} % / "
            f"legacy {row['max_percent_legacy']:.4f} % -> "
            f"{row['verdict_exact']} / {row['verdict_legacy']}, "
            f"flip={row['verdict_flip']}, margin {row['margin_pp_exact']:+.4f} pp, "
            f"displacement {row['instrument_displacement_pp']:.4f} pp, "
            f"reproduces recorded={row['reproduces_recorded']}"
        )
    print(f"flips: {payload['verdict_flips'] or 'none'}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())