"""Projection-error baseline for CASE-DG-EULER-VORTEX-ERR.

The case claims *error conservation*: over the periodic box the exact solution is
a pure translation, so the discretisation error should be a property of the mesh /
flux pair rather than a function of the elapsed transport time.  ``C2`` measures
the growth of ``e(t)`` on a log-log fit over the ladder, which says nothing about
the *level* the growth starts from.  This script supplies that level:

``e(0)`` is the error of the exact initial state **as represented in the discrete
space** -- the L2 projection for the DG family, exact 4x4 Gauss cell averages for
the P0 family -- i.e. the irreducible error the scheme starts with before a single
time step is taken.  Dividing the transported error by it,

    A(t) = e(t) / e(0),

gives the per-box-crossing amplification of the discretisation error.  A scheme
that conserves the error keeps ``A`` of order one; a secular error-amplification
mechanism (the failure mode the case exists to detect) drives ``A`` to grow with
``t``.  ``e(0)`` costs a projection and one error evaluation per family -- no time
stepping -- so this is cheap evidence, not a second ladder.

Output: ``case_euler_vortex_errconserve/projection_baseline.json``
(``--json PATH`` overrides), consumed by ``publish_case_euler_vortex_errconserve_report.py``.
"""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.case_euler_vortex_errconserve import (
    BACKGROUND, DOMAIN, FIELDS, PERIOD, STRENGTH, TOLERANCE,
    _initial_dg_state, _initial_p0_state, _operators,
    dg_vortex_error, vortex_error,
)


def dg_baseline(degree: int = 2, cells: int = 32) -> dict:
    config = {"task": "dg", "degree": degree, "cells": cells, "final_time": 0.0,
              "cfl": 0.2, "strength": STRENGTH}
    coeff, basis, nodes, _weights = _initial_dg_state(config)
    raw = dg_vortex_error(coeff, basis, nodes, DOMAIN, 0.0, STRENGTH, BACKGROUND,
                          period=PERIOD)
    errors = {name: float(raw[name]) for name in FIELDS}
    return {"task_key": f"dg_d{degree}_n{cells}", "task": "dg", "degree": degree,
            "cells": cells, "final_time": 0.0,
            "metric": "volume-quadrature relative L2 of the L2-projected t=0 state",
            "errors": errors, "max_relative_l2": max(errors.values())}


def p0_baseline(cells: int) -> dict:
    config = {"task": "p0", "cells": cells, "final_time": 0.0, "cfl": 0.4,
              "strength": STRENGTH}
    conservative = _initial_p0_state(config)
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / cells, (ymax - ymin) / cells
    x = xmin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dy
    raw = vortex_error(x, y, conservative, 0.0, STRENGTH, BACKGROUND, period=PERIOD)
    errors = {name: float(raw[name]["relative_l2"]) for name in FIELDS}
    return {"task_key": f"p0_n{cells}", "task": "p0", "cells": cells,
            "final_time": 0.0,
            "metric": "relative L2 of the exact t=0 cell averages",
            "errors": errors, "max_relative_l2": max(errors.values())}


def amplification(baseline: dict, records: list) -> list:
    """``A(t) = e(t)/e(0)`` for every record of the matching family."""
    out = []
    for record in sorted(records, key=lambda r: r["final_time"]):
        family = record["task_key"]
        task, degree, cells, t = record["task"], record.get("degree"), record["cells"], record["final_time"]
        if task == "dg":
            key = f"dg_d{degree}_n{cells}"
        else:
            key = f"p0_n{cells}"
        if key != baseline["task_key"] or t <= 0.0:
            continue
        out.append({
            "task_key": family,
            "final_time": t,
            "error": record["max_relative_l2"],
            "amplification": record["max_relative_l2"] / baseline["max_relative_l2"],
        })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir",
                        default="case_euler_vortex_errconserve/results")
    parser.add_argument("--json",
                        default="case_euler_vortex_errconserve/projection_baseline.json")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    records = [json.loads(path.read_text())
               for path in sorted(Path(args.results_dir).glob("*.json"))]
    baselines = [dg_baseline(), p0_baseline(64), p0_baseline(128), p0_baseline(256)]
    for baseline in baselines:
        baseline["amplification"] = amplification(baseline, records)

    payload = {
        "case": "EULER-VORTEX-ERR / projection baseline",
        "ledger_slug": "CASE-DG-EULER-VORTEX-ERR",
        "definition": ("e(0) = error of the exact t=0 state in the discrete space "
                       "(DG: L2 projection; P0: exact cell averages); "
                       "A(t) = e(t)/e(0) is the per-box-crossing amplification"),
        "tolerance_relative_l2": TOLERANCE,
        "box_crossing_time": 10.0,
        "baselines": baselines,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    for baseline in baselines:
        print(f"{baseline['task_key']}: e(0) = {100 * baseline['max_relative_l2']:.6f} %")
        for row in baseline["amplification"]:
            print(f"   t={row['final_time']:>4g}  e={100 * row['error']:.6f} %  "
                  f"A={row['amplification']:.3f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())