"""Resolution ladder for G3 of ``benchmarks.forward_facing_step``.

Drives ``validation/forward_step/wall_reflection_diag.py`` over a config matrix
and writes one JSON per task, so the ladder can be sharded across cluster jobs:

    python validation/forward_step/wall_reflection_ladder.py --task 3

Why a ladder: ``docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md`` §4(a) records the
embedded-wall reflection gate as an honest FAIL whose *physical attribution*
(embedded-wall flux vs. near-wall resolution / limiter) was never measured.
A single-mesh FAIL cannot tell those apart; the error-vs-mesh curve can.

Geometry note: this ladder places the last **fluid** cell so that it ends
exactly on the analytic wall face.  ``benchmarks/forward_facing_step.py``
originally masked ``solid[xc >= x_wall - dx/2]``, which starts the solid block
one cell early and therefore puts the discrete wall a full ``dx`` short of the
analytic wall at *every* resolution -- a reference-geometry error that the error
ladder cannot absorb (it is O(1) in cells, not O(dx) in the error norm).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.forward_step import wall_reflection_diag as diag  # noqa: E402

OUTDIR = Path(__file__).resolve().parent / "ladder"

# (route, nx, degree, limiter, tvb_constant, extra_time)
TASKS = [
    ("embedded", 60, 1, "tvb", 1.0, 0.08),
    ("embedded", 120, 1, "tvb", 1.0, 0.08),
    ("embedded", 240, 1, "tvb", 1.0, 0.08),
    ("embedded", 480, 1, "tvb", 1.0, 0.08),
    ("embedded", 120, 2, "tvb", 1.0, 0.08),
    ("embedded", 240, 2, "tvb", 1.0, 0.08),
    ("embedded", 480, 2, "tvb", 1.0, 0.08),
    ("embedded", 240, 2, "tvb", 0.0, 0.08),   # strict TVD (M = 0)
    ("embedded", 240, 2, None, 1.0, 0.08),    # no limiter: stability probe
    ("outer", 240, 2, "tvb", 1.0, 0.08),      # cross-route check
]


def _resolve(i: int) -> int:
    for cand in (i, i - 1):
        if 0 <= cand < len(TASKS):
            return cand
    raise SystemExit(f"task index {i} out of range (0..{len(TASKS) - 1})")


def run_task(idx: int) -> dict:
    route, nx, degree, limiter, tvb, extra = TASKS[idx]
    ncells = (nx, 1)
    kw = dict(degree=degree, limiter=limiter, tvb=tvb, extra_time=extra)
    if route == "embedded":
        run = diag.run_embedded(ncells, **kw)
    else:
        run = diag.run_outer(ncells, **kw)
    wr = diag.WallReflection(
        x_shock=diag._snap(diag.X_SHOCK, diag.XMAX / nx),
        x_wall=diag._snap(diag.X_WALL, diag.XMAX / nx),
        incident_mach=diag.MACH,
    )
    run.update(diag.plateau_stats(run, wr))
    run.update({"task": idx, "route": route, "limiter": limiter,
                "tvb_constant": tvb, "extra_time": extra, "degree": degree})
    # third route: the 1-D solver at the same effective resolution
    return run


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    task = 0
    for k, tok in enumerate(argv):
        if tok == "--task":
            task = int(argv[k + 1])
    idx = _resolve(task)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = run_task(idx)
    path = OUTDIR / f"task{idx:02d}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "profile"}, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())