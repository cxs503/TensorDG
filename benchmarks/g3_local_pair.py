"""Local (short-march) pair of the G3 arms: embedded wall vs boundary wall.

The full ladder (``benchmarks.g3_internal_wall_ladder``) needs ~10^3-10^4 RK steps
per run and belongs on the cluster.  This driver runs the *same two arms* with the
incident shock started closer to the wall (``x_shock=0.85`` instead of 0.4) and a
platue-width-matched ``extra_time``, which cuts the pre-reflection march by ~4x
and fits the control-plane time budget on the 60-cell mesh.

Everything else -- the analytic reference, the snapping of the wall onto a face,
the buffered plateau band, the CFL, the numerics -- is the shared gate code in
``benchmarks.forward_facing_step.gate_wall_reflection``, so the short-march rows
are directly comparable with the cluster ladder rows (same scorer, same 3 %
criterion).  Rows are merged into ``validation/g3_internal_wall/ladder.json``.

Usage::

    python -m benchmarks.g3_local_pair --arm solid
    python -m benchmarks.g3_local_pair --arm boundary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.forward_facing_step import gate_wall_reflection

OUT = Path("validation/g3_internal_wall/ladder.json")

ARMS = {
    # nx=60, wall face x=1.0, shock starts at 0.85 -> ~1/4 of the standard march
    "solid": dict(ncells=(32, 1), degree=1, limiter=None, x_shock=0.9,
                  extra_time=0.30, wall_mode="solid"),
    "boundary": dict(ncells=(32, 1), degree=1, limiter=None, x_shock=0.9,
                     extra_time=0.24, wall_mode="boundary"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    kwargs = ARMS[args.arm]
    name = args.name or f"shortmarch_{args.arm}_p1_nx60"
    row = gate_wall_reflection(**kwargs)
    row["name"] = name
    row["mode"] = row["wall_mode"]
    print(f"[{name}] steps={row['steps']} plateau={row['plateau_cells']} "
          f"rho_err={row['plateau_mean_rho_error_percent']} "
          f"p_err={row['plateau_mean_p_error_percent']} "
          f"worst={row['worst_relative_error_percent']} status={row['status']} "
          f"l1={row['l1_relative']}")

    rows = []
    if OUT.exists():
        data = json.loads(OUT.read_text())
        rows = [r for r in data.get("rows", []) if r["name"] != name]
    rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"configs": [r["name"] for r in rows], "rows": rows}, indent=1))
    print(f"merged into {OUT} ({len(rows)} rows)")


if __name__ == "__main__":  # pragma: no cover
    main()