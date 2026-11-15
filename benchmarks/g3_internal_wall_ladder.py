"""G3 closeout: shock reflecting off an *embedded* wall, resolution ladder + attribution.

`docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md` §4(a) left one gate open (honest
FAIL / WITHHELD): the ``(0, p, 0, 0)`` slip-wall flux that ``tensordg.euler_dg_2d``
puts on solid-fluid faces was *not* shown to reproduce the exact two-shock
solution (``benchmarks.euler_dg_2d_wall_bc.WallReflection``), and the residual was
attributed neither to the face treatment nor to resolution.  This driver answers
that question with two independent arms:

**Arm A -- resolution ladder (``wall_mode="solid"``).**  The wall is an internal
solid block, exactly the code path under test.  ``nx`` and the polynomial degree
sweep, everything else fixed.  Reading: the plateau error must *fall* with ``h``
(and with ``p``); a residual that sits still while ``h`` halves is a defect in
the face treatment, a residual that falls at the design order is a resolution
limit of a shock-capturing DG scheme.

**Arm B -- attribution (``wall_mode="boundary"``).**  The *same* Riemann data with
the wall moved onto the domain face (``bcs right="wall"``, no solid cells, domain
truncated at ``x_wall``), i.e. the boundary path used by every pre-existing
``WallReflection`` validation.  The two arms differ only in *where the wall is
implemented*.  Since both are integer-counted against the same analytic solution,
comparing their plateau errors separates "embedded-wall flux is wrong" from "DG
smears shocks on this mesh".  This arm needs no literature data: it is an
internal consistency check between two independent code paths.

Both arms report, per run: plateau samples (cell averages buffered away from the
reflected front and from the wall), their worst relative error, whole-domain
relative L1/L∞ of density and pressure, the observed step count, and the mass
residual.  Numbers are written to JSON so no figure or table in the report is
hand-copied.

Usage::

    python -m benchmarks.g3_internal_wall_ladder                 # full ladder
    python -m benchmarks.g3_internal_wall_ladder --index 2 --shards 8   # one shard
    python -m benchmarks.g3_internal_wall_ladder --smoke          # ~1 min, tiny meshes
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.forward_facing_step import gate_wall_reflection

OUT = Path("validation/g3_internal_wall")

# (name, kwargs) -- the ladder.  `x_shock` is kept at the documented 0.4 so the
# pre-reflection march is identical across rows; `extra_time` sets the plateau
# width in units of the reflected-shock travel, so it must grow slightly as the
# mesh coarsens (a fixed 0.08 leaves < 3 samples at nx=60).
CONFIGS = [
    ("solid_p1_nx60", dict(ncells=(60, 1), degree=1, limiter=None, extra_time=0.20)),
    ("solid_p1_nx120", dict(ncells=(120, 1), degree=1, limiter=None, extra_time=0.12)),
    ("solid_p1_nx240", dict(ncells=(240, 1), degree=1, limiter=None, extra_time=0.08)),
    ("solid_p2_nx120", dict(ncells=(120, 1), degree=2, limiter=None, extra_time=0.12)),
    ("solid_p2_nx240", dict(ncells=(240, 1), degree=2, limiter=None, extra_time=0.08)),
    ("solid_p1_nx240_tvb", dict(ncells=(240, 1), degree=1, limiter="tvb",
                                tvb_constant=1.0, extra_time=0.08)),
    ("solid_p2_nx240_tvb", dict(ncells=(240, 1), degree=2, limiter="tvb",
                                tvb_constant=1.0, extra_time=0.08)),
    ("boundary_p1_nx240", dict(ncells=(240, 1), degree=1, limiter=None,
                               extra_time=0.08, wall_mode="boundary")),
    ("boundary_p2_nx240", dict(ncells=(240, 1), degree=2, limiter=None,
                               extra_time=0.08, wall_mode="boundary")),
    ("boundary_p1_nx120", dict(ncells=(120, 1), degree=1, limiter=None,
                               extra_time=0.12, wall_mode="boundary")),
    ("solid_p1_nx480", dict(ncells=(480, 1), degree=1, limiter=None, extra_time=0.06)),
    ("solid_p2_nx480", dict(ncells=(480, 1), degree=2, limiter=None, extra_time=0.06)),
]

# A cheap subset that still exercises both arms (used by --smoke and by tests).
SMOKE = ["solid_p1_nx60", "boundary_p1_nx60"]


def _smoke_configs():
    """Coarse versions of the first solid row and of its boundary twin."""
    return [
        ("solid_p1_nx60", dict(ncells=(60, 1), degree=1, limiter=None, extra_time=0.20)),
        ("boundary_p1_nx60", dict(ncells=(60, 1), degree=1, limiter=None,
                                  extra_time=0.20, wall_mode="boundary")),
    ]


def run_one(name, kwargs, log=print):
    started = time.time()
    row = gate_wall_reflection(**kwargs)
    row["elapsed_seconds"] = time.time() - started
    row["name"] = name
    row["mode"] = kwargs.get("wall_mode", "solid")
    log(
        "[{name}] dx={dx:.5f} p={degree} mode={mode} steps={steps} "
        "plateau={n} rho_err={re}% p_err={pe}% worst={w}% L1(rho)={l1:.4%} "
        "status={status} pass={pass_}".format(
            name=name, dx=row["dx"], degree=row["degree"], mode=row["mode"],
            steps=row["steps"], n=row["plateau_cells"],
            re=_fmt(row["plateau_mean_rho_error_percent"]),
            pe=_fmt(row["plateau_mean_p_error_percent"]),
            w=_fmt(row["worst_relative_error_percent"]),
            l1=row["l1_relative"]["density"], status=row["status"],
            pass_=row["pass"],
        )
    )
    return row


def _fmt(value):
    return "n/a" if value is None else f"{value:+.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=None, help="shard index (0-based)")
    ap.add_argument("--shards", type=int, default=None, help="total shards")
    ap.add_argument("--smoke", action="store_true", help="coarse two-run subset")
    ap.add_argument("--out", default=str(OUT / "ladder.json"))
    args = ap.parse_args(argv)

    if args.smoke:
        configs, out = _smoke_configs(), Path(args.out).with_name("ladder_smoke.json")
    else:
        configs, out = CONFIGS, Path(args.out)
    if args.index is not None and args.shards:
        configs = [c for i, c in enumerate(configs) if i % args.shards == args.index]

    rows = [run_one(name, kwargs) for name, kwargs in configs]
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.index is None else f".shard{args.index}of{args.shards}"
    path = out.with_name(out.stem + suffix + out.suffix)
    path.write_text(json.dumps({"configs": [n for n, _ in configs], "rows": rows}, indent=1))
    print(f"wrote {path} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":  # pragma: no cover
    main()