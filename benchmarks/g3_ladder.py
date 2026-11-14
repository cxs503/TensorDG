"""G3 resolution ladder: does an interior-wall shock reflection converge to <3 %?

Anchor: ``REPRO-DG-CKS1998`` (Cockburn & Shu, JCP 141 (1998) 199-224, sec. 5).
The forward-facing-step capability (mesh-embedded solid obstacles) is merged; its
G3 gate -- a normal shock reflecting off an *embedded* wall against the exact
two-shock solution -- was recorded as an honest FAIL without an attribution:
"resolution or interior-wall modelling?" was left open (see
``docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md`` sec. 4a).

This driver turns that open question into a measurement: one shard per
``(ncells, degree, limiter)`` configuration, then a collected table + figure.

    python -m benchmarks.g3_ladder --shard 0 --shards 8     # one job per shard
    python -m benchmarks.g3_ladder --collect               # table + PNG + JSON
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

from benchmarks.forward_facing_step import gate_wall_reflection

OUTDIR = Path("validation/cks1998_g3_ladder")

#: cpu cost is ~0.1 s per step and steps ~ 1/dx, so grids stop at 480 cells.
CONFIGS = [
    {"tag": f"nx{nx}_p{deg}_none", "ncells": (nx, 1), "degree": deg, "limiter": None}
    for deg in (1, 2)
    for nx in (60, 120, 240, 480)
] + [
    {"tag": "nx240_p2_tvb", "ncells": (240, 1), "degree": 2, "limiter": "tvb",
     "tvb_constant": 1.0},
    {"tag": "nx240_p1_tvb", "ncells": (240, 1), "degree": 1, "limiter": "tvb",
     "tvb_constant": 1.0},
]


def run_shard(index, shards, outdir=OUTDIR):
    outdir.mkdir(parents=True, exist_ok=True)
    for k, config in enumerate(CONFIGS):
        if k % shards != index:
            continue
        kwargs = {key: value for key, value in config.items() if key != "tag"}
        print(f"[shard {index}] {config['tag']} ...", flush=True)
        row = gate_wall_reflection(**kwargs)
        row["tag"] = config["tag"]
        row["environment"] = {"python": platform.python_version(), "platform": platform.platform()}
        path = outdir / f"{config['tag']}.json"
        path.write_text(json.dumps(row, indent=2))
        print(f"[shard {index}] {config['tag']}: plateau_cells={row['plateau_cells']} "
              f"worst={row['worst_relative_error_percent']} status={row['status']} "
              f"pass={row['pass']} l1={row['l1_relative']}", flush=True)
    return 0


def _load(outdir=OUTDIR):
    rows = []
    for path in sorted(outdir.glob("nx*_p*_*.json")):
        rows.append(json.loads(path.read_text()))
    rows.sort(key=lambda r: (r["degree"], r["ncells"][0], str(r["limiter"])))
    return rows


def collect(outdir=OUTDIR):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load(outdir)
    if not rows:
        raise SystemExit(f"no rows in {outdir}; run the shards first")
    (outdir / "ladder.json").write_text(json.dumps({"rows": rows}, indent=2))

    header = ("tag", "dx", "plateau_cells", "worst_%", "mean_rho_%", "mean_p_%",
              "l1_rho", "l1_p", "linf_rho", "status", "pass", "steps")
    print(" | ".join(header))
    table = []
    for row in rows:
        fmt = lambda value, spec=".4g": "-" if value is None else format(value, spec)
        line = [row["tag"], fmt(row["dx"]), str(row["plateau_cells"]),
                fmt(row["worst_relative_error_percent"], ".3f"),
                fmt(row["plateau_mean_rho_error_percent"], ".3f"),
                fmt(row["plateau_mean_p_error_percent"], ".3f"),
                fmt(row["l1_relative"]["density"]), fmt(row["l1_relative"]["pressure"]),
                fmt(row["linf_relative"]["density"]), row["status"], str(row["pass"]),
                str(row["steps"])]
        print(" | ".join(line))
        table.append(line)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for degree in (1, 2):
        series = [r for r in rows if r["degree"] == degree and r["limiter"] is None]
        if not series:
            continue
        dx = [r["dx"] for r in series]
        worst = [abs(r["worst_relative_error_percent"]) if r["worst_relative_error_percent"] is not None
                 else math.nan for r in series]
        l1 = [r["l1_relative"]["density"] * 100.0 for r in series]
        axes[0].loglog(dx, worst, "o-", label=f"P{degree} plateau worst")
        axes[1].loglog(dx, l1, "s-", label=f"P{degree} L1 density")
    axes[0].axhline(3.0, color="k", ls="--", lw=1.0, label="3 % gate")
    axes[0].set_xlabel("dx"); axes[0].set_ylabel("worst plateau error [%]")
    axes[0].set_title("reflected plateau vs exact two-shock solution")
    axes[0].legend(fontsize=8); axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("dx"); axes[1].set_ylabel("L1(density) [%]")
    axes[1].set_title("whole-domain L1 against the exact solution")
    axes[1].legend(fontsize=8); axes[1].grid(True, which="both", alpha=0.3)
    fig.suptitle("CKS1998 interior-wall shock reflection (G3) — mesh ladder")
    fig.tight_layout()
    fig.savefig(outdir / "g3_ladder.png", dpi=150)
    print(f"wrote {outdir/'ladder.json'} and {outdir/'g3_ladder.png'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=len(CONFIGS))
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args(argv)
    if args.collect:
        return collect(args.outdir)
    index = 0 if args.shard is None else args.shard
    # accept both 0-based and 1-based shard numbering from the scheduler
    return run_shard(index % args.shards, args.shards, args.outdir)


if __name__ == "__main__":
    sys.exit(main())