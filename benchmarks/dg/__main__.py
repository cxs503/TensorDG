"""Package entry point for the DG unified verification surface.

    python -m benchmarks.dg --quick      # CI configuration: run the five criteria
                                         # through the registry (~1 min, same limits)
    python -m benchmarks.dg              # report configuration: full studies, figures,
                                         # report.md, benchmark table and evidence bundle

Both modes call the *same* production modules (:mod:`tensordg.advection_diffusion`,
:mod:`tensordg.dispersive`), so the CI smoke run and the reported numbers share one
execution path.  Exit status is 0 only when every criterion passes; the full mode
additionally fails when the evidence bundle cannot be re-verified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import benchmarks.dg as surface  # noqa: E402
from benchmarks.dg import evidence as evidence_mod  # noqa: E402


def quick_run(outdir: str, mission: int) -> int:
    """Run E1--E5 through the registry and serialise the criterion table."""
    os.makedirs(outdir, exist_ok=True)
    result = surface.run_surface(mode="quick")
    table = surface.benchmark_table(result["criteria"])
    payload = {
        "schema": "tensordg-dg-surface-quick/1",
        "mission_id": mission,
        "mode": "quick",
        "apriori_limits": surface.TOLERANCES,
        "criteria": result["criteria"],
        "benchmark": table,
        "verdict": result["verdict"],
        "wall_seconds": result["wall_seconds"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = os.path.join(outdir, "dg_surface_quick.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    for cid, row in table.items():
        print(f"{cid} got={row['got']:.4e} limit={row['tolerance']:.2e} "
              f"pass={row['pass']}", flush=True)
    print(f"verdict: {result['verdict']}  ({result['wall_seconds']:.1f} s)  -> {path}",
          flush=True)
    return 0 if result["verdict"] == "PASS" else 1


def _render(outdir: str, mission: int, data: dict) -> int:
    """report.md -> evidence bundle -> offline re-verification (from a run artifact)."""
    from benchmarks.dg import make_report

    os.makedirs(outdir, exist_ok=True)
    title, summary, markdown = make_report.build(data)
    with open(os.path.join(outdir, "report.md"), "w") as handle:
        handle.write(markdown)
    print(f"wrote {os.path.join(outdir, 'report.md')}", flush=True)

    evidence = evidence_mod.build_evidence(outdir, mission=mission)
    problems = evidence_mod.verify(os.path.join(outdir, evidence_mod.EVIDENCE_JSON))
    for problem in problems:
        print("EVIDENCE VIOLATION", problem, flush=True)

    gates = {cid: (row["got"], row["tolerance"], row["pass"])
             for cid, row in evidence["benchmark"].items()}
    print("gates (got, limit, pass):", json.dumps(gates, indent=2), flush=True)
    ok = all(g[2] for g in gates.values()) and not problems
    print(f"artifacts: {outdir}", flush=True)
    print("verdict:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


def full_run(outdir: str, mission: int) -> int:
    """Full studies -> report.md -> evidence bundle -> offline re-verification."""
    from benchmarks.dg import unified_verification

    os.makedirs(outdir, exist_ok=True)
    data = unified_verification.main(["--outdir", outdir])
    return _render(outdir, mission, data)


def render_only(outdir: str, mission: int) -> int:
    """Re-derive report.md + evidence from an existing run artifact (no re-solve)."""
    path = os.path.join(outdir, evidence_mod.RUN_JSON)
    if not os.path.isfile(path):
        print(f"missing run artifact: {path}", file=sys.stderr)
        return 2
    with open(path) as handle:
        data = json.load(handle)
    return _render(outdir, mission, data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="CI configuration: five criteria through the registry")
    parser.add_argument("--outdir", default=None,
                        help="output directory (default: the canonical artifact "
                             "directory for the full run, /tmp/dg-unified-quick for --quick)")
    parser.add_argument("--mission", type=int, default=14548)
    parser.add_argument("--render-only", action="store_true",
                        help="re-derive report.md + evidence bundle from the existing "
                             "run artifact instead of solving again")
    args = parser.parse_args(argv)

    if args.quick:
        outdir = os.path.abspath(args.outdir or "/tmp/dg-unified-quick")
        return quick_run(outdir, args.mission)
    outdir = os.path.abspath(
        args.outdir or os.path.join(ROOT, "docs", "benchmarks", "results",
                                    "dg-unified-verification"))
    if args.render_only:
        return render_only(outdir, args.mission)
    return full_run(outdir, args.mission)


if __name__ == "__main__":
    sys.exit(main())