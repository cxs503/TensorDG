"""Ingest the cluster-run digest of ``benchmarks.lax_resolution_closure``.

The compute node has no matplotlib, so the driver is run there **without**
figures and its metrics digest is printed into the job log:

    python -m benchmarks.lax_resolution_closure --skip-figures --output-dir /tmp/o \
      && python -c "import json; m=json.load(open('/tmp/o/metrics.json')); print('DIGEST'+json.dumps(...))"

This script parses that digest (so no number is ever transcribed by hand),
writes ``metrics.json`` in the standard artifact form, redraws the two figures
locally from the same numbers, and renders
``docs/CASE-LAX-DG-resolution-closure.md``.

Usage::

    python scripts/ingest_lax_closure_run.py /root/tp-agent/cluster/jobs/cjNNNN.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.lax_resolution_closure import (  # noqa: E402
    ARM_CELLS, CFL, CLOSURE_LADDER, DEGREE, DOMAIN, FINAL_TIME,
    LIMITER_ARMS, R3_BAND_WIDTH,
)
from scripts.render_lax_closure_doc import main as render_doc  # noqa: E402

OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_resolution_closure"
FIELDS = ("density", "velocity", "pressure")


def digest_from_log(path: Path) -> dict:
    """Pull the ``DIGEST{...}`` payload out of a queue job log."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        tail = record.get("output_tail") or ""
        index = tail.find("DIGEST")
        if index < 0:
            continue
        payload = tail[index + len("DIGEST"):]
        digest, _ = json.JSONDecoder().raw_decode(payload.strip())
        digest["_job"] = record.get("job_id")
        digest["_node"] = record.get("node")
        digest["_rc"] = record.get("rc")
        return digest
    raise SystemExit(f"no DIGEST payload in {path}")


def render_figures(metrics: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gate = metrics["gate"]
    tol = gate["tolerance"]
    legs = sorted(metrics["ladder"], key=lambda e: e["cells"])
    made = []

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for field, style in zip(FIELDS, ("o-", "s--", "^:")):
        ax.loglog([e["cells"] for e in legs], [e["l2"][field] for e in legs],
                  style, label=f"{field} (whole-field L2)")
    ax.axhline(tol, color="k", ls="--", lw=1.2, label=f"ledger bar {tol:.0%}")
    if gate.get("closure_mesh"):
        ax.axvline(gate["closure_mesh"], color="g", lw=1.0, alpha=0.6,
                   label=f"closure mesh N={gate['closure_mesh']}")
    ax.set_xlabel("cells N (p = 3, no limiter)")
    ax.set_ylabel("relative L2 vs exact")
    ax.set_title("CASE-LAX-DG: p3 h-refinement against the 3 % ledger bar")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = fig_dir / "closure_ladder.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption":
                 "Whole-field relative L2 of density/velocity/pressure versus cell "
                 "count on the registered p3 unlimited arm; dashed line = ledger "
                 "3 % bar, green line = the mesh at which it is met."})

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    width = 0.26
    arms = metrics["arms"]
    for k, field in enumerate(FIELDS):
        ax.bar([i + (k - 1) * width for i in range(len(arms))],
               [a["l2"][field] for a in arms], width, label=f"{field} L2")
    ax.axhline(tol, color="k", ls="--", lw=1.2, label="ledger bar")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([f"limiter={a['limiter']}" for a in arms])
    ax.set_yscale("log")
    ax.set_ylabel("relative L2 vs exact")
    ax.set_title(f"One-knob comparison at the production leg (p3, N={ARM_CELLS})")
    ax.legend(fontsize=8)
    path = fig_dir / "closure_arms.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption":
                 "Limiter arms at the production mesh p3 N=400 with the shock "
                 "limiter as the only changed knob; both limiters are worse than "
                 "leaving the discontinuities unlimited (troubled cells are "
                 "reduced to a linear reconstruction)."})
    return made


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    log = Path(argv[0])
    digest = digest_from_log(log)
    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "study": "resolution closure of ledger A0 (p3, whole-field relative L2 < 3 %)",
        "generated_at_utc": digest["generated_at_utc"],
        "config": {"degree": DEGREE, "ladder": list(CLOSURE_LADDER), "cfl": CFL,
                   "final_time": FINAL_TIME, "domain": list(DOMAIN),
                   "r3_band_width": R3_BAND_WIDTH,
                   "limiter_arms": [("none" if a is None else a) for a in LIMITER_ARMS],
                   "arm_cells": ARM_CELLS, "quick": False},
        "ladder": [dict(e, limiter="none", degree=DEGREE)
                   for e in digest["ladder"]],
        "arms": [dict(a, cells=ARM_CELLS, degree=DEGREE)
                 for a in digest["arms"]],
        "gate": digest["gate"],
        "wall_seconds": digest["wall_seconds"],
        "source": {"job": digest.get("_job"), "node": digest.get("_node"),
                   "log": str(log),
                   "note": ("driver ran with --skip-figures on the compute node "
                            "(no matplotlib there); the digest was printed to the "
                            "job log and ingested here, and the two figures were "
                            "redrawn locally from these numbers")},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    metrics["figures"] = render_figures(metrics)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    render_doc()
    summary = OUT / "summary.md"
    if not summary.exists():
        summary.write_text(
            "\n".join(["# CASE-LAX-DG resolution closure",
                       f"verdict {metrics['gate']['status']}, closure mesh "
                       f"{metrics['gate']['closure_mesh']}",
                       "see metrics.json / docs/CASE-LAX-DG-resolution-closure.md"]),
            encoding="utf-8")
    print(json.dumps({"verdict": metrics["gate"]["status"],
                      "closure_mesh": metrics["gate"]["closure_mesh"],
                      "job": metrics["source"]["job"],
                      "figures": [f["name"] for f in metrics["figures"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())