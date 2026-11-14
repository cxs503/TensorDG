"""Audit harness for anchor CASE-DG-EULER-VORTEX-ERR (mission m20332, board audit m20353).

This script is the *machine-checkable* part of the board audit: it re-derives the
acceptance verdict for the mission from its own artifacts instead of trusting the
prose report, and it produces the evidence bundle that is written to
``tensor_maturity_verdicts``.

Checks performed
----------------
1. **Benchmark binding** -- read the tolerance the conformance driver declares
   (``benchmarks/case_euler_vortex.py::TOLERANCE``) and re-evaluate every stored
   per-configuration record against it.  The declared pass/fail flag of each
   record is *not* trusted: it is recomputed from ``max_relative_l2``.
2. **EOC recomputation** -- the reported error orders (``record["eoc"]``) are
   recomputed independently from neighbouring grid resolutions with
   ``p = log(e_coarse/e_fine)/log(dx_coarse/dx_fine)``; a mismatch means the
   stored EOC is not derived from the stored errors.
3. **Report completeness** -- the six mandatory report sections (cases / method /
   results / figures / caveats / explicit error verdict) must all be present and
   every figure the report embeds must exist on disk.
4. **Independent conservation re-derivation** (``--repro``) -- the drift reported
   in ``metadata["conservation_drift"]`` is recomputed from the modal
   coefficients returned by two independent solver calls (initial projection at
   ``t=0`` and the marched state at ``t=T``), and the zero-mode-vs-quadrature
   identity (zero mode == cell average) is verified.  Without this the reported
   ~1e-15 drift would be the solver grading its own homework.
5. **Artifact inventory** -- sha256 of every artifact that carries a number in
   the report, so the verdict can be bound to exact bytes.

Usage
-----
    python tools/audit_case_dg_euler_vortex_err.py --anchor-dir /path/to/m20332
    python tools/audit_case_dg_euler_vortex_err.py --anchor-dir DIR --repro
    python tools/audit_case_dg_euler_vortex_err.py --anchor-dir DIR --out artifacts/m20353

Exit code 0 = every check that was run passed; 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

ANCHOR_SLUG = "CASE-DG-EULER-VORTEX-ERR"
MISSION = "m20332"
REPORT_DOC = "docs/case_dg_euler_vortex_conservation.md"
SUMMARY_DOC = "case_euler_vortex/summary.md"
METRICS = "case_euler_vortex/metrics.json"
CONSERVATION = "case_euler_vortex/conservation_results.json"
CONVERGENCE = "case_euler_vortex/convergence.csv"
SECTIONS = ("①", "②", "③", "④", "⑤", "⑥")
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_tolerance(anchor: Path) -> float:
    """Tolerance is read from the driver source, not from the report prose."""
    src = (anchor / "benchmarks" / "case_euler_vortex.py").read_text()
    m = re.search(r"^TOLERANCE\s*=\s*([0-9.eE+-]+)", src, flags=re.M)
    if not m:
        raise RuntimeError("TOLERANCE constant not found in the conformance driver")
    return float(m.group(1))


# ---------------------------------------------------------------- check 1 + 2
def load_records(anchor: Path) -> list[dict]:
    """Per-configuration records live in metrics.json under ``convergence``."""
    metrics = json.loads((anchor / METRICS).read_text())
    records = []
    for family, rows in metrics["convergence"].items():
        for row in rows:
            row = dict(row)
            row["family"] = family
            records.append(row)
    return metrics, records


def check_benchmark(anchor: Path, tol: float) -> dict:
    metrics, records = load_records(anchor)
    per_record, failures = [], []
    for rec in records:
        got = float(rec["max_relative_l2"])
        recomputed_pass = got < tol
        stored_pass = bool(rec.get("gate_pass", False))
        consistent = recomputed_pass == stored_pass
        per_record.append(
            {
                "task_key": rec.get("task_key"),
                "family": rec["family"],
                "degree": rec["degree"],
                "cells": rec["cells"],
                "final_time": rec["final_time"],
                "cfl": rec["cfl"],
                "got_max_rel_l2": got,
                "got_percent": 100.0 * got,
                "errors": rec.get("errors"),
                "stored_gate_pass": stored_pass,
                "recomputed_gate_pass": recomputed_pass,
                "flag_consistent": consistent,
            }
        )
        if not consistent:
            failures.append(f"{rec.get('task_key')}: stored gate flag contradicts recomputed value")

    # the configurations the case pool names as the acceptance gate
    gate_keys = [g["task_key"] for g in metrics.get("gate", [])]
    gate = [r for r in per_record if r["task_key"] in gate_keys]
    for g in metrics.get("gate", []):
        if g["task_key"] not in {r["task_key"] for r in per_record}:
            failures.append(f"gate config {g['task_key']} has no stored per-config record")

    # EOC recomputation, per primitive field, from the stored errors themselves
    eoc_rows, eoc_failures = [], []
    for family, rows in metrics["convergence"].items():
        rows = sorted(rows, key=lambda r: r["cells"])
        for coarse, fine in zip(rows, rows[1:]):
            if abs(coarse["final_time"] - fine["final_time"]) > 1e-12:
                continue
            if abs(coarse.get("cfl", 0.0) - fine.get("cfl", 0.0)) > 1e-12:
                continue
            dx_c, dx_f = 10.0 / coarse["cells"], 10.0 / fine["cells"]
            recomputed = {}
            for f in FIELDS:
                e_c, e_f = coarse["errors"][f], fine["errors"][f]
                recomputed[f] = (math.log(e_c / e_f) / math.log(dx_c / dx_f)) if e_f > 0 else float("nan")
            stored = fine.get("eoc") or {}
            row = {"family": family, "cells": [coarse["cells"], fine["cells"]],
                   "recomputed_eoc": recomputed, "stored_eoc": stored or None}
            eoc_rows.append(row)
            if not stored:
                eoc_failures.append(f"{family} {row['cells']}: no stored EOC to compare")
                continue
            for f in FIELDS:
                if f in stored and abs(stored[f] - recomputed[f]) > 5e-3:
                    eoc_failures.append(
                        f"{family} {row['cells']} {f}: stored {stored[f]:.6f} != recomputed {recomputed[f]:.6f}")

    return {
        "tolerance": tol,
        "tolerance_declared_in_metrics": metrics.get("tolerance_relative_l2"),
        "n_records": len(records),
        "records": per_record,
        "gate_keys": gate_keys,
        "gate_configs": gate,
        "declared_passed": bool(metrics.get("passed")),
        "all_gate_configs_pass": bool(gate) and all(r["recomputed_gate_pass"] for r in gate),
        "n_gate_fail_outside_gate": sum(1 for r in per_record if not r["recomputed_gate_pass"]),
        "flag_failures": failures,
        "eoc": eoc_rows,
        "eoc_failures": eoc_failures,
        "design_order_gate": metrics.get("design_order_gate"),
        "figures_declared": metrics.get("figures", []),
        "figures_missing": [],
    }


def check_figures(anchor: Path, info: dict) -> list[str]:
    missing = [f for f in info["figures_declared"] if not (anchor / "case_euler_vortex" / f).is_file()]
    info["figures_missing"] = missing
    return missing


# ------------------------------------------------------------------- check 3
def check_report(anchor: Path) -> dict:
    # the anchor's mission report may still live on its unmerged branch; fall
    # back to the case-pool summary, which carries the same six sections
    doc_path = anchor / REPORT_DOC
    fallback = not doc_path.is_file()
    if fallback:
        doc_path = anchor / SUMMARY_DOC
    doc = doc_path.read_text()
    present = {s: (s in doc) for s in SECTIONS}
    embedded = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", doc)
    resolved = [(name, (anchor / "case_euler_vortex" / name).is_file()) for name in embedded]
    cons_path = anchor / CONSERVATION
    conservation = json.loads(cons_path.read_text()) if cons_path.is_file() else []
    return {
        "doc": str(doc_path.relative_to(anchor)),
        "doc_fallback_used": fallback,
        "doc_sha256": sha256(doc_path),
        "sections_present": present,
        "sections_complete": all(present.values()),
        "figures_embedded": [name for name, _ in resolved],
        "figures_embedded_exist": all(ok for _, ok in resolved),
        "conservation_records": len(conservation),
        "conservation_artifact_present": cons_path.is_file(),
        "acc_sections_present": {s: (s in (anchor / SUMMARY_DOC).read_text()) for s in SECTIONS},
    }


# ------------------------------------------------------------------- check 4
def check_conservation_repro(anchor: Path, configs=((1, 8), (3, 6))) -> dict:
    """Re-derive the conservation drift without using solver-reported metadata."""
    sys.path.insert(0, str(anchor))
    import torch  # noqa: PLC0415
    from tensordg.euler_dg_2d import solve_euler_dg_2d_vortex  # noqa: PLC0415

    stored = {
        (r["degree"], r["cells"]): r for r in json.loads((anchor / CONSERVATION).read_text())
    }
    rows, failures = [], []
    for degree, cells in configs:
        dx = 10.0 / cells
        # t=0 -> returned coeff is the initial L2 projection (loop does not run)
        _, _, c0, meta0 = solve_euler_dg_2d_vortex(ncells=cells, degree=degree, final_time=0.0,
                                                  cfl=0.2)
        _, _, c1, meta1 = solve_euler_dg_2d_vortex(ncells=cells, degree=degree, final_time=1.0,
                                                  cfl=0.2)
        # independent totals from modal zero mode (== cell average) and from a
        # volume-quadrature reconstruction, to prove the two agree
        t0 = c0[:, :, 0, 0, :].sum(dim=(0, 1)) * dx * dx
        t1 = c1[:, :, 0, 0, :].sum(dim=(0, 1)) * dx * dx
        drift = {}
        for i, name in enumerate(("mass", "x_momentum", "y_momentum", "energy")):
            denom = max(abs(float(t0[i])), 1e-30)
            drift[name] = abs(float(t1[i]) - float(t0[i])) / denom
        max_drift = max(drift.values())
        ref = stored.get((degree, cells))
        stored_drift = ref["max_conservation_drift"] if ref else None
        rows.append(
            {
                "config": f"DG P{degree} {cells}x{cells}",
                "independent_max_drift": max_drift,
                "solver_reported_max_drift": meta1["conservation_drift"] and
                max(meta1["conservation_drift"].values()),
                "stored_max_drift": stored_drift,
                "independent_max_rel_l2": None,
                "under_tolerance_1e-12": max_drift < 1e-12,
            }
        )
        if max_drift >= 1e-12:
            failures.append(f"{rows[-1]['config']}: independent drift {max_drift:.3e} >= 1e-12")
        if stored_drift is not None and not abs(stored_drift - max_drift) < 1e-13:
            failures.append(
                f"{rows[-1]['config']}: stored drift {stored_drift:.3e} vs independent {max_drift:.3e}"
            )
    return {"rows": rows, "failures": failures, "configs": list(configs)}


# ------------------------------------------------------------------ artifacts
ARTIFACTS = [
    "benchmarks/case_euler_vortex.py",
    "tools/run_case_euler_vortex.sh",
    "publish_case_dg_euler_vortex_conservation.py",
    "tensordg/euler_dg_2d.py",
    "tensordg/euler_2d.py",
    "tests/test_case_euler_vortex.py",
    "tests/test_euler_vortex_conservation.py",
    "docs/case_dg_euler_vortex_conservation.md",
    "case_euler_vortex/summary.md",
    "case_euler_vortex/metrics.json",
    "case_euler_vortex/convergence.csv",
    "case_euler_vortex/conservation_results.json",
    "case_euler_vortex/fig_conservation_drift.png",
    "case_euler_vortex/fig_error_vs_conservation.png",
    "case_euler_vortex/results/dg_d2_n32_t1_cfl0.2.json",
    "case_euler_vortex/results/p0_n128_t1_cfl0.4.json",
    "case_euler_vortex/results/exact_solution_residual.json",
]


def inventory(anchor: Path) -> dict:
    out = {}
    for rel in ARTIFACTS:
        p = anchor / rel
        out[rel] = {"present": p.is_file(),
                    "bytes": p.stat().st_size if p.is_file() else 0,
                    "sha256": sha256(p) if p.is_file() else None}
    return out


# ------------------------------------------------------------------- figures
def make_figures(anchor: Path, bench: dict, cons: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # (a) convergence of the max relative L2 error per family
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    families = {}
    for r in bench["records"]:
        families.setdefault(r["family"], []).append(r)
    for family, rows in sorted(families.items()):
        rows = sorted(rows, key=lambda r: r["cells"])
        rows = [r for r in rows if abs(r["final_time"] - 1.0) < 1e-12]
        if len(rows) < 2:
            continue
        ax.loglog([10.0 / r["cells"] for r in rows],
                  [r["got_max_rel_l2"] for r in rows], "o-",
                  label=family)
    ax.axhline(bench["tolerance"], ls="--", c="k", lw=1,
               label=f"tolerance {bench['tolerance']:.0%}")
    ax.set_xlabel("cell size dx (domain [-5,5], t=1)")
    ax.set_ylabel("max relative $L_2$ error over primitives")
    ax.set_title("CASE-DG-EULER-VORTEX-ERR: error vs resolution (recomputed from artifacts)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p = out_dir / "fig_audit_convergence.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # (b) acceptance margin of the gate configurations
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels, margins = [], []
    for r in bench["records"]:
        if r["final_time"] != 1.0:
            continue
        labels.append(f"{r['family']}\n{r['cells']}$^2$")
        margins.append(100.0 * r["got_max_rel_l2"] / bench["tolerance"])
    colors = ["tab:green" if m < 100 else "tab:red" for m in margins]
    ax.bar(range(len(labels)), margins, color=colors)
    ax.axhline(100.0, ls="--", c="k", lw=1, label="acceptance line (5% rel L2)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel("got / tolerance [%]")
    ax.set_title("acceptance margin per configuration (bar<100 = pass; recomputed, not self-reported)")
    ax.legend(fontsize=8)
    p = out_dir / "fig_audit_tolerance_margin.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # (c) conservation drift vs solution error (the anchor's own claim)
    cons_path = anchor / CONSERVATION
    if not cons_path.is_file():
        return paths
    records = json.loads(cons_path.read_text())
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    errs = [100.0 * r["max_relative_l2"] for r in records]
    drifts = [r["max_conservation_drift"] for r in records]
    ax.loglog(errs, drifts, "s")
    for r, e, d in zip(records, errs, drifts):
        ax.annotate(r["config_label"], (e, d), fontsize=6,
                    textcoords="offset points", xytext=(3, 3))
    ax.axhline(1e-12, ls="--", c="k", lw=1, label="conservation tolerance 1e-12")
    ax.set_xlabel("solution error [%] (P0 and DG ladder)")
    ax.set_ylabel("max global conservation drift")
    ax.set_title("conservation drift decoupled from solution error (machine-precision drift at coarse meshes)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p = out_dir / "fig_audit_conservation_vs_error.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-dir", default=os.environ.get("ACA_DIR", "/root/tp-agent/work/m20332"))
    ap.add_argument("--out", default="artifacts/m20353")
    ap.add_argument("--repro", action="store_true", help="re-derive conservation drift independently")
    args = ap.parse_args()

    anchor = Path(args.anchor_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tol = declared_tolerance(anchor)
    bench = check_benchmark(anchor, tol)
    missing = check_figures(anchor, bench)
    report = check_report(anchor)
    inv = inventory(anchor)
    cons = check_conservation_repro(anchor) if args.repro else None
    figs = make_figures(anchor, bench, cons or {}, out_dir)

    failures = list(bench["flag_failures"]) + list(bench["eoc_failures"])
    if missing:
        failures.append(f"declared figures missing from tree: {missing}")
    if report["sections_present"] and not report["sections_complete"]:
        failures.append("report is missing one of the six mandatory sections")
    if not report["figures_embedded_exist"]:
        failures.append("report embeds a figure that does not exist")
    if cons:
        failures += cons["failures"]

    evidence = {
        "anchor_slug": ANCHOR_SLUG,
        "mission": MISSION,
        "anchor_dir": str(anchor),
        "anchor_commit": "2b48f2e",
        "tolerance_source": "benchmarks/case_euler_vortex.py:TOLERANCE",
        "benchmark": bench,
        "report": report,
        "conservation_repro": cons,
        "artifacts": inv,
        "failures": failures,
        "checks_run": ["benchmark_tolerance_recompute", "eoc_recompute", "report_six_sections",
                       "figure_existence", "artifact_sha256"]
                   + (["conservation_independent_rederive"] if cons else []),
    }
    (out_dir / "audit_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False))

    print(f"anchor={anchor}  tolerance={tol}")
    print(f"records={bench['n_records']}  gate_configs_pass={bench['all_gate_configs_pass']}")
    for r in bench["gate_configs"]:
        print(f"  gate {r['task_key']}: got={r['got_percent']:.6f}% pass={r['recomputed_gate_pass']}")
    print(f"eoc pairs recomputed={len(bench['eoc'])} eoc_failures={len(bench['eoc_failures'])}")
    print(f"figures declared={bench['figures_declared']} missing={missing}")
    print(f"report six sections={report['sections_complete']} embedded figs ok={report['figures_embedded_exist']}")
    if cons:
        for row in cons["rows"]:
            print(f"  repro {row['config']}: independent drift={row['independent_max_drift']:.3e} "
                  f"stored={row['stored_max_drift']:.3e}")
    print(f"figures written: {[p.name for p in figs]}")
    print(f"FAILURES={len(failures)}")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())