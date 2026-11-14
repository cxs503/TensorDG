#!/usr/bin/env python
"""Independent board-audit re-run of the CASE-DG-ADVECT-ORDER artifact.

Audit mission m20097 (cross-audit of m20026).  This driver is *independent* of
``tools/run_advection_porder_study.py`` (the m20026 deliverable): it drives the
library entry points directly (``tensordg.advection_1d.solve_advection`` /
``l2_error`` and ``tensordg.advection_2d_var.DGAdvection2D``) with the same
ladders that the m20026 tool declares, then compares every number against the
committed artifact copy in ``docs/board_audit_m20097/anchor_artifacts/``.

Zero figures, zero matplotlib: this script only produces numbers.

Outputs
-------
docs/board_audit_m20097/repro/repro_result.json   (machine readable)
stdout                                            (compact table)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tensordg.advection_1d import solve_advection, l2_error as l2_error_1d  # noqa: E402
from tensordg.advection_2d_var import DGAdvection2D, trig_profile  # noqa: E402
from tensordg.convergence import convergence_table, pairwise_orders  # noqa: E402

ART = ROOT / "docs" / "board_audit_m20097" / "anchor_artifacts"
OUT = ROOT / "docs" / "board_audit_m20097" / "repro"

# Ladders exactly as declared by tools/run_advection_porder_study.py (m20026, dbf5bdc).
LADDERS_1D = {0: [20, 40, 80, 160], 1: [10, 20, 40, 80], 2: [10, 20, 40, 80],
              3: [8, 16, 32, 64], 4: [8, 16, 32, 64]}
LADDERS_2D = {0: [8, 16, 32], 1: [8, 16, 32, 64], 2: [6, 12, 24], 3: [6, 12, 24]}


def _vel(ax: float, ay: float):
    def velocity(x, y, t):
        return torch.full_like(x, ax), torch.full_like(y, ay)
    return velocity


def run_1d() -> dict:
    out = {}
    for p, cells in LADDERS_1D.items():
        errs, runs = [], []
        for n in cells:
            centers, coeff, meta = solve_advection(
                ncells=n, degree=p, final_time=1.0, speed=1.0, cfl=0.08,
                time_integrator="rk4",
            )
            errs.append(l2_error_1d(centers, coeff, 1.0, speed=1.0))
            runs.append({"ncells": n, "dt": meta["dt"], "steps": meta["steps"]})
        tbl = convergence_table(cells, errs, theoretical_order=p + 1)
        out[f"p{p}"] = {
            "cells": cells,
            "errors_l2": errs,
            "observed_order_lsq": tbl["observed_order_lsq"],
            "pairwise_orders": tbl["pairwise_orders"],
            "theoretical_order": p + 1,
            "order_deviation": tbl["order_deviation"],
            "order_within_tolerance": tbl["order_within_tolerance"],
            "runs": runs,
        }
    return out


def run_2d() -> dict:
    vel = _vel(1.0, 0.5)
    out = {}
    for p, cells in LADDERS_2D.items():
        errs, runs = [], []
        for n in cells:
            solver = DGAdvection2D(p, n)
            coeff0 = solver.project(trig_profile)
            res = solver.solve(coeff0, vel, 1.0, cfl=0.08)
            err = solver.l2_error(res["coeff"], trig_profile, 1.0)
            errs.append(err["absolute_l2"])
            runs.append({"ncells": n, "dt": res["meta"]["dt"],
                         "steps": res["meta"]["steps"],
                         "mass_drift_rel": res["meta"]["mass_drift_rel"]})
        tbl = convergence_table(cells, errs, theoretical_order=p + 1)
        out[f"p{p}"] = {
            "cells": cells,
            "errors_l2": errs,
            "observed_order_lsq": tbl["observed_order_lsq"],
            "pairwise_orders": tbl["pairwise_orders"],
            "theoretical_order": p + 1,
            "order_deviation": tbl["order_deviation"],
            "order_within_tolerance": tbl["order_within_tolerance"],
            "runs": runs,
        }
    return out


def compare(tag: str, mine: dict, committed: dict) -> list[dict]:
    rows = []
    for key, m in mine.items():
        c = committed.get(key, {})
        c_err = c.get("errors_l2", [])
        rel = [abs(a - b) / abs(b) for a, b in zip(m["errors_l2"], c_err)] if c_err else []
        rows.append({
            "block": tag, "degree": key,
            "cells": m["cells"],
            "repro_errors": m["errors_l2"], "committed_errors": c_err,
            "max_rel_diff_errors": max(rel) if rel else None,
            "repro_order_lsq": round(m["observed_order_lsq"], 4),
            "committed_order_lsq": c.get("observed_order_lsq"),
            "repro_pairwise": [round(x, 4) for x in m["pairwise_orders"]],
            "committed_pairwise": c.get("pairwise_orders"),
        })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    committed = json.loads((ART / "porder_convergence.json").read_text())
    committed_1d = json.loads((ART / "porder_convergence_1d.json").read_text())

    r1d = run_1d()
    r2d = run_2d()

    rows = compare("1d", r1d, committed["study_1d"]["results"])
    rows += compare("2d", r2d, committed["study_2d"]["results"])

    # (a) does the *committed* artifact survive its own library's convergence_table?
    internal = []
    for tag, block, ref in (("1d", committed["study_1d"]["results"], committed_1d.get("study_1d", {}).get("results", {})),
                            ("2d", committed["study_2d"]["results"], {})):
        for key, v in block.items():
            pw = pairwise_orders(v["cells"], v["errors_l2"])
            tbl = convergence_table(v["cells"], v["errors_l2"], theoretical_order=v["theoretical_order"])
            internal.append({
                "block": tag, "degree": key,
                "artifact_pairwise": v["pairwise_orders"],
                "recomputed_pairwise_from_artifact_errors": [round(x, 4) for x in pw],
                "artifact_lsq": v["observed_order_lsq"],
                "recomputed_lsq": round(tbl["observed_order_lsq"], 4),
                "pairwise_matches": all(abs(a - b) < 5e-4 for a, b in zip(v["pairwise_orders"], pw)),
                "lsq_matches": abs(v["observed_order_lsq"] - tbl["observed_order_lsq"]) < 5e-4,
                "in_1d_sidecar_too": key in ref,
            })

    # (b) registered-benchmark judgement: tensor_benchmark_cases / REPRO-DG-CKS1998 A1
    bench = []
    for key in ("p1", "p2", "p3"):
        p = int(key[1])
        got_committed = committed["study_1d"]["results"][key]["observed_order_lsq"]
        got_repro = r1d[key]["observed_order_lsq"]
        gate = p + 0.5          # registered: order >= p + 0.5
        bench.append({
            "degree": key, "gate_order_ge": gate,
            "committed_order": round(got_committed, 4),
            "repro_order": round(got_repro, 4),
            "committed_pass": got_committed >= gate,
            "repro_pass": got_repro >= gate,
            "margin_committed": round(got_committed - gate, 4),
            "margin_repro": round(got_repro - gate, 4),
        })

    result = {
        "audit": "m20097 cross-audit of m20026 CASE-DG-ADVECT-ORDER",
        "committed_artifact_md5": {
            "porder_convergence.json": __import__("hashlib").md5(
                (ART / "porder_convergence.json").read_bytes()).hexdigest(),
            "porder_convergence_1d.json": __import__("hashlib").md5(
                (ART / "porder_convergence_1d.json").read_bytes()).hexdigest(),
        },
        "repro_vs_committed": rows,
        "artifact_internal_consistency": internal,
        "registered_benchmark_check": bench,
        "repro_1d": r1d,
        "repro_2d": r2d,
    }
    (OUT / "repro_result.json").write_text(json.dumps(result, indent=2, default=float))

    print("=" * 78)
    print("A. repro vs committed artifact (errors_l2)")
    for r in rows:
        print(f"{r['block']} {r['degree']}: max rel diff = {r['max_rel_diff_errors']:.3e} | "
              f"lsq committed={r['committed_order_lsq']} repro={r['repro_order_lsq']}")
    print("=" * 78)
    print("B. artifact internal consistency (pairwise_orders vs its own errors)")
    for r in internal:
        print(f"{r['block']} {r['degree']}: artifact={r['artifact_pairwise']} "
              f"recomputed={r['recomputed_pairwise_from_artifact_errors']} "
              f"match={r['pairwise_matches']} lsq_match={r['lsq_matches']}")
    print("=" * 78)
    print("C. registered benchmark REPRO-DG-CKS1998 A1 (order >= p+0.5)")
    for b in bench:
        print(f"{b['degree']}: gate>={b['gate_order_ge']} committed={b['committed_order']}"
              f" ({b['committed_pass']}) repro={b['repro_order']} ({b['repro_pass']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())