#!/usr/bin/env python3
"""Round-2 delta evidence for CASE-DG-ADVECT-ORDER (stage-matched RKDG).

Reads the per-degree ladder fragments of
``benchmarks/case_dg_advect_order_matched_rk.py`` (``frag_p*.json``), the
mechanically extracted reference table, and re-runs the fixed-sign time-order
probe, then writes a compact evidence bundle:

``validation/advection_order_matched_rk/metrics_delta.json``  -- all numbers
``docs/advorder_round2_delta.md``                             -- rendered tables

The point of the round is the *stage-matched* comparison: degree p is advanced
with exactly p+1 stages (SSP-RK(5,4) at p=4, where order 5 with 5 stages is
impossible), so both the rate and the stage count can be compared with the
reference column.  Every number here is computed, none is transcribed.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from benchmarks.case_dg_advect_order import fit_with_floor, fitted_order
from benchmarks.case_dg_advect_order_matched_rk import (DEGREES, DT_ORDER_TOL,
                                                        RATE_TOL, dt_order_probe)
from tensordg.runge_kutta import max_order_for_stages, tableau_for_degree

OUT = pathlib.Path("validation/advection_order_matched_rk")
REF = pathlib.Path("validation/advection_order/ref_table.json")
DOC = pathlib.Path("docs/advorder_round2_delta.md")


def main() -> dict:
    ref = json.loads(REF.read_text())
    rows_by_n = {r["N"]: r for r in ref["rows"]}
    column = {c["degree"]: c for c in ref["columns"]}
    comparison, checks = [], {}
    for degree in DEGREES:
        frag = json.loads((OUT / f"frag_p{degree}.json").read_text())
        tab = tableau_for_degree(degree)
        name = column[degree]["name"]
        meshes = [n for n in frag["cells"]]
        ref_l2 = [rows_by_n[n][name]["l2"] for n in meshes]
        got_l2 = frag["l2"]
        ref_fit, ref_n, _ = fit_with_floor(meshes, ref_l2)
        got_fit, got_n, dropped = fit_with_floor(meshes, got_l2)
        gap = None if got_fit is None else got_fit - ref_fit
        expected_order = min(degree + 1, max_order_for_stages(degree + 1))
        comparison.append({
            "degree": degree, "integrator": tab.name,
            "rk_stages": tab.stages, "rk_order": tab.order,
            "order_attainable": expected_order,
            "ref_column": name, "ref_stages": column[degree]["rk_stages"],
            "meshes": meshes, "got_l2": got_l2, "ref_l2": ref_l2,
            "ratio_got_over_ref": [g / r for g, r in zip(got_l2, ref_l2)],
            "ref_fitted_order": ref_fit, "got_fitted_order": got_fit,
            "order_gap_vs_ref": gap, "rate_points_used": got_n,
            "roundoff_meshes_dropped": dropped,
            "rate_ok": (gap is not None and abs(gap) <= RATE_TOL),
            "stages_matched": tab.stages == column[degree]["rk_stages"],
            "order_ok": tab.order == expected_order,
            "relative_rate_error_pct": (None if gap is None or ref_fit == 0
                                        else 100.0 * abs(gap) / abs(ref_fit)),
        })
    probes = [dt_order_probe(d) for d in DEGREES]
    checks["M1_rate"] = all(r["rate_ok"] for r in comparison)
    checks["M2_stages"] = all(r["stages_matched"] and r["order_ok"] for r in comparison)
    checks["M5_dt_order"] = all(p["fitted_time_order"] is not None
                                and abs(p["fitted_time_order"] - p["design_order"])
                                <= DT_ORDER_TOL for p in probes)
    payload = {
        "case": "CASE-DG-ADVECT-ORDER", "round": 2,
        "thresholds": {"RATE_TOL": RATE_TOL, "DT_ORDER_TOL": DT_ORDER_TOL},
        "reference": {"columns": column, "citation": ref["citation"],
                      "provenance_hashes": {k: ref["provenance"][k] for k in
                                            ("eprint_sha256", "table_block_sha256",
                                             "fetched_at")}},
        "comparison": comparison, "dt_order_probes": probes, "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "failed": [k for k, v in checks.items() if not v],
        "note": ("per-degree fragments may use different mesh ladders (the run "
                 "time per degree differs by an order of magnitude); each "
                 "fragment records its own ladder and every comparison uses the "
                 "reference row of the same N"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics_delta.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# CASE-DG-ADVECT-ORDER round 2 delta — stage-matched RKDG\n",
             "| p | integrator | stages | order | attainable | ref column | ref stages | ref 阶 | got 阶 | 差 | 相对误差 % | 判定 |",
             "|--:|---|---:|---:|---:|---|--:|---:|---:|---:|---:|:--:|"]
    for r in comparison:
        pct = "n/a" if r["relative_rate_error_pct"] is None else f"{r['relative_rate_error_pct']:.2f}"
        lines.append(f"| P{r['degree']} | {r['integrator']} | {r['rk_stages']} | {r['rk_order']} | "
                     f"{r['order_attainable']} | {r['ref_column']} | {r['ref_stages']} | "
                     f"{r['ref_fitted_order']:.3f} | {r['got_fitted_order']:.3f} | "
                     f"{r['order_gap_vs_ref']:+.3f} | {pct} | "
                     f"{'PASS' if r['rate_ok'] else 'FAIL'} |")
    lines += ["", "| p | 格式 | 设计阶 | 实测时间阶 | 拟合点 | 判定 |", "|--:|---|--:|---:|---:|:--:|"]
    for p in probes:
        lines.append(f"| P{p['degree']} | {p['integrator']} | {p['design_order']} | "
                     f"{p['fitted_time_order']:.3f} | {p['fit_points']} | "
                     f"{'PASS' if abs(p['fitted_time_order'] - p['design_order']) <= DT_ORDER_TOL else 'FAIL'} |")
    lines += ["", f"**checks**: {json.dumps(checks)}", f"**verdict**: {payload['verdict']}",
              "", "## L2 ladders (got vs ref, same N)", "",
              "| p | N | got L2 | ref L2 | got/ref |", "|--:|--:|--:|--:|--:|"]
    for r in comparison:
        for n, g, q, ratio in zip(r["meshes"], r["got_l2"], r["ref_l2"], r["ratio_got_over_ref"]):
            lines.append(f"| P{r['degree']} | {n} | {g:.4e} | {q:.4e} | {ratio:.3f} |")
    DOC.write_text("\n".join(lines) + "\n")
    (OUT.relative_to('.')).as_posix()
    print(json.dumps({"checks": checks, "verdict": payload["verdict"],
                      "rates": {f"P{r['degree']}": [r["ref_fitted_order"], r["got_fitted_order"]]
                                for r in comparison},
                      "probes": {f"P{p['degree']}": p["fitted_time_order"] for p in probes}},
                     indent=1))
    return payload


if __name__ == "__main__":
    main()
