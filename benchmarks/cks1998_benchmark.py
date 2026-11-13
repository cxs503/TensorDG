"""REPRO-DG-CKS1998 — benchmark table (reference vs got) + integration probe.

Two jobs, one entry point:

1. **Integration probe** (``integration_calls``): drives the shock-capturing
   entry points that were added to TensorDG for the CKS1998 reproduction
   *through the public API only* — ``solve_advection(..., limiter=,
   time_integrator="ssp_rk3", troubled_policy=)`` and
   ``apply_tvb_limiter(..., periodic=True, limit_all=True)``.  This is the
   non-test caller of the new code path; it is exercised by
   ``tests/test_repro_cks1998.py::test_benchmark_module_integration``.

2. **Reference-vs-got table** (``build_ref_vs_got``): reads the run artifacts
   written by ``validation/repro_cks1998/repro_cks1998.py`` and emits one row
   per benchmarked quantity with

       ref --- what the anchor says the number should be
       got --- what TensorDG actually produced
       tol --- the pre-declared tolerance
       verdict --- PASS/FAIL, machine decided

   ``ref`` provenance is declared per row and is one of

   * ``analytic``       — the exact (Riemann) solution / exact bounds;
   * ``paper_theorem``  — the order/floor the paper *proves* (p+1, O(h^{1/2}));
   * ``withheld``       — a literal cell of a paper table we could not verify
                          without the PDF, so it is *not* claimed as PASS.

Usage
-----
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_benchmark            # probe + table
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_benchmark --probe-only
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_benchmark \
        --artifacts validation/repro_cks1998/artifacts

Exit code 0 iff the integration probe passes and the emitted table contains no
FAIL rows that were not already expected (the Sod coarse-mesh FAIL is reported,
not hidden, and the exit code follows the *pre-declared gate* verdicts stored
by the reproduction driver).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tensordg.advection_1d import l1_error, l2_error, solve_advection  # noqa: E402
from tensordg.limiters import apply_tvb_limiter  # noqa: E402

DEFAULT_ARTIFACTS = REPO / "validation" / "repro_cks1998" / "artifacts"
OUT_JSON = REPO / "validation" / "repro_cks1998" / "benchmark_ref_vs_got.json"
OUT_MD = REPO / "validation" / "repro_cks1998" / "benchmark_ref_vs_got.md"


# ---------------------------------------------------------------------------
# 1. Integration probe — real calls into the public shock-capturing API
# ---------------------------------------------------------------------------
def _square_wave(x: torch.Tensor) -> torch.Tensor:
    xm = torch.remainder(x, 1.0)
    return ((xm > 0.25) & (xm < 0.75)).to(x.dtype)


def integration_calls(ncells: int = 80, degree: int = 1) -> dict:
    """Exercise every new public entry point once; return measured evidence.

    No assertions here — the caller/test decides.  Everything returned is a
    measured number so the probe doubles as a smoke benchmark.
    """
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    out: dict = {"calls": []}

    # (a) limited square wave through the solver entry point (added plumbing)
    c, coeff, meta = solve_advection(
        ncells=ncells, degree=degree, final_time=1.0, cfl=0.2,
        initial=_square_wave, limiter="tvb", tvb_constant=0.0,
        time_integrator="ssp_rk3", troubled_policy="all",
    )
    ti = torch.linspace(-1.0, 1.0, 21, dtype=coeff.dtype)
    leg = np.polynomial.legendre
    basis = torch.tensor(np.stack([leg.legval(ti.numpy(), np.eye(coeff.shape[1])[k])
                                   for k in range(coeff.shape[1])], axis=1))
    vals = coeff @ basis.T
    out["calls"].append({
        "call": "solve_advection(limiter='tvb', time_integrator='ssp_rk3', troubled_policy='all')",
        "ncells": ncells, "degree": degree, "steps": int(meta["steps"]),
        "limiter": meta["limiter"], "time_integrator": meta["time_integrator"],
        "pt_max": float(vals.max()), "pt_min": float(vals.min()),
        "overshoot_pct": 100 * max(0.0, float(vals.max()) - 1.0, 0.0 - float(vals.min())),
    })

    # (b) direct limiter call with the periodic wrap-around neighbours
    c2, coeff2, _ = solve_advection(ncells=ncells, degree=degree, final_time=1.0,
                                    cfl=0.2, initial=_square_wave,
                                    limiter=None, time_integrator="ssp_rk3")
    h = 1.0 / ncells
    coeff_2d = coeff2
    lim_p = apply_tvb_limiter(coeff_2d.unsqueeze(-1), h, tvb_constant=0.0,
                              periodic=True, limit_all=True).squeeze(-1)
    changed = int((lim_p - coeff2).abs().max(dim=1).values.gt(1e-14).sum())
    out["calls"].append({
        "call": "apply_tvb_limiter(periodic=True, limit_all=True)",
        "ncells": ncells, "degree": degree,
        "cells_modified": changed,
        "edge_cell_modified": bool((lim_p[0] - coeff2[0]).abs().max() > 1e-14),
        "last_cell_modified": bool((lim_p[-1] - coeff2[-1]).abs().max() > 1e-14),
    })

    # (c) convergence sanity through the same entry point (cheap, 3 meshes)
    errs = []
    for n in (20, 40, 80):
        cc, aa, _ = solve_advection(ncells=n, degree=degree, final_time=0.2, cfl=0.1)
        errs.append(l2_error(cc, aa, 0.2))
    out["calls"].append({
        "call": "solve_advection(default path) L2 error ladder",
        "ncells": [20, 40, 80], "l2_errors": errs,
        "observed_order": float(math.log(errs[0] / errs[-1]) / math.log(80 / 20)),
    })

    # (d) L1 metric used by the square-wave bench must be callable in the loop
    e_l1 = l1_error(c, coeff, 1.0, exact=_square_wave)
    out["calls"].append({"call": "l1_error(square wave)", "l1_error": float(e_l1),
                         "degrees_of_freedom": int(c.numel() * coeff.shape[1])})

    out["probe_pass"] = bool(
        out["calls"][0]["overshoot_pct"] <= 1e-9
        and out["calls"][1]["cells_modified"] > 0
        and out["calls"][1]["edge_cell_modified"]
        and out["calls"][1]["last_cell_modified"]
        and out["calls"][2]["observed_order"] > degree + 0.5
    )
    return out


# ---------------------------------------------------------------------------
# 2. Reference-vs-got table from the reproduction run artifacts
# ---------------------------------------------------------------------------
def _row(item, quantity, ref, got, tol, verdict, ref_source, note="", unit="",
         gated=False):
    """One ref-vs-got row.

    ``verdict`` is *only* ever the tolerance rule applied to (ref, got); the
    separate ``gated`` flag says whether the row is part of the pre-declared
    anchor gate.  The two used to be conflated: an out-of-tolerance *non-gate*
    row was printed as ``informative``, which hid a real 4.17 % violation
    (audit m15345 defect M2).  Laundering is now impossible by construction.
    """
    ref_f = float(ref) if ref is not None else None
    got_f = float(got) if got is not None else None
    if ref_f is not None and got_f is not None and ref_f != 0:
        rel = abs(got_f - ref_f) / abs(ref_f)
        diff = got_f - ref_f
    elif ref_f is not None and got_f is not None:
        rel, diff = None, got_f - ref_f
    else:
        rel, diff = None, None
    assert verdict in ("PASS", "FAIL", "UNSCORED"), verdict
    return {"item": item, "quantity": quantity, "ref": ref_f, "got": got_f,
            "unit": unit, "abs_diff": diff, "rel_diff": rel,
            "tolerance": tol, "verdict": verdict, "gated": bool(gated),
            "ref_source": ref_source, "note": note}


def _rule(ok: bool) -> str:
    """A scored row can only be PASS or FAIL — never anything else."""
    return "PASS" if ok else "FAIL"


def build_ref_vs_got(metrics: dict, profile: str) -> dict:
    """Turn one ``metrics.json`` payload into a ref-vs-got evidence table."""
    sec = metrics["sections"]
    gates = metrics["gates"]
    rows: list[dict] = []

    # A1 — smooth advection: ref = paper theorem order p+1, got = LSQ observed
    for key, block in sec["A1_smooth"]["by_degree"].items():
        p = int(key[1:])
        got_order = block["observed_order"]
        rows.append(_row(
            "A1_smooth_order", f"L2 observed order {key}", p + 1.0, got_order,
            f">= p+0.5 = {p + 0.5}", _rule(got_order >= p + 0.5),
            "paper_theorem",
            "Cockburn & Shu: DG(P^p) converges at O(h^{p+1}) for smooth data; "
            "the mission tolerance is the looser p+0.5", gated=True))

    # A2 — square wave: ref = O(h^{1/2}) L1 floor proved in the paper
    for key, block in sec["A2_discontinuous"]["by_degree"].items():
        got_order = block["l1_order"]
        rows.append(_row(
            "A2_discont_order", f"L1 observed order {key}", 0.5, got_order,
            ">= 0.5", _rule(got_order >= 0.5), "paper_theorem",
            "half-order L1 floor for limited schemes at discontinuities",
            gated=True))

    # E2 — Sod: numeric ref vs got at fixed stations + integral rel-L2
    sod_ref = metrics.get("sod_reference_stations", {})
    for tag, case in sec["E2_sod"]["cases"].items():
        gated = bool(case.get("gate_case", True))
        for comp in ("density", "velocity", "pressure"):
            err = case["rel_l2"][comp]
            rows.append(_row(
                f"E2_sod_{tag}", f"rel L2 error, {comp}", None, err,
                f"< {gates['sod_rel_l2']:.0%}", _rule(err < gates["sod_rel_l2"]),
                "analytic", "ref = exact Riemann solution (benchmarks.sod_exact)"
                + ("" if gated else "; grid below the pre-declared gate resolution"),
                unit="-", gated=gated))
    # the reference (analytic) vs got values themselves, station by station
    for tag, stations in sod_ref.items():
        # Supplementary evidence, never a gate row: the pre-declared E2 gate is the
        # *integral* relative L2 on the gate grid.  A station compares a DG cell
        # average with a point value of the exact solution, so it is scored with
        # the same 3 % rule (a violation still prints FAIL) but cannot make or
        # break the anchor.
        for st in stations:
            for comp in ("density", "velocity", "pressure"):
                ref, got = st["ref"][comp], st["got"][comp]
                if ref == 0:
                    # a purely zero reference has no relative scale: report, do not score
                    verdict = "UNSCORED"
                else:
                    verdict = _rule(abs(got - ref) <= gates["sod_rel_l2"] * abs(ref))
                rows.append(_row(
                    f"E2_sod_{tag}", f"{comp} at x={st['x']:.3f}", ref, got,
                    f"< {gates['sod_rel_l2']:.0%} rel", verdict, "analytic",
                    "cell-average vs pointwise exact solution (supplementary, not a gate)"
                    + ("" if ref != 0 else "; zero reference, no relative scale"),
                    unit="-", gated=False))

    # E3 — TVD/TVB bound: ref = exact bounds [0, 1] of the initial data
    for tag, case in sec["E3_tvd"]["cases"].items():
        tol = f"<= {gates['tvd_overshoot']:.0%} overshoot"
        strict = tag.endswith("strict_all")
        over = case["overshoot_pointwise_pct"] / 100.0
        rows.append({
            "item": "E3_tvd", "quantity": f"pointwise max, {tag}", "ref": 1.0,
            "got": case["pt_max"], "unit": "-", "abs_diff": case["pt_max"] - 1.0,
            "rel_diff": case["pt_max"] - 1.0, "tolerance": tol,
            "verdict": _rule(over <= gates["tvd_overshoot"]), "gated": strict,
            "ref_source": "analytic",
            "note": ("strict minmod policy (gate row)" if strict
                     else f"not a gate row ({case['limiter']}): reported with the same "
                          "tolerance rule, so a violation still shows as FAIL"),
        })

    gated_rows = [r for r in rows if r["gated"]]
    counts = {
        "gated_pass": sum(r["verdict"] == "PASS" for r in gated_rows),
        "gated_fail": sum(r["verdict"] == "FAIL" for r in gated_rows),
        "gated_total": len(gated_rows),
        "ungated_pass": sum(r["verdict"] == "PASS" for r in rows if not r["gated"]),
        "ungated_fail": sum(r["verdict"] == "FAIL" for r in rows if not r["gated"]),
        "unscored": sum(r["verdict"] == "UNSCORED" for r in rows),
        "total": len(rows),
    }
    counts["FAIL"] = counts["gated_fail"] + counts["ungated_fail"]
    counts["PASS"] = counts["gated_pass"] + counts["ungated_pass"]
    # what the anchor actually claims: every *gated* row inside its tolerance.
    assert counts["gated_fail"] == 0, (
        f"pre-declared gate rows out of tolerance: "
        f"{[r['quantity'] for r in gated_rows if r['verdict'] == 'FAIL']}")
    return {
        "anchor": metrics["anchor"],
        "profile": profile,
        "reference_provenance": {
            "analytic": "exact Riemann solution / exact data bounds — machine-decidable",
            "paper_theorem": "order/floor stated and proved in Cockburn & Shu",
            "withheld": "literal cells of the paper's tables — no PDF in the repo, "
                        "NOT claimed (see docs/repro_cks1998.md §4)",
        },
        "counts": counts,
        "gate_verdict": metrics["verdict"],
        "environment": metrics["environment"],
        "wall_seconds": metrics["wall_seconds"],
        "rows": rows,
    }


def _counts_line(counts: dict) -> str:
    return (f"gated PASS {counts['gated_pass']}/{counts['gated_total']} · "
            f"non-gated out-of-tolerance {counts['ungated_fail']} · "
            f"unscored {counts['unscored']} · rows {counts['total']}")


def render_md(table: dict) -> str:
    lines = [
        "# REPRO-DG-CKS1998 — reference vs got",
        "",
        f"Anchor: {table['anchor']}",
        f"Profile: `{table['profile']}` · {_counts_line(table['counts'])}",
        "",
        "`verdict` is the tolerance rule applied to (ref, got) **and nothing "
        "else**; `gated` marks the pre-declared anchor rows. A non-gated row "
        "that leaves its tolerance is printed as `FAIL`, so the table can no "
        "longer hide a violation behind the word `informative`.",
        "",
        "`ref` provenance: *analytic* = exact solution, *paper_theorem* = proved "
        "order/floor in Cockburn & Shu, *withheld* = paper table cell not "
        "claimed (no PDF). Regenerate with "
        "`python -m benchmarks.cks1998_benchmark`.",
        "",
        "| item | quantity | ref | got | abs diff | rel diff | tolerance | verdict | gated | ref source |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for r in table["rows"]:
        def fmt(v, nd=6):
            return "" if v is None else f"{v:.{nd}g}"
        lines.append(
            f"| {r['item']} | {r['quantity']} | {fmt(r['ref'])} | {fmt(r['got'])} | "
            f"{fmt(r['abs_diff'], 3)} | {fmt(r['rel_diff'], 3)} | {r['tolerance']} | "
            f"{r['verdict']} | {'yes' if r['gated'] else 'no'} | {r['ref_source']} |")
    lines += ["", "## Gate verdict (from the run artifacts)", "",
              "```json", json.dumps(table["gate_verdict"], indent=2), "```", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS),
                    help="dir holding <profile>/metrics.json")
    ap.add_argument("--profiles", default="full,quick",
                    help="comma list, first existing one is used for the table")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-md", default=str(OUT_MD))
    args = ap.parse_args(argv)

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    probe = integration_calls()
    print(json.dumps({"integration_probe": probe}, indent=2))
    if args.probe_only:
        return 0 if probe["probe_pass"] else 1

    root = Path(args.artifacts)
    chosen = None
    for prof in args.profiles.split(","):
        p = root / prof.strip() / "metrics.json"
        if p.exists():
            chosen = (prof.strip(), json.loads(p.read_text()))
            break
    if chosen is None:
        print(f"no metrics.json under {root}; run the reproduction driver first",
              file=sys.stderr)
        return 2
    profile, metrics = chosen
    table = build_ref_vs_got(metrics, profile)
    table["integration_probe"] = probe
    Path(args.out_json).write_text(json.dumps(table, indent=2))
    Path(args.out_md).write_text(render_md(table))
    print(json.dumps({"table": str(args.out_json), "profile": profile,
                      "counts": table["counts"], "probe_pass": probe["probe_pass"]},
                     indent=2))
    # exit code tracks the *gate* (pre-declared rows); non-gated violations are
    # reported honestly in the table but must not be able to flip the anchor
    return 0 if (probe["probe_pass"] and table["counts"]["gated_fail"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())