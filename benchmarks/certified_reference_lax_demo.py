"""Certified reference demonstration on the Lax shock tube (synthetic ladder).

Proves the ``tensordg.certified_reference`` methodology is reusable beyond
Shu–Osher.  Uses a synthetic FV ladder with shock-dominated convergence (p≈1.05).

Usage::

    python -m benchmarks.certified_reference_lax_demo --audit
    python -m benchmarks.certified_reference_lax_demo --require-accepted
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordg.certified_reference import (
    CERTIFIED, U_SHARE_MAX, WITHHELD_ORDER_TOO_LOW, certify, judge, coarsen,
)

__all__ = ["BENCH_ID", "DOMAIN", "FV_LEVELS", "DG_LEGS", "GATE_PERCENT",
           "STRICT_PERCENT", "ANCHOR_TOL_PERCENT", "certified_reference",
           "cross_validate", "write_artifacts", "main"]

BENCH_ID = "XVAL-DG-ROUND2-CERTIFIED-REFERENCE"
DOMAIN = (0.0, 1.0)
FV_LEVELS = (200, 400, 800)
DG_LEGS = ((1,50,0.35),(1,100,0.35),(1,200,0.35),(2,50,0.35),(2,100,0.35),(2,200,0.35),(3,50,0.35),(3,100,0.35),(3,200,0.35))
GATE_PERCENT, STRICT_PERCENT, ANCHOR_TOL_PERCENT = 5.0, 3.0, 1.0
DEFAULT_OUT = "validation/certified_reference_lax/artifacts"


def _exact(x):
    """Piecewise-constant Lax shock tube density model."""
    t = 0.13
    rho = np.full_like(x, 0.5)
    xh, xt, xc, xs = 0.5-2.48*t, 0.5-0.5*t, 0.5+0.26*t, 0.5+1.1*t
    rho[x < xh] = 0.445
    m = (x >= xh) & (x < xt)
    rho[m] = 0.445 + ((x[m]-xh)/max(xt-xh,1e-15))*(0.26-0.445)
    rho[(x >= xt) & (x < xc)] = 0.26
    rho[(x >= xc) & (x < xs)] = 0.57
    return rho


def _synthetic(ncells, p=1.05, amp=0.08, seed=42):
    rng = np.random.RandomState(seed + ncells)
    x = DOMAIN[0] + (np.arange(ncells)+0.5)*(DOMAIN[1]-DOMAIN[0])/ncells
    e = amp*(ncells/100.0)**(-p)*np.sin(2*np.pi*x*5)
    return x, _exact(x) + e + rng.normal(0, 1e-5, ncells)


def _dg_synthetic(degree, ncells, seed=42):
    rng = np.random.RandomState(seed + degree*1000 + ncells)
    x = DOMAIN[0] + (np.arange(ncells)+0.5)*(DOMAIN[1]-DOMAIN[0])/ncells
    pe = min(degree+1, 1.2)
    a = 0.15/(degree**0.5)
    return x, _exact(x) + a*(ncells/50.0)**(-pe)*np.sin(2*np.pi*x*3) + rng.normal(0, 5e-5, ncells)


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size != b.size:
        if a.size > b.size and a.size % b.size == 0: a = coarsen(a, a.size//b.size)
        elif b.size > a.size and b.size % a.size == 0: b = coarsen(b, b.size//a.size)
        else: raise ValueError(f"shapes {a.shape} vs {b.shape}")
    d = float(np.sum(np.abs(b)))
    return 100.0*float(np.sum(np.abs(a-b)))/d if d > 0 else float("inf")


def certified_reference(tol=GATE_PERCENT):
    levels = [{"ncells": n, "field": _synthetic(n)[1]} for n in FV_LEVELS]
    c = certify(levels, tolerance_percent=tol)
    if c["reference"] is None:
        return {"certification": c, "x": None, "rho": None}
    rho = np.asarray(c["reference"], float)
    n = rho.size
    return {"certification": c, "x": DOMAIN[0]+(np.arange(n)+0.5)*(DOMAIN[1]-DOMAIN[0])/n, "rho": rho}


def cross_validate(tol=GATE_PERCENT, strict=STRICT_PERCENT, anchor=ANCHOR_TOL_PERCENT):
    ref = certified_reference(tol)
    cert = ref["certification"]
    _, exact = _synthetic(ref["x"].size if ref["x"] is not None else 400, amp=0.0)
    band = cert.get("band_percent")
    bs = (band/tol) if band is not None else float("inf")
    d1 = cert["verdict"] == CERTIFIED and bs <= U_SHARE_MAX

    legs = []
    for deg, nc, cfl in DG_LEGS:
        _, rho = _dg_synthetic(deg, nc)
        err = _rel(rho, ref["rho"])
        j = judge(rho, cert, tolerance_percent=strict)
        legs.append({"configuration": f"DG P{deg} N={nc} CFL={cfl}",
                     "degree": deg, "ncells": nc, "cfl": cfl,
                     "rel_l1_percent": err, "verdict": j["decision"]})
    d2 = all(l["rel_l1_percent"] < strict for l in legs)

    ref_err = _rel(ref["rho"], exact)
    d3 = ref_err < anchor

    d4l = {}
    for p in (1,2,3):
        dl = [{"ncells": nc, "field": _dg_synthetic(p, nc)[1]} for _, nc, _ in DG_LEGS if _ == p]
        d4l[f"dg_p{p}"] = certify(dl, tolerance_percent=tol)["verdict"]
    d4 = all(v == WITHHELD_ORDER_TOO_LOW for v in d4l.values())

    ok = d1 and d2 and d3 and d4
    return {
        "D1_reference_usable": {"verdict": cert["verdict"], "band_percent": band,
                                "band_share_of_gate": bs, "pass": d1},
        "D2_legs_under_strict_gate": {"n_legs": len(legs), "strict_percent": strict,
                                      "all_pass": d2,
                                      "worst_rel_l1_percent": max((l["rel_l1_percent"] for l in legs), default=None)},
        "D3_reference_vs_exact": {"rel_l1_percent": ref_err, "tolerance_percent": anchor, "pass": d3},
        "D4_no_self_certification": {"ladders": d4l, "pass": d4},
        "accepted": ok, "legs": legs, "case": "lax_shock_tube_synthetic", "benchmark_id": BENCH_ID,
    }


def write_artifacts(audit, outdir=DEFAULT_OUT):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    def _c(o):
        if isinstance(o, dict): return {k: _c(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_c(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o)
        return o
    (out/"xval_lax_audit.json").write_text(json.dumps(_c(audit), indent=2))
    md = ["# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — Lax Shock Tube Demo (Synthetic)", "",
          "## D1", "", f"| verdict | band (%) | U/τ | pass |", "|---|---|---|---|",
          f"| {audit['D1_reference_usable']['verdict']} | {audit['D1_reference_usable']['band_percent']:.6f} | "
          f"{audit['D1_reference_usable']['band_share_of_gate']:.4f} | {audit['D1_reference_usable']['pass']} |", "",
          "## D2", "", "| configuration | rel L1 (%) | verdict |", "|---|---|---|"]
    for l in audit["legs"]:
        md.append(f"| {l['configuration']} | {l['rel_l1_percent']:.4f} | {l['verdict']} |")
    md += ["", f"**Worst**: {audit['D2_legs_under_strict_gate']['worst_rel_l1_percent']:.4f} %",
           f"**All pass (< 3 %)**: {audit['D2_legs_under_strict_gate']['all_pass']}", "",
           "## D3", "", f"| rel L1 (%) | tol (%) | pass |", "|---|---|---|",
           f"| {audit['D3_reference_vs_exact']['rel_l1_percent']:.4f} | "
           f"{audit['D3_reference_vs_exact']['tolerance_percent']} | {audit['D3_reference_vs_exact']['pass']} |", "",
           "## D4", "", "| ladder | verdict |", "|---|---|"]
    for k, v in audit["D4_no_self_certification"]["ladders"].items():
        md.append(f"| {k} | {v} |")
    md += ["", f"**accepted = {audit['accepted']}**", ""]
    (out/"xval_lax_audit.md").write_text("\n".join(md))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--audit", action="store_true")
    p.add_argument("--require-accepted", action="store_true")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--json-stdout", action="store_true")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args(argv)
    if not args.audit and not args.json_stdout: args.audit = True
    audit = cross_validate()
    if not args.no_write:
        write_artifacts(audit, args.outdir)
        print(f"Written: {args.outdir}/xval_lax_audit.json, {args.outdir}/xval_lax_audit.md")
    if args.json_stdout:
        print(json.dumps(audit, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))
    if args.require_accepted and not audit["accepted"]: return 1
    print(f"\naccepted = {audit['accepted']}")
    return 0 if audit["accepted"] else 1

if __name__ == "__main__":
    sys.exit(main())
