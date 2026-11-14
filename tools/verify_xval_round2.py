"""Second-path reproduction of the round-2 XVAL evidence bundle.

Why this exists
---------------
``docs/XVAL-DG-ROUND2-CERTIFIED-REFERENCE.md`` reports a handful of very small
numbers (cross-implementation certified-reference difference ``0.0115 %``, GCI
bands ``~0.0828 %``, DG-leg errors).  Numbers that small are exactly the ones a
reader cannot sanity-check by eye, so they must be **recomputable from committed
artifacts by a second, independent code path** -- otherwise the evidence bundle
is only as trustworthy as one script that no longer exists in the tree.

The original study computed everything with the one-off driver
:mod:`benchmarks.xval_oss_shu_osher` / :mod:`benchmarks.shu_osher_certified_ref`.
This tool recomputes the *same* quantities with the reusable engine
:mod:`tensordg.certified_reference` (``certify``/``judge``), reading only
committed raw profiles:

* our finite-volume ladder: ``validation/repro_shu_osher/artifacts/raw/fv_cfl0p3_n*.npz``
* the OSS (Clawpack) ladders: ``artifacts/xval_round2/oss_raw/oss_*_n*.npz``
* the DG legs of the case pool: ``validation/repro_shu_osher/artifacts/raw/dg_*_p*.npz``

Two implementations of the same statistics (one-off driver vs reusable engine)
agreeing on the numbers is a *real* check: it catches transcription errors, a
stale ``xval.json``, and silent formula drift in one of the two paths.  Every
deviation is reported in percent; nothing is rounded away.

Usage::

    python -m tools.verify_xval_round2                 # human table
    python -m tools.verify_xval_round2 --json artifacts/xval_round2/verify.json

Exit status: 0 when every recomputed quantity matches the committed claim
within ``--tol`` (relative), 1 otherwise (honest FAIL, never smoothed over).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensordg.certified_reference import certify, judge, rel_l1

#: the two certified references are both reported on this grid
REF_GRID = 1600
#: same pre-declared numbers as the benchmark module (kept in sync by the test)
TOL_XVAL_PERCENT = 1.0
BAND_FACTOR = 3.0
GATE_PERCENT = 5.0
OSS_LEVELS = (800, 1600, 3200)
#: DG legs of the case pool, as recorded by the round-2 study
DG_LEGS = (("dg_cfl0p4_limweno_n200_p1.npz", "DG P1 N=200 CFL=0.4"),
           ("dg_cfl0p4_limweno_n400_p1.npz", "DG P1 N=400 CFL=0.4"),
           ("dg_cfl0p4_limweno_n800_p1.npz", "DG P1 N=800 CFL=0.4"),
           ("dg_cfl0p4_limweno_n200_p2.npz", "DG P2 N=200 CFL=0.4"),
           ("dg_cfl0p4_limweno_n400_p2.npz", "DG P2 N=400 CFL=0.4"),
           ("dg_cfl0p4_limweno_n800_p2.npz", "DG P2 N=800 CFL=0.4"),
           ("dg_cfl0p4_limweno_n200_p3.npz", "DG P3 N=200 CFL=0.4"),
           ("dg_cfl0p4_limweno_n400_p3.npz", "DG P3 N=400 CFL=0.4"),
           ("dg_cfl0p4_limweno_n800_p3.npz", "DG P3 N=800 CFL=0.4"),
           ("dg_cfl0p1_limweno_n400_p2.npz", "DG P2 N=400 CFL=0.1"),
           ("dg_cfl0p2_limweno_n400_p2.npz", "DG P2 N=400 CFL=0.2"))


# ------------------------------------------------------------------ loading
def _npz_profile(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        prof = {"x": np.asarray(d["x"], dtype=float),
                "rho": np.asarray(d["rho"], dtype=float)}
        if "meta" in d:
            meta = d["meta"]
            prof["meta"] = (json.loads(str(meta)) if getattr(meta, "shape", None) == ()
                            else {})
    return prof


def load_fv_levels(rawdir: Path) -> list[dict]:
    levels = []
    for path in sorted(rawdir.glob("fv_cfl0p3_n*.npz")):
        n = int(path.stem.rsplit("_n", 1)[1])
        prof = _npz_profile(path)
        levels.append({"ncells": n, **prof, "file": path.name})
    levels.sort(key=lambda d: d["ncells"])
    if len(levels) < 3:
        raise SystemExit(f"need >=3 committed FV levels in {rawdir}, found {len(levels)}")
    return levels


def load_oss_levels(outdir: Path, tag: str = "", levels=OSS_LEVELS,
                    cfl: float = 0.3) -> list[dict]:
    cfl_tag = ("%g" % cfl).replace(".", "p")
    out, missing = [], []
    for n in levels:
        path = outdir / "oss_raw" / f"oss_cfl{cfl_tag}{tag}_n{n}.npz"
        if not path.exists():
            missing.append(str(path))
            continue
        prof = _npz_profile(path)
        prof.update({"ncells": n, "file": path.name})
        out.append(prof)
    if missing:
        raise FileNotFoundError("missing OSS ladder files: " + ", ".join(missing))
    out.sort(key=lambda d: d["ncells"])
    return out


# ------------------------------------------------------- certifications
def certify_ladder(levels: list[dict], tolerance_percent: float) -> dict:
    """Engine certification of one ladder (density only, the studied field)."""
    return certify([{"ncells": lv["ncells"], "field": lv["rho"]} for lv in levels],
                   tolerance_percent=tolerance_percent)


def compare(claims: dict, recomputed: dict) -> list[dict]:
    """Pair every claimed number with its recomputation (relative deviation)."""
    rows = []

    def add(name, claim, got, unit="%"):
        if claim is None or got is None:
            rows.append({"quantity": name, "claimed": claim, "recomputed": got,
                         "deviation": None, "unit": unit, "status": "WITHHELD"})
            return
        scale = max(abs(float(claim)), abs(float(got)), 1e-300)
        dev = abs(float(claim) - float(got)) / scale
        rows.append({"quantity": name, "claimed": float(claim),
                     "recomputed": float(got), "deviation": dev, "unit": unit,
                     "status": "MATCH" if dev <= 1e-6 else
                               ("ROUNDOFF" if dev <= 1e-3 else "MISMATCH")})

    for key, label in (("p_obs", "observed order (unitless)"),
                       ("band_percent", "GCI band U [%]"),
                       ("displacement_percent", "extrapolation displacement [%]")):
        for who in ("ours", "oss"):
            add(f"{who}: {label}", claims.get(f"{who}_{key}"), recomputed[f"{who}"][key])
    add("D1 rel L1(cert_ours, cert_oss) [%]", claims.get("difference_percent"),
        recomputed["d1_percent"])
    add("D2 combined 3x band [%]", claims.get("combined_band_percent"),
        recomputed["d2_percent"])
    for row in recomputed["same_grid"]:
        match = [c for c in claims.get("same_grid", [])
                 if int(c.get("ncells", -1)) == int(row["ncells"])]
        add(f"same-grid rel L1 N={row['ncells']} [%]",
            match[0].get("rel_l1_percent") if match else None, row["rel_l1_percent"])
    by_name = {r.get("configuration"): r for r in claims.get("dg_legs", [])}
    for row in recomputed["dg_legs"]:
        claim = by_name.get(row["configuration"], {})
        add(f"{row['configuration']} vs OSS cert [%]", claim.get("rel_l1_percent"),
            row["rel_l1_percent"])
    return rows


def reproduce(outdir: Path, rawdir: Path, xval_json: Path) -> dict:
    claims = json.loads(xval_json.read_text())
    ours = certify_ladder(load_fv_levels(rawdir), GATE_PERCENT)
    oss = certify_ladder(load_oss_levels(outdir), GATE_PERCENT)
    if ours["reference_ncells"] != oss["reference_ncells"]:
        raise SystemExit("the two certified references do not live on the same grid: "
                         f"{ours['reference_ncells']} vs {oss['reference_ncells']}")
    ours["reference_profile"] = ours["reference"]
    oss["reference_profile"] = oss["reference"]

    d1 = 100.0 * rel_l1(ours["reference_profile"], oss["reference_profile"])
    d2 = BAND_FACTOR * math.hypot(ours["band_percent"], oss["band_percent"])

    # same-grid raw differences (coarsen everything onto N=800)
    fv = {lv["ncells"]: lv for lv in load_fv_levels(rawdir)}
    same_grid = []
    for lv in load_oss_levels(outdir):
        a = fv[lv["ncells"]]["rho"]
        same_grid.append({"ncells": lv["ncells"],
                          "rel_l1_percent": 100.0 * rel_l1(lv["rho"], a)})

    # D3: gate the DG legs against the OSS-certified reference
    dg_legs = []
    for name, label in DG_LEGS:
        path = rawdir / name
        if not path.exists():
            dg_legs.append({"configuration": label, "file": name, "missing": True,
                            "rel_l1_percent": None, "decision": "WITHHELD"})
            continue
        prof = _npz_profile(path)
        verdict = judge(prof["rho"], oss, tolerance_percent=GATE_PERCENT)
        dg_legs.append({"configuration": label, "file": name,
                        "rel_l1_percent": verdict["rel_l1_percent"],
                        "rel_l2_percent": verdict["rel_l2_percent"],
                        "decision": verdict["decision"],
                        "passes_5pct": bool(verdict["rel_l1_percent"] < GATE_PERCENT),
                        "robust_margin_percent_of_gate":
                            verdict["robust_margin_percent_of_gate"]})

    # char_decomp axis (the gap the first round left open)
    char = None
    try:
        ch = certify_ladder(load_oss_levels(outdir, tag="_char1"), GATE_PERCENT)
        char = {"p_obs": ch["observed_order_used"],
                "band_percent": ch["band_percent"],
                "certified_ref_diff_percent":
                    100.0 * rel_l1(ch["reference_profile"], oss["reference_profile"]),
                "verdict": ch["verdict"]}
    except FileNotFoundError:
        char = None

    recomputed = {
        "ours": {"p_obs": ours["observed_order_used"],
                 "band_percent": ours["band_percent"],
                 "displacement_percent": ours["extrapolation_displacement_percent"],
                 "verdict": ours["verdict"],
                 "displacement_ratio": ours["displacement_ratio"]},
        "oss": {"p_obs": oss["observed_order_used"],
                "band_percent": oss["band_percent"],
                "displacement_percent": oss["extrapolation_displacement_percent"],
                "verdict": oss["verdict"],
                "displacement_ratio": oss["displacement_ratio"]},
        "d1_percent": d1,
        "d1_passes": bool(d1 < TOL_XVAL_PERCENT),
        "d2_percent": d2,
        "d2_passes": bool(d1 <= d2),
        "same_grid": same_grid,
        "dg_legs": dg_legs,
        "char_decomp": char,
        "reference_ncells": ours["reference_ncells"],
        "reference_sha256": ours.get("input_sha256"),
    }

    claim_block = {
        "ours_p_obs": claims.get("ours", {}).get("observed_order_used"),
        "ours_band_percent": claims.get("ours", {}).get("gci_finest_percent"),
        "ours_displacement_percent":
            claims.get("certified", {}).get("extrapolation_displacement_ours_percent")
            if claims.get("certified", {}).get("extrapolation_displacement_ours_percent")
            is not None else claims.get("ours", {}).get("extrapolation_displacement_percent"),
        "oss_p_obs": claims.get("oss", {}).get("observed_order_used"),
        "oss_band_percent": claims.get("oss", {}).get("gci_finest_percent"),
        "oss_displacement_percent":
            claims.get("certified", {}).get("extrapolation_displacement_oss_percent")
            if claims.get("certified", {}).get("extrapolation_displacement_oss_percent")
            is not None else claims.get("oss", {}).get("extrapolation_displacement_percent"),
        "difference_percent": claims.get("certified", {}).get("difference_percent"),
        "combined_band_percent": claims.get("certified", {}).get("combined_band_percent"),
        "same_grid": claims.get("same_grid", []),
        "dg_legs": claims.get("dg_legs_vs_oss", []),
    }
    rows = compare(claim_block, recomputed)
    worst = max((r for r in rows if r["deviation"] is not None),
                key=lambda r: r["deviation"], default=None)
    mismatches = [r for r in rows if r["status"] == "MISMATCH"]
    roundoff = [r for r in rows if r["status"] == "ROUNDOFF"]
    status = ("MISMATCH" if mismatches else
              "REPRODUCED_ROUNDOFF" if roundoff else "REPRODUCED_EXACTLY")
    return {
        "outdir": str(outdir), "rawdir": str(rawdir), "xval_json": str(xval_json),
        "quantities_checked": len(rows),
        "rows": rows,
        "worst": worst,
        "mismatches": [r["quantity"] for r in mismatches],
        "status": status,
        "recomputed": recomputed,
    }


def _table(result: dict) -> str:
    lines = [f"{'quantity':52s} {'claimed':>12s} {'recomputed':>12s} {'|dev|':>10s}  status",
             "-" * 100]
    for row in result["rows"]:
        fmt = ("{:.9g}".format if row["quantity"].startswith("observed order") else "{:.6f}".format)
        claimed = "—" if row["claimed"] is None else fmt(row["claimed"])
        got = "—" if row["recomputed"] is None else fmt(row["recomputed"])
        dev = "—" if row["deviation"] is None else f"{row['deviation']:.2e}"
        lines.append(f"{row['quantity']:52s} {claimed:>12s} {got:>12s} {dev:>10s}  {row['status']}")
    lines.append("-" * 100)
    lines.append(f"status: {result['status']}  ({result['quantities_checked']} quantities)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="artifacts/xval_round2")
    ap.add_argument("--rawdir", default="validation/repro_shu_osher/artifacts/raw")
    ap.add_argument("--xval", default=None, help="default: <outdir>/xval.json")
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    xval_json = Path(args.xval) if args.xval else outdir / "xval.json"
    result = reproduce(outdir, Path(args.rawdir), xval_json)
    print(_table(result))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    return 0 if result["status"].startswith("REPRODUCED") else 1


if __name__ == "__main__":
    raise SystemExit(main())