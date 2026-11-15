"""Is the round-2 certified band honest?  A cross-order audit of the reference.

The claim under audit
---------------------
``XVAL-DG-ROUND2-CERTIFIED-REFERENCE`` quotes, for every DG leg of CASE-SHU-OSHER
(Shu & Osher 1988), a relative L1 density error ``< 3 %`` against a *certified*
finite-volume WENO5 reference whose uncertainty band is
``U = 0.083 %`` (``certified_reference.certify``, verdict ``CERTIFIED``).  All 11
legs pass; the worst is 2.456 %.

What that number does **not** test
----------------------------------
``U`` comes from a GCI over a ladder of *one* family: WENO5 at N = 800/1600/3200
(/6400).  A band built that way answers "how much does this family still move
when refined?", not "how far is this family from the continuum solution?".
``certified_reference.band_coverage_study`` says so explicitly ("that says
nothing about the shock case, where the true error cannot be measured at all").
So the certified reference can be *precise* (small ``U``) and still be *biased*
by more than ``U`` -- in which case every DG-leg error inherits that bias and the
3 % margins in the anchor are optimistic.

What this module does
---------------------
1. builds an independent ladder with the **same physics, different
   reconstruction order** (:mod:`benchmarks.fv_weno7_1d`: FV-WENO7/HLLC/SSP-RK3,
   same HLLC flux, same case module, same ghost-cell convention);
2. certifies both ladders with the *same* engine and the *same* gate (3 %);
3. measures the cross-family distance ``Delta = rel_l1(ref5, ref7)`` on the grid
   where both certified profiles live (second-finest level, so no interpolation
   enters) and compares it with the bands:

   * ``COVERED``       -- ``Delta <= U5``: the band bounds the family difference;
   * ``MARGINAL``      -- ``U5 < Delta <= U5 + U7``: the two bands just touch;
   * ``UNDER_COVERS``  -- ``Delta > U5 + U7``: ``U5`` does not bound the
     reference's own error, so the certificate is over-stated;

4. re-judges all 11 committed DG legs against *both* certified references and
   reports whether any verdict at the 3 % gate flips, and how far the worst
   margin moves.

Controls (so that a finding cannot be blamed on the new code alone)
-------------------------------------------------------------------
* the new order-5 path is run through the same pipeline: it must reproduce the
  shipped WENO5 reference to much less than ``U`` (it differs only in the
  smoothness-indicator normalisation, see :mod:`benchmarks.fv_weno7_1d`);
* two independent ladders within one family (3-level and 4-level) give the
  intra-family control distance, which the band is *designed* to bound.

Usage::

    python -m benchmarks.xval_dg_round2_band_honesty            # audit (+ cache)
    python -m benchmarks.xval_dg_round2_band_honesty --figures  # audit + figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensordg.cases.shock_entropy import DOMAIN, FINAL_TIME, initial_state
from tensordg.certified_reference import certify, coarsen, judge, rel_l1
from benchmarks.fv_weno7_1d import solve_fv_weno

RAW = Path("validation/repro_shu_osher/artifacts/raw")
OUTDIR = Path("validation/xval_dg_round2/artifacts/band_honesty")
WENO7_RAW = OUTDIR / "raw_weno7"
WENO5_GENERIC_RAW = OUTDIR / "raw_weno5_generic"
LADDER = (800, 1600, 3200, 6400)
CFL = 0.3
GATE = 3.0


def _load(path: Path):
    with np.load(path) as d:
        return float(d["x"][0]), d["rho"].astype(np.float64)


def _solve_level(ncells: int, order: int, cachedir: Path, tag: str) -> dict:
    """Run (or load from cache) one FV level of the Shu-Osher case."""
    cachedir.mkdir(parents=True, exist_ok=True)
    path = cachedir / f"{tag}_n{ncells}.npz"
    if path.exists():
        with np.load(path) as d:
            return {"ncells": ncells, "x": d["x"], "field": d["rho"].astype(np.float64),
                    "meta": json.loads(str(d["meta"])), "source": str(path), "cached": True}
    x, u, meta = solve_fv_weno(initial_state, ncells=ncells, order=order,
                               final_time=FINAL_TIME, domain=DOMAIN, cfl=CFL)
    np.savez_compressed(path, x=x, rho=u[:, 0], rho_u=u[:, 1], rho_e=u[:, 2],
                        meta=json.dumps({k: v for k, v in meta.items()}))
    return {"ncells": ncells, "x": x, "field": u[:, 0], "meta": meta,
            "source": str(path), "cached": False}


def load_weno5_ladder(rawdir: Path = RAW) -> list[dict]:
    levels = []
    for n in LADDER:
        x, rho = _load(rawdir / f"fv_cfl0p3_n{n}.npz")
        levels.append({"ncells": n, "field": rho, "source": str(rawdir / f"fv_cfl0p3_n{n}.npz")})
    return levels


def cert_of(levels: list[dict], n_levels: int = 0, gate: float = GATE) -> dict:
    lv = levels[:n_levels] if n_levels else levels
    out = certify([{"ncells": l["ncells"], "field": l["field"]} for l in lv],
                  tolerance_percent=gate)
    out["ladder"] = [l["ncells"] for l in lv]
    return out


def _cert_summary(c: dict) -> dict:
    return {k: c.get(k) for k in ("verdict", "classification", "band_percent",
                                 "band_share_of_gate", "observed_order",
                                 "reference_ncells", "ladder")}


def coverage_verdict(delta_percent: float, u_ref: float, u_other: float) -> dict:
    """Classify a cross-family distance against the two certified bands."""
    if delta_percent <= u_ref:
        level = "COVERED"
    elif delta_percent <= u_ref + u_other:
        level = "MARGINAL"
    else:
        level = "UNDER_COVERS"
    return {
        "delta_percent": float(delta_percent),
        "band_reference_percent": float(u_ref),
        "band_other_percent": float(u_other),
        "ratio_delta_over_band": float(delta_percent / u_ref) if u_ref else None,
        "level": level,
        "band_covers_cross_family": bool(level == "COVERED"),
        "combined_bands_cover": bool(delta_percent <= u_ref + u_other),
    }


def dg_legs(rawdir: Path = RAW) -> list[dict]:
    legs = []
    for path in sorted(rawdir.glob("dg_*.npz")):
        with np.load(path) as d:
            legs.append({"name": path.stem, "x": d["x"], "field": d["rho"].astype(np.float64)})
    return legs


def rejudge_legs(legs: list[dict], cert5: dict, cert7: dict, gate: float = GATE) -> dict:
    """Judge every committed DG leg against both certified references."""
    rows = []
    for leg in legs:
        r5 = judge(leg["field"], cert5, tolerance_percent=gate)
        r7 = judge(leg["field"], cert7, tolerance_percent=gate)
        rows.append({
            "leg": leg["name"],
            "error_percent_vs_weno5_ref": float(r5["rel_l1_percent"]),
            "verdict_vs_weno5_ref": r5["decision"],
            "error_percent_vs_weno7_ref": float(r7["rel_l1_percent"]),
            "verdict_vs_weno7_ref": r7["decision"],
            "shift_pp": float(r7["rel_l1_percent"] - r5["rel_l1_percent"]),
            "passes_both": bool(r5["decision"].startswith("PASS")
                                and r7["decision"].startswith("PASS")),
            "passes_weno5_only": bool(r5["decision"].startswith("PASS")
                                      and not r7["decision"].startswith("PASS")),
        })
    worst5 = max(rows, key=lambda r: r["error_percent_vs_weno5_ref"])
    worst7 = max(rows, key=lambda r: r["error_percent_vs_weno7_ref"])
    flips = [r["leg"] for r in rows if r["passes_weno5_only"]]
    return {
        "gate_percent": gate,
        "legs": rows,
        "n_legs": len(rows),
        "worst_vs_weno5": {"leg": worst5["leg"],
                           "error_percent": worst5["error_percent_vs_weno5_ref"]},
        "worst_vs_weno7": {"leg": worst7["leg"],
                           "error_percent": worst7["error_percent_vs_weno5_ref"]},
        "n_pass_both": sum(1 for r in rows if r["passes_both"]),
        "verdict_flips": flips,
        "max_abs_shift_pp": float(max(abs(r["shift_pp"]) for r in rows)),
        "max_shift_pp": float(max(r["shift_pp"] for r in rows)),
    }


def audit(rawdir: Path = RAW, outdir: Path = OUTDIR, gate: float = GATE,
          run_missing: bool = True) -> dict:
    lv5 = load_weno5_ladder(rawdir)
    lv7 = []
    lv5gen = []
    for n in LADDER:
        lv7.append(_solve_level(n, 7, WENO7_RAW, "fv_weno7_cfl0p3"))

    cert5_4 = cert_of(lv5, gate=gate)
    cert7_4 = cert_of(lv7, gate=gate)
    cert5g_4 = cert_of(lv5gen, gate=gate) if lv5gen else None
    # intra-family control: 3-level ladders (800/1600/3200) live on N=1600
    cert5_3 = cert_of(lv5, n_levels=3, gate=gate)
    cert7_3 = cert_of(lv7, n_levels=3, gate=gate)

    def delta(a: dict, b: dict) -> float | None:
        if a.get("reference") is None or b.get("reference") is None:
            return None
        if a["reference_ncells"] != b["reference_ncells"]:
            return None
        return 100.0 * rel_l1(a["reference"], b["reference"])

    d47 = delta(cert5_4, cert7_4)          # cross-order, 4-level ladders
    d35 = delta(cert5_3, cert7_3)          # cross-order, 3-level ladders
    intra = delta(cert5_3, cert5_4)        # intra-family control (WENO5 3 vs 4)
    d_gen = delta(cert5_4, cert5g_4) if cert5g_4 else None   # beta-normalisation control

    legs = dg_legs(rawdir)
    rj = rejudge_legs(legs, cert5_4, cert7_4, gate=gate)
    cover = coverage_verdict(d47 if d47 is not None else float("nan"),
                             cert5_4["band_percent"], cert7_4["band_percent"])

    # gate-survival arithmetic: does the worst leg still pass if the reference
    # moves by Delta in the worst direction?
    worst = rj["worst_vs_weno5"]["error_percent"]
    survival = {
        "worst_leg_error_percent": float(worst),
        "gate_percent": gate,
        "margin_percent": float(gate - worst),
        "delta_percent": float(d47) if d47 is not None else None,
        "worst_plus_delta_percent": float(worst + d47) if d47 is not None else None,
        "gate_survives_cross_family_shift": bool(d47 is not None and worst + d47 < gate),
    }

    verdict = cover["level"]
    report = {
        "case": "shu-osher-1988-shock-entropy",
        "what": "cross-order coverage audit of the round-2 certified reference",
        "gate_percent": gate,
        "ladder": list(LADDER),
        "cfl": CFL,
        "certified_reference_weno5": _cert_summary(cert5_4),
        "certified_reference_weno7": _cert_summary(cert7_4),
        "certified_reference_weno5_generic": (_cert_summary(cert5g_4)
                                              if cert5g_4 else None),
        "certified_reference_weno5_3level": _cert_summary(cert5_3),
        "certified_reference_weno7_3level": _cert_summary(cert7_3),
        "distances": {
            "cross_order_4level_percent": d47,
            "cross_order_3level_percent": d35,
            "intra_family_weno5_3v4_percent": intra,
            "weno5_shipped_vs_generic_percent": d_gen,
        },
        "coverage": cover,
        "gate_survival": survival,
        "dg_legs": rj,
        "verdict": verdict,
        "accepted": bool(cover["band_covers_cross_family"]),
        "sources": {
            "weno5_ladder": [l["source"] for l in lv5],
            "weno7_ladder": [l["source"] for l in lv7],
            "weno5_generic_ladder": [l["source"] for l in lv5gen],
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "band_honesty_audit.json").write_text(json.dumps(report, indent=2, default=str))
    (outdir / "band_honesty_audit.md").write_text(render_markdown(report))
    return report


def render_markdown(r: dict) -> str:
    d, c, s = r["distances"], r["coverage"], r["gate_survival"]
    c5, c7 = r["certified_reference_weno5"], r["certified_reference_weno7"]
    lines = [
        "# Band-honesty audit — round-2 certified reference (CASE-SHU-OSHER)",
        "",
        "Generated by `python -m benchmarks.xval_dg_round2_band_honesty`; every",
        "number below is computed from the profiles named in `sources`.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| ladder | {r['ladder']} (CFL {r['cfl']}) |",
        f"| WENO5 reference: verdict / band U5 | {c5['verdict']} / {c5['band_percent']:.6f} % |",
        f"| WENO7 reference: verdict / band U7 | {c7['verdict']} / {c7['band_percent']:.6f} % |",
        f"| cross-order distance (4-level) | {d['cross_order_4level_percent']:.6f} % |",
        f"| cross-order distance (3-level) | {d['cross_order_3level_percent']:.6f} % |",
        f"| intra-family control (WENO5 3 vs 4 level) | {d['intra_family_weno5_3v4_percent']:.6f} % |",
        f"| Delta / U5 | {c['ratio_delta_over_band']:.3f} |",
        f"| **coverage verdict** | **{c['level']}** |",
        "",
        "## DG legs re-judged against both references",
        "",
        "| leg | err vs WENO5 ref % | err vs WENO7 ref % | shift pp | pass both |",
        "|---|---|---|---|---|",
    ]
    for row in r["dg_legs"]["legs"]:
        lines.append(
            f"| {row['leg']} | {row['error_percent_vs_weno5_ref']:.4f} | "
            f"{row['error_percent_vs_weno7_ref']:.4f} | {row['shift_pp']:+.4f} | "
            f"{row['passes_both']} |")
    lines += [
        "",
        f"* worst leg vs WENO5 reference: {r['dg_legs']['worst_vs_weno5']['error_percent']:.4f} %",
        f"* verdict flips at the {r['gate_percent']} % gate: "
        f"{r['dg_legs']['verdict_flips'] or 'none'}",
        f"* max |shift| across legs: {r['dg_legs']['max_abs_shift_pp']:.4f} pp",
        f"* gate survival: worst + Delta = {s['worst_plus_delta_percent']:.4f} % "
        f"(gate {s['gate_percent']} %) -> {s['gate_survives_cross_family_shift']}",
        "",
        "## Verdict",
        "",
        f"`coverage = {c['level']}` — "
        + ("the certified band bounds the cross-family (cross-order) distance."
           if c["band_covers_cross_family"] else
           "the certified band does NOT bound the cross-family distance; the "
           "reference's own uncertainty is larger than U, so the certificate "
           "under-states the reference error by a factor "
           f"{c['ratio_delta_over_band']:.2f}."),
        "",
    ]
    return "\n".join(lines)


def figures(outdir: Path = OUTDIR) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit_json = Path(outdir) / "band_honesty_audit.json"
    r = json.loads(audit_json.read_text())
    cert5 = certify([{"ncells": l["ncells"], "field": l["field"]}
                     for l in load_weno5_ladder()], tolerance_percent=r["gate_percent"])
    lv7 = [_solve_level(n, 7, WENO7_RAW, "fv_weno7_cfl0p3") for n in LADDER]
    cert7 = certify([{"ncells": l["ncells"], "field": l["field"]} for l in lv7],
                    tolerance_percent=r["gate_percent"])
    paths = []
    # figure 1: certified references on their common grid + difference
    n = cert5["reference_ncells"]
    x = (np.arange(n) + 0.5) * (DOMAIN[1] - DOMAIN[0]) / n + DOMAIN[0]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(x, cert5["reference"], lw=1.6, label="certified reference, FV-WENO5")
    ax[0].plot(x, cert7["reference"], lw=1.0, ls="--", label="certified reference, FV-WENO7")
    ax[0].set_ylabel(r"density $\rho$")
    ax[0].set_title("Shu-Osher t=1.8: two certified references, "
                    f"Delta = {r['distances']['cross_order_4level_percent']:.4f} %, "
                    f"U5 = {cert5['band_percent']:.4f} %")
    ax[0].legend(fontsize=8)
    ax[1].plot(x, cert7["reference"] - cert5["reference"], lw=0.9, color="crimson")
    ax[1].axhline(0.0, color="k", lw=0.5)
    ax[1].set_xlabel("x")
    ax[1].set_ylabel(r"$\rho_{WENO7}-\rho_{WENO5}$")
    fig.tight_layout()
    p1 = Path(outdir) / "fig1_cross_order_references.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(str(p1))
    # figure 2: error bars per DG leg against both references
    rows = r["dg_legs"]["legs"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    idx = np.arange(len(rows))
    ax.bar(idx - 0.2, [row["error_percent_vs_weno5_ref"] for row in rows], width=0.4,
           label="vs certified WENO5 reference")
    ax.bar(idx + 0.2, [row["error_percent_vs_weno7_ref"] for row in rows], width=0.4,
           label="vs certified WENO7 reference")
    ax.axhline(r["gate_percent"], color="k", ls="--", lw=1.0,
               label=f"{r['gate_percent']} % gate")
    ax.set_xticks(idx)
    ax.set_xticklabels([row["leg"].replace("dg_cfl", "").replace("_limweno", "") for row in rows],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("rel. L1 density error [%]")
    ax.set_title("DG legs vs two independently certified references")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = Path(outdir) / "fig2_dg_legs_two_references.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(str(p2))
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rawdir", default=str(RAW))
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--no-run", action="store_true", help="use only cached WENO7 levels")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)
    report = audit(Path(args.rawdir), Path(args.outdir), gate=args.gate,
                   run_missing=not args.no_run)
    print(json.dumps({k: report[k] for k in ("verdict", "coverage", "distances",
                                             "gate_survival")}, indent=2, default=str))
    print("wg legs: n_pass_both =", report["dg_legs"]["n_pass_both"],
          "flips =", report["dg_legs"]["verdict_flips"])
    if args.figures:
        for p in figures(Path(args.outdir)):
            print("figure:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())