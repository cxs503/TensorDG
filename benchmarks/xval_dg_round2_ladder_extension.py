"""Ladder-extension audit of the round-2 certified reference (three levels -> four).

What question this answers
--------------------------
``XVAL-DG-ROUND2-CERTIFIED-REFERENCE`` certifies the Shu--Osher reference from a
three-level FV ladder ``[800, 1600, 3200]``: the observed order comes from a
*single* successive difference ``d(800->1600) / d(1600->3200)`` and Roache's band
``U = Fs d_fine / (r^p - 1)`` inherits that one order estimate.  ``Ds
CASE-SHU-OSHER-round2.md`` §6 recorded the honest gap: a fourth level
(``N = 6400``) was submitted to the cluster and never dispatched, so the
asymptotic-range assumption behind ``p`` had no in-ladder evidence and the band
had never been confronted with a finer ladder.

**The band is a claim, and this module tests it.**  Extending the ladder gives a
*second* successive difference (order consistency) and a *newer* certified
profile.  If the new profile moves away from the old one by more than the old
band ``U``, the three-level band under-reports its own uncertainty -- an honest
FAIL that must be published, because every downstream "leg error < 3 %" verdict
is then compared against a reference whose stated noise is too small.

Predeclared criteria (fixed before the fourth level was generated)
-----------------------------------------------------------------
``L0``  successive differences strictly decrease across the four levels
        (``d(800->1600) > d(1600->3200) > d(3200->6400)``): the ladder is in an
        asymptotic range at all, which two differences cannot show.
``L1``  the extended ladder is ``CERTIFIED`` by the *same* engine call, same
        ``Fs``, same tolerance, no parameter changed.
``L2``  order consistency: ``|p4 - p3| / p3 <= 0.20`` where ``p3`` is the engine's
        order on ``[800,1600,3200]`` and ``p4`` its order on ``[1600,3200,6400]``.
``L3``  **band coverage (self-consistency)**: ``rel L1(cert_3level, cert_4level) <= U3``
        measured on the shared ``N = 1600`` grid.  ``U3`` is the three-level band,
        so this is the band's own falsifier.
``L4``  claim stability: (a) all eleven shipped DG legs still pass the strict
        ``< 3 %`` gate against the *four-level* reference; (b) no leg moves by more
        than ``0.5`` percentage points when the reference is upgraded -- otherwise
        the verdict would be a property of the ladder, not of the solver.

``accepted`` is the conjunction of ``L0``-``L4``; the CLI exits non-zero when it
is false, so a red run cannot be mistaken for a green one.  Nothing here is
tuned after the fact: the thresholds above are module constants and the only
inputs are committed raw profiles.

Usage::

    python -m benchmarks.xval_dg_round2_ladder_extension            # audit + artifacts
    python -m benchmarks.xval_dg_round2_ladder_extension --figures  # + 2 PNGs
    python -m benchmarks.xval_dg_round2_ladder_extension --json-stdout
    python -m benchmarks.xval_dg_round2_ladder_extension --require-accepted
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow ``python benchmarks/xval_dg_round2_ladder_extension.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordg.certified_reference import CERTIFIED, certify, coarsen, judge, rel_l1
from benchmarks import xval_dg_round2_certified as xval

__all__ = ["BASE_LEVELS", "EXT_LEVEL", "DEFAULT_RAW", "DEFAULT_OUT", "load_fv_level",
           "successive_differences", "audit", "write_artifacts", "figures", "main"]

CASE = ("Shu-Osher 1988 shock/entropy interaction (1D Euler, gamma=1.4, x in [-5,5], "
        "t=1.8, Sod-style Hugoniot left state, transmissive ends)")
BASE_LEVELS = (800, 1600, 3200)
EXT_LEVEL = 6400
GATE_PERCENT = xval.GATE_PERCENT            # 5 %: the gate the reference is used at
STRICT_PERCENT = xval.STRICT_PERCENT        # 3 %: owner precision standard
DEFAULT_RAW = xval.DEFAULT_RAW
DEFAULT_OUT = "validation/xval_dg_round2/artifacts"

#: Predeclared thresholds -- module constants, never recomputed from results.
ORDER_CONSISTENCY_MAX = 0.20     # L2, relative
LEG_STABILITY_MAX_PP = 0.5       # L4(b), percentage points


# --------------------------------------------------------------------- inputs
def level_path(rawdir: Path | str, ncells: int) -> Path:
    return Path(rawdir) / f"fv_cfl0p3_n{int(ncells)}.npz"


def load_fv_level(rawdir: Path | str, ncells: int) -> dict:
    """Load one committed FV-WENO5 level as ``{ncells, x, field}``."""
    path = level_path(rawdir, ncells)
    if not path.exists():
        raise FileNotFoundError(
            f"missing reference level {path}; generate it with "
            f"`python validation/xval_dg_round2/gen_fv_level.py --ncells {ncells} --out {path}`")
    blob = np.load(path, allow_pickle=False)
    x = np.asarray(blob["x"], dtype=float)
    rho = np.asarray(blob["rho"], dtype=float)
    if x.size != int(ncells) or rho.size != x.size:
        raise ValueError(f"{path}: expected {ncells} cells, got x={x.size} rho={rho.size}")
    if not np.all(np.isfinite(rho)) or rho.min() <= 0.0:
        raise ValueError(f"{path}: non-admissible density profile")
    return {"ncells": int(ncells), "x": x, "field": rho}


def ladders(rawdir: Path | str = DEFAULT_RAW) -> dict:
    levels = [load_fv_level(rawdir, n) for n in (*BASE_LEVELS, EXT_LEVEL)]
    return {"three": levels[:-1], "four": levels,
            "rawdir": str(rawdir), "levels": [lv["ncells"] for lv in levels]}


def successive_differences(levels: list[dict]) -> list[dict]:
    """Cell-average every level to the coarsest grid, then successive rel L1."""
    base = levels[0]["ncells"]
    prof = [coarsen(lv["field"], lv["ncells"] // base) for lv in levels]
    rows = []
    for i in range(len(prof) - 1):
        rows.append({
            "levels": [levels[i]["ncells"], levels[i + 1]["ncells"]],
            "rel_l1": float(rel_l1(prof[i], prof[i + 1])),
            "rel_l1_percent": 100.0 * float(rel_l1(prof[i], prof[i + 1])),
        })
    return rows


def leg_table(cert: dict, rawdir: Path | str = DEFAULT_RAW) -> list[dict]:
    """Every shipped DG leg judged against one certification."""
    rows = []
    for degree, ncells, cfl in xval.DG_LEGS:
        blob = np.load(Path(rawdir) / xval.leg_filename(degree, ncells, cfl),
                       allow_pickle=False)
        field = np.asarray(blob["rho"], dtype=float)
        res5 = judge(field, cert, tolerance_percent=GATE_PERCENT)
        res3 = judge(field, cert, tolerance_percent=STRICT_PERCENT)
        rows.append({
            "configuration": f"DG P{degree} N={ncells} CFL={cfl}",
            "degree": int(degree), "ncells": int(ncells), "cfl": float(cfl),
            "rel_l1_percent": res5["rel_l1_percent"],
            "rel_l2_percent": res5["rel_l2_percent"],
            "decision_5pct": res5["decision"],
            "decision_3pct": res3["decision"],
            "pass_strict": bool(res5["rel_l1_percent"] < STRICT_PERCENT),
        })
    return rows


# ---------------------------------------------------------------------- audit
def audit(rawdir: Path | str = DEFAULT_RAW) -> dict:
    """Run L0-L4 and return the whole audit (deterministic, no side effects)."""
    bun = ladders(rawdir)
    cert3 = certify(bun["three"], tolerance_percent=GATE_PERCENT)
    cert4 = certify(bun["four"], tolerance_percent=GATE_PERCENT)

    diffs = successive_differences(bun["four"])

    l0 = {
        "criterion": "successive differences strictly decrease over the four levels",
        "rows": diffs,
        "monotone": bool(all(diffs[i]["rel_l1"] > diffs[i + 1]["rel_l1"]
                             for i in range(len(diffs) - 1))),
    }
    l0["pass"] = bool(l0["monotone"])

    p3 = cert3.get("observed_order_used")
    p4 = cert4.get("observed_order_used")
    order_change = (None if p3 in (None, 0) or p4 is None else abs(p4 - p3) / abs(p3))
    l2 = {
        "criterion": f"|p4-p3|/p3 <= {ORDER_CONSISTENCY_MAX}",
        "p_three_levels": p3, "p_four_levels": p4,
        "relative_change": order_change,
        "threshold": ORDER_CONSISTENCY_MAX,
    }
    l2["pass"] = bool(order_change is not None and order_change <= ORDER_CONSISTENCY_MAX)

    ref3, ref4 = cert3.get("reference"), cert4.get("reference")
    shift = None
    if ref3 is not None and ref4 is not None:
        r = int(round(cert4["refinement_fine"]))
        ref4_on_ref3 = coarsen(ref4, r)
        if ref4_on_ref3.size != ref3.size:
            raise ValueError("the two certified profiles do not share a nesting grid")
        shift = float(rel_l1(ref4_on_ref3, ref3))
    band3 = cert3.get("band_fraction")
    l3 = {
        "criterion": "rel L1(cert_3level, cert_4level) <= U3",
        "shift_rel_l1": shift,
        "shift_percent": None if shift is None else 100.0 * shift,
        "band_fraction_U3": band3,
        "band_percent_U3": cert3.get("band_percent"),
        "coverage_ratio": (None if shift is None or not band3 else shift / band3),
        "grid": cert3.get("reference_ncells"),
    }
    l3["pass"] = bool(shift is not None and band3 not in (None, 0.0) and shift <= band3)

    legs3 = leg_table(cert3, rawdir)
    legs4 = leg_table(cert4, rawdir)
    moves = []
    for a, b in zip(legs3, legs4):
        moves.append({"configuration": a["configuration"],
                      "rel_l1_percent_3level": a["rel_l1_percent"],
                      "rel_l1_percent_4level": b["rel_l1_percent"],
                      "delta_pp": b["rel_l1_percent"] - a["rel_l1_percent"],
                      "pass_strict_4level": b["pass_strict"]})
    worst_move = max((abs(m["delta_pp"]) for m in moves), default=None)
    worst_leg4 = max(legs4, key=lambda r: r["rel_l1_percent"])
    l4 = {
        "criterion": (f"every leg < {STRICT_PERCENT} % against the 4-level reference; "
                      f"no leg moves by more than {LEG_STABILITY_MAX_PP} pp"),
        "legs_against_four_level": legs4,
        "legs_against_three_level": legs3,
        "moves": moves,
        "worst_leg_4level": worst_leg4["configuration"],
        "worst_rel_l1_percent_4level": worst_leg4["rel_l1_percent"],
        "max_abs_move_pp": worst_move,
        "all_legs_pass_strict": bool(all(r["pass_strict"] for r in legs4)),
    }
    l4["pass"] = bool(l4["all_legs_pass_strict"] and worst_move is not None
                      and worst_move <= LEG_STABILITY_MAX_PP)

    l1 = {
        "criterion": "the extended ladder is CERTIFIED by the unchanged engine call",
        "verdict": cert4.get("verdict"),
        "classification": cert4.get("classification"),
        "band_percent": cert4.get("band_percent"),
        "band_share_of_gate": cert4.get("band_share_of_gate"),
        "reference_ncells": cert4.get("reference_ncells"),
    }
    l1["pass"] = bool(cert4.get("verdict") == CERTIFIED)

    decisions = {"L0": l0["pass"], "L1": l1["pass"], "L2": l2["pass"],
                 "L3": l3["pass"], "L4": l4["pass"]}
    return {
        "case": CASE,
        "benchmark": {"benchmark_id": "XVAL-DG-ROUND2-CERTIFIED-REFERENCE",
                      "version": "3.0.0"},
        "workload": {"workload_id": "xval-dg-round2-ladder-extension", "version": "1"},
        "rawdir": str(rawdir),
        "ladder": {"three_levels": list(BASE_LEVELS), "four_levels": [*BASE_LEVELS, EXT_LEVEL],
                   "refinement": 2, "scheme": "FV-WENO5-JS/HLLC(Davis)/SSP-RK3, CFL=0.3"},
        "thresholds": {"gate_percent": GATE_PERCENT, "strict_percent": STRICT_PERCENT,
                       "order_consistency_max": ORDER_CONSISTENCY_MAX,
                       "leg_stability_max_pp": LEG_STABILITY_MAX_PP},
        "L0_successive_differences": l0,
        "L1_extension_certified": l1,
        "L2_order_consistency": l2,
        "L3_band_covers_shift": l3,
        "L4_claim_stable": l4,
        "certification_three_levels": {k: v for k, v in cert3.items()
                                       if k not in ("reference", "field_raw_finest")},
        "certification_four_levels": {k: v for k, v in cert4.items()
                                      if k not in ("reference", "field_raw_finest")},
        "decisions": decisions,
        "accepted": bool(all(decisions.values())),
    }


# ------------------------------------------------------------------ artifacts
def write_artifacts(out: dict, outdir: Path | str = DEFAULT_OUT) -> dict:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    js = d / "ladder_extension.json"
    md = d / "ladder_extension.md"
    js.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    md.write_text(_markdown(out))
    return {"json": str(js), "markdown": str(md)}


def _markdown(out: dict) -> str:
    l0, l1, l2 = out["L0_successive_differences"], out["L1_extension_certified"], out["L2_order_consistency"]
    l3, l4 = out["L3_band_covers_shift"], out["L4_claim_stable"]
    lines = [
        "# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — ladder extension (3 -> 4 levels)",
        "",
        f"Generated by `python -m benchmarks.xval_dg_round2_ladder_extension` from the committed",
        f"raw profiles (`{out['rawdir']}`); no number in this file is hand-typed.",
        "",
        f"* case: {out['case']}",
        f"* ladder: {out['ladder']['four_levels']} (r = {out['ladder']['refinement']}), "
        f"{out['ladder']['scheme']}",
        "",
        "## L0 successive differences",
        "",
        "| levels | rel L1 % |", "|---|---:|",
    ]
    for row in l0["rows"]:
        lines.append(f"| {row['levels'][0]} -> {row['levels'][1]} | {row['rel_l1_percent']:.6f} |")
    lines += ["", f"monotone decreasing: **{l0['monotone']}**", "",
              "## L1 extended ladder certified", "",
              f"* verdict: `{l1['verdict']}` (classification `{l1['classification']}`)",
              f"* band U: {l1['band_percent']:.6f} % = {100.0 * l1['band_share_of_gate']:.2f} % of the "
              f"{out['thresholds']['gate_percent']:g} % gate",
              f"* reference grid: N = {l1['reference_ncells']}", "",
              "## L2 order consistency", "",
              f"| ladder | observed order p |", "|---|---:|",
              f"| [800, 1600, 3200] | {l2['p_three_levels']:.6f} |",
              f"| [1600, 3200, 6400] | {l2['p_four_levels']:.6f} |",
              "", f"relative change = {l2['relative_change']:.4f} "
              f"(threshold {l2['threshold']})", "",
              "## L3 does the three-level band cover the shift?", "",
              f"| quantity | value |", "|---|---:|",
              f"| rel L1(cert3, cert4) on N = {l3['grid']} | {l3['shift_percent']:.6f} % |",
              f"| U3 (three-level band) | {l3['band_percent_U3']:.6f} % |",
              f"| coverage ratio shift/U3 | {l3['coverage_ratio']:.4f} |",
              "", f"covered: **{l3['pass']}**", "",
              "## L4 claim stability (eleven DG legs)", "",
              "| leg | rel L1 % (3-level ref) | rel L1 % (4-level ref) | delta pp | < 3 % |",
              "|---|---:|---:|---:|:--:|"]
    for m in l4["moves"]:
        lines.append(f"| {m['configuration']} | {m['rel_l1_percent_3level']:.6f} | "
                     f"{m['rel_l1_percent_4level']:.6f} | {m['delta_pp']:+.6f} | "
                     f"{'PASS' if m['pass_strict_4level'] else 'FAIL'} |")
    lines += ["", f"worst leg against the 4-level reference: {l4['worst_leg_4level']} "
              f"= {l4['worst_rel_l1_percent_4level']:.6f} %; "
              f"max |delta| = {l4['max_abs_move_pp']:.6f} pp", "",
              "## Decision", "",
              "| gate | pass |", "|---|:--:|"]
    for key, val in out["decisions"].items():
        lines.append(f"| {key} | {'PASS' if val else 'FAIL'} |")
    lines += ["", f"accepted = **{out['accepted']}**", ""]
    return "\n".join(lines)


def figures(out: dict, rawdir: Path | str = DEFAULT_RAW,
            outdir: Path | str = DEFAULT_OUT) -> dict:
    """Two PNGs: the two certified profiles with their bands, and leg stability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    bun = ladders(rawdir)
    cert3 = certify(bun["three"], tolerance_percent=GATE_PERCENT)
    cert4 = certify(bun["four"], tolerance_percent=GATE_PERCENT)
    ref3 = cert3["reference"]
    ref4 = coarsen(cert4["reference"], int(round(cert4["refinement_fine"])))
    x = bun["three"][1]["x"]  # the shared N = 1600 grid

    fig1 = d / "fig_ladder_extension_profiles.png"
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    band3 = cert3["band_fraction"]
    ax.fill_between(x, ref3 * (1 - band3), ref3 * (1 + band3), color="0.75", alpha=0.7,
                    label=f"3-level band U = {cert3['band_percent']:.4f} %")
    ax.plot(x, ref3, "k-", lw=1.3, label="certified ref, 3 levels [800,1600,3200]")
    ax.plot(x, ref4, "--", color="C0", lw=1.3, label="certified ref, 4 levels [1600,3200,6400]")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title("Shu-Osher t=1.8: does the three-level band cover the ladder-extension shift?")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-5, 5)
    delta = ref4 - ref3
    ax2.plot(x, 100.0 * delta / np.maximum(ref3, 1e-12), "-", color="C3", lw=1.0,
             label="relative shift (cert4 - cert3)")
    ax2.axhline(100.0 * band3, color="0.4", ls=":", lw=1.0, label="+U3")
    ax2.axhline(-100.0 * band3, color="0.4", ls=":", lw=1.0, label="-U3")
    ax2.set_xlabel("x")
    ax2.set_ylabel("shift / %")
    ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig1, dpi=150)
    plt.close(fig)

    fig2 = d / "fig_ladder_extension_legs.png"
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    idx = np.arange(len(out["L4_claim_stable"]["moves"]))
    a = [m["rel_l1_percent_3level"] for m in out["L4_claim_stable"]["moves"]]
    b = [m["rel_l1_percent_4level"] for m in out["L4_claim_stable"]["moves"]]
    ax.plot(idx, a, "o-", label="vs 3-level reference")
    ax.plot(idx, b, "s--", label="vs 4-level reference")
    ax.axhline(STRICT_PERCENT, color="C3", ls=":", label="3 % strict gate")
    ax.set_xticks(idx)
    ax.set_xticklabels([m["configuration"].replace("DG ", "") for m in
                        out["L4_claim_stable"]["moves"]], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("rel L1 / %")
    ax.set_title("Eleven DG legs against both certified references (verdict stability)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig2, dpi=150)
    plt.close(fig)
    return {"fig1": str(fig1), "fig2": str(fig2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rawdir", default=DEFAULT_RAW)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--json-stdout", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--require-accepted", action="store_true")
    args = ap.parse_args(argv)

    out = audit(args.rawdir)
    if args.json_stdout:
        print(json.dumps({k: v for k, v in out.items()
                          if k not in ("certification_three_levels",
                                       "certification_four_levels")},
                         indent=2, sort_keys=True, default=str))
    else:
        print(_markdown(out))
    if not args.no_write:
        paths = write_artifacts(out, args.outdir)
        print(f"wrote {paths['json']} and {paths['markdown']}", file=sys.stderr)
        if args.figures:
            figs = figures(out, args.rawdir, args.outdir)
            print(f"wrote {figs['fig1']} and {figs['fig2']}", file=sys.stderr)
    if args.require_accepted and not out["accepted"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())