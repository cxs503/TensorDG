"""Independent (engine-free) recompute of the ladder-extension gates L0-L4.

Why this exists
---------------
``benchmarks/xval_dg_round2_ladder_extension.py`` decides the predeclared gates
``L0``-``L4`` by calling the shipped engine ``tensordg.certified_reference``.
That is the right way to answer *"does the practice still certify a four-level
ladder?"*, but it is a single code path: if the engine has a systematic bug
(cell-average convention, denominator of the relative L1, the ``r^p - 1``
denominator, the choice of which raw level *is* the reference) then the audit
and the engine agree on the same wrong number.

This module recomputes every gate from the committed raw profiles with **its
own** arithmetic -- no import of ``tensordg.certified_reference``, no reuse of
its helpers -- and then reports the deviation from the engine artifact.  Its job
is to disagree if it can.  The independent numbers are also what the platform
result row is posted with, so the machine judge compares against arithmetic that
never touched the engine.

Definitions are taken from the predeclared protocol
(``artifacts/ladder_extension_protocol.json``) and from the cited methodology
(Roache GCI, ``Fs = 1.25``), not from the engine source.

Usage::

    python validation/xval_dg_round2/close_ladder_extension.py            # audit + artifacts
    python validation/xval_dg_round2/close_ladder_extension.py --json-stdout
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "validation" / "repro_shu_osher" / "artifacts" / "raw"
OUT = ROOT / "validation" / "xval_dg_round2" / "artifacts"
ENGINE_ARTIFACT = OUT / "ladder_extension.json"

BASE_LEVELS = (800, 1600, 3200)
EXT_LEVEL = 6400
FS_GCI = 1.25          # Roache (1994), the factor the methodology cites
REFINEMENT = 2
GATE_PERCENT = 5.0
STRICT_PERCENT = 3.0
ORDER_CONSISTENCY_MAX = 0.20
LEG_STABILITY_MAX_PP = 0.5
ROUNDOFF_FLOOR = 1e-10  # below this a difference carries no order information

#: the eleven shipped DG legs (P1/P2/P3 x N=200/400/800 x CFL=0.4, plus two CFL
#: sensitivity legs at P2 N=400) -- the same set the round-2 audit judges
DG_LEGS = [(1, 200, 0.4), (2, 200, 0.4), (3, 200, 0.4),
           (1, 400, 0.4), (2, 400, 0.4), (3, 400, 0.4),
           (1, 800, 0.4), (2, 800, 0.4), (3, 800, 0.4),
           (2, 400, 0.1), (2, 400, 0.2)]


# ------------------------------------------------------------------ own math
def cell_average(field, factor: int) -> np.ndarray:
    """Average ``factor`` contiguous cells -- plain column-mean reshape."""
    v = np.asarray(field, dtype=float)
    factor = int(factor)
    if factor == 1:
        return v.copy()
    if factor <= 0 or v.size % factor:
        raise ValueError(f"size {v.size} not divisible by {factor}")
    return v.reshape(-1, factor).mean(axis=1)


def rel_l1(a, b) -> float:
    """sum|a-b| / sum|b| (fraction, not percent)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.abs(a - b).sum() / np.abs(b).sum())


def order_log2(d_coarse: float, d_fine: float, r: float = REFINEMENT) -> float | None:
    """p from a pair of successive differences: p = ln(d_c/d_f)/ln(r)."""
    if d_coarse <= ROUNDOFF_FLOOR or d_fine <= ROUNDOFF_FLOOR:
        return None
    return float(math.log(d_coarse / d_fine) / math.log(r))


def roache_band_percent(d_fine_percent: float, p: float | None,
                        fs: float = FS_GCI, r: float = REFINEMENT) -> float | None:
    """GCI band U = Fs |d_fine| / (r^p - 1), in percent."""
    if p is None:
        return None
    return float(fs * abs(d_fine_percent) / (r ** p - 1.0))


# --------------------------------------------------------------------- inputs
#: the fourth level is kept out of the globbed round-2 raw directory (see
#: benchmarks/xval_dg_round2_ladder_extension.EXT_RAWDIR for the measurement)
EXT_RAWDIR = RAW.parent / "raw_ext"


def level_path(ncells: int) -> Path:
    path = RAW / f"fv_cfl0p3_n{int(ncells)}.npz"
    if path.exists():
        return path
    alt = EXT_RAWDIR / f"fv_cfl0p3_n{int(ncells)}.npz"
    return alt if alt.exists() else path


def load_level(ncells: int) -> np.ndarray:
    path = level_path(ncells)
    if not path.exists():
        raise FileNotFoundError(
            f"missing reference level {path}; generate it with "
            f"`python validation/xval_dg_round2/gen_fv_level.py --ncells {ncells} --out {path}`")
    rho = np.asarray(np.load(path, allow_pickle=False)["rho"], dtype=float)
    if rho.size != int(ncells):
        raise ValueError(f"{path}: expected {ncells} cells, got {rho.size}")
    if not np.all(np.isfinite(rho)) or rho.min() <= 0.0:
        raise ValueError(f"{path}: non-admissible density profile")
    return rho


def load_leg(degree: int, ncells: int, cfl: float) -> np.ndarray:
    tag = f"{cfl:g}".replace(".", "p")          # 0.4 -> "0p4", the shipped naming
    path = RAW / f"dg_cfl{tag}_limweno_n{ncells}_p{degree}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing DG leg {path}")
    return np.asarray(np.load(path, allow_pickle=False)["rho"], dtype=float)


# ---------------------------------------------------------------------- audit
def audit() -> dict:
    lv = {n: load_level(n) for n in (*BASE_LEVELS, EXT_LEVEL)}
    base = BASE_LEVELS[0]
    coarsened = {n: cell_average(lv[n], n // base) for n in lv}

    # ---- L0: successive differences, and how many are available
    d = {}
    for lo, hi in ((800, 1600), (1600, 3200), (3200, 6400)):
        d[(lo, hi)] = rel_l1(coarsened[lo], coarsened[hi])
    d1, d2, d3 = d[(800, 1600)], d[(1600, 3200)], d[(3200, 6400)]
    l0 = {
        "criterion": "d(800->1600) > d(1600->3200) > d(3200->6400) (strict)",
        "differences": [{"levels": list(k), "rel_l1": v, "rel_l1_percent": 100.0 * v}
                        for k, v in d.items()],
        "strictly_decreasing": bool(d1 > d2 > d3),
        "pass": bool(d1 > d2 > d3),
    }

    # ---- L1: the extended ladder certifies (own classifier, same rules)
    p3 = order_log2(d1, d2)
    p4 = order_log2(d2, d3)
    u3 = roache_band_percent(100.0 * d2, p3)
    u4 = roache_band_percent(100.0 * d3, p4)
    asymptotic = bool(p4 is not None and 0.0 < p4 <= 8.0)
    band_share = None if u4 is None else (u4 / 100.0) / GATE_PERCENT
    l1 = {
        "criterion": "four-level ladder is asymptotic (0 < p4 <= 8) and its band "
                     "occupies <= 1/3 of the gate",
        "p_three_levels": p3, "p_four_levels": p4,
        "band_percent_U3": u3, "band_percent_U4": u4,
        "band_share_of_gate_U4": band_share,
        "reference_ncells": BASE_LEVELS[-1],
        "pass": bool(asymptotic and band_share is not None and band_share <= 1.0 / 3.0),
    }

    # ---- L2: order consistency
    rel_change = None if (p3 is None or p4 is None) else abs(p4 - p3) / abs(p3)
    l2 = {
        "criterion": f"|p4 - p3| / p3 <= {ORDER_CONSISTENCY_MAX}",
        "p_three_levels": p3, "p_four_levels": p4, "relative_change": rel_change,
        "threshold": ORDER_CONSISTENCY_MAX,
        "pass": bool(rel_change is not None and rel_change <= ORDER_CONSISTENCY_MAX),
    }

    # ---- L3: does U3 cover the shift of the certified profile?
    # 3-level reference lives on the second-finest grid of [800,1600,3200] = 1600;
    # 4-level reference lives on 3200 -> coarsen by r to the shared 1600 grid.
    # Richardson extrapolation with own arithmetic, expressed on the second-finest
    # grid (the engine's documented convention): the finer level coarsened onto that
    # grid, corrected by (fine - coarse) / (r^p - 1).
    c3200_on_1600 = cell_average(lv[3200], REFINEMENT)
    ref3 = c3200_on_1600 + (c3200_on_1600 - lv[1600]) / (REFINEMENT ** p3 - 1.0)
    c6400_on_3200 = cell_average(lv[6400], REFINEMENT)
    ref4_on_3200 = c6400_on_3200 + (c6400_on_3200 - lv[3200]) / (REFINEMENT ** p4 - 1.0)
    ref4 = cell_average(ref4_on_3200, REFINEMENT)
    shift = rel_l1(ref4, ref3)
    coverage = None if not u3 else shift / (u3 / 100.0)
    l3 = {
        "criterion": "rel L1(cert3, cert4) on the shared N=1600 grid <= U3",
        "shift_rel_l1": shift, "shift_percent": 100.0 * shift,
        "band_percent_U3": u3, "coverage_ratio": coverage,
        "grid": 1600,
        "pass": bool(u3 is not None and shift <= u3 / 100.0),
    }

    # ---- L4: the eleven DG legs against both references
    moves = []
    for degree, ncells, cfl in DG_LEGS:
        field = load_leg(degree, ncells, cfl)
        e3 = rel_l1(cell_average(ref3, 1600 // ncells), field)
        e4 = rel_l1(cell_average(ref4, 1600 // ncells), field)
        moves.append({
            "configuration": f"DG P{degree} N={ncells} CFL={cfl}",
            "rel_l1_percent_3level": 100.0 * e3,
            "rel_l1_percent_4level": 100.0 * e4,
            "delta_pp": 100.0 * (e4 - e3),
            "pass_strict_4level": bool(100.0 * e4 < STRICT_PERCENT),
        })
    worst = max(moves, key=lambda r: r["rel_l1_percent_4level"])
    max_move = max(abs(r["delta_pp"]) for r in moves)
    l4 = {
        "criterion": f"every leg < {STRICT_PERCENT} % against the 4-level reference and "
                     f"no leg moves by more than {LEG_STABILITY_MAX_PP} pp",
        "moves": moves,
        "worst_leg_4level": worst["configuration"],
        "worst_rel_l1_percent_4level": worst["rel_l1_percent_4level"],
        "max_abs_move_pp": max_move,
        "all_legs_pass_strict": bool(all(r["pass_strict_4level"] for r in moves)),
        "pass": bool(all(r["pass_strict_4level"] for r in moves)
                     and max_move <= LEG_STABILITY_MAX_PP),
    }

    decisions = {"L0": l0["pass"], "L1": l1["pass"], "L2": l2["pass"],
                 "L3": l3["pass"], "L4": l4["pass"]}
    return {
        "independent_of_engine": True,
        "case": ("Shu-Osher 1988 shock/entropy interaction (1D Euler, gamma=1.4, x in [-5,5], "
                 "t=1.8, Mach-3 Hugoniot left state, transmissive ends)"),
        "rawdir": str(RAW),
        "ladder": {"levels": [*BASE_LEVELS, EXT_LEVEL], "refinement": REFINEMENT,
                   "scheme": "FV-WENO5-JS/HLLC(Davis)/SSP-RK3, CFL=0.3"},
        "thresholds": {"gate_percent": GATE_PERCENT, "strict_percent": STRICT_PERCENT,
                       "order_consistency_max": ORDER_CONSISTENCY_MAX,
                       "leg_stability_max_pp": LEG_STABILITY_MAX_PP, "fs_gci": FS_GCI},
        "L0_successive_differences": l0,
        "L1_extension_certified": l1,
        "L2_order_consistency": l2,
        "L3_band_covers_shift": l3,
        "L4_claim_stable": l4,
        "decisions": decisions,
        "accepted": bool(all(decisions.values())),
    }


def compare_with_engine(indep: dict, artifact: Path = ENGINE_ARTIFACT) -> dict:
    """Report every quantity where the independent recompute and the engine differ."""
    if not artifact.exists():
        return {"status": "engine_artifact_missing", "path": str(artifact)}
    eng = json.loads(artifact.read_text())
    pairs = [
        ("L0.d1_percent", indep["L0_successive_differences"]["differences"][0]["rel_l1_percent"],
         eng["L0_successive_differences"]["rows"][0]["rel_l1_percent"]),
        ("L0.d2_percent", indep["L0_successive_differences"]["differences"][1]["rel_l1_percent"],
         eng["L0_successive_differences"]["rows"][1]["rel_l1_percent"]),
        ("L0.d3_percent", indep["L0_successive_differences"]["differences"][2]["rel_l1_percent"],
         eng["L0_successive_differences"]["rows"][2]["rel_l1_percent"]),
        ("L1.band_U4_percent", indep["L1_extension_certified"]["band_percent_U4"],
         eng["L1_extension_certified"]["band_percent"]),
        ("L2.p4", indep["L2_order_consistency"]["p_four_levels"],
         eng["L2_order_consistency"]["p_four_levels"]),
        ("L3.shift_percent", indep["L3_band_covers_shift"]["shift_percent"],
         eng["L3_band_covers_shift"]["shift_percent"]),
    ]
    rows = []
    for name, mine, theirs in pairs:
        if mine is None or theirs is None:
            rows.append({"quantity": name, "independent": mine, "engine": theirs,
                         "abs_diff": None, "rel_diff": None, "agree": False})
            continue
        ad = abs(mine - float(theirs))
        rd = ad / abs(float(theirs)) if float(theirs) else None
        rows.append({"quantity": name, "independent": mine, "engine": float(theirs),
                     "abs_diff": ad, "rel_diff": rd,
                     "agree": bool(rd is not None and rd <= 1e-9)})
    eng_legs = {m["configuration"]: m["rel_l1_percent_4level"]
                for m in eng["L4_claim_stable"]["moves"]}
    worst_leg_diff = max(abs(m["rel_l1_percent_4level"] - eng_legs[m["configuration"]])
                         for m in indep["L4_claim_stable"]["moves"])
    same_decisions = all(indep["decisions"][k] == bool(eng["decisions"][k])
                         for k in indep["decisions"])
    return {
        "status": "compared",
        "artifact": str(artifact),
        "rows": rows,
        "max_rel_diff": max((r["rel_diff"] or 0.0) for r in rows),
        "worst_leg_abs_diff_pp": worst_leg_diff,
        "decisions_identical": bool(same_decisions),
        "all_agree": bool(all(r["agree"] for r in rows) and worst_leg_diff <= 1e-9
                          and same_decisions),
    }


def write(out: dict, cmp: dict, outdir: Path = OUT) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    js = outdir / "ladder_extension_independent.json"
    md = outdir / "ladder_extension_independent.md"
    payload = {"independent": out, "cross_check_vs_engine": cmp}
    js.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    l0, l1 = out["L0_successive_differences"], out["L1_extension_certified"]
    l2, l3, l4 = out["L2_order_consistency"], out["L3_band_covers_shift"], out["L4_claim_stable"]
    lines = [
        "# Ladder extension (3 -> 4 levels): independent recompute",
        "",
        "Every number below is produced by `validation/xval_dg_round2/close_ladder_extension.py`",
        "from the committed raw profiles, **without importing the engine** "
        "(`tensordg.certified_reference`).  It exists to disagree with the engine if it can.",
        "",
        "## L0 successive differences",
        "", "| levels | rel L1 % |", "|---|---:|",
    ]
    for row in l0["differences"]:
        lines.append(f"| {row['levels'][0]} -> {row['levels'][1]} | {row['rel_l1_percent']:.6f} |")
    lines += ["", f"strictly decreasing: **{l0['strictly_decreasing']}**", "",
              "## L1 / L2 order and band", "",
              "| quantity | value |", "|---|---:|",
              f"| p3 ([800,1600,3200]) | {l1['p_three_levels']:.6f} |",
              f"| p4 ([1600,3200,6400]) | {l1['p_four_levels']:.6f} |",
              f"| relative order change | {l2['relative_change']:.4f} |",
              f"| U3 (3-level band) | {l1['band_percent_U3']:.6f} % |",
              f"| U4 (4-level band) | {l1['band_percent_U4']:.6f} % |",
              f"| U4 share of gate | {100.0 * l1['band_share_of_gate_U4']:.3f} % |", "",
              "## L3 band coverage (the band's own falsifier)", "",
              "| quantity | value |", "|---|---:|",
              f"| shift rel L1(cert3, cert4) on N={l3['grid']} | {l3['shift_percent']:.6f} % |",
              f"| U3 | {l3['band_percent_U3']:.6f} % |",
              f"| coverage ratio shift/U3 | {l3['coverage_ratio']:.4f} |",
              f"| covered | **{l3['pass']}** |", "",
              "## L4 claim stability", "",
              "| leg | vs 3-level ref % | vs 4-level ref % | delta pp |", "|---|---:|---:|---:|"]
    for m in l4["moves"]:
        lines.append(f"| {m['configuration']} | {m['rel_l1_percent_3level']:.6f} | "
                     f"{m['rel_l1_percent_4level']:.6f} | {m['delta_pp']:+.6f} |")
    lines += ["", f"worst leg: {l4['worst_leg_4level']} = "
              f"{l4['worst_rel_l1_percent_4level']:.6f} %; max |delta| = "
              f"{l4['max_abs_move_pp']:.6f} pp", "",
              "## Independent vs engine", ""]
    if cmp.get("status") == "compared":
        lines += ["| quantity | independent | engine | rel diff | agree |", "|---|---:|---:|---:|:--:|"]
        for r in cmp["rows"]:
            rd = "—" if r["rel_diff"] is None else f"{r['rel_diff']:.3e}"
            lines.append(f"| {r['quantity']} | {r['independent']} | {r['engine']} | {rd} | "
                         f"{'yes' if r['agree'] else 'NO'} |")
        lines += ["", f"decisions identical: **{cmp['decisions_identical']}**; "
                  f"worst leg abs diff {cmp['worst_leg_abs_diff_pp']:.3e} pp; "
                  f"all agree: **{cmp['all_agree']}**", ""]
    else:
        lines += [f"engine artifact not available ({cmp.get('status')})", ""]
    lines += ["## Decision", "", "| gate | independent |", "|---|:--:|"]
    for k, v in out["decisions"].items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += ["", f"accepted (independent) = **{out['accepted']}**", ""]
    md.write_text("\n".join(lines))
    return {"json": str(js), "markdown": str(md)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json-stdout", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--require-accepted", action="store_true")
    args = ap.parse_args(argv)

    out = audit()
    cmp = compare_with_engine(out)
    if args.json_stdout:
        print(json.dumps({"independent": out, "cross_check_vs_engine": cmp},
                         indent=2, sort_keys=True, default=str))
    else:
        print(json.dumps(out["decisions"], indent=2, sort_keys=True))
    if not args.no_write:
        paths = write(out, cmp)
        print(f"wrote {paths['json']} and {paths['markdown']}", file=sys.stderr)
    if args.require_accepted and not out["accepted"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())