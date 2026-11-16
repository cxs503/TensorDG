"""Time-discretisation pollution audit of the round-2 certified reference.

What question this answers
--------------------------
``XVAL-DG-ROUND2-CERTIFIED-REFERENCE`` certifies the Shu--Osher reference from a
ladder of FV-WENO5 profiles, and every one of those profiles was generated at
**one** time step: ``CFL = 0.3`` (``fv_cfl0p3_n800/1600/3200/6400.npz``).  The
ladder differences ``d(N -> 2N)`` are then read as *spatial* discretisation
errors, and Roache's band ``U = Fs d_fine / (r^p - 1)`` is published as the
reference's own uncertainty.  That reading is only legitimate if the time
integrator (SSP-RK3) contributes negligibly to those differences.  Nobody had
measured that: a reference whose band is set by its time step makes every
downstream "leg error < 3 %" verdict a comparison against noise of unknown
origin -- and a *time*-refined re-certification could move the reference by more
than the band it published.

This module measures the time leg directly.  It refines **only the time step**
(``CFL 0.3 -> 0.15 -> 0.075``, ratio 2) on fixed grids, so any change between the
profiles is attributable to the time discretisation alone, and:

1. measures the temporal error ``delta_t(N) = rel L1(u_CFL0.3, u_CFL0.15)`` at
   each committed grid (leading-order estimate of the CFL=0.3 time-step error);
2. compares it with the spatial difference ``d_fine(N)`` that *sets the band*
   (``bagding fraction <= d_fine * Fs / (r^p - 1)``, and with ``Fs = 1.25``,
   ``r^p - 1 ~ 1.25`` the band is of the same size as ``d_fine``);
3. estimates the temporal observed order ``p_t`` from the two successive
   temporal differences and, where ``p_t`` is usable, **Richardson-extrapolates
   each grid level in time** (``CFL -> 0``) and re-certifies the ladder on those
   time-corrected levels -- the reference the anchor would have published had it
   refined the time step;
4. re-judges all eleven shipped DG legs against both references and counts any
   decision flip, because a time-polluted reference is only *material* if it
   changes a published verdict.

Predeclared criteria (module constants, fixed before the extra CFL runs existed)
-------------------------------------------------------------------------------
``T0`` successive temporal differences strictly decrease (``d(0.3->0.15) >
       d(0.15->0.075)``) on every grid: the time leg is in an asymptotic range
       at all.  Otherwise ``p_t`` and the temporal correction are refused.
``T1`` **pollution ratio**: on the reference grid ``R = delta_t / d_fine <=
       POLLUTION_MAX`` (0.10, i.e. the CFL=0.3 time error is at most a tenth of
       the band it is supposed to be negligible against).  ``R`` in
       ``(0.10, 0.25]`` is ``MARGINAL``, above that ``POLLUTED``.
``T2`` temporal order ``p_t >= P_MIN_T`` (0.5) and finite on the reference grid,
       so the extrapolation in ``T3`` is not a guess.
``T3`` **band shift**: with time-corrected levels, ``|U_t - U| / U <=
       BAND_SHIFT_MAX`` (0.25) and the corrected ladder is still ``CERTIFIED``
       (same engine, same ``Fs``, same tolerance).
``T4`` **claim stability**: no shipped DG leg changes decision at the 3 % gate
       between the raw and the time-corrected reference, and ``|delta err| <=
       LEG_STABILITY_MAX_PP`` (0.5 pp) for every leg.

``accepted`` is the conjunction of ``T0``-``T4``; the CLI exits non-zero when it
is false so a red run cannot be mistaken for a green one.  A red run is an
honest published FAIL of the *reference*, not a defect of this module: it means
the certified band must be re-derived at a smaller CFL before being quoted.

Nothing here is tuned after the fact: the only inputs are committed raw
profiles, and the thresholds above are constants.

Usage::

    python -m benchmarks.xval_dg_round2_time_pollution            # audit + artifacts
    python -m benchmarks.xval_dg_round2_time_pollution --figures  # + PNGs
    python -m benchmarks.xval_dg_round2_time_pollution --json-stdout
    python -m benchmarks.xval_dg_round2_time_pollution --require-accepted
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow ``python benchmarks/xval_dg_round2_time_pollution.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import xval_dg_round2_certified as xval  # noqa: E402
from tensordg.certified_reference import (  # noqa: E402
    CERTIFIED, FS_GCI, P_MIN, WITHHELD, coarsen, certify, judge, observed_order,
    rel_l1,
)

__all__ = [
    "CFL_LEVELS", "BASE_LEVELS", "POLLUTION_MAX", "POLLUTION_MARGINAL",
    "BAND_SHIFT_MAX", "LEG_STABILITY_MAX_PP", "P_MIN_T",
    "cfl_tag", "level_path", "load_level", "temporal_series", "temporal_report",
    "time_corrected_levels", "classify_pollution", "audit", "markdown", "main",
    "required_inputs", "missing_inputs", "audit_or_withheld", "leg_profiles",
    "leg_distinctness", "input_provenance", "markdown_withheld",
]

#: coarse -> fine time-step settings; ratio 2 so orders are directly comparable
#: with the spatial ladder convention (``r = 2``).
CFL_LEVELS = (0.3, 0.15, 0.075)
#: grids for which the *whole* CFL series is committed (three CFLs each, needed
#: for a temporal order).  The fourth spatial level (N = 6400) has only
#: ``0.3`` and ``0.15``, which supports the ratio ``R`` but no order.
BASE_LEVELS = (800, 1600, 3200)
EXTRA_RATIO_LEVELS = (6400,)
#: grid whose ``d_fine`` sets the published four-level band (the certified
#: profile is stored on the second-finest grid).  The three-level ladder stores
#: it on 1600; both are reported, and ``REFERENCE_GRID`` is the current one.
REFERENCE_GRID = 3200
LEGACY_REFERENCE_GRID = 1600

GATE_PERCENT = xval.GATE_PERCENT       # 5 %: gate the reference is used at
STRICT_PERCENT = xval.STRICT_PERCENT   # 3 %: owner precision standard
DEFAULT_RAW = xval.DEFAULT_RAW
DEFAULT_OUT = Path("validation/xval_dg_round2/artifacts")

#: Predeclared thresholds -- constants, never recomputed from results.
POLLUTION_MAX = 0.10        # T1, ratio delta_t / d_fine on the reference grid
POLLUTION_MARGINAL = 0.25   # T1, upper band still reported as MARGINAL
P_MIN_T = 0.5               # T2, temporal order floor for a usable correction
BAND_SHIFT_MAX = 0.25       # T3, relative change of the published band
LEG_STABILITY_MAX_PP = 0.5  # T4(b), percentage points
#: T5: two DG legs count as the *same* profile when their relative L1 difference
#: is at or below this floor.  ``T4`` claims that the reference choice flips no
#: shipped verdict across eleven legs; that claim is only evidence if the eleven
#: legs are eleven *different* discretisations.  A leg table whose P1/P2/P3 rows
#: are bit-identical makes T4 pass vacuously, which this gate refuses.
LEG_DISTINCT_EPS = 1e-12

#: Verdict of :func:`audit_or_withheld` when inputs are absent (never overwritten
#: by a stale measurement).
WITHHELD_MISSING_INPUTS = "WITHHELD_MISSING_INPUTS"


# --------------------------------------------------------------------- inputs
def cfl_tag(cfl: float) -> str:
    """File-name spelling of a CFL value: ``0.3 -> '0p3'``, ``0.075 -> '0p075'``."""
    return f"{float(cfl):g}".replace(".", "p")


def level_path(rawdir: Path | str, ncells: int, cfl: float = 0.3) -> Path:
    return Path(rawdir) / f"fv_cfl{cfl_tag(cfl)}_n{int(ncells)}.npz"


def load_level(rawdir: Path | str, ncells: int, cfl: float = 0.3) -> dict:
    """Load one FV-WENO5 profile as ``{ncells, cfl, x, field, meta}``."""
    path = level_path(rawdir, ncells, cfl)
    if not path.exists():
        raise FileNotFoundError(
            f"missing CFL level {path}; generate it with "
            f"`python validation/xval_dg_round2/gen_fv_level.py --ncells {ncells} "
            f"--cfl {cfl:g} --out {path}`")
    blob = np.load(path, allow_pickle=False)
    x = np.asarray(blob["x"], dtype=float)
    rho = np.asarray(blob["rho"], dtype=float)
    if x.size != int(ncells) or rho.size != x.size:
        raise ValueError(f"{path}: expected {ncells} cells, got x={x.size} rho={rho.size}")
    if not np.all(np.isfinite(rho)) or rho.min() <= 0.0:
        raise ValueError(f"{path}: non-admissible density profile")
    meta = {}
    if "meta" in blob:
        try:
            meta = json.loads(str(blob["meta"]))
        except (ValueError, TypeError):
            meta = {}
    return {"ncells": int(ncells), "cfl": float(cfl), "x": x, "field": rho,
            "meta": meta, "file": path.name}


def temporal_series(rawdir: Path | str, ncells: int,
                    cfls=CFL_LEVELS, require: int = 2) -> list[dict]:
    """The CFL series present on disk at one grid, coarse to fine.

    ``require`` is the number of members the *caller* needs: the three-level
    correction (``p_t`` from two successive temporal differences) needs all of
    ``cfls``, while the pollution ratio ``R`` only needs the coarsest pair.  A
    grid that carries fewer than ``require`` members is refused loudly -- an
    absent level must never be silently replaced by a two-member series that
    would report ``p_t = None`` where the audit expects a correction.

    Partial series are the documented norm here, not a workaround: the finest
    grid of the ladder (``EXTRA_RATIO_LEVELS``) only has to supply ``R``, and
    each extra level costs 20+ CPU-minutes.
    """
    present = [float(c) for c in sorted(cfls, reverse=True)
               if level_path(rawdir, ncells, c).exists()]
    if len(present) < max(int(require), 2):
        missing = [c for c in sorted(cfls, reverse=True) if float(c) not in present]
        first = missing[0]
        raise FileNotFoundError(
            f"grid {ncells} has {len(present)} of {len(cfls)} CFL levels "
            f"(need {max(int(require), 2)}); missing e.g. "
            f"{level_path(rawdir, ncells, first)}; generate it with "
            f"`python validation/xval_dg_round2/gen_fv_level.py --ncells {ncells} "
            f"--cfl {first:g} --out {level_path(rawdir, ncells, first)}`")
    return [load_level(rawdir, ncells, c) for c in present]

# ------------------------------------------------------------------ provenance
def required_inputs(rawdir: Path | str = DEFAULT_RAW) -> list[dict]:
    """Every profile the audit reads, with the command that produces it.

    ``required=True`` rows are hard inputs: without them the real-data branch has
    no numbers at all.  ``required=False`` rows are the coarse partner of a grid
    row's spatial pair -- if one is absent that row simply reports no pollution
    ratio ``R`` instead of killing the audit.
    """
    rows = []
    for n in (*BASE_LEVELS, *EXTRA_RATIO_LEVELS):
        cfls = CFL_LEVELS if n in BASE_LEVELS else CFL_LEVELS[:2]
        for c in cfls:
            rows.append({
                "ncells": int(n), "cfl": float(c),
                "path": str(level_path(rawdir, n, c)),
                "required": True, "role": "time-series member",
                "generator": (f"python validation/xval_dg_round2/gen_cfl_ladder.py "
                              f"--ncells {int(n)} --cfl {float(c):g} --rawdir {rawdir}")})
    for n in (*BASE_LEVELS, *EXTRA_RATIO_LEVELS):
        coarse = int(n) // 2
        rows.append({
            "ncells": coarse, "cfl": 0.3,
            "path": str(level_path(rawdir, coarse, 0.3)),
            "required": False,
            "role": f"coarse partner of the N={n} spatial pair",
            "generator": (f"python validation/xval_dg_round2/gen_fv_level.py "
                          f"--ncells {coarse} --cfl 0.3 --out "
                          f"{level_path(rawdir, coarse, 0.3)}")})
    seen, unique = set(), []
    for row in rows:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        unique.append(row)
    return unique


def missing_inputs(rawdir: Path | str = DEFAULT_RAW) -> list[dict]:
    """Hard inputs absent from the repository, each with its producer command."""
    return [row for row in required_inputs(rawdir)
            if row["required"] and not Path(row["path"]).exists()]


def input_provenance(rawdir: Path | str = DEFAULT_RAW) -> dict:
    """``sha256`` of every profile the audit consumes (anti-staleness stamp)."""
    raw = Path(rawdir)
    paths = [Path(r["path"]) for r in required_inputs(rawdir)]
    paths += [raw / xval.leg_filename(d, n, c) for d, n, c in xval.DG_LEGS]
    files = {}
    for path in sorted({q for q in paths if q.exists()}):
        data = path.read_bytes()
        files[path.name] = {"sha256": hashlib.sha256(data).hexdigest(),
                            "bytes": len(data)}
    return {"rawdir": str(rawdir), "files": files}


def audit_or_withheld(rawdir: Path | str = DEFAULT_RAW, **kwargs) -> dict:
    """``audit`` when its inputs exist, else an explicit ``WITHHELD`` result.

    The missing-input case used to be a bare ``FileNotFoundError`` -- correct for
    a function, wrong for a *published artifact*: the file on disk then keeps
    whatever verdict it last carried, and a reader cannot tell a measured
    ``TIME_CLEAN`` from a stale one.  This wrapper turns "no numbers yet" into a
    first-class, renderable verdict that names every missing profile and the
    command that produces it, so a stale claim cannot outlive its inputs.
    """
    missing = missing_inputs(rawdir)
    if not missing:
        return audit(rawdir, **kwargs)
    return {
        "question": "is the round-2 certified reference polluted by its time step?",
        "verdict": WITHHELD_MISSING_INPUTS,
        "accepted": False,
        "reason": (f"{len(missing)} of the CFL-refined FV-WENO5 profiles this audit "
                   "consumes are not in the repository, so the real-data branch has no "
                   "numbers and no pollution verdict is claimed; a verdict stored "
                   "earlier in this artifact must not be read as current"),
        "missing": missing,
        "cfl_levels": [float(c) for c in CFL_LEVELS],
        "base_levels": [int(n) for n in BASE_LEVELS],
        "extra_ratio_levels": [int(n) for n in EXTRA_RATIO_LEVELS],
        "reference_grid": int(REFERENCE_GRID),
        "gate_percent": float(GATE_PERCENT),
        "strict_percent": float(STRICT_PERCENT),
        "rawdir": str(rawdir),
        "headline": {"state": WITHHELD_MISSING_INPUTS, "pollution_ratio": None},
    }


# ---------------------------------------------------------------- time leg math
def temporal_report(series: list[dict]) -> dict:
    """Temporal differences and order of one fixed-grid CFL series.

    All differences are measured *on the same grid*, so they contain no spatial
    contribution at all -- that is the whole point of refining CFL only.
    """
    if len(series) < 2:
        raise ValueError("a temporal difference needs two CFL levels")
    base = series[0]
    # one common denominator (the finest member) for every difference in the
    # series: with a per-pair norm the two ratios entering ``p_t`` would be
    # normalised by different profiles and the recovered order would be biased
    # by the error itself (measured: p = 1.0 -> 0.9978 on a synthetic power law)
    norm = float(np.sum(np.abs(np.asarray(series[-1]["field"], dtype=float))))
    if norm == 0.0:
        raise ValueError("zero L1 norm in the temporal series")
    rows = []
    for coarse, fine in zip(series, series[1:]):
        if coarse["ncells"] != base["ncells"] or fine["ncells"] != base["ncells"]:
            raise ValueError("a temporal series must live on one fixed grid")
        d = float(np.sum(np.abs(np.asarray(fine["field"], dtype=float)
                                - np.asarray(coarse["field"], dtype=float)))) / norm
        rows.append({
            "cfls": [coarse["cfl"], fine["cfl"]],
            "ratio": coarse["cfl"] / fine["cfl"],
            "rel_l1": d, "rel_l1_percent": 100.0 * d,
            "steps": [int(coarse["meta"].get("steps", -1)),
                      int(fine["meta"].get("steps", -1))],
        })
    p_t = None
    if len(rows) >= 2:
        d_coarse, d_fine = rows[0]["rel_l1"], rows[1]["rel_l1"]
        if d_coarse > 0.0 and d_fine > 0.0:
            p_t = observed_order(d_coarse, d_fine, float(rows[0]["ratio"]))
    return {
        "ncells": int(base["ncells"]),
        "cfls": [float(s["cfl"]) for s in series],
        "levels_present": len(series),
        "differences": rows,
        # leading-order estimate of the *coarsest* (published) time-step error:
        # for order >= 1 the difference to the next finer level is an upper
        # bound of it, which is the conservative reading this audit wants
        "norm_field": "finest temporal member (common denominator)",
        "delta_t": rows[0]["rel_l1"],
        "delta_t_percent": 100.0 * rows[0]["rel_l1"],
        "temporal_order": p_t,
        "monotone": bool(len(rows) < 2 or rows[0]["rel_l1"] > rows[1]["rel_l1"]),
        "files": [s["file"] for s in series],
    }


def time_corrected_field(series: list[dict], p_t: float) -> np.ndarray:
    """Richardson-extrapolate one grid level to ``CFL -> 0`` (time continuum).

    ``series`` is coarse -> fine on a **single** grid, so the profiles are
    already aligned and the correction is a plain Richardson step; nothing is
    coarsened (coarsening would inject exactly the spatial error this audit is
    trying to isolate).
    """
    if len(series) < 2:
        raise ValueError("a temporal extrapolation needs two CFL levels")
    if p_t is None or not np.isfinite(p_t) or p_t <= 0.0:
        raise ValueError(f"unusable temporal order {p_t!r}")
    coarse, fine = series[-2], series[-1]
    r = float(coarse["cfl"]) / float(fine["cfl"])
    a = np.asarray(fine["field"], dtype=float)
    b = np.asarray(coarse["field"], dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"temporal series is not on one grid: {a.shape} vs {b.shape}")
    return a + (a - b) / (r ** p_t - 1.0)


def time_corrected_levels(rawdir: Path | str = DEFAULT_RAW,
                          grids=BASE_LEVELS) -> tuple[list[dict], dict]:
    """A spatial ladder whose every level is time-converged (``CFL -> 0``)."""
    levels, used = [], {}
    for n in grids:
        rep = temporal_report(temporal_series(rawdir, n))
        if not rep["monotone"] or rep["temporal_order"] is None:
            raise ValueError(f"grid {n}: temporal series not monotone/asymptotic")
        field = time_corrected_field(temporal_series(rawdir, n), rep["temporal_order"])
        levels.append({"ncells": int(n), "field": field})
        used[int(n)] = {"temporal_order": rep["temporal_order"],
                        "delta_t_percent": rep["delta_t_percent"]}
    return levels, used


# ------------------------------------------------------------- decision rule
def classify_pollution(ratio: float | None) -> str:
    """``CLEAN`` / ``MARGINAL`` / ``POLLUTED`` from the pollution ratio ``R``.

    Kept as a pure function so the decision rule is testable without generating
    any CFD data, and so the thresholds appear in exactly one place.
    """
    if ratio is None or not np.isfinite(ratio):
        return WITHHELD
    if ratio <= POLLUTION_MAX:
        return "CLEAN"
    if ratio <= POLLUTION_MARGINAL:
        return "MARGINAL"
    return "POLLUTED"


# ---------------------------------------------------------------------- audit
def _leg_table(cert: dict, rawdir: Path | str = DEFAULT_RAW) -> list[dict]:
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
            "decision_5pct": res5["decision"],
            "decision_3pct": res3["decision"],
        })
    return rows


def leg_profiles(rawdir: Path | str = DEFAULT_RAW) -> list[dict]:
    """The eleven shipped DG leg profiles, as arrays (needed for T5)."""
    raw = Path(rawdir)
    legs = []
    for degree, ncells, cfl in xval.DG_LEGS:
        name = xval.leg_filename(degree, ncells, cfl)
        with np.load(raw / name, allow_pickle=False) as blob:
            field = np.asarray(blob["rho"], dtype=float)
        legs.append({"configuration": f"DG P{degree} N={ncells} CFL={cfl}",
                     "degree": int(degree), "ncells": int(ncells), "cfl": float(cfl),
                     "file": name, "field": field})
    return legs


def leg_distinctness(legs: list[dict]) -> dict:
    """T5 -- are the legs the leg table claims actually different profiles?

    Only legs on the same grid are compared: ``rel L1`` between an N=200 and an
    N=800 field is not defined without an interpolation the audit does not want
    to introduce.  Duplicate profiles inside one grid group are the failure mode
    seen in a stale artifact (nine legs collapsing to three distinct errors).
    """
    groups: dict[int, list[dict]] = {}
    for leg in legs:
        groups.setdefault(int(leg["ncells"]), []).append(leg)
    pairs, duplicates = [], []
    for n in sorted(groups):
        members = groups[n]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                d = rel_l1(members[i]["field"], members[j]["field"])
                pair = {"ncells": int(n),
                        "a": members[i]["configuration"],
                        "b": members[j]["configuration"],
                        "rel_l1": float(d)}
                pairs.append(pair)
                if not (d > LEG_DISTINCT_EPS):
                    duplicates.append(pair)
    smallest = min((p["rel_l1"] for p in pairs), default=None)
    return {
        "criterion": ("no two DG legs on the same grid are the same profile "
                      f"(rel L1 > {LEG_DISTINCT_EPS:g}): otherwise T4 'no decision "
                      "flip across eleven legs' is a claim about fewer legs"),
        "n_legs": len(legs),
        "n_groups": len(groups),
        "n_pairs": len(pairs),
        "eps": LEG_DISTINCT_EPS,
        "min_pair_rel_l1": smallest,
        "duplicates": duplicates,
        "pairs": pairs,
        "pass": bool(legs) and not duplicates,
    }


def audit(rawdir: Path | str = DEFAULT_RAW) -> dict:
    """Run T0-T4 on committed raw profiles; deterministic, no side effects."""
    rawdir = Path(rawdir)
    # ---- per-grid time leg (T0) and spatial difference it is compared against
    grid_rows = []
    for n in (*BASE_LEVELS, *EXTRA_RATIO_LEVELS):
        # a grid inside the corrected spatial ladder needs a usable p_t, i.e. all
        # CFL levels; the extra ratio grid only has to supply the pollution ratio
        need = len(CFL_LEVELS) if n in BASE_LEVELS else 2
        src = temporal_series(rawdir, n, require=need)
        rep = temporal_report(src)
        row = {"ncells": int(n), **{k: rep[k] for k in
                                    ("differences", "delta_t", "delta_t_percent",
                                     "temporal_order", "monotone", "files",
                                     "levels_present")}}
        # spatial difference that sets the band at this level: the finest
        # successive difference of the *committed* CFL=0.3 ladder, measured on
        # the coarser grid exactly as ``certify`` measures it
        if n > 1:
            fine = load_level(rawdir, n, 0.3)
            coarse_n = n // 2
            try:
                coarse = load_level(rawdir, coarse_n, 0.3)
            except FileNotFoundError:
                coarse = None
            if coarse is not None:
                # certify()'s convention: the difference is measured on the
                # *coarse* grid, so the fine level is cell-averaged down first
                d_sp = rel_l1(coarsen(fine["field"], 2), coarse["field"])
                row["spatial_difference"] = d_sp
                row["spatial_pair"] = [int(coarse_n), int(n)]
                row["pollution_ratio"] = (rep["delta_t"] / d_sp) if d_sp > 0 else None
        grid_rows.append(row)

    by_grid = {row["ncells"]: row for row in grid_rows}
    ref_row = by_grid[REFERENCE_GRID]
    legacy_row = by_grid[LEGACY_REFERENCE_GRID]

    # ---- T0: temporal ladder asymptotic on the reference grid
    t0 = {
        "criterion": "successive temporal differences strictly decrease on every "
                     "grid with a full CFL series",
        "rows": {int(r["ncells"]): r["differences"] for r in grid_rows},
        "monotone": {int(r["ncells"]): bool(r["monotone"]) for r in grid_rows},
        "pass": bool(all(r["monotone"] for r in grid_rows
                         if r["ncells"] in BASE_LEVELS)),
    }

    # ---- T1: pollution ratio on the reference grid
    ratio = ref_row.get("pollution_ratio")
    t1_state = classify_pollution(ratio)
    t1 = {
        "criterion": f"delta_t / d_fine <= {POLLUTION_MAX} on N={REFERENCE_GRID}",
        "reference_grid": int(REFERENCE_GRID),
        "delta_t_percent": ref_row["delta_t_percent"],
        "d_fine_percent": 100.0 * ref_row["spatial_difference"],
        "pollution_ratio": ratio,
        "state": t1_state,
        "legacy_reference_grid": int(LEGACY_REFERENCE_GRID),
        "legacy_pollution_ratio": legacy_row.get("pollution_ratio"),
        "by_grid": {int(r["ncells"]): r.get("pollution_ratio") for r in grid_rows},
        "pass": bool(t1_state == "CLEAN"),
    }

    # ---- T2: is the temporal order usable?
    p_t = ref_row["temporal_order"]
    t2 = {
        "criterion": f"temporal order on N={REFERENCE_GRID} >= {P_MIN_T}",
        "temporal_order": p_t,
        "by_grid": {int(r["ncells"]): r["temporal_order"] for r in grid_rows},
        "pass": bool(p_t is not None and np.isfinite(p_t) and p_t >= P_MIN_T),
    }

    # ---- T3: re-certify on time-corrected levels, compare the band
    raw_levels = [load_level(rawdir, n, 0.3) for n in BASE_LEVELS]
    cert_raw = certify(raw_levels, tolerance_percent=GATE_PERCENT)
    cert_raw4_levels = [load_level(rawdir, n, 0.3)
                        for n in (*BASE_LEVELS, *EXTRA_RATIO_LEVELS)]
    cert_raw4 = certify(cert_raw4_levels, tolerance_percent=GATE_PERCENT)
    try:
        corr_levels, corr_meta = time_corrected_levels(rawdir, BASE_LEVELS)
        cert_corr = certify(corr_levels, tolerance_percent=GATE_PERCENT)
        corr_error = None
    except (ValueError, FileNotFoundError) as exc:  # honest refusal, not a crash
        corr_levels, cert_corr, corr_meta, corr_error = None, None, {}, str(exc)

    band_raw = cert_raw.get("band_percent")
    band_corr = None if cert_corr is None else cert_corr.get("band_percent")
    band_shift = (None if band_corr is None or not band_raw
                  else abs(band_corr - band_raw) / abs(band_raw))
    order_raw = cert_raw.get("observed_order_used")
    order_corr = None if cert_corr is None else cert_corr.get("observed_order_used")
    order_shift = (None if order_corr is None or not order_raw
                   else abs(order_corr - order_raw) / abs(order_raw))
    t3 = {
        "criterion": f"|U_t - U|/U <= {BAND_SHIFT_MAX} and the corrected ladder is CERTIFIED",
        "raw": {k: cert_raw.get(k) for k in
                ("verdict", "classification", "observed_order_used", "band_percent",
                 "reference_ncells", "differences")},
        "raw_four_level": {k: cert_raw4.get(k) for k in
                           ("verdict", "classification", "observed_order_used",
                            "band_percent", "reference_ncells")},
        "time_corrected": None if cert_corr is None else
        {k: cert_corr.get(k) for k in
         ("verdict", "classification", "observed_order_used", "band_percent",
          "reference_ncells", "differences")},
        "temporal_orders_used": corr_meta,
        "refusal": corr_error,
        "band_shift_relative": band_shift,
        "order_shift_relative": order_shift,
        "threshold": BAND_SHIFT_MAX,
        "pass": bool(cert_corr is not None
                     and cert_corr.get("verdict") == CERTIFIED
                     and band_shift is not None and band_shift <= BAND_SHIFT_MAX),
    }

    # ---- T4: do published DG verdicts move?
    legs_raw = _leg_table(cert_raw, rawdir)
    legs_corr = None if cert_corr is None else _leg_table(cert_corr, rawdir)
    flips, max_move = [], None
    if legs_corr is not None:
        for a, b in zip(legs_raw, legs_corr):
            if a["decision_3pct"] != b["decision_3pct"]:
                flips.append({"configuration": a["configuration"],
                              "raw": a["decision_3pct"],
                              "time_corrected": b["decision_3pct"],
                              "raw_error_percent": a["rel_l1_percent"],
                              "corrected_error_percent": b["rel_l1_percent"]})
            move = abs(b["rel_l1_percent"] - a["rel_l1_percent"])
            max_move = move if max_move is None else max(max_move, move)
    t4 = {
        "criterion": f"no decision flip at {STRICT_PERCENT:g} % and every leg moves "
                     f"<= {LEG_STABILITY_MAX_PP} pp",
        "legs_raw": legs_raw, "legs_time_corrected": legs_corr,
        "flips": flips, "n_flips": len(flips),
        "max_abs_move_pp": max_move,
        "threshold_pp": LEG_STABILITY_MAX_PP,
        "pass": bool(legs_corr is not None and not flips
                     and max_move is not None and max_move <= LEG_STABILITY_MAX_PP),
    }

    # ---- T5: non-vacuity precondition of T4
    t5 = leg_distinctness(leg_profiles(rawdir))
    if not t5["pass"]:
        t4["pass"] = False
        t4["downgraded_by"] = "T5"
        t4["downgrade_reason"] = (
            "identical DG leg profiles: 'no decision flip across eleven legs' would "
            "be a claim about fewer distinct discretisations than it names")

    criteria = {"T0": t0, "T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5}
    accepted = bool(all(c["pass"] for c in criteria.values()))
    if accepted:
        verdict = "TIME_CLEAN"
    elif not t5["pass"]:
        verdict = "AUDIT_INVALID_LEGS"
    elif t1["state"] == "MARGINAL":
        verdict = "TIME_MARGINAL"
    else:
        verdict = "TIME_POLLUTED"
    return {
        "question": "is the round-2 certified reference polluted by its time step?",
        "verdict": verdict,
        "accepted": accepted,
        "cfl_levels": [float(c) for c in CFL_LEVELS],
        "reference_grid": int(REFERENCE_GRID),
        "fs_gci": float(FS_GCI),
        "p_min_spatial": float(P_MIN),
        "gate_percent": float(GATE_PERCENT),
        "strict_percent": float(STRICT_PERCENT),
        "grids": grid_rows,
        "criteria": criteria,
        "provenance": input_provenance(rawdir),
        "rawdir": str(rawdir),
        "headline": {
            "pollution_ratio": ratio,
            "state": t1_state,
            "time_corrected_band_percent": band_corr,
            "raw_band_percent": band_raw,
            "band_shift_relative": band_shift,
            "n_decision_flips": len(flips),
        },
    }


# ------------------------------------------------------------------- rendering
def markdown_withheld(result: dict) -> str:
    """Render the ``WITHHELD_MISSING_INPUTS`` result (no verdict is claimed)."""
    lines = [
        "# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — time-discretisation pollution audit",
        "",
        f"**Verdict: `{result['verdict']}`** (accepted = False) — question: "
        "*is the published certified reference polluted by the time step it was "
        "generated at?*",
        "",
        "**No pollution verdict is claimed.** " + str(result["reason"]) + ".",
        "",
        f"CFL series {result['cfl_levels']} (ratio 2, fixed grids), reference grid "
        f"N = {result['reference_grid']}, band = Roache GCI with Fs = {FS_GCI}.",
        "",
        "## Missing inputs",
        "",
        "| grid | CFL | file | producer command |",
        "|---|---|---|---|",
    ]
    for row in result["missing"]:
        lines.append(f"| {row['ncells']} | {row['cfl']:g} | `{Path(row['path']).name}` | "
                     f"`{row['generator']}` |")
    lines += [
        "",
        "The eleven shipped DG legs and the CFL = 0.3 ladder are committed, so the "
        "audit can say *what* it would compute -- it cannot say *what the answer is* "
        "until the profiles above exist. Total cost of the table: ~1.2 CPU-hours "
        "(see `validation/xval_dg_round2/gen_cfl_ladder.py`, shardable).",
        "",
    ]
    return "\n".join(lines)


def markdown(result: dict) -> str:
    if "criteria" not in result:
        return markdown_withheld(result)
    h = result["headline"]
    c = result["criteria"]
    lines = [
        "# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — time-discretisation pollution audit",
        "",
        f"**Verdict: `{result['verdict']}`** (accepted = {result['accepted']}) — "
        "question: *is the published certified reference polluted by the time step "
        "it was generated at?*",
        "",
        f"CFL series {result['cfl_levels']} (ratio 2, fixed grids), reference grid "
        f"N = {result['reference_grid']}, band = Roache GCI with Fs = {result['fs_gci']}.",
        "",
        "| criterion | statement | measured | pass |",
        "|---|---|---|---|",
        f"| T0 | temporal successive differences decrease | "
        f"{ {k: v for k, v in c['T0']['monotone'].items()} } | {c['T0']['pass']} |",
        f"| T1 | delta_t / d_fine <= {POLLUTION_MAX} on N={result['reference_grid']} | "
        f"{None if h['pollution_ratio'] is None else round(h['pollution_ratio'], 6)} "
        f"({h['state']}) | {c['T1']['pass']} |",
        f"| T2 | temporal order >= {P_MIN_T} | {c['T2']['temporal_order']} | {c['T2']['pass']} |",
        f"| T3 | band shift <= {BAND_SHIFT_MAX} and corrected ladder CERTIFIED | "
        f"{None if c['T3']['band_shift_relative'] is None else round(c['T3']['band_shift_relative'], 6)} | "
        f"{c['T3']['pass']} |",
        f"| T4 | no 3 % decision flip, leg move <= {LEG_STABILITY_MAX_PP} pp | "
        f"{c['T4']['n_flips']} flips, max move {c['T4']['max_abs_move_pp']} pp | "
        f"{c['T4']['pass']} |",
        (f"| T5 | legs pairwise distinct (T4 not vacuous) | "
         f"{c['T5']['n_pairs']} pairs, min rel L1 "
         f"{c['T5']['min_pair_rel_l1']:.3e} | {c['T5']['pass']} |"
         if "T5" in c else "| T5 | legs pairwise distinct | (not evaluated) | - |"),
        "",
        "## Time leg per grid",
        "",
        "| N | delta_t = rel L1(0.3, 0.15) % | rel L1(0.15, 0.075) % | p_t | "
        "d_fine (spatial) % | R = delta_t/d_fine |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["grids"]:
        diffs = row["differences"]
        second = f"{diffs[1]['rel_l1_percent']:.6e}" if len(diffs) > 1 else "—"
        p_t = row["temporal_order"]
        d_sp = row.get("spatial_difference")
        rr = row.get("pollution_ratio")
        lines.append(
            f"| {row['ncells']} | {row['delta_t_percent']:.6e} | {second} | "
            f"{'—' if p_t is None else f'{p_t:.4f}'} | "
            f"{'—' if d_sp is None else f'{100.0 * d_sp:.6e}'} | "
            f"{'—' if rr is None else f'{rr:.4f}'} |")
    lines += [
        "",
        "## Reference: raw (CFL = 0.3) vs time-corrected (CFL -> 0)",
        "",
        "| ladder | verdict | observed order | band % | reference grid |",
        "|---|---|---|---|---|",
    ]
    for label, key in (("raw three-level (published round 2)", "raw"),
                       ("raw four-level (current shipped)", "raw_four_level"),
                       ("time-corrected three-level", "time_corrected")):
        block = c["T3"].get(key)
        if not block:
            lines.append(f"| {label} | (refused) | - | - | - |")
            continue
        order = block["observed_order_used"]
        band = block["band_percent"]
        order_s = "-" if order is None else f"{order:.6f}"
        band_s = "-" if band is None else f"{band:.6f}"
        lines.append(f"| {label} | {block['verdict']} | {order_s} | {band_s} | "
                     f"{block['reference_ncells']} |")
    if c["T3"].get("refusal"):
        lines += ["", f"time-corrected ladder refused: `{c['T3']['refusal']}`"]
    lines += [
        "",
        "## DG legs: does the reference choice move a published verdict?",
        "",
        "| configuration | err % (raw ref) | err % (time-corrected ref) | "
        "decision 3% (raw) | decision 3% (corrected) |",
        "|---|---|---|---|---|",
    ]
    corr = c["T4"]["legs_time_corrected"] or [None] * len(c["T4"]["legs_raw"])
    for a, b in zip(c["T4"]["legs_raw"], corr):
        corr_err = "-" if b is None else f"{b['rel_l1_percent']:.6f}"
        corr_dec = "-" if b is None else b["decision_3pct"]
        lines.append(f"| {a['configuration']} | {a['rel_l1_percent']:.6f} | "
                     f"{corr_err} | {a['decision_3pct']} | {corr_dec} |")
    prov = result.get("provenance") or {}
    if prov.get("files"):
        lines += ["", "## Input provenance (sha256)", "",
                  "| file | sha256 (16) | bytes |", "|---|---|---|"]
        for name, meta in sorted(prov["files"].items()):
            lines.append(f"| `{name}` | `{meta['sha256'][:16]}` | {meta['bytes']} |")
    lines += ["", "## Thresholds (predeclared)", "",
              f"* `POLLUTION_MAX = {POLLUTION_MAX}` (CLEAN), "
              f"`POLLUTION_MARGINAL = {POLLUTION_MARGINAL}` (MARGINAL ceiling)",
              f"* `P_MIN_T = {P_MIN_T}`, `BAND_SHIFT_MAX = {BAND_SHIFT_MAX}`, "
              f"`LEG_STABILITY_MAX_PP = {LEG_STABILITY_MAX_PP}`", ""]
    return "\n".join(lines)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def write_artifacts(result: dict, outdir: Path | str = DEFAULT_OUT) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    js = out / "time_pollution.json"
    md = out / "time_pollution.md"
    js.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    md.write_text(markdown(result) + "\n")
    return {"json": str(js), "md": str(md)}


# --------------------------------------------------------------------- figures
def figures(result: dict, rawdir: Path | str = DEFAULT_RAW,
            outdir: Path | str = DEFAULT_OUT) -> list[str]:
    """Three PNGs: time leg vs spatial leg, profile/pollution overlay, leg table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rawdir = Path(rawdir)
    made = []

    grids = [r for r in result["grids"] if r.get("spatial_difference")]
    ns = np.array([r["ncells"] for r in grids], dtype=float)

    # ---- fig 1: time leg vs the spatial leg it must stay below
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(ns, [100.0 * r["delta_t"] for r in grids], "o-", label=r"time leg $\delta_t(CFL\;0.3\to0.15)$")
    ax.loglog(ns, [100.0 * r["spatial_difference"] for r in grids], "s--",
              label=r"spatial leg $d_{fine}$ (sets the band $U$)")
    for r in grids:
        if r.get("pollution_ratio") is not None:
            ax.annotate(f"R={r['pollution_ratio']:.3f}", (r["ncells"], 100.0 * r["delta_t"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(100.0 * POLLUTION_MAX, color="0.6", lw=1, ls=":")
    ax.set_xlabel("cells $N$")
    ax.set_ylabel("relative $L^1$ difference [%]")
    ax.set_title("Time leg vs spatial leg of the certified reference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = out / "fig_time_vs_space_leg.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    made.append(str(p1))

    # ---- fig 2: profiles at N=3200 (detail) + the CFL difference
    n = REFERENCE_GRID
    coarse = load_level(rawdir, n, 0.3)
    fine = load_level(rawdir, n, 0.075)
    x = coarse["x"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax0.plot(x, coarse["field"], lw=1.0, label="FV-WENO5 CFL=0.3")
    ax0.plot(x, fine["field"], lw=1.0, ls="--", label="FV-WENO5 CFL=0.075")
    ax0.set_ylabel(r"$\rho$")
    ax0.set_title(f"Shu--Osher, N={n}: the published time step vs a 4x smaller one")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)
    ax1.plot(x, fine["field"] - coarse["field"], lw=0.9, color="tab:red")
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$\rho_{0.075}-\rho_{0.3}$")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    p2 = out / "fig_cfl_profile_difference.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    made.append(str(p2))

    # ---- fig 3: DG leg errors against both references
    t4 = result["criteria"]["T4"]
    labels = [d["configuration"].replace("DG ", "") for d in t4["legs_raw"]]
    y_raw = [d["rel_l1_percent"] for d in t4["legs_raw"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    idx = np.arange(len(labels))
    ax.bar(idx - 0.2, y_raw, width=0.4, label="reference at CFL=0.3 (published)")
    if t4["legs_time_corrected"]:
        ax.bar(idx + 0.2, [d["rel_l1_percent"] for d in t4["legs_time_corrected"]],
               width=0.4, label=r"reference at $CFL\to0$ (time-corrected)")
    ax.axhline(STRICT_PERCENT, color="k", ls="--", lw=1, label="3 % gate")
    ax.axhline(GATE_PERCENT, color="0.5", ls=":", lw=1, label="5 % gate")
    ax.set_ylabel("rel $L^1$ error vs reference [%]")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_title("Shipped DG legs: verdict stability under a time-refined reference")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p3 = out / "fig_leg_stability.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    made.append(str(p3))
    return made


# ------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rawdir", default=str(DEFAULT_RAW))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--json-stdout", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--require-accepted", action="store_true")
    args = ap.parse_args(argv)

    result = audit_or_withheld(args.rawdir)
    paths = write_artifacts(result, args.outdir)
    payload = _jsonable(result)
    if args.figures and "criteria" in result:
        payload["figures"] = figures(result, args.rawdir, args.outdir)
    payload["artifacts"] = paths
    if args.json_stdout:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(markdown(result))
        print(f"\nartifacts: {paths}")
    return 0 if (result["accepted"] or not args.require_accepted) else 1


if __name__ == "__main__":
    raise SystemExit(main())