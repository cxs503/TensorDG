"""Certified reference for cases without a closed-form solution (round-2 practice).

Why this module exists
----------------------
Every ``error < tolerance`` statement about a case without an exact solution is
an error *against some reference*, and the credibility of that reference is the
whole argument.  Quoting a single fine-grid run as "the reference" leaves the
reference's own discretisation error unquantified: a measured 0.6 % can then
mean 0.6 % or 0.6 % +/- 0.3 %, and nobody can tell which.

Round 2 of CASE-SHU-OSHER closed that gap for one case with an ad-hoc driver
(:mod:`benchmarks.shu_osher_certified_ref`).  This module turns the practice into
a **reusable, case-independent engine** and adds the three machine decisions the
one-off script left to prose:

1. **Convergence-case classification.**  An observed order is only meaningful if
   the ladder is in its asymptotic range.  Data that stagnates (``p`` near 0),
   oscillates, or sits on the round-off floor must be *refused*, not averaged
   into a number -- the one-off driver happily returns ``p = 0.04`` and a
   "reference" built from it.
2. **Gate validity (vacuity) rule.**  A reference is only usable at tolerance
   ``tau`` if its own uncertainty band ``U`` is small against that tolerance:
   ``U / tau <= u_share_max`` (default ``1/3``).  Otherwise the comparison is
   vacuous -- it cannot distinguish PASS from FAIL -- and the verdict is
   ``WITHHELD``.  This is the written-down form of "the reference's uncertainty
   must be much smaller than the gate".
3. **Margin robustness.**  A candidate passes only if ``err + U < tau``; a
   reading that passes but whose margin is smaller than the reference
   uncertainty is reported as ``PASS_MARGINAL`` instead of ``PASS``, so a gate
   can never be passed by reference noise.

Method (unchanged from round 2, so the two rounds stay comparable)
-----------------------------------------------------------------
* differences and observed orders from a **nesting** ladder of profiles, all
  differences measured on the coarsest grid so no reconstruction ambiguity
  enters;
* Richardson extrapolation of the finest level towards the continuum limit,
  which becomes the certified profile;
* Roache's grid-convergence index (``Fs = 1.25``) as the uncertainty band of the
  certified reference.

Consistency identity (a genuine trap in this practice)
-----------------------------------------------------
The certified profile is stored on the **second-finest** grid, so its distance
to the *raw* level it replaces is *not* ``U`` -- it is the discretisation error
of the second-finest level, i.e. ``r**p * U / Fs`` (measured: 2.07x for
CASE-SHU-OSHER).  Comparing that displacement against ``U`` and concluding "the
band is understated" is a **wrong reading**: the band belongs to the finest
level.  :func:`certify` therefore reports
``extrapolation_displacement_expected_percent`` next to the measured value and
``displacement_ratio``, and the identity is asserted by the test suite.

References
----------
* P. J. Roache, *Perspective: a method for uniform reporting of grid refinement
  studies*, ASME J. Fluids Eng. **116** (1994) 405--413 (GCI, ``Fs = 1.25`` for
  a three-grid study).
* ASME V&V 20-2009 (reference-uncertainty accounting in a validation
  comparison).
* I. B. Celik et al., *Procedure for estimation and reporting of uncertainty due
  to discretization in CFD applications*, ASME J. Fluids Eng. **130** (2008)
  078001 (asymptotic-range check, three-case classification).
* C.-W. Shu & S. Osher, J. Comput. Phys. **77** (1988) 439--471 (the case whose
  round-2 certification this engine reproduces).

Usage::

    from tensordg.certified_reference import certify, judge

    cert = certify(levels, tolerance_percent=5.0)     # levels: coarse -> fine
    cert["verdict"]                                   # CERTIFIED / WITHHELD_*
    judge(dg_profile_on_any_nesting_grid, cert)       # PASS / PASS_MARGINAL / ...

Reachability note
-----------------
Every public name here is exported at the package root, so
``from tensordg import certify, judge`` works.  Five names collide with
:mod:`tensordg.convergence` (``observed_order``, ``richardson_extrapolate``,
``coarsen``, ``rel_l1``, ``gci``) and are therefore exported under the
``certified_`` prefix (``certified_observed_order``, ...), following the
disambiguation convention the package already uses for ``euler_sound_speed`` /
``swe_sound_speed``; the remaining six (``certify``, ``judge``, ``classify``,
``digest_levels``, ``audit_ladder``, ``band_coverage_study``) are exported
bare.  History, for the record: when this module first landed it could not be
reached from the root at all, which left the package contract gate
(``tests/test_public_api_contract.py``) RED with 11 of its 15 missing symbols
coming from this module -- i.e. the module was the main contributor to that
RED, not a bystander (audit ``docs/AUDIT-m17539-XVAL-DG-ROUND2.md`` finding
B1/M1).  The export block above fixes that; the gate is green as of the
``XVAL-DG-ROUND2-CERTIFIED-REFERENCE`` closure.

Command line (reproduces the CASE-SHU-OSHER round-2 numbers from the committed
raw profiles and audits every shipped DG leg)::

    python -m tensordg.certified_reference --rawdir \
        validation/repro_shu_osher/artifacts/raw --outdir artifacts/m17539
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

__all__ = [
    "FS_GCI",
    "U_SHARE_MAX",
    "P_MIN",
    "P_MAX",
    "ROUNDOFF_FLOOR",
    "coarsen",
    "rel_l1",
    "observed_order",
    "richardson_extrapolate",
    "gci",
    "classify",
    "certify",
    "judge",
    "audit_ladder",
    "band_coverage_study",
    "digest_levels",
]

#: Roache factor of safety for a three-grid study.
FS_GCI = 1.25
#: A reference may consume at most this share of the tolerance it is used at.
U_SHARE_MAX = 1.0 / 3.0
#: Lowest observed order still accepted as "in the asymptotic range".  A
#: shock-capturing ladder typically shows p ~ 1; anything below half an order is
#: a stagnating ladder, not a converging one.
P_MIN = 0.5
#: Highest observed order accepted.  Above this the "convergence" is more likely
#: a round-off floor, a metric artifact or a superconvergent fluke.
P_MAX = 6.0
#: Relative differences below this are treated as floating-point noise.
ROUNDOFF_FLOOR = 1e-12

CERTIFIED = "CERTIFIED"
WITHHELD_TOO_FEW_LEVELS = "WITHHELD_TOO_FEW_LEVELS"
WITHHELD_NON_NESTING = "WITHHELD_NON_NESTING"
WITHHELD_STAGNANT = "WITHHELD_STAGNANT"
WITHHELD_ORDER_TOO_LOW = "WITHHELD_ORDER_TOO_LOW"
WITHHELD_ORDER_TOO_HIGH = "WITHHELD_ORDER_TOO_HIGH"
WITHHELD_ROUNDOFF_FLOOR = "WITHHELD_ROUNDOFF_FLOOR"
WITHHELD_BAND_CONSUMES_GATE = "WITHHELD_BAND_CONSUMES_GATE"

PASS = "PASS"
PASS_MARGINAL = "PASS_MARGINAL"
FAIL = "FAIL"
WITHHELD = "WITHHELD"


# ------------------------------------------------------------------ primitives
def coarsen(values, factor: int) -> np.ndarray:
    """Cell-average a fine cell-centred profile down to a coarser grid."""
    v = np.asarray(values, dtype=float)
    factor = int(factor)
    if factor <= 0:
        raise ValueError("coarsening factor must be positive")
    if factor == 1:
        return v
    if v.size % factor:
        raise ValueError(f"size {v.size} not divisible by coarsening factor {factor}")
    return v.reshape(-1, factor).mean(axis=1)


def rel_l1(a, b) -> float:
    """Relative L1 difference of two profiles on the same grid (as a fraction)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    denom = float(np.sum(np.abs(b)))
    if denom == 0.0:
        raise ValueError("reference profile has zero L1 norm")
    return float(np.sum(np.abs(a - b)) / denom)


def observed_order(d_coarse: float, d_fine: float, r: float = 2.0) -> float:
    """Observed order of accuracy from two successive differences."""
    if d_fine <= 0.0 or d_coarse <= 0.0:
        raise ValueError("differences must be positive to estimate an order")
    if r <= 0.0 or r == 1.0:
        raise ValueError("refinement ratio must be positive and != 1")
    return float(math.log(d_coarse / d_fine) / math.log(r))


def richardson_extrapolate(fine, coarse, p: float, r: float = 2.0) -> np.ndarray:
    """Richardson estimate of the continuum profile, on the *fine* grid."""
    fine = np.asarray(fine, dtype=float)
    coarse = np.asarray(coarse, dtype=float)
    factor = int(round(fine.size / coarse.size))
    sub = coarsen(coarse, factor)
    if sub.shape != fine.shape:
        raise ValueError(f"cannot align {fine.shape} with {np.asarray(coarse).shape}")
    return fine + (fine - sub) / (r ** p - 1.0)


def gci(d_coarse: float, d_fine: float, p: float, fs: float = FS_GCI,
        r: float = 2.0) -> float:
    """Roache grid-convergence index of the *fine* level (as a fraction)."""
    return float(fs * d_fine / (r ** p - 1.0))


def classify(p: float | None, d_fine: float, *, p_min: float = P_MIN,
             p_max: float = P_MAX,
             roundoff_floor: float = ROUNDOFF_FLOOR) -> tuple[str, str]:
    """Three-case classification of a ladder (Celik et al. 2008 style).

    Returns ``(case, verdict)`` where ``case`` is one of ``"monotonic"``,
    ``"stagnant"``, ``"order_too_low"``, ``"order_too_high"``,
    ``"roundoff_floor"`` and ``verdict`` is ``CERTIFIED`` or the matching
    ``WITHHELD_*`` reason.  Refusing non-asymptotic data is the point: a
    certified reference built from a stagnating ladder is a number with no
    meaning.
    """
    if d_fine <= roundoff_floor:
        return "roundoff_floor", WITHHELD_ROUNDOFF_FLOOR
    if p is None or not np.isfinite(p):
        return "stagnant", WITHHELD_STAGNANT
    if p <= 0.0:
        return "stagnant", WITHHELD_STAGNANT
    if p < p_min:
        return "order_too_low", WITHHELD_ORDER_TOO_LOW
    if p > p_max:
        return "order_too_high", WITHHELD_ORDER_TOO_HIGH
    return "monotonic", CERTIFIED


def digest_levels(levels) -> str:
    """sha256 over grid sizes + profiles: provenance for the certification."""
    h = hashlib.sha256()
    for lv in levels:
        h.update(str(int(lv["ncells"])).encode())
        h.update(np.ascontiguousarray(np.asarray(lv["field"], dtype=np.float64)).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- certification
def certify(levels, *, tolerance_percent: float, fs: float = FS_GCI,
            u_share_max: float = U_SHARE_MAX, p_min: float = P_MIN,
            p_max: float = P_MAX, roundoff_floor: float = ROUNDOFF_FLOOR) -> dict:
    """Certify a reference profile from a nesting ladder of profiles.

    ``levels`` is a list of ``{"ncells": int, "field": array}`` in **coarse to
    fine** order; ``tolerance_percent`` is the gate the reference will be used
    at (it enters only the gate-validity rule, never the numbers).

    Returns a JSON-able dict (arrays excluded) whose ``verdict`` is ``CERTIFIED``
    only if the ladder is asymptotic *and* the band fits the gate.  The certified
    profile itself is returned under ``"reference"`` (numpy array, on the grid of
    the second-finest level, i.e. the grid the finer level coarsens onto
    exactly).
    """
    if len(levels) < 3:
        return {"verdict": WITHHELD_TOO_FEW_LEVELS, "classification": "insufficient_levels",
                "gate_usable": False, "tolerance_percent": float(tolerance_percent),
                "n_levels": len(levels), "reference": None, "band_percent": None,
                "band_share_of_gate": None, "reasons": ["a GCI needs three levels"],
                "extrapolation_displacement_percent": None, "displacement_ratio": None,
                "reference_ncells": None, "ladder": [int(lv["ncells"]) for lv in levels]}
    levels = sorted(levels, key=lambda lv: int(lv["ncells"]))
    base = int(levels[0]["ncells"])
    for lv in levels:
        if int(lv["ncells"]) % base:
            return {"verdict": WITHHELD_NON_NESTING, "classification": "non_nesting",
                "gate_usable": False, "tolerance_percent": float(tolerance_percent),
                "ladder": [int(x["ncells"]) for x in levels], "reference": None,
                "band_percent": None, "band_share_of_gate": None,
                "reasons": ["grids do not nest"], "reference_ncells": None}
    prof = [coarsen(lv["field"], int(lv["ncells"]) // base) for lv in levels]
    diffs = []
    for i in range(len(prof) - 1):
        # normalised by the *finer* profile, so the differences are the
        # same numbers the shipped round-2 driver reports
        d = rel_l1(prof[i], prof[i + 1])
        diffs.append({"levels": [int(levels[i]["ncells"]), int(levels[i + 1]["ncells"])],
                      "refinement": int(levels[i + 1]["ncells"]) // max(int(levels[i]["ncells"]), 1),
                      "rel_l1": d, "rel_l1_percent": 100.0 * d})
    orders = []
    for i in range(len(diffs) - 1):
        r_loc = float(levels[i + 2]["ncells"]) / float(levels[i + 1]["ncells"])
        # a zero difference carries no order information (round-off floor or a
        # ladder that stopped moving): keep it as None instead of raising, so the
        # ladder is *refused* by :func:`classify` rather than silently accepted
        if diffs[i]["rel_l1"] <= 0.0 or diffs[i + 1]["rel_l1"] <= 0.0:
            p_loc = None
        else:
            p_loc = observed_order(diffs[i]["rel_l1"], diffs[i + 1]["rel_l1"], r_loc)
        orders.append({"levels": [int(levels[k]["ncells"]) for k in (i, i + 1, i + 2)],
                       "refinement": r_loc, "observed_order": p_loc})
    d_fine = diffs[-1]["rel_l1"]
    p_used = orders[-1]["observed_order"]
    r_fine = float(levels[-1]["ncells"]) / float(levels[-2]["ncells"])
    case, verdict = classify(p_used, d_fine, p_min=p_min, p_max=p_max,
                             roundoff_floor=roundoff_floor)

    band = (gci(diffs[-2]["rel_l1"], d_fine, p_used, fs=fs, r=r_fine)
            if p_used is not None and p_used > 0.0 else float("inf"))
    band_percent = 100.0 * band
    band_share = band_percent / float(tolerance_percent)
    reasons = []
    if verdict != CERTIFIED:
        reasons.append(f"ladder classified {case}")
    band_ok = bool(np.isfinite(band_percent) and band_share <= u_share_max)
    if not band_ok:
        # the band reason is always recorded; it *becomes* the verdict only when
        # the ladder itself is asymptotic, so the primary defect is what shows
        reasons.append(
            f"band {band_percent:.6f} % is {band_share*100:.1f} % of the "
            f"{float(tolerance_percent):g} % gate (max {u_share_max*100:.1f} %)")
        if verdict == CERTIFIED:
            verdict = WITHHELD_BAND_CONSUMES_GATE
    gate_usable = bool(verdict == CERTIFIED)

    ref_raw = np.asarray(levels[-2]["field"], dtype=float)
    fine_on_ref = coarsen(levels[-1]["field"], int(round(r_fine)))
    reference = None
    displacement_percent = displacement_expected_percent = None
    displacement_ratio = None
    if p_used is not None and p_used > 0.0 and np.isfinite(p_used):
        reference = richardson_extrapolate(fine_on_ref, ref_raw, p_used, r=r_fine)
        # measured displacement of the certified profile from the raw level it
        # replaces, and the identity that *should* hold for it
        displacement = rel_l1(reference, ref_raw)
        expected = rel_l1(fine_on_ref, ref_raw) * r_fine ** p_used / (r_fine ** p_used - 1.0)
        displacement_percent = 100.0 * displacement
        displacement_expected_percent = 100.0 * expected
        displacement_ratio = (displacement / expected) if expected > 0.0 else None

    out = {
        "verdict": verdict,
        "classification": case,
        "reasons": reasons,
        "gate_usable": gate_usable,
        "tolerance_percent": float(tolerance_percent),
        "u_share_max": float(u_share_max),
        "fs_gci": float(fs),
        "p_min": float(p_min),
        "p_max": float(p_max),
        "ladder": [int(lv["ncells"]) for lv in levels],
        "differences": diffs,
        "orders": orders,
        "observed_order_used": p_used if p_used is None else float(p_used),
        "refinement_fine": r_fine,
        "band_fraction": None if not np.isfinite(band) else float(band),
        "band_percent": None if not np.isfinite(band_percent) else float(band_percent),
        "band_share_of_gate": None if not np.isfinite(band_share) else float(band_share),
        "min_resolvable_error_percent": None if not np.isfinite(band_percent)
        else float(band_percent),
        "reference_ncells": int(levels[-2]["ncells"]),
        "reference_grid_size": int(ref_raw.size),
        "extrapolation_displacement_percent": displacement_percent,
        "extrapolation_displacement_expected_percent": displacement_expected_percent,
        "displacement_ratio": displacement_ratio,
        "input_sha256": digest_levels(levels),
        "reference": reference,
        "field_raw_finest": np.asarray(levels[-1]["field"], dtype=float),
    }
    return out


def judge(candidate, certification: dict, *, tolerance_percent: float | None = None) -> dict:
    """Compare one candidate profile against a certification.

    The candidate may live on any nesting grid of the reference (it is
    cell-averaged down to the reference grid).  Decisions:

    ``WITHHELD``       the reference itself is not certified / not usable here
    ``FAIL``           ``err >= tau``
    ``PASS``           ``err + U < tau`` -- margin larger than reference noise
    ``PASS_MARGINAL``  ``err < tau <= err + U`` -- passes, but not resolvably
    """
    tau = float(certification["tolerance_percent"] if tolerance_percent is None
                else tolerance_percent)
    ref = certification.get("reference")
    if ref is None:
        return {"decision": WITHHELD, "reason": certification["verdict"],
                "rel_l1_percent": None, "tolerance_percent": tau}
    cand = np.asarray(candidate, dtype=float)
    if cand.size >= ref.size:
        factor = int(round(cand.size / ref.size))
        if factor * ref.size != cand.size:
            raise ValueError(f"candidate size {cand.size} is not a nesting multiple "
                             f"of the reference grid {ref.size}")
        cand_on_ref = coarsen(cand, factor)
    else:
        factor = int(round(ref.size / cand.size))
        if factor * cand.size != ref.size:
            raise ValueError(f"reference size {ref.size} is not a nesting multiple "
                             f"of the candidate grid {cand.size}")
        cand_on_ref = cand
        ref = coarsen(ref, factor)
    err = rel_l1(cand_on_ref, ref)
    err_percent = 100.0 * err
    band = certification.get("band_percent")
    # the gate-validity rule is re-evaluated at the tolerance actually asked
    # about: a reference certified for a 5 % gate is not automatically usable at
    # a 3 % gate
    share_max = float(certification.get("u_share_max", U_SHARE_MAX))
    gate_usable = bool(certification.get("classification") == "monotonic"
                       and certification.get("verdict") in (CERTIFIED, WITHHELD_BAND_CONSUMES_GATE)
                       and band is not None and band / tau <= share_max)
    if not gate_usable:
        decision = WITHHELD
        reason = certification["verdict"]
    elif err_percent >= tau:
        decision, reason = FAIL, "error above tolerance"
    elif band is not None and err_percent + band < tau:
        decision, reason = PASS, "margin larger than the reference band"
    else:
        decision, reason = PASS_MARGINAL, "error below tolerance but margin inside the reference band"
    return {
        "decision": decision,
        "reason": reason,
        "rel_l1_percent": err_percent,
        "rel_l2_percent": 100.0 * float(np.sqrt(np.sum((cand_on_ref - ref) ** 2) /
                                                np.sum(ref ** 2))),
        "tolerance_percent": tau,
        "band_percent": band,
        "band_share_of_gate": None if band is None else band / tau,
        "robust_margin_percent_of_gate": None if not gate_usable or band is None
        else 100.0 * (tau - err_percent - band) / tau,
        "reference_verdict": certification["verdict"],
        "candidate_grid": int(np.asarray(candidate).size),
        "reference_grid": int(ref.size),
    }


def audit_ladder(ladders: dict, *, tolerance_percent: float) -> dict:
    """Certify several named ladders and summarise which ones are usable.

    ``ladders`` maps a name to a list of ``{"ncells", "field"}`` levels.  The
    summary is the machine answer to "can this solver certify itself?".
    """
    out = {}
    for name, levels in ladders.items():
        cert = certify(levels, tolerance_percent=tolerance_percent)
        out[name] = {k: v for k, v in cert.items()
                     if k not in ("reference", "field_raw_finest")}
        out[name]["_reference"] = cert.get("reference")
    usable = [n for n, c in out.items() if c["gate_usable"]]
    return {"tolerance_percent": float(tolerance_percent),
            "ladders": out,
            "usable_certifiers": usable,
            "n_usable": len(usable),
            "n_total": len(out)}


def band_coverage_study(n_cases: int = 200, seed: int = 0,
                        levels=(200, 400, 800), tolerance_percent: float = 5.0) -> dict:
    """Does the band ``U`` actually cover the reference's own error?

    Synthetic ladders with a **known** exact solution and a two-term error model
    ``A*N^-p + B*N^-q`` (``q > p``) are certified exactly like real data, and the
    true relative L1 error of the certified profile is compared with its band.
    The point is falsifiable: a band that is too tight shows up as
    ``coverage_fraction < 1`` or ``worst_true_over_band > 1``.

    Limitation: the smooth ladders here recover a high observed order, so the
    extrapolation removes almost all of the error and the band is *very*
    conservative. That says nothing about the shock case, where the true error
    cannot be measured at all (no exact solution) and the observed order is ~1.
    """
    rng = np.random.default_rng(seed)
    cover = 0
    ratios = []
    n_ok = 0
    for _ in range(int(n_cases)):
        p = rng.uniform(0.8, 3.0)
        q = rng.uniform(p + 0.5, p + 3.0)
        a = rng.uniform(0.1, 1.0)
        b = a * rng.uniform(0.0, 1.0)
        lv = []
        for n in levels:
            x = (np.arange(n) + 0.5) * (10.0 / n) - 5.0
            lv.append({"ncells": int(n), "field": 1.0 + 0.2 * x + a * n ** (-p) + b * n ** (-q)})
        cert = certify(lv, tolerance_percent=tolerance_percent)
        if cert["reference"] is None:
            continue
        n_ref = cert["reference_ncells"]
        x = (np.arange(n_ref) + 0.5) * (10.0 / n_ref) - 5.0
        true_percent = 100.0 * rel_l1(cert["reference"], 1.0 + 0.2 * x)
        ratios.append(true_percent / cert["band_percent"])
        cover += int(true_percent <= cert["band_percent"])
        n_ok += 1
    return {
        "n_cases": n_ok,
        "seed": int(seed),
        "levels": [int(n) for n in levels],
        "coverage_fraction": (cover / n_ok) if n_ok else None,
        "median_true_over_band": float(np.median(ratios)) if ratios else None,
        "worst_true_over_band": float(max(ratios)) if ratios else None,
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()
                if not (isinstance(v, np.ndarray) and v.ndim)}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------- driver
def _load_profile(rawdir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(rawdir) / name, allow_pickle=True) as d:
        return np.asarray(d["x"], dtype=float), np.asarray(d["rho"], dtype=float)


def _shu_osher_audit(rawdir: Path, tolerance_percent: float) -> dict:
    """The shipped CASE-SHU-OSHER ladder set: FV certifier + 3 DG families."""
    raw = Path(rawdir)
    fv_levels = []
    for n in (800, 1600, 3200):
        _, rho = _load_profile(raw, f"fv_cfl0p3_n{n}.npz")
        fv_levels.append({"ncells": n, "field": rho})
    ladders = {"fv_weno5": fv_levels}
    for degree in (1, 2, 3):
        lv = []
        for n in (200, 400, 800):
            _, rho = _load_profile(raw, f"dg_cfl0p4_limweno_n{n}_p{degree}.npz")
            lv.append({"ncells": n, "field": rho})
        ladders[f"dg_p{degree}"] = lv
    audit = audit_ladder(ladders, tolerance_percent=tolerance_percent)
    fv = audit["ladders"]["fv_weno5"]
    legs = []
    ref_cert = certify(fv_levels, tolerance_percent=tolerance_percent)
    leg_specs = [(1, 200, 0.4), (1, 400, 0.4), (1, 800, 0.4),
                 (2, 200, 0.4), (2, 400, 0.4), (2, 800, 0.4),
                 (3, 200, 0.4), (3, 400, 0.4), (3, 800, 0.4),
                 (2, 400, 0.1), (2, 400, 0.2)]
    for degree, ncells, cfl in leg_specs:
        name = f"dg_cfl{cfl:g}".replace(".", "p") + f"_limweno_n{ncells}_p{degree}.npz"
        _, rho = _load_profile(raw, name)
        res = judge(rho, ref_cert)
        res.update({"configuration": f"DG P{degree} N={ncells} CFL={cfl}",
                    "degree": degree, "ncells": ncells, "cfl": cfl, "file": name})
        res["judge_3pct"] = judge(rho, ref_cert, tolerance_percent=3.0)
        legs.append(res)
    audit["fv_certification"] = {k: v for k, v in fv.items() if k != "_reference"}
    audit["dg_legs"] = legs
    counts = {}
    for lg in legs:
        counts[lg["decision"]] = counts.get(lg["decision"], 0) + 1
    counts3 = {}
    for lg in legs:
        d3 = lg["judge_3pct"]["decision"]
        counts3[d3] = counts3.get(d3, 0) + 1
    audit["decision_counts_5pct"] = counts
    audit["decision_counts_3pct"] = counts3
    audit["worst_leg"] = max(legs, key=lambda lg: lg["rel_l1_percent"])["configuration"]
    audit["band_coverage"] = band_coverage_study(tolerance_percent=tolerance_percent)
    return audit, ref_cert


def _markdown(audit: dict) -> str:
    fv = audit["fv_certification"]
    lines = [
        "# Certified-reference audit (round-2 practice, engine form)",
        "",
        f"Gate used for the gate-validity rule: **{audit['tolerance_percent']:g} %**, "
        f"max band share {fv['u_share_max']*100:.1f} %, Fs = {fv['fs_gci']}.",
        "",
        "## Which ladders may certify a reference?",
        "",
        "| ladder | levels | p (obs) | band U % | U / gate | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in audit["ladders"].items():
        p = c.get("observed_order_used")
        u = c.get("band_percent")
        share = c.get("band_share_of_gate")
        lines.append(
            f"| {name} | {c.get('ladder')} | {'n/a' if p is None else f'{p:.4f}'} | "
            f"{'n/a' if u is None else f'{u:.6f}'} | "
            f"{'n/a' if share is None else f'{share:.3f}'} | {c['verdict']} |")
    lines += ["", f"Usable certifiers: {audit['usable_certifiers']} "
                  f"({audit['n_usable']}/{audit['n_total']}).", "",
              "## Shipped DG legs judged against the certified FV reference", "",
              "| configuration | err % | + U | decision (5 % gate) | decision (3 % gate) |",
              "|---|---|---|---|---|"]
    for lg in audit["dg_legs"]:
        lines.append(f"| {lg['configuration']} | {lg['rel_l1_percent']:.6f} | "
                     f"{lg['rel_l1_percent'] + lg['band_percent']:.6f} | {lg['decision']} | "
                     f"{lg['judge_3pct']['decision']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rawdir", default="validation/repro_shu_osher/artifacts/raw")
    ap.add_argument("--outdir", default="artifacts/m17539")
    ap.add_argument("--tolerance", type=float, default=5.0)
    args = ap.parse_args(argv)
    audit, ref_cert = _shu_osher_audit(Path(args.rawdir), args.tolerance)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "certified_reference_audit.json").write_text(
        json.dumps(_jsonable(audit), indent=2, sort_keys=True) + "\n")
    (outdir / "certified_reference_audit.md").write_text(_markdown(audit))
    np.savez_compressed(outdir / "certified_reference_fv.npz",
                        reference=ref_cert["reference"])
    print(_markdown(audit))
    print(f"wrote {outdir}/certified_reference_audit.{{json,md}}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())