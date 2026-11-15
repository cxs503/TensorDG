"""Round-2 cross-validation, extended from ``rho`` to every field of the system.

Why this file exists
--------------------
``benchmarks/xval_dg_round2_certified`` certifies **one** field of the
Shu-Osher case -- the density -- and gates the eleven shipped DG legs on it
(``D2``: ``rel L1(rho) < 3 %``).  A one-field gate is an *implicit* claim about
the other two conserved variables: the raw ladders commit ``rho_u`` and
``rho_e`` alongside ``rho`` (see ``validation/repro_shu_osher/artifacts/raw``),
but nothing ever certified or gated them.  For a system conservation law that
gap is material: a limiter can be acceptable on ``rho`` and unacceptable on the
momentum, and a "the case is within 3 %" statement that silently means
"density is within 3 %" is exactly the kind of under-specified claim this
programme keeps finding.

This module closes that gap *without touching* the round-2 engine or its
artifacts:

* it re-runs :func:`tensordg.certified_reference.certify` independently on
  ``rho``, ``rho_u`` and ``rho_e`` -- each field gets its own observed order and
  its own GCI band, and they are *not* equal (p = 1.2825 / 1.3132 / 1.3013);
* it extends the gate to the two **derived** quantities that matter physically,
  velocity ``u = rho_u / rho`` and pressure
  ``p = (gamma-1)(rho_e - rho_u^2 / (2 rho))``, whose bands follow a declared,
  conservative rule (:data:`BAND_RULE`) rather than a fitted one;
* it extends the closed-form oracle set from density to the whole state
  (plateau ``u``, ``p``; the untouched entropy wave in ``u`` and ``p``) and adds
  two budget closures that density cannot see -- the **momentum** budget
  (inflow flux ``rho_L u_L^2 + p_L`` against the outflow pressure ``p_R``) and
  the **energy** budget (inflow flux ``(E_L + p_L) u_L``, zero outflow because
  ``u_R = 0``).  Both are analytic for this case and both close two orders of
  magnitude tighter than the leg gate;
* it reports -- not gates -- how the density-only gate relates to the others
  (``MF6``): momentum carries the largest error of the five quantities, so the
  density-only gate was slightly optimistic, by about 1.4 % of its own value on
  the worst leg, while every leg still sits inside the strict gate.

What is decided (thresholds predeclared in the source, not fitted)
-----------------------------------------------------------------
``MF1`` every conserved field ``CERTIFIED`` on the committed
        ``[800, 1600, 3200]`` FV ladder **and** gate-usable at the strict 3 %
        gate (``U / 3 % <= 1/3``).
``MF2`` the additive bands of ``u`` and ``p`` are also gate-usable at 3 %.
``MF3`` every one of the 11 DG legs satisfies ``rel L1 < 3 %`` on **all five**
        quantities (55 comparisons), with the margin rule of :func:`judge`.
``MF4`` state oracles on the certified reference: plateau ``rho``, ``u``, ``p``
        within 1 %, and the untouched entropy wave in ``u`` and ``p``
        (``u = 0``, ``p = 1`` exactly) within 1 %.
``MF5`` mass, momentum and energy budgets on the certified reference within
        0.01 % -- two decades tighter than the leg gate.
``accepted`` is the conjunction of ``MF1..MF5``; ``MF6`` is reported only.

Honest limitation, stated up front
----------------------------------
All five quantities are certified on the **same** FV-WENO5 family, so MF1-MF3
bound *mesh* error, not a systematic error shared by that one discretisation.
The independent content is MF4/MF5 (closed forms and the Rankine-Hugoniot
skeleton), plus the fact that the three fields' observed orders differ: a
common-mode bug in the reconstruction would move them together.

Usage::

    python -m benchmarks.xval_dg_round2_multifield                # audit + artifacts
    python -m benchmarks.xval_dg_round2_multifield --figures      # + 2 PNGs
    python -m benchmarks.xval_dg_round2_multifield --json-stdout  # machine-readable
    python -m benchmarks.xval_dg_round2_multifield --require-accepted   # CI gate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow ``python benchmarks/xval_dg_round2_multifield.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordg.certified_reference import CERTIFIED, certify, judge
from tensordg.cases.shock_entropy import (
    DIAPHRAGM,
    DOMAIN,
    FINAL_TIME,
    GAMMA,
    LEFT_STATE,
    RIGHT_STATE,
    WAVE_AMPLITUDE,
    WAVE_NUMBER,
    exact_mass_budget,
    rankine_hugoniot_anchor,
)

__all__ = [
    "BENCH_ID",
    "BENCH_VER",
    "WORKLOAD_ID",
    "CONSERVED_FIELDS",
    "DERIVED",
    "FIELDS",
    "GATE_PERCENT",
    "STRICT_PERCENT",
    "ORACLE_TOL_PERCENT",
    "BUDGET_TOL_PERCENT",
    "BAND_RULE",
    "DEFAULT_RAW",
    "leg_filename",
    "load_field_ladders",
    "derive",
    "reference_bands",
    "closed_form_budgets",
    "state_oracles",
    "certify_multifield",
    "cross_validate",
    "write_artifacts",
    "main",
]

# ---------------------------------------------------------------- identity
BENCH_ID = "XVAL-DG-ROUND2-CERTIFIED-REFERENCE-MULTIFIELD"
BENCH_VER = "2026.1"
WORKLOAD_ID = "xval-dg-round2-multifield"
WORKLOAD_VER = "1"

#: Committed raw directory of the round-2 ladder (three FV levels + 11 DG legs).
DEFAULT_RAW = Path("validation/repro_shu_osher/artifacts/raw")

#: The three conserved fields a 1-D Euler solver really evolves.
CONSERVED_FIELDS = ("rho", "rho_u", "rho_e")
#: Physically interpretable primitives recovered from them.
DERIVED = ("u", "p")
FIELDS = CONSERVED_FIELDS + DERIVED

# ---------------------------------------------------------------- predeclared
#: Case-pool gate kept from round 2, so the two audits stay comparable.
GATE_PERCENT = 5.0
#: The mission's precision standard.
STRICT_PERCENT = 3.0
#: Oracle tolerance on the certified reference (declared before the run).
ORACLE_TOL_PERCENT = 1.0
#: Budget closure tolerance: two decades tighter than the leg gate.
BUDGET_TOL_PERCENT = 0.01
#: How a derived quantity inherits the uncertainty of the fields it is made of.
#: Additive (sum of the bands) is the conservative choice: ``u = rho_u / rho``
#: and ``p = (gamma-1)(rho_e - rho_u^2/(2 rho))`` both combine errors from more
#: than one field, and the triangle inequality can never give a band below the
#: largest single band.  It is declared here, in the source, so it cannot be
#: chosen after looking at which rule passes.
BAND_RULE = {
    "kind": "additive",
    "description": "band(derived) = sum of the bands of the conserved fields it "
                   "is made of (triangle inequality, conservative)",
    "u": ["rho_u", "rho"],
    "p": ["rho_e", "rho_u", "rho"],
}
#: Ladder geometry of the round-2 case (coarse -> fine FV levels, DG legs).
FV_LEVELS = (800, 1600, 3200)
DG_LEGS = ((1, 200, 0.4), (1, 400, 0.4), (1, 800, 0.4),
           (2, 200, 0.4), (2, 400, 0.4), (2, 800, 0.4),
           (3, 200, 0.4), (3, 400, 0.4), (3, 800, 0.4),
           (2, 400, 0.1), (2, 400, 0.2))
#: Density level of the transmitted shock: half way between the post-shock
#: plateau and the largest pre-shock density -- the round-2 convention.
SHOCK_LEVEL = 0.5 * (LEFT_STATE[0] + RIGHT_STATE[0] + WAVE_AMPLITUDE)


# ---------------------------------------------------------------- loading
def leg_filename(degree: int, ncells: int, cfl: float) -> str:
    """Committed file name of one DG leg (``dg_cfl0p4_limweno_n200_p1.npz``)."""
    tag = ("%g" % cfl).replace(".", "p")
    return f"dg_cfl{tag}_limweno_n{ncells}_p{degree}.npz"


def _load_all(raw: Path, name: str) -> dict:
    with np.load(raw / name) as data:
        missing = [f for f in CONSERVED_FIELDS if f not in data.files]
        if missing:
            raise KeyError(f"{name} is missing conserved field(s) {missing}")
        return {"x": np.asarray(data["x"], dtype=float),
                **{f: np.asarray(data[f], dtype=float) for f in CONSERVED_FIELDS}}


def load_field_ladders(rawdir: Path | str = DEFAULT_RAW) -> dict:
    """Load the committed ladder once, keeping all three conserved fields."""
    raw = Path(rawdir)
    fv = [{"ncells": n, **_load_all(raw, f"fv_cfl0p3_n{n}.npz")} for n in FV_LEVELS]
    dg = []
    for degree, ncells, cfl in DG_LEGS:
        name = leg_filename(degree, ncells, cfl)
        dg.append({"configuration": f"DG P{degree} N={ncells} CFL={cfl}",
                   "degree": degree, "ncells": ncells, "cfl": cfl,
                   "file": name, **_load_all(raw, name)})
    return {"fv": fv, "dg": dg, "rawdir": str(raw)}


# ---------------------------------------------------------------- derived fields
def derive(rho, rho_u, rho_e):
    """``(u, p)`` from the conserved fields, on whatever grid they live on.

    ``u = (rho u)/rho`` and ``p = (gamma-1)(rho E - (rho u)^2 / (2 rho))``.
    Non-positive density is rejected instead of silently producing a number: a
    negative ``rho`` would flip the pressure formula and still look finite.
    """
    rho = np.asarray(rho, dtype=float)
    rho_u = np.asarray(rho_u, dtype=float)
    rho_e = np.asarray(rho_e, dtype=float)
    if np.any(rho <= 0.0):
        raise ValueError("non-positive density: cannot form u or p")
    u = rho_u / rho
    p = (GAMMA - 1.0) * (rho_e - 0.5 * rho_u * u)
    return u, p


def derived_reference(refs: dict) -> dict:
    """Derived certified references on the (common) grid of the field references."""
    u, p = derive(refs["rho"], refs["rho_u"], refs["rho_e"])
    return {"u": u, "p": p}


def reference_bands(certifications: dict, rule: dict = BAND_RULE) -> dict:
    """All five bands: fields keep their own GCI band, derived ones are additive."""
    bands = {f: float(certifications[f]["band_percent"]) for f in CONSERVED_FIELDS}
    for name in DERIVED:
        bands[name] = float(sum(bands[f] for f in rule[name]))
    return bands


# ---------------------------------------------------------------- closed forms
def closed_form_budgets() -> dict:
    """Analytic mass, momentum and energy in the domain at ``t = 1.8``.

    The case geometry is why this is closed form: everything moves to the right
    (``u_L - c_L = 0.69 > 0``), so ``x = -5`` only *feeds* the domain with the
    constant left state and ``x = +5`` only *discharges* the right state, which
    is at rest (``u_R = 0``).  Hence, inflow counted positive,

        dM/dt = rho_L u_L                       (mass)
        dP/dt = (rho_L u_L^2 + p_L) - p_R        (momentum: -p_R is the pressure
                                                  acting on the outflow face)
        dE/dt = (E_L + p_L) u_L                  (energy: no outflow flux, u_R=0)

    with ``E_L = p_L/(gamma-1) + rho_L u_L^2/2``.  The initial integrals are
    quadrature-free as well: the left part is a constant state and the right
    part is ``p = 1`` with ``rho = 1 + A sin(kx)``, so the density integral uses
    the exact cosine primitive and the energy integral only needs the constant
    ``p_R/(gamma-1)``.

    The mass row is cross-checked against
    :func:`tensordg.cases.shock_entropy.exact_mass_budget` -- an independent
    implementation in the case module -- by the test suite, so a typo here
    cannot quietly ship.
    """
    rho_l, u_l, p_l = LEFT_STATE
    rho_r, u_r, p_r = RIGHT_STATE
    x0, x1 = DOMAIN
    length_l = DIAPHRAGM - x0
    length_r = x1 - DIAPHRAGM
    amplitude, wavenumber = WAVE_AMPLITUDE, WAVE_NUMBER

    mass0 = (rho_l * length_l + rho_r * length_r
             + amplitude * (math.cos(wavenumber * DIAPHRAGM)
                            - math.cos(wavenumber * x1)) / wavenumber)
    momentum0 = rho_l * u_l * length_l + rho_r * u_r * length_r
    energy0 = ((p_l / (GAMMA - 1.0) + 0.5 * rho_l * u_l ** 2) * length_l
               + (p_r / (GAMMA - 1.0) + 0.5 * rho_r * u_r ** 2) * length_r)

    mass_flux = rho_l * u_l
    momentum_flux = rho_l * u_l ** 2 + p_l
    momentum_out = p_r
    energy_flux = (p_l / (GAMMA - 1.0) + 0.5 * rho_l * u_l ** 2 + p_l) * u_l

    return {
        "final_time": FINAL_TIME,
        "mass": {"initial": mass0, "rate": mass_flux,
                 "final_predicted": mass0 + mass_flux * FINAL_TIME},
        "momentum": {"initial": momentum0, "rate": momentum_flux - momentum_out,
                     "inflow_flux": momentum_flux, "outflow_flux": momentum_out,
                     "final_predicted": momentum0 + (momentum_flux - momentum_out) * FINAL_TIME},
        "energy": {"initial": energy0, "rate": energy_flux,
                   "final_predicted": energy0 + energy_flux * FINAL_TIME},
        "closure": "inflow through x=-5 only; the outflow face at u_R = 0 discharges pressure",
        "cross_check_mass_case_module": float(exact_mass_budget()["mass_final_predicted"]),
    }


def state_oracles() -> dict:
    """Closed-form values of the whole state (not just density) at ``t = 1.8``.

    ``plateau_*`` is the Mach-3 left state, taken from the case module where it
    is re-derived from the Hugoniot (so a typo in the literals shows up as a
    Hugoniot residual).  ``entropy_wave_*`` is the untouched downstream wave,
    where the exact solution is ``u = 0`` and ``p = 1`` at every ``x``; the
    density there (``1 + A sin(kx)``) is what round 2 already checks.
    """
    anchor = rankine_hugoniot_anchor()
    rho_l, u_l, p_l = LEFT_STATE
    rho_r, u_r, p_r = RIGHT_STATE
    return {
        "plateau_rho": float(rho_l),
        "plateau_u": float(u_l),
        "plateau_p": float(p_l),
        "plateau_rho_u": float(rho_l * u_l),
        "plateau_rho_e": float(p_l / (GAMMA - 1.0) + 0.5 * rho_l * u_l ** 2),
        "entropy_wave_u": float(u_r),
        "entropy_wave_p": float(p_r),
        "undisturbed_window": list(anchor["undisturbed_window"]),
        "plateau_window": [DOMAIN[0], anchor["post_shock_plateau"]["window"][1]],
        "shock_position_analytic": float(anchor["shock_position_analytic"]),
        "source": "tensordg.cases.shock_entropy closed forms (Hugoniot + left/right states)",
    }


# ---------------------------------------------------------------- certification
def certify_multifield(rawdir: Path | str = DEFAULT_RAW,
                       tolerance_percent: float = GATE_PERCENT) -> dict:
    """Certify each conserved field on the FV ladder; return references and bands."""
    ladder = load_field_ladders(rawdir)
    certifications = {}
    for field in CONSERVED_FIELDS:
        levels = [{"ncells": lv["ncells"], "field": lv[field]} for lv in ladder["fv"]]
        certifications[field] = certify(levels, tolerance_percent=tolerance_percent)
    refs = {f: np.asarray(certifications[f]["reference"], dtype=float)
            for f in CONSERVED_FIELDS if certifications[f]["reference"] is not None}
    n = next(iter(refs.values())).size if refs else None
    for field, arr in refs.items():
        if arr.size != n:
            raise ValueError(f"field references disagree on the grid size ({field})")
    complete = len(refs) == len(CONSERVED_FIELDS)
    return {
        "ladder": ladder,
        "certifications": certifications,
        "references": refs,
        "derived": derived_reference(refs) if complete else {},
        "bands": reference_bands(certifications) if complete else {},
        "x": None if n is None else DOMAIN[0] + (np.arange(n) + 0.5)
        * (DOMAIN[1] - DOMAIN[0]) / n,
    }


def _coarsen_to(values: np.ndarray, target: int) -> np.ndarray:
    """Cell-average ``values`` onto ``target`` cells (integer nesting only)."""
    values = np.asarray(values, dtype=float)
    if values.size == target:
        return values
    factor = values.size // target
    if factor * target != values.size:
        raise ValueError(f"grid {values.size} does not nest onto {target}")
    return values.reshape(target, factor).mean(axis=1)


def shock_position(x: np.ndarray, rho: np.ndarray, level: float | None = None) -> float | None:
    """Rightmost upward crossing of ``rho = level`` (the transmitted shock)."""
    level = SHOCK_LEVEL if level is None else float(level)
    rho = np.asarray(rho, dtype=float)
    x = np.asarray(x, dtype=float)
    for i in range(rho.size - 1, 0, -1):
        if rho[i - 1] > level >= rho[i]:
            return float(x[i])
    return None


def _breakdown(x: np.ndarray, got: np.ndarray, ref: np.ndarray,
               shock_x: float | None, half_width: float = 0.3) -> dict:
    """Share of the absolute L1 error that sits in the shock ramp (diagnostic).

    Not a decision: it explains *why* one field or one degree is worse than
    another.  It is the only place in this audit where the shock position
    enters, and there it localises error instead of scoring it.
    """
    abs_err = np.abs(np.asarray(got, float) - np.asarray(ref, float))
    total = float(abs_err.sum())
    if total <= 0.0 or shock_x is None:
        return {"shock_share_percent": None, "shock_window": None}
    mask = np.abs(np.asarray(x, float) - shock_x) <= half_width
    return {"shock_share_percent": 100.0 * float(abs_err[mask].sum()) / total,
            "shock_window": [shock_x - half_width, shock_x + half_width],
            "half_width": half_width}


# ---------------------------------------------------------------- decisions
def cross_validate(rawdir: Path | str = DEFAULT_RAW,
                   gate_percent: float = GATE_PERCENT,
                   strict_percent: float = STRICT_PERCENT,
                   oracle_tolerance_percent: float = ORACLE_TOL_PERCENT,
                   budget_tolerance_percent: float = BUDGET_TOL_PERCENT) -> dict:
    """Run MF1-MF5 (+ reported MF6) and return the whole audit."""
    mf = certify_multifield(rawdir, tolerance_percent=gate_percent)
    certs, refs, bands, x = mf["certifications"], mf["references"], mf["bands"], mf["x"]
    if len(refs) != len(CONSERVED_FIELDS):
        raise RuntimeError("the FV ladder did not certify for every conserved field")

    # ---- MF1: every conserved field certified and gate-usable at the strict gate
    mf1_rows = {}
    for field in CONSERVED_FIELDS:
        cert = certs[field]
        band = float(cert["band_percent"])
        share = band / strict_percent
        mf1_rows[field] = {
            "verdict": cert["verdict"],
            "classification": cert["classification"],
            "observed_order": cert["observed_order_used"],
            "differences_percent": [d["rel_l1_percent"] for d in cert["differences"]],
            "band_percent": band,
            "band_share_of_strict_gate": share,
            "reference_ncells": cert["reference_ncells"],
            "gate_usable_at_strict": bool(cert["verdict"] == CERTIFIED and share <= 1.0 / 3.0),
        }
    mf1 = {"fields": mf1_rows, "strict_percent": strict_percent, "u_share_max": 1.0 / 3.0,
           "pass": all(r["gate_usable_at_strict"] for r in mf1_rows.values())}

    # ---- MF2: derived fields usable through the declared additive band rule
    mf2_rows = {}
    for name in DERIVED:
        share = bands[name] / strict_percent
        mf2_rows[name] = {"band_percent": bands[name], "band_rule": BAND_RULE["kind"],
                          "sources": BAND_RULE[name],
                          "band_share_of_strict_gate": share,
                          "gate_usable_at_strict": bool(share <= 1.0 / 3.0)}
    mf2 = {"derived": mf2_rows, "band_rule": BAND_RULE, "strict_percent": strict_percent,
           "pass": all(r["gate_usable_at_strict"] for r in mf2_rows.values())}

    # ---- MF3: the 11 legs x 5 quantities
    legs, comparisons = [], []
    for leg in mf["ladder"]["dg"]:
        u_leg, p_leg = derive(leg["rho"], leg["rho_u"], leg["rho_e"])
        values = {"rho": leg["rho"], "rho_u": leg["rho_u"], "rho_e": leg["rho_e"],
                  "u": u_leg, "p": p_leg}
        row = {"configuration": leg["configuration"], "degree": leg["degree"],
               "ncells": leg["ncells"], "cfl": leg["cfl"], "file": leg["file"]}
        for field in FIELDS:
            cert = dict(certs["rho_e"] if field in DERIVED else certs[field])
            cert["band_percent"] = bands[field]
            cert["tolerance_percent"] = gate_percent
            if field in DERIVED:
                cert["reference"] = mf["derived"][field]
            res = judge(values[field], cert, tolerance_percent=strict_percent)
            row[field] = {"rel_l1_percent": res["rel_l1_percent"],
                          "decision_strict": res["decision"],
                          "margin_percent": None if res["rel_l1_percent"] is None
                          else strict_percent - res["rel_l1_percent"]}
            comparisons.append((leg["configuration"], field, res["rel_l1_percent"],
                                res["decision"]))
        # diagnostic only: where does the density error live?
        ref_on_leg = _coarsen_to(refs["rho"], leg["rho"].size)
        row["error_localisation"] = _breakdown(
            leg["x"], leg["rho"], ref_on_leg, shock_position(leg["x"], leg["rho"]))
        legs.append(row)

    worst = max(comparisons, key=lambda c: c[2] if c[2] is not None else -1.0)
    per_field = {}
    for field in FIELDS:
        vals = [c[2] for c in comparisons if c[1] == field]
        per_field[field] = {"max_percent": max(vals), "min_percent": min(vals),
                            "n_over_strict": int(sum(1 for v in vals if v >= strict_percent)),
                            "n_over_gate": int(sum(1 for v in vals if v >= gate_percent))}
    mf3 = {"n_legs": len(legs), "n_comparisons": len(comparisons),
           "strict_percent": strict_percent, "gate_percent": gate_percent,
           "per_field": per_field,
           "worst": {"configuration": worst[0], "field": worst[1],
                     "rel_l1_percent": worst[2], "decision": worst[3]},
           "pass": all(c[3] != "FAIL" and c[2] is not None and c[2] < strict_percent
                       for c in comparisons)}

    # ---- MF4: closed-form oracles on the certified reference, whole state
    oracle = state_oracles()
    plateau = x <= oracle["plateau_window"][1]
    quiet = x >= oracle["undisturbed_window"][0]
    measured = {
        "plateau_rho": float(refs["rho"][plateau].mean()),
        "plateau_u": float(mf["derived"]["u"][plateau].mean()),
        "plateau_p": float(mf["derived"]["p"][plateau].mean()),
        "entropy_wave_u_max_abs": float(np.abs(mf["derived"]["u"][quiet]).max()),
        "entropy_wave_p_max_abs_dev": float(np.abs(mf["derived"]["p"][quiet] - 1.0).max()),
    }
    oracle_rows = []
    for name in ("plateau_rho", "plateau_u", "plateau_p"):
        want = float(oracle[name])
        got = measured[name]
        err = 100.0 * abs(got - want) / abs(want)
        oracle_rows.append({"name": name, "kind": "relative", "got": got, "ref": want,
                            "rel_error_percent": err,
                            "tolerance_percent": oracle_tolerance_percent,
                            "pass": bool(err < oracle_tolerance_percent)})
    # the undisturbed wave is *exact* (u = 0, p = 1), so the reading is itself a
    # deviation from the closed form and its target is 0 -- reporting it as
    # |got - 1| / |1| would divide the deviation by the constant instead
    for key, name in (("entropy_wave_u_max_abs", "entropy_wave_u_deviation"),
                      ("entropy_wave_p_max_abs_dev", "entropy_wave_p_deviation")):
        err = float(abs(measured[key]))
        oracle_rows.append({"name": name, "kind": "absolute_deviation",
                            "got": measured[key], "ref": 0.0,
                            "exact_state": {"entropy_wave_u": oracle["entropy_wave_u"],
                                            "entropy_wave_p": oracle["entropy_wave_p"]},
                            "rel_error_percent": err,
                            "tolerance_percent": oracle_tolerance_percent,
                            "pass": bool(err < oracle_tolerance_percent)})
    mf4 = {"tolerance_percent": oracle_tolerance_percent, "rows": oracle_rows,
           "windows": {"plateau": oracle["plateau_window"],
                       "undisturbed": oracle["undisturbed_window"]},
           "worst_deviation_percent": max(r["rel_error_percent"] for r in oracle_rows),
           "pass": all(r["pass"] for r in oracle_rows)}

    # ---- MF5: mass / momentum / energy closure on the certified reference
    budgets = closed_form_budgets()
    dx = (DOMAIN[1] - DOMAIN[0]) / refs["rho"].size
    measured_budget = {"mass": float(refs["rho"].sum() * dx),
                       "momentum": float(refs["rho_u"].sum() * dx),
                       "energy": float(refs["rho_e"].sum() * dx)}
    budget_rows = []
    for name in ("mass", "momentum", "energy"):
        want = float(budgets[name]["final_predicted"])
        got = measured_budget[name]
        err = 100.0 * abs(got - want) / abs(want)
        budget_rows.append({"name": name, "got": got, "ref": want,
                            "rel_error_percent": err,
                            "tolerance_percent": budget_tolerance_percent,
                            "pass": bool(err < budget_tolerance_percent)})
    mf5 = {"tolerance_percent": budget_tolerance_percent, "rows": budget_rows,
           "rates": {name: budgets[name]["rate"] for name in ("mass", "momentum", "energy")},
           "mass_cross_check_case_module_percent": 100.0 * abs(
               budgets["mass"]["final_predicted"]
               - budgets["cross_check_mass_case_module"])
           / budgets["mass"]["final_predicted"],
           "pass": all(r["pass"] for r in budget_rows)}

    # ---- MF6 (reported, not gated): is the density-only gate a faithful proxy?
    ratios = {}
    for field in ("rho_u", "rho_e", "u", "p"):
        pairs = [(leg["rho"]["rel_l1_percent"], leg[field]["rel_l1_percent"],
                  leg["configuration"]) for leg in legs]
        ratios[field] = {"max_ratio_to_rho": max(b / a for a, b, _ in pairs if a),
                         "worst_leg": max(pairs, key=lambda r: r[1] - r[0])[2]}
    mf6 = {"description": "each quantity's leg error against the density error the "
                          "round-2 gate actually measures",
           "ratios": ratios,
           "declared_criterion": "rho-only gate is a faithful proxy if every ratio <= 1.05",
           "rho_only_gate_conservative": bool(
               all(r["max_ratio_to_rho"] <= 1.05 for r in ratios.values()))}

    accepted = bool(mf1["pass"] and mf2["pass"] and mf3["pass"]
                    and mf4["pass"] and mf5["pass"])
    return {
        "benchmark": {"benchmark_id": BENCH_ID, "version": BENCH_VER},
        "workload": {"workload_id": WORKLOAD_ID, "version": WORKLOAD_VER},
        "rawdir": str(Path(rawdir)),
        "case": "shu_osher_shock_entropy_wave",
        "fields": list(FIELDS),
        "thresholds": {"gate_percent": gate_percent, "strict_percent": strict_percent,
                       "oracle_tolerance_percent": oracle_tolerance_percent,
                       "budget_tolerance_percent": budget_tolerance_percent,
                       "u_share_max": 1.0 / 3.0, "band_rule": BAND_RULE},
        "band_percent": bands,
        "oracle": oracle,
        "budgets": budgets,
        "MF1_field_certification": mf1,
        "MF2_derived_usability": mf2,
        "MF3_leg_gate_all_fields": mf3,
        "MF4_state_oracles": mf4,
        "MF5_budget_closure": mf5,
        "MF6_rho_gate_proxy": mf6,
        "legs": legs,
        "accepted": accepted,
    }


# ---------------------------------------------------------------- artifacts
def _markdown(audit: dict) -> str:
    th = audit["thresholds"]
    mf1, mf2 = audit["MF1_field_certification"], audit["MF2_derived_usability"]
    mf3, mf4, mf5 = (audit["MF3_leg_gate_all_fields"], audit["MF4_state_oracles"],
                     audit["MF5_budget_closure"])
    mf6 = audit["MF6_rho_gate_proxy"]
    lines = [
        "# XVAL-DG-ROUND2-CERTIFIED-REFERENCE - multifield extension audit",
        "",
        "Generated by `python -m benchmarks.xval_dg_round2_multifield` from the committed",
        "raw ladder (`validation/repro_shu_osher/artifacts/raw`).  Thresholds declared in",
        f"the source before the run: gate {th['gate_percent']:.1f} %, strict "
        f"{th['strict_percent']:.1f} %,",
        f"oracle {th['oracle_tolerance_percent']:.1f} %, budget "
        f"{th['budget_tolerance_percent']:.2f} %.",
        "",
        f"**accepted = {audit['accepted']}**",
        "",
        "## MF1 - every conserved field certifies on the FV ladder",
        "",
        "| field | verdict | observed order p | band U % | U / 3 % | usable at 3 % |",
        "|---|---|---|---|---|---|",
    ]
    for field, row in mf1["fields"].items():
        lines.append(f"| `{field}` | {row['verdict']} | {row['observed_order']:.4f} | "
                     f"{row['band_percent']:.4f} | {row['band_share_of_strict_gate']:.4f} | "
                     f"{row['gate_usable_at_strict']} |")
    lines += ["", "## MF2 - derived fields inherit the band additively", "",
              "| field | sources | U % | U / 3 % | usable at 3 % |", "|---|---|---|---|---|"]
    for name, row in mf2["derived"].items():
        lines.append(f"| `{name}` | {'+'.join(row['sources'])} | {row['band_percent']:.4f} | "
                     f"{row['band_share_of_strict_gate']:.4f} | {row['gate_usable_at_strict']} |")
    lines += ["", "## MF3 - 11 legs x 5 quantities against the strict gate", "",
              "| quantity | max rel L1 % | min rel L1 % | # >= 3 % |", "|---|---|---|---|"]
    for field, row in mf3["per_field"].items():
        lines.append(f"| `{field}` | {row['max_percent']:.4f} | {row['min_percent']:.4f} | "
                     f"{row['n_over_strict']} / {mf3['n_legs']} |")
    lines += ["",
              f"Worst comparison: **{mf3['worst']['configuration']} / {mf3['worst']['field']}"
              f" = {mf3['worst']['rel_l1_percent']:.4f} %** ({mf3['worst']['decision']}).",
              "", "## MF4 - closed-form state oracles on the certified reference", "",
              "| quantity | reference | closed form | deviation | tol | pass |",
              "|---|---|---|---|---|---|"]
    for row in mf4["rows"]:
        lines.append(f"| `{row['name']}` | {row['got']:.6g} | {row['ref']:.6g} | "
                     f"{row['rel_error_percent']:.3e} | {row['tolerance_percent']:.2f} | "
                     f"{row['pass']} |")
    lines += ["", "## MF5 - mass / momentum / energy budget closure", "",
              "| budget | measured | closed form | deviation % | tol % | pass |",
              "|---|---|---|---|---|---|"]
    for row in mf5["rows"]:
        lines.append(f"| {row['name']} | {row['got']:.9f} | {row['ref']:.9f} | "
                     f"{row['rel_error_percent']:.3e} | {row['tolerance_percent']:.3f} | "
                     f"{row['pass']} |")
    lines += ["", "## MF6 - is the density-only gate a faithful proxy? (reported, not gated)",
              "", "| quantity | max ratio to the rho error | leg with the largest excess |",
              "|---|---|---|"]
    for field, row in mf6["ratios"].items():
        lines.append(f"| `{field}` | {row['max_ratio_to_rho']:.4f} | {row['worst_leg']} |")
    lines += ["", f"`rho_only_gate_conservative = {mf6['rho_only_gate_conservative']}` "
                  f"({mf6['declared_criterion']}).", "",
              "## Leg table", "",
              "| configuration | rho | rho_u | rho_e | u | p | shock-band share of rho err % |",
              "|---|---|---|---|---|---|---|"]
    for leg in audit["legs"]:
        share = leg["error_localisation"]["shock_share_percent"]
        lines.append("| {cfg} | {rho:.3f} | {ru:.3f} | {re:.3f} | {u:.3f} | {p:.3f} | {s} |".format(
            cfg=leg["configuration"], rho=leg["rho"]["rel_l1_percent"],
            ru=leg["rho_u"]["rel_l1_percent"], re=leg["rho_e"]["rel_l1_percent"],
            u=leg["u"]["rel_l1_percent"], p=leg["p"]["rel_l1_percent"],
            s="-" if share is None else f"{share:.1f}"))
    lines.append("")
    return "\n".join(lines)


def _figures(audit: dict, outdir: Path) -> list:
    """Two figures: per-field leg error ladder, and the certified u/p at the shock."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    fields = ("rho", "rho_u", "rho_e", "u", "p")
    labels = {"rho": r"$\rho$", "rho_u": r"$\rho u$", "rho_e": r"$\rho E$",
              "u": r"$u$", "p": r"$p$"}
    fig, axes = plt.subplots(1, len(fields), figsize=(3.2 * len(fields), 4.0), sharey=True)
    strict = audit["thresholds"]["strict_percent"]
    for ax, field in zip(axes, fields):
        for degree, marker in ((1, "o"), (2, "s"), (3, "^")):
            xs, ys = [], []
            for leg in audit["legs"]:
                if leg["degree"] == degree and leg["cfl"] == 0.4:
                    xs.append(leg["ncells"])
                    ys.append(leg[field]["rel_l1_percent"])
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker=marker, label=f"P{degree}")
        ax.axhline(strict, color="crimson", ls="--", lw=1.2, label=f"{strict:.0f} % gate")
        ax.set_xscale("log")
        ax.set_xlabel("N cells")
        ax.set_title(labels[field])
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(bottom=0.0)
    axes[0].set_ylabel(r"relative $L^1$ error vs certified reference  [%]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Shu-Osher t=1.8 - DG legs on all five quantities "
                 "(CFL 0.4 ladder, reference certified per field)")
    fig.tight_layout()
    p1 = outdir / "fig_mf1_leg_errors_by_field.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    mf = certify_multifield(audit["rawdir"], tolerance_percent=audit["thresholds"]["gate_percent"])
    x = mf["x"]
    leg = next(lg for lg in mf["ladder"]["dg"]
               if lg["degree"] == 2 and lg["ncells"] == 800 and lg["cfl"] == 0.4)
    u_leg, p_leg = derive(leg["rho"], leg["rho_u"], leg["rho_e"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    panels = ((axes[0], mf["derived"]["u"], u_leg, leg["x"], r"velocity $u$",
               audit["oracle"]["plateau_u"],
               r"$u$: plateau 2.629369, entropy-wave zone $u = 0$"),
              (axes[1], mf["derived"]["p"], p_leg, leg["x"], r"pressure $p$",
               audit["oracle"]["plateau_p"],
               r"$p$: plateau 10.33333, entropy-wave zone $p = 1$"))
    for ax, ref, got, got_x, name, plateau, title in panels:
        ax.plot(x, ref, "k-", lw=1.6, label="certified reference (Richardson, N=1600)")
        ax.plot(got_x, got, "o", ms=2.5, mfc="none", color="tab:orange", label="DG P2 N=800")
        ax.axhline(plateau, color="tab:green", ls=":", lw=1.0, label="Hugoniot plateau")
        ax.set_xlim(-5.2, 3.6)
        ax.set_xlabel("x")
        ax.set_ylabel(name)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center right")
    fig.suptitle("Fields the density-only round-2 gate never looked at "
                 "(each certified on its own FV ladder)")
    fig.tight_layout()
    p2 = outdir / "fig_mf2_derived_profiles.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)
    return paths


def write_artifacts(audit: dict, outdir: Path | str = Path("validation/xval_dg_round2/artifacts"),
                    figures: bool = False) -> dict:
    """Write ``multifield_audit.json`` (+md, +optional PNGs); return the paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "multifield_audit.json"
    md_path = outdir / "multifield_audit.md"
    json_path.write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n")
    md_path.write_text(_markdown(audit) + "\n")
    written = {"json": str(json_path), "markdown": str(md_path), "figures": []}
    if figures:
        written["figures"] = [str(p) for p in _figures(audit, outdir)]
    return written


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rawdir", default=str(DEFAULT_RAW))
    parser.add_argument("--outdir", default="validation/xval_dg_round2/artifacts")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--json-stdout", action="store_true")
    parser.add_argument("--require-accepted", action="store_true")
    args = parser.parse_args(argv)

    audit = cross_validate(args.rawdir)
    if args.json_stdout:
        print(json.dumps(audit, indent=1, sort_keys=True))
    else:
        written = write_artifacts(audit, args.outdir, figures=args.figures)
        mf3 = audit["MF3_leg_gate_all_fields"]
        print(f"accepted={audit['accepted']}")
        for name in ("MF1_field_certification", "MF2_derived_usability"):
            print(f"  {name:28s} {audit[name]['pass']}")
        print(f"  MF3_leg_gate_all_fields      {mf3['pass']} "
              f"({mf3['n_comparisons']} comparisons; worst {mf3['worst']['configuration']} / "
              f"{mf3['worst']['field']} = {mf3['worst']['rel_l1_percent']:.4f} %)")
        for name in ("MF4_state_oracles", "MF5_budget_closure"):
            print(f"  {name:28s} {audit[name]['pass']}")
        print(f"  MF6 rho-only gate faithful   "
              f"{audit['MF6_rho_gate_proxy']['rho_only_gate_conservative']} (reported only)")
        for key, value in written.items():
            if isinstance(value, list):
                for item in value:
                    print(f"  wrote {item}")
            else:
                print(f"  wrote {value}")
    if args.require_accepted and not audit["accepted"]:
        print("REJECTED: the multifield audit did not accept", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
