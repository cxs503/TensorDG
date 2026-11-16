"""Power calibration of the CKS1998 §5.3 time-error-attribution instrument.

Why this module exists
----------------------
``benchmarks/cks1998_vortex_order_certification.py`` decides ``SPATIAL_ISOLATED``
from two legs (pre-registered there as ``TIME_FRACTION_MAX = 5 %`` and
``ORDER_DRIFT_MAX = 1 %``)::

    eta   = |e(CFL) - e(CFL/beta)| / e(CFL)     temporal share of the fine-mesh error
    drift = |p(CFL) - p(CFL/beta)| / p(CFL)     order movement under dt -> dt/beta

That study reported ``eta = 0.00 %`` and concluded ``SPATIAL_ISOLATED``, and its
own §5 left one item open ("second-level ``dt -> dt/beta**2`` refinement not
run").  The complementary question was never asked: **could the eta leg have
fired at all in the admissible configuration space?**  A gate that cannot fire
is vacuous, and a verdict resting on a vacuous leg is unsupported -- the
"a REAL criterion must state its power and its rejected negative control" rule
the owner imposed on the platform's criteria.

This module measures that power on the *same* case, the *same* solver path
(``run_case`` is imported, not re-implemented) and without touching any
pre-registered constant of the certified ladder:

1. ``eta(CFL)`` on a CFL ladder inside the solver's admissible window
   ``0 < CFL <= ADMISSIBLE_CFL_MAX`` (the window is a hard guard in
   ``tensordg.euler_dg_2d.solve_euler_dg_2d``); a log-log fit ``eta ~ A CFL**s``
   extrapolates the CFL at which the fitted eta reaches ``ETA_MAX``;
2. a **positive control**: the same mesh and the same spatial operator, marched
   with a *prescribed* ``dt`` that is a multiple of the production step (the
   solver's ``dt=`` knob bypasses the CFL guard).  If the instrument fires there
   (``eta > ETA_MAX`` on a finite run) it has demonstrated detection power, and
   the calibration then says where that power lives -- inside or outside the
   admissible CFL window;
3. the **second-level refinement** ``dt -> dt/beta**2`` the certified study
   declared not run: ``omega = |e(CFL/beta) - e(CFL/beta**2)| / e(CFL/beta)``;
4. the **order-drift leg** at the production CFL, i.e. the leg the verdict
   actually leans on, reported (not re-tuned).

The verdict is one of

* ``GATE_INFORMATIVE`` -- the fitted crossing sits inside the admissible window,
  so a ``SILENT`` reading there really excludes time contamination;
* ``GATE_VACUOUS_IN_ENVELOPE`` -- the crossing lies outside it; the eta leg
  cannot fire on any admissible run of this case, hence ``eta = 0.00 %`` carries
  no information and the verdict stands on the drift leg alone;
* ``SCALING_UNRESOLVED`` -- no credible power law, nothing is claimed.

Pure functions are torch-free and unit tested (``tests/test_cks1998_time_power.py``);
the numeric payload is produced on the cluster and published with its raw records.

Usage
-----
::

    python -m benchmarks.cks1998_vortex_time_power --degrees 1 \
        --shard-out validation/cks1998_vortex_2d/time_power_p1.json
    python -m benchmarks.cks1998_vortex_time_power --assemble
    python -m benchmarks.cks1998_vortex_time_power --local-smoke    # seconds
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.cks1998_vortex_order_certification import (
    CFL0,
    DOMAIN,
    FIELD,
    FINAL_TIME,
    error_norms,
    initial,
    NORM,
    OUT_DIR,
    TIME_PAIR,
    TIME_REFINEMENT,
    run_case,
    time_error_fraction,
)
import tensordg.euler_dg_2d as dg2

REPO = Path(__file__).resolve().parents[1]
POWER_OUT = OUT_DIR / "time_power.json"

#: The production CFL the certified verdict was taken at.
CFL_BASE = CFL0
#: Refinement factor of the certified study (``dt -> dt / beta``).
BETA = TIME_REFINEMENT
#: Second-level refinement: ``dt -> dt / beta**2``.
BETA2 = BETA * BETA
#: Pre-registered eta bound of the certified study, kept verbatim: this module
#: does not re-preregister it, it measures whether the bound can ever be hit.
ETA_MAX = 0.05
#: Order-drift bound of the certified study (reported for completeness).
DRIFT_MAX = 0.01
#: Hard admissible-CFL guard of ``solve_euler_dg_2d``: ``0 < cfl <= 0.5``.
ADMISSIBLE_CFL_MAX = 0.5
#: CFL ladder (all inside the admissible window).  ``CFL_BASE/2`` and
#: ``CFL_BASE*2`` bracket the production point instead of putting it on an edge.
CFL_LADDER = (0.05, 0.1, 0.2, 0.4, 0.5)
#: The ladder points used for the power fit.  ``CFL_BASE/2`` and ``2*CFL_BASE``
#: bracket the production point; the window edge (0.5) is measured but excluded
#: from the fit, since it is not a regular point of the family.
FIT_CFLS = (0.05, 0.1, 0.2, 0.4)
#: CFLs that the solver is expected to *refuse*; recorded as evidence that the
#: admissible window is hard, not a soft convention.
REJECTED_PROBES = (0.6,)
#: Positive-control knots: prescribed ``dt = k * dt(CFL_BASE)``.
DT_KNOTS = (1, 2, 4, 8, 16, 32)
#: An error that grows by more than this factor relative to the production CFL
#: is treated as a lost run, not as a time-error measurement.
BLOWUP_FACTOR = 1.0e3
#: Minimal coefficient of determination for the power law to be used.
SCALING_R2_MIN = 0.9


# ---------------------------------------------------------------------------
# Pure estimators (torch-free, unit-testable)
# ---------------------------------------------------------------------------
def eta_share(base_error: float, refined_error: float) -> float:
    """``|e(CFL) - e(CFL/beta)| / e(CFL)`` -- the certified eta leg."""
    return time_error_fraction([base_error, refined_error])


def loglog_fit(xs, ys):
    """Least-squares ``log y = log A + s log x``; returns ``(A, s, r2)``."""
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if x > 0.0 and y > 0.0 and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        raise ValueError("need at least two positive finite points for a fit")
    lx = np.log([p[0] for p in pairs])
    ly = np.log([p[1] for p in pairs])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return math.exp(float(intercept)), float(slope), r2


def cfl_at_eta(amplitude: float, exponent: float, eta_target: float) -> float:
    """Extrapolate the CFL at which ``A * CFL**s`` reaches ``eta_target``."""
    if amplitude <= 0.0 or eta_target <= 0.0:
        raise ValueError("amplitude and eta_target must be positive")
    if exponent == 0.0:
        raise ValueError("a flat power law never reaches the target")
    ratio = eta_target / amplitude
    return float(ratio ** (1.0 / exponent))


def classify_leg(eta: float, eta_max: float = ETA_MAX) -> str:
    """``FIRE`` when the eta leg reports time contamination, else ``SILENT``."""
    if not math.isfinite(eta):
        return "NOT_MEASURABLE"
    return "FIRE" if eta > eta_max else "SILENT"


def power_verdict(scaling: dict, admissible_max: float) -> str:
    """Could the eta leg have fired on an *admissible* configuration?

    * ``GATE_INFORMATIVE``: the fitted crossing CFL is inside the admissible
      window -- a ``SILENT`` reading at that CFL is informative;
    * ``GATE_VACUOUS_IN_ENVELOPE``: the crossing is above the window, so no
      admissible run reaches ``eta_max``;
    * ``SCALING_UNRESOLVED``: no credible power law (bad fit, non-positive or
      non-finite exponent, or no window).
    """
    exponent = scaling.get("exponent")
    crossing = scaling.get("cfl_at_eta_max")
    r2 = scaling.get("r2")
    if exponent is None or crossing is None or r2 is None or admissible_max is None:
        return "SCALING_UNRESOLVED"
    if not (math.isfinite(exponent) and math.isfinite(crossing)
            and math.isfinite(admissible_max)):
        return "SCALING_UNRESOLVED"
    if exponent <= 0.0 or r2 < SCALING_R2_MIN:
        return "SCALING_UNRESOLVED"
    return "GATE_INFORMATIVE" if crossing <= admissible_max else \
        "GATE_VACUOUS_IN_ENVELOPE"


def stability_envelope(records, reference_cfl: float = CFL_BASE,
                       blowup: float = BLOWUP_FACTOR) -> dict:
    """Separate finite, production-comparable runs from lost ones.

    ``records`` carry ``cfl`` and ``error``.  A run whose error is non-finite or
    exceeds ``blowup`` times the production-CFL error is ``lost``.
    """
    ref = [r for r in records if abs(float(r["cfl"]) - reference_cfl) < 1e-12]
    if not ref:
        return {"status": "NO_REFERENCE", "stable_cfl_max": None}
    ref_error = float(ref[0]["error"])
    if not math.isfinite(ref_error) or ref_error <= 0.0:
        return {"status": "BAD_REFERENCE", "stable_cfl_max": None}
    stable, lost = [], []
    for r in sorted(records, key=lambda x: float(x["cfl"])):
        e = float(r["error"])
        item = {"cfl": float(r["cfl"]), "error": e}
        (lost if (not math.isfinite(e) or e > blowup * ref_error) else stable).append(item)
    return {"status": "OK", "reference_cfl": reference_cfl,
            "reference_error": ref_error, "blowup_factor": blowup,
            "stable_cfl_max": max((x["cfl"] for x in stable), default=None),
            "stable": stable, "lost": lost}


def etas_from_records(records, beta: float = BETA) -> list[dict]:
    """Build ``eta`` samples from any record set with a ``cfl`` field.

    Every base CFL whose ``CFL/beta`` partner exists yields one sample; records
    flagged ``unstable``/``rejected`` are skipped, so no arithmetic ever runs on
    a refused or diverged configuration.
    """
    by_cfl = {}
    for r in records:
        if r.get("unstable") or r.get("rejected"):
            continue
        by_cfl[round(float(r["cfl"]), 12)] = r
    samples = []
    for cfl in sorted(by_cfl):
        partner = next((o for o in by_cfl
                        if abs(o - cfl / beta) < 1e-12 * max(cfl, 1e-30)), None)
        if partner is None:
            continue
        base = float(by_cfl[cfl]["error"])
        refined = float(by_cfl[partner]["error"])
        if not (math.isfinite(base) and math.isfinite(refined)) or base <= 0.0:
            continue
        eta = eta_share(base, refined)
        samples.append({"cfl": cfl, "cfl_refined": partner, "error": base,
                        "error_refined": refined, "eta": eta,
                        "leg": classify_leg(eta),
                        "production_cfl": abs(cfl - CFL_BASE) < 1e-12})
    samples.sort(key=lambda s: s["cfl"])
    return samples


def second_refinement(records, beta: float = BETA) -> dict:
    """``omega``: is the *refined* step already time-converged?

    ``omega = |e(CFL/beta) - e(CFL/beta**2)| / e(CFL/beta)`` at the production
    CFL.  Small ``omega`` closes the certified study's limitation 4; an ``omega``
    comparable to ``eta`` means one refinement was not enough to sit in the
    temporal asymptotic range.
    """
    want = {round(CFL_BASE, 12): "base",
            round(CFL_BASE / beta, 12): "refined",
            round(CFL_BASE / beta ** 2, 12): "refined2"}
    found = {}
    for r in records:
        key = round(float(r["cfl"]), 12)
        if key in want and not r.get("unstable") and not r.get("rejected"):
            found[want[key]] = r
    if len(found) < 3:
        return {"status": "INSUFFICIENT_LEVELS", "available": sorted(found)}
    e = {k: float(v["error"]) for k, v in found.items()}
    if any(not math.isfinite(v) or v <= 0.0 for v in e.values()):
        return {"status": "NON_FINITE"}
    eta_1 = eta_share(e["base"], e["refined"])
    omega = abs(e["refined"] - e["refined2"]) / e["refined"]
    return {"status": "OK", "errors": e, "eta_one_level": eta_1,
            "omega_second_level": omega,
            "omega_over_eta": (omega / eta_1) if eta_1 > 0.0 else None,
            "temporal_asymptote_reached": bool(omega <= ETA_MAX)}


def order_drift_leg(records, degree: int, beta: float = BETA) -> dict:
    """Order movement under ``dt -> dt/beta`` at the production CFL."""
    want = [round(CFL_BASE, 12), round(CFL_BASE / beta, 12)]
    per_cfl = {c: {} for c in want}
    for r in records:
        key = round(float(r["cfl"]), 12)
        if (key in per_cfl and r.get("kind") == "ladder"
                and r.get("mesh") is not None and not r.get("rejected")
                and math.isfinite(float(r["error"]))):
            per_cfl[key][int(r["mesh"])] = r
    design = degree + 1
    meshes = sorted(set(per_cfl[want[0]]) & set(per_cfl[want[1]]))
    if len(meshes) < 2:
        return {"status": "INSUFFICIENT_MESHES", "design_order": design}
    coarse_n, fine_n = meshes[0], meshes[-1]
    orders = {}
    for cfl in want:
        c, f = per_cfl[cfl][coarse_n], per_cfl[cfl][fine_n]
        ec, ef = float(c["error"]), float(f["error"])
        if not (math.isfinite(ec) and math.isfinite(ef)) or ec <= 0 or ef <= 0:
            return {"status": "NON_FINITE", "design_order": design}
        orders[cfl] = math.log(ec / ef) / math.log(float(c["h"]) / float(f["h"]))
    p_base, p_red = orders[want[0]], orders[want[1]]
    drift = abs(p_red - p_base) / abs(p_base) if p_base != 0.0 else float("inf")
    return {"status": "OK", "design_order": design, "order_base": p_base,
            "order_refined": p_red, "drift": drift, "drift_max": DRIFT_MAX,
            "leg": "FIRE" if drift > DRIFT_MAX else "SILENT"}


def positive_control(records) -> dict:
    """Classify the prescribed-``dt`` arm (detection power, not a verdict).

    Each knot ``k`` (``dt = k * dt_production``) is paired with ``k/beta`` so the
    *same* eta instrument is applied at an effective CFL of ``CFL_BASE * k``.
    """
    runs = {int(r["dt_multiple"]): r for r in records
            if r.get("kind") == "dt_control"}
    if not runs:
        return {"status": "NOT_RUN"}
    table = []
    for k in sorted(runs):
        r = runs[k]
        finite = math.isfinite(float(r["error"])) and not r.get("unstable")
        eta = None
        partner = runs.get(k / BETA) if k % BETA == 0 else None
        if finite and partner is not None and math.isfinite(
                float(partner["error"])) and not partner.get("unstable") \
                and float(r["error"]) > 0.0:
            eta = eta_share(float(r["error"]), float(partner["error"]))
        table.append({"dt_multiple": k,
                      "effective_cfl": r["effective_cfl"],
                      "steps": r.get("steps"), "error": r["error"],
                      "eta": eta,
                      "leg": classify_leg(eta) if eta is not None
                      else "NOT_MEASURABLE",
                      "finite": finite})
    fired = [x for x in table if x["leg"] == "FIRE" and x["finite"]]
    return {"status": "OK" if table else "NOT_RUN", "runs": table,
            "fired": bool(fired),
            "first_firing_effective_cfl": (min(x["effective_cfl"] for x in fired)
                                           if fired else None),
            "detection_demonstrated": bool(fired)}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def _record(degree, ncells, kind, cfl=None, log=print, **extra):
    """One solver run, with refused/diverged configurations recorded, not raised."""
    started = time.time()
    try:
        r = run_case(degree, ncells, cfl=CFL_BASE if cfl is None else cfl)
    except ValueError as exc:                     # solver guard (CFL window)
        rec = {"degree": degree, "mesh": ncells, "cfl": cfl, "kind": kind,
               "rejected": True, "reason": str(exc)}
        rec.update(extra)
        log(f"  p{degree} n={ncells} cfl={cfl} REFUSED: {exc}")
        return rec
    error = float(r["norms"][FIELD][NORM])
    rec = {"degree": degree, "mesh": ncells, "h": r["h"],
           "cfl": cfl if cfl is not None else CFL_BASE, "kind": kind,
           "steps": r["steps"], "dt": r["dt"], "error": error,
           "wall_seconds": r["wall_seconds"], "dofs": r["dofs"]}
    rec.update(extra)
    log(f"  p{degree} n={ncells} cfl={rec['cfl']:<8g} steps={rec['steps']:<5d} "
        f"e={error:.6e} dt={r['dt']} ({time.time()-started:.1f}s)")
    return rec


def _record_dt(degree, ncells, dt_multiple, dt_production, log=print, **extra):
    """Positive-control run: prescribed ``dt`` (bypasses the solver's CFL guard).

    Same initial condition, same spatial operator, same error norms as the
    ladder runs -- only the time step is prescribed, which is the documented
    ``dt=`` verification knob of ``solve_euler_dg_2d``.
    """
    dt = dt_production * dt_multiple
    started = time.time()
    try:
        x, y, coeff, metadata = dg2.solve_euler_dg_2d(
            initial, ncells=(ncells, ncells), degree=degree, domain=DOMAIN,
            final_time=FINAL_TIME, cfl=CFL_BASE, dt=dt, periodic=(True, True))
    except (ValueError, FloatingPointError) as exc:
        rec = {"degree": degree, "mesh": ncells, "cfl": None,
               "kind": "dt_control", "dt": dt, "dt_multiple": dt_multiple,
               "effective_cfl": CFL_BASE * dt_multiple, "error": float("nan"),
               "unstable": True, "reason": str(exc)}
        rec.update(extra)
        log(f"  p{degree} n={ncells} dt x{dt_multiple} DIVERGED: {exc}")
        return rec
    norms = error_norms(coeff, x, y, degree, FINAL_TIME)
    error = float(norms[FIELD][NORM])
    # ``cfl=None``: the control arm is keyed by its prescribed step, never by a
    # CFL -- otherwise a diverged knot would shadow the ladder record with the
    # same (mesh, cfl) pair and poison the order-drift leg.
    rec = {"degree": degree, "mesh": ncells,
           "h": (DOMAIN[0][1] - DOMAIN[0][0]) / ncells, "cfl": None,
           "kind": "dt_control", "steps": metadata["steps"], "dt": dt,
           "dt_multiple": dt_multiple, "effective_cfl": CFL_BASE * dt_multiple,
           "error": error, "dofs": metadata["dofs"],
           "wall_seconds": round(time.time() - started, 2)}
    rec.update(extra)
    log(f"  p{degree} n={ncells} dt x{dt_multiple:<3d} steps={rec['steps']:<4d} "
        f"e={error:.6e} ({rec['wall_seconds']}s)")
    return rec


def sweep(degree: int, meshes=None, log=print) -> list[dict]:
    """All runs of one degree: CFL ladder, refused probes, dt control, beta**2."""
    if meshes is None:
        meshes = TIME_PAIR[degree]
    coarse_n, fine_n = meshes
    records = []
    cfls = sorted(set(CFL_LADDER) | {CFL_BASE / BETA2})
    for cfl in cfls:
        records.append(_record(degree, fine_n, "ladder", cfl=cfl, log=log))
    # the production pair's coarse mesh, at the base CFL and its beta partner,
    # feeds the order-drift leg only
    for cfl in (CFL_BASE, CFL_BASE / BETA):
        records.append(_record(degree, coarse_n, "ladder", cfl=cfl, log=log))
    # refused probes: evidence that the CFL window is a hard guard
    for cfl in REJECTED_PROBES:
        records.append(_record(degree, fine_n, "refused_probe", cfl=cfl, log=log))
    # positive control: prescribed dt = k * dt(CFL_BASE), which bypasses the guard
    dt_prod = next((r["dt"] for r in records
                    if r.get("kind") == "ladder" and r.get("mesh") == fine_n
                    and abs(r["cfl"] - CFL_BASE) < 1e-12 and r.get("dt")), None)
    if dt_prod:
        for k in DT_KNOTS:
            records.append(_record_dt(degree, fine_n, k, dt_prod, log=log))
    return records


def assemble(records: list[dict], beta: float = BETA) -> dict:
    """Turn raw records into the calibration payload (pure, no solver)."""
    payload = {"field": FIELD, "norm": NORM, "beta": beta, "beta2": beta * beta,
               "eta_max": ETA_MAX, "drift_max": DRIFT_MAX,
               "production_cfl": CFL_BASE, "admissible_cfl_max": ADMISSIBLE_CFL_MAX,
               "fit_cfls": list(FIT_CFLS), "degrees": {}}
    for degree in sorted(set(int(r["degree"]) for r in records)):
        own = [r for r in records if int(r["degree"]) == degree]
        ladder_meshes = [int(r["mesh"]) for r in own
                         if r.get("kind") == "ladder" and not r.get("rejected")]
        if not ladder_meshes:
            continue
        fine_n = max(ladder_meshes)
        fine = [r for r in own if int(r["mesh"]) == fine_n]
        envelope = stability_envelope(
            [{"cfl": r["cfl"], "error": r["error"]} for r in fine
             if r.get("kind") == "ladder" and not r.get("rejected")])
        lost = {round(x["cfl"], 12) for x in envelope.get("lost", [])}
        marked = [dict(r, unstable=bool(r.get("unstable"))
                       or (r.get("kind") == "ladder"
                           and round(float(r.get("cfl") or 0.0), 12) in lost))
                  for r in own]
        # eta / second refinement are single-mesh quantities: the production
        # pair's coarse mesh and the dt-control arm must not leak into them
        # (a cfl keyed lookup would otherwise average two different meshes).
        fine_marked = [r for r in marked
                       if r.get("kind") == "ladder" and int(r["mesh"]) == fine_n]
        samples = etas_from_records(fine_marked)
        fit_samples = [s for s in samples
                       if any(abs(s["cfl"] - c) < 1e-12 for c in FIT_CFLS)]
        scaling = {"status": "INSUFFICIENT_SAMPLES"}
        if len(fit_samples) >= 2:
            try:
                amplitude, exponent, r2 = loglog_fit(
                    [s["cfl"] for s in fit_samples],
                    [s["eta"] for s in fit_samples])
                scaling = {"status": "OK", "amplitude": amplitude,
                           "exponent": exponent, "r2": r2, "eta_max": ETA_MAX,
                           "cfl_at_eta_max": cfl_at_eta(amplitude, exponent, ETA_MAX),
                           "n_points": len(fit_samples)}
            except ValueError as exc:                 # pragma: no cover
                scaling = {"status": "FIT_FAILED", "reason": str(exc)}
        verdict = power_verdict(scaling, ADMISSIBLE_CFL_MAX)
        control = positive_control(marked)
        production = [s for s in samples if s["production_cfl"]]
        payload["degrees"][f"p{degree}"] = {
            "fine_mesh": fine_n,
            "envelope": envelope,
            "eta_samples": samples,
            "scaling": scaling,
            "verdict": verdict,
            "positive_control": control,
            "refused_probes": [{"cfl": r["cfl"], "reason": r.get("reason")}
                               for r in own if r.get("rejected")],
            "production_eta": production[0]["eta"] if production else None,
            "production_leg": production[0]["leg"] if production else None,
            "second_refinement": second_refinement(fine_marked),
            "drift_leg": order_drift_leg(marked, degree, beta=beta),
            "runs": sorted(marked, key=lambda r: (int(r["mesh"]),
                                                  float(r.get("cfl") or 0.0))),
        }
    payload["verdicts"] = {k: v["verdict"] for k, v in payload["degrees"].items()}
    return payload


def render_markdown(payload: dict) -> str:
    lines = ["# CKS1998 §5.3 — time-attribution instrument power calibration", ""]
    lines.append(f"production CFL = {payload['production_cfl']}, beta = "
                 f"{payload['beta']}, eta_max = {payload['eta_max']}, "
                 f"admissible CFL <= {payload['admissible_cfl_max']}")
    lines.append("")
    for degree, bundle in payload["degrees"].items():
        lines += [f"## {degree} (fine mesh {bundle['fine_mesh']})", "",
                  "| CFL | e(CFL) | e(CFL/beta) | eta | leg |",
                  "|---:|---:|---:|---:|---|"]
        for s in bundle["eta_samples"]:
            lines.append(f"| {s['cfl']:g} | {s['error']:.6e} | "
                         f"{s['error_refined']:.6e} | {s['eta']:.3e} | {s['leg']} |")
        lines += ["", f"* scaling: {bundle['scaling']}",
                  f"* refused probes: {bundle['refused_probes']}",
                  f"* positive control: {bundle['positive_control'].get('fired')} "
                  f"(first firing effective CFL "
                  f"{bundle['positive_control'].get('first_firing_effective_cfl')})",
                  f"* second refinement: {bundle['second_refinement']}",
                  f"* drift leg: {bundle['drift_leg']}",
                  f"* **verdict: {bundle['verdict']}**", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degrees", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--shard-out", type=Path, default=None)
    ap.add_argument("--assemble", type=Path, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=POWER_OUT)
    ap.add_argument("--local-smoke", action="store_true",
                    help="cheap single-degree sanity sweep (seconds)")
    args = ap.parse_args(argv)

    if args.assemble is not None:
        shards = args.assemble or sorted(OUT_DIR.glob("time_power_p*.json"))
        records = []
        for path in shards:
            records.extend(json.loads(Path(path).read_text())["runs"])
        payload = assemble(records)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))
        md = render_markdown(payload)
        args.out.with_suffix(".md").write_text(md)
        print(md)
        return 0

    if args.local_smoke:
        records = sweep(1, meshes=(4, 6), log=lambda *a: print(*a, file=sys.stderr))
        payload = assemble(records)
        print(json.dumps({"verdicts": payload["verdicts"],
                          "scaling": payload["degrees"]["p1"]["scaling"],
                          "control": payload["degrees"]["p1"]["positive_control"]},
                         indent=1)[:2000])
        return 0

    all_records = []
    for degree in args.degrees:
        print(f"[degree {degree}]", file=sys.stderr)
        recs = sweep(degree, log=lambda *a: print(*a, file=sys.stderr))
        if args.shard_out is not None:
            out = args.shard_out
            if len(args.degrees) > 1:
                out = out.with_name(f"{out.stem}_p{degree}{out.suffix}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"degree": degree, "runs": recs}, indent=1))
            print(f"shard -> {out}", file=sys.stderr)
        all_records.extend(recs)
    if args.shard_out is None:
        payload = assemble(all_records)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))
        print(render_markdown(payload))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())