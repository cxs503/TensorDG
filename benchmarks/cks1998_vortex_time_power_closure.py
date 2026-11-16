"""Closure of the CKS1998 §5.3 time-attribution instrument power calibration.

Why this module exists
----------------------
``benchmarks/cks1998_vortex_time_power.py`` (mission m22636) calibrated the
power of the certified study's ``eta`` leg and returned
``GATE_VACUOUS_IN_ENVELOPE``: the log-log law fitted to ``eta(CFL)`` puts the
5 % crossing at ``CFL ~ 7.4``, far outside the solver's hard admissible window
``0 < CFL <= 0.5``.  Its own §5 listed three open items, none of which was
closed:

1. **the fit used two points** (``CFL = 0.2`` and ``0.4``), so the exponent
   ``s = 2.82`` had no uncertainty -- the ladder carried ``CFL = 0.05, 0.1`` but
   their ``CFL/beta`` partners (``0.0125, 0.025``) were never run, and the
   sample builder can only pair a base CFL with an existing partner;
2. **the second-level refinement** ``dt -> dt/beta**2`` (itself the open
   limitation 4 of the certified study) returned ``INSUFFICIENT_LEVELS``
   because ``CFL/beta**2 = 0.0125`` was missing;
3. **the mesh was the local ``n = 10``**, not the certified ``TIME_PAIR``
   fine mesh, so "the same vacuity holds on the production configuration" was
   an extrapolation rather than a measurement.

This module closes all three *without touching a single pre-registered
constant* and, on top of the extra samples, adds an estimator the earlier work
did not have:

**temporal-order cross-check.**  With three time levels the time-error share
can be measured twice, and the *ratio* of the two shares is itself a
measurement::

    eta    = |e(c) - e(c/beta)|   / e(c)              share, one refinement
    omega  = |e(c/beta) - e(c/beta**2)| / e(c/beta)   share, second refinement
    q_obs  = -ln(omega / eta) / ln(beta)              implied temporal order

because for a temporally dominated error ``e(c) ~ tau c**q`` the shares satisfy
``omega/eta = beta**-q``.  ``q_obs`` is *independent* of the CFL-ladder power
law (it is a single-CFL, three-level Richardson-type estimate), so comparing it
with the fitted exponent ``s`` tests the instrument against itself:

* ``CONSISTENT``      -- ``|s - q_obs| / q_obs <= ORDER_TOL``: two independent
  routes give the same temporal order, so the power law that produces the
  vacuity verdict is supported;
* ``DISAGREE``        -- the difference is declared, the verdict stays as
  measured but is flagged ``ORDER_INCONSISTENT`` (no claim of support);
* ``UNRESOLVED``      -- ``omega`` or ``eta`` is zero / non-finite: nothing is
  claimed about the order.

The bias of ``q_obs`` is ``O(eta)`` (it is exact only for ``S = 0``, with ``S``
the CFL-independent spatial error), i.e. ``~1e-6`` here -- four orders below any
tolerance of interest, and stated explicitly in the case document.

Verdicts
--------
``power_jackknife`` fits the law to every sample and to every leave-one-out
subset.  The closure verdict is therefore stated with its uncertainty rather
than as a point value:

* ``GATE_VACUOUS_CERTIFIED``  -- every leave-one-out fit crosses above the
  admissible window: the vacuity of the ``eta`` leg is robust to dropping any
  single CFL;
* ``GATE_INFORMATIVE_CERTIFIED`` -- every leave-one-out fit crosses *inside* the
  window: the instrument can fire, and m22636's verdict would be overturned;
* ``CLOSURE_UNDERPOWERED``    -- fewer than three samples, or the leave-one-out
  crossings straddle the window edge: neither claim is certified.

Pure functions are torch-free and unit tested
(``tests/test_cks1998_vortex_time_power_closure.py``); the numeric payload is
produced by the solver sweep and published with its raw records.

Usage
-----
::

    python -m benchmarks.cks1998_vortex_time_power_closure \
        --degree 1 --meshes 10 32 \
        --shard-out validation/cks1998_vortex_2d/time_power_closure_p1.json
    python -m benchmarks.cks1998_vortex_time_power_closure --assemble
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from benchmarks.cks1998_vortex_order_certification import (
    CFL0,
    OUT_DIR,
    TIME_PAIR,
    TIME_REFINEMENT,
)
from benchmarks.cks1998_vortex_time_power import (
    ADMISSIBLE_CFL_MAX,
    BETA,
    BETA2,
    DT_KNOTS,
    ETA_MAX,
    FIT_CFLS,
    REJECTED_PROBES,
    SCALING_R2_MIN,
    _record,
    _record_dt,
    cfl_at_eta,
    classify_leg,
    eta_share,
    etas_from_records,
    loglog_fit,
    order_drift_leg,
    positive_control,
    power_verdict,
    second_refinement,
    stability_envelope,
)

CLOSURE_OUT = OUT_DIR / "time_power_closure.json"

#: CFL ladder that is *partner complete* under ``beta = 4``: every base CFL used
#: by the fit has its ``CFL/beta`` partner in the set, and the production CFL
#: also has its ``CFL/beta**2`` partner.  This is exactly what m22636 lacked.
EXTENDED_CFLS = (CFL0 / BETA2, CFL0 / BETA / 2, CFL0 / BETA, CFL0 / 2,
                 CFL0, CFL0 * 2, ADMISSIBLE_CFL_MAX)
#: CFLs entering the power fit (the window edge 0.5 is measured, not fitted: it
#: is not a regular point of the family, and it has no partner in the ladder).
EXTENDED_FIT_CFLS = (0.05, 0.1, 0.2, 0.4)
#: Minimum number of ``eta`` samples for a jackknife.
MIN_SAMPLES = 3
#: Admissible relative gap between the fitted exponent ``s`` and the three-level
#: temporal order ``q_obs``.  Pre-registered here, before any number is looked
#: at; it is *not* a physical tolerance but an instrument-consistency band.
ORDER_TOL = 0.10
#: Measurable replacement scale for the ``eta`` leg, proposed (not enacted) in
#: the case document: it is the order of magnitude this instrument actually
#: resolves on the case, i.e. a bound a *measurable* criterion can be stated
#: against.  Nothing in the certified ladder is changed by it.
PROPOSED_ETA_BOUND = 1.0e-4
#: ``omega/eta`` below this is numerically indistinguishable from "the refined
#: level is already time converged to round-off": no order is claimed.  At the
#: measured share scale (``eta ~ 1e-6``) this band sits *below* double-precision
#: resolution, so it is a defensive guard rather than a working branch; the
#: reachable degradation path is ``OMEGA_ZERO`` (see the case document §5).
RATIO_FLOOR = 1.0e-12


# ---------------------------------------------------------------------------
# Pure estimators
# ---------------------------------------------------------------------------
def level_triples(records, beta: float = BETA) -> list[dict]:
    """Per-CFL three-level measurements ``(eta, omega, q_obs)``.

    A triple needs ``c``, ``c/beta`` and ``c/beta**2`` on the *same* mesh;
    refused, diverged and non-finite records are skipped, so no arithmetic ever
    runs on a lost configuration.  ``q_obs`` is ``None`` (with a reason) when
    ``eta`` or ``omega`` is zero -- the log of a zero ratio is not a number, and
    inventing one would manufacture an order.
    """
    m = measurable_by_cfl(records)
    out = []
    for cfl in sorted(m):
        p1, p2 = round(cfl / beta, 12), round(cfl / beta ** 2, 12)
        if p1 not in m or p2 not in m:
            continue
        e0, e1, e2 = (float(m[cfl]["error"]), float(m[p1]["error"]),
                      float(m[p2]["error"]))
        eta = eta_share(e0, e1)
        omega = abs(e1 - e2) / e1
        item = {"cfl": cfl, "errors": {"base": e0, "refined": e1,
                                       "refined2": e2},
                "eta": eta, "omega": omega,
                "omega_over_eta": (omega / eta) if eta > 0.0 else None,
                "q_obs": None, "reason": None}
        if eta <= 0.0:
            item["reason"] = "ETA_ZERO"
        elif omega <= 0.0:
            item["reason"] = "OMEGA_ZERO"
        elif omega / eta < RATIO_FLOOR:      # pragma: no cover - unreachable
            item["reason"] = "RATIO_BELOW_FLOOR"   # in double precision here
        else:
            item["q_obs"] = -math.log(omega / eta) / math.log(beta)
        item["production_cfl"] = abs(cfl - CFL0) < 1e-12
        out.append(item)
    return out


def measurable_by_cfl(records) -> dict:
    """``{cfl: record}`` for finite, non-refused records that carry a CFL."""
    m = {}
    for r in records:
        if r.get("unstable") or r.get("rejected"):
            continue
        cfl = r.get("cfl")
        if cfl is None:
            continue
        try:
            err = float(r["error"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(err):
            continue
        m[round(float(cfl), 12)] = r
    return m


def fit_law(samples, eta_max: float = ETA_MAX) -> dict:
    """Log-log fit ``eta = A CFL**s`` on finite positive samples."""
    pts = [s for s in samples
           if s.get("eta") is not None and math.isfinite(float(s["eta"]))
           and float(s["eta"]) > 0.0 and float(s["cfl"]) > 0.0]
    if len(pts) < 2:
        return {"status": "INSUFFICIENT_SAMPLES", "n_points": len(pts)}
    amp, exp, r2 = loglog_fit([s["cfl"] for s in pts], [s["eta"] for s in pts])
    return {"status": "OK", "amplitude": amp, "exponent": exp, "r2": r2,
            "eta_max": eta_max,
            "cfl_at_eta_max": (cfl_at_eta(amp, exp, eta_max)
                               if exp > 0.0 else None),
            "n_points": len(pts),
            "cfls": [s["cfl"] for s in pts]}


def power_jackknife(samples, eta_max: float = ETA_MAX,
                    admissible_max: float = ADMISSIBLE_CFL_MAX,
                    min_samples: int = MIN_SAMPLES) -> dict:
    """Power law with leave-one-out uncertainty and a certified verdict.

    Returns the full fit, every leave-one-out fit, the range of the fitted
    exponent and of the extrapolated crossing, and a verdict that only claims
    vacuity/informativeness when *every* leave-one-out fit agrees.
    """
    pts = [s for s in samples
           if s.get("eta") is not None and math.isfinite(float(s["eta"]))
           and float(s["eta"]) > 0.0]
    full = fit_law(pts, eta_max)
    out = {"full": full, "n_points": len(pts),
           "min_samples": min_samples, "jackknife": [],
           "exponent_range": None, "crossing_range": None,
           "verdict": "CLOSURE_UNDERPOWERED"}
    if full.get("status") != "OK" or len(pts) < min_samples:
        out["reason"] = ("INSUFFICIENT_SAMPLES" if len(pts) < min_samples
                         else "FIT_FAILED")
        return out
    loo_crossings, loo_exponents = [], []
    for i, drop in enumerate(pts):
        rest = [p for j, p in enumerate(pts) if j != i]
        fit = fit_law(rest, eta_max)
        if fit.get("status") != "OK" or fit.get("cfl_at_eta_max") is None:
            continue
        loo_crossings.append(fit["cfl_at_eta_max"])
        loo_exponents.append(fit["exponent"])
        out["jackknife"].append({"dropped_cfl": drop["cfl"],
                                 "exponent": fit["exponent"],
                                 "cfl_at_eta_max": fit["cfl_at_eta_max"],
                                 "r2": fit["r2"]})
    if not loo_crossings:                          # pragma: no cover - guard
        out["reason"] = "NO_LEAVE_ONE_OUT_FITS"
        return out
    out["exponent_range"] = [min(loo_exponents), max(loo_exponents)]
    out["crossing_range"] = [min(loo_crossings), max(loo_crossings)]
    point = power_verdict(full, admissible_max)
    all_up = all(c > admissible_max for c in loo_crossings)
    all_in = all(c <= admissible_max for c in loo_crossings)
    out["verdict"] = ("GATE_VACUOUS_CERTIFIED" if all_up
                      else "GATE_INFORMATIVE_CERTIFIED" if all_in
                      else "CLOSURE_UNDERPOWERED")
    out["point_verdict"] = point
    out["all_leave_one_out_agree"] = bool(all_up or all_in)
    out["admissible_cfl_max"] = admissible_max
    return out


def window_sup(samples, admissible_max: float = ADMISSIBLE_CFL_MAX) -> dict:
    """Supremum of the instrument over the *admissible* window.

    A vacuous threshold says nothing; the supremum is still a statement.  If the
    largest share measured on any admissible configuration is ``eta_sup``, then
    "time error contributes at most ``eta_sup`` of the error on every admissible
    run of this case" is certified by the ladder itself -- no extrapolation, no
    pre-registered constant touched.  ``bound_density`` reports the same share
    against a candidate replacement bound of ``1e-4`` (the measurable scale
    proposed in the case document), purely as information for the ledger owner.
    """
    inside = [s for s in samples
              if s.get("eta") is not None
              and math.isfinite(float(s["eta"]))
              and float(s["cfl"]) <= admissible_max]
    if not inside:
        return {"status": "NO_ADMISSIBLE_SAMPLES", "admissible_cfl_max":
                admissible_max}
    top = max(inside, key=lambda s: float(s["eta"]))
    sup = float(top["eta"])
    return {"status": "OK", "admissible_cfl_max": admissible_max,
            "eta_sup": sup, "at_cfl": float(top["cfl"]),
            "n_admissible": len(inside),
            "sup_over_eta_max": (sup / ETA_MAX) if ETA_MAX > 0 else None,
            "below_proposed_1e4": bool(sup <= PROPOSED_ETA_BOUND)}


def order_agreement(fitted_exponent, temporal_order,
                    tol: float = ORDER_TOL) -> dict:
    """Compare the fitted CFL exponent with the three-level temporal order."""
    out = {"tolerance": tol, "fitted_exponent": fitted_exponent,
           "temporal_order": temporal_order, "relative_gap": None,
           "status": "UNRESOLVED"}
    if fitted_exponent is None or temporal_order is None:
        out["reason"] = "MISSING_ESTIMATE"
        return out
    try:
        s, q = float(fitted_exponent), float(temporal_order)
    except (TypeError, ValueError):                # pragma: no cover - guard
        out["reason"] = "NON_NUMERIC"
        return out
    if not (math.isfinite(s) and math.isfinite(q)) or q == 0.0:
        out["reason"] = "NON_FINITE"
        return out
    gap = abs(s - q) / abs(q)
    out["relative_gap"] = gap
    out["status"] = "CONSISTENT" if gap <= tol else "DISAGREE"
    return out


def closure_verdict(power: dict, agreements: list[dict]) -> str:
    """Fold the power verdict and the order cross-checks into one string."""
    base = power.get("verdict", "CLOSURE_UNDERPOWERED")
    if base == "CLOSURE_UNDERPOWERED":
        return "CLOSURE_UNDERPOWERED"
    statuses = [a.get("status") for a in agreements]
    if "DISAGREE" in statuses:
        return f"{base}_ORDER_INCONSISTENT"
    return base


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep(degree: int, meshes=None, cfls=None, dt_knots=None,
          drift_pair: bool = True, log=print) -> list[dict]:
    """Partner-complete CFL ladder on each mesh, plus the positive control.

    ``meshes`` defaults to the local ``n = 10`` of m22636 plus the certified
    ``TIME_PAIR`` fine mesh, so mesh independence is measured, not assumed.
    """
    if meshes is None:
        meshes = (10, TIME_PAIR[degree][1])
    cfls = tuple(EXTENDED_CFLS if cfls is None else cfls)
    knots = tuple(DT_KNOTS if dt_knots is None else dt_knots)
    records = []
    for n in meshes:
        for cfl in cfls:
            records.append(_record(degree, n, "ladder", cfl=cfl, log=log,
                                   mesh_role=("production" if n == TIME_PAIR[degree][1]
                                              else "local")))
        for cfl in (() if not drift_pair and not knots else REJECTED_PROBES):
            records.append(_record(degree, n, "refused_probe", cfl=cfl, log=log,
                                   mesh_role=("production" if n == TIME_PAIR[degree][1]
                                              else "local")))
        dt_prod = next((r["dt"] for r in records
                        if r.get("kind") == "ladder" and int(r["mesh"]) == n
                        and r.get("cfl") is not None
                        and abs(float(r["cfl"]) - CFL0) < 1e-12 and r.get("dt")),
                       None)
        if dt_prod and knots:
            for k in knots:
                records.append(_record_dt(degree, n, k, dt_prod, log=log,
                                          mesh_role=("production"
                                                     if n == TIME_PAIR[degree][1]
                                                     else "local")))
    # coarse mesh of the certified pair, for the order-drift leg only
    coarse_n, fine_n = TIME_PAIR[degree]
    if drift_pair and coarse_n not in meshes:
        for cfl in (CFL0, CFL0 / BETA):
            records.append(_record(degree, coarse_n, "ladder", cfl=cfl, log=log,
                                   mesh_role="pair_coarse"))
    return records


def assemble(records: list[dict], beta: float = BETA) -> dict:
    """Per-degree, per-mesh payload: samples, jackknife, three levels, verdict."""
    payload = {"case": "CKS1998-5.3-isentropic-vortex",
               "anchor": "REPRO-DG-CKS1998",
               "production_cfl": CFL0, "beta": beta, "beta2": beta * beta,
               "eta_max": ETA_MAX, "admissible_cfl_max": ADMISSIBLE_CFL_MAX,
               "order_tol": ORDER_TOL, "extended_cfls": list(EXTENDED_CFLS),
               "fit_cfls": list(EXTENDED_FIT_CFLS),
               "degrees": {}, "verdicts": {}}
    degrees = sorted({int(r["degree"]) for r in records if r.get("degree") is not None})
    for degree in degrees:
        own = [r for r in records if int(r["degree"]) == degree]
        ladder = [r for r in own if r.get("kind") == "ladder"]
        meshes = sorted({int(r["mesh"]) for r in ladder})
        bundle = {"meshes": meshes, "per_mesh": {}}
        agreements = []
        for n in meshes:
            fine = [r for r in ladder if int(r["mesh"]) == n]
            samples = etas_from_records(fine, beta=beta)
            fit_samples = [s for s in samples
                           if any(abs(s["cfl"] - c) < 1e-12
                                  for c in EXTENDED_FIT_CFLS)]
            power = power_jackknife(fit_samples)
            triples = level_triples(fine, beta=beta)
            prod_triple = next((t for t in triples if t["production_cfl"]), None)
            agreement = order_agreement(
                (power.get("full") or {}).get("exponent"),
                (prod_triple or {}).get("q_obs"))
            agreements.append(agreement)
            bundle["per_mesh"][str(n)] = {
                "mesh_role": next((r.get("mesh_role") for r in fine
                                   if r.get("mesh_role")), None),
                "envelope": stability_envelope(fine),
                "eta_samples": samples,
                "power": power,
                "level_triples": triples,
                "production_triple": prod_triple,
                "order_agreement": agreement,
                "window_sup": window_sup(samples),
                "second_refinement": second_refinement(fine, beta=beta),
                "drift_leg": order_drift_leg(ladder, degree, beta=beta),
                "positive_control": positive_control(fine),
                "refused_probes": [{"cfl": r.get("cfl"),
                                    "reason": r.get("reason")}
                                   for r in own if r.get("rejected")],
            }
        # the certified configuration is the production mesh; the verdict is
        # only certified if the local mesh agrees with it
        prod_key = str(TIME_PAIR[degree][1])
        primary = bundle["per_mesh"].get(prod_key) or \
            bundle["per_mesh"][str(meshes[-1])]
        verdicts = {m: v["power"]["verdict"] for m, v in bundle["per_mesh"].items()}
        bundle["primary_mesh"] = int(prod_key if prod_key in bundle["per_mesh"]
                                    else meshes[-1])
        bundle["mesh_verdicts"] = verdicts
        bundle["mesh_independent"] = len(set(verdicts.values())) == 1
        bundle["closure_verdict"] = closure_verdict(primary["power"], agreements)
        payload["degrees"][f"p{degree}"] = bundle
        payload["verdicts"][f"p{degree}"] = bundle["closure_verdict"]
    return payload


def render_markdown(payload: dict) -> str:
    lines = ["# CKS1998 §5.3 — power-calibration closure", "",
             f"production CFL {payload['production_cfl']}, beta "
             f"{payload['beta']}, eta_max {payload['eta_max']}, admissible "
             f"CFL <= {payload['admissible_cfl_max']}, order tol "
             f"{payload['order_tol']}", ""]
    for degree, bundle in payload["degrees"].items():
        lines += [f"## {degree} (primary mesh {bundle['primary_mesh']})", "",
                  f"* mesh verdicts: {bundle['mesh_verdicts']}",
                  f"* mesh independent: {bundle['mesh_independent']}",
                  f"* closure verdict: **{bundle['closure_verdict']}**", ""]
        for mesh, m in bundle["per_mesh"].items():
            full = m["power"].get("full") or {}
            lines += [f"### mesh {mesh}", "",
                      "| CFL | e(CFL) | e(CFL/beta) | eta | leg |",
                      "|---:|---:|---:|---:|---|"]
            for s in m["eta_samples"]:
                lines.append(f"| {s['cfl']:g} | {s['error']:.6e} | "
                             f"{s['error_refined']:.6e} | {s['eta']:.3e} | "
                             f"{s['leg']} |")
            lines += ["",
                      f"* fit: s = {full.get('exponent')}, r2 = "
                      f"{full.get('r2')}, n = {full.get('n_points')}, "
                      f"crossing = {full.get('cfl_at_eta_max')}",
                      f"* jackknife exponent range "
                      f"{m['power'].get('exponent_range')}, crossing range "
                      f"{m['power'].get('crossing_range')}",
                      f"* three-level q_obs: "
                      f"{(m.get('production_triple') or {}).get('q_obs')}, "
                      f"agreement {m['order_agreement'].get('status')} "
                      f"(gap {m['order_agreement'].get('relative_gap')})",
                      f"* window sup: {m['window_sup']}",
                      f"* second refinement: {m['second_refinement']}",
                      f"* drift leg: {m['drift_leg']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--meshes", type=int, nargs="*", default=None)
    ap.add_argument("--cfls", type=float, nargs="*", default=None,
                    help="CFL subset (default: the partner-complete ladder)")
    ap.add_argument("--dt-knots", type=int, nargs="*", default=None,
                    help="positive-control dt multiples (default: all)")
    ap.add_argument("--no-drift-pair", action="store_true",
                    help="skip the coarse mesh of the certified pair")
    ap.add_argument("--shard-out", type=Path, default=None)
    ap.add_argument("--assemble", nargs="*", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=CLOSURE_OUT)
    args = ap.parse_args(argv)

    log = lambda *a: print(*a, file=sys.stderr)      # noqa: E731
    if args.assemble is not None:
        shards = args.assemble or sorted(OUT_DIR.glob("time_power_closure_p*.json"))
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

    started = time.time()
    recs = sweep(args.degree,
                 meshes=tuple(args.meshes) if args.meshes else None,
                 cfls=tuple(args.cfls) if args.cfls else None,
                 dt_knots=tuple(args.dt_knots) if args.dt_knots is not None else None,
                 drift_pair=not args.no_drift_pair, log=log)
    log(f"[degree {args.degree}] {len(recs)} runs in {time.time()-started:.1f}s")
    if args.shard_out is not None:
        args.shard_out.parent.mkdir(parents=True, exist_ok=True)
        args.shard_out.write_text(json.dumps({"degree": args.degree,
                                              "runs": recs}, indent=1))
        log(f"shard -> {args.shard_out}")
        return 0
    payload = assemble(recs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(render_markdown(payload))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())