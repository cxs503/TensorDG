"""Set-level *feature* gates: "the profile passed, did the feature?".

Why this module exists
----------------------
``certify_functional`` / ``judge_functional`` judge **one** functional at a time
against one certified reference.  That is necessary but not sufficient as an
acceptance rule, because a validation claim of the form "the wave train / vortex
core / wall extremum is right to X %" is a claim about a *set* of functionals:

* an **integral** functional (``rms``, ``l1``, a windowed mass or momentum
  budget) is insensitive to where the energy sits,
* an **extremum** functional (peak value, peak location) is insensitive to how
  much energy there is elsewhere.

A scheme can pass every integral functional and still destroy the extremum (the
Shu--Osher wave train at L1-passing DG legs loses 18--68 % of its amplitude --
see ``docs/XVAL-DG-ROUND2-FEATURE-CERTIFICATION.md``).  A gate that accepts such
a scheme is *vacuous* for the feature it is supposed to protect.  This module is
the machine-checkable version of the recommendation that closed that audit:

    a case-level gate is passed only if, for **every required functional kind**,
    at least ``required[kind]`` functionals pass against a usable certified
    reference, each within its band-aware tolerance.

Two anti-vacuity rules are built in, because "no functional of this kind was
evaluated" must never read as a pass:

* a kind with fewer judgeable functionals than required is ``GATE_WITHHELD``,
  never ``GATE_PASS`` (:func:`judge_feature_set`);
* :func:`gate_teeth_check` re-runs the gate on the reference itself (must pass)
  and on a deliberately perturbed candidate (must fail), so a mis-wired gate --
  a battery that cannot fail -- is detected by the gate's own test, not by a
  reviewer reading the verdict.

Cell-average extremum deconvolution
-----------------------------------
Extremum-type functionals measured on **cell averages** carry a sampling bias
that is a property of the representation, not of the solver: for a smooth field
the cell mean over a cell of width ``h`` is

    ``<f>_i = f(x_i) + h^2/24 f''(x_i) + O(h^4)``

so ``max_i <f>_i`` (and ``argmax``) is only O(h^2) accurate *at best*, and the
location is quantised to cell centres, which makes the location ladder a
staircase (measured O(h) jitter, e.g. ``peak_x`` on the Shu--Osher wave train).
:func:`cell_average_extremum` removes both effects to leading order without any
knowledge of the continuum solution:

1. fit the parabola through ``(<f>_{j-1}, <f>_j, <f>_{j+1})`` at the discrete
   maximum ``j`` -> vertex value and sub-cell vertex position (kills the
   quantisation: two independent O(h^2) errors become one O(h^4) fit);
2. read the curvature ``f''`` off the same parabola and subtract the known
   ``h^2/24 f''`` cell-average bias.

On analytic cell averages this takes the recovered peak from O(h^2) to O(h^4)
(``tests/test_feature_gate.py::test_deconvolution_restores_fourth_order``).  The
estimator is *representation-aware, not case-aware*: it never sees the exact
solution, so it cannot smuggle the answer into the measurement.  When the
discrete extremum sits on the domain edge, the spacing is non-uniform, or the
parabola opens the wrong way, the estimator refuses and falls back to the raw
cell value with an explicit ``status`` -- a degraded estimate is reported, not
silently repaired.

Nothing here is case-specific: every function takes arrays (or a mapping of
named functional values) and the gate is configured by the caller.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np

from .certified_reference import FAIL, PASS, PASS_MARGINAL, WITHHELD, judge_functional

__all__ = [
    "EST_RAW",
    "EST_PARABOLA",
    "EST_DECONVOLVED",
    "GATE_PASS",
    "GATE_PASS_MARGINAL",
    "GATE_FAIL",
    "GATE_WITHHELD",
    "DEFAULT_REQUIRED",
    "cell_average_extremum",
    "kind_of",
    "judge_feature_set",
    "gate_teeth_check",
]

EST_RAW = "raw_cell_average"
EST_PARABOLA = "parabola_vertex"
EST_DECONVOLVED = "deconvolved"

GATE_PASS = "GATE_PASS"
GATE_PASS_MARGINAL = "GATE_PASS_MARGINAL"
GATE_FAIL = "GATE_FAIL"
GATE_WITHHELD = "GATE_WITHHELD"

#: default set-level requirement: one integral *and* one extremum functional
DEFAULT_REQUIRED = {"integral": 1, "extremum": 1}

#: decisions that count as a pass
_PASSING = (PASS, PASS_MARGINAL)


def cell_average_extremum(values, *, x=None, mode: str = "max",
                          correct_bias: bool = True,
                          uniform_tol: float = 1e-9) -> dict:
    """Locate and deconvolve an extremum of a cell-average profile.

    Parameters
    ----------
    values : sequence of float
        Cell averages of the field (one value per cell, any spacing).
    x : sequence of float, optional
        Cell centres.  Defaults to ``i + 0.5`` (unit spacing), which is all the
        estimator needs when only the *value* is wanted.
    mode : {"max", "min"}
    correct_bias : bool
        Subtract the ``h^2/24 f''`` cell-average bias from the parabola vertex.
        ``False`` returns the raw vertex (the intermediate estimator).
    uniform_tol : float
        Relative tolerance on spacing uniformity; non-uniform grids fall back to
        the raw estimate rather than being silently fit.

    Returns
    -------
    dict
        ``value`` / ``position`` are the best estimate; ``raw_value`` /
        ``raw_position`` the cell-average argmax; ``estimator`` says which one
        ``value`` came from; ``status`` is ``"ok"`` for the deconvolved path or
        ``"boundary"`` / ``"nonuniform"`` / ``"degenerate"`` (with
        ``estimator == EST_RAW``) when the fit was refused; ``curvature`` is the
        fitted ``f''`` (``None`` when refused).
    """
    f = np.asarray(values, dtype=float)
    if f.ndim != 1 or f.size < 3:
        raise ValueError(f"need a 1-D profile of >= 3 cells, got shape {f.shape}")
    if not np.all(np.isfinite(f)):
        raise ValueError("profile contains non-finite values")
    if mode not in ("max", "min"):
        raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
    xv = (np.arange(f.size, dtype=float) + 0.5 if x is None
          else np.asarray(x, dtype=float))
    if xv.shape != f.shape:
        raise ValueError(f"x {xv.shape} and values {f.shape} differ")
    if not np.all(np.isfinite(xv)):
        raise ValueError("x contains non-finite values")

    pick = np.argmax if mode == "max" else np.argmin
    j = int(pick(f))
    raw = {"value": float(f[j]), "position": float(xv[j])}
    base = {
        "mode": mode, "ncells": int(f.size), "index": j,
        "raw_value": raw["value"], "raw_position": raw["position"],
        "value": raw["value"], "position": raw["position"],
        "curvature": None, "bias_correction": 0.0,
        "estimator": EST_RAW, "status": "ok",
    }
    if j == 0 or j == f.size - 1:
        base["status"] = "boundary"
        return base

    dx_all = np.diff(xv)
    dx = float(np.mean(dx_all))
    if not math.isfinite(dx) or dx <= 0.0:
        base["status"] = "nonuniform"
        return base
    if float(np.max(np.abs(dx_all - dx))) > uniform_tol * abs(dx):
        base["status"] = "nonuniform"
        return base

    y0, y1, y2 = float(f[j - 1]), float(f[j]), float(f[j + 1])
    curvature = (y0 - 2.0 * y1 + y2) / (dx * dx)   # f'' at x_j
    # parabola through the three cell means, in units of dx
    a = 0.5 * (y0 - 2.0 * y1 + y2)                 # coefficient of (xi/dx)^2
    b = 0.5 * (y2 - y0)                            # coefficient of (xi/dx)
    if a == 0.0 or (mode == "max" and a > 0.0) or (mode == "min" and a < 0.0):
        base["status"] = "degenerate"   # flat or wrong-sign parabola
        base["curvature"] = curvature
        return base

    offset = -b / (2.0 * a)                        # in units of dx
    vertex = y1 - b * b / (4.0 * a)
    bias = (dx * dx / 24.0) * curvature if correct_bias else 0.0
    base.update({
        "value": float(vertex - bias),
        "position": float(xv[j] + offset * dx),
        "curvature": float(curvature),
        "bias_correction": float(bias),
        "estimator": EST_DECONVOLVED if correct_bias else EST_PARABOLA,
        "vertex_value": float(vertex),
        "subcell_offset": float(offset * dx),
    })
    return base


def kind_of(name: str, table: Mapping[str, str] | None = None) -> str:
    """Functional kind used by the gate; ``table`` maps name -> kind.

    Unknown names default to ``"other"`` so a caller cannot accidentally satisfy
    a requirement by adding a functional of an undeclared kind.
    """
    if table is None:
        return "other"
    return table.get(name, "other")


def judge_feature_set(candidate: Mapping[str, float], *,
                      references: Mapping[str, Mapping],
                      tolerance_percent: float = 3.0,
                      kinds: Mapping[str, str] | None = None,
                      required: Mapping[str, int] | None = DEFAULT_REQUIRED,
                      allow_marginal: bool = False) -> dict:
    """Judge one candidate against a *set* of certified functional references.

    ``references`` maps functional name -> certification dict, i.e. the output of
    :func:`tensordg.certified_reference.certify_functional` (the ``kind`` of each
    functional comes from ``kinds``, so the reference dict itself stays exactly
    what the engine produced).  ``candidate`` maps functional name -> value.

    The gate is a *strict* set rule: for every ``kind`` in ``required`` at least
    ``required[kind]`` functionals must reach ``PASS`` (``PASS_MARGINAL`` only
    counts when ``allow_marginal`` is set).  Missing / withheld / zero-reference
    functionals are reported, and a kind that cannot reach its quota because
    functionals are missing is ``GATE_WITHHELD`` -- never a pass.
    """
    req = dict(DEFAULT_REQUIRED if required is None else required)
    if not req:
        raise ValueError("required is empty: a gate with no requirement is vacuous")
    for kind, need in req.items():
        if int(need) < 1:
            raise ValueError(f"required[{kind!r}] = {need}: quotas must be >= 1")

    per: dict[str, dict] = {}
    for name, cert in references.items():
        kind = kind_of(name, kinds)
        entry = {"kind": kind, "decision": WITHHELD, "reason": "not in candidate",
                 "error_percent": None, "reference_verdict": None}
        if name in candidate:
            judgement = judge_functional(float(candidate[name]), cert,
                                         tolerance_percent=tolerance_percent)
            entry.update(judgement)
            entry["kind"] = kind
        else:
            entry["reference_verdict"] = cert.get("verdict")
        per[name] = entry
    missing = sorted(set(candidate) - set(references))

    kind_status: dict[str, dict] = {}
    for kind, need in req.items():
        members = [n for n, e in per.items() if e["kind"] == kind]
        strict = [n for n in members if per[n]["decision"] == PASS]
        marginal = [n for n in members if per[n]["decision"] == PASS_MARGINAL]
        withheld = [n for n in members if per[n]["decision"] == WITHHELD]
        failed = [n for n in members if per[n]["decision"] == FAIL]
        credited = list(strict) + (list(marginal) if allow_marginal else [])
        entry = {
            "required": int(need), "n_available": len(members),
            "n_pass": len(strict), "n_pass_marginal": len(marginal),
            "n_fail": len(failed), "n_withheld": len(withheld),
            "passing": strict, "failing": failed, "withheld": withheld,
        }
        if len(credited) >= need:
            entry["status"] = GATE_PASS
        elif len(members) - len(withheld) < int(need):
            # the quota is unreachable on the evidence that exists
            entry["status"] = GATE_WITHHELD
            entry["reason"] = ("fewer judgeable functionals of this kind than "
                               "the quota: missing evidence is not a pass")
        elif allow_marginal and len(credited) + len(marginal) >= need:
            entry["status"] = GATE_PASS_MARGINAL
        else:
            entry["status"] = GATE_FAIL
        kind_status[kind] = entry

    statuses = [s["status"] for s in kind_status.values()]
    if GATE_FAIL in statuses:
        verdict = GATE_FAIL
    elif GATE_WITHHELD in statuses:
        verdict = GATE_WITHHELD
    elif GATE_PASS_MARGINAL in statuses:
        verdict = GATE_PASS_MARGINAL
    else:
        verdict = GATE_PASS
    if not per and not missing:
        verdict = GATE_WITHHELD
    return {
        "verdict": verdict,
        "tolerance_percent": float(tolerance_percent),
        "allow_marginal": bool(allow_marginal),
        "required": {k: int(v) for k, v in req.items()},
        "kinds": kind_status,
        "functionals": per,
        "candidate_only": missing,
        "reason": _verdict_reason(verdict, kind_status),
    }


def _verdict_reason(verdict: str, kind_status: Mapping[str, Mapping]) -> str:
    if verdict == GATE_PASS:
        return "every required functional kind met its quota"
    bad = [f"{k}: {v['status']}" for k, v in sorted(kind_status.items())
           if v["status"] != GATE_PASS]
    return "; ".join(bad) if bad else "no functional evaluated"


def gate_teeth_check(references: Mapping[str, Mapping], *,
                     tolerance_percent: float = 3.0,
                     kinds: Mapping[str, str] | None = None,
                     required: Mapping[str, int] | None = DEFAULT_REQUIRED,
                     perturbation_percent: float = 10.0,
                     allow_marginal: bool = False) -> dict:
    """Do two things at once: pass the reference, and fail a perturbed one.

    A gate battery is only evidence if it **can** fail.  The candidate equal to
    the certified values must reach ``GATE_PASS`` (the gate is satisfiable, i.e.
    not vacuous by construction), and the same candidate with every functional
    displaced by ``perturbation_percent`` must not pass.  Either half failing
    means the gate is mis-wired and its verdicts must not be trusted.
    """
    ref_values = {}
    for name, cert in references.items():
        value = cert.get("certified_value")
        if value is not None:
            ref_values[name] = float(value)
    identity = judge_feature_set(ref_values, references=references,
                                 tolerance_percent=tolerance_percent, kinds=kinds,
                                 required=required, allow_marginal=allow_marginal)
    scale = 1.0 + float(perturbation_percent) / 100.0
    perturbed = {}
    for i, (name, value) in enumerate(sorted(ref_values.items())):
        # alternate the sign so the perturbation cannot cancel inside a kind
        perturbed[name] = value * (scale if i % 2 == 0 else 1.0 / scale)
    moved = judge_feature_set(perturbed, references=references,
                              tolerance_percent=tolerance_percent, kinds=kinds,
                              required=required, allow_marginal=allow_marginal)
    ok = identity["verdict"] == GATE_PASS and moved["verdict"] != GATE_PASS
    return {
        "accepts_identity": identity["verdict"],
        "rejects_perturbed_verdict": moved["verdict"],
        "perturbation_percent": float(perturbation_percent),
        "has_teeth": bool(ok),
        "n_reference_functionals": len(ref_values),
        "identity_verdict": identity["verdict"],
    }
