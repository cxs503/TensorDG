"""Independent reproducibility audit for the CASE-LAX-DG evidence chain.

Motivation (verification debt): every published CASE-LAX-DG verdict is read out
of a committed ``metrics.json``.  The 78 contract tests in ``tests/`` cross-check
those JSONs against the rendered documents, but **none of them re-executes the
solver**, so an artifact that stopped reproducing after a later merge stays green
forever.  The repository already carries a symptom of exactly that: the
``lax_asymptotic_probe`` artifact and the ``lax_resolution_requirement`` artifact
disagree on the *same* ladder legs at the 1e-4..7e-3 relative level while
reporting identical step counts.

This driver answers three questions independently of the artifact suite:

R1 ``determinism``  does the current solver return bit-identical fields when the
                    same configuration is run twice, and does the bit pattern
                    depend on the BLAS/OpenMP thread count?
R2 ``reproduction`` does a fresh run of the current code reproduce each tracked
                    artifact's numbers?
R3 ``materiality``  where an artifact no longer reproduces, does the drift flip
                    the ledger verdict it supports (3 % bar) or a published
                    qualitative claim?
R4 ``quadrature``   how much of each published reading is bias from the artifacts'
                    10-point Gauss norm rather than discretisation error, and does
                    the finer (20-point) reading flip any PASS/FAIL?

Independence choices (deliberate, to keep the audit from re-using the very code
it audits):

* the DG solver is the object under test, so it *is* re-used
  (``tensordg.euler_dg_1d.solve_euler_dg_1d``, library defaults);
* the *measurement* is re-implemented here: own Legendre recurrence, own
  Gauss-Legendre nodes, own relative L1/L2 reduction.  ``benchmarks.lax_case_pool``
  is **not** imported for ``modal_fields`` / ``_weighted_l1`` / ``_weighted_l2``;
* the exact Riemann solution is taken from ``benchmarks.riemann_exact.sample``
  (the closed-form reference already arbitrated by
  ``lax_reference_crosscheck``), not from a stashed profile;
* the 3 % tolerance is imported from ``benchmarks.lax_case_pool`` on purpose:
  the audit must judge with the *registered* bar, not one of its own choosing.

Usage::

    python -m benchmarks.lax_reproducibility_audit                # full audit
    python -m benchmarks.lax_reproducibility_audit --jobs 6       # parallel legs
    python -m benchmarks.lax_reproducibility_audit --quick        # 2 legs only

Artifacts written to ``docs/benchmarks/results/lax_reproducibility_audit/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_reproducibility_audit"

CASE = "CASE-LAX-DG"
ANCHOR = "CASE-LAX-DG"
LEFT = (0.445, 0.698, 3.528)
RIGHT = (0.500, 0.000, 0.571)
GAMMA = 1.4
DIAPHRAGM = 0.5
FINAL_TIME = 0.13
CFL = 0.35

FIELDS = ("density", "velocity", "pressure")

#: fixed physical half-band around each wave, exactly the pool's R3 region rule
WAVE_BAND = 0.02
C3_MIN_ORDER = 1.0
R3_ORDER_FACTOR = 0.85

#: the artifacts' measurement convention (``benchmarks.lax_case_pool.METRIC``)
ARTIFACT_ORDER = 10
#: converged reading used only to quantify the *quadrature* bias of the bar
CONVERGED_ORDER = 20

#: registered legs of the audit: (cells, degree, label)
LEGS = (
    (400, 1, "p1.N400"),
    (200, 3, "p3.N200"),
    (400, 3, "p3.N400"),   # A0 production leg (honest FAIL)
    (800, 3, "p3.N800"),   # resolution requirement / closure mesh
    (800, 2, "p2.N800"),   # certified configuration
)

#: bitwise reproducibility bar between two runs of the same code
DETERMINISM_TOL = 1e-12
#: an artifact is called "reproducible" when every shared reading agrees here.
#: Not bitwise: the exact Riemann solution is sampled at nodes that can sit arbitrarily
#: close to a discontinuity, so last-bit trajectory differences (different BLAS/threads/
#: machine) are amplified to ~1e-9 in the norms.  Anything above 1e-6 is a real change.
REPRODUCTION_TOL = 1e-6
#: below this relative drift a non-reproducing artifact is quantified as noise
DRIFT_NOISE_FLOOR = 1e-2


# ---------------------------------------------------------------------------
# independent measurement layer
# ---------------------------------------------------------------------------

def _legendre_table(xi: np.ndarray, modes: int) -> np.ndarray:
    """Legendre polynomials P_0..P_{modes-1} at ``xi`` via the three-term recurrence.

    Written out explicitly (instead of ``np.polynomial.legendre.legval``) so the
    audit does not share an implementation with the code it audits.
    """
    table = np.empty((xi.size, modes), dtype=np.float64)
    table[:, 0] = 1.0
    if modes > 1:
        table[:, 1] = xi
    for k in range(1, modes - 1):
        table[:, k + 1] = ((2 * k + 1) * xi * table[:, k] - k * table[:, k - 1]) / (k + 1)
    return table


def primitive_on_gauss_nodes(x, coeff, quadrature_order: int):
    """Conservative -> primitive variables on ``quadrature_order`` Gauss nodes per cell.

    Returns ``(points, primitive, dx, weights)`` where ``points``/``primitive``
    are ``(cells, quadrature_order)`` and ``weights`` are the reference-interval
    Gauss weights (mapped to physical space by the caller through ``0.5*dx``).
    """
    conserved = coeff.detach().cpu().numpy().astype(np.float64)
    centres = x.detach().cpu().numpy().astype(np.float64)
    cells, modes, _ = conserved.shape
    dx = float((centres[-1] - centres[0]) / (cells - 1))
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    basis = _legendre_table(nodes, modes)                      # (q, modes)
    nodal = np.einsum("cmv,qm->cqv", conserved, basis)          # (cells, q, vars)
    rho = nodal[..., 0]
    mom = nodal[..., 1]
    energy = nodal[..., 2]
    primitive = {
        "density": rho,
        "velocity": mom / rho,
        "pressure": (GAMMA - 1.0) * (energy - 0.5 * mom * mom / rho),
    }
    points = centres[:, None] + 0.5 * dx * nodes[None, :]
    return points, primitive, dx, weights


def _relative_norm(points, values, exact, dx, weights, power: int) -> float:
    cell_weight = 0.5 * dx * weights[None, :] * np.ones_like(values)
    diff = np.abs(values - exact) ** power
    reference = np.abs(exact) ** power
    numerator = np.sum(cell_weight * diff) ** (1.0 / power)
    denominator = np.sum(cell_weight * reference) ** (1.0 / power)
    return float(numerator / denominator)


def exact_fields(points):
    from benchmarks.riemann_exact import sample
    return sample(points, FINAL_TIME, LEFT, RIGHT, diaphragm=DIAPHRAGM, gamma=GAMMA)


def wave_band_mask(points) -> np.ndarray:
    """True away from the four wave centres (fixed physical half-band, pool's R3 rule)."""
    from benchmarks.lax_case_pool import exact_positions, exact_waves
    positions = exact_positions(exact_waves())
    mask = np.ones_like(points, dtype=bool)
    for name in ("rarefaction_head", "rarefaction_tail", "contact", "shock"):
        mask &= np.abs(points - positions[name]) > WAVE_BAND
    return mask


def _relative_norm_masked(points, values, exact, dx, weights, power: int,
                          mask) -> float:
    cell_weight = 0.5 * dx * weights[None, :] * np.ones_like(values) * mask
    numerator = np.sum(cell_weight * np.abs(values - exact) ** power) ** (1.0 / power)
    denominator = np.sum(cell_weight * np.abs(exact) ** power) ** (1.0 / power)
    return float(numerator / denominator)


def measure(x, coeff, quadrature_order: int = 20):
    """Relative L1/L2 of the three primitive fields, own quadrature and reduction.

    ``analytic_*`` is the same reading with the four wave bands masked out: it is
    what the pool's R3 gate uses, so the audit can re-derive C3/R3 on fresh numbers.
    """
    points, primitive, dx, weights = primitive_on_gauss_nodes(x, coeff, quadrature_order)
    exact = exact_fields(points)
    mask = wave_band_mask(points)
    l1 = {f: _relative_norm(points, primitive[f], exact[f], dx, weights, 1) for f in FIELDS}
    l2 = {f: _relative_norm(points, primitive[f], exact[f], dx, weights, 2) for f in FIELDS}
    analytic_l1 = {f: _relative_norm_masked(points, primitive[f], exact[f], dx, weights,
                                            1, mask) for f in FIELDS}
    return {"l1_errors": l1, "l2_errors": l2, "analytic_l1_errors": analytic_l1,
            "analytic_fraction": float(mask.mean()),
            "density_min": float(primitive["density"].min()),
            "pressure_min": float(primitive["pressure"].min())}


# ---------------------------------------------------------------------------
# leg runner (executed in worker processes)
# ---------------------------------------------------------------------------

def _run_leg(task) -> dict:
    cells, degree, threads = task
    import torch
    if threads:
        torch.set_num_threads(threads)
    from tensordg.euler_dg_1d import solve_euler_dg_1d

    started = time.time()
    x, coeff, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=cells, degree=degree,
                                       final_time=FINAL_TIME, cfl=CFL)
    wall = time.time() - started
    measurements = {order: measure(x, coeff, quadrature_order=order)
                    for order in (ARTIFACT_ORDER, CONVERGED_ORDER)}
    primary = measurements[ARTIFACT_ORDER]
    # digest of the raw coefficients: the strongest possible reproducibility witness
    raw = coeff.detach().cpu().numpy().astype(np.float64)
    digest = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
    return {
        "cells": cells, "degree": degree, "label": f"p{degree}.N{cells}",
        "threads": int(torch.get_num_threads()),
        "steps": int(meta["steps"]),
        "dt_mean": float(meta.get("dt_mean", FINAL_TIME / max(meta["steps"], 1))),
        "dt_last": float(meta["dt"]),
        "quadrature_order": ARTIFACT_ORDER,
        "coeff_sha256": digest,
        "wall_seconds": float(wall),
        "l1_errors": primary["l1_errors"],
        "l2_errors": primary["l2_errors"],
        "analytic_l1_errors": primary["analytic_l1_errors"],
        "analytic_fraction": primary["analytic_fraction"],
        "density_min": primary["density_min"],
        "pressure_min": primary["pressure_min"],
        "measurements": {str(order): payload for order, payload in measurements.items()},
    }


# ---------------------------------------------------------------------------
# artifact extractors: every tracked JSON that publishes a reading for a leg
# ---------------------------------------------------------------------------

def _artifact_path(*parts) -> Path:
    return ROOT.joinpath("docs", "benchmarks", "results", *parts)


def _pool_rows(legs):
    payload = json.loads(_artifact_path("lax_case_pool", "metrics.json").read_text())
    rows = []
    for entry in payload["ladder"]:
        for norm, key in (("l1", "errors"), ("l2", "l2_errors")):
            for field, value in entry[key].items():
                rows.append({"norm": norm, "field": field, "committed": float(value),
                             "cells": entry["cells"], "degree": entry["degree"]})
    return payload.get("generated_at_utc", "unknown"), rows


def _requirement_rows(legs):
    payload = json.loads(_artifact_path("lax_resolution_requirement", "metrics.json").read_text())
    rows = []
    for entry in payload["legs"]:
        for norm, key in (("l1", "l1_errors"), ("l2", "l2_errors")):
            for field, value in entry[key].items():
                rows.append({"norm": norm, "field": field, "committed": float(value),
                             "cells": entry["cells"], "degree": entry["degree"]})
    return payload.get("generated_at_utc", "unknown"), rows


def _certified_rows(legs):
    payload = json.loads(_artifact_path("lax_certified_configuration", "metrics.json").read_text())
    rows = []
    for entry in payload["legs"]:
        for norm, key in (("l1", "l1_errors"), ("l2", "l2_errors")):
            for field, value in entry[key].items():
                rows.append({"norm": norm, "field": field, "committed": float(value),
                             "cells": entry["cells"], "degree": entry["degree"]})
    return payload.get("generated_at_utc", "unknown"), rows


def _probe_rows(legs):
    payload = json.loads(_artifact_path("lax_asymptotic_probe", "metrics.json").read_text())
    rows = []
    for entry in payload["ladder"]:
        # the probe publishes relative L1 only, under the key "errors"
        for field, value in entry["errors"].items():
            rows.append({"norm": "l1", "field": field, "committed": float(value),
                         "cells": entry["cells"], "degree": entry["degree"]})
    return payload.get("generated_at_utc", "unknown"), rows


def _closure_rows(legs):
    payload = json.loads(_artifact_path("lax_resolution_closure", "metrics.json").read_text())
    rows = []
    for entry in payload["ladder"]:
        for norm, key in (("l1", "l1"), ("l2", "l2")):
            for field, value in entry[key].items():
                rows.append({"norm": norm, "field": field, "committed": float(value),
                             "cells": entry["cells"], "degree": 3})
    for arm in payload.get("arms", []):
        if arm.get("limiter") not in (None, "none"):
            continue
        for norm, key in (("l1", "l1"), ("l2", "l2")):
            for field, value in arm.get(key, {}).items():
                rows.append({"norm": norm, "field": field, "committed": float(value),
                             "cells": entry.get("cells", 400), "degree": 3,
                             "limiter": "none"})
    return payload.get("generated_at_utc", "unknown"), rows


ARTIFACTS = (
    ("lax_case_pool", _pool_rows),
    ("lax_resolution_requirement", _requirement_rows),
    ("lax_certified_configuration", _certified_rows),
    ("lax_asymptotic_probe", _probe_rows),
    ("lax_resolution_closure", _closure_rows),
)


def compare(runs, extractor):
    """Compare every shared (leg, norm, field) reading against a fresh run."""
    index = {}
    for run in runs:
        index.setdefault((run["cells"], run["degree"]), run)
    rows = []
    for entry in extractor(None)[1]:
        run = index.get((entry["cells"], entry["degree"]))
        if run is None:
            continue
        key = "l1_errors" if entry["norm"] == "l1" else "l2_errors"
        rerun = float(run[key][entry["field"]])
        row = {
            "leg": f"p{entry['degree']}.N{entry['cells']}",
            "norm": entry["norm"], "field": entry["field"],
            "committed": entry["committed"], "rerun": rerun,
            "relative_difference": (abs(rerun - entry["committed"])
                                    / max(abs(entry["committed"]), 1e-300)),
        }
        if "limiter" in entry:
            row["limiter"] = entry["limiter"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

def determinism_verdict(runs):
    """Same configuration, different repeats / thread counts -> identical coefficients?"""
    groups = {}
    for run in runs:
        groups.setdefault((run["cells"], run["degree"]), []).append(run)
    rows = []
    for (cells, degree), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        digests = {run["coeff_sha256"] for run in group}
        worst = 0.0
        for field in FIELDS:
            values = [run["l2_errors"][field] for run in group]
            worst = max(worst, (max(values) - min(values)) / max(min(values), 1e-300))
        rows.append({
            "leg": f"p{degree}.N{cells}",
            "repeats": len(group),
            "threads": [run["threads"] for run in group],
            "identical_coefficients": len(digests) == 1,
            "worst_relative_l2_spread": float(worst),
            "verdict": "PASS" if len(digests) == 1 and worst <= DETERMINISM_TOL else "FAIL",
        })
    if not rows:
        return {"available": False, "reason": "no repeated leg in this run"}
    return {"available": True, "tolerance": DETERMINISM_TOL, "rows": rows,
            "verdict": "PASS" if all(r["verdict"] == "PASS" for r in rows) else "FAIL"}


def _ledger_bar():
    from benchmarks.lax_case_pool import ledger_acceptance_tolerance
    return float(ledger_acceptance_tolerance())


def quadrature_verdict(runs):
    """How much of each published reading is quadrature bias rather than discretisation error?

    The pool measures the relative norms on uniform ``ARTIFACT_ORDER`` Gauss nodes, so a
    cell whose exact solution jumps between two nodes carries a quadrature error.
    Re-measuring the *same* field with more nodes isolates that bias.
    """
    bar = _ledger_bar()
    rows = []
    for run in runs:
        coarse = run["measurements"][str(ARTIFACT_ORDER)]["l2_errors"]
        fine = run["measurements"][str(CONVERGED_ORDER)]["l2_errors"]
        for field in FIELDS:
            rows.append({
                "leg": run["label"], "field": field, "norm": "l2",
                "order_10": coarse[field], "order_20": fine[field],
                "quadrature_bias": (abs(coarse[field] - fine[field])
                                    / max(abs(fine[field]), 1e-300)),
                "verdict_10": "PASS" if coarse[field] < bar else "FAIL",
                "verdict_20": "PASS" if fine[field] < bar else "FAIL",
                "flips_verdict": (coarse[field] < bar) != (fine[field] < bar),
            })
    worst = max((row["quadrature_bias"] for row in rows), default=0.0)
    return {"tolerance": bar, "rows": rows, "worst_quadrature_bias": float(worst),
            "flips": [row["leg"] + "." + row["field"] for row in rows if row["flips_verdict"]],
            "verdict": ("PASS" if not any(row["flips_verdict"] for row in rows)
                        else "FLIPPED")}


def gate_order(coarse_cells, coarse_error, fine_cells, fine_error) -> float:
    return float(np.log(coarse_error / fine_error)
                 / np.log(fine_cells / coarse_cells))


def gates_verdict(runs):
    """Re-derive the pool's C3 / R3 order gates from the fresh runs.

    C3: whole-field relative L1 order >= 1.0.  R3: wave-band-masked relative L1
    order >= 0.85 * (p + 1).  Both use the finest pair available per degree, which
    is the pool's own gating convention, so a difference can only come from the
    solver, not from the criterion.
    """
    rows = []
    by_degree = {}
    for run in runs:
        by_degree.setdefault(run["degree"], {})[run["cells"]] = run
    for degree, legs in sorted(by_degree.items()):
        cells = sorted(legs)
        if len(cells) < 2:
            continue
        coarse, fine = cells[-2], cells[-1]
        requirement_r3 = R3_ORDER_FACTOR * (degree + 1)
        for field in FIELDS:
            c3 = gate_order(coarse, legs[coarse]["l1_errors"][field],
                            fine, legs[fine]["l1_errors"][field])
            r3 = gate_order(coarse, legs[coarse]["analytic_l1_errors"][field],
                            fine, legs[fine]["analytic_l1_errors"][field])
            rows.append({
                "degree": degree, "field": field, "pair": [coarse, fine],
                "c3_order": c3, "c3_requirement": C3_MIN_ORDER,
                "c3_verdict": "PASS" if c3 >= C3_MIN_ORDER else "FAIL",
                "r3_order": r3, "r3_requirement": requirement_r3,
                "r3_verdict": "PASS" if r3 >= requirement_r3 else "FAIL",
            })
    if not rows:
        return {"available": False, "reason": "no degree has two legs in this run"}
    gated_c3 = min(row["c3_order"] for row in rows if row["degree"] == max(by_degree))
    return {"available": True, "rows": rows,
            "c3_gated_order": float(gated_c3),
            "verdict": "PASS" if all(row["c3_verdict"] == "PASS"
                                     and row["r3_verdict"] == "PASS" for row in rows)
                       else "FAIL"}


def materiality_verdict(reproduction, runs):
    """Does any drift flip the 3 % ledger verdict a reading supports?"""
    bar = _ledger_bar()
    by_leg = {(run["cells"], run["degree"]): run for run in runs}
    checks = []

    def add(label, leg, field, committed, rerun):
        checks.append({
            "claim": label, "leg": leg, "field": field, "tolerance": bar,
            "committed_value": committed, "rerun_value": rerun,
            "committed_verdict": "PASS" if committed < bar else "FAIL",
            "rerun_verdict": "PASS" if rerun < bar else "FAIL",
            "flips_verdict": (committed < bar) != (rerun < bar),
        })

    for artifact, payload in reproduction["by_artifact"].items():
        for row in payload["rows"]:
            if row["norm"] != "l2":
                continue
            key = row["leg"]
            cells, degree = int(key.split(".N")[1]), int(key[1])
            run = by_leg.get((cells, degree))
            if run is None:
                continue
            add(f"{artifact} whole-field rel L2 bar", key, row["field"],
                row["committed"], float(run["l2_errors"][row["field"]]))

    return {
        "tolerance": bar,
        "checks": checks,
        "flips": [c for c in checks if c["flips_verdict"]],
        "verdict": "STABLE" if not any(c["flips_verdict"] for c in checks) else "FLIPPED",
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover - best effort provenance
        return "unknown"


def run_audit(legs=LEGS, repeat_leg=(400, 3), threads_pair=(None, 1), jobs=1):
    tasks = [(cells, degree, None) for cells, degree, _ in legs]
    if repeat_leg is not None:
        # R1: same configuration twice, and once with a pinned thread count
        tasks.append((repeat_leg[0], repeat_leg[1], threads_pair[1]))
        tasks.append((repeat_leg[0], repeat_leg[1], None))

    started = time.time()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            runs = list(pool.map(_run_leg, tasks))
    else:
        runs = [_run_leg(task) for task in tasks]

    reproduction = {"tolerance": REPRODUCTION_TOL, "by_artifact": {}}
    for name, extractor in ARTIFACTS:
        generated, _ = extractor(legs)
        rows = compare(runs, extractor)
        worst = max((row["relative_difference"] for row in rows), default=0.0)
        reproduction["by_artifact"][name] = {
            "artifact": str(_artifact_path(name, "metrics.json").relative_to(ROOT)),
            "generated_at_utc": generated,
            "rows": rows,
            "shared_readings": len(rows),
            "worst_relative_difference": float(worst),
            "verdict": "REPRODUCIBLE" if rows and worst <= REPRODUCTION_TOL else
                       ("NOT_COMPARED" if not rows else "STALE"),
        }

    determinism = determinism_verdict(runs)
    materiality = materiality_verdict(reproduction, runs)
    quadrature = quadrature_verdict(runs)
    gates = gates_verdict(runs)

    stale = [name for name, payload in reproduction["by_artifact"].items()
             if payload["verdict"] == "STALE"]
    return {
        "case": CASE,
        "anchor": ANCHOR,
        "question": ("do the tracked CASE-LAX-DG artifacts still reproduce from the "
                     "current solver, is the solver deterministic, and does any drift "
                     "flip the 3 % ledger bar?"),
        "independent_layers": {
            "solver": "tensordg.euler_dg_1d.solve_euler_dg_1d (library defaults, cfl=0.35)",
            "measurement": "re-implemented in this module (own Legendre recurrence + "
                           "Gauss-Legendre reduction); benchmarks.lax_case_pool norms NOT reused",
            "reference": "benchmarks.riemann_exact.sample (closed form)",
            "tolerance": "benchmarks.lax_case_pool.ledger_acceptance_tolerance()",
        },
        "config": {"legs": [[cells, degree] for cells, degree, _ in legs],
                   "cfl": CFL, "final_time": FINAL_TIME, "gamma": GAMMA,
                   "artifact_quadrature_order": ARTIFACT_ORDER,
                   "converged_quadrature_order": CONVERGED_ORDER, "jobs": jobs,
                   "repeat_leg": list(repeat_leg) if repeat_leg else None},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
            "machine": platform.machine(),
            "git_head": _git_head(),
            "cpu_count": os.cpu_count(),
        },
        "runs": runs,
        "reproduction": reproduction,
        "determinism": determinism,
        "materiality": materiality,
        "quadrature": quadrature,
        "gates": gates,
        "verdict": "PASS" if (not stale and determinism.get("verdict") in (None, "PASS")
                              and materiality["verdict"] == "STABLE"
                              and quadrature["verdict"] == "PASS") else "ATTENTION",
        "stale_artifacts": stale,
        "wall_seconds": float(time.time() - started),
    }


def write_outputs(metrics, out_dir=OUT):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    # digest covers the payload *without* the digest field, so it can be re-verified
    payload.pop("metrics_sha256", None)
    digest = hashlib.sha256(
        json.dumps(payload, indent=2, sort_keys=True).encode()).hexdigest()
    payload["metrics_sha256"] = digest
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out_dir / "metrics.json").write_text(text)
    (out_dir / "metrics.sha256").write_text(f"{digest}  metrics.json\n")
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1,
                        help="legs executed in parallel worker processes")
    parser.add_argument("--quick", action="store_true",
                        help="only the two decisive legs (p3.N400, p3.N800)")
    parser.add_argument("--repeat-leg", default="400:3",
                        help="leg run twice for the determinism check (cells:degree)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    legs = tuple(leg for leg in LEGS if not args.quick or leg[2] in ("p3.N400", "p3.N800"))
    repeat_leg = None
    if args.repeat_leg and args.repeat_leg.lower() not in ("none", ""):
        cells, degree = args.repeat_leg.split(":")
        repeat_leg = (int(cells), int(degree))
    metrics = run_audit(legs=legs, repeat_leg=repeat_leg, jobs=args.jobs)
    digest = write_outputs(metrics, args.out)

    print(f"[repro] git HEAD          : {metrics['environment']['git_head']}")
    print(f"[repro] determinism       : {metrics['determinism'].get('verdict')}")
    for name, payload in metrics["reproduction"]["by_artifact"].items():
        print(f"[repro] {name:28s} {payload['verdict']:14s} "
              f"worst rel diff {payload['worst_relative_difference']:.3e} "
              f"({payload['shared_readings']} readings)")
    print(f"[repro] ledger bar        : {metrics['materiality']['tolerance']:.4g} "
          f"-> {metrics['materiality']['verdict']}")
    print(f"[repro] quadrature bias   : worst {metrics['quadrature']['worst_quadrature_bias']:.3e}"
          f" -> {metrics['quadrature']['verdict']}"
          f" (flips {metrics['quadrature']['flips']})")
    print(f"[repro] C3/R3 on fresh run: {metrics['gates'].get('verdict')}")
    print(f"[repro] overall verdict   : {metrics['verdict']}")
    print(f"[repro] metrics sha256    : {digest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())