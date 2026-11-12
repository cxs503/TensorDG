"""Independent re-verification of CASE-DG-PREFINEMENT-EXPONENTIAL.

Why this script exists
----------------------
The shipped gate ``tests/test_p_refinement_exponential.py`` only *reads*
``docs/benchmarks/results/p-refinement-exponential/p_refinement_results.json``
(12 assertions over a stored artifact, ~0.14 s, no solver call).  A stored JSON
can drift away from the code that supposedly produced it, so on its own it is a
self-certifying gate.  This script re-derives the published numbers from the
shipped solver along axes that are *independent* of the original run:

1. **Quadrature independence** — the L2 error is re-measured with an
   NumPy-only path at several Gauss orders (nq = 16/32/64) instead of the
   library's ``l2_error`` (nq = 2p+4).  If the published error is really the
   discretisation error, all orders must agree to ~1e-13 relative.
2. **Norm independence** — the same runs are scored in L1 and L-inf; an
   exponential p-law must survive the change of norm (it is not a quadrature or
   norm artefact).
3. **Integrator independence** — classical RK4 vs. SSP-RK3 (Cockburn & Shu
   1998) for the *same* spatial operator and the *same* dt.  Agreement shows the
   published error is spatial, not temporal.
4. **Artifact cross-check** — recomputed errors vs. the stored JSON, with the
   artifact's sha256 recorded for provenance.
5. **Falsification control of the metric** — ``p_decay_diagnostics`` is fed
   synthetic error sequences with *known* laws (constant / algebraic p^-1, p^-2,
   p^-3 / exponential sigma = 0.5, 1.0).  This quantifies how discriminating the
   predeclared criterion ``sigma > 0.5`` actually is: an algebraic p^-2 sequence
   also clears it, so the criterion alone does not prove exponentiality.

Artifact: ``docs/benchmarks/results/p-refinement-exponential/independent_check.json``

Run::

    python -m benchmarks.p_refinement_independent_check                # p=1..6
    python -m benchmarks.p_refinement_independent_check --degrees 1 2 3 --cfl 0.08
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensordg.advection_1d import l2_error, solve_advection
from tensordg.convergence import p_decay_diagnostics

SCHEMA = "tensordg-p-refinement-independent/1"
OUTDIR = Path("docs") / "benchmarks" / "results" / "p-refinement-exponential"
PRIMARY_ARTIFACT = OUTDIR / "p_refinement_results.json"
NCELLS = 16
FINAL_TIME = 1.0
RK3_CFL_DIVISOR = 8.0  # keeps SSP-RK3 temporal error below the spatial error
SPEED = 1.0


def sha256_of(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legendre_basis(xi: np.ndarray, degree: int) -> np.ndarray:
    """Legendre P_0..P_p at ``xi`` (NumPy only, no library helpers)."""
    coeffs = np.zeros((degree + 1, degree + 1))
    for k in range(degree + 1):
        coeffs[k, k] = 1.0
    return np.stack(
        [np.polynomial.legendre.legval(xi, coeffs[k]) for k in range(degree + 1)],
        axis=1,
    )


def norms_independent(centers: torch.Tensor, coeff: torch.Tensor, final_time: float,
                      speed: float = SPEED, nq: int = 64) -> dict:
    """L2 / L1 / Linf error of the modal solution, NumPy path, ``nq``-point Gauss.

    Deliberately re-implements the evaluation (own quadrature order, own basis
    assembly, own norm reduction) so that it cannot inherit a bug from
    :func:`tensordg.advection_1d.l2_error`.
    """
    c_np = coeff.detach().cpu().numpy().astype(np.float64)
    xc_np = centers.detach().cpu().numpy().astype(np.float64)
    ncells, nmodes = c_np.shape
    degree = nmodes - 1
    h = 1.0 / ncells
    xi, w = np.polynomial.legendre.leggauss(nq)
    phi = _legendre_basis(xi, degree)                 # (nq, p+1)
    uh = c_np @ phi.T                                  # (ncells, nq)
    x = xc_np[:, None] + 0.5 * h * xi[None, :]
    diff = uh - np.sin(2.0 * math.pi * (x - speed * final_time))
    l2 = math.sqrt(0.5 * h * float(np.sum(w[None, :] * diff ** 2)))
    l1 = 0.5 * h * float(np.sum(w[None, :] * np.abs(diff)))
    linf = float(np.max(np.abs(diff)))
    return {"l2": l2, "l1": l1, "linf": linf, "nq": nq}


def falsification_control(degrees: tuple[int, ...]) -> dict:
    """Feed the metric sequences with known decay laws; report the measured sigma."""
    p = np.asarray(degrees, dtype=float)
    p -= p[0]
    scale = 1.0e-2
    cases = {
        "constant": np.full_like(p, 1.0),
        "algebraic_p^-1": 1.0 / (p + 1.0),
        "algebraic_p^-2": 1.0 / (p + 1.0) ** 2,
        "algebraic_p^-3": 1.0 / (p + 1.0) ** 3,
        "exponential_sigma_0.5": np.exp(-0.5 * p),
        "exponential_sigma_1.0": np.exp(-1.0 * p),
        "exponential_sigma_3.93": np.exp(-3.927 * p),
    }
    out = {}
    for name, seq in cases.items():
        diag = p_decay_diagnostics(degrees, [float(scale * v) for v in seq])
        out[name] = {
            "lsq_decay_rate": diag["lsq_decay_rate"],
            "monotone_decreasing": diag["monotone_decreasing"],
            "superlinear": diag["superlinear"],
            "passes_predeclared_sigma_gt_0.5": diag["lsq_decay_rate"] > 0.5,
        }
    return out


def temporal_refinement(degrees: tuple[int, ...], ncell: int = NCELLS,
                        cfl: float = 0.08, halvings: int = 2) -> dict:
    """Error vs. halved CFL: proves the fixed-CFL (cheap) runs are temporally safe.

    The published study refined CFL until the L2 error moved by <1 %.  A live
    gate cannot afford that loop, so this function measures *how much* the error
    changes when the same run is repeated at cfl, cfl/2, cfl/4 — the last-step
    relative change is the temporal-error share of the reported number.
    """
    out = {}
    for degree in degrees:
        seq = []
        prev = None
        for k in range(halvings + 1):
            c = cfl / (2 ** k)
            centers, coeff, meta = solve_advection(
                ncells=ncell, degree=degree, final_time=FINAL_TIME,
                speed=SPEED, cfl=c, time_integrator="rk4",
            )
            e = norms_independent(centers, coeff, FINAL_TIME, nq=64)["l2"]
            seq.append({"cfl": c, "steps": meta["steps"], "l2": e,
                        "rel_change_vs_previous": None if prev is None else abs(e - prev) / e})
            prev = e
            print(f"  temporal p={degree} cfl={c:.4f} steps={meta['steps']} l2={e:.6e}",
                  flush=True)
        out[degree] = {"levels": seq,
                       "final_rel_change": seq[-1]["rel_change_vs_previous"]}
    return out


def run_independent(degrees: tuple[int, ...], ncell: int = NCELLS,
                    cfl: float = 0.08, cross_integrator: bool = True) -> dict:
    levels = []
    for degree in degrees:
        t0 = time.time()
        centers, coeff, meta = solve_advection(
            ncells=ncell, degree=degree, final_time=FINAL_TIME,
            speed=SPEED, cfl=cfl, time_integrator="rk4",
        )
        lib_l2 = l2_error(centers, coeff, FINAL_TIME, speed=SPEED)
        nq_scan = {str(nq): norms_independent(centers, coeff, FINAL_TIME, nq=nq)
                   for nq in (2 * degree + 4, 16, 32, 64)}
        entry = {
            "degree": degree,
            "steps": meta["steps"],
            "dt": meta["dt"],
            "library_l2": lib_l2,
            "nq_scan": nq_scan,
            "quadrature_spread_rel": max(v["l2"] for v in nq_scan.values())
            / min(v["l2"] for v in nq_scan.values()) - 1.0,
            "l1": nq_scan["64"]["l1"],
            "linf": nq_scan["64"]["linf"],
            "wall_seconds": time.time() - t0,
        }
        if cross_integrator:
            # SSP-RK3 is 3rd order: at a CFL that is already safe for RK4 its
            # temporal error swamps the spatial error for high p (measured: at
            # cfl=0.08, p=6, the raw RK3 error is 116x the RK4 one -- a temporal
            # artefact, not a spatial disagreement), so dt is reduced there.
            rk3_cfl = cfl if degree <= 3 else cfl / RK3_CFL_DIVISOR
            c3, k3, meta3 = solve_advection(
                ncells=ncell, degree=degree, final_time=FINAL_TIME,
                speed=SPEED, cfl=rk3_cfl, time_integrator="ssp_rk3",
            )
            e3 = norms_independent(c3, k3, FINAL_TIME, nq=64)["l2"]
            entry["ssp_rk3_l2"] = e3
            entry["integrator_rel_gap"] = abs(e3 - lib_l2) / lib_l2
            entry["ssp_rk3_steps"] = meta3["steps"]
            entry["ssp_rk3_cfl"] = rk3_cfl
        levels.append(entry)
        print(f"  p={degree}: lib_l2={lib_l2:.6e} dt={meta['dt']:.3e} "
              f"steps={meta['steps']} qspread={entry['quadrature_spread_rel']:.1e} "
              f"({entry['wall_seconds']:.1f}s)", flush=True)

    errors = [lv["library_l2"] for lv in levels]
    diag = p_decay_diagnostics(degrees, errors)
    primary = {}
    if PRIMARY_ARTIFACT.exists():
        primary = json.loads(PRIMARY_ARTIFACT.read_text())
    stored = {d: e for d, e in zip(primary.get("results", {}).get("degrees", []),
                                   primary.get("results", {}).get("errors", []))}
    cross = {}
    for lv in levels:
        d = lv["degree"]
        if d in stored:
            cross[d] = {
                "stored": stored[d],
                "recomputed": lv["library_l2"],
                "rel_gap": abs(stored[d] - lv["library_l2"]) / lv["library_l2"],
            }
    return {
        "schema": SCHEMA,
        "case": "CASE-DG-PREFINEMENT-EXPONENTIAL",
        "config": {"ncells": ncell, "cfl": cfl, "final_time": FINAL_TIME,
                   "degrees": list(degrees), "integrator": "rk4",
                   "cross_integrator": "ssp_rk3"},
        "levels": levels,
        "diagnostics": diag,
        "criterion": {"sigma": diag["lsq_decay_rate"], "threshold": 0.5,
                      "pass": diag["lsq_decay_rate"] > 0.5},
        "artifact_provenance": {
            "primary_artifact": str(PRIMARY_ARTIFACT),
            "primary_sha256": sha256_of(PRIMARY_ARTIFACT),
        },
        "artifact_cross_check": cross,
        "falsification_control": falsification_control(tuple(degrees)),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--cfl", type=float, default=0.08)
    ap.add_argument("--ncells", type=int, default=NCELLS)
    ap.add_argument("--no-integrator-cross", action="store_true")
    ap.add_argument("--temporal-check", action="store_true",
                    help="halve CFL (cf. the driver's convergence loop) for a few degrees")
    ap.add_argument("--out", type=str, default=str(OUTDIR / "independent_check.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    degrees = tuple(sorted(args.degrees))
    print(f"Independent re-verification: p={degrees}, ncells={args.ncells}, cfl={args.cfl}")
    report = run_independent(degrees, args.ncells, args.cfl,
                             cross_integrator=not args.no_integrator_cross)
    if args.temporal_check:
        print("Temporal refinement check (demonstrates cfl default is temporally safe)")
        report["temporal_refinement"] = temporal_refinement(degrees, args.ncells, args.cfl)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== summary ===")
    print(f"sigma (independent recompute) = {report['diagnostics']['lsq_decay_rate']:.4f}")
    print(f"threshold 0.5 -> {'PASS' if report['criterion']['pass'] else 'FAIL'}")
    worst_q = max(lv["quadrature_spread_rel"] for lv in report["levels"])
    print(f"worst quadrature spread (rel) = {worst_q:.2e}")
    gaps = [v["rel_gap"] for v in report["artifact_cross_check"].values()]
    if gaps:
        print(f"worst stored-vs-recomputed gap (rel) = {max(gaps):.2e}")
    ig = [lv["integrator_rel_gap"] for lv in report["levels"] if "integrator_rel_gap" in lv]
    if ig:
        print(f"worst RK4-vs-SSPRK3 gap (rel) = {max(ig):.2e}")
    print("\nfalsification control (does sigma>0.5 discriminate?):")
    for name, v in report["falsification_control"].items():
        print(f"  {name:24s} sigma={v['lsq_decay_rate']:.3f} "
              f"passes_gate={v['passes_predeclared_sigma_gt_0.5']}")
    print(f"\nartifact written: {out}")
    return report


if __name__ == "__main__":
    main()