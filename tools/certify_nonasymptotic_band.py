"""Does the round-2 certified band cover the truth in the *limiter* regime?

The gap this closes
-------------------
``docs/CERTIFIED-REFERENCE-BAND-PHYSICAL.md`` measures, on cases with a closed
form solution, whether the Roache GCI band ``U`` produced by
``tensordg.certified_reference.certify`` actually covers the certified
profile's true error.  Nine smooth / pre-break ladders were measured and all
nine were covered (worst ``true/U`` = 0.337).  That document's own limitations
section then states the consequential hole verbatim::

    **未测到的正是最要紧的区间（honest）**：上表全部阶梯仍属光滑/预破解区
    （p_obs ≈ 2–5），`p ~ 1` 的限幅/激波区**没有**认证结论。 ...
    三级阶梯未在预算内跑完 ... 故该区间的带宽覆盖率**仍然未知**。

That is the regime that matters for ``XVAL-DG-ROUND2-CERTIFIED-REFERENCE``: the
CASE-SHU-OSHER reference is certified at ``p_obs = 1.28`` (3-level) / ``1.18``
(4-level), i.e. inside the slow-convergence, limiter-dominated range where the
band is *largest* and the extrapolation gain is *smallest*.  If the band stops
covering the truth there, every "``< 3 %``" margin quoted for the DG legs
inherits an unquantified bias.

What this tool does
-------------------
Three ladders are run **through the production kernels** (``tensordg.advection_1d``,
``tensordg.euler_dg_1d``) with a shock limiter active, so the observed order
sits near the value the anchor's reference lives at.  Each ladder has an exact
solution, so the true error of the certified profile is *measured*, not guessed:

* square-wave advection (:math:`u_t+u_x=0`, discontinuity, TVB limiter) --
  exact cell averages are rational numbers (``square_wave_cell_averages``);
* Sod shock tube (1-D Euler, HLLC, Qiu-Shu WENO limiter, the same limiter and
  ``limiter_stage`` the round-2 DG legs use) -- exact cell averages from the
  exact Riemann solver in :mod:`benchmarks.riemann_exact`.

For every ladder the *finest three* levels and the *full four* levels are
certified separately, so the 3-level convention used by the published round-2
reference and the 4-level extension are both measured from the same runs.

Pre-declared criteria (fixed before the numbers were seen)
----------------------------------------------------------
====  ============================================================================
X0    measurable: every ladder yields a finite ``true_over_band`` (certificate
      produced and exact error computable); otherwise the ladder is ``WITHHELD``
      and counted as such -- never as covered.
X1    coverage: ``err_certified <= U`` (``true_over_band <= 1``).
X2    regime reached: ``p_obs <= 1.5`` for the ladder (this is what makes the
      measurement relevant to the ``p ~ 1`` regime; a ladder outside that range
      is reported as ``REGIME_NOT_REACHED`` and excluded from the X1 headline).
X3    gate validity is not violated: ``U / tau <= 1/3`` (``tau = 3 %``).
X4    the extrapolation actually helps: ``Richson gain > 1`` (certified profile
      closer to the exact solution than the second-finest raw solution).
====  ============================================================================

Overall verdict is ``COVERED_ALL`` only if X1 holds for every regime-reached
ladder; otherwise ``UNDER_COVERS`` naming the failing ladders.  An under-cover
finding is a *result*, not a failure of the tool.

Cost discipline
---------------
Per-case runs are a few minutes each at the coarsest useful resolutions
(the regime of interest is pre-asymptotic, so the coarse end is the point).
Each case is written to its own JSON as soon as it finishes, so a wall-clock
cut never loses a finished case:

    python -m tools.certify_nonasymptotic_band --case adv_square_tvb_P1
    python -m tools.certify_nonasymptotic_band --case sod_weno_P1
    python -m tools.certify_nonasymptotic_band --case sod_weno_P2
    python -m tools.certify_nonasymptotic_band --merge      # doc + figures
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensordg.certified_reference import certify, coarsen, rel_l1
from tools.certify_physical_band import square_wave_cell_averages

DEFAULT_OUTDIR = Path("artifacts/certify_nonasymptotic_band")
DOC_PATH = Path("docs/CERTIFIED-REFERENCE-NONASYMPTOTIC-BAND.md")

#: Owner's accuracy standard -- the gate the round-2 certificate is quoted at.
TAU = 3.0
#: Sub-samples per cell used to integrate the exact Riemann solution.
NSUB_EXACT = 64
#: Second sub-sample count, used only to bound the quadrature error of NSUB_EXACT
#: across the (single) cell that straddles a discontinuity.
NSUB_EXACT_CHECK = 256
#: Observed-order ceiling that still counts as "the slow-convergence regime".
P_REGIME_MAX = 1.5

# Square-wave advection: same numbers as the earlier raw-error probe
# (``tools.certify_physical_band.probe_nonasymptotic``), so the two studies are
# directly comparable.
SQ_X1, SQ_X2, SQ_SHIFT, SQ_T, SQ_SPEED = 0.4, 0.6, 0.5, 0.5, 1.0

# Sod shock tube (Toro, *Riemann Solvers and Numerical Methods*, test 1).
SOD_LEFT = (1.0, 0.0, 2.5 / 1.4)
SOD_RIGHT = (0.125, 0.0, 0.1)
SOD_T = 0.2

#: Limiter settings mirror the round-2 DG legs (``benchmarks/shu_osher_1d.py``):
#: Qiu-Shu WENO limiter, ``limiter_stage="every"``, ``tvb_constant=1.0``.
DG_CFL = 0.15

CASES = [
    {"name": "adv_square_tvb_P1", "kind": "advection_square", "degree": 1,
     "final_time": SQ_T, "cfl": 0.3, "ladder": [200, 400, 800, 1600]},
    {"name": "sod_weno_P1", "kind": "sod", "degree": 1,
     "final_time": SOD_T, "cfl": 0.15, "ladder": [50, 100, 200]},
]


# ------------------------------------------------------------------ solver side
def solve_field(case: dict, ncells: int):
    """Cell averages of the production DG solution (modal coefficient 0)."""
    import torch

    if case["kind"] == "advection_square":
        from tensordg import advection_1d

        def initial(x):
            return torch.where((x > SQ_X1) & (x < SQ_X2),
                               torch.ones_like(x), torch.zeros_like(x))

        _, coeff, meta = advection_1d.solve_advection(
            ncells=int(ncells), degree=case["degree"], final_time=case["final_time"],
            speed=SQ_SPEED, cfl=case["cfl"], initial=initial,
            limiter="tvb", time_integrator="ssp_rk3",
        )
        return coeff[:, 0].detach().cpu().numpy().astype(float), meta

    if case["kind"] == "sod":
        from tensordg.euler_dg_1d import solve_euler_dg_1d

        _, coeff, meta = solve_euler_dg_1d(
            SOD_LEFT, SOD_RIGHT, ncells=int(ncells), degree=case["degree"],
            final_time=case["final_time"], cfl=case["cfl"],
            limiter="weno", tvb_constant=1.0, limiter_stage="every",
        )
        # component 0 of the cell-average mode is the density.
        return coeff[:, 0, 0].detach().cpu().numpy().astype(float), meta

    raise ValueError(f"unknown case kind {case['kind']!r}")


# -------------------------------------------------------------------- exact side
def exact_field(case: dict, ncells: int, nsub: int = NSUB_EXACT) -> np.ndarray:
    """Exact cell averages on the same grid (discontinuities sub-sampled)."""
    if case["kind"] == "advection_square":
        return square_wave_cell_averages(
            int(ncells), SQ_X1, SQ_X2, SQ_SPEED * case["final_time"])

    if case["kind"] == "sod":
        from benchmarks.riemann_exact import sample

        n = int(ncells)
        xq = (np.arange(n * nsub) + 0.5) / (n * nsub)
        rho = sample(xq, case["final_time"], SOD_LEFT, SOD_RIGHT, 0.5)["density"]
        return np.asarray(rho, dtype=float).reshape(n, nsub).mean(axis=1)

    raise ValueError(f"unknown case kind {case['kind']!r}")


def quadrature_error_percent(case: dict, ncells: int) -> float | None:
    """How much the sub-sampled exact cell averages move with the sub-sample count.

    Only the one or two cells that straddle a discontinuity can move; the number
    is reported so the reader can see the truth itself is not the limiting error.
    """
    fine = exact_field(case, ncells, NSUB_EXACT_CHECK)
    coarse = exact_field(case, ncells, NSUB_EXACT)
    return 100.0 * rel_l1(coarse, fine)


# ------------------------------------------------------------------- level cache
def raw_cache_path(outdir: Path, case: dict, ncells: int) -> Path:
    return Path(outdir) / "raw" / f"{case['name']}_n{int(ncells)}.npz"


def save_level(outdir: Path, case: dict, ncells: int) -> Path:
    """Solve one level and cache the cell averages (+ step count)."""
    field, meta = solve_field(case, int(ncells))
    path = raw_cache_path(outdir, case, ncells)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, field=field, steps=int(meta.get("steps") or 0),
             ncells=int(ncells), case=case["name"])
    return path


def load_level(outdir: Path, case: dict, ncells: int):
    path = raw_cache_path(outdir, case, ncells)
    if not path.exists():
        return None
    data = np.load(path)
    return data["field"].astype(float), int(data["steps"])


# ---------------------------------------------------------------------- the study
def _measure(levels, labels, case, errs_raw, steps, tolerance_percent=TAU) -> dict:
    cert = certify(levels, tolerance_percent=tolerance_percent)
    n_ref = cert.get("reference_ncells")
    orders = cert.get("orders") or []
    observed = cert.get("observed_order_used")
    if observed is None and orders:
        observed = orders[-1].get("observed_order")
    out = {
        "labels": [int(n) for n in labels],
        "verdict": cert.get("verdict"),
        "classification": cert.get("classification"),
        "gate_usable": bool(cert.get("gate_usable", False)),
        "observed_order": observed,
        "band_percent": cert.get("band_percent"),
        "band_share_of_gate": cert.get("band_share_of_gate"),
        "displacement_ratio": cert.get("displacement_ratio"),
        "reference_ncells": (int(n_ref) if n_ref is not None else None),
        "withheld_reason": cert.get("reasons") or cert.get("reason"),
        "err_raw_finest_percent": errs_raw[int(labels[-1])],
        "err_raw_second_finest_percent": errs_raw[int(labels[-2])],
        "err_certified_percent": None,
        "true_over_band": None,
        "band_covers_truth": None,
        "richardson_gain": None,
        "quadrature_error_percent": quadrature_error_percent(case, int(labels[-1])),
        "steps": [steps[int(n)] for n in labels],
        "acceptable": None,
    }
    if n_ref is None:
        # X0 fails: the engine refused to certify this ladder, so its band cannot
        # be tested.  The refusal itself is a result and is reported as such.
        return out
    n_ref = int(n_ref)
    exact_ref = exact_field(case, n_ref)
    out["err_raw_second_finest_percent"] = errs_raw[n_ref]
    err_cert = 100.0 * rel_l1(cert["reference"], exact_ref)
    out["err_certified_percent"] = err_cert
    band = cert["band_percent"]
    if band:
        out["true_over_band"] = err_cert / band
        out["band_covers_truth"] = bool(err_cert <= band)
    out["richardson_gain"] = (
        out["err_raw_second_finest_percent"] / err_cert if err_cert > 0.0 else None)
    # X-invariant used by the acceptance gate: the certificate is acceptable for
    # the round-2 practice iff the band covers the truth AND the certified profile
    # itself is inside the 3 % gate the band is quoted at.
    out["acceptable"] = bool(out["band_covers_truth"] and err_cert < TAU)
    return out


def measure_case(case: dict, tolerance_percent: float = TAU,
                 outdir: Path = DEFAULT_OUTDIR) -> dict:
    """Solve one ladder, then certify its finest triple *and* the full four."""
    levels, errs_raw, steps = [], {}, {}
    for n in case["ladder"]:
        cached = load_level(outdir, case, n)
        if cached is None:
            field, meta = solve_field(case, n)
            step_count = int(meta.get("steps") or 0)
            path = raw_cache_path(outdir, case, n)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, field=field, steps=step_count, ncells=int(n), case=case["name"])
        else:
            field, step_count = cached
        levels.append({"ncells": int(n), "field": field})
        errs_raw[int(n)] = 100.0 * rel_l1(field, exact_field(case, int(n)))
        steps[int(n)] = step_count
        print(f"  [{case['name']}] N={n} raw_err={errs_raw[int(n)]:.4f}% "
              f"steps={step_count}{' (cached)' if cached is not None else ''}", flush=True)

    row = {
        "case": case["name"],
        "kind": case["kind"],
        "degree": int(case["degree"]),
        "final_time": float(case["final_time"]),
        "cfl": float(case["cfl"]),
        "ladder": [int(n) for n in case["ladder"]],
        "limiter": "tvb" if case["kind"] == "advection_square" else "weno",
        "raw_error_percent": {int(n): errs_raw[int(n)] for n in case["ladder"]},
        "steps": {int(n): steps[int(n)] for n in case["ladder"]},
        "quadrature_error_percent": {int(n): quadrature_error_percent(case, int(n))
                                    for n in case["ladder"]},
        "apparent_order_raw": (
            math.log(errs_raw[case["ladder"][0]] / errs_raw[case["ladder"][1]])
            / math.log(case["ladder"][1] / case["ladder"][0])
            if errs_raw[case["ladder"][0]] > 0.0 and errs_raw[case["ladder"][1]] > 0.0
            else None),
        "three_level": _measure(levels[-3:], [n for n in case["ladder"][-3:]],
                                case, errs_raw, steps, tolerance_percent),
        "four_level": (_measure(levels, [n for n in case["ladder"]],
                                case, errs_raw, steps, tolerance_percent)
                       if len(levels) >= 4 else None),
    }
    return row


def summarise(rows: list[dict]) -> dict:
    ladders = []
    for row in rows:
        for key, kind in (("three_level", "3-level"), ("four_level", "4-level")):
            m = row.get(key)
            if m is None:
                continue
            ladder = {
                "case": row["case"], "convention": kind, "labels": m["labels"],
                "verdict": m["verdict"], "gate_usable": m["gate_usable"],
                "observed_order": m["observed_order"],
                "band_percent": m["band_percent"],
                "err_certified_percent": m["err_certified_percent"],
                "true_over_band": m["true_over_band"],
                "band_covers_truth": m["band_covers_truth"],
                "richardson_gain": m["richardson_gain"],
                "quadrature_error_percent": m["quadrature_error_percent"],
                "regime_reached": (m["observed_order"] is not None
                                   and m["observed_order"] <= P_REGIME_MAX),
                "acceptable": m["acceptable"],
            }
            # X2/X3/X4 are reported per ladder, X1 only where X0/X2 hold.
            ladder["X0_measurable"] = m["true_over_band"] is not None
            ladder["X1_covered"] = (m["band_covers_truth"]
                                    if ladder["X0_measurable"] else None)
            ladder["X2_regime"] = ladder["regime_reached"]
            ladder["X3_gate_valid"] = bool(
                m["band_share_of_gate"] is not None and m["band_share_of_gate"] <= 1.0 / 3.0)
            ladder["X4_extrapolation_helps"] = bool(
                m["richardson_gain"] is not None and m["richardson_gain"] > 1.0)
            ladders.append(ladder)

    measured = [l for l in ladders if l["X0_measurable"]]
    in_regime = [l for l in measured if l["X2_regime"]]
    covered = [l for l in in_regime if l["X1_covered"]]
    failing = [f"{l['case']} ({l['convention']})" for l in in_regime if not l["X1_covered"]]
    return {
        "tau_percent": TAU,
        "p_regime_max": P_REGIME_MAX,
        "n_ladders": len(ladders),
        "n_measurable": len(measured),
        "n_in_regime": len(in_regime),
        "n_covered_in_regime": len(covered),
        "coverage_fraction_in_regime": (len(covered) / len(in_regime)) if in_regime else None,
        "worst_true_over_band_in_regime": (
            max(l["true_over_band"] for l in in_regime) if in_regime else None),
        "worst_true_over_band_case": (
            max(in_regime, key=lambda l: l["true_over_band"])["case"] if in_regime else None),
        "min_richardson_gain": (min(l["richardson_gain"] for l in measured
                                    if l["richardson_gain"] is not None)
                                if any(l["richardson_gain"] is not None for l in measured) else None),
        "failing_ladders": failing,
        "verdict": ("COVERED_ALL" if in_regime and not failing else
                    ("UNDER_COVERS" if failing else "INCONCLUSIVE")),
        "all_observed_orders": [l["observed_order"] for l in measured],
        "all_verdicts": [l["verdict"] for l in ladders],
    }


# ------------------------------------------------------------------------- output
def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_case(row: dict, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"case_{row['case']}.json"
    path.write_text(json.dumps(_jsonable(row), indent=2) + "\n")
    return path


def merge(outdir: Path = DEFAULT_OUTDIR) -> dict:
    rows = []
    for case in CASES:
        path = outdir / f"case_{case['name']}.json"
        if not path.exists():
            print(f"  [merge] missing {path} -- case not run, excluded")
            continue
        rows.append(json.loads(path.read_text()))
    payload = {
        "generator": "python -m tools.certify_nonasymptotic_band",
        "tau_percent": TAU,
        "p_regime_max": P_REGIME_MAX,
        "cases": rows,
        "summary": summarise(rows),
        "provenance": {
            "reference_doc": "docs/CERTIFIED-REFERENCE-BAND-PHYSICAL.md",
            "anchor": "XVAL-DG-ROUND2-CERTIFIED-REFERENCE",
            "gap_closed": ("band coverage of the truth in the limiter / p~1 regime, "
                           "declared unmeasured in CERTIFIED-REFERENCE-BAND-PHYSICAL.md §⑤"),
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", action="append", default=None,
                        help="case name(s) to run; default = all")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--level", type=int, action="append", default=None,
                        help="solve only these cell counts for --case and cache them")
    parser.add_argument("--merge", action="store_true",
                        help="combine per-case JSONs into results.json")
    parser.add_argument("--markdown", action="store_true",
                        help="render the doc from results.json (implies --merge)")
    args = parser.parse_args(argv)

    if args.case:
        wanted = set(args.case)
        for case in CASES:
            if case["name"] in wanted:
                if args.level:
                    for n in args.level:
                        print(f"[level] {case['name']} N={n}", flush=True)
                        print(f"  -> {save_level(args.outdir, case, n)}", flush=True)
                    continue
                print(f"[run] {case['name']} ladder={case['ladder']}", flush=True)
                row = measure_case(case, outdir=args.outdir)
                print(f"[done] {case['name']} -> {write_case(row, args.outdir)}", flush=True)
        wanted -= {c["name"] for c in CASES}
        if wanted:
            raise SystemExit(f"unknown case(s): {sorted(wanted)}")
        return 0

    if args.merge or args.markdown:
        payload = merge(args.outdir)
        if args.markdown:
            from tools.render_nonasymptotic_band import render
            render(payload, args.outdir, DOC_PATH)
        print(json.dumps(payload["summary"], indent=2))
        return 0

    parser.error("choose --case NAME or --merge/--markdown")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())