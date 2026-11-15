"""CKS1998 §5.1 smooth 2-D Euler case: *certified* order, not a point estimate.

Why this module exists
----------------------
``benchmarks/cks1998_smooth_2d.py`` measures the observed order of accuracy of
the RKDG solver on the paper's own smooth multidimensional test case
(``rho = 1 + 0.2 sin(pi (x + y - 2t))``, ``u = v = 1``, ``p = 1``, periodic
``[0, 2]^2``) and the platform gate ``sm2d_order_p{1,2,3}`` compares *one*
number per degree against the design order ``p + 1`` within 3 %.

A single number is not a verification statement.  On a finite ladder of meshes
an order estimate carries

* an **estimator spread** (adjacent-pair order vs least-squares order vs the
  order from the next-coarser pair),
* an **asymptotic-range** question (an order measured outside the asymptotic
  range is not the order of the scheme),
* a **space/time coupling** (the solver is SSP-RK3, so at fixed ``CFL`` the
  temporal error shrinks together with the mesh error and can mask a deficient
  spatial operator).

This module turns each of those into a number and then answers one question
that a point estimate cannot answer:

    is the 3 % order gate **certified** (deviation + estimator uncertainty both
    inside the tolerance, on meshes that are demonstrably asymptotic), or is it
    only a **point estimate that happens to pass** (``UNCERTIFIED``)?

Definitions (all pre-registered here, before any number is looked at)
--------------------------------------------------------------------
``e_i`` integrals ``L1(rho_h - rho_exact)`` on meshes ``h_i`` (fine -> coarse
indexing below), ``r_i = h_{i} / h_{i+1}`` the actual mesh ratio.

* adjacent-pair order: ``p_i = ln(e_i / e_{i+1}) / ln r_i``
* least-squares order: ``p_lsq`` = slope of ``ln e`` vs ``ln h`` over a level set
* Richardson extrapolated error:
  ``e_R = e_fine + (e_fine - e_coarse)/(r^p - 1)``.  If the ladder is asymptotic
  and ``p`` is the true order, ``e_R -> 0``; the **extrapolation residual**
  ``rho_R = |e_R| / e_fine`` therefore asks "does the measured order *explain*
  the error decay?"  A Roache GCI is deliberately **not** reported here: the
  functional being extrapolated is the error itself, so the GCI of an error
  ladder is ``O(1)`` by construction and says nothing
* asymptotic ratio: ``sigma_i = p_{i+1} / p_i`` (order on the *coarser* pair over
  the order on the finer pair); asymptotic range is declared for
  ``0.9 <= sigma <= 1.1``
* estimator uncertainty of the finest-pair order:
  ``U = max |X - p_fine|`` over the alternative estimators
  ``X in {p_lsq(all levels), p_lsq(finest three), p_next_pair}``
* relative deviation ``delta = |p_fine - (p+1)| / (p+1)``

Verdicts (never silently upgraded):

* ``CERTIFIED``   -- ``delta + U/(p+1) <= tol``, asymptotic range holds **and**
  the measured order extrapolates the error to zero (``rho_R <= 3 %``).
  Under the pre-registered tolerance the scheme is verified, and the tolerance
  is wider than the estimator uncertainty.
* ``UNVERIFIED_BAND`` -- ``delta <= tol`` but the band does not fit: the point
  estimate passes, the *claim* "the order is within tol of p+1" is not
  supported by the ladder.  This is the honest reading of a gate that is
  narrower than the measurement uncertainty.
* ``FAIL``        -- ``delta > tol``: the scheme is outside tolerance on the
  finest mesh available.

Time-error separation runs the finest pair at ``CFL`` and at ``CFL/beta``:
``eta = |e(CFL) - e(CFL/beta)| / e(CFL)`` is the fraction of the error that the
time integrator owns.  The spatial order is treated as isolated only if
``eta <= 0.05`` **and** the order moves by less than 1 % between the two runs.

Cluster usage (one shard per configuration so the sweep can be scattered)::

    python -m benchmarks.cks1998_order_certification --config-index 3 --num-configs 12
    python -m benchmarks.cks1998_order_certification --assemble validation/.../shard_*.json
    python -m benchmarks.cks1998_order_certification --local-smoke   # tiny, seconds
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "validation" / "cks1998_order_certification"
SHARD_DIR = ARTIFACTS / "shards"

#: Pre-registered acceptance numbers (declared before the sweep was run).
TOLERANCE_REL = 0.03          # the platform gate: |order - (p+1)| / (p+1) <= 3 %
SIGMA_BAND = (0.9, 1.1)       # asymptotic-range band for p_coarse / p_fine
EXTRAPOLATION_RESIDUAL_MAX = 0.03   # |e_R|/e_fine for a consistent ladder
TIME_FRACTION_MAX = 0.05      # eta: temporal share of the error must be <= 5 %
ORDER_DRIFT_MAX = 0.01        # relative order change under dt -> dt/beta

FIELD = "density"
NORM = "L1"


# ---------------------------------------------------------------------------
# Pure estimators (torch-free, unit-testable)
# ---------------------------------------------------------------------------
def adjacent_order(e_coarse: float, e_fine: float, ratio: float) -> float:
    """``ln(e_coarse / e_fine) / ln(ratio)``."""
    if e_coarse <= 0.0 or e_fine <= 0.0:
        raise ValueError("errors must be positive")
    if ratio <= 1.0:
        raise ValueError(f"mesh ratio must exceed 1, got {ratio}")
    return math.log(e_coarse / e_fine) / math.log(ratio)


def least_squares_order(h: list[float], errors: list[float]) -> float:
    """Slope of ``ln e`` vs ``ln h`` (errors strictly decreasing required)."""
    if len(h) != len(errors) or len(h) < 2:
        raise ValueError("need >= 2 matched samples")
    xs = [math.log(v) for v in h]
    ys = [math.log(v) for v in errors]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("degenerate mesh ladder")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def richardson_extrapolated(f_coarse: float, f_fine: float, ratio: float,
                            order: float) -> float:
    """Discretisation limit from two levels and a known order."""
    denom = ratio ** order - 1.0
    if denom == 0.0:
        raise ValueError("order 0 has no Richardson limit")
    return f_fine + (f_fine - f_coarse) / denom


def extrapolation_residual(e_coarse: float, e_fine: float, ratio: float,
                           order: float) -> float:
    """``|e_R| / e_fine`` with ``e_R`` the Richardson-extrapolated error."""
    if e_fine <= 0.0:
        raise ValueError("residual needs a positive fine error")
    return abs(richardson_extrapolated(e_coarse, e_fine, ratio, order)) / e_fine


def asymptotic_ratio(order_coarse_pair: float, order_fine_pair: float) -> float:
    """``p_{i+1} / p_i`` for the two finest pairs."""
    if order_fine_pair == 0.0:
        raise ValueError("fine-pair order is zero")
    return order_coarse_pair / order_fine_pair


def certify(ladder: list[dict], design_order: int, tolerance: float = TOLERANCE_REL,
            sigma_band: tuple[float, float] = SIGMA_BAND) -> dict:
    """Judge a mesh ladder (coarse -> fine) against the design order.

    ``ladder`` entries need ``h`` and ``error`` (``L1`` of ``rho``); they are
    sorted by ``h`` internally.  Returns the full evidence bundle plus the
    three-valued verdict documented in the module docstring.
    """
    ordered = sorted(ladder, key=lambda r: r["h"], reverse=True)  # coarse -> fine
    if len(ordered) < 3:
        raise ValueError("certification needs at least 3 levels")
    pairs = []
    for coarse, fine in zip(ordered, ordered[1:]):
        ratio = coarse["h"] / fine["h"]
        pairs.append({
            "h_coarse": coarse["h"], "h_fine": fine["h"], "ratio": ratio,
            "order": adjacent_order(coarse["error"], fine["error"], ratio),
            "richardson_limit": richardson_extrapolated(
                coarse["error"], fine["error"], ratio,
                1.0),  # placeholder order filled below once p_fine is known
        })
    p_fine = pairs[-1]["order"]
    p_next = pairs[-2]["order"] if len(pairs) > 1 else None
    p_lsq_all = least_squares_order([r["h"] for r in ordered],
                                    [r["error"] for r in ordered])
    p_lsq_three = least_squares_order([r["h"] for r in ordered[-3:]],
                                      [r["error"] for r in ordered[-3:]])
    # Richardson limit / GCI of the finest pair with the finest-pair order
    ratio = pairs[-1]["ratio"]
    pairs[-1]["richardson_limit"] = richardson_extrapolated(
        ordered[-2]["error"], ordered[-1]["error"], ratio, p_fine)
    residual = extrapolation_residual(ordered[-2]["error"], ordered[-1]["error"],
                                      ratio, p_fine)
    pairs[-1]["extrapolation_residual"] = residual

    estimators = {"p_lsq_all": p_lsq_all, "p_lsq_finest3": p_lsq_three}
    if p_next is not None:
        estimators["p_next_pair"] = p_next
    uncertainty = max(abs(v - p_fine) for v in estimators.values())
    delta = abs(p_fine - design_order) / design_order
    delta_band = delta + uncertainty / design_order
    sigma = asymptotic_ratio(p_next, p_fine) if p_next is not None else None
    asymptotic_ok = sigma is not None and sigma_band[0] <= sigma <= sigma_band[1]

    consistent = residual <= EXTRAPOLATION_RESIDUAL_MAX
    if delta > tolerance:
        verdict = "FAIL"
    elif delta_band <= tolerance and asymptotic_ok and consistent:
        verdict = "CERTIFIED"
    else:
        verdict = "UNVERIFIED_BAND"
    return {
        "design_order": design_order,
        "verdict": verdict,
        "pairs": pairs,
        "order_fine": p_fine,
        "order_next": p_next,
        "estimators": estimators,
        "uncertainty": uncertainty,
        "delta_rel": delta,
        "delta_band_rel": delta_band,
        "asymptotic_ratio": sigma,
        "asymptotic_ok": asymptotic_ok,
        "tolerance_rel": tolerance,
        "sigma_band": list(sigma_band),
        "extrapolation_residual": residual,
        "extrapolation_consistent": consistent,
        "richardson_limit": pairs[-1]["richardson_limit"],
        "levels": [{"h": r["h"], "error": r["error"], "ncells": r.get("ncells")}
                   for r in ordered],
    }


def time_error_fraction(errors: list[float]) -> float:
    """``|e(CFL) - e(CFL/beta)| / e(CFL)`` for the two time-step runs."""
    if len(errors) != 2:
        raise ValueError("need exactly the base-CFL and the reduced-CFL error")
    base, reduced = errors
    if base <= 0.0:
        raise ValueError("base error must be positive")
    return abs(base - reduced) / base


# ---------------------------------------------------------------------------
# Certify the ladder the platform gate actually used
# ---------------------------------------------------------------------------
#: stdout of the L4 workload run whose metrics are in ``tensorbench_results``
#: (mission 20922, cluster job cj3053): the ``sm2d_order_p*`` rows are built from
#: exactly these three-point ladders.
PLATFORM_STDOUT = (Path(__file__).resolve().parents[1] / "validation"
                   / "repro_cks1998" / "artifacts" / "l4_2d" / "stdout.txt")
#: side length of the periodic box of the CKS1998 §5.1 smooth 2-D case
DOMAIN_LENGTH = 2.0


def from_workload_stdout(path: Path = PLATFORM_STDOUT) -> tuple[list[dict], dict]:
    """Ladder records out of a platform workload stdout document.

    Returns ``(records, platform_orders)``.  ``records`` are ordinary ladder
    entries (``h = 2 / ncells``), so the very numbers the platform scored can be
    re-certified here without re-running anything.  ``platform_orders`` keeps the
    order the workload *stored* next to each ladder, for reconciliation.
    """
    text = path.read_text()
    doc = json.loads(text[text.index("{"):])
    ladders = doc["validation"]["smooth_2d"]["ladders"]
    records, orders = [], {}
    for name, ladder in ladders.items():
        degree = int(name[1:])
        orders[degree] = {"observed_order": ladder["observed_order"],
                          "design_order": ladder["design_order"]}
        for ncells, error in zip(ladder["cells"], ladder["l1_density"]):
            records.append({"kind": "ladder", "degree": degree, "ncells": int(ncells),
                            "cfl": None, "h": DOMAIN_LENGTH / int(ncells),
                            "error": float(error), "source": str(path)})
    return records, orders


# ---------------------------------------------------------------------------
# Cluster configuration list (deterministic, so shards are reproducible)
# ---------------------------------------------------------------------------
#: Order ladders per degree.  Meshes are chosen so that the *finest* pair of
#: each degree still sits in the asymptotic range (verified by the sweep, not
#: assumed).  ``final_time = 0.5`` is half a period of the sine wave.
LADDERS = {
    1: (16, 32, 64),
    2: (12, 16, 24, 32),
    3: (8, 12, 16, 24),
}
#: Mesh *pair* used for the space/time separation runs (same pair at both CFLs,
#: so the order at reduced ``dt`` is directly comparable with the order at the
#: base ``dt`` -- keeping the finest pair here would only cost time steps).
TIME_PAIR = {1: (16, 32), 2: (12, 16), 3: (8, 12)}
#: ``dt -> dt / beta`` factor for the time-error run.
TIME_REFINEMENT = 4
FINAL_TIME = 0.5


def configs(ladders=None, time_pair=None, final_time: float = FINAL_TIME) -> list[dict]:
    """Deterministic configuration list: one entry = one solver run."""
    ladders = ladders or LADDERS
    time_pair = time_pair or TIME_PAIR
    out = []
    for degree, meshes in sorted(ladders.items()):
        for ncells in meshes:
            out.append({"kind": "ladder", "degree": degree, "ncells": ncells,
                        "cfl": 0.2, "final_time": final_time})
    for degree, meshes in sorted(time_pair.items()):
        for ncells in meshes:
            for cfl0 in (0.2, 0.2 / TIME_REFINEMENT):
                out.append({"kind": "time", "degree": degree, "ncells": ncells,
                            "cfl": cfl0, "final_time": final_time})
    return out


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def run_config(cfg: dict) -> dict:
    """Run one configuration and return a JSON-serialisable record."""
    from benchmarks.cks1998_smooth_2d import run_case

    result = run_case(cfg["degree"], cfg["ncells"], None, 1.0, False,
                      final_time=cfg["final_time"], cfl0=cfg["cfl"])
    return {
        **cfg,
        "h": result["h"],
        "error": result["norms"][FIELD][NORM],
        "error_rel": result["norms"][FIELD]["L1_rel"],
        "linf": result["norms"][FIELD]["Linf"],
        "steps": result["steps"],
        "dt": result["dt"],
        "wall_seconds": result["wall_seconds"],
        "all_fields_L1_rel": {k: v["L1_rel"] for k, v in result["norms"].items()},
    }


def assemble(records: list[dict], tolerance: float = TOLERANCE_REL) -> dict:
    """Turn raw runs into per-degree certification bundles."""
    out = {"tolerance_rel": tolerance, "sigma_band": list(SIGMA_BAND),
           "time_fraction_max": TIME_FRACTION_MAX,
           "extrapolation_residual_max": EXTRAPOLATION_RESIDUAL_MAX, "field": FIELD, "norm": NORM,
           "degrees": {},
           "runs": sorted(records, key=lambda r: (r["degree"], r["ncells"], r.get("cfl", 0.0)))}
    for degree in sorted({r["degree"] for r in records}):
        ladder = [r for r in records
                  if r["degree"] == degree and r["kind"] == "ladder"]
        ladder.sort(key=lambda r: r["ncells"])
        bundle = {"ladder": ladder}
        if len(ladder) >= 3:
            bundle["certification"] = certify(
                [{"h": r["h"], "error": r["error"], "ncells": r["ncells"]} for r in ladder],
                degree + 1, tolerance)
        else:
            bundle["certification"] = {"verdict": "INSUFFICIENT_LEVELS"}
        time_runs = [r for r in records if r["degree"] == degree and r["kind"] == "time"]
        if time_runs:
            by_cfl = {}
            for run in time_runs:
                by_cfl.setdefault(run["cfl"], []).append(run)
            cfls = sorted(by_cfl)
            bundle["time_separation"] = {"cfls": cfls, "pairs": {}}
            for cfl in cfls:
                runs = sorted(by_cfl[cfl], key=lambda r: r["ncells"])
                if len(runs) != 2:
                    continue
                coarse, fine = runs
                bundle["time_separation"]["pairs"][f"cfl{cfl:g}"] = {
                    "h_coarse": coarse["h"], "h_fine": fine["h"],
                    "error_coarse": coarse["error"], "error_fine": fine["error"],
                    "order": adjacent_order(coarse["error"], fine["error"],
                                            coarse["h"] / fine["h"]),
                }
            if len(cfls) == 2 and all(len(by_cfl[c]) == 2 for c in cfls):
                base, reduced = cfls
                fine_base = max(by_cfl[base], key=lambda r: r["ncells"])
                fine_reduced = max(by_cfl[reduced], key=lambda r: r["ncells"])
                eta = time_error_fraction([fine_base["error"], fine_reduced["error"]])
                order_drift = abs(
                    bundle["time_separation"]["pairs"][f"cfl{reduced:g}"]["order"]
                    - bundle["time_separation"]["pairs"][f"cfl{base:g}"]["order"]
                ) / max(abs(bundle["time_separation"]["pairs"][f"cfl{base:g}"]["order"]), 1e-30)
                bundle["time_separation"].update({
                    "cfl_base": base, "cfl_reduced": reduced, "beta": TIME_REFINEMENT,
                    "eta": eta, "temporal_share_ok": eta <= TIME_FRACTION_MAX,
                    "order_drift": order_drift,
                    "order_drift_ok": order_drift <= ORDER_DRIFT_MAX,
                    "spatial_isolated": bool(eta <= TIME_FRACTION_MAX
                                             and order_drift <= ORDER_DRIFT_MAX),
                })
        out["degrees"][f"p{degree}"] = bundle
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_markdown(payload: dict) -> str:
    lines = ["| P^k | 设计阶 | 最细对观测阶 | 估计子离散 U | δ | δ+U/(k+1) | σ=p(粗对)/p(细对) "
             "| 渐近区 | ρ_R | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, bundle in payload["degrees"].items():
        cert = bundle["certification"]
        if "order_fine" not in cert:
            lines.append(f"| {name} | {cert.get('verdict')} | | | | | | | | |")
            continue
        lines.append(
            f"| {name} | {cert['design_order']} | {cert['order_fine']:.4f} "
            f"| {cert['uncertainty']:.4f} | {cert['delta_rel']*100:.2f} % "
            f"| {cert['delta_band_rel']*100:.2f} % "
            f"| {cert['asymptotic_ratio']:.3f} | {'yes' if cert['asymptotic_ok'] else 'no'} "
            f"| {cert['extrapolation_residual']*100:.3f} % | **{cert['verdict']}** |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-index", type=int)
    parser.add_argument("--num-configs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=SHARD_DIR)
    parser.add_argument("--assemble", nargs="*", type=Path)
    parser.add_argument("--from-platform-artifact", action="store_true",
                        help="certify the ladder the platform gate actually scored")
    parser.add_argument("--local-smoke", action="store_true",
                        help="tiny ladder, seconds, for unit-level debugging")
    args = parser.parse_args(argv)

    if args.assemble is not None:
        patterns = args.assemble or [SHARD_DIR / "shard_*.json"]
        files = []
        for pattern in patterns:
            files.extend(sorted(glob.glob(str(pattern))))
        records = []
        for path in files:
            records.extend(json.loads(Path(path).read_text())["runs"])
        payload = assemble(records)
        payload["shard_files"] = [str(p) for p in files]
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        out = ARTIFACTS / "certification.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        (ARTIFACTS / "certification.md").write_text(render_markdown(payload) + "\n")
        print(render_markdown(payload))
        print(f"\nwrote {out}")
        return 0

    if args.from_platform_artifact:
        records, platform_orders = from_workload_stdout()
        payload = assemble(records)
        payload["source"] = str(PLATFORM_STDOUT)
        payload["platform_orders"] = {f"p{d}": v for d, v in sorted(platform_orders.items())}
        for degree, info in payload["platform_orders"].items():
            bundle = payload["degrees"][degree]["certification"]
            # the workload's stored order is a least-squares fit; reconciliation
            # of the gate's own number against an independent recomputation
            info["recomputed_lsq"] = bundle["estimators"]["p_lsq_all"]
            info["reconciliation_rel"] = abs(info["observed_order"]
                                             - bundle["estimators"]["p_lsq_all"]) / abs(
                info["observed_order"])
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        out = ARTIFACTS / "certification.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        (ARTIFACTS / "certification.md").write_text(render_markdown(payload) + "\n")
        print(render_markdown(payload))
        print(json.dumps(payload["platform_orders"], indent=1))
        print(f"\nwrote {out}")
        return 0

    if args.local_smoke:
        cfgs = configs(ladders={1: (8, 16, 32)}, time_pair={1: (8, 16)}, final_time=0.25)
        records = [run_config(c) for c in cfgs]
        payload = assemble(records)
        print(render_markdown(payload))
        return 0

    all_configs = configs()
    if args.config_index is None:
        records = [run_config(c) for c in all_configs]
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "runs_all.json").write_text(json.dumps({"runs": records}, indent=2) + "\n")
        print(f"{len(records)} runs -> {ARTIFACTS / 'runs_all.json'}")
        return 0

    if args.num_configs < 1:
        raise SystemExit("--num-configs must be >= 1")
    # ``{i}`` is substituted by the scheduler, which may count from 0 or from 1;
    # the residue is a surjection onto 0..n-1 either way, so both conventions
    # partition the configuration list exactly once.
    residue = args.config_index % args.num_configs
    mine = [c for i, c in enumerate(all_configs) if i % args.num_configs == residue]
    records = [run_config(c) for c in mine]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"shard_{residue:02d}.json"
    path.write_text(json.dumps({"runs": records}, indent=2) + "\n")
    print(f"{len(records)} runs -> {path}")
    print("SHARD_JSON " + json.dumps({"shard": residue, "runs": records}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())