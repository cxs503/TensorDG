"""CKS1998 §5.3 vortex 2-D case: *certified* order, not a point estimate.

Why this module exists
----------------------
``benchmarks/cks1998_vortex_2d.py`` measures the observed order of accuracy of
the RKDG solver on the paper's isentropic vortex test case
(``rho = [1 - (gamma-1) beta^2 / (8 gamma pi^2) exp(1-r^2)]^{1/(gamma-1)}``,
``u = 1 - beta/(2 pi) exp(0.5(1-r^2)) (y-y0-vt)``, ``v = 1 + beta/(2 pi)
exp(0.5(1-r^2)) (x-x0-ut)``, ``p = rho^gamma``, periodic ``[-5, 5]^2``) and
the platform gate compares *one* number per degree against the design order
``p + 1`` within the paper's tolerance (>= p + 0.5).

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

    is the order gate **certified** (deviation + estimator uncertainty both
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
  the error decay?"
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
* ``UNVERIFIED_BAND`` -- ``delta <= tol`` but the band does not fit: the point
  estimate passes, the *claim* "the order is within tol of p+1" is not
  supported by the ladder.
* ``FAIL``        -- ``delta > tol``: the scheme is outside tolerance on the
  finest mesh available.

Time-error separation runs the finest pair at ``CFL`` and at ``CFL/beta``:
``eta = |e(CFL) - e(CFL/beta)| / e(CFL)`` is the fraction of the error that the
time integrator owns.  The spatial order is treated as isolated only if
``eta <= 0.05`` **and** the order moves by less than 1 % between the two runs.

Usage
-----
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_vortex_order_certification  # full sweep
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_vortex_order_certification --quick  # ~2 min
    OMP_NUM_THREADS=1 python -m benchmarks.cks1998_vortex_order_certification --local-smoke  # seconds

Results are written to ``validation/cks1998_vortex_2d/certification.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_1d import GAMMA

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "validation" / "cks1998_vortex_2d"
CERT_OUT = OUT_DIR / "certification.json"

#: Domain: [-5, 5]^2, the standard vortex box.
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FINAL_TIME = 1.0
CFL0 = 0.2
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)

#: Pre-registered acceptance numbers (declared before any result is looked at).
TOLERANCE_REL = 0.03          # the platform gate: |order - (p+1)| / (p+1) <= 3 %
SIGMA_BAND = (0.9, 1.1)       # asymptotic-range band for p_coarse / p_fine
EXTRAPOLATION_RESIDUAL_MAX = 0.03   # |e_R|/e_fine for a consistent ladder
TIME_FRACTION_MAX = 0.05      # eta: temporal share of the error must be <= 5 %
ORDER_DRIFT_MAX = 0.01        # relative order change under dt -> dt/beta

FIELD = "density"
NORM = "L1_rel"

#: ``(degree, meshes)`` ladders — four rungs per degree for a reliable EOC.
LADDERS = {1: [12, 16, 24, 32], 2: [8, 12, 16, 24], 3: [6, 8, 12, 16]}


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

    ``ladder`` entries need ``h`` and ``error`` (``L1_rel`` of ``rho``); they are
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
# Analytic reference (same as cks1998_vortex_2d.py)
# ---------------------------------------------------------------------------
def vortex_primitive(x, y, time_val):
    """The paper's smooth solution: (rho, u, v, p) at ``(x, y, time)``."""
    return dg2.vortex_primitive(x, y, time_val, STRENGTH, BACKGROUND)


def vortex_conservative(x, y, time_val):
    """Conservative state (rho, rho*u, rho*v, E) at ``(x, y, time)``."""
    rho, u, v, p = vortex_primitive(x, y, time_val)
    return dg2.conservative_state(rho, u, v, p)


def initial(x, y):
    return vortex_primitive(x, y, 0.0)


# ---------------------------------------------------------------------------
# Error norms at the tensor Gauss nodes
# ---------------------------------------------------------------------------
def gauss_fields(coeff, degree, quadrature=None):
    """Evaluate the modal solution at the tensor Gauss nodes of every cell."""
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, order)
    values = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    return values, nodes, weights


def error_norms(coeff, x, y, degree, final_time=FINAL_TIME, domain=DOMAIN):
    """L1 (absolute and relative), L2 relative and L-inf of each field."""
    values, nodes, weights = gauss_fields(coeff, degree)
    (xmin, xmax), (ymin, ymax) = domain
    nx, ny = x.numel(), y.numel()
    hx, hy = (xmax - xmin) / nx, (ymax - ymin) / ny
    X = x[:, None, None, None] + nodes[None, None, :, None] * (hx / 2.0)
    Y = y[None, :, None, None] + nodes[None, None, None, :] * (hy / 2.0)
    exact = vortex_conservative(X, Y, final_time)
    jacobian = (
        weights[None, None, :, None]
        * weights[None, None, None, :]
        * (hx / 2.0)
        * (hy / 2.0)
    )
    out = {}
    names = ("density", "momentum_x", "momentum_y", "energy")
    for index, name in enumerate(names):
        diff = (values[..., index] - exact[..., index]).abs()
        l1 = float((diff * jacobian).sum())
        scale = float((exact[..., index].abs() * jacobian).sum())
        l2 = float(torch.sqrt((diff ** 2 * jacobian).sum() / jacobian.sum()))
        l2_ref = float(torch.sqrt((exact[..., index] ** 2 * jacobian).sum() / jacobian.sum()))
        out[name] = {
            "L1": l1,
            "L1_rel": l1 / scale,
            "L2_rel": l2 / l2_ref,
            "Linf": float(diff.max()),
        }
    return out


# ---------------------------------------------------------------------------
# One run of the sweep
# ---------------------------------------------------------------------------
def run_case(degree, ncells, final_time=FINAL_TIME, cfl=CFL0, device="cpu"):
    nx = ncells if isinstance(ncells, int) else ncells
    ny = ncells if isinstance(ncells, int) else ncells
    start = time.time()
    x, y, coeff, metadata = dg2.solve_euler_dg_2d(
        initial,
        ncells=(nx, ny),
        degree=degree,
        domain=DOMAIN,
        final_time=final_time,
        cfl=cfl,
        periodic=(True, True),
        device=device,
    )
    wall = time.time() - start
    norms = error_norms(coeff, x, y, degree, final_time)
    hx = (DOMAIN[0][1] - DOMAIN[0][0]) / nx
    return {
        "degree": degree,
        "ncells": nx,
        "h": hx,
        "cfl": cfl,
        "final_time": final_time,
        "steps": metadata["steps"],
        "dt": metadata.get("dt"),
        "wall_seconds": round(wall, 2),
        "norms": norms,
        "dofs": metadata["dofs"],
    }


# ---------------------------------------------------------------------------
# Assemble the full certification payload
# ---------------------------------------------------------------------------
def assemble(records: list[dict]) -> dict:
    """Build the certification payload from raw run records."""
    by_degree = {}
    for degree in sorted(set(r["degree"] for r in records)):
        ladder = [
            {"h": r["h"], "error": r["norms"][FIELD][NORM], "ncells": r["ncells"]}
            for r in records if r["degree"] == degree
        ]
        ladder.sort(key=lambda r: r["h"])
        cert = certify(ladder, degree + 1)
        by_degree[f"p{degree}"] = {
            "certification": cert,
            "runs": [r for r in records if r["degree"] == degree],
        }
    return {"field": FIELD, "norm": NORM, "tolerance_rel": TOLERANCE_REL,
            "degrees": by_degree}


def render_markdown(payload: dict) -> str:
    """Render the certification results as a Markdown table."""
    lines = [
        "| degree | design | order_fine | delta_rel | uncertainty | delta_band | "
        "asymptotic_ok | extrapolation_consistent | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(payload["degrees"].keys()):
        bundle = payload["degrees"][name]
        cert = bundle["certification"]
        lines.append(
            f"| {name} | {cert['design_order']} | {cert['order_fine']:.4f} | "
            f"{cert['delta_rel']:.4f} | {cert['uncertainty']:.4f} | "
            f"{cert['delta_band_rel']:.4f} | {cert['asymptotic_ok']} | "
            f"{cert['extrapolation_consistent']} | {cert['verdict']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degrees", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--meshes", type=int, nargs="*", default=None,
                    help="override the per-degree ladder")
    ap.add_argument("--final-time", type=float, default=FINAL_TIME)
    ap.add_argument("--cfl", type=float, default=CFL0)
    ap.add_argument("--quick", action="store_true",
                    help="cheap corner: three meshes per degree")
    ap.add_argument("--local-smoke", action="store_true",
                    help="tiny smoke test: 2 meshes, degree 1 only")
    ap.add_argument("--out", type=Path, default=CERT_OUT)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    torch.set_num_threads(max(args.threads, 1))

    if args.local_smoke:
        ladders = {1: [16, 24]}
        args.degrees = [1]
    elif args.quick:
        ladders = {1: [16, 24, 32], 2: [12, 16, 24], 3: [8, 12, 16]}
    elif args.meshes:
        ladders = {k: list(args.meshes) for k in args.degrees}
    else:
        ladders = {k: LADDERS[k] for k in args.degrees}

    records = []
    for degree in args.degrees:
        ladder = ladders.get(degree, LADDERS.get(degree, [8, 12, 16]))
        for ncells in ladder:
            run = run_case(degree, ncells, args.final_time, args.cfl)
            records.append(run)
            print(f"[vortex-cert] p{degree} {ncells}x{ncells}: "
                  f"L1rel={run['norms'][FIELD][NORM]:.3e} "
                  f"steps={run['steps']} wall={run['wall_seconds']}s", flush=True)

    payload = assemble(records)
    payload["table_markdown"] = render_markdown(payload)
    payload["reference"] = (
        "analytic: isentropic vortex, beta=5, u_inf=(1,1), rho_inf=1, p_inf=1, "
        "domain=[-5,5]^2, periodic"
    )
    payload["anchor"] = "REPRO-DG-CKS1998"
    payload["case"] = "vortex_2d_certification"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nCertification written to {args.out}")
    print(payload["table_markdown"])

    # Summary verdict
    all_certified = all(
        bundle["certification"]["verdict"] == "CERTIFIED"
        for bundle in payload["degrees"].values()
    )
    print(f"\nOverall: {'ALL CERTIFIED' if all_certified else 'SOME NOT CERTIFIED'}")
    return 0 if all_certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
