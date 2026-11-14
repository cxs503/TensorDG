"""CASE-DG-SOD-SHOCK: Sod (1978) shock tube -- density **peak position** and **plateau**.

Anchor
------
CASE-DG-SOD-SHOCK (算例池).  Unlike the earlier whole-field L2 Sod verification
(``benchmarks/sod_dg_benchmark.py``), this case gates *feature* quantities of the
density profile, which whole-field norms hide:

* ``rho_peak``     -- the star-region density maximum, i.e. the left-star value
                     ``rho_L*`` that the rarefaction fan asymptotes to;
* ``x_peak``       -- **where** that density peak starts, i.e. the rarefaction
                     *tail* position ``x_tail = x_d + (u* - c*_L) t``;
* ``rho_plateau``  -- the constant-density shelf between contact and shock,
                     ``rho_R*`` (the classic "plateau" behind the shock);
* ``plateau_width``-- the width of that shelf, ``(S - u*) t``;
* plus the two shelf boundaries ``x_contact`` and ``x_shock``.

Reference
---------
Sod, G. A. (1978), "A survey of several finite difference methods for systems of
nonlinear hyperbolic conservation laws", *J. Comput. Phys.* **27**(1), 1-31,
doi:10.1016/0021-9991(78)90023-2 -- initial data ``(rho,u,p)_L=(1,0,1)``,
``(rho,u,p)_R=(0.125,0,0.1)``, ``gamma=1.4``, diaphragm at ``x_d=0.5``, snapshot
``t=0.2`` on ``[0,1]``.  The exact closed-form wave structure used here is the
Sod solution as reproduced by Toro (2009, *Riemann Solvers and Numerical Methods
for Fluid Dynamics*, 3rd ed., Sec. 4.3 and Table 4.1); the repository's
independent exact Riemann solver :mod:`benchmarks.riemann_exact` computes the
star state and the wave speeds.  Published star state
``p*=0.303130, u*=0.927453, rho_L*=0.426319, rho_R*=0.265574, S=1.752156`` is
asserted against that solver in ``tests/test_case_sod_peak_plateau.py`` (zero
hand-copying of the target numbers into the pass/fail path: every reference
value below is *computed*, the literature constants are only a cross-check).

Tolerance
---------
``< 4 %`` relative on **every** gated feature; positions are measured relative
to the domain length ``L = 1`` (and additionally reported relative to the
feature's own width, which is the sharper normalisation).

Usage
-----
    python -m benchmarks.case_sod_peak_plateau --profile quick
    python -m benchmarks.case_sod_peak_plateau --profile full --outdir artifacts/case_sod_peak_plateau
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from benchmarks.riemann_exact import GAMMA, sample, star_state
from tensordg.curved_geometry import legendre_eval_matrix
from tensordg.euler_1d import solve_riemann
from tensordg.euler_dg_1d import solve_euler_dg_1d

LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.1)
DIAPHRAGM = 0.5
DOMAIN = (0.0, 1.0)
FINAL_TIME = 0.2
TOLERANCE = 0.04          # < 4 % on every gated feature (anchor criterion)
TRIM = 0.10               # trim 10 % of each window to stay clear of smearing
PEAK_EPS = 0.01           # "peak onset" level: 1 % below the measured peak
SUBSAMPLES = 8            # pointwise DG samples per cell for crossing detection


# ---------------------------------------------------------------------------
# exact (closed-form) features
# ---------------------------------------------------------------------------
def exact_features(final_time: float = FINAL_TIME, left=LEFT, right=RIGHT,
                   gamma: float = GAMMA, diaphragm: float = DIAPHRAGM) -> dict:
    """Closed-form Sod wave speeds, plateaus and their positions at ``final_time``."""
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    p_star, u_star = star_state(left, right, gamma)
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)

    # Sod: both branches are compressive enough that left is a fan, right a shock.
    if not (p_star < p_l and p_star > p_r):
        raise ValueError("not the Sod wave pattern (need fan/shock)")

    # left rarefaction -> star: isentropic, rho = rho_l (p*/p_l)^(1/gamma)
    rho_l_star = rho_l * (p_star / p_l) ** (1.0 / gamma)
    c_l_star = math.sqrt(gamma * p_star / rho_l_star)
    # right shock: Rankine-Hugoniot density ratio and shock speed
    ratio = p_star / p_r
    rho_r_star = rho_r * ((ratio + (gamma - 1.0) / (gamma + 1.0))
                          / ((gamma - 1.0) / (gamma + 1.0) * ratio + 1.0))
    shock_speed_rh = (rho_r_star * u_star - rho_r * u_r) / (rho_r_star - rho_r)
    shock_speed = u_r + a_r * math.sqrt(
        (gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma))

    x_head = diaphragm + (u_l - a_l) * final_time
    x_tail = diaphragm + (u_star - c_l_star) * final_time
    x_contact = diaphragm + u_star * final_time
    x_shock = diaphragm + shock_speed * final_time

    return {
        "p_star": p_star, "u_star": u_star,
        "rho_peak": rho_l_star, "rho_plateau": rho_r_star,
        "c_l_star": c_l_star, "a_l": a_l, "a_r": a_r,
        "shock_speed": shock_speed, "shock_speed_rankine_hugoniot": shock_speed_rh,
        "x_head": x_head, "x_peak": x_tail, "x_tail": x_tail,
        "x_contact": x_contact, "x_shock": x_shock,
        "peak_width": x_contact - x_tail,
        "plateau_width": x_shock - x_contact,
        "left_density": rho_l, "right_density": rho_r,
        "final_time": final_time, "gamma": gamma, "diaphragm": diaphragm,
    }


# ---------------------------------------------------------------------------
# numerical -> pointwise density
# ---------------------------------------------------------------------------
def pointwise_primitive(x_centers, coeff, degree: int, subsamples: int = SUBSAMPLES,
                        domain=DOMAIN, gamma: float = GAMMA):
    """Evaluate the modal DG solution inside every cell.

    ``coeff`` is ``(ncells, degree+1, 3)`` in the unnormalised-Legendre basis
    used by :mod:`tensordg.euler_dg_1d` (``coeff[:, k]`` multiplies ``P_k(xi)``,
    ``xi in [-1, 1]``).
    """
    coeff = np.asarray(coeff, dtype=np.float64)
    ncells = coeff.shape[0]
    lower, upper = domain
    dx = (upper - lower) / ncells
    centers = np.asarray(x_centers, dtype=np.float64).reshape(-1)
    xi = -1.0 + (2.0 * np.arange(subsamples) + 1.0) / subsamples
    L = legendre_eval_matrix(degree, xi)                      # (sub, degree+1)
    point = np.einsum("cdv,sd->csv", coeff, L)                # (ncells, sub, 3)
    x_fine = (centers[:, None] + 0.5 * dx * xi[None, :]).reshape(-1)
    rho = point[..., 0].reshape(-1)
    mom = point[..., 1].reshape(-1)
    energy = point[..., 2].reshape(-1)
    safe = np.where(np.abs(rho) > 1e-14, rho, np.nan)
    velocity = mom / safe
    pressure = (gamma - 1.0) * (energy - 0.5 * mom ** 2 / safe)
    return x_fine, {"density": rho, "velocity": velocity, "pressure": pressure}


def _crossing(x, y, start: int, level: float, rising: bool) -> float:
    """First level crossing of ``y`` scanning forward from ``start`` (linear interpolation)."""
    n = y.size
    for i in range(max(start, 1), n):
        hit = (y[i] >= level) if rising else (y[i] <= level)
        if hit:
            y0, y1 = y[i - 1], y[i]
            if y1 == y0:
                return float(x[i])
            frac = (level - y0) / (y1 - y0)
            frac = min(max(frac, 0.0), 1.0)
            return float(x[i - 1] + frac * (x[i] - x[i - 1]))
    return float("nan")


def _crossing_left(x, y, start: int, level: float) -> float:
    """First level crossing of ``y`` scanning **backwards** from ``start``."""
    for i in range(min(start, y.size - 1), 0, -1):
        if y[i - 1] >= level:
            y0, y1 = y[i - 1], y[i]
            if y1 == y0:
                return float(x[i - 1])
            frac = (level - y0) / (y1 - y0)
            frac = min(max(frac, 0.0), 1.0)
            return float(x[i - 1] + frac * (x[i] - x[i - 1]))
    return float("nan")


def _window_mask(x, lo, hi):
    return (x >= lo) & (x <= hi)


def measure_density_features(x_fine, rho, feat: dict, trim: float = TRIM,
                             peak_eps: float = PEAK_EPS) -> dict:
    """Extract peak value/position and plateau value/width from a density profile.

    Measurement definition (declared before looking at the results):

    * windows come from the *exact* wave structure (an a-priori definition, not
      a fit to the numerical data), each trimmed by ``trim`` of its width so
      that contact/shock smearing cannot leak into the value metrics;
    * thresholds are data-driven (``0.5 (rho_peak + rho_plateau)`` etc.);
    * the density **peak shelf** is the run between the rarefaction tail and the
      contact discontinuity; inside the fan the density is *higher* than the
      shelf value, so the shelf's left edge ``x_peak`` is located by walking
      **leftwards** from the middle of the shelf until the profile rises above
      ``(1 + peak_eps) * rho_peak`` (linear interpolation of that crossing).
    """
    x = np.asarray(x_fine, dtype=np.float64)
    y = np.asarray(rho, dtype=np.float64)

    # --- peak shelf (rarefaction tail .. contact) ---------------------------
    p_lo, p_hi = feat["x_tail"], feat["x_contact"]
    wp = p_hi - p_lo
    inner = _window_mask(x, p_lo + trim * wp, p_hi - trim * wp)
    rho_peak = float(np.max(y[inner]))
    i_shelf = int(np.argmin(np.abs(x - 0.5 * (p_lo + p_hi))))
    x_peak = _crossing_left(x, y, i_shelf, (1.0 + peak_eps) * rho_peak)

    # --- plateau shelf (contact .. shock) -----------------------------------
    q_lo, q_hi = feat["x_contact"], feat["x_shock"]
    wq = q_hi - q_lo
    inner_q = _window_mask(x, q_lo + trim * wq, q_hi - trim * wq)
    rho_plateau = float(np.mean(y[inner_q]))
    plateau_flatness = float(np.max(np.abs(y[inner_q] - rho_plateau)) / rho_plateau)

    # --- shelf boundaries ---------------------------------------------------
    level_contact = 0.5 * (rho_peak + rho_plateau)
    level_shock = 0.5 * (rho_plateau + feat["right_density"])
    i_mid = int(np.argmin(np.abs(x - 0.5 * (p_lo + p_hi))))
    x_contact = _crossing(x, y, i_mid, level_contact, rising=False)
    i_qmid = int(np.argmin(np.abs(x - 0.5 * (q_lo + q_hi))))
    x_shock = _crossing(x, y, i_qmid, level_shock, rising=False)

    star = _window_mask(x, p_lo, feat["x_shock"])
    return {
        "rho_peak": rho_peak,
        "x_peak": x_peak,
        "rho_plateau": rho_plateau,
        "plateau_flatness": plateau_flatness,
        "x_contact": x_contact,
        "x_shock": x_shock,
        "peak_width": x_contact - x_peak,
        "plateau_width": x_shock - x_contact,
        "rho_max_star": float(np.max(y[star])),
        "rho_peak_overshoot_pct": 100.0 * (float(np.max(y[star])) / rho_peak - 1.0),
        "level_contact": level_contact,
        "level_shock": level_shock,
    }


def compare_features(meas: dict, feat: dict, domain_length: float = 1.0) -> list[dict]:
    """ref-vs-got table with relative errors.

    Gated rows (``pass`` is True/False) normalise positions by the domain length
    ``L`` -- that is the ``< 4 %`` criterion of the anchor.  Supplementary rows
    (``pass is None``) re-normalise the same position error by the feature's own
    width, which is the sharper statement, and record the peak-shelf flatness.
    """
    rows = []
    spec = [
        ("rho_peak", "density peak value", "value", "relative", "rho_peak"),
        ("x_peak", "density peak onset x_tail", "position", "domain", "x_tail"),
        ("rho_plateau", "post-contact plateau value", "value", "relative", "rho_plateau"),
        ("x_contact", "contact position", "position", "domain", "x_contact"),
        ("x_shock", "shock position", "position", "domain", "x_shock"),
        ("plateau_width", "plateau width", "length", "relative", "plateau_width"),
        ("peak_width", "peak-shelf width", "length", "relative", "peak_width"),
    ]
    for key, label, kind, norm, ref_key in spec:
        ref, got = feat[ref_key], meas[key]
        err = abs(got - ref)
        rel = err / domain_length if norm == "domain" else err / abs(ref)
        rows.append({
            "metric": key, "label": label, "kind": kind, "ref": ref, "got": got,
            "abs_err": err, "rel_err_pct": 100.0 * rel,
            "pass": bool(rel <= TOLERANCE),
            "normalisation": "L" if norm == "domain" else "feature value",
        })
    for key, ref_key, width_key, label in (
        ("x_peak", "x_tail", "peak_width", "peak onset error / peak-shelf width"),
        ("x_contact", "x_contact", "plateau_width", "contact error / plateau width"),
        ("x_shock", "x_shock", "plateau_width", "shock error / plateau width"),
        ("plateau_width", "plateau_width", "plateau_width", "plateau width error / plateau width"),
    ):
        err = abs(meas[key] - feat[ref_key])
        rows.append({
            "metric": f"{key}_over_width", "label": label, "kind": "supplementary",
            "ref": feat[ref_key], "got": meas[key], "abs_err": err,
            "rel_err_pct": 100.0 * err / abs(feat[width_key]), "pass": None,
            "normalisation": f"own width ({width_key})",
        })
    rows.append({
        "metric": "plateau_flatness", "label": "max deviation inside plateau interior",
        "kind": "supplementary", "ref": 0.0, "got": meas["plateau_flatness"],
        "abs_err": meas["plateau_flatness"], "rel_err_pct": 100.0 * meas["plateau_flatness"],
        "pass": None, "normalisation": "plateau value",
    })
    rows.append({
        "metric": "rho_peak_overshoot", "label": "Gibbs overshoot above the peak value",
        "kind": "supplementary", "ref": 0.0, "got": meas["rho_peak_overshoot_pct"],
        "abs_err": meas["rho_peak_overshoot_pct"], "rel_err_pct": meas["rho_peak_overshoot_pct"],
        "pass": None, "normalisation": "peak value",
    })
    return rows


# ---------------------------------------------------------------------------
# single run
# ---------------------------------------------------------------------------
def run_single(degree: int, ncells: int, cfl: float = 0.2, limiter=None,
               final_time: float = FINAL_TIME, device: str = "cpu") -> dict:
    """One solver run (P0 = finite-volume limit, P1-P3 = modal DG) + feature extraction."""
    started = time.time()
    if degree == 0:
        # P0 has no modal DOF: use the repository's finite-volume P0 solver.
        x, fields, meta = solve_riemann(
            LEFT, RIGHT, ncells=ncells, final_time=final_time, diaphragm=DIAPHRAGM,
            domain=DOMAIN, cfl=cfl, flux="hllc", device=device,
        )
        x_np = x.detach().cpu().numpy().reshape(-1)
        prim = {k: v.detach().cpu().numpy() for k, v in fields.items()}
        x_fine = x_np                       # piecewise constant: cell centres are exact samples
    else:
        x, coeff, meta = solve_euler_dg_1d(
            LEFT, RIGHT, ncells=ncells, degree=degree, final_time=final_time,
            diaphragm=DIAPHRAGM, domain=DOMAIN, cfl=cfl, limiter=limiter, device=device,
        )
        x_np = x.detach().cpu().numpy().reshape(-1)
        coeff_np = coeff.detach().cpu().numpy()
        x_fine, prim = pointwise_primitive(x_np, coeff_np, degree)
    feat = exact_features(final_time)
    meas = measure_density_features(x_fine, prim["density"], feat)
    rows = compare_features(meas, feat)
    gated = [r for r in rows if r["pass"] is not None]
    return {
        "degree": degree, "ncells": ncells, "cfl": cfl,
        "limiter": limiter if limiter else "none",
        "steps": meta["steps"], "dt_mean": meta["dt_mean"],
        "wall_seconds": time.time() - started,
        "x": x_np, "x_fine": x_fine, "prim": prim,
        "features": feat, "measured": meas, "table": rows,
        "max_rel_err_pct": max(r["rel_err_pct"] for r in gated),
        "pass": all(r["pass"] for r in gated),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, bool):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
PROFILES = {
    "smoke": {
        "ladder": [(0, 100, 0.4), (1, 50, 0.3), (2, 50, 0.3), (3, 50, 0.3)],
        "cfl_sweep": [(0, 100, 0.2), (0, 100, 0.4)],
        "limiter_sweep": [(2, 50, 0.3, None), (2, 50, 0.3, "tvb")],
    },
    "quick": {
        "ladder": [(0, 400, 0.4), (1, 200, 0.2), (2, 100, 0.15), (3, 50, 0.1)],
        "cfl_sweep": [(0, 400, 0.1), (0, 400, 0.2), (0, 400, 0.4)],
        "limiter_sweep": [(2, 100, 0.15, None), (2, 100, 0.15, "tvb"), (2, 100, 0.15, "weno")],
    },
    "full": {
        "ladder": [(0, 200, 0.4), (0, 400, 0.4), (0, 800, 0.4), (0, 1600, 0.4),
                   (1, 100, 0.2), (1, 200, 0.2), (1, 400, 0.2),
                   (2, 50, 0.15), (2, 100, 0.15), (2, 200, 0.15),
                   (3, 25, 0.1), (3, 50, 0.1), (3, 100, 0.1)],
        "cfl_sweep": [(0, 400, 0.1), (0, 400, 0.2), (0, 400, 0.4), (0, 400, 0.5),
                      (2, 100, 0.05), (2, 100, 0.15), (2, 100, 0.3)],
        "limiter_sweep": [(2, 100, 0.15, None), (2, 100, 0.15, "tvb"), (2, 100, 0.15, "weno"),
                          (3, 100, 0.1, None), (3, 100, 0.1, "tvb"), (3, 100, 0.1, "weno")],
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    parser.add_argument("--outdir", default="artifacts/case_sod_peak_plateau")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = PROFILES[args.profile]
    feat = exact_features()

    ladder = [run_single(d, n, c) for d, n, c in cfg["ladder"]]
    cfl_runs = [run_single(d, n, c) for d, n, c in cfg["cfl_sweep"]]
    lim_runs = [run_single(d, n, c, limiter=l) for d, n, c, l in cfg["limiter_sweep"]]

    # ---- console summary ---------------------------------------------------
    print("CASE-DG-SOD-SHOCK : Sod 1978 density peak position / plateau")
    print(f"exact: rho_peak={feat['rho_peak']:.6f} at x_tail={feat['x_tail']:.6f}; "
          f"rho_plateau={feat['rho_plateau']:.6f} width={feat['plateau_width']:.6f} "
          f"(x in [{feat['x_contact']:.6f}, {feat['x_shock']:.6f}])")
    print(f"tolerance {TOLERANCE * 100:.1f}% on every gated feature\n")
    header = f"{'cfg':>18} {'rho_peak%':>9} {'x_peak%':>8} {'rho_plat%':>9} {'x_c%':>7} {'x_s%':>7} {'W%':>7} {'pass':>5}"
    print(header)
    for r in ladder + lim_runs + cfl_runs:
        by = {row["metric"]: row for row in r["table"]}
        tag = f"P{r['degree']} N={r['ncells']} cfl={r['cfl']:.2f} {str(r['limiter'])[:4]}"
        print(f"{tag:>18} "
              + " ".join(f"{by[k]['rel_err_pct']:8.3f}" for k in
                         ("rho_peak", "x_peak", "rho_plateau", "x_contact", "x_shock", "plateau_width"))
              + f" {'PASS' if r['pass'] else 'FAIL':>5}")

    # ---- artifacts ---------------------------------------------------------
    def strip(run):
        out = {k: v for k, v in run.items() if k not in ("x", "x_fine", "prim")}
        return out

    metrics = {
        "case": "CASE-DG-SOD-SHOCK",
        "profile": args.profile,
        "reference": {
            "source": "Sod 1978, J. Comput. Phys. 27(1):1-31, doi:10.1016/0021-9991(78)90023-2; "
                      "Toro 2009 3rd ed. Sec. 4.3 / Table 4.1",
            "comment": "closed-form Sod wave structure at t=0.2, computed by benchmarks.riemann_exact",
            "features": feat,
        },
        "tolerance_pct": TOLERANCE * 100.0,
        "measurement_definition": {
            "trim": TRIM, "peak_eps": PEAK_EPS, "subsamples": SUBSAMPLES,
            "windows": "from the exact wave structure (a priori), trimmed by 10%",
            "thresholds": "data-driven mid-levels",
        },
        "ladder": [strip(r) for r in ladder],
        "cfl_sensitivity": [strip(r) for r in cfl_runs],
        "limiter_sensitivity": [strip(r) for r in lim_runs],
        "all_pass": all(r["pass"] for r in ladder),
    }
    (outdir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))

    with (outdir / "features.csv").open("w") as handle:
        handle.write("config,degree,ncells,cfl,limiter,metric,ref,got,rel_err_pct,pass\n")
        for r in ladder + cfl_runs + lim_runs:
            cfg_name = f"P{r['degree']}_N{r['ncells']}_cfl{r['cfl']}_{r['limiter']}"
            for row in r["table"]:
                handle.write(f"{cfg_name},{r['degree']},{r['ncells']},{r['cfl']},{r['limiter']},"
                             f"{row['metric']},{row['ref']:.10g},{row['got']:.10g},"
                             f"{row['rel_err_pct']:.6f},{row['pass']}\n")

    reference = ladder[0]
    with (outdir / "profiles.csv").open("w") as handle:
        handle.write("x,density_num,density_exact\n")
        ex = sample(reference["x_fine"], FINAL_TIME, LEFT, RIGHT, DIAPHRAGM, GAMMA)
        for xi, rn, re in zip(reference["x_fine"], reference["prim"]["density"], ex["density"]):
            handle.write(f"{xi:.10g},{rn:.10g},{re:.10g}\n")

    if not args.no_figures:
        _figures(outdir, ladder, cfl_runs, lim_runs, feat)

    print(f"\nartifacts -> {outdir}  (all_pass={metrics['all_pass']})")
    return 0 if metrics["all_pass"] else 1


def _figures(outdir, ladder, cfl_runs, lim_runs, feat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- figure 1: profile + feature markers --------------------------------
    ref = ladder[0]
    ex = sample(ref["x_fine"], FINAL_TIME, LEFT, RIGHT, DIAPHRAGM, GAMMA)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(ref["x_fine"], ref["prim"]["density"], lw=1.3, label=f"TensorDG P{ref['degree']} N={ref['ncells']}")
    ax.plot(ref["x_fine"], ex["density"], "--", lw=1.0, color="k", label="Sod exact (t=0.2)")
    ax.axhline(feat["rho_peak"], color="C3", lw=0.6, ls=":")
    ax.axhline(feat["rho_plateau"], color="C2", lw=0.6, ls=":")
    for key, lab in (("x_peak", "peak onset"), ("x_contact", "contact"), ("x_shock", "shock")):
        ax.axvline(feat[key], color="0.5", lw=0.6, ls=":")
        ax.annotate(lab, (feat[key], 0.06), rotation=90, fontsize=7, ha="right")
    ax.set_xlim(0.2, 0.95)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("Sod shock tube: density peak shelf and post-contact plateau")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "density_peak_plateau.png", dpi=140)
    plt.close(fig)

    # --- figure 2: feature-error convergence --------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for degree in sorted({r["degree"] for r in ladder}):
        runs = sorted([r for r in ladder if r["degree"] == degree], key=lambda r: r["ncells"])
        for metric, marker in (("rho_peak", "o"), ("x_peak", "s"), ("rho_plateau", "^"),
                               ("plateau_width", "d")):
            pts = [(r["ncells"], {row["metric"]: row for row in r["table"]}[metric]["rel_err_pct"])
                   for r in runs]
            ax.plot([p[0] for p in pts], [max(p[1], 1e-6) for p in pts],
                    marker=marker, ls="--", lw=0.8, ms=4,
                    label=f"P{degree} {metric}")
    ax.axhline(TOLERANCE * 100, color="C3", lw=1.0, label="4 % tolerance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells N")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Feature-error convergence")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "feature_convergence.png", dpi=140)
    plt.close(fig)

    # --- figure 3: sensitivity -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, runs, xkey, title in (
        (axes[0], cfl_runs, "cfl", "CFL sensitivity"),
        (axes[1], lim_runs, "limiter", "limiter sensitivity"),
    ):
        labels = [f"P{r['degree']}N{r['ncells']}\n{r[xkey]}" for r in runs]
        width = 0.2
        for j, metric in enumerate(("rho_peak", "x_peak", "rho_plateau", "plateau_width")):
            vals = [{row["metric"]: row for row in r["table"]}[metric]["rel_err_pct"] for r in runs]
            ax.bar(np.arange(len(runs)) + j * width, vals, width=width, label=metric)
        ax.axhline(TOLERANCE * 100, color="C3", lw=1.0)
        ax.set_xticks(np.arange(len(runs)) + 1.5 * width)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("relative error [%]")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "sensitivity.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())