"""Shu--Osher shock / entropy-wave interaction: DG case-pool driver.

Case identity
-------------
Shu & Osher (1988), *Efficient implementation of essentially non-oscillatory
shock-capturing schemes*, J. Comput. Phys. **77** 439--471 (the standard
shock/entropy-wave interaction example, and its restatement as test case 3 of
Shu & Osher (1989), J. Comput. Phys. **83** 32--78): a Mach-3 shock at ``x=-4``
in a gamma=1.4 ideal gas runs into a stationary sinusoidal density wave and,
downstream of the shock, converts it into high-frequency entropy waves.

Why this driver exists
----------------------
The existing verification (:mod:`benchmarks.shu_osher_1d`, ``docs/shu_osher_verification.md``)
fixed a 300-cell P3 solution as *the* reference and compared limiter policies at
150 cells.  A benchmark that feeds a case pool needs something else: an
*error-versus-refinement* curve against a reference whose own error is shown to
be small, plus analytic anchors.  This driver provides

* a resolution ladder ``N = 200, 400, 800`` for ``P1/P2/P3`` (WENO limiter),
* a converged reference from an **independent discretisation**
  (``benchmarks.fv_weno5_1d``: FV-WENO5/HLLC/SSP-RK3) at two refinements,
  so its own grid-convergence can be exhibited,
* a CFL-sensitivity sweep (parameter sensitivity axis 2),
* three *analytic* anchors that need no literature table: post-shock plateau,
  contact-discontinuity position and transmitted-shock position from the
  Rankine--Hugoniot relations, plus an exact mass budget.

Any number this file emits is computed, never transcribed.

Artifacts (all inside the worktree)::

    runs/shu_osher_case_pool/raw/<key>.npz     one file per configuration
    runs/shu_osher_case_pool/metrics.json      merged comparison tables
    runs/shu_osher_case_pool/profiles.npz      profiles for plotting

Usage::

    python -m benchmarks.shu_osher_case_pool ladder --ncells 200 --degree 1
    python -m benchmarks.shu_osher_case_pool reference --kind fv --ncells 12800
    python -m benchmarks.shu_osher_case_pool merge
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

import sys

if __package__ in (None, ""):  # allow ``python benchmarks/shu_osher_case_pool.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensordg.cases.shock_entropy as _case  # noqa: E402
from benchmarks.shu_osher_1d import (  # noqa: E402
    DIAPHRAGM, DOMAIN, FINAL_TIME, LEFT, initial_state, solve_case, shock_position,
)

OUT = Path("runs/shu_osher_case_pool")
RAW = OUT / "raw"

#: resolution ladder: every ``ncells`` divides the finest grid used below (3200)
LADDER = [(200, 1), (400, 1), (800, 1), (200, 2), (400, 2), (800, 2),
          (200, 3), (400, 3), (800, 3)]
#: reference pair (independent FV-WENO5), used to show the reference converged.
#: The coarse leg of the pair is also the coarsest grid every ladder entry divides.
REFERENCE = (1600, 3200)
#: CFL sensitivity at fixed grid/order (800 cells P2 also divides the reference grid)
CFL_SWEEP = (0.1, 0.2, 0.4)
CFL_CELLS, CFL_DEGREE = 400, 2
#: DG-side self-convergence leg, taken from the loftiest ladder pair (P3)
DG_PAIR = [(800, 3), (400, 3)]


# --------------------------------------------------------------- analytic part
def rankine_hugoniot_anchor() -> dict:
    """Wave skeleton of the case, computed by the library case module.

    Delegation (instead of a second copy of the Rankine--Hugoniot algebra) keeps
    the benchmark and the solver library from drifting apart: the shipped case
    definition in :mod:`tensordg.cases.shock_entropy` is the single owner.
    """
    return _case.rankine_hugoniot_anchor()


def exact_mass_budget() -> dict:
    """Exact global mass budget at the final time (delegated to the case module)."""
    return _case.exact_mass_budget()


# ------------------------------------------------------------------ DG ladder
def _key(tag: str, **kw) -> str:
    parts = [tag] + [f"{k}{v}" for k, v in sorted(kw.items())]
    return "_".join(parts).replace(".", "p")


def run_dg(ncells: int, degree: int, limiter: str = "weno", cfl: float = 0.4,
           tag: str = "dg") -> Path:
    """Solve one DG configuration and persist ``x, rho, rho_u, rho_E`` cell averages.

    ``benchmarks.shu_osher_1d.solve_case`` returns the *full conserved state*
    cell averages, shape ``(ncells, 3)`` (density, momentum, total energy); the
    density is column 0.  Keeping the three columns separate matters: an earlier
    revision of this driver stored the whole ``(ncells, 3)`` array under ``rho``,
    so the reported ``rho`` extrema were really extrema over ``(rho, rho u,
    rho E)`` (``rho E`` reaches ~39 on the post-shock plateau) and every profile
    comparison was against the wrong quantity.
    """
    key = _key(tag, n=ncells, p=degree, lim=limiter, cfl=cfl)
    path = RAW / f"{key}.npz"
    if path.exists():
        print(f"[skip] {key} (cached)", flush=True)
        return path
    x, state, meta = solve_case(ncells, degree, limiter, cfl=cfl)
    state = np.asarray(state, dtype=float)
    if state.ndim != 2 or state.shape[1] != 3:
        raise ValueError(f"expected cell-averaged conserved state (ncells, 3), got {state.shape}")
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, rho=state[:, 0], rho_u=state[:, 1], rho_e=state[:, 2],
             meta=json.dumps(meta))
    rho = state[:, 0]
    print(f"[done] {key}: steps={meta['steps']} wall={meta['wall_time_s']:.1f}s "
          f"rho=[{rho.min():.4f},{rho.max():.4f}]", flush=True)
    return path


def run_fv(ncells: int, cfl: float = 0.3, tag: str = "fv") -> Path:
    """Solve one independent FV-WENO5 configuration (reference-solver leg)."""
    from benchmarks.fv_weno5_1d import solve_fv_weno5

    key = _key(tag, n=ncells, cfl=cfl)
    path = RAW / f"{key}.npz"
    if path.exists():
        print(f"[skip] {key} (cached)", flush=True)
        return path
    x, u, meta = solve_fv_weno5(initial_state, ncells=ncells, cfl=cfl, final_time=FINAL_TIME,
                                domain=DOMAIN)
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, rho=u[:, 0], rho_u=u[:, 1], rho_e=u[:, 2], meta=json.dumps(meta))
    print(f"[done] {key}: steps={meta['steps']} wall={meta['wall_time_s']:.1f}s", flush=True)
    return path


def state_column_regression(path: Path) -> dict:
    """Quantify the state-column mix-up that once corrupted this driver.

    An earlier revision of :func:`run_dg` persisted the *whole* conserved state
    ``(ncells, 3) = (rho, rho u, rho E)`` under the name ``rho``, so the reported
    "density peak" was really the largest of the three components (``rho E``
    reaches ~40 on the post-shock plateau, i.e. ~10x the density).  Given one
    (correct) raw artifact, this reproduces what the old convention would have
    reported: the extrema, and the *same* analytic anchors (quiet-zone plateau,
    shock position) recomputed on the per-cell maximum over the three columns.

    Keeping it a plain function (rather than a block inside :func:`merge`) is
    what makes the defect testable without the full ladder: any raw artifact
    suffices, and the exhibit cannot silently lose its input path again.
    """
    path = Path(path)
    blob = np.load(path, allow_pickle=False)
    for column in ("rho", "rho_u", "rho_e"):
        if column not in blob.files:
            raise ValueError(f"{path} does not store the conserved state columns")
    rho, rho_u, rho_e = blob["rho"], blob["rho_u"], blob["rho_e"]
    state = np.stack((rho, rho_u, rho_e))
    whole_max, whole_min = float(state.max()), float(state.min())
    good = profile_metrics(blob["x"], rho)
    bad = profile_metrics(blob["x"], state.max(axis=0))
    analytic = rankine_hugoniot_anchor()
    anchor = abs(analytic["shock_position_analytic"])

    def shock_err(position: float) -> float:
        return 100.0 * abs(position - analytic["shock_position_analytic"]) / anchor

    return {
        "artifact": str(path),
        "ncells": int(rho.size),
        "degree": int(json.loads(str(blob["meta"]))["degree"]) if "meta" in blob.files else -1,
        "cfl": float(json.loads(str(blob["meta"]))["cfl"]) if "meta" in blob.files else float("nan"),
        "density_max_correct": float(rho.max()),
        "density_max_if_whole_state_used": whole_max,
        "overstatement_factor": float(whole_max / rho.max()),
        "density_min_correct": float(rho.min()),
        "density_min_if_whole_state_used": whole_min,
        "negative_cells_if_whole_state_used": int((state < 0.0).any(axis=0).sum()),
        "column_extrema": {
            "rho": [float(rho.min()), float(rho.max())],
            "rho_u": [float(rho_u.min()), float(rho_u.max())],
            "rho_e": [float(rho_e.min()), float(rho_e.max())],
        },
        "plateau_dev_percent_correct": good["plateau_rho_dev_percent"],
        "plateau_dev_percent_if_whole_state_used": bad["plateau_rho_dev_percent"],
        "shock_position_correct": good["shock_position"],
        "shock_position_if_whole_state_used": bad["shock_position"],
        "shock_position_rel_error_percent_correct": shock_err(good["shock_position"]),
        "shock_position_rel_error_percent_if_whole_state_used":
            shock_err(bad["shock_position"]),
        "columns": ["rho", "rho_u", "rho_e"],
        "note": "rho E is ~40 on the post-shock plateau, so using it as density inflates "
                "the reported peak ~10x; the quiet-zone plateau anchor and the shock "
                "position then both miss by orders of magnitude, i.e. the corruption is "
                "caught by the analytic anchors, not by the positivity check",
    }


# --------------------------------------------------------------------- metrics
def coarsen(values: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1 or values.shape[0] % factor:
        raise ValueError(f"factor {factor} does not divide {values.shape[0]}")
    return values.reshape(values.shape[0] // factor, factor).mean(axis=1)


def relative_errors(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """Relative L1/L2/Linf of two profiles on the *same* grid.

    The shape check is deliberate: the first revision of ``merge`` coarsened the
    coarse DG profile instead of the fine reference (the factor is
    ``reference_cells // ncells``, which belongs on the *reference*), which either
    raised deep inside ``coarsen`` or silently broadcast the wrong arrays.
    """
    if candidate.shape != reference.shape:
        raise ValueError(f"profile shapes differ: {candidate.shape} vs {reference.shape}")
    diff = candidate - reference
    l1 = float(np.sum(np.abs(diff)) / np.sum(np.abs(reference)))
    l2 = float(np.sqrt(np.sum(diff ** 2) / np.sum(reference ** 2)))
    return {"rel_l1": l1, "rel_l1_percent": 100.0 * l1, "rel_l2_percent": 100.0 * l2,
            "linf_percent": 100.0 * float(np.max(np.abs(diff))) / float(np.max(np.abs(reference)))}


def preshock_sine_error(x: np.ndarray, rho: np.ndarray, x_start: float = 2.6) -> float:
    """Relative L1 error against the *exact* unshocked state on ``x >= x_start``.

    Downstream of the transmitted shock nothing has been touched yet, so the
    density there is still the initial sine ``1 + 0.2 sin(5x)`` exactly: this is
    a genuine closed-form anchor, not a reference solution.
    """
    mask = x >= x_start
    exact = 1.0 + 0.2 * np.sin(5.0 * x[mask])
    return 100.0 * float(np.sum(np.abs(rho[mask] - exact)) / np.sum(np.abs(exact)))


def profile_metrics(x: np.ndarray, rho: np.ndarray, plateau_end: float = -3.0) -> dict:
    """Analytic-anchor residuals for one density profile.

    ``plateau_end = -3.0`` keeps the window strictly inside the quiet zone
    (``x <= -2.75``, see :func:`rankine_hugoniot_anchor`), where the continuous
    solution is the *constant* left state to machine precision.  The material
    interface between the initial left gas and the shocked gas sits at
    ``x = 0.733``; inside ``[-2.75, 0.733]`` the incident sine wave has already
    been converted into entropy/acoustic waves, so it is not a valid anchor.
    """
    plateau = rho[(x >= -5.0) & (x <= plateau_end)]
    return {
        "ncells": int(rho.size),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "plateau_rho_mean": float(plateau.mean()),
        "plateau_rho_dev_percent": 100.0 * float(np.max(np.abs(plateau - LEFT[0]))) / LEFT[0],
        "shock_position": shock_position(x, rho),
        "preshock_sine_rel_l1_percent": preshock_sine_error(x, rho),
        "mass": float(np.sum(rho) * (DOMAIN[1] - DOMAIN[0]) / rho.size),
    }


def load_raw(path: Path) -> dict:
    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["meta"])) if "meta" in blob.files else {}
    return {"x": blob["x"], "rho": blob["rho"], "meta": meta}


def merge(reference_cells: int = 25600) -> dict:
    """Compare the DG ladder against the converged FV-WENO5 reference."""
    analytic = rankine_hugoniot_anchor()
    budget = exact_mass_budget()
    ref_path = RAW / f"{_key('fv', n=reference_cells, cfl=0.3)}.npz"
    if not ref_path.exists():
        raise SystemExit(f"missing reference {ref_path}; run the 'reference' stage first")
    ref = load_raw(ref_path)
    x_ref, rho_ref = ref["x"], ref["rho"]
    out: dict = {
        "case": "shu_osher_shock_entropy_wave",
        "reference": {
            "source": "independent FV-WENO5/HLLC/SSP-RK3 (benchmarks/fv_weno5_1d.py)",
            "ncells": reference_cells, "meta": ref["meta"],
        },
        "configuration": {
            "gamma": 1.4, "domain": list(DOMAIN), "diaphragm": DIAPHRAGM,
            "left_state": list(LEFT), "right_density": "1 + 0.2 sin(5x)",
            "final_time": FINAL_TIME,
        },
        "analytic": analytic,
        "mass_budget": budget,
        "reference_anchor": profile_metrics(x_ref, rho_ref),
    }
    # ---- reference self-convergence (independent solver, two refinements) ----
    avail = sorted(int(p.stem.split("_n")[1])
                   for p in RAW.glob("fv_cfl0p3_n*.npz"))
    avail = [n for n in avail if n < reference_cells and reference_cells % n == 0]
    if avail:
        coarse_n = avail[-1]  # finest available coarser leg
        h = load_raw(RAW / f"{_key('fv', n=coarse_n, cfl=0.3)}.npz")
        out["reference_self_convergence"] = relative_errors(
            h["rho"], coarsen(rho_ref, reference_cells // coarse_n))
        out["reference_self_convergence"]["coarse_ncells"] = int(coarse_n)
    # ---- DG ladder ----------------------------------------------------------
    rows = []
    for ncells, degree in LADDER:
        p = RAW / f"{_key('dg', n=ncells, p=degree, lim='weno', cfl=0.4)}.npz"
        if not p.exists():
            continue
        d = load_raw(p)
        factor = reference_cells // ncells
        row = {"ncells": ncells, "degree": degree, "limiter": "weno",
               "cfl": 0.4, "steps": d["meta"]["steps"],
               "wall_time_s": d["meta"]["wall_time_s"]}
        row.update(relative_errors(d["rho"], coarsen(rho_ref, factor)))
        row.update({k: v for k, v in profile_metrics(d["x"], d["rho"]).items()
                    if k in ("rho_min", "rho_max", "plateau_rho_dev_percent",
                             "shock_position", "preshock_sine_rel_l1_percent")})
        row["shock_position_rel_error_percent"] = 100.0 * abs(
            row["shock_position"] - analytic["shock_position_analytic"]) / abs(
            analytic["shock_position_analytic"])
        row["mass_abs_error_percent_of_initial"] = 100.0 * abs(
            profile_metrics(d["x"], d["rho"])["mass"] - budget["mass_final_predicted"]) / budget[
                "mass_initial"]
        row["finite"] = bool(np.isfinite(d["rho"]).all() and d["rho"].min() > 0.0)
        row["passes_5pct"] = bool(row["rel_l1_percent"] < 5.0)
        rows.append(row)
    out["ladder"] = sorted(rows, key=lambda r: (r["degree"], r["ncells"]))
    # ---- DG self-convergence leg (same scheme, two grids) -------------------
    dg_pairs = []
    fine_n, fine_p = DG_PAIR[0]
    coarse_n, _ = DG_PAIR[1]
    fp = RAW / f"{_key('dg', n=fine_n, p=fine_p, lim='weno', cfl=0.4)}.npz"
    cp = RAW / f"{_key('dg', n=coarse_n, p=fine_p, lim='weno', cfl=0.4)}.npz"
    if fp.exists() and cp.exists():
        f, c = load_raw(fp), load_raw(cp)
        dg_pairs.append({"fine_ncells": fine_n, "coarse_ncells": coarse_n, "degree": fine_p,
                         "self_convergence": relative_errors(coarsen(f["rho"], 2), c["rho"]),
                         "vs_fv_reference": relative_errors(
                             f["rho"], coarsen(rho_ref, reference_cells // fine_n))})
    out["dg_self_convergence"] = dg_pairs
    # ---- CFL sensitivity ----------------------------------------------------
    cfl_rows = []
    for cfl in CFL_SWEEP:
        p = RAW / f"{_key('dg', n=CFL_CELLS, p=CFL_DEGREE, lim='weno', cfl=cfl)}.npz"
        if not p.exists():
            continue
        d = load_raw(p)
        row = {"cfl": cfl, "ncells": CFL_CELLS, "degree": CFL_DEGREE,
               "steps": d["meta"]["steps"], "wall_time_s": d["meta"]["wall_time_s"]}
        row.update(relative_errors(d["rho"], coarsen(rho_ref, reference_cells // CFL_CELLS)))
        cfl_rows.append(row)
    out["cfl_sweep"] = sorted(cfl_rows, key=lambda r: r["cfl"])
    if len(cfl_rows) >= 2:
        spread = max(r["rel_l1_percent"] for r in cfl_rows) - min(
            r["rel_l1_percent"] for r in cfl_rows)
        out["cfl_sweep_spread_percent_points"] = float(spread)
    # ---- regression exhibit: the state-column mix-up that broke this driver ----
    # ``run_dg`` used to persist the whole conserved state (ncells, 3) under the
    # name ``rho``; recomputing the extrema and the analytic anchors on that
    # (wrong) quantity makes the defect measurable instead of anecdotal.
    probe_n, probe_p = LADDER[0]
    probe = RAW / f"{_key('dg', n=probe_n, p=probe_p, lim='weno', cfl=0.4)}.npz"
    if probe.exists():
        out["state_column_regression"] = state_column_regression(probe)
    # ---- profiles for plotting ---------------------------------------------
    prof = {"x_ref": x_ref, "rho_ref": rho_ref}
    for ncells, degree in LADDER:
        p = RAW / f"{_key('dg', n=ncells, p=degree, lim='weno', cfl=0.4)}.npz"
        if p.exists():
            d = load_raw(p)
            prof[f"x_n{ncells}_p{degree}"] = d["x"]
            prof[f"rho_n{ncells}_p{degree}"] = d["rho"]
    if avail:
        h = load_raw(RAW / f"{_key('fv', n=avail[-1], cfl=0.3)}.npz")
        prof["x_fv_half"], prof["rho_fv_half"] = h["x"], h["rho"]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "profiles.npz", **prof)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[merge] wrote {OUT/'metrics.json'} ({len(rows)} ladder rows)", flush=True)
    return out


# ------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("ladder", help="one DG ladder configuration")
    p.add_argument("--ncells", type=int, required=True)
    p.add_argument("--degree", type=int, required=True)
    p.add_argument("--limiter", default="weno")
    p.add_argument("--cfl", type=float, default=0.4)

    p = sub.add_parser("all-ladder", help="every DG ladder configuration in this process")
    p.add_argument("--reverse", action="store_true", help="coarse-first (cheap-first) order")

    p = sub.add_parser("reference", help="independent FV-WENO5 reference at one refinement")
    p.add_argument("--ncells", type=int, default=REFERENCE[0])
    p.add_argument("--cfl", type=float, default=0.3)

    p = sub.add_parser("merge", help="merge raw artifacts into metrics.json")
    p.add_argument("--reference-cells", type=int, default=REFERENCE[1])
    args = ap.parse_args(argv)

    if args.stage == "ladder":
        run_dg(args.ncells, args.degree, args.limiter, args.cfl)
    elif args.stage == "all-ladder":
        configs = list(LADDER)
        if args.reverse:
            configs = configs[::-1]
        for ncells, degree in configs:
            run_dg(ncells, degree)
    elif args.stage == "reference":
        run_fv(args.ncells, args.cfl)
    elif args.stage == "merge":
        merge(args.reference_cells)


if __name__ == "__main__":
    main()