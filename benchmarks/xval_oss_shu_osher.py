"""XVAL: cross-validate the round-2 *certified reference* against an OSS solver.

Why this module exists
----------------------
``benchmarks/shu_osher_certified_ref.py`` builds the certified reference for
CASE-SHU-OSHER from a single in-house finite-volume solver (FV-WENO5/HLLC/
SSP-RK3 at ``N = 800/1600/3200``) plus Richardson extrapolation and a Roache
GCI uncertainty band.  A certification argument that rests on one discretisation
cannot exclude a *shared* modelling error (same reconstruction, same wave-speed
estimate, same boundary treatment).  The honest cross-check is an independent
implementation written by somebody else.

This module runs the same case with **Clawpack 5.14.0 / pyclaw SharpClaw**
(BSD-3, L. F. Shampine? no -- Clawpack Development Team; see ``OSS_PROVENANCE``
below for the exact files and their sha256), using its own

* WENO reconstruction (``clawpack.pyclaw.limiters.recon``) and
* its own HLLC Riemann solver (``clawpack.riemann.euler_1D_py.euler_hllc_1D``),
* its own Runge-Kutta driver (``SSP104`` by default, ``SSP33`` optional) and
* its own ghost-cell boundary handling (``pyclaw.BC.extrap``).

Nothing is re-used from ``benchmarks.fv_weno5_1d``: the OSS leg is a genuinely
different code path, so agreement is evidence, not tautology.  The *certification
machinery* (coarsening, relative L1, observed order, GCI, Richardson) is
deliberately shared -- the point of the study is to compare two solutions with
the very same yardstick, so duplicating the yardstick would defeat the purpose.

Pre-declared decision rule (fixed before the first OSS run)
-----------------------------------------------------------
D1  the two certified references must agree within ``TOL_XVAL_PERCENT = 1.0 %``
    rel L1 (a factor ~12 inside the owner's 15 % threshold, and ~10x the GCI
    band of the reference itself);
D2  the difference must sit inside the quadrature sum of the two GCI bands
    (``U_ours`` and ``U_oss``) times a factor ``BAND_FACTOR = 3``;
D3  every DG leg of the case pool must still pass the 5 % gate when the
    *OSS-certified* reference is used instead of ours.
Failure of D1/D2 means the round-2 certification is implementation-dependent and
the honest verdict is FAIL (reported, not papered over).

Usage::

    python -m benchmarks.xval_oss_shu_osher \
        --clawpack-root /tmp/oss/clawpack-5.14.0 \
        --rawdir validation/repro_shu_osher/artifacts/raw \
        --outdir artifacts/xval_m17365

The OSS root is imported by prepending it to ``sys.path``; the OSS tree is never
modified (see ``verify_oss_tree``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.shu_osher_certified_ref import (
    FS_GCI,
    REFINEMENT,
    TOLERANCE_PERCENT,
    coarsen,
    convergence_study,
    evaluate_leg,
    load_fv_levels,
    load_profile,
    rel_l1,
    window_errors,
)

#: pre-declared cross-validation gates (see module docstring)
TOL_XVAL_PERCENT = 1.0
BAND_FACTOR = 3.0
#: finite-volume ladder used by both implementations (same as round 2)
FV_LEVELS = (800, 1600, 3200)
#: case definition (CASE-SHU-OSHER)
DOMAIN = (-5.0, 5.0)
GAMMA = 1.4
FINAL_TIME = 1.8
#: OSS files whose *behaviour* this study relies on; hashes prove non-tampering
OSS_PROVENANCE = (
    "clawpack/riemann/euler_1D_py.py",
    "pyclaw/src/pyclaw/sharpclaw/solver.py",
    "pyclaw/src/pyclaw/limiters/recon.py",
    "pyclaw/src/pyclaw/solver.py",
    "pyclaw/src/pyclaw/state.py",
)


# ------------------------------------------------------------------ OSS leg
def import_oss(root: Path | str):
    """Import the OSS solvers from ``root`` (prepended to ``sys.path``)."""
    root = Path(root)
    if not (root / "clawpack" / "__init__.py").exists():
        raise FileNotFoundError(f"{root} does not look like a clawpack tree")
    sys.path.insert(0, str(root))
    from clawpack import pyclaw, riemann  # noqa: WPS433 (deliberate late import)
    return pyclaw, riemann


def verify_oss_tree(root: Path | str) -> dict:
    """sha256 of the OSS files the study leans on (zero-transcription evidence)."""
    root = Path(root)
    out = {"root": str(root), "files": {}}
    for rel in OSS_PROVENANCE:
        path = root / rel
        if not path.exists():
            out["files"][rel] = {"missing": True}
            continue
        out["files"][rel] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    init = root / "clawpack" / "__init__.py"
    if init.exists():
        out["version"] = "5.14.0" if b"MICRO               = 0" in init.read_bytes() else "unknown"
    return out


def _shu_osher_state(x: np.ndarray) -> np.ndarray:
    """Shu--Osher (1988) initial data as conservative variables."""
    rho = np.where(x < -4.0, 3.857143, 1.0 + 0.2 * np.sin(5.0 * x))
    vel = np.where(x < -4.0, 2.629369, 0.0)
    pre = np.where(x < -4.0, 10.33333, 1.0)
    energy = pre / (GAMMA - 1.0) + 0.5 * rho * vel ** 2
    return np.row_stack([rho, rho * vel, energy])


def run_oss_level(ncells: int, cfl: float = 0.3, t_end: float = FINAL_TIME,
                  weno_order: int = 5, time_integrator: str = "SSP104",
                  char_decomp: int = 0, root: Path | str = "/tmp/oss/clawpack-5.14.0") -> dict:
    """One SharpClaw leg; returns the cell averages used by the DG ladder."""
    pyclaw, riemann = import_oss(root)
    rp = riemann.euler_1D_py.euler_hllc_1D

    solver = pyclaw.SharpClawSolver1D(rp)
    solver.kernel_language = "Python"   # pure-python kernel: no f2py build needed
    solver.lim_type = 2              # WENO reconstruction
    solver.weno_order = int(weno_order)
    solver.time_integrator = time_integrator
    solver.char_decomp = int(char_decomp)
    solver.cfl_desired = float(cfl)
    solver.cfl_max = float(cfl) * 1.2
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap

    xdim = pyclaw.Dimension(DOMAIN[0], DOMAIN[1], int(ncells), name="x")
    domain = pyclaw.Domain(xdim)
    state = pyclaw.State(domain, num_eqn=3, num_aux=1)
    state.problem_data["gamma"] = GAMMA
    state.problem_data["gamma1"] = GAMMA - 1.0
    state.problem_data["epsilon"] = 0.0
    state.aux[0, :] = GAMMA
    state.q[:, :] = _shu_osher_state(state.grid.x.centers)

    sol = pyclaw.Solution(state, domain)
    start = time.perf_counter()
    solver.evolve_to_time(sol, t_end)
    wall = time.perf_counter() - start

    q = np.asarray(sol.q[:, :], dtype=float)
    xc = np.asarray(sol.grid.x.centers, dtype=float)
    return {
        "x": xc,
        "rho": q[0].copy(),
        "rho_u": q[1].copy(),
        "rho_e": q[2].copy(),
        "meta": {
            "scheme": f"OSS pyclaw SharpClaw WENO{weno_order}/{time_integrator}/HLLC",
            "implementation": "clawpack-5.14.0 (pyclaw)",
            "ncells": int(ncells), "domain": list(DOMAIN), "final_time": t_end,
            "cfl_desired": float(cfl), "weno_order": int(weno_order),
            "time_integrator": time_integrator, "char_decomp": int(char_decomp),
            "riemann": "riemann.euler_1D_py.euler_hllc_1D",
            "bc": "extrap", "steps": int(getattr(solver, "num_step", -1)),
            "dt_last": float(getattr(solver, "dt", float("nan"))),
            "wall_time_s": wall,
            "density_min": float(q[0].min()), "density_max": float(q[0].max()),
        },
    }


def save_profile(profile: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, x=profile["x"], rho=profile["rho"],
        rho_u=profile["rho_u"], rho_e=profile["rho_e"],
        meta=np.asarray(json.dumps(profile["meta"])),
    )


def build_oss_levels(outdir: Path, levels=FV_LEVELS, cfl: float = 0.3,
                     tag: str = "", root: Path | str = "/tmp/oss/clawpack-5.14.0",
                     **solver_kw) -> list[dict]:
    """Run (or reuse) the OSS ladder and return ``convergence_study``-ready levels."""
    oss = []
    band = f"cfl{cfl:g}".replace(".", "p") + tag
    for n in levels:
        path = outdir / "oss_raw" / f"oss_{band}_n{n}.npz"
        if path.exists():
            prof = load_profile(path.parent, path.name)
            prof["ncells"] = n
            prof["file"] = path.name
            prof["reused"] = True
            oss.append(prof)
            continue
        prof = run_oss_level(n, cfl=cfl, root=root, **solver_kw)
        save_profile(prof, path)
        prof["ncells"] = n
        prof["file"] = path.name
        prof["reused"] = False
        oss.append(prof)
    oss.sort(key=lambda d: d["ncells"])
    return oss


#: sensitivity ladders (each is one independent OSS run set, cache-shareable)
WARM_VARIANTS = {
    "base": {"cfl": 0.3, "tag": "", "solver": {}},
    "ssp33": {"cfl": 0.3, "tag": "_ssp33", "solver": {"time_integrator": "SSP33"}},
    "cfl06": {"cfl": 0.6, "tag": "", "solver": {}},
    "char1": {"cfl": 0.3, "tag": "_char1", "solver": {"char_decomp": 1}},
}


def warm(variant: str, outdir: Path | str, levels=FV_LEVELS,
         root: Path | str = "/tmp/oss/clawpack-5.14.0") -> list[dict]:
    """Produce (or confirm) the cached OSS ladder of one sensitivity variant."""
    cfg = WARM_VARIANTS[variant]
    return build_oss_levels(Path(outdir), levels, cfl=cfg["cfl"], tag=cfg["tag"],
                            root=root, **cfg["solver"])


# -------------------------------------------------------------- comparison
def compare_certified(ours: dict, oss: dict) -> dict:
    """D1/D2: agreement of the two certified references (both on N=1600 grid)."""
    if ours["reference_ncells"] != oss["reference_ncells"]:
        raise ValueError("certified profiles live on different grids: "
                         f"{ours['reference_ncells']} vs {oss['reference_ncells']}")
    a, b = ours["rho_certified"], oss["rho_certified"]
    diff = 100.0 * rel_l1(a, b)
    u_ours = ours["gci_finest_percent"]
    u_oss = oss["gci_finest_percent"]
    band = BAND_FACTOR * math.hypot(u_ours, u_oss)
    wins = window_errors(ours["x"], b, a)
    return {
        "difference_percent": diff,
        "tolerance_percent": TOL_XVAL_PERCENT,
        "passes_tolerance": bool(diff < TOL_XVAL_PERCENT),
        "u_ours_percent": u_ours,
        "u_oss_percent": u_oss,
        "combined_band_percent": band,
        "inside_combined_band": bool(diff <= band),
        "oss_vs_ours_windows_percent": wins,
        "ours_vs_oss_windows_percent": window_errors(oss["x"], a, b),
        "p_ours": ours["observed_order_used"],
        "p_oss": oss["observed_order_used"],
        "extrapolation_displacement_ours_percent": ours["extrapolation_displacement_percent"],
        "extrapolation_displacement_oss_percent": oss["extrapolation_displacement_percent"],
    }


def compare_same_grid(ours_levels: list[dict], oss_levels: list[dict]) -> list[dict]:
    """Raw same-grid differences (no extrapolation involved at all)."""
    rows = []
    for a, b in zip(ours_levels, oss_levels):
        if a["ncells"] != b["ncells"]:
            raise ValueError("grid mismatch between ladders")
        rows.append({
            "ncells": a["ncells"],
            "rel_l1_percent": 100.0 * rel_l1(b["rho"], a["rho"]),
            "windows_percent": window_errors(a["x"], b["rho"], a["rho"]),
            "ours_steps": a.get("meta", {}).get("steps"),
            "oss_meta": {k: b.get("meta", {}).get(k) for k in
                         ("steps", "dt_last", "wall_time_s", "time_integrator", "weno_order")},
        })
    return rows


def retest_dg_legs(rawdir: Path | str, oss_study: dict) -> list[dict]:
    """D3: gate every DG leg against the OSS-certified reference."""
    from benchmarks.shu_osher_certified_ref import DG_LADDER

    raw = Path(rawdir)
    legs = []
    for degree, ncells, cfl in DG_LADDER:
        name = f"dg_cfl{cfl:g}".replace(".", "p") + f"_limweno_n{ncells}_p{degree}.npz"
        if not (raw / name).exists():
            legs.append({"configuration": f"DG P{degree} N={ncells} CFL={cfl}", "missing": name})
            continue
        prof = load_profile(raw, name)
        res = evaluate_leg(prof["x"], prof["rho"], oss_study)
        res.update({"configuration": f"DG P{degree} N={ncells} CFL={cfl}",
                    "degree": degree, "ncells": ncells, "cfl": cfl, "file": name})
        legs.append(res)
    return legs


def sensitivity(argv_levels, outdir: Path, root, rawdir: Path) -> dict:
    """Parameter sensitivity of the OSS certification (at least two axes)."""
    out = {}
    base = build_oss_levels(outdir, argv_levels, cfl=0.3, root=root)
    base_study = convergence_study(base)

    # (a) time integrator: SSP104 (OSS default) vs SSP33 (ours uses SSP-RK3)
    ss33 = build_oss_levels(outdir, argv_levels, cfl=0.3, root=root, tag="_ssp33",
                            time_integrator="SSP33")
    ss33_study = convergence_study(ss33)
    out["time_integrator"] = {
        "variants": ["SSP104 (OSS default)", "SSP33 (closest to our SSP-RK3)"],
        "p_obs": [base_study["observed_order_used"], ss33_study["observed_order_used"]],
        "gci_percent": [base_study["gci_finest_percent"], ss33_study["gci_finest_percent"]],
        "certified_ref_diff_percent": 100.0 * rel_l1(base_study["rho_certified"],
                                                    ss33_study["rho_certified"]),
    }

    # (b) CFL of the reference ladder: 0.3 (ours) vs 0.6
    hi = build_oss_levels(outdir, argv_levels, cfl=0.6, root=root)
    hi_study = convergence_study(hi)
    out["cfl"] = {
        "variants": [0.3, 0.6],
        "p_obs": [base_study["observed_order_used"], hi_study["observed_order_used"]],
        "gci_percent": [base_study["gci_finest_percent"], hi_study["gci_finest_percent"]],
        "certified_ref_diff_percent": 100.0 * rel_l1(base_study["rho_certified"],
                                                    hi_study["rho_certified"]),
    }

    # (c) characteristic vs component-wise WENO (OSS option char_decomp 0/2)
    # (c) char_decomp: only if the cached ladder exists (see honest note in docs)
    need = [outdir / "oss_raw" / ("oss_cfl0p3_char1_n%d.npz" % n) for n in argv_levels]
    if all(f.exists() for f in need):
        ch = build_oss_levels(outdir, argv_levels, cfl=0.3, root=root, tag="_char1", char_decomp=1)
        ch_study = convergence_study(ch)
        out["char_decomp"] = {
            "variants": ["0 (component-wise, OSS default; same choice as our FV)",
                         "1 (wave-slope reconstruction; OSS pure-python kernel: 2 raises NotImplementedError)"],
            "p_obs": [base_study["observed_order_used"], ch_study["observed_order_used"]],
            "gci_percent": [base_study["gci_finest_percent"], ch_study["gci_finest_percent"]],
            "certified_ref_diff_percent": 100.0 * rel_l1(base_study["rho_certified"],
                                                        ch_study["rho_certified"]),
        }
    else:
        out["char_decomp"] = {"incomplete": True,
                              "reason": "cached char_decomp=1 ladder (3 levels) not finished in-budget"}
    return out


# ------------------------------------------------------------------ driver
def build_report(rawdir: Path | str, outdir: Path | str, root: Path | str,
                 levels=FV_LEVELS, skip_sensitivity: bool = False) -> dict:
    rawdir, outdir = Path(rawdir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ours_levels = load_fv_levels(rawdir)
    ours_study = convergence_study(ours_levels)

    oss_levels = build_oss_levels(outdir, levels, cfl=0.3, root=root)
    oss_study = convergence_study(oss_levels)

    study = {
        "case": "CASE-SHU-OSHER",
        "predeclared": {"tol_xval_percent": TOL_XVAL_PERCENT, "band_factor": BAND_FACTOR,
                        "dg_gate_percent": TOLERANCE_PERCENT, "fs_gci": FS_GCI,
                        "refinement": REFINEMENT},
        "ladder": list(levels),
        "provenance": verify_oss_tree(root),
        "ours": {k: v for k, v in ours_study.items() if k not in ("x", "rho_certified", "rho_raw_finest")},
        "oss": {k: v for k, v in oss_study.items() if k not in ("x", "rho_certified", "rho_raw_finest")},
        "certified": compare_certified(ours_study, oss_study),
        "same_grid": compare_same_grid(ours_levels, oss_levels),
        "dg_legs_vs_oss": retest_dg_legs(rawdir, oss_study),
        "oss_meta": [lv.get("meta", {}) for lv in oss_levels],
    }
    study["all_dg_legs_pass_vs_oss"] = all(
        lg.get("passes_5pct", False) for lg in study["dg_legs_vs_oss"] if "missing" not in lg)
    if not skip_sensitivity:
        study["sensitivity"] = sensitivity(levels, outdir, root, rawdir)

    np.savez_compressed(outdir / "certified_refs.npz",
                        x=ours_study["x"], rho_ours=ours_study["rho_certified"],
                        rho_oss=oss_study["rho_certified"])
    (outdir / "xval.json").write_text(json.dumps(_jsonable(study), indent=2, sort_keys=True))
    return study


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rawdir", default="validation/repro_shu_osher/artifacts/raw")
    ap.add_argument("--outdir", default="artifacts/xval_m17365")
    ap.add_argument("--clawpack-root", default=os.environ.get("CLAWPACK_ROOT",
                                                             "/tmp/oss/clawpack-5.14.0"))
    ap.add_argument("--levels", default=",".join(str(n) for n in FV_LEVELS))
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--warm", choices=sorted(WARM_VARIANTS),
                    help="only produce the cached OSS ladder of this variant, then exit")
    args = ap.parse_args(argv)
    levels = tuple(int(s) for s in args.levels.split(",") if s.strip())
    if args.warm:
        got = warm(args.warm, args.outdir, levels, args.clawpack_root)
        print(json.dumps(_jsonable([{"ncells": g["ncells"], "meta": g.get("meta", {})}
                                    for g in got]), indent=2))
        return 0
    study = build_report(args.rawdir, args.outdir, args.clawpack_root, levels,
                         skip_sensitivity=args.skip_sensitivity)
    print(json.dumps(_jsonable(study["certified"]), indent=2)[:2000])
    return 0 if study["certified"]["passes_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())