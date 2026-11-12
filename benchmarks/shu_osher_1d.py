"""Shu--Osher shock / entropy-wave interaction: reference solution and scheme comparison.

Physical setup (Shu & Osher, *J. Comput. Phys.* **83** (1989) 32--78, test case 3):
a Mach-3 right-moving shock at ``x = -4`` in a gamma = 1.4 ideal gas sweeps over a
smooth sinusoidal density wave and generates high-frequency entropy oscillations
downstream of the shock.  The case is the standard stress test for *shock limiters*:
a too-diffusive limiter erases the entropy waves, an inactive one leaves Gibbs
oscillations around the shock.  It has no closed-form solution, so the verification
anchor is a self-converged fine-grid solution (see ``docs/shu_osher_verification.md``).

Only cell averages (the ``k = 0`` modal coefficient, i.e. the conservative projection)
are compared, so coarse/fine grids can be related by a plain volume average.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_1d import solve_periodic_euler_dg_1d

# The case definition (domain, diaphragm position, final time, left state and
# initial data) lives in the library, in ``tensordg.cases.shock_entropy``; this
# benchmark re-exports it so that existing call sites keep working.  One owner
# per constant means a benchmark cannot silently drift from the shipped case.
from tensordg.cases.shock_entropy import (  # noqa: F401  (re-export)
    DIAPHRAGM,
    DOMAIN,
    FINAL_TIME,
    LEFT_STATE as LEFT,
    initial_state,
)
#: high-frequency window that only exists if the scheme resolves the entropy waves
WAVE_WINDOW = (0.0, 4.2)


def solve_case(ncells, degree=2, limiter="weno", cfl=0.12, tvb_constant=1.0,
               limiter_stage="every", final_time=FINAL_TIME, device="cpu"):
    """Run one configuration and return ``(x_centers, cell_averages, metadata)``."""
    # Single owner: the case (data, anchors, solver call) lives in the library at
    # ``tensordg.cases.shock_entropy``; this driver only folds the modal state into
    # the ``(ncells, 3)`` cell-average array the comparison code expects.
    from tensordg.cases.shock_entropy import solve_dg_case

    out = solve_dg_case(ncells, degree=degree, limiter=limiter, cfl=cfl,
                        final_time=final_time, device=device,
                        tvb_constant=tvb_constant, limiter_stage=limiter_stage)
    state = np.stack((out["rho"], out["rho_u"], out["rho_e"]), axis=-1)
    return out["x"], state, out["meta"]


def coarsen(values: np.ndarray, factor: int) -> np.ndarray:
    """Volume-average ``values`` (uniform grid) by an integer ``factor``."""
    if factor < 1 or values.shape[0] % factor:
        raise ValueError("refinement factor must be a positive divisor of the array length")
    return values.reshape(values.shape[0] // factor, factor).mean(axis=1)


def density_error(scheme_rho, reference_rho, factor):
    """Relative L1 / L2 error of a coarse density profile against a fine reference."""
    ref = coarsen(reference_rho, factor)
    diff = scheme_rho - ref
    l1 = float(np.sum(np.abs(diff)) / np.sum(np.abs(ref)))
    l2 = float(np.sqrt(np.sum(diff ** 2) / np.sum(ref ** 2)))
    return {"relative_l1": l1, "relative_l1_percent": 100.0 * l1, "relative_l2": l2}


def shock_position(x, rho, level=2.0):
    """Interpolated x of the *rightmost* upward crossing of ``rho = level``.

    The left state (rho = 3.857143) also lies above ``level``, so the scan must
    start at the far end of the domain: the transmitted shock is the last place
    where the density falls through ``level`` from behind.
    """
    above = np.nonzero(rho >= level)[0]
    if above.size == 0:
        return float("nan")
    if above[-1] == rho.size - 1:
        return float(x[-1])
    i = int(above[-1]) + 1
    x0, x1 = x[i - 1], x[i]
    r0, r1 = rho[i - 1], rho[i]
    if r1 == r0:
        return float(x1)
    return float(x0 + (level - r0) * (x1 - x0) / (r1 - r0))


def wave_metrics(x, rho, window=WAVE_WINDOW):
    """Resolution / admissibility proxies for the post-shock entropy waves."""
    mask = (x >= window[0]) & (x <= window[1])
    profile = rho[mask]
    if profile.size < 3:
        return {"n_local_extrema": 0, "wave_amplitude": 0.0, "total_variation": 0.0}
    slope = np.diff(profile)
    sign = np.sign(slope)
    sign[sign == 0] = 1
    flips = int(np.count_nonzero(np.diff(sign) != 0))
    return {
        "n_local_extrema": flips,
        "wave_amplitude": float(profile.max() - profile.min()),
        "total_variation": float(np.sum(np.abs(slope))),
    }


def evaluate(np_profile, ref_rho, ref_cells, x, meta):
    """Bundle every metric for one configuration."""
    factor = ref_cells // np_profile.shape[0]
    out = {
        "ncells": int(np_profile.shape[0]),
        "degree": int(meta["degree"]),
        "limiter": meta["limiter"],
        "limiter_stage": meta["limiter_stage"],
        "steps": int(meta["steps"]),
        "wall_time_s": float(meta["wall_time_s"]),
        "density_min": float(np_profile[:, 0].min()),
        "density_max": float(np_profile[:, 0].max()),
        "shock_position": shock_position(x, np_profile[:, 0]),
    }
    out.update(density_error(np_profile[:, 0], ref_rho, factor))
    out.update(wave_metrics(x, np_profile[:, 0]))
    return out


# ---------------------------------------------------------------- study driver
CONFIGS = [
    # (ncells, degree, limiter); every ncells must divide the reference grid
    (150, 1, "none"), (150, 1, "tvb"), (150, 1, "weno"),
    (150, 2, "none"), (150, 2, "tvb"), (150, 2, "weno"),
    (150, 3, "none"), (150, 3, "weno"),
]


def _cached(cache, ncells, degree, limiter, cfl, cache_dir=None):
    """Solve one configuration, reusing an on-disk result when present.

    A full sweep costs minutes of single-core work, so every configuration is
    memoised to ``<output_dir>/_cache``; an interrupted study resumes instead of
    recomputing.  The persisted arrays are the cell averages actually compared,
    so a resumed run is bit-identical to an uninterrupted one.
    """
    key = (int(ncells), int(degree), str(limiter), float(cfl))
    if key in cache:
        return cache[key]
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"case_n{ncells}_p{degree}_{limiter}_cfl{cfl}.npz"
        if path.exists():
            blob = np.load(path, allow_pickle=False)
            meta = json.loads(str(blob["meta"]))
            cache[key] = (blob["x"], blob["avg"], meta)
            return cache[key]
    result = solve_case(ncells, degree, limiter, cfl=cfl)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=result[0], avg=result[1], meta=json.dumps(result[2]))
    cache[key] = result
    return result


def run_study(ref_cells=300, ref_degree=3, ref_limiter="weno", configs=None,
              output_dir="shu_osher_results", cfl=0.4,
              cross_reference=(600, 1, "none"), verbose=True):
    """Reference solution + configuration sweep + machine-readable artifact.

    Two independent trust anchors for the reference are recorded:

    * ``reference_self_convergence`` -- the same scheme one level coarser must
      agree with the reference to within the 3 % gate;
    * ``reference_cross_scheme`` -- a different discretisation family (P1, no
      limiter) on a *finer* grid, coarsened onto the reference grid.

    Agreement of both anchors bounds how much of any tabulated error can be
    blamed on the reference itself.
    """
    configs = CONFIGS if configs is None else configs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    cache_dir = out / "_cache"

    x_ref, avg_ref, meta_ref = _cached(cache, ref_cells, ref_degree, ref_limiter, cfl, cache_dir)
    rho_ref = avg_ref[:, 0]

    x_half, avg_half, meta_half = _cached(cache, ref_cells // 2, ref_degree, ref_limiter, cfl,
                                          cache_dir)
    self_conv = density_error(avg_half[:, 0], rho_ref, 2)

    cross = None
    if cross_reference:
        c_cells, c_degree, c_limiter = cross_reference
        x_c, avg_c, meta_c = _cached(cache, c_cells, c_degree, c_limiter, cfl, cache_dir)
        cross = density_error(rho_ref, avg_c[:, 0], c_cells // ref_cells)
        cross.update({"ncells": int(c_cells), "degree": int(c_degree), "limiter": c_limiter,
                      "wall_time_s": float(meta_c["wall_time_s"]),
                      "steps": int(meta_c["steps"])})

    rows = []
    profiles = {"x_ref": x_ref, "rho_ref": rho_ref}
    for ncells, degree, limiter in configs:
        if ref_cells % ncells:
            if verbose:
                print(f"skip n={ncells}: does not divide the reference grid", flush=True)
            continue
        x, avg, meta = _cached(cache, ncells, degree, limiter, cfl, cache_dir)
        row = evaluate(avg, rho_ref, ref_cells, x, meta)
        row["finite"] = bool(np.isfinite(avg).all())
        row["reference_cells"] = int(ref_cells)
        rows.append(row)
        profiles[f"rho_{ncells}_{degree}_{limiter}"] = avg[:, 0]
        profiles[f"x_{ncells}_{degree}_{limiter}"] = x
        if verbose:
            print(f"n={ncells:4d} P{degree} {limiter:4s} "
                  f"L1={100 * row['relative_l1']:7.4f}%  L2={100 * row['relative_l2']:7.4f}%  "
                  f"extrema={row['n_local_extrema']:3d}  rho=[{row['density_min']:.3f},"
                  f"{row['density_max']:.3f}]  x_sh={row['shock_position']:.3f}  "
                  f"steps={row['steps']:5d}  {row['wall_time_s']:6.1f}s", flush=True)

    gate = {
        "reference_self_convergence_percent": 100.0 * self_conv["relative_l1"],
        "reference_self_convergence_below_3pct": bool(100.0 * self_conv["relative_l1"] < 3.0),
        "reference_cross_scheme_percent": (
            None if cross is None else 100.0 * cross["relative_l1"]
        ),
        "all_runs_finite": all(r["finite"] for r in rows),
        "best_relative_l1_percent": min((r["relative_l1_percent"] for r in rows),
                                        default=float("nan")),
        "best_relative_l1_under_3pct": bool(
            min((r["relative_l1_percent"] for r in rows), default=float("inf")) < 3.0
        ),
    }
    summary = {
        "case": "shu_osher",
        "gamma": GAMMA,
        "left_state": LEFT,
        "diaphragm": DIAPHRAGM,
        "domain": DOMAIN,
        "final_time": FINAL_TIME,
        "cfl": cfl,
        "reference": {"ncells": ref_cells, "degree": ref_degree, "limiter": ref_limiter,
                      "steps": int(meta_ref["steps"]), "wall_time_s": meta_ref["wall_time_s"]},
        "reference_self_convergence": self_conv,
        "reference_cross_scheme": cross,
        "gate": gate,
        "runs": rows,
    }
    (out / "shu_osher_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(out / "shu_osher_profiles.npz", **profiles)
    if verbose:
        print(json.dumps(gate, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Shu--Osher shock/entropy-wave comparison")
    parser.add_argument("--reference-cells", type=int, default=300)
    parser.add_argument("--reference-degree", type=int, default=3)
    parser.add_argument("--reference-limiter", default="weno")
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--output-dir", default="shu_osher_results")
    args = parser.parse_args()
    run_study(ref_cells=args.reference_cells, ref_degree=args.reference_degree,
              ref_limiter=args.reference_limiter, output_dir=args.output_dir, cfl=args.cfl)


if __name__ == "__main__":
    main()