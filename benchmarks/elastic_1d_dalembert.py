"""Verify the 1-D DG elastodynamics solver against the d'Alembert exact solution.

Case anchor: ``CASE-DG-DALEMBERT-ELASTIC-1D`` (canonical definition lives in
:mod:`tensordg.cases.elastic_rod_1d`, this module only drives and scores it).

Scenarios (both closed form, see ``tensordg/elastic_1d.py`` and the case
module):

* ``pulse`` -- periodic bar, ``c = sqrt(E/rho) = 1``, released with a Gaussian
  velocity pulse ``v0 = exp(-((x-0.5)/0.05)^2)``.  The exact solution is the
  d'Alembert superposition of two counter-propagating half-amplitude pulses and
  contains no physical dispersion.  Headline metric: relative L2 error of
  ``sigma(x,t)`` against that closed form, gate ``< 2 %``.
* ``standing`` -- bar clamped at x=0 and free at x=L, released from the first
  longitudinal eigenmode: checks the reflection rules and the exact
  eigenfrequency ``f1 = c/(4L)``.
* ``reflection`` -- a single right-going stress pulse hitting the fixed end:
  the wall stress must double (``2 sigma_max``).

Besides the L2 error the script measures *physical* anchors (packet positions,
measured wave speed, probe arrival time, eigefrequency, wall amplification,
energy conservation) so that ``benchmark_ref_vs_got.json`` contains an explicit
reference-vs-computed row per quantity.

Usage::

    python -m benchmarks.elastic_1d_dalembert --output-dir validation/elastic_1d_dalembert
    python -m benchmarks.elastic_1d_dalembert --quick        # ~1 min smoke run

Writes into the output directory: ``elastic_1d_dalembert.json``,
``benchmark_ref_vs_got.json``, ``sigma_profile_ref_vs_got.csv``,
``elastic_1d_dalembert.png`` and ``run_manifest.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg import elastic_1d as solver
from tensordg.cases import elastic_rod_1d as case

PULSE_CENTER = case.PULSE_CENTER
PULSE_WIDTH = case.PULSE_WIDTH
REL_TOL_PCT = 2.0


# ---------------------------------------------------------------------------
# driver helpers
# ---------------------------------------------------------------------------
def pulse_case(meshes, degree, final_time, cfl, nsample):
    """Mesh sequence for the periodic d'Alembert pulse."""
    rows = []
    for n in meshes:
        centers, coeff, meta = case.solve_case("pulse", n, degree, final_time, cfl)
        x, v, s = solver.stress_profile(centers, coeff, nsample=nsample)
        v_ex, s_ex = case.exact_pulse(x, final_time)
        shift = meta["c"] * final_time
        window = 0.5 * shift
        centroid_left = _centroid(x, s, PULSE_CENTER - window, PULSE_CENTER)
        centroid_right = _centroid(x, s, PULSE_CENTER, PULSE_CENTER + window)
        energy = solver.elastic_energy(centers, coeff, rho=case.RHO,
                                       young=case.YOUNG)
        energy_exact = _exact_energy()
        rows.append({
            "cells": n,
            "steps": meta["steps"],
            "dt": meta["dt"],
            "rel_l2_v": solver.relative_error(x, v, v_ex),
            "rel_l2_sigma": solver.relative_error(x, s, s_ex),
            "linf_sigma": float((s - s_ex).abs().max()),
            "sigma_peak_num": float(s.abs().max()),
            "sigma_peak_exact": float(s_ex.abs().max()),
            "energy": energy,
            "energy_exact": energy_exact,
            "energy_drift": abs(energy - energy_exact) / energy_exact,
            "centroid_left": centroid_left,
            "centroid_right": centroid_right,
            "centroid_exact": {"left": PULSE_CENTER - shift,
                               "right": PULSE_CENTER + shift},
            "pulse_speed_error": max(
                abs(centroid_left - (PULSE_CENTER - shift)),
                abs(centroid_right - (PULSE_CENTER + shift)),
            ) / shift if shift else 0.0,
        })
    for a, b in zip(rows[:-1], rows[1:]):
        b["eoc_sigma"] = math.log(a["rel_l2_sigma"] / b["rel_l2_sigma"], 2.0)
        b["eoc_v"] = math.log(a["rel_l2_v"] / b["rel_l2_v"], 2.0)
    return rows


def _exact_energy() -> float:
    """Exact energy of the initial pulse: ``0.5 * rho * integral v0^2 dx``."""
    return 0.5 * case.RHO * PULSE_WIDTH * math.sqrt(math.pi / 2.0)


def _centroid(x, field, lo, hi):
    """First moment of ``|field|`` inside ``[lo, hi]`` (smooth pulse position)."""
    mask = (x >= lo) & (x <= hi)
    weight = field[mask].abs()
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("zero mass in window")
    return float((x[mask] * weight).sum()) / total


def standing_case(meshes, degree, final_time, cfl, nsample):
    """Mesh sequence for the clamped-free eigenmode release."""
    rows = []
    for n in meshes:
        centers, coeff, meta = case.solve_case("standing", n, degree,
                                               final_time, cfl)
        x, v, s = solver.stress_profile(centers, coeff, nsample=nsample)
        v_ex, s_ex, _u_ex = case.exact_standing(x, final_time)
        energy = solver.elastic_energy(centers, coeff, rho=case.RHO,
                                       young=case.YOUNG)
        k = math.pi / 2.0
        energy_exact = 0.25 * (k ** 2)
        left, right = case.boundary_values(coeff)
        rows.append({
            "cells": n,
            "steps": meta["steps"],
            "dt": meta["dt"],
            "rel_l2_v": solver.relative_error(x, v, v_ex),
            "rel_l2_sigma": solver.relative_error(x, s, s_ex),
            "energy": energy,
            "energy_exact": energy_exact,
            "energy_drift": abs(energy - energy_exact) / energy_exact,
            "sigma_wall": float(left[1]),
            "sigma_free_end": float(right[1]),
            "sigma_wall_exact": float(s_ex.max()),
        })
    for a, b in zip(rows[:-1], rows[1:]):
        b["eoc_sigma"] = math.log(a["rel_l2_sigma"] / b["rel_l2_sigma"], 2.0)
    return rows


# ---------------------------------------------------------------------------
# physical anchors: reference (closed form) vs computed
# ---------------------------------------------------------------------------
def _row(name, ref, got, criterion=REL_TOL_PCT, unit="-", source="", note=""):
    ref, got = float(ref), float(got)
    abs_err = abs(got - ref)
    rel = (abs_err / abs(ref) * 100.0) if ref != 0.0 else float("nan")
    return {
        "quantity": name, "unit": unit, "ref": ref, "got": got,
        "abs_error": abs_err,
        "rel_error_pct": rel if not math.isnan(rel) else None,
        "criterion_pct": criterion,
        "verdict": "PASS" if (not math.isnan(rel) and rel < criterion) else
                   ("PASS" if math.isnan(rel) and abs_err < 1e-6 else "FAIL"),
        "ref_source": source, "note": note,
    }


def measured_wave_speed(degree, cfl, ncells, t1=0.25, t2=0.30):
    """Packet displacement between two times -> measured propagation speed."""
    positions = {}
    for t in (t1, t2):
        x, _v, s, _m = case.sample("pulse", ncells, degree, t, cfl)
        positions[t] = _centroid(x, s, PULSE_CENTER + 0.05, 0.995)
    return (positions[t2] - positions[t1]) / (t2 - t1)


def measured_arrival_time(degree, cfl, ncells, x_probe=0.8, nt=11):
    """Time of the stress peak at a probe point, parabolic refinement."""
    t_exact = case.probe_arrival_time(x_probe)
    span = 0.04
    times = np.linspace(t_exact - span, t_exact + span, nt)
    values, xs = [], None
    for t in times:
        x, _v, s, _m = case.sample("pulse", ncells, degree, float(t), cfl)
        if xs is None:
            xs = x
            idx = int(torch.argmin((x - x_probe).abs()))
        values.append(abs(float(s[idx])))
    values = np.asarray(values)
    j = int(np.argmax(values))
    # parabola through the three samples around the discrete maximum
    if 0 < j < len(times) - 1:
        y0, y1, y2 = values[j - 1], values[j], values[j + 1]
        denom = (y0 - 2.0 * y1 + y2)
        shift = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
        t_got = float(times[j] + shift * (times[1] - times[0]))
    else:
        t_got = float(times[j])
    return t_got, t_exact


def measured_eigenfrequency(degree, cfl, ncells, lo=0.7, hi=1.4, iters=8):
    """Zero crossing of ``sigma`` at the clamped wall -> ``f1 = c/(4L)``.

    ``sigma(0,t) = E A k cos(omega t)`` vanishes first at ``t = 1/(4 f1)``; a
    bisection on the wall trace avoids any spectral post-processing.
    """
    def wall_sigma(t):
        _c, coeff, _m = case.solve_case("standing", ncells, degree, float(t), cfl)
        left, _right = case.boundary_values(coeff)
        return float(left[1])

    a, b = lo, hi
    fa = wall_sigma(a)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        fm = wall_sigma(mid)
        if fm * fa > 0.0:
            a, fa = mid, fm
        else:
            b = mid
    t_zero = 0.5 * (a + b)
    return 1.0 / (4.0 * t_zero), case.clamped_free_frequency(1)


def anchor_rows(degree, cfl, ncells=400, ncells_dynamic=200,
                ncells_freq=20):
    """All reference-vs-computed rows of the case."""
    rows = []

    # 1. headline: sigma profile against the closed form
    centers, coeff, meta = case.solve_case("pulse", ncells, degree,
                                           case.PULSE_TIME, cfl)
    x, v, s = solver.stress_profile(centers, coeff, nsample=2 * degree + 3)
    v_ex, s_ex = case.exact_pulse(x, case.PULSE_TIME)
    rel_l2 = solver.relative_error(x, s, s_ex) * 100.0
    rows.append({
        "quantity": "relative L2 error of sigma(x,t) vs d'Alembert "
                    f"(N={ncells}, P{degree}, t={case.PULSE_TIME})",
        "unit": "%", "ref": 0.0, "got": rel_l2, "abs_error": rel_l2,
        "rel_error_pct": rel_l2, "criterion_pct": REL_TOL_PCT,
        "verdict": "PASS" if rel_l2 < REL_TOL_PCT else "FAIL",
        "ref_source": "exact d'Alembert superposition (ref = 0 by definition)",
        "note": "the row value *is* the error in %, compare against the 2% gate"})

    # 2. peak stress of one travelling packet = rho c A / 2
    rows.append(_row("peak stress of one packet sigma_max",
                     case.peak_stress_amplitude(), float(s.abs().max()),
                     source="Graff 1975 Ch.1: two half-amplitude d'Alembert waves"))
    energy = solver.elastic_energy(centers, coeff, rho=case.RHO,
                                   young=case.YOUNG)

    # 3. packet positions = L/2 +- c t
    x_l, x_r = case.front_positions(case.PULSE_TIME)
    rows.append(_row("right-going packet position x_R",
                     x_r, _centroid(x, s, PULSE_CENTER, PULSE_CENTER + 0.45),
                     unit="L", source="x_R = L/2 + c t"))
    rows.append(_row("left-going packet position x_L",
                     x_l, _centroid(x, s, PULSE_CENTER - 0.45, PULSE_CENTER),
                     unit="L", source="x_L = L/2 - c t"))

    # 4. measured wave speed from packet displacement
    c_meas = measured_wave_speed(degree, cfl, ncells_dynamic)
    rows.append(_row("measured bar wave speed c", case.C,
                     c_meas, unit="L/T", source="c = sqrt(E/rho)"))

    # 5. probe arrival time = |x_p - L/2| / c
    t_got, t_ref = measured_arrival_time(degree, cfl, ncells_dynamic)
    rows.append(_row("pulse arrival time at probe x=0.8", t_ref, t_got,
                     unit="T", source="t = |x_p - L/2| / c",
                     note="parabolic refinement over 21 snapshots"))

    # 6. clamped-free eigenfrequency f1 = c/(4L)
    f_got, f_ref = measured_eigenfrequency(degree, cfl, ncells_freq)
    rows.append(_row("clamped-free first eigenfrequency f1", f_ref, f_got,
                     unit="1/T", source="f1 = c/(4L)",
                     note=f"bisection on the wall stress zero crossing, "
                          f"8 bisection steps at N={ncells_freq} P{degree} "
                          f"(bracket [0.7,1.4])"))

    # 7. fixed-end reflection: wall stress doubles
    # the reflected peak lives on one cell, so this anchor needs a resolved
    # pulse width: >= 6 cells per pulse width at P2
    refl_cells = max(100, min(ncells, 200))
    _c, coeff, _m = case.solve_case("reflection", refl_cells, degree,
                                    case.REFLECTION_TIME, cfl)
    _left, right = case.boundary_values(coeff)
    rows.append(_row("wall stress at the fixed end / pulse amplitude",
                     2.0 * case.REFLECTION_AMPLITUDE, float(right[1]),
                     source="Graff 1975 Ch.1 reflection rule at a fixed end",
                     note=f"pulse width resolves 6 cells at N={refl_cells} P{degree}"))

    # 8. discrete elastic energy conservation (periodic bar)
    e_ref = _exact_energy()
    rows.append(_row("elastic energy at t (periodic, exactly conserved)",
                     e_ref, energy, unit="-",
                     source="0.5 rho * integral v0^2 dx (d'Alembert splitting "
                            "conserves energy)"))

    return rows


def write_profile_csv(path: Path, ncells: int, degree: int, cfl: float):
    """Point-by-point ref-vs-got stress profile at the headline time."""
    x, v, s, _meta = case.sample("pulse", ncells, degree, case.PULSE_TIME, cfl)
    v_ex, s_ex = case.exact_pulse(x, case.PULSE_TIME)
    lines = ["x,sigma_ref,sigma_got,abs_error,v_ref,v_got"]
    for xi, sr, sg, vr, vg in zip(x.tolist(), s_ex.tolist(), s.tolist(),
                                  v_ex.tolist(), v.tolist()):
        lines.append(f"{xi:.9f},{sr:.12e},{sg:.12e},{abs(sg - sr):.6e},"
                     f"{vr:.12e},{vg:.12e}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# parameter studies
# ---------------------------------------------------------------------------
def degree_sweep(ncells, degrees, final_time, cfl):
    out = []
    order = degree = None
    for p in degrees:
        x, _v, s, meta = case.sample("pulse", ncells, p, final_time, cfl)
        v_ex, s_ex = case.exact_pulse(x, final_time)
        out.append({"degree": p, "cells": ncells, "steps": meta["steps"],
                    "rel_l2_sigma": solver.relative_error(x, s, s_ex)})
    return out


def cfl_sweep(ncells, degree, final_time, cfls):
    out = []
    for cfl in cfls:
        x, _v, s, meta = case.sample("pulse", ncells, degree, final_time, cfl)
        v_ex, s_ex = case.exact_pulse(x, final_time)
        out.append({"cfl": cfl, "cells": ncells, "degree": degree,
                    "steps": meta["steps"], "dt": meta["dt"],
                    "rel_l2_sigma": solver.relative_error(x, s, s_ex)})
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure_profiles(pulse_rows, standing_rows, degree, cfl, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    best = pulse_rows[-1]
    x, v, s, _meta = case.sample("pulse", best["cells"], degree, case.PULSE_TIME,
                                 cfl, nsample=6)
    v_ex, s_ex = case.exact_pulse(x, case.PULSE_TIME)
    axes[0, 0].plot(x, v_ex, "-", color="0.35", label="exact")
    axes[0, 0].plot(x, v, "--", color="C0", lw=1.2,
                    label=f"DG P{degree}, {best['cells']} cells")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("v")
    axes[0, 0].set_title("d'Alembert pulse: velocity")
    axes[0, 0].legend()

    axes[0, 1].plot(x, s_ex, "-", color="0.35", label="exact")
    axes[0, 1].plot(x, s, "--", color="C3", lw=1.2,
                    label=f"DG P{degree}, {best['cells']} cells")
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel(r"$\sigma$")
    axes[0, 1].set_title("stress split into two counter-propagating pulses")
    axes[0, 1].legend()

    for row in pulse_rows:
        axes[1, 0].loglog(row["cells"], row["rel_l2_sigma"], "o-",
                          label=f"sigma (EOC {row.get('eoc_sigma', float('nan')):.2f})")
        axes[1, 0].loglog(row["cells"], row["rel_l2_v"], "s--", label="v")
    axes[1, 0].set_xlabel("cells"); axes[1, 0].set_ylabel("relative L2 error")
    axes[1, 0].set_title("pulse case: convergence")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].legend()

    best_s = standing_rows[-1]
    x, v, s, _meta = case.sample("standing", best_s["cells"], degree,
                                 case.STANDING_TIME, cfl, nsample=6)
    v_ex, s_ex, _u = case.exact_standing(x, case.STANDING_TIME)
    axes[1, 1].plot(x, s_ex, "-", color="0.35", label="exact")
    axes[1, 1].plot(x, s, "--", color="C2", lw=1.2,
                    label=f"DG P{degree}, {best_s['cells']} cells")
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel(r"$\sigma$")
    axes[1, 1].set_title(f"clamped-free eigenmode, t = {case.STANDING_TIME}")
    axes[1, 1].legend()

    fig.suptitle("TensorDG linear elastodynamics verification "
                 "(CASE-DG-DALEMBERT-ELASTIC-1D)")
    fig.tight_layout()
    path = out_dir / "elastic_1d_dalembert.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=case.DEFAULT_DEGREE)
    parser.add_argument("--pulse-time", type=float, default=case.PULSE_TIME)
    parser.add_argument("--standing-time", type=float, default=case.STANDING_TIME)
    parser.add_argument("--cfl", type=float, default=case.DEFAULT_CFL)
    parser.add_argument("--meshes", type=int, nargs="+", default=list(case.MESHES))
    parser.add_argument("--standing-meshes", type=int, nargs="+",
                        default=list(case.STANDING_MESHES))
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--cfls", type=float, nargs="+",
                        default=[0.1, 0.3, 0.8])
    parser.add_argument("--output-dir", default="validation/elastic_1d_dalembert")
    parser.add_argument("--max-relative-error", type=float, default=0.02)
    parser.add_argument("--probe-cells", type=int, default=None,
                        help="cells for the time-tracking anchors "
                             "(default: middle of --meshes)")
    parser.add_argument("--freq-cells", type=int, default=20,
                        help="cells for the eigenfrequency bisection")
    parser.add_argument("--quick", action="store_true",
                        help="small-size smoke run (~1 min)")
    args = parser.parse_args(argv)

    if args.quick:
        args.meshes = [50, 100]
        args.standing_meshes = [40, 80]
        args.degrees = [1, 2]
        args.cfls = [0.1, 0.5]

    started = time.time()
    nsample = max(2 * args.degree + 3, 6)
    pulse = pulse_case(args.meshes, args.degree, args.pulse_time, args.cfl, nsample)
    standing = standing_case(args.standing_meshes, args.degree,
                             args.standing_time, args.cfl, nsample)
    degrees_tbl = degree_sweep(args.meshes[len(args.meshes) // 2],
                               args.degrees, args.pulse_time, args.cfl)
    cfl_tbl = cfl_sweep(args.meshes[len(args.meshes) // 2], args.degree,
                        args.pulse_time, args.cfls)
    probe_cells = args.probe_cells or args.meshes[len(args.meshes) // 2]
    anchors = anchor_rows(args.degree, args.cfl, ncells=args.meshes[-1],
                          ncells_dynamic=probe_cells,
                          ncells_freq=args.freq_cells)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure = figure_profiles(pulse, standing, args.degree, args.cfl, out_dir)
    profile_csv = write_profile_csv(out_dir / "sigma_profile_ref_vs_got.csv",
                                    args.meshes[-1], args.degree, args.cfl)

    worst_pulse = max(r["rel_l2_sigma"] for r in pulse)
    worst_standing = max(r["rel_l2_sigma"] for r in standing)
    gate = {
        "criteria": {"rel_l2_sigma_pulse": args.max_relative_error,
                     "rel_l2_sigma_standing": args.max_relative_error},
        "worst": {"rel_l2_sigma_pulse": worst_pulse,
                  "rel_l2_sigma_standing": worst_standing},
        "passed": bool(worst_pulse < args.max_relative_error
                       and worst_standing < args.max_relative_error),
    }
    payload = {
        "case": case.CASE_KEY,
        "parameters": {**case.parameters(),
                       "degree": args.degree, "cfl": args.cfl,
                       "pulse_time": args.pulse_time,
                       "standing_time": args.standing_time,
                       "meshes": args.meshes,
                       "standing_meshes": args.standing_meshes},
        "pulse_case": pulse,
        "standing_case": standing,
        "degree_sweep": degrees_tbl,
        "cfl_sweep": cfl_tbl,
        "gate": gate,
        "figure": str(figure),
        "profile_csv": str(profile_csv),
    }
    json_path = out_dir / "elastic_1d_dalembert.json"
    json_path.write_text(json.dumps(payload, indent=2))

    bench = {
        "case": case.CASE_KEY,
        "metric": "stress profile sigma(x,t) vs d'Alembert two-wave exact "
                  "solution (no physical dispersion)",
        "tolerance_pct": REL_TOL_PCT,
        "settings": {"degree": args.degree, "cfl": args.cfl,
                     "ncells_profile": args.meshes[-1],
                     "ncells_dynamic": probe_cells,
                     "ncells_freq": args.freq_cells,
                     "pulse_time": args.pulse_time},
        "rows": anchors,
        "worst_rel_error_pct": max(r["rel_error_pct"] for r in anchors
                                   if r["rel_error_pct"] is not None),
        "all_rows_pass": all(r["verdict"] == "PASS" for r in anchors),
        "l2_gate": gate,
        "repro_cmd": ("python -m benchmarks.elastic_1d_dalembert "
                      f"--output-dir {out_dir}"),
    }
    bench_path = out_dir / "benchmark_ref_vs_got.json"
    bench_path.write_text(json.dumps(bench, indent=2))

    elapsed = time.time() - started
    manifest = {
        "case": case.CASE_KEY,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "command": " ".join([Path(sys.executable).name, "-m",
                             "benchmarks.elastic_1d_dalembert",
                             "--output-dir", str(out_dir)]),
        "quick": bool(args.quick),
        "elapsed_seconds": round(elapsed, 2),
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform()},
        "artifacts": {},
    }
    for p in (json_path, bench_path, profile_csv, figure):
        manifest["artifacts"][p.name] = {"bytes": p.stat().st_size,
                                         "sha256": _sha256(p)}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{'case':>9} {'cells':>6} {'steps':>7} {'relL2 v':>11} "
          f"{'relL2 sigma':>12} {'EOC sigma':>9}")
    for row in pulse:
        print(f"{'pulse':>9} {row['cells']:>6d} {row['steps']:>7d} "
              f"{row['rel_l2_v']:>11.3e} {row['rel_l2_sigma']:>12.3e} "
              f"{row.get('eoc_sigma', float('nan')):>9.3f}")
    for row in standing:
        print(f"{'standing':>9} {row['cells']:>6d} {row['steps']:>7d} "
              f"{row['rel_l2_v']:>11.3e} {row['rel_l2_sigma']:>12.3e} "
              f"{row.get('eoc_sigma', float('nan')):>9.3f}")
    print("\nreference vs computed:")
    for row in anchors:
        rel = row["rel_error_pct"]
        rel_s = "n/a" if rel is None else f"{rel:.3e}%"
        print(f"  {row['verdict']:<4} {row['quantity'][:52]:<52} "
              f"ref {row['ref']:.6g}  got {row['got']:.6g}  err {rel_s}")
    print(f"\ngate (<{args.max_relative_error:.0%}): "
          f"{'PASS' if gate['passed'] else 'FAIL'}"
          f"  worst pulse {worst_pulse:.3e}, worst standing {worst_standing:.3e}")
    print(f"json: {json_path}")
    print(f"benchmark: {bench_path}")
    print(f"figure: {figure}")
    print(f"elapsed: {elapsed:.1f} s")
    return 0 if (gate["passed"] and bench["all_rows_pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())