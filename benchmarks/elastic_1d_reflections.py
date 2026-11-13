"""Long-time (multi-reflection) d'Alembert verification of the 1-D elastic bar.

The shipped d'Alembert case (``docs/CASE-DG-DALEMBERT-ELASTIC-1D.md``) grades the
solver against the *single-image* closed form, which is only valid while the two
half-packets have not yet wrapped around the periodic seam -- i.e. for
``c t <= 0.45`` of the 1.0-long bar (documented limitation #3 of that report) and
never past the second reflection at the ends.  That leaves the physically
important regime untested: cumulative numerical dispersion/phase error
accumulates over *many* transits, and it is exactly the quantity a
wave-propagation code is judged on.

This driver closes that gap with a reference that is exact for **all** times: the
image (reflection-series) solution

    V_periodic(y)  = sum_n V(y - n L)                      (L: seam period)
    V_clamped_free(y): odd about x=0, even about x=L  ->  4L-periodic,
                       V(y + 2L) = -V(y)                   (mixed reflection signs)

    v(x,t)     = [V(x-ct) + V(x+ct)] / 2
    sigma(x,t) = rho c [V(x+ct) - V(x-ct)] / 2

Each transit adds one image with the sign the physical end imposes, so the
comparison stays a genuine analytic benchmark to arbitrary time.  Grading metric
is the energy norm ``||v-v_ex||^2 + ||(sigma-sigma_ex)/(rho c)||^2`` normalised by
its exact counterpart: the plain relative L2 error of ``sigma`` degenerates when
the exact stress vanishes (e.g. the periodic 2-transit reassembly at ``t=L/c``,
where ``sigma_ex == 0`` exactly, so the denominator is zero).

Outputs: ``elastic_1d_reflections.json`` (transit sweep + convergence),
``benchmark_ref_vs_got.json`` (row-wise verdicts), ``reflection_profiles.csv``,
``elastic_1d_reflections.png`` and ``run_manifest.json`` with sha256 digests.

Usage::

    python -m benchmarks.elastic_1d_reflections \
        --output-dir validation/elastic_1d_reflections
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tensordg.elastic_1d import (
    _dalembert_fixed_free_images,
    _dalembert_periodic_images,
    _seam_limit,
    dalembert_pulse,
    elastic_energy,
    elastic_wave_speed,
    gaussian_pulse,
    relative_error,
    solve_elastic_1d,
    stress_profile,
)

ROOT = Path(__file__).resolve().parent.parent

#: Anchor tolerance of the case under test (feature_ledger acceptance, <2%).
REL_TOL_PCT = 2.0

LENGTH = 1.0
RHO = 1.0
YOUNG = 1.0
C = elastic_wave_speed(RHO, YOUNG)
CENTER = 0.5
WIDTH = 0.05
DEGREE = 2
CFL = 0.3

#: ``c t / L`` values (one unit = one full traversal of the bar).
PERIODIC_TRANSITS = (0.2, 1.0, 2.0, 4.0)
CLAMPED_TRANSITS = (0.5, 1.5, 3.5)
SWEEP_CELLS = 200
CONV_TRANSIT = 2.5   # non-integer: sigma_exact does not vanish here
CONV_CELLS = (50, 100, 200)
PROFILE_CELLS = 200
CSV_SERIES = (("periodic", 1.0), ("periodic", 2.0), ("periodic", 4.0),
              ("clamped-free", 1.5), ("clamped-free", 3.5))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zero_stress(x: torch.Tensor) -> torch.Tensor:
    """Stress-free release: the d'Alembert decomposition assumes ``sigma(0)=0``."""
    return torch.zeros_like(x)


def exact(scenario: str, x: torch.Tensor, t: float):
    """Image-summed exact solution, valid for any ``t``."""
    if scenario == "periodic":
        return _dalembert_periodic_images(x, t, LENGTH, RHO, YOUNG, CENTER, WIDTH)
    if scenario == "clamped-free":
        return _dalembert_fixed_free_images(x, t, LENGTH, RHO, YOUNG, CENTER, WIDTH)
    raise ValueError(f"unknown scenario {scenario!r}")


def solve(scenario: str, ncells: int, final_time: float, degree: int = DEGREE,
          cfl: float = CFL):
    bc = {"periodic": "periodic", "clamped-free": "fixed-free"}[scenario]
    return solve_elastic_1d(
        ncells=ncells, degree=degree, final_time=final_time, length=LENGTH,
        rho=RHO, young=YOUNG, cfl=cfl, bc=bc,
        initial_velocity=gaussian_pulse, initial_stress=zero_stress)


def energy_norm_error(x, v, s, v_ex, s_ex, rho=RHO, c=C) -> float:
    """``||(dv, ds/(rho c))|| / ||(v_ex, s_ex/(rho c))||`` -- never degenerate."""
    dv = v - v_ex
    ds = (s - s_ex) / (rho * c)
    num = float(torch.sum(dv * dv + ds * ds))
    den = float(torch.sum(v_ex * v_ex + (s_ex / (rho * c)) ** 2))
    if den <= 0.0:                                       # pragma: no cover
        raise ValueError("exact state has zero energy norm")
    return math.sqrt(num / den)


def measure(scenario: str, ncells: int, final_time: float,
            degree: int = DEGREE, cfl: float = CFL) -> dict:
    """Solve one configuration and grade it against the image-sum solution."""
    centers, coeff, meta = solve(scenario, ncells, final_time, degree, cfl)
    x, v, s = stress_profile(centers, coeff, nsample=2 * degree + 3)
    v_ex, s_ex = exact(scenario, x, final_time)
    v_norm = float(torch.sum(v_ex * v_ex))
    s_norm = float(torch.sum(s_ex * s_ex))
    # At integer transits V(x-ct) == V(x+ct) exactly, so sigma_exact is
    # identically zero and a relative L2 error of sigma is undefined -- that is
    # why the graded metric is the energy norm (it stays finite either way).
    stress_degenerate = s_norm <= 1e-16 * max(v_norm, 1e-300)
    return {
        "scenario": scenario,
        "bc": meta["bc"],
        "ncells": ncells,
        "degree": degree,
        "cfl": cfl,
        "final_time": final_time,
        "transits": C * final_time / LENGTH,
        "steps": meta["steps"],
        "energy_norm_error": energy_norm_error(x, v, s, v_ex, s_ex),
        "rel_l2_velocity": relative_error(x, v, v_ex) if v_norm > 0 else None,
        "rel_l2_stress": (None if stress_degenerate
                          else relative_error(x, s, s_ex)),
        "stress_is_degenerate": stress_degenerate,
        "exact_stress_norm": math.sqrt(s_norm),
        "exact_velocity_norm": math.sqrt(v_norm),
        "exact_peak_velocity": float(v_ex.abs().max()),
        "numerical_peak_velocity": float(v.abs().max()),
        "energy_initial": elastic_energy(*solve_elastic_1d(
            ncells=ncells, degree=degree, final_time=0.0, length=LENGTH, rho=RHO,
            young=YOUNG, cfl=cfl,
            bc="periodic" if scenario == "periodic" else "fixed-free",
            initial_velocity=gaussian_pulse, initial_stress=zero_stress)[:2],
            rho=RHO, young=YOUNG),
        "energy_final": elastic_energy(centers, coeff, rho=RHO, young=YOUNG),
        "_profile": (x, v, s, v_ex, s_ex),
    }


def _row(quantity: str, ref: float, got: float, tol_pct: float = REL_TOL_PCT,
         note: str = "") -> dict:
    if ref == 0.0:
        rel_pct = abs(got) * 100.0
    else:
        rel_pct = abs(got - ref) / abs(ref) * 100.0
    return {
        "quantity": quantity,
        "ref": ref,
        "got": got,
        "rel_error_pct": rel_pct,
        "tolerance_pct": tol_pct,
        "verdict": "PASS" if rel_pct < tol_pct else "FAIL",
        "note": note,
    }


def build_anchors(periodic_sweep, clamped_sweep) -> tuple[list[dict], dict]:
    """Row-wise reference-vs-computed table (every ref is computed, not typed)."""
    rows: list[dict] = []
    extras: dict = {}

    by_transit = {(m["scenario"], round(m["transits"], 6)): m
                  for m in periodic_sweep + clamped_sweep}

    # 1-3: profile accuracy after many transits (the new regime).
    for scenario, transits, label in (
            ("periodic", 2.0, "periodic"),
            ("periodic", 4.0, "periodic"),
            ("clamped-free", 3.5, "clamped-free")):
        if (scenario, transits) not in by_transit:      # --quick smoke runs
            continue
        m = by_transit[(scenario, transits)]
        rows.append(_row(
            f"{label} bar, energy-norm error of (v, sigma) vs image-series exact "
            f"solution after {transits:g} transits (N={m['ncells']}, P{m['degree']})",
            0.0, m["energy_norm_error"] * 100.0,
            note="graded value is a percentage error, ref = 0"))

    # 4: periodic reassembly at one full transit -- analytic peak is 1.0.
    m1 = by_transit.get(("periodic", 1.0))
    if m1 is not None:
        rows.append(_row("periodic reassembly peak velocity at t = L/c (exact 1.0)",
                         1.0, m1["numerical_peak_velocity"]))

    # 5: clamped-free peak velocity matches the image solution.
    m35 = by_transit.get(("clamped-free", 3.5)) or max(
        [m for m in clamped_sweep], key=lambda m: m["transits"], default=None)
    if m35 is not None:
        rows.append(_row(f"clamped-free peak velocity at {m35['transits']:g} transits",
                         m35["exact_peak_velocity"], m35["numerical_peak_velocity"]))

    # 6: discrete energy conservation across the longest periodic run.
    m4 = (by_transit.get(("periodic", 4.0)) or max(
        periodic_sweep, key=lambda m: m["transits"], default=None))
    if m4 is not None:
        drift = abs(m4["energy_final"] - m4["energy_initial"]) / m4["energy_initial"]
        rows.append(_row(
            f"periodic elastic energy drift over {m4['transits']:g} transits",
            0.0, drift * 100.0,
            note="graded value is a percentage drift, ref = 0"))
        extras[f"energy_drift_{m4['transits']:g}_transits"] = drift

    # 7: boundary-condition leakage of the numerical solution (clamped end v=0,
    #    free end sigma=0), normalised by the field scale.
    if m35 is not None:
        m = m35
        x, v, s, _v_ex, _s_ex = m["_profile"]
        h = LENGTH / m["ncells"]
        left = v[x < h]
        right = s[x > LENGTH - h]
        rows.append(_row("clamped end velocity leakage |v(0)| / max|v|",
                         0.0, float(left.abs().max()) / float(v.abs().max()) * 100.0,
                         note="graded value is a percentage leak, ref = 0"))
        rows.append(_row("free end stress leakage max|sigma(L)| / max|sigma|",
                         0.0, float(right.abs().max())
                         / float(s.abs().max()) * 100.0,
                         note="graded value is a percentage leak, ref = 0"))

    # 8: the reference itself -- image sum must reduce to the single-image closed
    #    form before the seam, otherwise the benchmark grades against nonsense.
    x = torch.linspace(0.001, 0.999, 4001, dtype=torch.float64)
    v_ref, s_ref = dalembert_pulse(x, 0.2, RHO, YOUNG, CENTER, WIDTH, LENGTH)
    v_img, s_img = _dalembert_periodic_images(x, 0.2, LENGTH, RHO, YOUNG,
                                              CENTER, WIDTH)
    rows.append(_row("image sum vs single-image closed form at t=0.2 (max deviation)",
                     0.0, float(max((v_ref - v_img).abs().max(),
                                    (s_ref - s_img).abs().max())) * 100.0,
                     note="graded value is a percentage of the unit pulse, ref = 0"))

    # 9: image sum must satisfy the mixed boundary conditions at the ends.
    tap = torch.tensor([0.0, LENGTH], dtype=torch.float64)
    leak = 0.0
    for t in CLAMPED_TRANSITS + (7.5,):
        vt, st = _dalembert_fixed_free_images(tap, t, LENGTH, RHO, YOUNG,
                                              CENTER, WIDTH)
        leak = max(leak, abs(float(vt[0])), abs(float(st[1])))
    rows.append(_row("image sum: |v(0)| and |sigma(L)| for t up to 7.5 (max)",
                     0.0, leak * 100.0,
                     note="graded value is a percentage of the unit pulse, ref = 0"))

    # 10: the old single-image form must refuse to be a silent wrong reference.
    guard = "NO_RAISE"
    try:
        dalembert_pulse(tap, 0.6, RHO, YOUNG, CENTER, WIDTH, LENGTH)
    except ValueError:
        guard = "ValueError"
    rows.append({
        "quantity": "dalembert_pulse(c*t=0.6) raises instead of returning a "
                    "wrapped-away field",
        "ref": "ValueError", "got": guard, "rel_error_pct": 0.0 if guard == "ValueError" else 100.0,
        "tolerance_pct": REL_TOL_PCT,
        "verdict": "PASS" if guard == "ValueError" else "FAIL",
        "note": f"seam limit c*t <= {_seam_limit(LENGTH, CENTER):g}",
    })
    return rows, extras


def reference_validity_scan(times=(0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50)):
    """How far the shipped single-image form is from the image sum, vs ``c t``.

    The nominal limit of ``dalembert_pulse`` is ``c t = min(center, L-center)``
    (the packet *centre* reaching the seam), but the Gaussian tail crosses
    earlier: departing by ``center - c t`` widths, the dropped image is of order
    ``exp(-((center - c t)/width)**2)``.  Measured here, never assumed, so the
    report can state the practical validity window instead of the nominal one.
    """
    x = torch.linspace(0.001, 0.999, 4001, dtype=torch.float64)
    rows = []
    for t in times:
        _, s_plain = dalembert_pulse(x, t, RHO, YOUNG, CENTER, WIDTH, LENGTH,
                                     strict=False)
        _, s_img = _dalembert_periodic_images(x, t, LENGTH, RHO, YOUNG,
                                              CENTER, WIDTH)
        rows.append({"c_t_over_L": C * t / LENGTH,
                     "max_stress_deviation": float((s_plain - s_img).abs().max()),
                     "deviation_pct": float((s_plain - s_img).abs().max())
                     / (0.5 * RHO * C) * 100.0})
    practical = [r["c_t_over_L"] for r in rows
                 if r["max_stress_deviation"] < 1e-4] or [0.0]
    smallest_bad = [r["c_t_over_L"] for r in rows if r["deviation_pct"] >= REL_TOL_PCT]
    return {
        "nominal_limit_c_t": _seam_limit(LENGTH, CENTER),
        "three_width_rule_c_t": CENTER - 3.0 * WIDTH,
        "measured_practical_limit_c_t": max(practical),
        "first_c_t_at_or_above_tolerance": min(smallest_bad) if smallest_bad else None,
        "scan": rows,
        "note": "deviation is max|sigma_plain - sigma_image| / (rho c / 2), i.e. "
                "as a percentage of the unit-pulse stress amplitude",
    }


def wrong_symmetry_reference(x: torch.Tensor, t: float) -> torch.Tensor:
    """Deliberately wrong reference: mirror (free) reflection at the clamped end.

    A plausible sign error in the reflection series.  The metric must reject it,
    which proves the profile gate is not vacuous.
    """
    period = 4.0 * LENGTH
    r = torch.remainder(x + C * t, period)
    r2 = torch.remainder(x - C * t, period)
    out = []
    for yy in (r, r2):
        fold = torch.where(yy > 2.0 * LENGTH, period - yy, yy)
        out.append(gaussian_pulse(torch.where(fold > LENGTH, 2.0 * LENGTH - fold,
                                              fold), CENTER, WIDTH))
    g_plus, g_minus = out
    return 0.5 * (g_minus + g_plus), 0.5 * RHO * C * (g_plus - g_minus)


def build_fault_injection(clamped_sweep) -> dict:
    """Prove the gate has teeth: a wrong-sign reference must be rejected."""
    m = max(clamped_sweep, key=lambda r: r["transits"])
    x, v, s, _v_ex, _s_ex = m["_profile"]
    v_wrong, s_wrong = wrong_symmetry_reference(x, m["final_time"])
    err = energy_norm_error(x, v, s, v_wrong, s_wrong)
    return {
        "scenario": "clamped-free",
        "transits": m["transits"],
        "ncells": m["ncells"],
        "degree": m["degree"],
        "wrong_reference": "free (mirror) reflection at the clamped end",
        "energy_norm_error_vs_wrong_reference": err,
        "energy_norm_error_vs_correct_reference": m["energy_norm_error"],
        "detected": err > REL_TOL_PCT / 100.0,
        "note": "the same numerical solution is graded against a wrong-sign "
                "reflection series; a gate that passes this would be vacuous",
    }


def make_figure(path: Path, sweeps, convergence: dict, fault: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    ax = axes[0, 0]
    per = [r for r in sweeps if r["scenario"] == "periodic"
           and r["transits"] in (2.0, 4.0)] or \
          [r for r in sweeps if r["scenario"] == "periodic"]
    for m, style in zip(per, ("-", "--")):
        x, v, s, _v_ex, s_ex = m["_profile"]
        ax.plot(x, s, style, lw=1.2,
                label=f"DG N={m['ncells']} P{m['degree']}, {m['transits']:g} transits")
        ax.plot(x, s_ex, ":", lw=2.0, color="0.35",
                label=f"image-series exact, t={m['final_time']:g}")
    ax.set_xlabel("x / L")
    ax.set_ylabel(r"$\sigma$")
    ax.set_title("(a) periodic bar, stress after 2 and 4 transits")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    m = max([r for r in sweeps if r["scenario"] == "clamped-free"],
            key=lambda r: r["transits"])
    x, v, s, v_ex, s_ex = m["_profile"]
    ax.plot(x, v, "-", lw=1.2, label="DG v")
    ax.plot(x, v_ex, ":", lw=2.0, color="0.35", label="exact v")
    ax.plot(x, s, "-", lw=1.2, label=r"DG $\sigma$")
    ax.plot(x, s_ex, ":", lw=2.0, color="0.15", label=r"exact $\sigma$")
    ax.set_xlabel("x / L")
    ax.set_title(f"(b) clamped-free bar after {m['transits']:g} transits "
                 "(mixed reflections)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for scenario, marker in (("periodic", "o"), ("clamped-free", "s")):
        pts = sorted([r for r in sweeps if r["scenario"] == scenario],
                     key=lambda r: r["transits"])
        ax.semilogy([p["transits"] for p in pts],
                    [max(p["energy_norm_error"], 1e-16) * 100 for p in pts],
                    marker + "-", label=scenario)
    ax.axhline(REL_TOL_PCT, color="crimson", ls="--", lw=1.0,
               label=f"{REL_TOL_PCT:g}% acceptance")
    ax.set_xlabel("transits = c t / L  (1 = one full traversal)")
    ax.set_ylabel("energy-norm error [%]")
    ax.set_title("(c) error accumulation over repeated reflections (N=200, P2)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    ns = [c["ncells"] for c in convergence["cases"]]
    errs = [c["energy_norm_error"] for c in convergence["cases"]]
    ax.loglog(ns, errs, "o-", label="energy-norm error")
    if convergence.get("eoc") is not None:
        ref = errs[0] * (np.array(ns) / ns[0]) ** (-(DEGREE + 1))
        ax.loglog(ns, ref, "k:", lw=1.0,
                  label=f"slope -(p+1) = -{DEGREE + 1} (EOC={convergence['eoc']:.2f})")
    ax.set_xlabel("cells N")
    ax.set_ylabel("energy-norm error at 2 transits")
    ax.set_title(f"(d) long-time convergence at t={CONV_TRANSIT:g} "
                 f"({CONV_TRANSIT:g} transits, P{DEGREE})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("TensorDG -- d'Alembert verification past the first seam: "
                 f"multi-reflection image series (wrong-sign reference error "
                 f"{fault['energy_norm_error_vs_wrong_reference']*100:.1f}%)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="validation/elastic_1d_reflections")
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--cells", type=int, default=SWEEP_CELLS)
    parser.add_argument("--tracked-transits", type=float, nargs="+",
                        default=list(PERIODIC_TRANSITS))
    parser.add_argument("--clamped-transits", type=float, nargs="+",
                        default=list(CLAMPED_TRANSITS))
    parser.add_argument("--conv-cells", type=int, nargs="+", default=list(CONV_CELLS))
    parser.add_argument("--max-relative-error", type=float, default=REL_TOL_PCT / 100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.tracked_transits = [0.2, 1.0]
        args.clamped_transits = [0.5]
        args.conv_cells = [50, 100]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if True:
        periodic = [measure("periodic", args.cells, t, args.degree, args.cfl)
                    for t in args.tracked_transits]
        clamped = [measure("clamped-free", args.cells, t, args.degree, args.cfl)
                   for t in args.clamped_transits]
        sweeps = periodic + clamped

        # long-time convergence (also the dispersion-error decay with N)
        conv_cases = [measure("periodic", n, CONV_TRANSIT, args.degree, args.cfl)
                      for n in args.conv_cells]
        eoc = None
        if len(conv_cases) >= 2:
            (n0, e0), (n1, e1) = ((conv_cases[0]["ncells"],
                                   conv_cases[0]["energy_norm_error"]),
                                  (conv_cases[-1]["ncells"],
                                   conv_cases[-1]["energy_norm_error"]))
            if e0 > 0 and e1 > 0:
                eoc = math.log(e0 / e1) / math.log(n1 / n0)

        rows, extras = build_anchors(periodic, clamped)
        fault = build_fault_injection(clamped)
        rows.append({
            "quantity": "fault injection: same solution graded against a "
                        "wrong-sign reflection series must be rejected",
            "ref": 1.0, "got": 1.0 if fault["detected"] else 0.0,
            "rel_error_pct": 0.0 if fault["detected"] else 100.0,
            "tolerance_pct": REL_TOL_PCT,
            "verdict": "PASS" if fault["detected"] else "FAIL",
            "note": f"wrong-reference error "
                    f"{fault['energy_norm_error_vs_wrong_reference']*100:.2f}%",
        })
        rows.append({
            "quantity": "fault injection: wrong reference must not be accepted "
                        "by the graded metric",
            "ref": REL_TOL_PCT / 100,
            "got": fault["energy_norm_error_vs_wrong_reference"],
            "rel_error_pct": (0.0 if fault["energy_norm_error_vs_wrong_reference"]
                              > REL_TOL_PCT / 100 else 100.0),
            "tolerance_pct": 1.0,
            "verdict": "PASS" if (fault["energy_norm_error_vs_wrong_reference"]
                                  > REL_TOL_PCT / 100) else "FAIL",
            "note": "got must exceed the acceptance threshold",
        })

    all_pass = all(r["verdict"] == "PASS" for r in rows)
    worst = max(r["rel_error_pct"] for r in rows)

    profiles = out / "reflection_profiles.csv"
    with profiles.open("w") as fh:
        fh.write("scenario,final_time,x,v,s,v_exact,sigma_exact\n")
        for m in sweeps:
            if (m["scenario"], m["transits"]) not in CSV_SERIES:
                continue
            x, v, s, v_ex, s_ex = m["_profile"]
            for xi, vi, si, ve, se in zip(x.tolist(), v.tolist(), s.tolist(),
                                          v_ex.tolist(), s_ex.tolist()):
                fh.write(f"{m['scenario']},{m['final_time']},{xi:.10f},"
                         f"{vi:.12e},{si:.12e},{ve:.12e},{se:.12e}\n")

    png = out / "elastic_1d_reflections.png"
    if not args.no_figure:
        make_figure(png, sweeps, {"cases": conv_cases, "eoc": eoc}, fault)

    def clean(ms):
        return [{k: v for k, v in m.items() if not k.startswith("_")} for m in ms]

    payload = {
        "case": "dalembert-elastic-rod-1d",
        "kind": "long-time multi-reflection image-series verification",
        "git_sha": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {"length": LENGTH, "rho": RHO, "young": YOUNG, "c": C,
                   "center": CENTER, "width": WIDTH, "degree": args.degree,
                   "cfl": args.cfl, "sweep_cells": args.cells,
                   "transit_unit": "c t / L"},
        "reference": {
            "periodic": "sum_n V(x -/+ c t - n L)",
            "clamped-free": "4L-periodic continuation, odd about x=0, even about x=L",
            "literature": "K. F. Graff, Wave Motion in Elastic Solids (1975), "
                          "ch. 1: reflection sign at clamped (v=0) and free "
                          "(sigma=0) ends",
        },
        "transit_sweep": clean(sweeps),
        "convergence": {"transit": CONV_TRANSIT, "cases": clean(conv_cases),
                        "eoc": eoc, "theoretical_eoc": args.degree + 1},
        "fault_injection": fault,
        "reference_validity": reference_validity_scan(),
        "energy_drift": extras,
        "elapsed_seconds": round(time.time() - started, 3),
        "platform": platform.platform(),
    }
    (out / "elastic_1d_reflections.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")

    bench = {
        "case": "dalembert-elastic-rod-1d",
        "kind": "benchmark_ref_vs_got",
        "tolerance_pct": REL_TOL_PCT,
        "reference": payload["reference"],
        "config": payload["config"],
        "rows": rows,
        "all_rows_pass": all_pass,
        "worst_rel_error_pct": worst,
        "repro_cmd": "python -m benchmarks.elastic_1d_reflections "
                     f"--output-dir {args.output_dir}",
        "git_sha": payload["git_sha"],
    }
    (out / "benchmark_ref_vs_got.json").write_text(
        json.dumps(bench, indent=2, sort_keys=True) + "\n")

    manifest = {
        "case": "dalembert-elastic-rod-1d",
        "repro_cmd": bench["repro_cmd"],
        "git_sha": payload["git_sha"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "all_rows_pass": all_pass,
        "elapsed_note": "wall clock of the producing run",
        "artifacts": {},
    }
    for f in sorted(out.iterdir()):
        if f.is_file() and f.name != "run_manifest.json":
            manifest["artifacts"][f.name] = {
                "bytes": f.stat().st_size, "sha256": _sha256(f)}
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[reflections] rows: {len(rows)}, all_pass={all_pass}, "
          f"worst={worst:.4f}% , EOC={eoc if eoc is None else round(eoc, 3)}, "
          f"elapsed={payload['elapsed_seconds']}s")
    for f in ("periodic", "clamped-free"):
        for m in [x for x in sweeps if x["scenario"] == f]:
            print(f"  {f:13s} t={m['final_time']:.2f} ({m['transits']:.2f} transits) "
                  f"energy-norm err = {m['energy_norm_error']*100:.5f}% "
                  f"steps={m['steps']}")
    return 0 if all_pass else 1


if __name__ == "__main__":                               # pragma: no cover
    raise SystemExit(main())