"""Benchmark: d'Alembert pulse crossing a material interface (layered rod).

Verification target
-------------------
A right-going stress pulse launched in layer 1 hits an impedance jump at
``x = interface`` and splits exactly into

    sigma_reflected = R * sigma_incident,   R = (Z2 - Z1)/(Z1 + Z2)
    sigma_transmitted = T * sigma_incident, T = 2 Z2/(Z1 + Z2),   Z = sqrt(rho E)

R and T are exact for the linear velocity--stress system, so the acceptance
criterion is ``|R_num - R|, |T_num - T| < 2%`` (ledger anchor
CASE-DG-DALEMBERT-ELASTIC-1D, layered extension).  Nothing in the repository
verified a *material interface* before (only d'Alembert end reflections, which
are sign flips); this benchmark exercises the heterogeneous interface Riemann
solver in :mod:`tensordg.elastic_1d_layered`.

Run::

    python -m benchmarks.elastic_1d_layered --output-dir validation/elastic_1d_layered
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from tensordg.elastic_1d import gaussian_pulse, solve_elastic_1d
from tensordg.elastic_1d_layered import (
    exact_rightgoing_scattering,
    impedance,
    layered_energy,
    layered_stress_profile,
    rightgoing_pulse_initial_state,
    scattering_coefficients,
    solve_elastic_1d_layered,
)

TOLERANCE = 0.02          # 2 % ledger criterion
# Energy is only guaranteed to the accuracy of SSP-RK3 time integration
# (dt^3 per step, cumulative in t); measured drift on this case is ~3e-4
# at CFL 0.3, so the invariant is stated at 0.1 % (see docs ⑤).
ENERGY_TOL = 1e-3
LENGTH = 4.0
INTERFACE = 2.0
PULSE_CENTER = 1.0
PULSE_WIDTH = 0.05
T_MEASURE = 1.4           # after the interface interaction, before any return
T_INCIDENT = 0.5          # reference amplitude, pulse still inside layer 1

CASES = (
    dict(key="contrast", label="Z2/Z1 = 4 (stiff, heavy)",
         rho2=4.0, young2=4.0),
    dict(key="impedance_matched", label="Z2 = Z1 (no reflection)",
         rho2=2.0, young2=0.5),
    dict(key="soft", label="Z2/Z1 = 0.5 (compliant)",
         rho2=1.0, young2=0.25),
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def layered_material(ncells: int, interface: float, length: float,
                     rho2: float, young2: float):
    n1 = int(round(interface / length * ncells))
    rho = [1.0] * n1 + [rho2] * (ncells - n1)
    young = [1.0] * n1 + [young2] * (ncells - n1)
    return rho, young, n1


def refine_peak(x: torch.Tensor, s: torch.Tensor, lo: float, hi: float):
    """Signed extremum of ``s`` in ``[lo, hi]`` with a 3-point parabola."""
    mask = (x >= lo) & (x <= hi)
    xs, ss = x[mask], s[mask]
    i = int(torch.argmax(ss.abs()))
    if 0 < i < len(ss) - 1:
        y0, y1, y2 = (float(ss[i - 1].abs()), float(ss[i].abs()),
                      float(ss[i + 1].abs()))
        curv = y0 - 2.0 * y1 + y2
        delta = 0.0 if abs(curv) < 1e-300 else 0.5 * (y0 - y2) / curv
        peak = y1 - 0.25 * (y0 - y2) * delta
        step = float(xs[i + 1] - xs[i])
        return float(torch.sign(ss[i])) * peak, float(xs[i] + delta * step)
    return float(ss[i]), float(xs[i])


def relative_l2(numerical: torch.Tensor, exact: torch.Tensor) -> float:
    num = torch.sqrt(torch.sum((numerical - exact) ** 2))
    den = torch.sqrt(torch.sum(exact ** 2))
    return float(num / den) if float(den) > 0 else float("nan")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True).strip()
    except Exception:                                   # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# one scattering case
# ---------------------------------------------------------------------------
def scattering_case(spec, ncells: int, degree: int):
    rho2, young2 = spec["rho2"], spec["young2"]
    rho, young, _n1 = layered_material(ncells, INTERFACE, LENGTH, rho2, young2)
    v0, s0 = rightgoing_pulse_initial_state(INTERFACE, 1.0, 1.0,
                                            PULSE_CENTER, PULSE_WIDTH)
    c2 = math.sqrt(young2 / rho2)
    z1, z2 = impedance(1.0, 1.0), impedance(rho2, young2)
    r_theory, t_theory = scattering_coefficients(1.0, 1.0, rho2, young2)

    common = dict(ncells=ncells, degree=degree, length=LENGTH, rho=rho,
                  young=young, bc="fixed-free", initial_velocity=v0,
                  initial_stress=s0)

    # reference incident amplitude (pulse still inside layer 1)
    x1, cf1, meta1 = solve_elastic_1d_layered(final_time=T_INCIDENT, **common)
    xs1, _v1, ss1 = layered_stress_profile(x1, cf1, 8)
    inc_peak, inc_x = refine_peak(xs1, ss1, PULSE_CENTER + T_INCIDENT - 0.4,
                                  PULSE_CENTER + T_INCIDENT + 0.4)

    # post-interaction field
    x2, cf2, meta2 = solve_elastic_1d_layered(final_time=T_MEASURE, **common)
    xs, vs, ss = layered_stress_profile(x2, cf2, 8)
    v_exact, s_exact = exact_rightgoing_scattering(
        xs, T_MEASURE, INTERFACE, LENGTH, 1.0, 1.0, rho2, young2,
        PULSE_CENTER, PULSE_WIDTH)

    x_ref = 2.0 * INTERFACE - PULSE_CENTER - T_MEASURE
    x_tra = INTERFACE + c2 * (T_MEASURE - (INTERFACE - PULSE_CENTER))
    ref_peak, ref_x = refine_peak(xs, ss, max(0.0, x_ref - 0.4), INTERFACE - 0.05)
    tra_peak, tra_x = refine_peak(xs, ss, INTERFACE + 0.05, x_tra + 0.4)

    r_num = ref_peak / inc_peak
    t_num = tra_peak / inc_peak
    energy0 = layered_energy(cf1, LENGTH / ncells, rho, young)
    energy1 = layered_energy(cf2, LENGTH / ncells, rho, young)

    # wave-speed diagnostic: interface crossing time of the transmitted peak
    crossing = (INTERFACE - PULSE_CENTER) + abs(x_tra - INTERFACE) / c2

    return dict(
        key=spec["key"], label=spec["label"],
        rho2=rho2, young2=young2, z1=z1, z2=z2,
        c1=1.0, c2=c2, impedance_ratio=z2 / z1,
        r_theory=r_theory, t_theory=t_theory,
        r_numeric=r_num, t_numeric=t_num,
        rel_error_R=abs(r_num - r_theory),
        rel_error_T=abs(t_num - t_theory) / abs(t_theory),
        incident_peak=inc_peak, incident_x=inc_x,
        reflected_peak=ref_peak, reflected_x=ref_x,
        transmitted_peak=tra_peak, transmitted_x=tra_x,
        continuity_1_plus_R_minus_T=r_theory + 1.0 - t_theory,
        field_rel_l2=relative_l2(ss, s_exact),
        velocity_rel_l2=relative_l2(vs, v_exact),
        energy_initial=energy0, energy_final=energy1,
        energy_rel_drift=abs(energy1 - energy0) / energy0,
        crossing_time=float(crossing),
        ncells=ncells, degree=degree, steps=meta2["steps"], dt=meta2["dt"],
        samples=dict(x=xs.tolist(), sigma_numeric=ss.tolist(),
                     sigma_exact=s_exact.tolist(), v_numeric=vs.tolist(),
                     v_exact=v_exact.tolist()),
    )


# ---------------------------------------------------------------------------
# uniform reduction (layered code path == homogeneous code path)
# ---------------------------------------------------------------------------
def uniform_reduction(ncells: int = 50, degree: int = 2, final_time: float = 0.2):
    v0 = lambda x: gaussian_pulse(x, 0.5, 0.05)       # noqa: E731
    s0 = lambda x: torch.zeros_like(x)                # noqa: E731
    _c, homogeneous, meta_h = solve_elastic_1d(
        ncells=ncells, degree=degree, final_time=final_time,
        initial_velocity=v0, initial_stress=s0)
    _c2, layered, meta_l = solve_elastic_1d_layered(
        ncells=ncells, degree=degree, final_time=final_time,
        rho=(1.0,), young=(1.0,), initial_velocity=v0, initial_stress=s0)
    return dict(ncells=ncells, degree=degree, final_time=final_time,
                max_abs_diff=float((homogeneous - layered).abs().max()),
                steps_homogeneous=meta_h["steps"],
                steps_layered=meta_l["steps"])


# ---------------------------------------------------------------------------
# convergence in p and in N
# ---------------------------------------------------------------------------
def convergence_spec():
    spec = CASES[0]
    ladder = []
    for degree in (1, 2, 3):
        out = scattering_case(spec, ncells=400, degree=degree)
        ladder.append(dict(axis="p", degree=degree, ncells=400,
                           field_rel_l2=out["field_rel_l2"],
                           rel_error_R=out["rel_error_R"],
                           rel_error_T=out["rel_error_T"]))
    for ncells in (200, 400, 800):
        out = scattering_case(spec, ncells=ncells, degree=2)
        ladder.append(dict(axis="h", degree=2, ncells=ncells,
                           field_rel_l2=out["field_rel_l2"],
                           rel_error_R=out["rel_error_R"],
                           rel_error_T=out["rel_error_T"]))
    return ladder


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(results, ladder, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6), sharex=False)
    for col, res in enumerate(results):
        s = {k: np.asarray(v) for k, v in res["samples"].items()}
        ax = axes[0, col]
        ax.plot(s["x"], s["sigma_exact"], "k-", lw=2.2, label="d'Alembert (exact)")
        ax.plot(s["x"], s["sigma_numeric"], "r--", lw=1.3, label="DG (p=2, N=400)")
        ax.axvline(INTERFACE, color="0.55", ls=":", lw=1.0)
        ax.set_title(f"{res['key']}: Z$_2$/Z$_1$ = {res['impedance_ratio']:.2f}\n"
                     f"R: {res['r_numeric']:+.4f} vs {res['r_theory']:+.4f} | "
                     f"T: {res['t_numeric']:.4f} vs {res['t_theory']:.4f}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel(r"$\sigma$")
        ax.set_xlim(0.5, 3.2); ax.grid(alpha=0.3); ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(s["x"], s["sigma_numeric"] - s["sigma_exact"], color="C3", lw=1.0)
        ax.axvline(INTERFACE, color="0.55", ls=":", lw=1.0)
        ax.set_xlabel("x"); ax.set_ylabel(r"$\sigma_{num}-\sigma_{exact}$")
        ax.set_xlim(0.5, 3.2); ax.grid(alpha=0.3)
    fig.suptitle("Layered rod: right-going pulse through a material interface "
                 "(t = %.1f, c$_1$ = 1)" % T_MEASURE, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    idx = np.arange(len(results))
    width = 0.36
    axes[0].bar(idx - width / 2, [r["r_theory"] for r in results], width,
                label="theory", color="0.4")
    axes[0].bar(idx + width / 2, [r["r_numeric"] for r in results], width,
                label="DG", color="C0")
    axes[0].set_xticks(idx, [r["key"] for r in results])
    axes[0].set_ylabel("R (reflected / incident)")
    axes[0].set_title("Reflection coefficient")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].bar(idx - width / 2, [r["t_theory"] for r in results], width,
                label="theory", color="0.4")
    axes[1].bar(idx + width / 2, [r["t_numeric"] for r in results], width,
                label="DG", color="C0")
    axes[1].set_xticks(idx, [r["key"] for r in results])
    axes[1].set_ylabel("T (transmitted / incident)")
    axes[1].set_title("Transmission coefficient")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path.with_name("elastic_1d_layered_coefficients.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    p_ladder = [r for r in ladder if r["axis"] == "p"]
    h_ladder = [r for r in ladder if r["axis"] == "h"]
    axes[0].semilogy([r["degree"] for r in p_ladder],
                     [r["field_rel_l2"] for r in p_ladder], "o-")
    axes[0].set_xlabel("polynomial degree p"); axes[0].set_ylabel("relative L2 error")
    axes[0].set_title("p-convergence (N = 400)"); axes[0].grid(alpha=0.3, which="both")
    axes[1].loglog([r["ncells"] for r in h_ladder],
                   [r["field_rel_l2"] for r in h_ladder], "s-")
    axes[1].set_xlabel("cells N"); axes[1].set_ylabel("relative L2 error")
    axes[1].set_title("h-convergence (p = 2)"); axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path.with_name("elastic_1d_layered_convergence.png"), dpi=140)
    plt.close(fig)
    return [path, path.with_name("elastic_1d_layered_coefficients.png"),
            path.with_name("elastic_1d_layered_convergence.png")]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="validation/elastic_1d_layered")
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--from-existing", action="store_true",
                        help="re-derive anchors/manifest from cases.json "
                             "(no re-solve, figures untouched)")
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    reduction = uniform_reduction()
    existing = outdir / "cases.json"
    reuse = args.from_existing and existing.exists()
    if reuse:
        results = json.loads(existing.read_text())["cases"]
        prev = json.loads((outdir / "elastic_1d_layered.json").read_text())
        ladder = prev.get("convergence", [])
        results = [dict(r, samples=None) for r in results]
    else:
        results = [scattering_case(spec, args.cells, args.degree)
                   for spec in CASES]
        ladder = [] if args.skip_convergence else convergence_spec()

    # --- machine-judged anchors -------------------------------------------
    anchors = []
    for res in results:
        anchors.append(dict(
            quantity=f"R_{res['key']}", reference=res["r_theory"],
            value=res["r_numeric"],
            rel_error=res["rel_error_R"],
            passed=bool(res["rel_error_R"] < TOLERANCE)))
        anchors.append(dict(
            quantity=f"T_{res['key']}", reference=res["t_theory"],
            value=res["t_numeric"],
            rel_error=res["rel_error_T"],
            passed=bool(res["rel_error_T"] < TOLERANCE)))
        anchors.append(dict(
            quantity=f"field_L2_{res['key']}", reference=0.0,
            value=res["field_rel_l2"], rel_error=res["field_rel_l2"],
            passed=bool(res["field_rel_l2"] < TOLERANCE)))
        anchors.append(dict(
            quantity=f"energy_drift_{res['key']}", reference=0.0,
            value=res["energy_rel_drift"], rel_error=res["energy_rel_drift"],
            passed=bool(res["energy_rel_drift"] < ENERGY_TOL)))
    anchors.append(dict(quantity="uniform_reduction_max_abs_diff", reference=0.0,
                        value=reduction["max_abs_diff"],
                        rel_error=reduction["max_abs_diff"],
                        passed=bool(reduction["max_abs_diff"] < 1e-12)))
    if ladder:
        # every refinement must improve the field error of the previous rung
        for axis in ("p", "h"):
            rungs = [r for r in ladder if r["axis"] == axis]
            for prev, cur in zip(rungs, rungs[1:]):
                anchors.append(dict(
                    quantity=f"monotone_{axis}_{prev['degree']}x{prev['ncells']}"
                             f"->{cur['degree']}x{cur['ncells']}",
                    reference=prev["field_rel_l2"], value=cur["field_rel_l2"],
                    rel_error=cur["field_rel_l2"] / prev["field_rel_l2"],
                    passed=bool(cur["field_rel_l2"] < prev["field_rel_l2"])))

    all_passed = all(a["passed"] for a in anchors)

    payload = dict(
        schema="tensordg-elastic-layered-scattering/1",
        case_key="dalembert-elastic-rod-1d-layered",
        ledger_anchor="CASE-DG-DALEMBERT-ELASTIC-1D",
        generated_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        git_commit=git_commit(),
        environment=dict(python=platform.python_version(),
                         torch=torch.__version__, platform=platform.platform()),
        parameters=dict(length=LENGTH, interface=INTERFACE,
                        pulse_center=PULSE_CENTER, pulse_width=PULSE_WIDTH,
                        t_measure=T_MEASURE, t_incident=T_INCIDENT,
                        bc="fixed-free", cfl=0.3,
                        ncells=args.cells, degree=args.degree,
                        tolerance=TOLERANCE),
        uniform_reduction=reduction,
        convergence=ladder, anchors=anchors, all_passed=all_passed,
    )

    # sampled fields go to CSV + figures; the JSON keeps only scalars
    csv_path = outdir / "scattering_profiles.csv"
    if reuse:
        case_records = results
    else:
        with csv_path.open("w") as fh:
            fh.write("case,x,sigma_numeric,sigma_exact,v_numeric,v_exact\n")
            for res in results:
                s = res["samples"]
                for i in range(len(s["x"])):
                    fh.write(",".join([
                        res["key"], f"{s['x'][i]:.10f}",
                        f"{s['sigma_numeric'][i]:.12e}", f"{s['sigma_exact'][i]:.12e}",
                    f"{s['v_numeric'][i]:.12e}", f"{s['v_exact'][i]:.12e}"]))
    if reuse:
        fig_paths = sorted(outdir.glob("*.png"))
    else:
        fig_paths = make_figures(results, ladder,
                                 outdir / "elastic_1d_layered_profiles.png")
    case_records = [{k: v for k, v in res.items() if k != "samples"}
                    for res in results]

    results_json = outdir / "elastic_1d_layered.json"
    results_json.write_text(json.dumps(
        dict(payload, cases=case_records), indent=2))
    ref_json = outdir / "benchmark_ref_vs_got.json"
    ref_json.write_text(json.dumps(dict(
        schema="tensordg-benchmark-ref-vs-got/1",
        case_key="dalembert-elastic-rod-1d-layered",
        tolerance=TOLERANCE, anchors=anchors, all_passed=all_passed), indent=2))
    (outdir / "cases.json").write_text(json.dumps(
        dict(schema="tensordg-elastic-layered-cases/1", cases=case_records),
        indent=2))

    artifacts = [results_json, ref_json, csv_path, outdir / "cases.json"]
    artifacts += [Path(p) for p in fig_paths]
    manifest = dict(
        schema="tensordg-run-manifest/1",
        command="python -m benchmarks.elastic_1d_layered --output-dir %s"
                % args.output_dir,
        git_commit=git_commit(),
        artifacts=[dict(path=str(p), sha256=sha256_file(p),
                        bytes=p.stat().st_size) for p in artifacts],
        all_passed=all_passed,
    )
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[elastic_1d_layered] anchors {sum(a['passed'] for a in anchors)}"
          f"/{len(anchors)} passed; all_passed={all_passed}")
    for res in case_records:
        print(f"  {res['key']:<18} R {res['r_numeric']:+.6f} vs {res['r_theory']:+.6f}"
              f"  ({res['rel_error_R']:.3%})   T {res['t_numeric']:.6f} vs"
              f" {res['t_theory']:.6f}  ({res['rel_error_T']:.3%})"
              f"   L2 {res['field_rel_l2']:.3%}")
    print(f"  uniform reduction max|diff| = {reduction['max_abs_diff']:.3e}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())