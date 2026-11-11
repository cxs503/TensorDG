"""REPRO-DG-CKS1998 — reproduction of the Cockburn & Shu (1998) benchmark set.

External anchor
---------------
Cockburn, B. & Shu, C.-W., *The Runge-Kutta discontinuous Galerkin method for
conservation laws* (SIAM Review 43(1) 2001 / JCP 141 (1998), "V: multidimensional
systems"), benchmark set for the 1-D RKDG scheme:

  A1  linear advection, smooth data   -> L2 order k+1, paper tolerance k+1/2
  A2  linear advection, square wave   -> limited scheme, O(h^{1/2}) L1 floor
  E2  Sod shock tube via DG vs exact  -> relative L2 < 3 %
  E3  TVB/TVD property of limiter     -> no new extrema (overshoot <= 1 %)

Everything is driven through the *existing* TensorDG public API
(``tensordg.advection_1d``, ``tensordg.euler_dg_1d``, ``tensordg.limiters``,
``tensordg.convergence``); the only additions are the limiter/SSP-RK3 plumbing
in ``solve_advection`` and the periodic option of the TVB limiter, both needed
to run the paper's limited-scheme tests at all.

Usage
-----
    OMP_NUM_THREADS=1 python validation/repro_cks1998/repro_cks1998.py --profile quick
    OMP_NUM_THREADS=1 python validation/repro_cks1998/repro_cks1998.py --profile full

Artifacts (written under ``validation/repro_cks1998/artifacts/<profile>/``,
``--outdir`` overrides):
    metrics.json   every number used by the report, plus the gate verdicts
    convergence.csv / sod.csv / tvd.csv   the tables
    summary.md     programmatically rendered report table (zero hand-copied digits)
    *.png          error-decay, Sod and limiter figures
Exit code 0 iff all pre-declared gates pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tensordg.advection_1d import l1_error, l2_error, solve_advection  # noqa: E402
from tensordg.convergence import observed_order, pairwise_orders  # noqa: E402
from tensordg.euler_dg_1d import solve_euler_dg_1d  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-declared design (fixed before any result is inspected)
# ---------------------------------------------------------------------------
SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)
SOD_T = 0.2
SOD_STATIONS = (0.10, 0.30, 0.45, 0.55, 0.70, 0.90)

PROFILES = {
    "quick": {
        "smooth_cells": [16, 32, 64],
        "discont_cells": [40, 80, 160],
        "sod_cases": [(2, 200), (2, 400)],
        "tvd_cells": 80,
        "tvd_degrees": [1, 2],
    },
    "full": {
        "smooth_cells": [20, 40, 80, 160, 320],
        "discont_cells": [40, 80, 160, 320],
        "sod_cases": [(2, 200), (2, 400), (1, 400)],
        "tvd_cells": 100,
        "tvd_degrees": [1, 2, 3],
    },
}

GATES = {
    # A1: the paper's tolerance for the smooth advection order.
    "smooth_order_tolerance": 0.5,     # order >= degree + 0.5
    # A2: half-order floor for limited advection of a discontinuity.
    "discont_l1_floor": 0.5,
    # E2: Sod relative L2 must beat 3 %.
    "sod_rel_l2": 0.03,
    # E3: no new extrema beyond 1 % of the initial data range.
    "tvd_overshoot": 0.01,
}


def square_wave(x: torch.Tensor) -> torch.Tensor:
    """CKS square wave: u0 = 1 on (0.25, 0.75), 0 elsewhere, periodic."""
    xm = torch.remainder(x, 1.0)
    return ((xm > 0.25) & (xm < 0.75)).to(x.dtype)


def peak_metrics(centers, coeff, npts: int = 21):
    """Cell-average and pointwise extrema of the limited DG polynomial."""
    avg = coeff[:, 0]
    xi = torch.linspace(-1.0, 1.0, npts, dtype=coeff.dtype)
    basis = torch.stack([
        torch.as_tensor(np.polynomial.legendre.legval(xi.numpy(), np.eye(coeff.shape[1])[k]))
        for k in range(coeff.shape[1])
    ], dim=1)
    values = coeff @ basis.T                      # [ncells, npts]
    return {
        "avg_min": float(avg.min()), "avg_max": float(avg.max()),
        "pt_min": float(values.min()), "pt_max": float(values.max()),
    }


def overshoot_pct(peaks, lo: float = 0.0, hi: float = 1.0, scale: float = 1.0) -> dict:
    """Overshoot/undershoot as a fraction of the initial data range."""
    avg_over = max(0.0, peaks["avg_max"] - hi, lo - peaks["avg_min"]) / scale
    pt_over = max(0.0, peaks["pt_max"] - hi, lo - peaks["pt_min"]) / scale
    return {"cell_average": avg_over, "pointwise": pt_over}


# ---------------------------------------------------------------------------
# A1: smooth linear advection -> L2 order
# ---------------------------------------------------------------------------
def run_smooth_advection(profile: dict) -> dict:
    rows, summary = [], {}
    for degree in (1, 2, 3):
        errs = []
        for ncells in profile["smooth_cells"]:
            t0 = time.time()
            c, coeff, meta = solve_advection(
                ncells=ncells, degree=degree, final_time=0.5, cfl=0.3,
            )
            err = l2_error(c, coeff, final_time=0.5)
            errs.append(err)
            rows.append({
                "case": "A1_smooth_sine", "degree": degree, "ncells": ncells,
                "h": 1.0 / ncells, "l2_error": err, "steps": meta["steps"],
                "seconds": round(time.time() - t0, 2),
            })
        res = list(profile["smooth_cells"])   # coarse -> fine resolution n = 1/h
        order = observed_order(res, errs)
        summary[f"p{degree}"] = {
            "l2_errors": errs,
            "observed_order": order,
            "pairwise_orders": pairwise_orders(res, errs),
            "theory_order": degree + 1,
            "gate": degree + GATES["smooth_order_tolerance"],
            "pass": bool(order >= degree + GATES["smooth_order_tolerance"]),
        }
    return {"rows": rows, "by_degree": summary,
            "pass": all(v["pass"] for v in summary.values())}


# ---------------------------------------------------------------------------
# A2: discontinuous advection -> limited scheme, order and extrema
# ---------------------------------------------------------------------------
def run_discontinuous_advection(profile: dict) -> dict:
    rows, summary = [], {}
    for degree in (1, 2, 3):
        l1s, l2s = [], []
        for ncells in profile["discont_cells"]:
            t0 = time.time()
            c, coeff, meta = solve_advection(
                ncells=ncells, degree=degree, final_time=1.0, cfl=0.2,
                initial=square_wave, limiter="tvb", tvb_constant=0.0,
                time_integrator="ssp_rk3",
            )
            e1 = l1_error(c, coeff, 1.0, exact=square_wave, jumps=(0.25, 0.75))
            e2 = l2_error(c, coeff, 1.0, exact=square_wave)
            peaks = peak_metrics(c, coeff)
            over = overshoot_pct(peaks)
            l1s.append(e1)
            l2s.append(e2)
            rows.append({
                "case": "A2_square_wave_limited", "degree": degree,
                "ncells": ncells, "h": 1.0 / ncells, "l1_error": e1,
                "l2_error": e2, "avg_max": peaks["avg_max"],
                "pt_max": peaks["pt_max"], "pt_min": peaks["pt_min"],
                "overshoot_cell_average_pct": 100 * over["cell_average"],
                "overshoot_pointwise_pct": 100 * over["pointwise"],
                "steps": meta["steps"], "seconds": round(time.time() - t0, 2),
            })
        res = list(profile["discont_cells"])
        o1 = observed_order(res, l1s) if len(res) >= 2 else float("nan")
        summary[f"p{degree}"] = {
            "l1_errors": l1s, "l2_errors": l2s,
            "l1_order": o1,
            "l1_order_pairs": pairwise_orders(res, l1s) if len(res) >= 2 else [],
            "l2_order": observed_order(res, l2s) if len(res) >= 2 else float("nan"),
            "pass": bool(len(res) < 2 or o1 >= GATES["discont_l1_floor"]),
        }
    return {"rows": rows, "by_degree": summary,
            "pass": all(v["pass"] for v in summary.values())}


# ---------------------------------------------------------------------------
# E2: Sod shock tube via DG vs exact Riemann solution
# ---------------------------------------------------------------------------
def run_sod(profile: dict) -> dict:
    from benchmarks.sod_exact import cell_averages
    from tensordg.euler_1d import GAMMA

    rows, summary = [], {}
    finest = {}
    for degree, ncells in profile["sod_cases"]:
        finest[degree] = max(ncells, finest.get(degree, 0))
    for degree, ncells in profile["sod_cases"]:
        t0 = time.time()
        x, coeff, meta = solve_euler_dg_1d(
            SOD_LEFT, SOD_RIGHT, ncells=ncells, degree=degree,
            final_time=SOD_T, cfl=0.12, limiter="tvb", tvb_constant=1.0,
            limiter_stage="final",
        )
        avg = coeff[:, 0]
        rho = avg[:, 0]
        vel = avg[:, 1] / rho
        pre = (GAMMA - 1.0) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / rho)
        _, ref = cell_averages(ncells, SOD_T)
        got = {"density": rho.numpy(), "velocity": vel.numpy(), "pressure": pre.numpy()}
        # fixed x-stations -> literal ref (exact) vs got (cell average) numbers
        stations = []
        for xs in SOD_STATIONS:
            idx = min(int(xs * ncells), ncells - 1)
            stations.append({
                "x": round((idx + 0.5) / ncells, 6),
                "ref": {k: float(ref[k][idx]) for k in got},
                "got": {k: float(got[k][idx]) for k in got},
            })
        rel = {}
        for key in ("density", "velocity", "pressure"):
            diff = got[key] - ref[key]
            rel[key] = float(np.sqrt(np.mean(diff ** 2)) / np.sqrt(np.mean(ref[key] ** 2)))
        tag = f"p{degree}_n{ncells}"
        worst = max(rel.values())
        gate_case = ncells == finest[degree]
        summary[tag] = {
            "degree": degree, "ncells": ncells, "steps": meta["steps"],
            "limiter": meta["limiter"], "rel_l2": rel, "worst_rel_l2": worst,
            # the 3 % gate is declared on the finest mesh of each degree; the
            # coarser cases are reported to show the convergence trend
            "gate_case": gate_case,
            "stations": stations,
            "pass": bool(worst < GATES["sod_rel_l2"]),
            "seconds": round(time.time() - t0, 2),
        }
        rows.append({"case": "E2_sod_dg", "degree": degree, "ncells": ncells,
                     **{f"rel_l2_{k}": v for k, v in rel.items()},
                     "worst_rel_l2": worst, "steps": meta["steps"],
                     "seconds": round(time.time() - t0, 2)})
    return {"rows": rows, "cases": summary,
            "pass": all(v["pass"] for v in summary.values() if v["gate_case"])}


# ---------------------------------------------------------------------------
# E3: TVD/TVB property — limited vs unlimited (control)
# ---------------------------------------------------------------------------
def run_tvd(profile: dict) -> dict:
    """Limited vs unlimited control, under both troubled-cell policies.

    ``tvb``  = the published Cockburn-Shu indicator (release monotone cells)
    ``all``  = strict policy, minmod every cell (no released bulge)
    """
    rows, summary = [], {}
    for degree in profile["tvd_degrees"]:
        variants = ((None, "tvb", "unlimited"), ("tvb", "tvb", "tvb_M0"),
                    ("tvb", "all", "strict_all"))
        for limiter, policy, tag in variants:
            t0 = time.time()
            c, coeff, meta = solve_advection(
                ncells=profile["tvd_cells"], degree=degree, final_time=1.0,
                cfl=0.2, initial=square_wave, limiter=limiter,
                tvb_constant=0.0, time_integrator="ssp_rk3",
                troubled_policy=policy,
            )
            peaks = peak_metrics(c, coeff)
            over = overshoot_pct(peaks)
            row = {
                "case": "E3_tvd", "degree": degree, "limiter": tag,
                "ncells": profile["tvd_cells"], **peaks,
                "overshoot_cell_average_pct": 100 * over["cell_average"],
                "overshoot_pointwise_pct": 100 * over["pointwise"],
                "steps": meta["steps"], "seconds": round(time.time() - t0, 2),
            }
            rows.append(row)
            summary[f"p{degree}_{tag}"] = row
    strict = [v for k, v in summary.items() if k.endswith("strict_all")]
    unlimited = [v for k, v in summary.items() if k.endswith("unlimited")]
    return {
        "rows": rows, "cases": summary,
        "pass": all(v["overshoot_pointwise_pct"] / 100 <= GATES["tvd_overshoot"]
                    for v in strict),
        # the control documents that the gate is not trivially satisfied
        "control_violates": [v["overshoot_pointwise_pct"] / 100 > GATES["tvd_overshoot"]
                             for v in unlimited],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def figure_smooth(smooth: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    h = None
    for key, val in smooth["by_degree"].items():
        deg = int(key[1:])
        n = len(val["l2_errors"])
        h = np.array(PROFILES[CURRENT_PROFILE]["smooth_cells"][:n], dtype=float)
        ax.loglog(1.0 / h, val["l2_errors"], "o-",
                  label=f"DG P{deg}: order {val['observed_order']:.2f} (target {deg + 1})")
        ref = val["l2_errors"][0] * (h[0] / h) ** (deg + 0.5)
        ax.loglog(1.0 / h, ref, "k--", lw=0.8, alpha=0.5,
                  label=f"slope {deg}+1/2 gate" if deg == 1 else None)
    ax.set_xlabel("1/h (cells)")
    ax.set_ylabel(r"$L^2$ error")
    ax.set_title("A1 linear advection, smooth data (CKS1998)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_tvd(tvd: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for degree in PROFILES[CURRENT_PROFILE]["tvd_degrees"]:
        for limiter, policy, tag, style in (
                (None, "tvb", "unlimited", ":"),
                ("tvb", "tvb", "TVB M=0 (paper)", "--"),
                ("tvb", "all", "strict minmod", "-")):
            c, coeff, _ = solve_advection(
                ncells=PROFILES[CURRENT_PROFILE]["tvd_cells"], degree=degree,
                final_time=1.0, cfl=0.2, initial=square_wave, limiter=limiter,
                tvb_constant=0.0, time_integrator="ssp_rk3",
                troubled_policy=policy,
            )
            xi = torch.linspace(-1, 1, 11, dtype=coeff.dtype)
            basis = torch.stack([
                torch.as_tensor(np.polynomial.legendre.legval(xi.numpy(), np.eye(coeff.shape[1])[k]))
                for k in range(coeff.shape[1])
            ], dim=1)
            values = (coeff @ basis.T).numpy()          # [ncells, npts]
            h = 1.0 / c.numel()
            xs = (c.numpy()[:, None] + 0.5 * h * xi.numpy()[None, :]).ravel()
            lbl = f"P{degree} {tag}"
            ax.plot(xs, values.ravel(), style, lw=1.0, label=lbl,
                    color=f"C{degree - 1}")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xlabel("x")
    ax.set_ylabel("$u_h$")
    ax.set_title("E3 square wave at t=1: limiter vs unlimited (CKS1998)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_sod(sod: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from benchmarks.sod_exact import cell_averages

    degree, ncells = max(PROFILES[CURRENT_PROFILE]["sod_cases"])
    x, coeff, _ = solve_euler_dg_1d(
        SOD_LEFT, SOD_RIGHT, ncells=ncells, degree=degree, final_time=SOD_T,
        cfl=0.12, limiter="tvb", tvb_constant=1.0, limiter_stage="final",
    )
    avg = coeff[:, 0]
    rho = avg[:, 0].numpy()
    _, ref = cell_averages(ncells, SOD_T)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(x.numpy(), rho, "o", ms=2.5, label=f"DG P{degree}, N={ncells}")
    ax.plot(x.numpy(), ref["density"], "k-", lw=1.0, label="exact Riemann solution")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("E2 Sod shock tube at t=0.2 (CKS1998 DG)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_summary(payload: dict, outdir: Path) -> str:
    s = payload["sections"]
    lines = ["# REPRO-DG-CKS1998 — measured vs anchor", "",
             f"profile `{payload['profile']}` · device `{payload['environment']['device']}` · "
             f"torch {payload['environment']['torch']} · {payload['environment']['cpu']}", ""]
    lines += ["## A1 linear advection, smooth data — L2 order (gate: order >= p+0.5)", "",
              "| degree | h | L2 error | observed order | theory | gate |", "|---:|---:|---:|---:|---:|---|"]
    for row in s["A1_smooth"]["rows"]:
        o = s["A1_smooth"]["by_degree"][f"p{row['degree']}"]
        lines.append(f"| {row['degree']} | 1/{row['ncells']} | {row['l2_error']:.3e} | "
                     f"{o['observed_order']:.3f} | {o['theory_order']} | {o['gate']:.1f} |")
    lines += ["", "## A2 square wave, TVB M=0 + SSP-RK3 (gate: L1 order >= 0.5)", "",
              "| degree | h | L1 error | L2 error | pointwise max | overshoot % |", "|---:|---:|---:|---:|---:|---:|"]
    for row in s["A2_discontinuous"]["rows"]:
        lines.append(f"| {row['degree']} | 1/{row['ncells']} | {row['l1_error']:.3e} | "
                     f"{row['l2_error']:.3e} | {row['pt_max']:.5f} | {row['overshoot_pointwise_pct']:.3f} |")
    lines += ["", "## E2 Sod via DG vs exact (gate: relative L2 < 3 %)", "",
              "| degree | cells | rho | u | p | worst |", "|---:|---:|---:|---:|---:|---:|"]
    for tag, c in s["E2_sod"]["cases"].items():
        lines.append(f"| {c['degree']} | {c['ncells']} | {100*c['rel_l2']['density']:.3f}% | "
                     f"{100*c['rel_l2']['velocity']:.3f}% | {100*c['rel_l2']['pressure']:.3f}% | "
                     f"{100*c['worst_rel_l2']:.3f}% |")
    lines += ["", "## E3 TVD property (gate: overshoot <= 1 %, declared on the strict policy)", "",
              "| degree | scheme | avg max | pointwise max | pointwise min | overshoot % |", "|---:|---|---:|---:|---:|---:|"]
    for tag, c in s["E3_tvd"]["cases"].items():
        lines.append(f"| {c['degree']} | {c['limiter']} | {c['avg_max']:.5f} | {c['pt_max']:.5f} | "
                     f"{c['pt_min']:.5f} | {c['overshoot_pointwise_pct']:.3f} |")
    verdict = payload["verdict"]
    lines += ["", "## Verdict", ""]
    for key, val in verdict.items():
        if key in ("all_pass",):
            continue
        lines.append(f"- **{key}**: {'PASS' if val else 'FAIL'}")
    lines.append(f"- **all_pass**: {verdict['all_pass']}")
    text = "\n".join(lines) + "\n"
    (outdir / "summary.md").write_text(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    global CURRENT_PROFILE
    CURRENT_PROFILE = args.profile
    profile = PROFILES[args.profile]
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    outdir = (Path(args.outdir) if args.outdir
             else REPO / "validation" / "repro_cks1998" / "artifacts" / args.profile)
    outdir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    smooth = run_smooth_advection(profile)
    discont = run_discontinuous_advection(profile)
    sod = run_sod(profile)
    tvd = run_tvd(profile)

    for name, section in (("A1_smooth", smooth), ("A2_discontinuous", discont),
                          ("E2_sod", sod), ("E3_tvd", tvd)):
        keys = list(section["rows"][0].keys()) if section["rows"] else []
        with open(outdir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(section["rows"])

    verdict = {
        "A1_smooth_order": smooth["pass"],
        "A2_discontinuous_order": discont["pass"],
        "E2_sod_three_percent": sod["pass"],
        "E3_tvd_no_new_extrema": tvd["pass"],
    }
    verdict["all_pass"] = all(verdict.values())

    payload = {
        "anchor": "Cockburn & Shu 1998/2001 RKDG benchmark set (SIAM Review 43(1), JCP 141)",
        "profile": args.profile,
        "gates": GATES,
        "environment": {
            "device": "cpu", "torch": torch.__version__,
            "numpy": np.__version__, "python": platform.python_version(),
            "cpu": platform.machine(),
            "torch_threads": torch.get_num_threads(),
            "host_loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
        },
        "config": profile,
        "sections": {
            "A1_smooth": smooth, "A2_discontinuous": discont,
            "E2_sod": sod, "E3_tvd": tvd,
        },
        "sod_reference_stations": {
            tag: case["stations"] for tag, case in sod["cases"].items()
        },
        "verdict": verdict,
        "tvd_control_violates_gate": tvd["control_violates"],
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (outdir / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))

    figure_smooth(smooth, outdir / "a1_l2_order.png")
    figure_tvd(tvd, outdir / "e3_limiter.png")
    figure_sod(sod, outdir / "e2_sod.png")
    render_summary(payload, outdir)

    print(json.dumps({"verdict": verdict, "wall_seconds": payload["wall_seconds"],
                      "outdir": str(outdir)}, indent=2))
    return 0 if verdict["all_pass"] else 1


CURRENT_PROFILE = "quick"

if __name__ == "__main__":
    raise SystemExit(main())