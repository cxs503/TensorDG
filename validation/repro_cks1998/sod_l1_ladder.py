"""Sod shock-tube L1 mesh ladder for the limited modal DG solution.

REPRO-DG-CKS1998 round 2 (TensorDG).  The pre-registered criteria that this
driver machine-checks are written down *before* the run in
``docs/REPRO-DG-CKS1998-round2.md`` (section 2) and are mirrored in ``CRITERIA``.

Why this exists
---------------
The round-1 report (``docs/repro_cks1998.md``) gates Sod on the relative L2 of
the *cell averages at the finest mesh* and honestly records that the N=200 rung
misses the 3 % gate (worst component 4.17 %).  What it never quantified is the
*convergence order of the limited solution at the shock*: the CKS1998 claim is
that a TVD/TVB-limited DG solution loses order at discontinuities (the paper
proves O(h) for the TVB scheme in the scalar case), so a mesh ladder must show
a slope far below the design order p+1 and must not be extrapolated from a
single rung.  This driver measures that slope for the 1-D Euler system, with an
unlimited control run so "the limiter is actually active" is falsifiable.

Usage
-----
    OMP_NUM_THREADS=1 python validation/repro_cks1998/sod_l1_ladder.py
    OMP_NUM_THREADS=1 python validation/repro_cks1998/sod_l1_ladder.py --cells 100 200 400 800 1600
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks._modal_1d import legendre_values  # noqa: E402
from benchmarks.sod_exact import cell_averages, sample  # noqa: E402
from tensordg.euler_1d import GAMMA  # noqa: E402
from tensordg.euler_dg_1d import solve_euler_dg_1d  # noqa: E402

SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)
SOD_T = 0.2
CFL = 0.12
TVB = 1.0
FIELDS = ("density", "velocity", "pressure")
QUADRATURE = 8

# Pre-registered criteria (fixed before the run; see docs §2).
CRITERIA = {
    "C1_monotone": "relative L1 error strictly decreases on every rung, every field",
    "C2_order_bracket": "density L1 order on the finest pair within [0.5, 2.0] "
                        "(limiter half-order floor .. below the design order p+1=3)",
    "C3_cross_check": "cell-average relative L2 at overlapping meshes reproduces the "
                      "round-1 E2 numbers to <=1e-6 relative",
    "C4_gate_extension": "report whether the finest rung meets the 3% relative-L2 gate "
                         "(verdict recorded, gate itself unchanged)",
    "C5_not_roundoff": "finest mesh relative L1 error > 1e-10 (not a roundoff floor)",
    "C6_limiter_active": "unlimited control at the finest mesh has a strictly larger "
                         "density overshoot than the limited run",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover - non-git checkouts
        return "unknown"


def _primitive_at_quadrature(coeff, degree: int, quadrature: int):
    """Pointwise primitive state at Gauss points of every cell.

    ``coeff`` holds the modal coefficients of the *conserved* variables
    (rho, rho*u, E) in the Legendre basis P_k(xi) with xi in [-1, 1].
    """
    values = coeff.detach().cpu().numpy()
    cells = values.shape[0]
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    phi = legendre_values(nodes, degree)                 # (q, modes)
    local = np.einsum("cms,qm->cqs", values, phi)        # (cells, q, 3)
    rho = local[:, :, 0]
    mom = local[:, :, 1]
    energy = local[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        vel = np.where(np.abs(rho) > 0.0, mom / rho, 0.0)
    pressure = (GAMMA - 1.0) * (energy - 0.5 * mom * vel)
    x = (np.arange(cells)[:, None] + 0.5 + 0.5 * nodes[None, :]) / cells
    return x, {"density": rho, "velocity": vel,
               "pressure": pressure}, weights / 2.0


def _exact_at(x, ncells: int):
    fields = sample(x, final_time=SOD_T, diaphragm=0.5)
    return {k: fields[k].reshape(ncells, -1) for k in FIELDS}


def _l1(weights, diff, ref, ncells):
    dx = 1.0 / ncells
    abs_err = float(np.sum(weights[None, :] * np.abs(diff)) * dx)
    norm = float(np.sum(weights[None, :] * np.abs(ref)) * dx)
    return abs_err, abs_err / norm


def run_rung(degree: int, ncells: int, limiter):
    t0 = time.time()
    _, coeff, meta = solve_euler_dg_1d(
        SOD_LEFT, SOD_RIGHT, ncells=ncells, degree=degree, final_time=SOD_T,
        cfl=CFL, limiter=limiter, tvb_constant=TVB, limiter_stage="final",
    )
    x, num, weights = _primitive_at_quadrature(coeff, degree, QUADRATURE)
    ref = _exact_at(x, ncells)
    centres, ref_avg = cell_averages(ncells, SOD_T)
    avg = coeff[:, 0].detach().cpu().numpy()
    got_avg = {
        "density": avg[:, 0],
        "velocity": avg[:, 1] / avg[:, 0],
        "pressure": (GAMMA - 1.0) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / avg[:, 0]),
    }
    out = {
        "degree": degree, "ncells": ncells, "limiter": limiter,
        "steps": int(meta["steps"]), "seconds": round(time.time() - t0, 3),
        "h": 1.0 / ncells,
    }
    for key in FIELDS:
        abs_l1, rel_l1 = _l1(weights, num[key] - ref[key], ref[key], ncells)
        diff_avg = got_avg[key] - ref_avg[key]
        rel_l2_cellavg = float(np.sqrt(np.mean(diff_avg ** 2))
                              / np.sqrt(np.mean(ref_avg[key] ** 2)))
        out[f"l1_abs_{key}"] = abs_l1
        out[f"l1_rel_{key}"] = rel_l1
        out[f"l2_cellavg_rel_{key}"] = rel_l2_cellavg
    out["l1_rel_worst"] = max(out[f"l1_rel_{k}"] for k in FIELDS)
    out["l2_cellavg_worst"] = max(out[f"l2_cellavg_rel_{k}"] for k in FIELDS)
    out["density_max"] = float(num["density"].max())
    out["density_min"] = float(num["density"].min())
    out["pressure_min"] = float(num["pressure"].min())
    out["overshoot_density"] = float(max(num["density"].max() - 1.0,
                                        0.125 - num["density"].min()))
    out["finite"] = bool(np.all(np.isfinite(num["density"]))
                         and np.all(num["pressure"] > 0.0))
    return out


def pairwise_order(errors, hs):
    """Observed order between successive rungs (coarse -> fine)."""
    rows = []
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        order = math.log(ratio) / math.log(hs[i] / hs[i + 1])
        rows.append({"coarse": i, "fine": i + 1, "order": order})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--cells", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--control-cells", type=int, default=None,
                        help="mesh for the unlimited control run (default: finest)")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "validation" / "repro_cks1998" / "artifacts" / "sod_l1_ladder")
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    cells = sorted(args.cells)
    control_cells = args.control_cells or cells[-1]

    print(f"[ladder] degree={args.degree} cells={cells} cfl={CFL} tvb={TVB}", flush=True)
    rungs = []
    for ncells in cells:
        row = run_rung(args.degree, ncells, "tvb")
        rungs.append(row)
        print(f"[ladder] N={ncells:5d} steps={row['steps']:6d} "
              f"l1_rho={row['l1_rel_density']:.6e} "
              f"l2_avg_worst={row['l2_cellavg_worst']:.6e} "
              f"({row['seconds']:.1f}s)", flush=True)

    control = run_rung(args.degree, control_cells, None)
    print(f"[control] unlimited N={control_cells} overshoot={control['overshoot_density']:.6e} "
          f"finite={control['finite']} ({control['seconds']:.1f}s)", flush=True)

    hs = [r["h"] for r in rungs]
    orders = {f"l1_rel_{k}": pairwise_order([r[f"l1_rel_{k}"] for r in rungs], hs)
              for k in FIELDS}
    orders["l1_rel_worst"] = pairwise_order([r["l1_rel_worst"] for r in rungs], hs)
    orders["l2_cellavg_worst"] = pairwise_order([r["l2_cellavg_worst"] for r in rungs], hs)

    # C3 cross-check against the round-1 quick/full artifacts, if present.
    published = {}
    for profile in ("quick", "full"):
        path = REPO / "validation" / "repro_cks1998" / "artifacts" / profile / "metrics.json"
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        for tag, case in blob.get("sections", {}).get("E2_sod", {}).get("cases", {}).items():
            if case.get("degree") != args.degree:
                continue
            published[(profile, tag)] = {
                "ncells": case["ncells"], "rel_l2": case["rel_l2"],
                "worst": case["worst_rel_l2"],
            }
    cross = []
    for (profile, tag), item in sorted(published.items()):
        rung = next((r for r in rungs if r["ncells"] == item["ncells"]), None)
        if rung is None:
            continue
        diffs = {k: abs(rung[f"l2_cellavg_rel_{k}"] - item["rel_l2"][k])
                 / max(abs(item["rel_l2"][k]), 1e-30) for k in FIELDS}
        cross.append({"profile": profile, "tag": tag, "ncells": item["ncells"],
                      "rel_diff": diffs, "max_rel_diff": max(diffs.values())})

    finest = rungs[-1]
    dens_orders = [o["order"] for o in orders["l1_rel_density"]]
    checks = {
        "C1_monotone": all(
            rungs[i][f"l1_rel_{k}"] > rungs[i + 1][f"l1_rel_{k}"]
            for k in FIELDS for i in range(len(rungs) - 1)),
        "C2_order_bracket": 0.5 <= dens_orders[-1] <= 2.0,
        "C3_cross_check": bool(cross) and all(c["max_rel_diff"] <= 1e-6 for c in cross),
        "C4_gate_extension": bool(finest["l2_cellavg_worst"] < 0.03),
        "C5_not_roundoff": bool(finest["l1_rel_density"] > 1e-10),
        "C6_limiter_active": bool(control["overshoot_density"] > finest["overshoot_density"]),
    }
    gate_used = [c for c in cross if c["ncells"] == finest["ncells"]]
    verdict = all(checks.values())

    metrics = {
        "anchor": "REPRO-DG-CKS1998 round 2 / Sod L1 ladder",
        "git_commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "platform": platform.platform(),
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset")},
        "config": {"degree": args.degree, "cells": cells, "cfl": CFL,
                   "tvb_constant": TVB, "limiter_stage": "final",
                   "quadrature": QUADRATURE, "final_time": SOD_T,
                   "left": SOD_LEFT, "right": SOD_RIGHT,
                   "control_cells": control_cells},
        "criteria": CRITERIA,
        "rungs": rungs,
        "control": control,
        "orders": orders,
        "cross_check_round1": cross,
        "gate_mesh_in_round1": [c["tag"] for c in gate_used],
        "checks": checks,
        "verdict": verdict,
        "wall_seconds": round(sum(r["seconds"] for r in rungs) + control["seconds"], 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    with (out_dir / "l1_ladder.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["degree", "ncells", "h", "limiter", "steps",
                         *[f"l1_rel_{k}" for k in FIELDS], "l1_rel_worst",
                         "l2_cellavg_worst", "overshoot_density", "seconds"])
        for r in rungs + [control]:
            writer.writerow([r["degree"], r["ncells"], f"{r['h']:.6f}", r["limiter"],
                             r["steps"], *[f"{r[f'l1_rel_{k}']:.12e}" for k in FIELDS],
                             f"{r['l1_rel_worst']:.12e}",
                             f"{r['l2_cellavg_worst']:.12e}",
                             f"{r['overshoot_density']:.12e}", r["seconds"]])

    render_summary(metrics, out_dir)
    render_figures(metrics, out_dir)
    print(f"[ladder] verdict={'PASS' if verdict else 'FAIL'} "
          f"checks={ {k: v for k, v in checks.items()} }", flush=True)
    return 0 if verdict else 1


def render_summary(metrics: dict, out_dir: Path) -> None:
    lines = [
        "# Sod L1 网格阶梯（REPRO-DG-CKS1998 round 2 驱动工件）",
        "",
        f"- git `{metrics['git_commit'][:12]}` · 生成于 {metrics['generated_at']}",
        f"- degree P{metrics['config']['degree']} · cfl={CFL} · tvb M={TVB} · "
        f"limiter_stage=final · 求积 {QUADRATURE} 点",
        f"- 墙钟 {metrics['wall_seconds']} s",
        "",
        "## 阶梯（相对 L1，8 点 Gauss 逐单元积分）",
        "",
        "| N | steps | L1 ρ | L1 u | L1 p | L1 最差 | 单元平均相对 L2 最差 | 密度超调 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["rungs"]:
        lines.append(
            f"| {r['ncells']} | {r['steps']} | {r['l1_rel_density']:.6e} | "
            f"{r['l1_rel_velocity']:.6e} | {r['l1_rel_pressure']:.6e} | "
            f"{r['l1_rel_worst']:.6e} | {r['l2_cellavg_worst']:.6e} | "
            f"{r['overshoot_density']:.3e} |")
    control = metrics["control"]
    lines += [
        f"| {control['ncells']} (unlimited) | {control['steps']} | "
        f"{control['l1_rel_density']:.6e} | {control['l1_rel_velocity']:.6e} | "
        f"{control['l1_rel_pressure']:.6e} | {control['l1_rel_worst']:.6e} | "
        f"{control['l2_cellavg_worst']:.6e} | {control['overshoot_density']:.3e} |",
        "",
        "## 观测阶（相邻两级的 log 斜率）",
        "",
        "| 量 | " + " | ".join(f"{a}→{b}" for a, b in
                                zip([r["ncells"] for r in metrics["rungs"]][:-1],
                                    [r["ncells"] for r in metrics["rungs"]][1:])) + " |",
        "|---|" + "---:|" * (len(metrics["rungs"]) - 1),
    ]
    for key, rows in metrics["orders"].items():
        lines.append(f"| {key} | " + " | ".join(f"{o['order']:.4f}" for o in rows) + " |")
    lines += ["", "## 预登记判据", "", "| 判据 | 含义 | 结果 |", "|---|---|---|"]
    for key, text in metrics["criteria"].items():
        ok = metrics["checks"].get(key)
        mark = "PASS" if ok else "FAIL"
        lines.append(f"| {key} | {text} | **{mark}** |")
    lines += ["", f"**总判定：{'PASS' if metrics['verdict'] else 'FAIL'}**", ""]
    if metrics["cross_check_round1"]:
        lines += ["## 与 round-1 工件的交叉核对（C3）", "",
                  "| profile | 用例 | N | 最大相对差 |", "|---|---|---:|---:|"]
        for c in metrics["cross_check_round1"]:
            lines.append(f"| {c['profile']} | {c['tag']} | {c['ncells']} | "
                         f"{c['max_rel_diff']:.3e} |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


def render_figures(metrics: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rungs = metrics["rungs"]
    hs = [r["h"] for r in rungs]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for key, label in (("l1_rel_density", "ρ"), ("l1_rel_velocity", "u"),
                       ("l1_rel_pressure", "p")):
        ax.loglog(hs, [r[key] for r in rungs], "o-", label=f"L1 {label}")
    design = metrics["config"]["degree"] + 1
    ref = [rungs[0]["l1_rel_density"] * (hs[0] / h) ** design for h in hs]
    ax.loglog(hs, ref, "k--", lw=1, label=f"design order h^{design}")
    fit = metrics["orders"]["l1_rel_density"][-1]["order"]
    ref2 = [rungs[-1]["l1_rel_density"] * (h / hs[-1]) ** fit for h in hs]
    ax.loglog(hs, ref2, "r:", lw=1.2, label=f"observed slope {fit:.2f}")
    ax.set_xlabel("h = 1/N")
    ax.set_ylabel("relative L1 error")
    ax.set_title("Sod L1 ladder, TVB-limited modal DG (P%d)" % metrics["config"]["degree"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "l1_ladder.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [str(r["ncells"]) for r in rungs] + [f"{metrics['control']['ncells']}\n(unltd)"]
    worst = [r["l2_cellavg_worst"] for r in rungs] + [metrics["control"]["l2_cellavg_worst"]]
    colours = ["#2b6cb0"] * len(rungs) + ["#c53030"]
    ax.bar(range(len(worst)), worst, color=colours)
    ax.axhline(0.03, color="k", ls="--", lw=1, label="3 % gate (round-1, unchanged)")
    ax.set_xticks(range(len(worst)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("N (cells); last bar = unlimited control")
    ax.set_ylabel("worst cell-average relative L2")
    ax.set_title("Gate status per rung (limited vs unlimited control)")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "gate_status.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())