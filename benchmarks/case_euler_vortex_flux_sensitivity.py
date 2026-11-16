#!/usr/bin/env python3
"""Flux sensitivity of the isentropic-vortex error-conservation result.

Why this exists
---------------
The published CASE-DG-EULER-VORTEX-ERR verdict is scoped to ``HLLC + SSP-RK3``
(see ``case_euler_vortex_errconserve/CASE-DG-EULER-VORTEX-ERR-m22549-report.md``,
section ②).  A reviewer is entitled to ask the obvious adversarial question:

    Is "the error is a mesh property, not a time property" (alpha << 1) a
    statement about the *DG discretisation on this mesh*, or is it an artefact
    of the particular Riemann solver -- with HLLC being unusually accurate on
    smooth isentropic flow?

This script answers it by re-running the *same design family*
(``dg_d2_n32_cfl0.2``: P2, 32x32 cells, CFL 0.2, beta = 5, diagonal transport)
with a deliberately more diffusive interface flux, Rusanov / local
Lax-Friedrichs (``flux="rusanov"`` in :mod:`tensordg.euler_dg_2d`), on the same
checkpoint ladder t = 1, 2, 5, 10 and through exactly the same measurement
chain the case gate uses (volume-quadrature relative L2 vs the periodic
min-image analytic solution, ``benchmarks/verify_euler_dg_2d._vortex_error``).

Nothing about the solver, the reference or the metric is duplicated here: the
run reuses ``benchmarks.case_euler_vortex_errconserve``'s operators, initial
state, measurements and growth-exponent estimator, so a difference between the
two series is attributable to the flux alone.

Usage
-----
    # one checkpoint (used as a compute-queue shard)
    python3 benchmarks/case_euler_vortex_flux_sensitivity.py --checkpoint 7

    # after every checkpoint has landed
    python3 benchmarks/case_euler_vortex_flux_sensitivity.py --aggregate

Checkpoints are indexed ``flux * len(TIMES) + time_index`` with
``FLUXES = ("hllc", "rusanov")`` and ``TIMES = (1.0, 2.0, 5.0, 10.0)``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.case_euler_vortex_errconserve import (  # noqa: E402
    BACKGROUND,
    DOMAIN,
    FIELDS,
    STRENGTH,
    T_REF,
    _dg_measurements,
    _exponent_theil_sen,
    _initial_dg_state,
    _fit_window,
    _operators,
    _reference_kwargs,
)
from benchmarks.verify_euler_dg_2d import _vortex_error as dg_vortex_error  # noqa: E402
import tensordg.euler_dg_2d as dg2  # noqa: E402

FLUXES = ("hllc", "rusanov")
TIMES = (1.0, 2.0, 5.0, 10.0)
CELLS = 32
DEGREE = 2
CFL = 0.2
OUTPUT_DIR = REPO_ROOT / "case_euler_vortex_flux_sensitivity"

CHECKPOINTS = [(flux, t) for flux in FLUXES for t in TIMES]


def checkpoint_path(flux: str, final_time: float) -> Path:
    return OUTPUT_DIR / "results" / f"dg_d2_n32_{flux}_t{final_time:g}.json"


def run_checkpoint(flux: str, final_time: float) -> dict:
    started = time.time()
    _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=CELLS,
        degree=DEGREE,
        final_time=final_time,
        cfl=CFL,
        strength=STRENGTH,
        background_velocity=BACKGROUND,
        domain=DOMAIN,
        flux=flux,
    )
    nodes, weights, basis, _, _ = _operators(DEGREE)
    reference_kwargs = _reference_kwargs({}, final_time)
    errors = dg_vortex_error(
        coeff, basis, nodes, DOMAIN, final_time, STRENGTH, BACKGROUND, **reference_kwargs
    )
    final = _dg_measurements(coeff, basis, nodes, weights)
    coeff0, basis0, nodes0, weights0 = _initial_dg_state(
        {"degree": DEGREE, "cells": CELLS, "final_time": final_time}
    )
    initial = _dg_measurements(coeff0, basis0, nodes0, weights0)
    record = {
        "task": "dg",
        "family": "flux_sensitivity",
        "flux": flux,
        "degree": DEGREE,
        "cells": CELLS,
        "final_time": final_time,
        "cfl": CFL,
        "steps": meta["steps"],
        "dt": meta["dt"],
        "dt_note": "final (clamped) step; steps*dt is not the elapsed time",
        "dofs": meta["dofs"],
        "errors": {name: float(errors[name]) for name in FIELDS},
        "metric": "volume-quadrature relative L2 (L2-projected modal state)",
        "reference_model": "min_image",
        "solver_flux": meta["flux"],
        "final": final,
        "initial": initial,
        "wall_seconds": time.time() - started,
    }
    assert meta["flux"] == flux, "solver metadata disagrees with the requested flux"
    return record


def series(records: list, flux: str) -> list:
    rows = [r for r in records if r["flux"] == flux]
    return sorted(rows, key=lambda r: r["final_time"])


def growth(times, errors) -> dict:
    """The case's own growth statistic: exponent over t >= T_REF plus spread."""
    window = _fit_window(times, errors, T_REF)
    if window is None:
        return {"alpha": float("nan"), "alpha_theil_sen": float("nan"),
                "points": 0, "growth_factor": float("nan")}
    lt, le = window
    alpha = float(np.polyfit(lt, le, 1)[0])
    return {
        "alpha": alpha,
        "alpha_theil_sen": _exponent_theil_sen(lt, le),
        "points": int(len(lt)),
        "growth_factor": float(errors[-1] / errors[0]),
    }


def aggregate() -> dict:
    records = []
    missing = []
    for flux, final_time in CHECKPOINTS:
        path = checkpoint_path(flux, final_time)
        if not path.exists():
            # A checkpoint that has not landed is reported as missing, never
            # silently interpolated: the fit window shrinks or goes NaN, which is
            # the honest reading of an incomplete ladder.
            missing.append(f"{flux}@t={final_time:g}")
            continue
        records.append(json.loads(path.read_text()))

    summary = {}
    for flux in FLUXES:
        rows = series(records, flux)
        times = [r["final_time"] for r in rows]
        max_error = [max(r["errors"][f] for f in FIELDS) for r in rows]
        summary[flux] = {
            "times": times,
            "max_error_percent": [100.0 * e for e in max_error],
            "per_field_percent": {
                f: [100.0 * r["errors"][f] for r in rows] for f in FIELDS
            },
            "steps": [r["steps"] for r in rows],
            "conservation_drift": [
                max(
                    abs(r["final"]["invariants"][k] - r["initial"]["invariants"][k])
                    for k in ("mass", "momentum_x", "momentum_y", "energy")
                )
                for r in rows
            ],
            "entropy_drift": [
                abs(r["final"]["invariants"]["entropy"] - r["initial"]["invariants"]["entropy"])
                / abs(r["initial"]["invariants"]["entropy"])
                for r in rows
            ],
            **growth(times, max_error),
        }

    # Paired comparison at every checkpoint: same mesh, same metric, same reference.
    pairs = []
    for i, final_time in enumerate(TIMES):
        pair_series = {flux: [r for r in series(records, flux)
                              if r["final_time"] == final_time] for flux in FLUXES}
        if not all(pair_series.values()):
            continue
        err = {flux: max(pair_series[flux][0]["errors"][f] for f in FIELDS) for flux in FLUXES}
        pairs.append({
            "final_time": final_time,
            "max_error_percent": {flux: 100.0 * err[flux] for flux in FLUXES},
            "rusanov_over_hllc": float(err["rusanov"] / err["hllc"]),
        })

    verdict = {
        "alpha_within_unit": {flux: bool(summary[flux]["alpha"] <= 1.0) for flux in FLUXES},
        "tolerance_percent": 6.0,
        "t1_within_tolerance": {
            flux: bool(max(summary[flux]["per_field_percent"][f][0] for f in FIELDS) <= 6.0)
            for flux in FLUXES
        },
        "conservation_within_1e-10": {
            flux: bool(max(summary[flux]["conservation_drift"]) < 1e-10) for flux in FLUXES
        },
        "entropy_within_1e-3": {
            flux: bool(max(summary[flux]["entropy_drift"]) < 1e-3) for flux in FLUXES
        },
        "max_error_ratio": float(max((p["rusanov_over_hllc"] for p in pairs), default=float("nan"))),
        "alpha_delta": float(abs(summary["hllc"]["alpha"] - summary["rusanov"]["alpha"])),
        "complete": not missing,
    }
    verdict["flux_invariant_claim"] = bool(
        all(verdict["alpha_within_unit"].values())
        and all(verdict["t1_within_tolerance"].values())
        and all(verdict["conservation_within_1e-10"].values())
    )

    bundle = {
        "case": "EULER-VORTEX-FLUX-SENSITIVITY",
        "ledger_slug": "CASE-DG-EULER-VORTEX-ERR",
        "design_family": f"dg_d{DEGREE}_n{CELLS}_cfl{CFL}",
        "fluxes": list(FLUXES),
        "times": list(TIMES),
        "records": len(records),
        "missing": missing,
        "summary": summary,
        "pairs": pairs,
        "verdict": verdict,
        "note": (
            "Rusanov is the local Lax-Friedrichs two-wave solver: same DG operator, "
            "same SSP-RK3, same mesh, same metric/reference. Only the interface flux differs."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "flux_sensitivity.json").write_text(json.dumps(bundle, indent=1) + "\n")
    (OUTPUT_DIR / "flux_sensitivity.md").write_text(markdown(bundle))
    try:
        make_figures(bundle)
    except Exception as exc:  # pragma: no cover - figures are a nicety, not evidence
        print(f"[figure] skipped: {exc}")
    return bundle


def markdown(bundle: dict) -> str:
    rows = [
        "# CASE-DG-EULER-VORTEX-ERR：数值通量敏感性（HLLC vs Rusanov）",
        "",
        "本文件由 `benchmarks/case_euler_vortex_flux_sensitivity.py --aggregate` 从 "
        "`results/*.json` 程序化生成，数字零手抄。",
        "",
        f"设计族：`{bundle['design_family']}`（P{DEGREE}，{CELLS}×{CELLS}，CFL {CFL}，"
        f"β = {STRENGTH}，对角输运）。检查点 t = {', '.join(f'{t:g}' for t in TIMES)}。",
        "两族除界面通量外完全相同：同一 DG 算子、同一 SSP-RK3、同一网格、"
        "同一相对 L2 度量与同一周期 min-image 解析参考。",
        "",
        "## 误差（最大场，相对 L2，%）",
        "",
        "| t | HLLC | Rusanov | Rusanov/HLLC | 6% 容差 |",
        "|---|---|---|---|---|",
    ]
    for pair in bundle["pairs"]:
        if not all(np.isfinite(list(pair["max_error_percent"].values()))):
            continue
        rows.append(
            f"| {pair['final_time']:g} | {pair['max_error_percent']['hllc']:.4f}% | "
            f"{pair['max_error_percent']['rusanov']:.4f}% | "
            f"{pair['rusanov_over_hllc']:.2f}× | "
            f"{'✓' if max(pair['max_error_percent'].values()) <= 6.0 else '✗'} |"
        )
    if bundle["missing"]:
        rows += ["", f"**未落地检查点**：{', '.join(bundle['missing'])}（增长统计仅在 ≥3 个检查点上拟合）", ""]
    rows += ["", "## 增长统计（与门禁同一估计量）", "",
             "| 通量 | α（最小二乘） | α（Theil–Sen） | e(10)/e(1) | 步数 |", "|---|---|---|---|---|"]
    for flux in FLUXES:
        s = bundle["summary"][flux]
        rows.append(
            f"| {flux} | {s['alpha']:.4f} | {s['alpha_theil_sen']:.4f} | "
            f"{s['growth_factor']:.3f} | {s['steps'][-1]} |"
        )
    v = bundle["verdict"]
    rows += ["", "## 判定", "", "| 项 | HLLC | Rusanov |", "|---|---|---|",
             f"| t=1 最大场 < 6% | {v['t1_within_tolerance']['hllc']} | {v['t1_within_tolerance']['rusanov']} |",
             f"| α ≤ 1 | {v['alpha_within_unit']['hllc']} | {v['alpha_within_unit']['rusanov']} |",
             f"| 守恒漂移 < 1e-10 | {v['conservation_within_1e-10']['hllc']} | "
             f"{v['conservation_within_1e-10']['rusanov']} |",
             f"| 熵漂移 < 1e-3 | {v['entropy_within_1e-3']['hllc']} | "
             f"{v['entropy_within_1e-3']['rusanov']} |",
             "",
             f"* 最大 Rusanov/HLLC 误差比 = {v['max_error_ratio']:.2f}×；"
             f"|Δα| = {v['alpha_delta']:.4f}。",
             f"* 「误差守恒是离散/网格属性，不是某一 Riemann 通量的属性」："
             f"**{v['flux_invariant_claim']}**。",
             ""]
    return "\n".join(rows)


def make_figures(bundle: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"hllc": ("o-", "HLLC (published)"), "rusanov": ("s--", "Rusanov / LF")}
    for flux in FLUXES:
        s = bundle["summary"][flux]
        style, label = styles[flux]
        axes[0].loglog(s["times"], s["max_error_percent"], style, label=label)
        axes[1].semilogy(s["times"], s["entropy_drift"], style, label=label)
    axes[0].axhline(6.0, color="crimson", ls=":", label="6% acceptance")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("max field relative L2 (%)")
    axes[0].set_title("Error vs transport time, P2/32$^2$")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("relative entropy drift")
    axes[1].set_title("Entropy budget (same statistic as C4)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_flux_sensitivity.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# quick ladder: the same comparison on a coarse mesh, cheap enough for a
# short interactive run (the production n=32 ladder needs the compute queue)
# ---------------------------------------------------------------------------

QUICK_CELLS = 12
QUICK_TIMES = (1.0, 2.0, 5.0, 10.0)


def quick(cells=None, times=None) -> dict:
    """Both fluxes on a coarse mesh, in one process; writes ``quick_ladder.json``.

    Resumable: every checkpoint is appended to the JSON as soon as it is done, so
    a run that hits an interactive wall-clock budget can be re-invoked and picks
    up where it stopped.
    """
    cells = QUICK_CELLS if cells is None else int(cells)
    times = QUICK_TIMES if times is None else tuple(float(t) for t in times)
    path = OUTPUT_DIR / "quick_ladder_raw.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    if done.get("cells") not in (None, cells) or done.get("times") not in (None, list(times)):
        done = {}
    raw = done.get("ladder", {})
    for flux in FLUXES:
        rows = raw.get(flux, [])
        finished = {r["final_time"] for r in rows}
        for final_time in times:
            if final_time in finished:
                continue
            started = time.time()
            _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
                ncells=cells, degree=DEGREE, final_time=final_time, cfl=CFL,
                strength=STRENGTH, background_velocity=BACKGROUND, domain=DOMAIN, flux=flux,
            )
            nodes, _weights, basis, _d, _s = _operators(DEGREE)
            errors = dg_vortex_error(
                coeff, basis, nodes, DOMAIN, final_time, STRENGTH, BACKGROUND,
                **_reference_kwargs({}, final_time)
            )
            max_percent = 100.0 * max(float(errors[f]) for f in FIELDS)
            rows.append({
                "final_time": final_time,
                "steps": meta["steps"],
                "max_error_percent": max_percent,
                "per_field_percent": {f: 100.0 * float(errors[f]) for f in FIELDS},
                "wall_seconds": time.time() - started,
            })
            print(f"[quick] {flux} t={final_time:g} steps={meta['steps']} "
                  f"max={max_percent:.4f}% ({time.time() - started:.1f}s)", flush=True)
            rows.append({"final_time": final_time, "steps": meta["steps"],
                         "max_error_percent": max_percent,
                         "per_field_percent": {f: 100.0 * float(errors[f]) for f in FIELDS},
                         "wall_seconds": time.time() - started})
            raw[flux] = rows
            path.write_text(json.dumps({"cells": cells, "times": list(times),
                                        "ladder": raw}, indent=1) + "\n")
    # one row per checkpoint (a resumed run may have appended the same t twice)
    ladder = {}
    for flux in FLUXES:
        by_time = {}
        for row in raw.get(flux, []):
            by_time[float(row["final_time"])] = row
        ladder[flux] = [by_time[t] for t in sorted(by_time)]
    if not all(sorted(r["final_time"] for r in rows) == sorted(times) for rows in ladder.values()):
        print("[quick] incomplete ladder saved to", path)
        return {"cells": cells, "times": list(times), "ladder": ladder, "incomplete": True}
    summary = {
        flux: growth([r["final_time"] for r in rows], [r["max_error_percent"] for r in rows])
        for flux, rows in ladder.items()
    }
    bundle = {
        "case": "EULER-VORTEX-FLUX-SENSITIVITY-QUICK",
        "ledger_slug": "CASE-DG-EULER-VORTEX-ERR",
        "cells": cells,
        "degree": DEGREE,
        "cfl": CFL,
        "times": list(QUICK_TIMES),
        "note": (
            f"Coarse-mesh ({cells}x{cells}) short-budget ladder. The production "
            "32x32 design-family ladder is produced by --checkpoint/--aggregate on the "
            "compute queue; this ladder is pre-asymptotic and is reported as such."
        ),
        "ladder": ladder,
        "growth": summary,
        "ratio_at_t1": float(
            ladder["rusanov"][0]["max_error_percent"] / ladder["hllc"][0]["max_error_percent"]
        ),
        "max_ratio": float(max(
            r_u["max_error_percent"] / r_h["max_error_percent"]
            for r_u, r_h in zip(ladder["rusanov"], ladder["hllc"])
        )),
        "alpha_within_unit": {flux: bool(summary[flux]["alpha"] <= 1.0) for flux in FLUXES},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "quick_ladder.json").write_text(json.dumps(bundle, indent=1) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        styles = {"hllc": ("o-", "HLLC"), "rusanov": ("s--", "Rusanov / LF")}
        for flux, rows in ladder.items():
            style, label = styles[flux]
            ax.semilogy([r["final_time"] for r in rows],
                        [r["max_error_percent"] for r in rows], style, label=label)
        ax.axhline(6.0, color="crimson", ls=":", label="6% acceptance")
        ax.set_xlabel("t")
        ax.set_ylabel("max field relative L2 (%)")
        ax.set_title(f"Flux sensitivity, P{DEGREE}/{QUICK_CELLS}$^2$ (pre-asymptotic)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig_quick_flux_sensitivity.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"[figure] skipped: {exc}")
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=int, default=None,
                        help=f"index in [0, {len(CHECKPOINTS) - 1}] = flux * {len(TIMES)} + time index")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the checkpoint table")
    parser.add_argument("--quick", action="store_true",
                        help="coarse both-flux ladder in one process (resumable)")
    parser.add_argument("--quick-cells", type=int, default=QUICK_CELLS)
    parser.add_argument("--quick-times", default=None,
                        help="comma separated, default %s" % (QUICK_TIMES,))
    args = parser.parse_args(argv)

    if args.list:
        for index, (flux, final_time) in enumerate(CHECKPOINTS):
            print(index, flux, final_time)
        return 0
    if args.checkpoint is not None:
        flux, final_time = CHECKPOINTS[args.checkpoint]
        record = run_checkpoint(flux, final_time)
        path = checkpoint_path(flux, final_time)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + "\n")
        print(f"[{args.checkpoint}] {flux} t={final_time:g} max="
              f"{100 * max(record['errors'].values()):.4f}% steps={record['steps']} "
              f"wall={record['wall_seconds']:.0f}s -> {path}")
        return 0
    if args.quick:
        qtimes = None if args.quick_times is None else [
            float(x) for x in args.quick_times.split(",")]
        bundle = quick(args.quick_cells, qtimes)
        print(json.dumps({k: bundle[k] for k in ("growth", "ratio_at_t1", "max_ratio",
                                                "alpha_within_unit")}, indent=1))
        return 0
    if args.aggregate:
        bundle = aggregate()
        print(json.dumps(bundle["verdict"], indent=1))
        return 0
    parser.error("pass --checkpoint N, --aggregate or --list")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())