#!/usr/bin/env python3
"""T2 unified core-path benchmark (anchor GAP-LBM-T2-GPU-COREPATH).

Runs the pre-registered protocol of ``tensorlbm.core_path`` on the
T3-validated case ``taylor-green-2d`` and writes, programmatically,

    outputs/core_path_t2/summary.json   every arm + verdict + machine context
    outputs/core_path_t2/fig_*.png      field contour / trade-off / memory

Nothing in the summary is hand-typed: every number is measured here, and every
throughput number carries the verdict produced by ``admissibility``.

Usage:
    python benchmarks/core_path_t2_bench.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tensorlbm.core_path import (  # noqa: E402
    ACCURACY_TOL,
    CONSISTENCY_TOL,
    DTYPE_CONSISTENCY_TOL,
    T3_CASE,
    CoreConfig,
    CorePath,
    probe_backends,
    run_arm,
    taylor_green_2d,
)

OUT = HERE.parent / "outputs" / "core_path_t2"


def machine_context() -> dict:
    try:
        nproc = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        nproc = os.cpu_count()
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:  # pragma: no cover
        load1 = load5 = load15 = float("nan")
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "nproc_affinity": nproc,
        "loadavg": [round(load1, 2), round(load5, 2), round(load15, 2)],
    }


def arms(quick: bool, fast: bool = False):
    """The arm list.  GPU arms are probed, never emulated."""
    if quick:
        return [
            CoreConfig(shape=(32, 32), kernel="reference", dtype="float64"),
            CoreConfig(shape=(32, 32), kernel="fused", dtype="float64"),
        ], dict(steps=300, record_every=100, repeat=1)
    return [
        # primary: T3 matrix row-5 configuration, all four kernel/dtype combos
        CoreConfig(shape=(64, 64), kernel="reference", dtype="float64"),
        CoreConfig(shape=(64, 64), kernel="fused", dtype="float64"),
        CoreConfig(shape=(64, 64), kernel="fused_slices", dtype="float64"),
        CoreConfig(shape=(64, 64), kernel="reference", dtype="float32"),
        CoreConfig(shape=(64, 64), kernel="fused", dtype="float32"),
        CoreConfig(shape=(64, 64), kernel="fused_slices", dtype="float32"),
        # second back-end (declared dependency), same case
        CoreConfig(shape=(64, 64), kernel="reference", dtype="float64",
                   backend="torch-cpu"),
        CoreConfig(shape=(64, 64), kernel="fused", dtype="float64",
                   backend="torch-cpu"),
        CoreConfig(shape=(64, 64), kernel="fused_slices", dtype="float64",
                   backend="torch-cpu"),
        # resolution scaling point (float64, numpy only)
        CoreConfig(shape=(128, 128), kernel="reference", dtype="float64"),
        CoreConfig(shape=(128, 128), kernel="fused", dtype="float64"),
        CoreConfig(shape=(128, 128), kernel="fused_slices", dtype="float64"),
    ], dict(steps=600 if fast else 1200, record_every=100, repeat=1 if fast else 2)


def run_all(quick: bool = False, fast: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    arm_cfgs, proto = arms(quick, fast)

    probes = {k: v.as_dict() for k, v in probe_backends().items()}
    results, fields = [], {}

    # per (shape, dtype) reference field for the consistency gate
    baselines = {}
    for cfg in arm_cfgs:
        steps = proto["steps"] if cfg.shape == (64, 64) or quick else 600
        if cfg.shape == (128, 128) and not quick:
            steps = 300 if fast else 600
        key = (cfg.shape, cfg.dtype)
        if cfg.kernel == "reference":
            t0 = time.perf_counter()
            res, u_ref = run_arm(cfg, steps=steps, record_every=proto["record_every"],
                                 repeat=proto["repeat"])
            baselines[key] = u_ref
            results.append(res.as_dict())
            print(f"[arm] {cfg.ident():52s} walk={time.perf_counter()-t0:6.2f}s "
                  f"mlups={res.mlups:7.2f} rel_err={res.accuracy_rel_err:.3e} "
                  f"{res.verdict}")
        else:
            t0 = time.perf_counter()
            res, u_fin = run_arm(cfg, steps=steps, record_every=proto["record_every"],
                                 repeat=proto["repeat"],
                                 reference_field=baselines.get(key))
            results.append(res.as_dict())
            if cfg.shape == (64, 64) and cfg.backend == "numpy" \
                    and cfg.dtype == "float64" and cfg.kernel == "fused_slices":
                fields["field_fused_64"] = u_fin
            if cfg.shape == (128, 128):
                fields["field_fused_128"] = u_fin
            print(f"[arm] {cfg.ident():52s} walk={time.perf_counter()-t0:6.2f}s "
                  f"mlups={res.mlups:7.2f} rel_err={res.accuracy_rel_err:.3e} "
                  f"cons={res.consistency_rel_diff} {res.verdict}")

    # --- pairwise ratios (programmatic, only admissible arms count) -------
    def find(kernel, dtype, shape=(64, 64), backend="numpy"):
        for r in results:
            c = r["config"]
            if (c["kernel"], c["dtype"], tuple(c["shape"]), c["backend"]) == \
                    (kernel, dtype, tuple(shape), backend):
                return r
        return None

    ratios = {}
    for shape in ((64, 64), (128, 128)):
        ref = find("reference", "float64", shape)
        for kern in ("fused", "fused_slices"):
            fus = find(kern, "float64", shape)
            if not (ref and fus):
                continue
            ratios[f"{kern}_over_reference_mlups_{shape[0]}"] = (
                None if not (ref["perf_admissible"] and fus["perf_admissible"])
                else round(fus["mlups"] / ref["mlups"], 4))
            ratios[f"{kern}_over_reference_live_bytes_{shape[0]}"] = round(
                fus["declared_live_bytes"] / ref["declared_live_bytes"], 4)
            ratios[f"{kern}_over_reference_peak_bytes_{shape[0]}"] = (
                round(fus["measured_peak_bytes"] / ref["measured_peak_bytes"], 4))
    for kernel in ("reference", "fused", "fused_slices"):
        f64, f32 = find(kernel, "float64"), find(kernel, "float32")
        if f64 and f32:
            ok = f64["perf_admissible"] and f32["perf_admissible"]
            ratios[f"float32_over_float64_mlups_{kernel}"] = (
                round(f32["mlups"] / f64["mlups"], 4) if ok else None)
            ratios[f"float32_over_float64_peak_bytes_{kernel}"] = round(
                f32["measured_peak_bytes"] / f64["measured_peak_bytes"], 4)
            ratios[f"float32_accuracy_rel_err_{kernel}"] = f32["accuracy_rel_err"]

    summary = {
        "anchor": "GAP-LBM-T2-GPU-COREPATH",
        "case": {
            "t3_row": T3_CASE,
            "analytic": "amplitude ~ exp(-2 nu k^2 t), u0=0.01, nu=0.05",
            "protocol": proto,
            "timing": "min over repeat blocks after one warm-up block; single core",
        },
        "preregistered_gates": {
            "accuracy_tol": ACCURACY_TOL,
            "consistency_tol_float64": CONSISTENCY_TOL,
            "consistency_tol_by_dtype": DTYPE_CONSISTENCY_TOL,
            "rule": "perf number admissible iff accuracy AND consistency gates pass",
        },
        "backends": probes,
        "withheld": [
            {"backend": n, "kind": p["kind"],
             "reason": p["reason"],
             "grade": "WITHHELD",
             "note": "GPU core-path arm not executed: no device on this host; "
                     "no CPU fallback used as a stand-in"}
            for n, p in probes.items() if not p["available"]
        ],
        "machine": machine_context(),
        "arms": results,
        "ratios": ratios,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    render(summary, fields)
    print(f"\nwrote {OUT/'summary.json'}")
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def render(summary: dict, fields: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in summary["arms"]]
    labels = [f"{a['config']['kernel']}/{a['config']['dtype']}/"
              f"{a['config']['backend']}/{a['config']['shape'][0]}" for a in arms]

    # fig 1: field contour (T3 case, fused arm)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in (
            (axes[0], "field_fused_64", "fused, 64x64"),
            (axes[1], "field_fused_128", "fused, 128x128")):
        u = fields.get(key)
        if u is None:
            ax.set_axis_off()
            continue
        im = ax.imshow(u[0], origin="lower", cmap="RdBu_r")
        ax.set_title(f"u_x after run — {title}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("T3 case taylor-green-2d on the unified core path")
    fig.tight_layout()
    fig.savefig(OUT / "fig_field_contours.png", dpi=130)
    plt.close(fig)

    # fig 2: accuracy vs throughput trade-off (only admissible arms are filled)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for a, lab in zip(arms, labels):
        ok = a["perf_admissible"]
        ax.scatter(a["mlups"], 100 * a["accuracy_rel_err"], s=60,
                   facecolors=("C0" if ok else "none"),
                   edgecolors=("C0" if ok else "C3"), linewidths=1.4)
        ax.annotate(lab, (a["mlups"], 100 * a["accuracy_rel_err"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6.5)
    ax.axhline(100 * summary["preregistered_gates"]["accuracy_tol"], color="k",
               ls="--", lw=1)
    ax.text(0.02, 0.98,
            "dashed = pre-registered gate (fails => no MLUPS published)",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.set_xlabel("throughput [MLUPS]")
    ax.set_ylabel("decay-rate relative error [%]")
    ax.set_title("accuracy / performance trade-off (filled = admissible)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tradeoff.png", dpi=130)
    plt.close(fig)

    # fig 3: memory inventory vs measured peak
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    idx = np.arange(len(arms))
    ax.bar(idx - 0.2, [a["declared_live_bytes"] / 1024 for a in arms], width=0.4,
           label="declared live state [KiB]")
    ax.bar(idx + 0.2, [a["measured_peak_bytes"] / 1024 for a in arms], width=0.4,
           label="measured tracemalloc peak [KiB]")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("KiB")
    ax.set_title("memory management: fused vs reference kernel")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_memory.png", dpi=130)
    plt.close(fig)

    # fig 4: fitted decay vs analytic for every arm
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(arms))
    ax.bar(x - 0.2, [a["decay_rate_analytic"] for a in arms], width=0.4,
           label="analytic 2 nu k^2")
    ax.bar(x + 0.2, [a["decay_rate_fit"] for a in arms], width=0.4,
           label="fitted from run")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("decay rate [1/step]")
    ax.set_title("T3 case gate: fitted decay rate vs analytic")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_decay.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="reduced step counts so a single-core run fits a short slot")
    a = ap.parse_args()
    run_all(quick=a.quick, fast=a.fast)