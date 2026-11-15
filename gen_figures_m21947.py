#!/usr/bin/env python3
"""Generate figures for CASE-DG-EULER-VORTEX-ERR report from existing metrics.

Produces:
  fig_error_vs_time.png       - Error growth over time for all families
  fig_error_conservation.png  - Error amplification factor A(t) = e(t)/e(0)
  fig_spatial_order.png       - Spatial convergence order at t=1 and t=10
  fig_error_field.png         - Error field visualization (conceptual)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = Path("case_euler_vortex_errconserve/metrics.json")
OUT = Path("case_euler_vortex_errconserve")

# Color palette for families
FAMILY_COLORS = {
    "dg_d1_n32_cfl0.2": "#1f77b4",
    "dg_d1_n64_cfl0.2": "#aec7e8",
    "dg_d2_n32_cfl0.2": "#2ca02c",
    "dg_d2_n32_cfl0.2_diraxis0": "#98df8a",
    "dg_d2_n32_cfl0.2_dirobl30": "#ff7f0e",
    "dg_d2_n32_cfl0.4": "#ffbb78",
    "dg_d2_n64_cfl0.2": "#d62728",
    "dg_d3_n32_cfl0.2": "#ff9896",
    "p0_n64": "#9467bd",
    "p0_n128": "#c5b0d5",
    "p0_n256": "#8c564b",
}

FAMILY_LABELS = {
    "dg_d1_n32_cfl0.2": "DG-P1/32²",
    "dg_d1_n64_cfl0.2": "DG-P1/64²",
    "dg_d2_n32_cfl0.2": "DG-P2/32²",
    "dg_d2_n32_cfl0.2_diraxis0": "DG-P2/32² (axis)",
    "dg_d2_n32_cfl0.2_dirobl30": "DG-P2/32² (30°)",
    "dg_d2_n32_cfl0.4": "DG-P2/32² CFL=0.4",
    "dg_d2_n64_cfl0.2": "DG-P2/64²",
    "dg_d3_n32_cfl0.2": "DG-P3/32²",
    "p0_n64": "P0/64²",
    "p0_n128": "P0/128²",
    "p0_n256": "P0/256²",
}


def load_data():
    m = json.loads(METRICS.read_text())
    records = m.get("records", [])
    gates = m.get("gates", {})
    return records, gates


def fig_error_vs_time(records):
    """Figure 1: Max field relative L2 error vs time for all families."""
    families = {}
    for r in records:
        fam = r.get("family", r["task_key"])
        families.setdefault(fam, []).append(r)

    fig, ax = plt.subplots(figsize=(10, 6))

    for fam, recs in sorted(families.items()):
        recs.sort(key=lambda r: r["final_time"])
        times = [r["final_time"] for r in recs]
        errors = [r["max_percent"] for r in recs]
        color = FAMILY_COLORS.get(fam, "#333333")
        label = FAMILY_LABELS.get(fam, fam)
        ax.plot(times, errors, "o-", color=color, label=label, linewidth=1.5, markersize=5)

    # 6% tolerance line
    ax.axhline(y=6.0, color="red", linestyle="--", linewidth=1.5, label="6% tolerance (Yee et al. 2000)")

    ax.set_xlabel("Transport time t (box crossings)", fontsize=12)
    ax.set_ylabel("Max field relative L₂ error (%)", fontsize=12)
    ax.set_title("CASE-DG-EULER-VORTEX-ERR: Error Growth vs Transport Time", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig_error_vs_time_m21947.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def fig_amplification(records):
    """Figure 2: Error amplification factor A(t) = e(t)/e(0)."""
    # Get projection baseline
    baseline_path = OUT / "projection_baseline.json"
    if not baseline_path.exists():
        print("  No projection baseline, skipping amplification figure")
        return None

    baseline = json.loads(baseline_path.read_text())

    families = {}
    for r in records:
        fam = r.get("family", r["task_key"])
        families.setdefault(fam, []).append(r)

    fig, ax = plt.subplots(figsize=(10, 6))

    for fam, recs in sorted(families.items()):
        recs.sort(key=lambda r: r["final_time"])
        # Find matching baseline
        task = recs[0].get("task", "dg")
        degree = recs[0].get("degree", 0)
        cells = recs[0].get("cells", 32)
        if task == "dg":
            bkey = f"dg_d{degree}_n{cells}"
        else:
            bkey = f"p0_n{cells}"

        # Find baseline error
        b_err = None
        if isinstance(baseline, list):
            for b in baseline:
                if b.get("task_key") == bkey:
                    b_err = b.get("max_relative_l2")
                    break
        elif isinstance(baseline, dict):
            b_err = baseline.get("max_relative_l2")

        if b_err is None or b_err <= 0:
            continue

        times = [r["final_time"] for r in recs if r["final_time"] > 0]
        amps = [r["max_percent"] / (100.0 * b_err) for r in recs if r["final_time"] > 0]

        color = FAMILY_COLORS.get(fam, "#333333")
        label = FAMILY_LABELS.get(fam, fam)
        ax.plot(times, amps, "o-", color=color, label=label, linewidth=1.5, markersize=5)

    # A=1 line (no amplification)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="A=1 (no amplification)")

    ax.set_xlabel("Transport time t", fontsize=12)
    ax.set_ylabel("Amplification factor A(t) = e(t)/e(0)", fontsize=12)
    ax.set_title("Error Amplification Factor: A(t) = e(t)/e(0)", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig_amplification_m21947.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def fig_spatial_order(records, gates):
    """Figure 3: Spatial convergence order at t=1 and t=10."""
    # Extract order data from gates
    c9 = gates.get("C9_design_order", {})
    checkpoints = c9.get("checkpoints", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, t_label in zip(axes, ["t=1", "t=10"]):
        t_data = [c for c in checkpoints if t_label.replace("=", "_") in c.get("label", "").lower()
                  or (t_label == "t=1" and "t1" in c.get("label", "").lower())
                  or (t_label == "t=10" and "t10" in c.get("label", "").lower())]

        if not t_data:
            # Try matching by looking at the label
            t_data = [c for c in checkpoints if f"_{t_label.replace('=', '')}" in c.get("label", "").lower()
                      or f"t{t_label.split('=')[1]}" in c.get("label", "").lower()]

        degrees = [c.get("degree", c.get("p", 0)) for c in t_data]
        orders = [c.get("observed_order", c.get("order", 0)) for c in t_data]
        targets = [c.get("target_order", c.get("target", 0)) for c in t_data]

        x = np.arange(len(degrees))
        width = 0.35

        bars1 = ax.bar(x - width/2, orders, width, label="Observed order", color="#2ca02c")
        bars2 = ax.bar(x + width/2, targets, width, label="Target (0.8×(p+1))", color="#ff7f0e")

        ax.set_xticks(x)
        ax.set_xticklabels([f"P{d}" for d in degrees])
        ax.set_ylabel("Convergence order", fontsize=11)
        ax.set_title(f"Spatial Order at {t_label}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("CASE-DG-EULER-VORTEX-ERR: Spatial Convergence Order", fontsize=13, y=1.02)
    fig.tight_layout()
    path = OUT / "fig_spatial_order_m21947.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def fig_error_distribution(records):
    """Figure 4: Error distribution by field at design checkpoint."""
    # Get the design checkpoint: dg_d2_n32 at t=1
    design_rec = None
    for r in records:
        if r.get("task_key") == "dg_d2_n32_t1_cfl0.2":
            design_rec = r
            break

    if design_rec is None:
        print("  No design checkpoint record, skipping distribution figure")
        return None

    errors = design_rec["errors"]
    fields = ["density", "velocity_x", "velocity_y", "pressure"]
    values = [errors[f] * 100 for f in fields]  # convert to %

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(fields, values, color=colors, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}%", ha="center", va="bottom", fontsize=10)

    ax.axhline(y=6.0, color="red", linestyle="--", linewidth=1.5, label="6% tolerance")
    ax.set_ylabel("Relative L₂ error (%)", fontsize=11)
    ax.set_title("DG-P2/32² @ t=1: Error by Field", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = OUT / "fig_error_distribution_m21947.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def main():
    print("Loading metrics...")
    records, gates = load_data()
    print(f"  {len(records)} records loaded")

    print("\nGenerating figures...")
    fig_error_vs_time(records)
    fig_amplification(records)
    fig_spatial_order(records, gates)
    fig_error_distribution(records)

    print("\nDone!")


if __name__ == "__main__":
    main()
