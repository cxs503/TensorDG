"""Report/figures for the 2-D limiter design-order study.

Reads ``validation/limiter_design_order_2d/sweep_P{1,2}.json`` (written by
``benchmarks.limiter_design_order_2d``), merges them, prints the markdown tables
used in ``docs/limiter_design_order_2d.md`` and draws

* ``docs/figures/limiter_design_order_2d_accuracy.png``  -- log-log error vs h
  with the design-order reference slopes,
* ``docs/figures/limiter_design_order_2d_activity.png``  -- flagged-cell
  fraction and observed order versus the TVB constant ``M``.

Run::

    PYTHONPATH=. python -m benchmarks.report_limiter_design_order_2d
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "validation" / "limiter_design_order_2d"
FIG = ROOT / "docs" / "figures"

COLOURS = {
    "none": "#111111",
    "tvb_M1": "#d62728",
    "tvb_M10": "#ff7f0e",
    "tvb_M100": "#2ca02c",
    "weno_M1": "#9467bd",
    "weno_M100": "#17becf",
}
MARKERS = {"none": "o", "tvb_M1": "s", "tvb_M10": "^", "tvb_M100": "v", "weno_M1": "D", "weno_M100": "P"}


def load():
    payloads = []
    for degree in (1, 2):
        path = VAL / f"sweep_P{degree}.json"
        if path.exists():
            payloads.append(json.loads(path.read_text()))
    runs = [row for payload in payloads for row in payload["runs"]]
    scale = payloads[0]["second_derivative_scale"] if payloads else {}
    setup = {key: payloads[0][key] for key in ("final_time", "cfl", "domain", "strength")} if payloads else {}
    return runs, scale, setup, payloads


def table_by_degree(runs, degree, field="density"):
    configs = []
    for row in runs:
        if row["degree"] == degree and row["config"] not in configs:
            configs.append(row["config"])
    order = [name for name in COLOURS if name in configs]
    meshes = sorted({row["cells"] for row in runs if row["degree"] == degree})
    lines = [
        "| config | " + " | ".join(f"{n}$^2$ rel $L^2$ [{0.0 if False else ''}%]" .replace("[]", "") for n in meshes) + " | EOC (last pair) | flagged mean |",
        "|---" * (len(meshes) + 3) + "|",
    ]
    for config in order:
        cells = []
        for mesh in meshes:
            match = [r for r in runs if r["degree"] == degree and r["config"] == config and r["cells"] == mesh]
            if not match:
                cells.append("&mdash;")
                continue
            row = match[0]
            value = 100.0 * row["errors"][field]
            flagged = f" ({row['flagged_mean'] * 100:.2f} % flagged)" if row["flagged_mean"] else ""
            cells.append(f"{value:.3e}{flagged}")
        match = [r for r in runs if r["degree"] == degree and r["config"] == config and r["cells"] == meshes[-1]]
        last_eoc = match[0]["eoc"].get(field) if match else None
        eoc_text = f"{last_eoc:.2f}" if last_eoc is not None else "&mdash;"
        mean_flag = match[0]["flagged_mean"] * 100.0 if match else 0.0
        lines.append(f"| {config} | " + " | ".join(cells) + f" | {eoc_text} | {mean_flag:.2f} % |")
    return "\n".join(lines)


def figure_accuracy(runs):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for axis, degree in zip(axes, (1, 2)):
        for config in COLOURS:
            group = sorted(
                (r for r in runs if r["degree"] == degree and r["config"] == config),
                key=lambda r: r["cells"],
            )
            if not group:
                continue
            h = [r["h"] for r in group]
            error = [r["errors"]["density"] for r in group]
            axis.loglog(h, error, marker=MARKERS[config], color=COLOURS[config], label=config,
                        linewidth=1.4, markersize=5)
        h_ref = [0.7, 0.2]
        first = [
            r for r in runs
            if r["degree"] == degree and r["config"] == "none"
        ]
        if first:
            anchor = min(first, key=lambda r: abs(r["h"] - 0.7))
            order = degree + 1
            axis.loglog(
                h_ref,
                [anchor["errors"]["density"] * (h / anchor["h"]) ** order for h in h_ref],
                "k--", linewidth=1.0, alpha=0.6, label=f"slope {order} (design)",
            )
        axis.set_title(f"P{degree}: analytic isentropic vortex")
        axis.set_xlabel("h = 10 / N")
        axis.set_ylabel(r"relative $L^2$ error of $\rho$")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("2-D DG on smooth flow: the TVB constant caps the observed order")
    fig.tight_layout()
    path = FIG / "limiter_design_order_2d_accuracy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_activity(runs, scale):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axis = axes[0]

    def finest_per_M(degree, family="tvb"):
        # keep the finest mesh for every M so the curve is one point per M
        chosen = {}
        for row in runs:
            if row["degree"] != degree or not row["config"].startswith(family):
                continue
            m = float(row["config"].split("_M")[1])
            if m not in chosen or row["cells"] > chosen[m]["cells"]:
                chosen[m] = row
        return [chosen[m] for m in sorted(chosen)]

    # measured flagged fractions come from the sweep itself
    for degree, style in ((1, "o-"), (2, "s-")):
        points = finest_per_M(degree)
        if not points:
            continue
        m = [float(r["config"].split("_M")[1]) for r in points]
        f = [100.0 * r["flagged_mean"] for r in points]
        mesh = points[-1]["cells"]
        axis.semilogx(m, f, style, label=f"P{degree} ({mesh}$^2$), flagged cells")
    axis.axvline(scale.get("max_over_conserved", float("nan")), color="k", linestyle="--",
                 alpha=0.6, label=r"analytic $\max|u''|$")
    axis.set_xlabel("TVB constant M")
    axis.set_ylabel("flagged cells [%] (mean over calls)")
    axis.set_title("Limiter activity vs M (h fixed per degree)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1]
    for degree, style in ((1, "o-"), (2, "s-")):
        points = finest_per_M(degree)
        base = sorted((r for r in runs if r["degree"] == degree and r["config"] == "none"),
                      key=lambda r: r["cells"])
        base_eoc = base[-1]["eoc"]["density"] if base else None
        if not points:
            continue
        m, observed = [], []
        for row in points:
            if row["eoc"]["density"] is None:
                continue
            m.append(float(row["config"].split("_M")[1]))
            observed.append(row["eoc"]["density"])
        axis.semilogx(m, observed, style, label=f"P{degree} observed order")
        axis.axhline(degree + 1, color=COLOURS.get(f"tvb_M1", "k"), linestyle=":",
                     alpha=0.5, linewidth=1.0)
        if base_eoc:
            axis.axhline(base_eoc, color="k", linestyle="--", alpha=0.6, linewidth=0.9,
                         label=f"P{degree} unlimited ({base_eoc:.2f})" if degree == 1 else None)
    axis.set_xlabel("TVB constant M")
    axis.set_ylabel("observed order of $\\rho$ error")
    axis.set_title("Accuracy price of an active limiter")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "limiter_design_order_2d_activity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    runs, scale, setup, payloads = load()
    if not runs:
        raise SystemExit("no sweep data; run benchmarks.limiter_design_order_2d first")
    print("setup:", json.dumps(setup))
    print("second-derivative scale:", json.dumps(scale, indent=2))
    for degree in (1, 2):
        print(f"\n### P{degree} density error\n")
        print(table_by_degree(runs, degree))
    print("\nfigures:", figure_accuracy(runs), figure_activity(runs, scale))
    merged = {
        "timestamp": payloads[0]["timestamp"],
        "setup": setup,
        "second_derivative_scale": scale,
        "runs": runs,
    }
    (VAL / "sweep.json").write_text(json.dumps(merged, indent=2))
    print("merged ->", VAL / "sweep.json")


if __name__ == "__main__":
    main()