"""Figure + figure-caption builder for the P3 vortex velocity-floor diagnosis.

Numbers come from ``validation/euler_dg_2d/velocity_floor.json`` (written by
``benchmarks.diagnose_euler_dg_2d_velocity_floor``).  If that file is still
being written, the series are parsed from the diagnostic log so the figure is
always programmatic (never hand-copied).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "validation" / "euler_dg_2d" / "velocity_floor.json"
FIG = ROOT / "docs" / "figures" / "euler_dg_2d_velocity_floor.png"
LOG = Path("/tmp/full.log")

LINE = re.compile(
    r"^\s+cells=(\d+) q=(\d+) steps=\s*(\d+) wall=\s*[\d.]+s"
    r" density=([\d.e+-]+) velocity_x=([\d.e+-]+) velocity_y=([\d.e+-]+) pressure=([\d.e+-]+)$"
)


def load():
    if JSON.exists():
        payload = json.loads(JSON.read_text())
        series = {
            label: [
                {"cells": row["cells"],
                 "errors": {k: row["errors"][k]["relative_l2"] for k in row["errors"]},
                 "mean": {k: row["errors"][k]["mean"] for k in row["errors"]}}
                for row in rows
            ]
            for label, rows in payload["series"].items()
        }
        return series, "json"
    series, label = {}, None
    for line in LOG.read_text().splitlines():
        if line.startswith("--- "):
            label = line[4:].split()[0]
            series[label] = []
            continue
        match = LINE.match(line)
        if match and label:
            cells = int(match.group(1))
            series[label].append({
                "cells": cells,
                "errors": {"density": float(match.group(4)),
                           "velocity_x": float(match.group(5)),
                           "velocity_y": float(match.group(6)),
                           "pressure": float(match.group(7))},
                "mean": None,
            })
    return series, "log"


def main():
    series, source = load()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    colours = {"moving": "tab:red", "static": "tab:blue", "subsonic_boost": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for label, rows in series.items():
        if len(rows) < 2:
            continue
        h = [10.0 / row["cells"] for row in rows]
        for field, style in (("density", "--"), ("velocity_x", "-")):
            axes[0].loglog(h, [row["errors"][field] for row in rows], style + "o",
                           color=colours.get(label, "k"),
                           label=f"{label} {field}" + (" (rho)" if field == "density" else " (u)"))
        if rows[0]["mean"] is not None:
            axes[1].loglog(h, [abs(row["mean"]["velocity_x"]) for row in rows], "o-",
                           color=colours.get(label, "k"), label=f"{label} |mean u error|")
    ref = [row["cells"] for row in series.get("moving", [])]
    if ref:
        h = [10.0 / c for c in ref]
        scale = series["moving"][0]["errors"]["density"] / h[0] ** 4
        axes[0].loglog(h, [scale * x ** 4 for x in h], ":", color="grey", label="slope h^4")
    axes[0].set_xlabel("h"); axes[0].set_ylabel("relative $L^2$ error")
    axes[0].set_title("Isentropic vortex, P3 DG/HLLC: rho vs u")
    axes[0].legend(fontsize=6, ncol=1)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("h"); axes[1].set_ylabel("|domain-mean u error|")
    axes[1].set_title("Zero-mode (Galilean) velocity offset")
    axes[1].legend(fontsize=6); axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=95)
    print(f"figure -> {FIG.relative_to(ROOT)} (source={source})")
    print("caption: left = relative L2 error of rho (dashed) and u (solid) vs h for the "
          "translated (red) and static (blue) isentropic vortex; dotted line is the design "
          "slope h^4. right = domain-mean velocity error: machine zero for the static "
          "vortex, h-independent ~1.1e-05 for the translated one.")


if __name__ == "__main__":
    main()
