#!/usr/bin/env python3
"""Plot the dry-bank shoreline rest-state artifact.

Reads ``validation/swe_thacker/shoreline_rest_state_results.json`` (written by
``run_shoreline_rest.py``) and draws

1. the free surface ``eta(x)`` of the N = 64 runs together with the bed and the
   analytic still-water level ``eta0`` -- the fixed filter holds a flat line,
   the pre-fix filter leaves a jagged, non-constant surface;
2. the spurious velocity ``max|u|`` before/after the fix on a log scale.

No SWE solve happens here, so the figure cannot disagree with the artifact.

Usage::

    python validation/swe_thacker/plot_shoreline_rest.py \
        [--results <json>] [--out docs/figures/swe_shoreline_rest.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation" / "swe_thacker" / "shoreline_rest_state_results.json"
OUT = ROOT / "docs" / "figures" / "swe_shoreline_rest.png"

TITLES = {
    "parabolic_hump": r"$z_b = 1.5 - 0.5x^2$: shoreline inside a cell",
    "tent_on_face": r"$z_b = |x|/1.5$: shoreline on a cell face",
}


def _bed(name: str, x):
    return [1.5 - 0.5 * v ** 2 for v in x] if name == "parabolic_hump" else \
        [abs(v) / 1.5 for v in x]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    data = json.loads(args.results.read_text())
    eta0 = data["exact_solution"]["eta"]
    cases = {case["name"]: case for case in data["cases"]}
    legacy = data["legacy_filter_contrast"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, (name, case) in zip(axes[:2], cases.items()):
        fixed = next(r for r in case["runs"]
                     if r["ncells"] == 64 and r["degree"] == 1)
        old = next(r for r in legacy
                   if r["case"] == name and r["ncells"] == 64 and r["degree"] == 1)
        x = fixed["x"]
        ax.plot(x, _bed(name, x), color="0.35", lw=1.2, label="bed $z_b$")
        ax.axhline(eta0, color="k", ls=":", lw=1.0, label=r"analytic $\eta_0$")
        ax.plot(x, fixed["eta_profile"], color="tab:blue", lw=1.4,
                label="fixed filter ($\\theta$-scaling)")
        ax.plot(x, old["eta_profile"], color="tab:red", lw=1.0, ls="--",
                label="pre-fix filter (flatten to average)")
        ax.set_title(TITLES[name] + "\nP1, N = 64, t = 1 s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel(r"$\eta = h + z_b$ [m]")
        ax.set_ylim(eta0 - 0.35, eta0 + 0.6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower center")

    ax = axes[2]
    labels, values_fixed, values_legacy = [], [], []
    for name, case in cases.items():
        for run in case["runs"]:
            key = f"{name.split('_')[0]}\nP{run['degree']} N{run['ncells']}"
            labels.append(key)
            values_fixed.append(max(run["u_max"], 1e-16))
            match = next(r for r in legacy if r["case"] == name
                         and r["degree"] == run["degree"]
                         and r["ncells"] == run["ncells"]) if run["ncells"] == 64 else None
            values_legacy.append(max(match["u_max"], 1e-16) if match else float("nan"))
    idx = range(len(labels))
    ax.bar([i - 0.2 for i in idx], values_fixed, width=0.4, color="tab:blue",
           label="fixed")
    ax.bar([i + 0.2 for i in idx], values_legacy, width=0.4, color="tab:red",
           label="pre-fix (N = 64 only)")
    ax.set_yscale("log")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.axhline(1e-8, color="k", ls=":", lw=1.0, label="pass criterion 1e-8 m/s")
    ax.set_ylabel(r"spurious $\max|u|$ [m/s]")
    ax.set_title("rest state: spurious current")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    fig.suptitle("Still lake with a dry bank: positivity filter before/after "
                 f"(git {data['provenance']['git_commit_short']})", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())