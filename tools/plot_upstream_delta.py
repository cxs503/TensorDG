#!/usr/bin/env python3
"""Render the upstream-delta figure from the audit artifact (no hand-typed numbers)."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THEME_COLORS = {
    "physics_core": "#c0392b", "verification": "#e67e22", "benchmarks": "#2980b9",
    "tests": "#27ae60", "docs": "#7f8c8d", "artifacts": "#8e44ad",
    "ci": "#16a085", "api": "#d35400", "packaging": "#2c3e50", "other": "#95a5a6",
    "none": "#bdc3c7",
}
LEVEL_COLORS = {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "INFO": "#7f8c8d"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()
    d = json.load(open(ns.artifact, encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    fig.suptitle("TensorDG upstream delta @ %s  (base %s, identical=%s)"
                 % (d["head"]["sha"][:7], d["delta"]["base_ref"],
                    d["delta"]["identical"]), fontsize=11)

    # --- panel A: churn per recent commit, coloured by theme -------------
    ax = axes[0]
    recs = [r for r in d["recent_commits"] if not r["merge"]]
    recs = list(reversed(recs))                      # oldest at bottom
    labels, vals, colors = [], [], []
    for r in recs:
        theme = d["theme_evolution"]["per_commit"][
            [p["short"] for p in d["theme_evolution"]["per_commit"]].index(r["short"])]["theme"]
        # ASCII only: the container has no CJK font, commit subjects are Chinese
        labels.append("%s [%s]" % (r["short"], theme))
        vals.append(r["churn"])
        colors.append(THEME_COLORS.get(theme, "#95a5a6"))
    y = np.arange(len(recs))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("changed lines (insertions+deletions)")
    ax.set_title("A. non-merge commit churn, %d commits" % len(recs), fontsize=9)
    for yi, v in zip(y, vals):
        ax.text(v * 1.02, yi, str(v), va="center", fontsize=6.5)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=THEME_COLORS[t])
                       for t in sorted({l.split("[")[1].rstrip("]") for l in labels})],
              labels=sorted({l.split("[")[1].rstrip("]") for l in labels}),
              fontsize=6, loc="lower right", frameon=False)

    # --- panel B: churn by theme + unmerged-branch bar -------------------
    ax = axes[1]
    churn = d["theme_evolution"]["churn_by_theme"]
    names = list(churn)
    cols = [THEME_COLORS.get(n, "#95a5a6") for n in names]
    x = np.arange(len(names) + 1)
    values = list(churn.values())
    br = d["unmerged_upstream_branches"]
    values.append(sum(b["insertions"] for b in br))
    names.append("UNMERGED\n(upstream)")
    cols.append("#111111")
    ax.bar(x, values, color=cols, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("changed lines")
    ax.set_title("B. churn by theme (last %d commits) vs unmerged work"
                 % d["acceptance"]["recent_commit_count"], fontsize=9)
    for xi, v in zip(x, values):
        ax.text(xi, v * 1.01, str(v), ha="center", fontsize=6.5)

    # --- panel C: risk tally by level ------------------------------------
    ax = axes[2]
    tally = d["risk_tally_by_level"]
    levels = list(tally)
    ax.bar(np.arange(len(levels)), list(tally.values()),
           color=[LEVEL_COLORS.get(l, "#95a5a6") for l in levels],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("risk tags on recent commits")
    ax.set_title("C. risk annotations by level", fontsize=9)
    for i, v in enumerate(tally.values()):
        ax.text(i, v + 0.05, str(v), ha="center", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(ns.out) or ".", exist_ok=True)
    fig.savefig(ns.out, dpi=140)
    print("wrote", ns.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())