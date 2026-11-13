#!/usr/bin/env python3
"""Render the figures of the m17042 upstream-diff patrol report.

Inputs are the machine-readable censuses produced by
``tools/patrol_upstream_diff.py --write``; nothing here recomputes git state, so
the figures cannot drift from the JSON that the report quotes (every number in
the report is injected from these JSON files).

    python tools/patrol_upstream_diff.py --write --out artifacts/patrol_upstream_diff
    python tools/patrol_upstream_diff.py --limit 60 --no-branches \
        --write --out artifacts/patrol_upstream_diff/window60
    python tools/render_patrol_figures.py

Figures land in ``docs/figures/patrol_m17042/`` -- that directory is tracked in this
repository, whereas binaries under ``validation/**/figures/`` are the ones the merge
gate drops (see risk code R8 in the patrol report).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "patrol_upstream_diff"
FIG = ROOT / "docs" / "figures" / "patrol_m17042"

THEME_ORDER = ["core-solver", "tests", "benchmarks", "validation-artifacts", "tooling", "docs", "scratch", "run-artifacts", "other", "loose-scripts"]
THEME_COLOR = {
    "core-solver": "#1f4e79",
    "tests": "#2e8b57",
    "benchmarks": "#4682b4",
    "validation-artifacts": "#d4a017",
    "tooling": "#8f6fbf",
    "docs": "#c0504d",
    "scratch": "#7f7f7f",
    "run-artifacts": "#e08a3c",
    "other": "#bdbdbd",
    "loose-scripts": "#9e9e9e",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fig_landing_themes(window15: dict, window60: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax = axes[0]
    commits = window15["commits"]
    labels = [f"#{c['pr']}\n{c['mission']}" if c["mission"] else c["short"] for c in commits]
    bottoms = [0] * len(commits)
    x = range(len(commits))
    for theme in THEME_ORDER:
        vals = [c["themes"].get(theme, 0) for c in commits]
        if not any(vals):
            continue
        ax.bar(x, vals, bottom=bottoms, label=theme, color=THEME_COLOR[theme], width=0.72)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    empties = [i for i, c in enumerate(commits) if "R3-CONTENT-EMPTY-LANDING" in c["risks"]]
    for i in empties:
        ax.annotate(
            "content-empty\nlanding (0 files)",
            xy=(i, 0.4),
            xytext=(i, max(bottoms or [1]) * 0.55),
            ha="center",
            fontsize=8,
            color="#b00020",
            arrowprops=dict(arrowstyle="->", color="#b00020", lw=1.0),
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("files in landing (first-parent diff)")
    ax.set_title(f"TensorDG landings: {len(commits)} first-parent commits on {window15['upstream']}\nfiles per landing, split by top-level theme")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    s60 = window60["summary"]
    plumbing_files = s60["files_touched"] - s60["plumbing_read_blind_files_hidden"]
    bars = ax.bar(
        ["plumbing reading\n(diff-tree, no -m)", "first-parent reading\n(git diff <sha>^1 <sha>)"],
        [plumbing_files, s60["files_touched"]],
        color=["#b0b0b0", "#1f4e79"],
        width=0.55,
    )
    for rect, value in zip(bars, [plumbing_files, s60["files_touched"]]):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 8, str(value), ha="center", fontsize=9)
    ax.set_ylabel(f"files reported over {s60['commits']} commits")
    ax.set_title(
        "Baseline contrast: the plumbing form hides every true merge\n"
        f"{s60['plumbing_read_blind_commits']} merges / {s60['plumbing_read_blind_files_hidden']} files invisible "
        f"(window: last {s60['commits']} first-parent commits)"
    )
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / "fig_landing_themes.png", dpi=150)
    plt.close(fig)


def fig_branch_zoo(window15: dict, out: Path) -> None:
    branches = window15["branches"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.0, 1.45]})

    ax = axes[0]
    states = branches["states"]
    order = ["MERGED_REF", "CONTENT_LANDED", "NO_DELTA", "DRIFTED", "LOST_FILES", "NO_MERGE_BASE"]
    labels = [s for s in order if s in states]
    colors = {"MERGED_REF": "#7f7f7f", "CONTENT_LANDED": "#2e8b57", "NO_DELTA": "#bdbdbd",
              "DRIFTED": "#d4a017", "LOST_FILES": "#b00020", "NO_MERGE_BASE": "#8f6fbf"}
    vals = [states[s] for s in labels]
    bars = ax.bar(labels, vals, color=[colors[s] for s in labels], width=0.6)
    for rect, value in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 1.5, str(value), ha="center", fontsize=9)
    ax.set_ylabel("local branches")
    ax.set_title(
        f"Branch audit: {branches['branches_scanned']} local branches vs upstream\n"
        f"{states.get('LOST_FILES', 0)} carry files that never reached upstream "
        f"({branches['absent_file_hits']} file hits)"
    )
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    detail = window15["lost_referenced"]["detail"]
    ranked = sorted(
        ((p, d) for p, d in detail.items() if d.get("consumer")),
        key=lambda kv: -sum(1 for r in branches["rows"] if kv[0] in r["absent_paths"]),
    )[:12]
    path_labels = [p for p, _ in ranked][::-1]
    counts = [sum(1 for r in branches["rows"] if p in r["absent_paths"]) for p, _ in ranked][::-1]
    ax.barh(range(len(path_labels)), counts, color="#b00020", height=0.62)
    ax.set_yticks(range(len(path_labels)))
    ax.set_yticklabels([p if len(p) < 46 else "…" + p[-44:] for p in path_labels], fontsize=7.5)
    ax.set_xlabel("branches holding the file")
    ax.set_title(
        "Absent from upstream but still consumed by upstream docs/tests\n"
        f"{window15['lost_referenced']['consumer_referenced_count']} consumer-referenced paths of "
        f"{window15['lost_referenced']['scanned']} absent paths"
    )
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / "fig_branch_zoo.png", dpi=150)
    plt.close(fig)


def main() -> int:
    window15 = load(ART / "patrol.json")
    window60 = load(ART / "window60" / "patrol.json")
    FIG.mkdir(parents=True, exist_ok=True)
    fig_landing_themes(window15, window60, FIG)
    fig_branch_zoo(window15, FIG)
    for name in ("fig_landing_themes.png", "fig_branch_zoo.png"):
        print(f"wrote {FIG / name} ({(FIG / name).stat().st_size} bytes)")
    # print the numbers the report quotes, so prose can be injected verbatim
    print(json.dumps({
        "window15": window15["summary"],
        "window60": {k: window60["summary"][k] for k in
                     ("commits", "content_empty_landings", "plumbing_read_blind_commits",
                      "plumbing_read_blind_files_hidden", "files_touched")},
        "branches": {k: window15["branches"][k] for k in ("branches_scanned", "states", "absent_file_hits")},
        "lost_referenced": {k: window15["lost_referenced"][k] for k in
                            ("scanned", "referenced_count", "consumer_referenced_count", "by_category")},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())