"""Aggregate the G3 resolution ladder into a table, a CSV and a figure.

Reads ``validation/forward_step/ladder/task*.json`` (written by
``wall_reflection_ladder.py``) and emits

* ``ladder.csv``  -- one row per run, every number read from the JSONs;
* ``ladder.md``   -- the same table in Markdown;
* ``wall_reflection_ladder.png`` -- (a) plateau error vs ``dx`` on log-log axes
  per polynomial degree with the 3 % criterion drawn in, (b) the near-wall
  pressure/density profile of the finest run against the exact two-shock
  solution.

Nothing in the figure or the table is typed by hand.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE / "ladder"
CRITERION = 3.0  # percent, the pre-registered G3 tolerance


def load():
    rows = []
    for path in sorted(LADDER.glob("task*.json")):
        d = json.loads(path.read_text())
        d["_path"] = path.name
        rows.append(d)
    return rows


def table(rows):
    head = ("task", "route", "nx", "p", "limiter", "M", "dx", "steps",
            "plateau_cells", "plateau_width_cells", "rho_err_%", "p_err_%",
            "worst_%", "l1_rho", "linf_rho", "l1_p", "linf_p", "wall_face")
    out = [head]
    for r in rows:
        def g(key, fmt="{:.4g}"):
            v = r.get(key)
            return "" if v is None else (fmt.format(v) if isinstance(v, float) else str(v))
        out.append((r["task"], r["route"], r["ncells"][0], r["degree"],
                    str(r["limiter"]), r.get("tvb_constant"), g("dx"),
                    r["steps"], r.get("plateau_cells"),
                    g("plateau_width_cells", "{:.3g}"),
                    g("plateau_rho_error_percent", "{:.3f}"),
                    g("plateau_p_error_percent", "{:.3f}"),
                    g("worst_cell_error_percent", "{:.3f}"),
                    g("l1_density"), g("linf_density"), g("l1_pressure"),
                    g("linf_pressure"), g("wall_face_actual", "{:.6g}")))
    return out


def write_csv(tab, path):
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows(tab)


def write_md(tab, path):
    lines = ["| " + " | ".join(str(c) for c in tab[0]) + " |",
             "|" + "---|" * len(tab[0])]
    for row in tab[1:]:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    path.write_text("\n".join(lines) + "\n")


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with_err = [r for r in rows if r.get("worst_cell_error_percent") is not None]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for degree, marker in ((1, "o"), (2, "s")):
        pts = sorted([r for r in with_err if r["degree"] == degree and r["route"] == "embedded"],
                     key=lambda r: r["dx"])
        if not pts:
            continue
        xs = [r["dx"] for r in pts]
        ys = [abs(r["worst_cell_error_percent"]) for r in pts]
        ax.loglog(xs, ys, marker + "-", label=f"embedded wall, P{degree}")
        for r, x, yv in zip(pts, xs, ys):
            ax.annotate(f"{r['ncells'][0]}", (x, yv), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
    ax.axhline(CRITERION, color="k", ls="--", lw=1,
               label=f"pre-registered criterion {CRITERION:g} %")
    ax.axhline(100.0, color="0.6", ls=":", lw=1, label="100 % (no compression)")
    ax.set_xlabel("$\\Delta x$")
    ax.set_ylabel("worst plateau error vs exact two-shock [%]")
    ax.set_title("(a) resolution ladder")
    ax.legend(fontsize=7)

    # profile panel: the finest embedded run
    finest = max(with_err, key=lambda r: r["ncells"][0]) if with_err else None
    if finest is not None:
        x = np.asarray(finest["profile"]["x"])
        p = np.asarray(finest["profile"]["p"])
        rho = np.asarray(finest["profile"]["rho"])
        bx.plot(x, p, label="DG pressure")
        bx.plot(x, rho, label="DG density")
        bx.axhline(finest["reflected_p_exact"], color="C0", ls="--", lw=1,
                   label=f"exact plateau p={finest['reflected_p_exact']:.3g}")
        bx.axhline(finest["reflected_rho_exact"], color="C1", ls="--", lw=1,
                   label=f"exact plateau rho={finest['reflected_rho_exact']:.3g}")
        bx.axvline(finest["wall_face_actual"], color="k", lw=1, label="wall face")
        bx.set_title(f"(b) N={finest['ncells'][0]} P{finest['degree']} at t={finest['final_time']:.4g}")
        bx.set_xlabel("x")
        bx.set_ylabel("cell average")
        bx.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    rows = load()
    if not rows:
        print("no ladder artifacts yet")
        return 1
    tab = table(rows)
    write_csv(tab, HERE / "ladder.csv")
    write_md(tab, HERE / "ladder.md")
    figure(rows, HERE / "wall_reflection_ladder.png")
    print("\n".join(" | ".join(str(c) for c in row) for row in tab))
    print(f"wrote {HERE/'ladder.csv'}, {HERE/'ladder.md'}, {HERE/'wall_reflection_ladder.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
