"""Render the G3 embedded-wall reflection ladder: table + convergence figure.

Reads ``validation/forward_step/g3/g3_ladder.json`` (written by
``scripts.run_g3_wall_reflection_ladder``; every number is printed by the run, so
nothing is hand-copied into the report) and emits

* ``docs/figures/g3_ladder.png`` -- reflected-plateau error and whole-domain L1
  error against the mesh size, one curve per wall arm, with the 3 % gate line;
* a markdown table on stdout to paste (programmatically) into the case doc.

Usage::

    PYTHONPATH=. python -m scripts.report_g3_ladder [--json PATH] [--out PNG]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def table(payload: dict) -> str:
    header = ("| arm | P^k | nx | dx | steps | plateau cells | plateau worst % | "
              "mean rho % | mean p % | L1 rho | Linf rho | mass resid | status |")
    lines = [header, "|" + "---|" * 13]
    for r in payload["rows"]:
        def fmt(value, spec):
            return "--" if value is None else format(value, spec)
        lines.append(
            "| {} | {} | {} | {:.4e} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["wall_mode"], r["degree"], r["ncells"][0], r["dx"], r["steps"],
                fmt(r["plateau_cells"], "d"), fmt(r["worst_pct"], ".2f"),
                fmt(r["mean_rho_pct"], ".2f"), fmt(r["mean_p_pct"], ".2f"),
                fmt(r["l1_rho"], ".3e"), fmt(r["linf_rho"], ".3e"),
                fmt(r["mass_residual"], ".2e"), r["status"]))
    return "\n".join(lines)


def figure(payload: dict, out: Path) -> Path | None:
    rows = [r for r in payload["rows"]]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = {"boundary": "tab:blue", "solid": "tab:red"}
    for ax, key, ylabel, title in (
        (axes[0], "worst_pct", "reflected-plateau error [%]",
         "plateau vs exact two-shock solution"),
        (axes[1], "l1_rho", "whole-domain relative L1 (rho)",
         "whole-domain L1, no sampling band"),
    ):
        for arm in ("boundary", "solid"):
            pts = [(r["dx"], r[key]) for r in rows
                   if r["wall_mode"] == arm and r.get(key) is not None]
            if not pts:
                continue
            # one marker per (dx, degree): connect the P1 rungs of that arm
            for degree, style in ((1, "o-"), (2, "s--")):
                sub = [(r["dx"], r[key], r["degree"]) for r in rows
                       if r["wall_mode"] == arm and r.get(key) is not None
                       and r["degree"] == degree]
                if not sub:
                    continue
                sub.sort()
                ax.loglog([p[0] for p in sub], [p[1] for p in sub], style,
                          color=colours[arm], markerfacecolor="none",
                          label=f"{arm} arm, P{degree}")
        if key == "worst_pct":
            ax.axhline(3.0, color="k", ls=":", lw=1.2, label="3 % gate")
        ax.set_xlabel("dx")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("G3: shock reflection off an embedded solid wall vs the domain-wall path "
                 "(CKS1998 forward-step capability)", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    return out


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="validation/forward_step/g3/g3_ladder.json")
    parser.add_argument("--out", default="docs/figures/g3_ladder.png")
    parser.add_argument("--table-out", default="validation/forward_step/g3/g3_ladder.md")
    args = parser.parse_args(argv)
    payload = load(Path(args.json))
    text = table(payload)
    Path(args.table_out).write_text(text + "\n")
    print(text)
    made = figure(payload, Path(args.out))
    print(f"\nfigure: {made}")
    return payload


if __name__ == "__main__":
    main()
