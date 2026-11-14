"""Gate evaluation + figures for the DMR mesh-refinement ladder.

Reads ``validation/dmr/refinement/*/metrics.json`` (written by
``scripts/run_dmr_refinement.py``) and the pre-registered gate list in
``validation/dmr/refinement/gates.json``, and emits

    validation/dmr/refinement/summary.json   machine-readable verdicts + numbers
    validation/dmr/refinement/summary.md     the same as a table
    validation/dmr/refinement/fig1_*.png     profile / convergence figures

No number is typed by hand: every table entry is copied from a metrics.json or
recomputed here from the stored profiles.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POST_RHO = 8.0
POST_P = 116.5
PRE_RHO = 1.4
INTERIOR_Y = (0.50, 0.60, 0.70, 0.80)


def load_runs(root: Path):
    runs = []
    for p in sorted(root.glob("*/metrics.json")):
        runs.append(json.loads(p.read_text()))
    return runs


def _row_key(front, target):
    """metrics keys are the cell-centre heights actually used (strings)."""
    keys = [float(k) for k in front]
    return min(keys, key=lambda k: abs(k - target))


def ladder(runs, degree=1):
    rs = [r for r in runs if r["degree"] == degree]
    return sorted(rs, key=lambda r: r["grid"][0])


def observed_rate(xs, ys):
    """Least-squares slope of log(y) vs log(x); ys must be positive."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = ys > 0
    if ok.sum() < 2:
        return None
    a, _ = np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)
    return float(a)


def evaluate(gates_doc, runs):
    gates = {g["id"]: {"id": g["id"], "claim": g["claim"],
                       "pass_if": g["pass_if"], "evidence": {}}
             for g in gates_doc["gates"]}
    p1 = ladder(runs, 1)
    if len(p1) < 2:
        raise SystemExit("need at least two P1 rungs to evaluate the gates")
    coarse, fine = p1[0], p1[-1]
    nxs = [r["grid"][0] for r in p1]
    dxs = [r["dx"] for r in p1]

    # G1 interior plateau inside 3 % at every rung
    ev = {r["grid"][0]: {"rho": r["plateau_rel_error"]["mid"]["rho"],
                        "p": r["plateau_rel_error"]["mid"]["p"]} for r in p1}
    ok = all(abs(v["rho"]) <= 0.03 and abs(v["p"]) <= 0.03 for v in ev.values())
    gates["G1"]["evidence"] = ev
    gates["G1"]["verdict"] = "PASS" if ok else "FAIL"

    # G2 convergence of the interior plateau error
    a = abs(coarse["plateau_rel_error"]["mid"]["rho"])
    b = abs(fine["plateau_rel_error"]["mid"]["rho"])
    gates["G2"]["evidence"] = {"coarse": {"nx": coarse["grid"][0], "abs_rel": a},
                               "fine": {"nx": fine["grid"][0], "abs_rel": b},
                               "rate": observed_rate(dxs, [abs(r["plateau_rel_error"]["mid"]["rho"]) for r in p1])}
    gates["G2"]["verdict"] = "PASS" if b < a else "FAIL"

    # G3 interior front position error, bounded number of cells
    ev = {}
    for r in p1:
        worst = 0.0
        for y in INTERIOR_Y:
            k = str(_row_key(r["front_error_cells"], y))
            v = r["front_error_cells"][k]
            if v is not None:
                worst = max(worst, abs(v))
        ev[r["grid"][0]] = worst
    gates["G3"]["evidence"] = ev
    gates["G3"]["verdict"] = "PASS" if max(ev.values()) <= 2.0 else "FAIL"

    # G4 top-row front error shrinks in physical units
    def top_err(r):
        k = str(_row_key(r["front_error"], 0.90))
        return abs(r["front_error"][k])

    gates["G4"]["evidence"] = {"coarse": {"nx": coarse["grid"][0], "abs_phys": top_err(coarse)},
                               "fine": {"nx": fine["grid"][0], "abs_phys": top_err(fine)},
                               "ratio": top_err(fine) / top_err(coarse)}
    gates["G4"]["verdict"] = "PASS" if top_err(fine) <= 0.5 * top_err(coarse) else "FAIL"

    # G5 smear width constant in cells
    ws = [r["smear_width_cells"] for r in p1]
    ratio = fine["smear_width_cells"] / coarse["smear_width_cells"]
    gates["G5"]["evidence"] = {r["grid"][0]: r["smear_width_cells"] for r in p1}
    gates["G5"]["evidence"]["ratio_finest_over_coarsest"] = ratio
    gates["G5"]["verdict"] = "PASS" if 0.5 <= ratio <= 2.0 else "FAIL"

    # G6 fixed-physical-offset sample improves and enters the 3 % band
    ev = {r["grid"][0]: {"rel_error_rho": r["samples_behind_front"]["0.1phys"]["rel_error"]["rho"],
                         "offset_cells": r["samples_behind_front"]["0.1phys"]["offset_cells"]}
          for r in p1}
    e_coarse = abs(ev[coarse["grid"][0]]["rel_error_rho"])
    e_fine = abs(ev[fine["grid"][0]]["rel_error_rho"])
    gates["G6"]["evidence"] = ev
    gates["G6"]["verdict"] = ("PASS" if (e_fine < e_coarse and e_fine <= 0.03)
                              else "FAIL")

    # G7 L1 convergence where the exact solution is piecewise constant
    ls = [r["undisturbed_l1"]["l1_rel"] for r in p1]
    rate = observed_rate(dxs, ls)
    gates["G7"]["evidence"] = {"l1_rel": dict(zip(nxs, ls)), "rate": rate}
    gates["G7"]["verdict"] = "PASS" if (rate is not None and rate >= 0.5) else "FAIL"

    # G8 the 0.10 sample sits inside the captured front at 64x16, in the
    #    plateau at the finest rung
    c_off = coarse["samples_behind_front"]["0.1phys"]["offset_cells"]
    c_w = coarse["smear_width_cells"]
    f_off = fine["samples_behind_front"]["0.1phys"]["offset_cells"]
    f_w = fine["smear_width_cells"]
    gates["G8"]["evidence"] = {"coarse": {"offset_cells": c_off, "smear_width_cells": c_w},
                               "fine": {"offset_cells": f_off, "smear_width_cells": f_w}}
    gates["G8"]["verdict"] = "PASS" if (c_off < c_w and f_off > f_w) else "FAIL"
    return gates


def figures(root: Path, runs, out_dir: Path):
    made = []
    p1 = ladder(runs, 1)
    p2 = ladder(runs, 2)

    # fig 1: density profile through the front at y ~ 0.70, all rungs
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for r in p1:
        ax[0].plot(r["profile_x"], r["profile_rho"], lw=0.9,
                   label=f"P1 {r['grid'][0]}x{r['grid'][1]}")
    for r in p2:
        ax[0].plot(r["profile_x"], r["profile_rho"], lw=1.1, ls="--",
                   label=f"P2 {r['grid'][0]}x{r['grid'][1]}")
    ax[0].axhline(POST_RHO, color="k", lw=1.0, ls=":", label="exact post-shock rho=8")
    ax[0].axhline(PRE_RHO, color="k", lw=1.0, ls=":", label="exact pre-shock rho=1.4")
    ax[0].set_xlim(0.0, 1.4)
    ax[0].set_ylim(0.5, 9.0)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("density")
    ax[0].set_title("incident-front profile, y ~ 0.70, t = 0.02")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    # right: interior plateau error vs h (log-log) with the pre-registered band
    for label, rs, marker in (("P1", p1, "o"), ("P2", p2, "s")):
        if not rs:
            continue
        hs = [r["dx"] for r in rs]
        e_rho = [abs(r["plateau_rel_error"]["mid"]["rho"]) for r in rs]
        e_p = [abs(r["plateau_rel_error"]["mid"]["p"]) for r in rs]
        ax[1].loglog(hs, e_rho, marker + "-", label=f"{label} |rho err| (interior)")
        ax[1].loglog(hs, e_p, marker + "--", label=f"{label} |p err| (interior)")
    ax[1].axhline(0.03, color="r", ls=":", label="3 % standard")
    ax[1].set_xlabel("h = 4 / nx")
    ax[1].set_ylabel("relative plateau error")
    ax[1].set_title("post-shock plateau far from all boundaries")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = out_dir / "fig1_profile_and_plateau_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    # fig 2: per-row contamination profile (sample 8 cells behind the front)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for r in p1:
        rp = r["row_profile"]
        ax.plot(rp["rel_error_from_measured_front"], rp["y"], "o-", ms=3, lw=0.8,
                label=f"P1 {r['grid'][0]}x{r['grid'][1]}")
    ax.axvline(0.0, color="k", lw=0.8)
    for band, txt in ((0.03, "3 %"), (-0.03, "3 %")):
        ax.axvline(band, color="r", ls=":", lw=0.9)
    ax.set_xlabel("relative density error, 8 cells behind the MEASURED front")
    ax.set_ylabel("y")
    ax.set_title("row-by-row post-shock density error (top-boundary contamination)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig2_row_error_profile.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    # fig 3: front-position error in cells vs y, all rungs
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in p1:
        ys = [float(k) for k in r["front_error_cells"]]
        es = [r["front_error_cells"][k] for k in r["front_error_cells"]]
        ax.plot(ys, es, "o-", ms=4, lw=0.9, label=f"P1 {r['grid'][0]}x{r['grid'][1]}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("y")
    ax.set_ylabel("front position error [cells]")
    ax.set_title("incident front captured position vs exact plane-front law")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig3_front_error_cells.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)
    return made


def write_markdown(path: Path, gates, runs):
    p1 = ladder(runs, 1)
    lines = ["# DMR refinement ladder — gate verdicts", ""]
    lines.append("| gate | verdict | pass_if | evidence |")
    lines.append("|---|---|---|---|")
    for gid in sorted(gates):
        g = gates[gid]
        lines.append(f"| {gid} | **{g['verdict']}** | {g['pass_if']} | "
                     f"`{json.dumps(g['evidence'], default=str)}` |")
    lines += ["", "## P1 ladder", "",
              "| grid | dx | steps | interior rho err | interior p err | "
              "top-window rho err | smear width [cells] | max front err [cells, y<=0.8] "
              "| front err @y~0.90 [cells] | L1 rel (undisturbed) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in p1:
        k = str(_row_key(r["front_error_cells"], 0.90))
        lines.append(
            f"| {r['grid'][0]}x{r['grid'][1]} | {r['dx']:.5f} | {r['steps']} | "
            f"{r['plateau_rel_error']['mid']['rho']:+.4%} | "
            f"{r['plateau_rel_error']['mid']['p']:+.4%} | "
            f"{r['plateau_rel_error']['top']['rho']:+.4%} | "
            f"{r['smear_width_cells']:.2f} | {r['max_front_error_cells']:.2f} | "
            f"{r['front_error_cells'][k]:.2f} | {r['undisturbed_l1']['l1_rel']:.4f} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="validation/dmr/refinement")
    args = ap.parse_args()
    root = Path(args.dir)
    gates_doc = json.loads((root / "gates.json").read_text())
    runs = load_runs(root)
    for r in runs:
        print(f"loaded {r['grid']} P{r['degree']} "
              f"({r['steps']} steps, {r['wall_seconds']:.0f} s)")
    gates = evaluate(gates_doc, runs)
    figs = figures(root, runs, root)
    summary = {
        "anchor": gates_doc["anchor"],
        "case": gates_doc["case"],
        "protocol": gates_doc["protocol"],
        "pre_registration": gates_doc["pre_registration"],
        "runs": [{k: r[k] for k in ("grid", "degree", "steps", "wall_seconds", "dx")}
                 for r in runs],
        "gates": gates,
        "n_pass": sum(1 for g in gates.values() if g["verdict"] == "PASS"),
        "n_fail": sum(1 for g in gates.values() if g["verdict"] == "FAIL"),
        "figures": figs,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_markdown(root / "summary.md", gates, runs)
    print(json.dumps({k: (v["verdict"] if isinstance(v, dict) else v)
                      for k, v in gates.items()}, indent=2))
    print(f"PASS {summary['n_pass']} / FAIL {summary['n_fail']} -> {root/'summary.json'}")


if __name__ == "__main__":
    main()