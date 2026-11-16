"""E2 gate-margin study for REPRO-DG-CKS1998 (Sod, wall boundary).

Why this exists
---------------
The anchor gate ``E2`` is "DG Sod relative L2 < 3 % on the finest mesh of each
degree" (N=400 at P2).  The reproduction reports one PASS (2.368 %) and one
honest FAIL on a coarser grid (4.173 % at N=200) and explains the FAIL as
"resolution deficiency, not scheme error".  That explanation was never measured:
a single grid point cannot tell a converging scheme from a lucky one, and a gate
that sits 0.6 %-points away from its threshold is only meaningful if the margin
is *stable* under the free knobs of the scheme (CFL, TVB constant M).

This script closes that verification debt with three sweeps, all with the same
gate configuration (``limiter="tvb"``, ``tvb_constant=1``, ``limiter_stage=
"final"``, t=0.2, exact Riemann cell averages as reference):

  ``ladder``  P1/P2/P3 x N in the configured cell list -> observed order of the
              relative L2 error and the *crossing resolution* N*(3 %).
  ``cfl``     P2/N=400 with cfl in the configured list -> is the gate a knife
              edge in the time-step knob?
  ``tvb``     P2/N=400 with M in the configured list -> is the gate an artefact
              of one lucky limiter constant?

Honesty rules
-------------
* every row records ``pass`` from the *same* 3 % rule as the anchor gate;
* the crossing resolution is a log-log interpolation between the two grids that
  bracket 3 %, never an extrapolation from a single point;
* ``--study assemble`` only reads the per-sweep JSON files -- it re-derives the
  orders, the crossing and the verdicts from the recorded numbers, so the report
  digits are machine-generated, not hand-copied.

Usage
-----
    python validation/repro_cks1998/sod_gate_margin.py --study ladder
    python validation/repro_cks1998/sod_gate_margin.py --study cfl
    python validation/repro_cks1998/sod_gate_margin.py --study tvb
    python validation/repro_cks1998/sod_gate_margin.py --study assemble
    python validation/repro_cks1998/sod_gate_margin.py --study all     # ~15 min

Artifacts land in ``validation/repro_cks1998/artifacts/sod_gate_margin/``.
Exit code: 0 if the sweeps ran; the *gate* verdict is reported, never forced.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tensordg.convergence import observed_order, pairwise_orders  # noqa: E402
from tensordg.euler_1d import GAMMA  # noqa: E402
from tensordg.euler_dg_1d import solve_euler_dg_1d  # noqa: E402

OUTDIR = REPO / "validation" / "repro_cks1998" / "artifacts" / "sod_gate_margin"

# --- pre-declared study design (fixed before any number was inspected) -------
SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)
SOD_T = 0.2
GATE_TOL = 0.03                 # the anchor's E2 tolerance
GATE = {"cfl": 0.12, "tvb_constant": 1.0, "limiter": "tvb",
        "limiter_stage": "final", "degree": 2, "ncells": 400}

DESIGN = {
    "ladder": {
        "cells": {1: [200, 400, 800], 2: [200, 300, 400, 600], 3: [100, 200, 400]},
        "note": "one power-law fit per degree; N*=crossing of the 3 % line",
    },
    "cfl": {"cfl": [0.06, 0.12, 0.24, 0.48], "degree": 2, "ncells": 400},
    "tvb": {"tvb_constant": [0.0, 1.0, 10.0, 100.0], "degree": 2, "ncells": 400},
}


# ---------------------------------------------------------------------------
# one DG Sod run -> the gate metric
# ---------------------------------------------------------------------------
def run_case(degree: int, ncells: int, cfl: float, tvb_constant: float) -> dict:
    """Relative L2 of (rho, u, p) against the exact Riemann cell averages.

    The gate number is scored against the breakpoint-split **exact** cell average
    (``validation.repro_cks1998.station_reference.exact_cell_averages``).  The legacy
    fixed 8-point Gauss rule -- what this function used to call, while the module
    docstring already promised the exact reference -- is still evaluated on the *same*
    numerical solution and recorded as ``*_legacy``, so every row carries its own
    before/after and the instrument cannot change a verdict unobserved.
    """
    from validation.repro_cks1998.station_reference import (
        exact_cell_averages, fixed_rule_cell_averages)

    t0 = time.time()
    x, coeff, meta = solve_euler_dg_1d(
        SOD_LEFT, SOD_RIGHT, ncells=ncells, degree=degree, final_time=SOD_T,
        cfl=cfl, limiter=GATE["limiter"], tvb_constant=tvb_constant,
        limiter_stage=GATE["limiter_stage"],
    )
    avg = coeff[:, 0]
    rho = avg[:, 0]
    vel = avg[:, 1] / rho
    pre = (GAMMA - 1.0) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / rho)
    _, ref = exact_cell_averages(ncells, SOD_T)
    _, ref_legacy = fixed_rule_cell_averages(ncells, SOD_T)
    got = {"density": rho.numpy(), "velocity": vel.numpy(), "pressure": pre.numpy()}

    def _rel(reference):
        out = {}
        for key in ("density", "velocity", "pressure"):
            diff = got[key] - reference[key]
            out[key] = float(np.sqrt(np.mean(diff ** 2))
                             / np.sqrt(np.mean(np.asarray(reference[key]) ** 2)))
        return out

    rel, rel_legacy = _rel(ref), _rel(ref_legacy)
    worst = max(rel.values())
    worst_legacy = max(rel_legacy.values())
    return {
        "degree": degree, "ncells": ncells, "cfl": cfl,
        "tvb_constant": tvb_constant, "limiter": meta["limiter"],
        "steps": meta["steps"], "rel_l2": rel, "worst_rel_l2": worst,
        "pass": bool(worst < GATE_TOL),
        "rel_l2_legacy": rel_legacy, "worst_rel_l2_legacy": worst_legacy,
        "pass_legacy": bool(worst_legacy < GATE_TOL),
        "instrument_delta_pp": 100.0 * (worst_legacy - worst),
        "verdict_flip": bool((worst < GATE_TOL) != (worst_legacy < GATE_TOL)),
        "seconds": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------
def study_ladder(out: Path) -> dict:
    rows = []
    for degree, cells in DESIGN["ladder"]["cells"].items():
        for n in cells:
            rows.append(run_case(degree, n, GATE["cfl"], GATE["tvb_constant"]))
            print(f"ladder P{degree} N={n}: worst {rows[-1]['worst_rel_l2']:.5f} "
                  f"({rows[-1]['seconds']} s)", file=sys.stderr, flush=True)
    payload = {"study": "ladder", "design": DESIGN["ladder"], "gate": GATE,
               "gate_tol": GATE_TOL, "rows": rows, **_env()}
    (out / "ladder.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


def study_cfl(out: Path) -> dict:
    rows = [run_case(DESIGN["cfl"]["degree"], DESIGN["cfl"]["ncells"], c,
                     GATE["tvb_constant"]) for c in DESIGN["cfl"]["cfl"]]
    payload = {"study": "cfl", "design": DESIGN["cfl"], "gate": GATE,
               "gate_tol": GATE_TOL, "rows": rows, **_env()}
    (out / "cfl.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


def study_tvb(out: Path) -> dict:
    rows = [run_case(DESIGN["tvb"]["degree"], DESIGN["tvb"]["ncells"], GATE["cfl"], m)
            for m in DESIGN["tvb"]["tvb_constant"]]
    payload = {"study": "tvb", "design": DESIGN["tvb"], "gate": GATE,
               "gate_tol": GATE_TOL, "rows": rows, **_env()}
    (out / "tvb.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


def _env() -> dict:
    return {"environment": {"device": "cpu", "torch": torch.__version__,
                            "numpy": np.__version__,
                            "python": platform.python_version(),
                            "cpu": platform.machine(),
                            "torch_threads": torch.get_num_threads(),
                            "host_loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None}}


# ---------------------------------------------------------------------------
# analysis (re-derived from the recorded rows; nothing is hand copied)
# ---------------------------------------------------------------------------
def crossing(cells: list[float], errs: list[float], tol: float = GATE_TOL) -> dict | None:
    """Log-log interpolation of the resolution where ``errs`` crosses ``tol``.

    Returns ``None`` when the error never crosses inside the measured window
    (that is *not* the same as "the scheme converges faster than measured").
    """
    order = list(zip(cells, errs))          # coarse -> fine
    for (n0, e0), (n1, e1) in zip(order, order[1:]):
        if (e0 - tol) * (e1 - tol) <= 0 and e0 != e1:
            q = math.log(e1 / e0) / math.log(n1 / n0) if e1 > 0 and e0 > 0 else 1.0
            n_star = n0 * (tol / e0) ** (1.0 / q) if q != 0 else float("nan")
            return {"between": [n0, n1], "local_order": q, "n_star": n_star}
    return None


def analyse_ladder(payload: dict) -> dict:
    by_degree: dict[str, dict] = {}
    for degree in sorted({r["degree"] for r in payload["rows"]}):
        rows = sorted((r for r in payload["rows"] if r["degree"] == degree),
                      key=lambda r: r["ncells"])
        cells = [r["ncells"] for r in rows]
        errs = [r["worst_rel_l2"] for r in rows]
        per_component = {}
        for comp in ("density", "velocity", "pressure"):
            e = [r["rel_l2"][comp] for r in rows]
            per_component[comp] = {
                "errors": e,
                "observed_order": observed_order(cells, e),
                "pairwise_orders": pairwise_orders(cells, e),
                "crossing": crossing(cells, e),
            }
        by_degree[f"p{degree}"] = {
            "cells": cells, "worst_rel_l2": errs,
            "gate_case": {"ncells": max(cells),
                          "worst_rel_l2": errs[-1],
                          "pass": bool(errs[-1] < GATE_TOL),
                          "margin_pp": 100.0 * (GATE_TOL - errs[-1])},
            "observed_order": observed_order(cells, errs),
            "pairwise_orders": pairwise_orders(cells, errs),
            "crossing": crossing(cells, errs),
            "monotone_decreasing": all(a > b for a, b in zip(errs, errs[1:])),
            "components": per_component,
        }
    return by_degree


def analyse_knob(payload: dict, knob: str) -> dict:
    rows = sorted(payload["rows"], key=lambda r: r[knob])
    vals = [r[knob] for r in rows]
    errs = [r["worst_rel_l2"] for r in rows]
    fails = [r[knob] for r in rows if not r["pass"]]
    return {
        "knob": knob, "values": vals, "worst_rel_l2": errs,
        "passes": [r["pass"] for r in rows],
        "n_fail": len(fails), "fail_values": fails,
        "spread_pp": 100.0 * (max(errs) - min(errs)),
        # the honest statement: does *every* value of the free knob keep the gate?
        "gate_holds_for_all_knob_values": len(fails) == 0,
    }


def instrument_block(ladder: dict, cfl: dict, tvb: dict) -> dict:
    """Before/after of the reference-instrument swap, row by row (never hand-copied)."""
    rows = list(ladder["rows"]) + list(cfl["rows"]) + list(tvb["rows"])
    deltas = [(abs(r["instrument_delta_pp"]), r) for r in rows
              if "instrument_delta_pp" in r]
    flips = [r for r in rows if r.get("verdict_flip")]
    worst = max(deltas)[1] if deltas else None
    return {
        "gate_reference": "breakpoint-split exact cell average "
                          "(validation.repro_cks1998.station_reference)",
        "legacy_reference": "fixed 8-point Gauss rule (benchmarks.riemann_exact)",
        "rows": len(rows),
        "rows_with_legacy": len(deltas),
        "verdict_flips": [
            {"degree": r["degree"], "ncells": r["ncells"], "cfl": r["cfl"],
             "tvb_constant": r["tvb_constant"],
             "worst_rel_l2": r["worst_rel_l2"],
             "worst_rel_l2_legacy": r["worst_rel_l2_legacy"]} for r in flips],
        "n_flips": len(flips),
        "max_abs_delta_pp": None if worst is None else float(abs(worst["instrument_delta_pp"])),
        "worst_row": None if worst is None else {
            "degree": worst["degree"], "ncells": worst["ncells"],
            "delta_pp": worst["instrument_delta_pp"],
            "worst_rel_l2": worst["worst_rel_l2"],
            "worst_rel_l2_legacy": worst["worst_rel_l2_legacy"]},
    }


def assemble(out: Path) -> dict:
    ladder = json.loads((out / "ladder.json").read_text())
    cfl = json.loads((out / "cfl.json").read_text())
    tvb = json.loads((out / "tvb.json").read_text())
    metrics = {
        "anchor": "REPRO-DG-CKS1998",
        "topic": "E2 (Sod < 3 %) gate margin and sensitivity to cfl / TVB constant",
        "gate_config": GATE, "gate_tol": GATE_TOL,
        "design": DESIGN,
        "ladder": {"rows": ladder["rows"], "by_degree": analyse_ladder(ladder)},
        "cfl_sensitivity": {**analyse_knob(cfl, "cfl"), "rows": cfl["rows"]},
        "tvb_sensitivity": {**analyse_knob(tvb, "tvb_constant"), "rows": tvb["rows"]},
        "instrument": instrument_block(ladder, cfl, tvb),
        "environment": ladder["environment"],
        "wall_seconds": round(sum(r["seconds"] for p in (ladder, cfl, tvb)
                                  for r in p["rows"]), 1),
    }
    # honest top-level verdict: the anchor gate is reproduced *and* robust
    gate_cfg_row = [r for r in ladder["rows"]
                    if r["degree"] == GATE["degree"] and r["ncells"] == GATE["ncells"]
                    and r["cfl"] == GATE["cfl"]
                    and r["tvb_constant"] == GATE["tvb_constant"]]
    by_degree = analyse_ladder(ladder)
    cfl_blk = analyse_knob(cfl, "cfl")
    tvb_blk = analyse_knob(tvb, "tvb_constant")
    metrics["verdict"] = {
        "gate_case_reproduced": bool(gate_cfg_row and gate_cfg_row[0]["pass"]),
        "ladder_orders_positive": all(v["observed_order"] > 0
                                      for v in by_degree.values()),
        "cfl_gate_holds": cfl_blk["gate_holds_for_all_knob_values"],
        "tvb_gate_holds": tvb_blk["gate_holds_for_all_knob_values"],
        "gate_case_rel_l2": gate_cfg_row[0]["worst_rel_l2"] if gate_cfg_row else None,
    }
    metrics["verdict"]["all_pass"] = all(
        bool(v) for k, v in metrics["verdict"].items() if k != "gate_case_rel_l2")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    with open(out / "ladder.csv", "w", newline="") as fh:
        keys = ["degree", "ncells", "cfl", "tvb_constant", "limiter", "steps",
                "rel_l2_density", "rel_l2_velocity", "rel_l2_pressure",
                "worst_rel_l2", "pass", "worst_rel_l2_legacy", "pass_legacy", "seconds"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in ladder["rows"]:
            w.writerow({"degree": r["degree"], "ncells": r["ncells"], "cfl": r["cfl"],
                        "tvb_constant": r["tvb_constant"], "limiter": r["limiter"],
                        "steps": r["steps"],
                        **{f"rel_l2_{k}": v for k, v in r["rel_l2"].items()},
                        "worst_rel_l2": r["worst_rel_l2"], "pass": r["pass"],
                        "worst_rel_l2_legacy": r.get("worst_rel_l2_legacy"),
                        "pass_legacy": r.get("pass_legacy"),
                        "seconds": r["seconds"]})
    with open(out / "knob_sensitivity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["study", "knob", "worst_rel_l2", "pass"])
        for study, knob in (("cfl", "cfl"), ("tvb", "tvb_constant")):
            for r in (cfl if study == "cfl" else tvb)["rows"]:
                w.writerow([study, r[knob], r["worst_rel_l2"], r["pass"]])

    (out / "summary.md").write_text(render_summary(metrics))
    figure(metrics, out)
    print(json.dumps(metrics["verdict"], indent=2))
    return metrics


def render_summary(m: dict) -> str:
    lines = ["# REPRO-DG-CKS1998 · E2 gate margin (Sod, 3 % on worst component)", "",
             f"Gate configuration: `{json.dumps(m['gate_config'])}` · "
             f"tolerance {m['gate_tol']:.0%} · re-derived from the swept rows.", ""]
    lines += ["## Ladder (per-degree rel-L2 vs N)", "",
              "| degree | cells | worst rel L2 (%) | observed order | pairwise | N*(3 %) | gate @Nmax | margin |",
              "|---:|---|---:|---:|---|---:|---|---:|"]
    for key, blk in m["ladder"]["by_degree"].items():
        cells = ", ".join(str(c) for c in blk["cells"])
        errs = " / ".join(f"{100 * e:.3f}" for e in blk["worst_rel_l2"])
        cr = blk["crossing"]
        cr_s = f"{cr['n_star']:.0f}" if cr else "not crossed"
        pw = ", ".join(f"{p:.2f}" for p in blk["pairwise_orders"])
        lines.append(f"| {key} | {cells} | {errs} | {blk['observed_order']:.3f} | {pw} | "
                     f"{cr_s} | {blk['gate_case']['pass']} | "
                     f"{blk['gate_case']['margin_pp']:.3f} pp |")
    lines += ["", "## Sensitivity of the gate at P2/N=400", "",
              "| knob | values | worst rel L2 (%) | PASS | spread | gate holds for all |",
              "|---|---|---|---|---:|---|"]
    for name, blk in (("cfl", m["cfl_sensitivity"]), ("TVB M", m["tvb_sensitivity"])):
        vals = ", ".join(f"{v:g}" for v in blk["values"])
        errs = " / ".join(f"{100 * e:.3f}" for e in blk["worst_rel_l2"])
        lines.append(f"| {name} | {vals} | {errs} | "
                     f"{sum(blk['passes'])}/{len(blk['passes'])} | "
                     f"{blk['spread_pp']:.3f} pp | {blk['gate_holds_for_all_knob_values']} |")
    inst = m.get("instrument") or {}
    if inst:
        lines += ["", "## Reference instrument (exact vs the legacy fixed rule)", "",
                  f"rows scored: {inst['rows_with_legacy']} / {inst['rows']} · "
                  f"verdict flips: {inst['n_flips']} · "
                  f"max |delta| = {inst['max_abs_delta_pp']:.4f} pp", ""]
        worst = inst.get("worst_row")
        if worst:
            lines.append(f"largest single-row shift: P{worst['degree']} N={worst['ncells']} "
                         f"{worst['worst_rel_l2_legacy'] * 100:.4f} % (legacy) -> "
                         f"{worst['worst_rel_l2'] * 100:.4f} % (exact)")
    lines += ["", "## Verdict (machine-derived)", "", "```json",
              json.dumps(m["verdict"], indent=2), "```", ""]
    return "\n".join(lines)


def figure(m: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for key, blk in m["ladder"]["by_degree"].items():
        ax.loglog(blk["cells"], blk["worst_rel_l2"], "o-",
                  label=f"DG {key.upper()}, order {blk['observed_order']:.2f}")
        cr = blk["crossing"]
        if cr:
            ax.plot([cr["n_star"]], [m["gate_tol"]], "x", ms=7,
                    color=ax.lines[-1].get_color())
    ax.axhline(m["gate_tol"], color="k", ls="--", lw=0.8, label="E2 gate 3 %")
    ax.set_xlabel("cells N")
    ax.set_ylabel("relative $L^2$ error (worst component)")
    ax.set_title("CKS1998 E2: Sod error vs resolution (x = N*(3 %))")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "e2_gate_margin_ladder.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    cfl = m["cfl_sensitivity"]
    tvb = m["tvb_sensitivity"]
    ax.plot(cfl["values"], [100 * e for e in cfl["worst_rel_l2"]], "o-", label="cfl sweep")
    ax.axhline(100 * m["gate_tol"], color="k", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("cfl (P2, N=400)")
    ax.set_ylabel("worst rel $L^2$ (%)")
    ax2 = ax.twiny()
    ax2.plot(tvb["values"], [100 * e for e in tvb["worst_rel_l2"]], "s--", color="C1",
             label="TVB M sweep")
    ax2.set_xscale("symlog", linthresh=1.0)
    ax2.set_xlabel("TVB constant M (P2, N=400)")
    ax.set_title("E2 gate margin under the free knobs (dashed = 3 % gate)")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "e2_gate_margin_knobs.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="all",
                    choices=["ladder", "cfl", "tvb", "assemble", "all"])
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args(argv)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    if args.study in ("ladder", "all"):
        study_ladder(out)
    if args.study in ("cfl", "all"):
        study_cfl(out)
    if args.study in ("tvb", "all"):
        study_tvb(out)
    if args.study in ("assemble", "all"):
        metrics = assemble(out)
        return 0 if metrics["verdict"]["all_pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())