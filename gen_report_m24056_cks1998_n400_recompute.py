"""Report figures for m24056: CKS1998 E2 Sod P2 gate — N=400 from-scratch recompute.

Closes the ``not_run`` debt left by m23250 (report 20883): the repository
published ``p2_n400`` worst-field relative L2 = 2.368 % (PASS) but the current
code reads 3.8384 % (cached profile, FAIL). This ticket re-runs the documented
configuration from scratch on the committed tree and sweeps the pre-registered
configuration axes to settle whether the published number is a *configuration*
or a *code-state* mismatch.

Figures (all numbers read from the committed receipt JSON, nothing hand-typed):

* ``fig1_gate_readings.png``  -- worst-field (velocity) relative L2 at N=200 and
  N=400: published (grey), cached profile (orange), this ticket's from-scratch
  reading (blue), with the 3 % gate (black dashed).
* ``fig2_sweep_attribution.png`` -- the 7 pre-registered configurations' worst
  relative L2 against the published 2.368 % (black dashed) and the 3 % gate
  (red dashed); the distance of each bar from the published value is annotated.
* ``fig3_velocity_profile.png`` -- velocity profile at N=400 (current code,
  documented config) vs the exact Riemann cell averages (the analytic reference).

Usage::

    python gen_report_m24056_cks1998_n400_recompute.py \
        --receipt <sod_p2_gate_recompute.json> --out-dir /tmp/m24056_figs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

PUBLISHED = 2.368  # percent, published p2_n400 worst-field (velocity)
CACHED = 3.8384    # percent, committed solver profile (cj3966)
GATE = 3.0         # percent, gate tolerance
# m23250 measured on the current code (N=200 leg, 52.6 s):
M23250_N200 = 6.1256  # percent, legacy instrument


def load_receipt(path: str) -> dict:
    return json.loads(Path(path).read_text())


def fig1_gate_readings(receipt: dict, out: Path) -> None:
    """Published vs cached vs this ticket's from-scratch reading, both grids."""
    runs = {r["config"]["tag"]: r for r in receipt["runs"]}
    doc = runs["documented"]
    n400_scratch = 100.0 * doc["worst_legacy"]

    labels = ["N=200", "N=400"]
    published = [4.173, PUBLISHED]
    current = [M23250_N200, n400_scratch]
    cached = [None, CACHED]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.bar(x - w, published, w, label="published (round2 docs)", color="0.75")
    b2 = ax.bar(x, [c if c is not None else 0 for c in cached], w,
                label="cached profile (cj3966)", color="tab:orange")
    ax.bar(x + w, current, w, label="current code, from scratch", color="tab:blue")
    for xi, p in zip(x, published):
        ax.text(xi - w, p + 0.08, f"{p:.3f}", ha="center", fontsize=8)
    for xi, c in zip(x, current):
        ax.text(xi + w, c + 0.08, f"{c:.4f}", ha="center", fontsize=8)
    ax.axhline(GATE, color="k", ls="--", lw=1.2)
    ax.text(1.42, GATE + 0.06, "3 % gate", ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("worst-field (velocity) relative L2, %")
    ax.set_title("CKS1998 E2 · DG P2 Sod gate: published vs current code")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(published), max(current)) * 1.18)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig2_sweep_attribution(receipt: dict, out: Path) -> None:
    """The 7 pre-registered configs vs the published number (attribution)."""
    runs = receipt["runs"]
    tags = [r["config"]["tag"] for r in runs]
    vals = [100.0 * r["worst_legacy"] for r in runs]
    gaps = [r["gap_to_published_pp"] for r in runs]

    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    colors = ["tab:blue" if g < 0.05 else "tab:grey" for g in gaps]
    ax.bar(x, vals, color=colors)
    ax.axhline(PUBLISHED, color="k", ls="--", lw=1.2)
    ax.text(len(tags) - 0.4, PUBLISHED + 0.05, f"published {PUBLISHED:.3f} %",
             ha="right", fontsize=8)
    ax.axhline(GATE, color="r", ls="--", lw=1.2)
    ax.text(len(tags) - 0.4, GATE + 0.05, "3 % gate", ha="right", fontsize=8, color="r")
    for xi, v, g in zip(x, vals, gaps):
        ax.text(xi, v + 0.06, f"{v:.3f}", ha="center", fontsize=7)
        ax.text(xi, -0.42, f"Δpub {g:+.3f} pp", ha="center", fontsize=6.5, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, fontsize=8)
    ax.set_ylabel("worst-field relative L2, %")
    ax.set_title("Pre-registered sweep: does any neighbour reproduce 2.368 %?")
    ax.set_ylim(0, max(max(vals), PUBLISHED) * 1.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig3_velocity_profile(receipt: dict, out: Path) -> None:
    """Velocity profile at N=400 (current code) vs exact Riemann cell averages."""
    from benchmarks.shock_reference_quadrature_audit import SOD, SOD_TIME
    from validation.repro_cks1998.station_reference import exact_cell_averages
    from tensordg.euler_dg_1d import solve_euler_dg_1d
    from benchmarks.sod_exact import GAMMA

    ncells = 400
    _, coeff, _ = solve_euler_dg_1d(
        SOD[0], SOD[1], ncells=ncells, degree=2, final_time=SOD_TIME,
        cfl=0.12, limiter="tvb", tvb_constant=1.0, limiter_stage="final",
        positivity="scale",
    )
    avg = coeff[:, 0]
    vel = (avg[:, 1] / avg[:, 0]).numpy()
    xc = (np.arange(ncells) + 0.5) / ncells
    _, exact = exact_cell_averages(ncells, SOD_TIME)
    vel_exact = exact["velocity"]

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    ax.plot(xc, vel_exact, "k-", lw=1.6, label="exact Riemann (cell avg)")
    ax.plot(xc, vel, "b.", ms=3, label="DG P2 TVB, N=400 (current code)")
    ax.axvline(0.5, color="0.8", lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("velocity u")
    ax.set_title("Sod t=0.2 · velocity profile (N=400, current code)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--out-dir", default="/tmp/m24056_figs")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    receipt = load_receipt(args.receipt)
    fig1_gate_readings(receipt, out / "fig1_gate_readings.png")
    fig2_sweep_attribution(receipt, out / "fig2_sweep_attribution.png")
    fig3_velocity_profile(receipt, out / "fig3_velocity_profile.png")
    print("figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
