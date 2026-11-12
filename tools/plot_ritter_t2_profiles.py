"""Generate Ritter dam-break profile figure + artifacts for mission m16081 (t*=2).

Runs the P1 DG benchmark at the non-dimensional time t* = t·c0/h0 = 2
(i.e. t = 2·√(h0/g) = 0.6386 s for h0 = 1 m, g = 9.81 m/s²), compares against
the Ritter (1892) exact self-similar solution, and writes:
  * docs/figs/ritter_t2_profiles.png   (h and u profiles vs exact)
  * artifacts/ritter_t2_metrics.json   (machine-readable metrics)
  * runs/ritter_t2_repro.sh            (one-click reproduction)
"""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.ritter_dam_break import run_ritter, ritter_exact

ROOT = Path(__file__).resolve().parent.parent
GRAVITY = 9.81


def main():
    h0 = 1.0
    g = GRAVITY
    c0 = math.sqrt(g * h0)
    # Non-dimensional time t* = t·c0/h0 = 2  ->  t = 2·h0/c0 = 2·√(h0/g)
    t_star = 2.0
    t_final = t_star * h0 / c0

    # P1 DG, 200 cells, domain wide enough that the wet front stays inside
    result = run_ritter(
        ncells=200, degree=1, h0=h0, final_time=t_final,
        domain=(-5.0, 5.0), cfl=0.15, flux_kind="hll",
        limiter=None, rk_order=3, dry_tol=1e-8,
    )

    x = np.asarray(result["x"])
    h_num = np.asarray(result["h_num"])
    u_num = np.asarray(result["u_num"])
    h_ex = np.asarray(result["h_ex"])
    u_ex = np.asarray(result["u_ex"])

    # ---- Figure: h and u profiles vs Ritter exact ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].plot(x, h_ex, "k-", lw=1.8, label="Ritter (1892) exact")
    axes[0].plot(x, h_num, "C0--", lw=1.4,
                 label=f"DG P1, N=200  (rel L1 h = {result['rel_l1_h']:.3%})")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("h [m]")
    axes[0].set_title(f"Ritter dam break, t* = {t_star}  (t = {t_final:.4f} s)")
    axes[0].axvline(result["wet_front_exact"], color="C3", ls=":", lw=1.2,
                     label=f"exact front 2c0t = {result['wet_front_exact']:.3f} m")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, u_ex, "k-", lw=1.8, label="Ritter (1892) exact")
    axes[1].plot(x, u_num, "C0--", lw=1.4,
                 label=f"DG P1, N=200  (rel L1 u = {result['rel_l1_u']:.3%})")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("u [m/s]")
    axes[1].set_title(f"Ritter dam break, t* = {t_star}  (t = {t_final:.4f} s)")
    axes[1].axvline(result["wet_front_exact"], color="C3", ls=":", lw=1.2)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig_dir = ROOT / "docs" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "ritter_t2_profiles.png"
    fig.savefig(fig_path, dpi=150)

    # ---- Artifacts ----
    art_dir = ROOT / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    with open(art_dir / "ritter_t2_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # ---- Repro script ----
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    repro = f"""#!/usr/bin/env bash
# One-click reproduction of the Ritter (1892) dam-break case at t* = 2 (m16081).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m benchmarks.ritter_dam_break --ncells 200 --degree 1 --time {t_final:.6f} --output-dir /tmp/ritter_t2_repro
python -m pytest tests/test_swe_1d.py -k "Ritter" -q
"""
    with open(runs_dir / "ritter_t2_repro.sh", "w") as f:
        f.write(repro)
    (runs_dir / "ritter_t2_repro.sh").chmod(0o755)

    # ---- Summary ----
    print("=" * 70)
    print(f"Ritter dam break @ t* = {t_star}  (t = {t_final:.4f} s, h0 = {h0} m, g = {g} m/s²)")
    print(f"  c0 = {c0:.4f} m/s,  exact front = {result['wet_front_exact']:.4f} m")
    print(f"  Steps: {result['steps']},  wall time: {result['wall_time']:.2f} s")
    print(f"  Wet cells: {result['wet_cells']}")
    print(f"  rel L1 h: {result['rel_l1_h']:.4%}   (tolerance < 5%)")
    print(f"  rel L2 h: {result['rel_l2_h']:.4%}")
    print(f"  rel L1 u: {result['rel_l1_u']:.4%}   (tolerance < 5%)")
    print(f"  rel L2 u: {result['rel_l2_u']:.4%}")
    print(f"  Wet front error: {result['wet_front_error_cells']:.2f} cells")
    print(f"  Mass error: {result['mass_error']:.4e}")
    print(f"  h range: [{result['h_min']:.2e}, {result['h_max']:.2f}]  (no negative depth)")
    print(f"  Figure: {fig_path}")
    print(f"  Artifact: {art_dir / 'ritter_t2_metrics.json'}")
    print(f"  Repro: {runs_dir / 'ritter_t2_repro.sh'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
