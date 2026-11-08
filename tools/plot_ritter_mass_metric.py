"""Reproduce the Ritter mass-metric evidence (artifact JSON + figure).

Runs the two benchmark configurations, one control run where the wet front
stays far from the walls, rewrites
``validation/swe_1d_dry_bed/ritter_results.json`` with the corrected mass
metrics and writes ``docs/figs/ritter_mass_metric.png``.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.ritter_dam_break import run_ritter
from tensordg.swe_1d import reconstruct_fields, solve_swe_1d

ROOT = Path(__file__).resolve().parent.parent


def control(domain, ncells, t_final, cfl=0.3):
    """Dry-bed run with a hard-zero dry side and no seeding."""
    x, coeff, zb, meta = solve_swe_1d(
        lambda xx: torch.where(xx < 0.0, torch.ones_like(xx), torch.zeros_like(xx)),
        lambda xx: torch.zeros_like(xx), lambda xx: torch.zeros_like(xx),
        ncells=ncells, degree=1, final_time=t_final, domain=domain, cfl=cfl,
        flux_kind="hll")
    h, u, _ = reconstruct_fields(coeff, zb)
    dx = (domain[1] - domain[0]) / ncells
    mass0 = 1.0 * abs(domain[0])
    return (abs(float(h.sum()) * dx - mass0) / mass0, meta["steps"],
            h.numpy(), x.numpy())


def main() -> dict:
    res = {f"P1_N{n}": run_ritter(ncells=n) for n in (100, 200)}
    (ROOT / "validation" / "swe_1d_dry_bed").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "validation" / "swe_1d_dry_bed" / "ritter_results.json", "w") as f:
        json.dump(res, f, indent=2)

    drift_ctrl, steps_ctrl, _, _ = control((-3.0, 3.0), 100, 0.3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    r200 = res["P1_N200"]
    x = np.asarray(r200["x"])
    axes[0].plot(x, r200["h_ex"], "k-", lw=1.6, label="Ritter exact")
    axes[0].plot(x, r200["h_num"], "C0--", lw=1.4, label=f"DG P1, N=200  (rel L1 h = {r200['rel_l1_h']:.3%})")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("h [m]")
    axes[0].set_title("Ritter dry-bed, t = 0.5 s")
    axes[0].axvline(3.0, color="C3", ls=":", lw=1.2)
    axes[0].annotate("transmissive wall\n(front 2c0t = 3.132 m\npasses it at t = 0.479 s)",
                     xy=(3.0, 0.35), xytext=(0.2, 0.55), fontsize=7.5,
                     arrowprops=dict(arrowstyle="->", lw=0.8, color="C3"))
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylim(-0.02, 1.15)

    labels = ["P1_N100", "P1_N200"]
    legacy = [res[k]["mass_error_legacy"] for k in labels]
    fixed = [res[k]["mass_error"] for k in labels]
    pos = np.arange(len(labels))
    axes[1].bar(pos - 0.2, legacy, 0.4, color="C1",
                label="legacy metric (analytic denominator)")
    axes[1].bar(pos + 0.2, fixed, 0.4, color="C2",
                label="consistent metric (discrete initial mass)")
    axes[1].axhline(drift_ctrl, color="C0", ls="--", lw=1.2,
                    label=f"control (-3,3) N=100 t=0.3: {drift_ctrl:.1e}")
    axes[1].set_yscale("log")
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels([l.replace("_", " ") for l in labels])
    axes[1].set_ylabel("relative mass drift")
    axes[1].set_title("Benchmark mass metric before / after the fix")
    axes[1].legend(fontsize=7, loc="upper right")
    for i, (l, f) in enumerate(zip(legacy, fixed)):
        axes[1].text(i - 0.2, l * 1.4, f"{l:.1e}", ha="center", fontsize=7)
        axes[1].text(i + 0.2, f * 1.4, f"{f:.1e}", ha="center", fontsize=7)
    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "ritter_mass_metric.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {"results": res, "control_drift": drift_ctrl, "control_steps": steps_ctrl,
            "figure_b64": b64, "figure_path": str(out)}


if __name__ == "__main__":
    d = main()
    print(json.dumps({k: v for k, v in d.items() if k != "figure_b64"}, indent=2, default=str))
    Path("/tmp/ritter_fig_b64.txt").write_text(d["figure_b64"])
    Path("/tmp/ritter_evidence.json").write_text(json.dumps(
        {"results": d["results"], "control_drift": d["control_drift"],
         "control_steps": d["control_steps"]}, indent=2))
