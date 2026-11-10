"""Generate the numbers and the lake-at-rest figure for the helper-contract
fix report (mission m9711).

Everything printed/written here is computed at run time -- nothing is
hand-copied into the report.

Run:  PYTHONPATH=. python validation/swe_thacker/helper_contract_report_data.py
Outputs:
  docs/figures/swe_wb_lake_at_rest_check.png
  /tmp/report_numbers.json     (consumed by the report builder)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg.swe_1d import (
    create_hydrostatic_initial_conditions,
    estimate_well_balanced_error_bound,
    solve_swe_1d,
    validate_well_balanced_setup,
)

OUT_FIG = Path("docs/figures/swe_wb_lake_at_rest_check.png")
OUT_JSON = Path("/tmp/report_numbers.json")
G = 9.81
ETA0 = 2.0


def zb_curved(x):
    return 0.3 * torch.cos(2.0 * math.pi * x)


def helper_fix_numbers():
    """Before/after numbers for the two helper defects."""
    rows = []
    for ncells in (8, 16, 32, 64, 128, 256):
        for degree in (1, 2):
            b = estimate_well_balanced_error_bound(
                ncells, degree, 1.0, bed_curvature=1.0)
            rows.append({
                "ncells": ncells,
                "degree": degree,
                "dx": b["dx"],
                "estimate_after": b["eta_error_estimate"],
                "returned_before": min(b["eta_error_estimate"], 1e-10),
            })

    accept = []
    for amp in (0.0, 1e-8, 1e-5, 1e-3):
        def initial_h(xx, amp=amp):
            return ETA0 - zb_curved(xx) + amp * torch.sin(2.0 * math.pi * xx)

        res = validate_well_balanced_setup(
            initial_h, lambda xx: torch.zeros_like(xx), zb_curved)
        accept.append({
            "amplitude": amp,
            "eta_deviation_max": res["eta_deviation_max"],
            "eta_variance": res["eta_variance"],
            "accepted_before_variance": bool(res["eta_variance"] < 1e-10),
            "accepted_after_supnorm": bool(res["is_valid"]),
        })
    return rows, accept


def lake_at_rest():
    """Well-balanced vs standard path on the same lake-at-rest initial data."""
    ncells, degree, tfinal = 32, 1, 0.5
    initial_h, initial_u = create_hydrostatic_initial_conditions(zb_curved, ETA0)

    out = {}
    profiles = {}
    for tag, wb in (("well_balanced", True), ("standard", False)):
        x, coeff, zb_faces, meta = solve_swe_1d(
            initial_h, initial_u, zb_curved,
            ncells=ncells, degree=degree, final_time=tfinal,
            domain=(0.0, 1.0), cfl=0.3, well_balanced=wb,
        )
        x_np = x.detach().cpu().numpy().ravel()
        h = coeff[:, 0, 0].detach().cpu().numpy().ravel()
        hu = coeff[:, 0, 1].detach().cpu().numpy().ravel()
        bed = (0.5 * (zb_faces[:, 0] + zb_faces[:, 1])).detach().cpu().numpy().ravel()
        u = np.where(h > 1e-9, hu / np.maximum(h, 1e-9), 0.0)
        eta = h + bed
        out[tag] = {
            "eta_error_max": float(np.abs(eta - ETA0).max()),
            "u_max": float(np.abs(u).max()),
            "ncells": ncells,
            "degree": degree,
            "final_time": tfinal,
        }
        profiles[tag] = (x_np, eta, bed, u)

    out["reduction_eta"] = (out["standard"]["eta_error_max"]
                            / out["well_balanced"]["eta_error_max"])
    out["reduction_u"] = (out["standard"]["u_max"]
                          / max(out["well_balanced"]["u_max"], 1e-30))

    # ---- figure: analytic comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    x_np, eta_wb, bed, _ = profiles["well_balanced"]
    _, eta_std, _, _ = profiles["standard"]
    ax.plot(x_np, bed, "k-", lw=1.2, label=r"bed $z_b=0.3\cos(2\pi x)$")
    ax.plot(x_np, eta_wb, "o", ms=3, color="seagreen",
            label="well_balanced=True $\\eta$")
    ax.plot(x_np, eta_std, "s", ms=3, color="crimson", mfc="none",
            label="well_balanced=False $\\eta$")
    ax.axhline(ETA0, color="0.3", ls="--", lw=1.2,
               label="exact lake at rest $\\eta_0=2$ m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\eta=h+z_b$ [m]")
    ax.set_title(f"(a) free surface, N={ncells}, P{degree}, t={tfinal} s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower center")

    ax = axes[1]
    labels = ["max $|\\eta-\\eta_0|$ [m]", "max $|u|$ [m/s]"]
    wb_vals = [out["well_balanced"]["eta_error_max"], out["well_balanced"]["u_max"]]
    std_vals = [out["standard"]["eta_error_max"], out["standard"]["u_max"]]
    idx = np.arange(len(labels))
    ax.bar(idx - 0.2, wb_vals, 0.4, color="seagreen", label="well_balanced=True")
    ax.bar(idx + 0.2, std_vals, 0.4, color="crimson", label="well_balanced=False")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("(b) residual vs analytic solution")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    print(f"wrote {OUT_FIG}")
    return out


def main():
    rows, accept = helper_fix_numbers()
    lake = lake_at_rest()
    payload = {"bound_rows": rows, "validation_acceptance": accept,
               "lake_at_rest": lake}
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()