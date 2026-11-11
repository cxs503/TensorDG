"""m15113: regenerate the two stable oblique-shock cases (params identical to
docs/benchmarks/results/oblique_shock/results.json) and emit report figures.

Outputs (worktree-relative):
  artifacts/m15113/wedge_M2_theta15_density.png
  artifacts/m15113/normal_M4_beta90_density.png
  artifacts/m15113/figs.json   (record subset for the report, zero hand-copy)
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.oblique_shock import (
    Case,
    _plot_case,
    run_case,
)

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "m15113"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    # params copied from the committed results.json (wedge row)
    Case("wedge_M2_theta15", "M=2 wedge class, theta=15 deg (PR #247 claimed case)",
         "wedge", 2.0, 15.0, ncells=40, degree=1, final_time=0.03, cfl=0.2,
         tvb_constant=100.0),
    # params copied from the committed results.json (normal row)
    Case("normal_M4_beta90", "grid-aligned control",
         "beta", 4.0, 90.0, ncells=24, degree=1, final_time=0.02, cfl=0.1,
         tvb_constant=1.0),
]

records = []
for case in CASES:
    record, measured, solution, _ = run_case(case)
    png = _plot_case(record, measured, solution, OUT)
    record["figure"] = str(png)
    records.append(record)
    print(f"[{record['case']}] beta_err={record['beta_error_deg']:.4f} deg "
          f"rho_err={record['rho_ratio_error']:.3e} "
          f"p_err={record['p_ratio_error']:.3e} "
          f"theta_err={record['theta_error_deg']:.4f} deg "
          f"preserve={record['preserve_l1']:.4f} gates={record['gates']}",
          flush=True)

slim = {r["case"]: {k: r[k] for k in (
    "beta_deg", "beta_hat_deg", "beta_error_deg", "rho_ratio_exact",
    "rho_ratio_hat", "rho_ratio_error", "p_ratio_exact", "p_ratio_hat",
    "p_ratio_error", "theta_exact_deg", "theta_hat_deg", "theta_error_deg",
    "preserve_l1", "steps", "gates", "figure")} for r in records}
with open(OUT / "figs.json", "w") as handle:
    json.dump(slim, handle, indent=2)
print("done", flush=True)
