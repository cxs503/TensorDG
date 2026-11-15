"""CI-scale certification run of the no-MCMP two-phase XVAL.

Same code path and same summary schema as
``benchmarks/phasefield_laplace_xval.py`` (the leg functions are imported, not
re-implemented), but sized to fit a short local budget so that the committed
evidence is reproducible without a cluster slot.  The full-resolution scan is
submitted separately and lands in ``summary_full.json``.

    python benchmarks/phasefield_certify_ci.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import phasefield_laplace_xval as xval  # noqa: E402


def main() -> int:
    torch.set_num_threads(1)
    os.makedirs(xval.OUTDIR, exist_ok=True)
    summary = {
        "anchor": "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY",
        "model": "phase-field / Allen-Cahn two-phase LBM (single component, no MCMP)",
        "upstream_pattern": "lssfau/walberla two-phase (no MCMP) tree: phase-field Allen-Cahn showcase",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "quick": True,
        "scale": "CI certification (n<=64, <=2000 steps per case)",
        "tolerances": {"sigma_profile": xval.TOL_SIGMA_PROFILE,
                       "laplace": xval.TOL_LAPLACE,
                       "laplace_target": xval.TARGET_LAPLACE,
                       "dpR_dispersion": xval.TOL_DPR_DISPERSION},
        "parameters": {"sigma_target": xval.SIGMA_TARGET, "ell_target": xval.ELL_TARGET,
                       "mobility": xval.MOBILITY, "omega": xval.OMEGA,
                       "advection": "central", "dtype": "float64"},
    }

    def dump():
        with open(os.path.join(xval.OUTDIR, "summary_ci.json"), "w") as fh:
            json.dump(summary, fh, indent=1)

    print("== Leg A ==", flush=True)
    summary["leg_a_planar"] = [xval.leg_a_planar(64, 1000)]
    dump()

    print("== Leg B ==", flush=True)
    drops = [xval.leg_b_droplet(64, R, 1500, capture=(R == 14.0)) for R in (12.0, 14.0)]
    summary["leg_b_droplet"] = drops
    ratios = [d["ratio_dp_over_sigma_R"] for d in drops]
    prod = [d["dp_times_R"] for d in drops]
    mean_prod = sum(prod) / len(prod)
    summary["leg_b_summary"] = {
        "ratio_mean": sum(ratios) / len(ratios),
        "ratio_max_abs_dev_from_1": max(abs(v - 1.0) for v in ratios),
        "dpR_mean": mean_prod,
        "dpR_dispersion": max(abs(v - mean_prod) / mean_prod for v in prod),
        "max_rel_error_vs_sigma_over_R": max(d["rel_error_vs_sigma_over_R"] for d in drops),
        "max_spurious_current": max(d["spurious_current"] for d in drops),
        "pass_15pct": bool(max(abs(v - 1.0) for v in ratios) <= xval.TOL_LAPLACE),
        "pass_3pct": bool(max(abs(v - 1.0) for v in ratios) <= xval.TARGET_LAPLACE),
        "pass_dpR_dispersion": bool(max(abs(v - mean_prod) / mean_prod for v in prod)
                                    <= xval.TOL_DPR_DISPERSION),
    }
    print("  " + json.dumps(summary["leg_b_summary"], default=str), flush=True)
    dump()

    print("== Leg C ==", flush=True)
    summary["leg_c_ell_study"] = [xval.leg_b_droplet(64, 16.0, 800, ell=e)
                                  for e in (3.0, 4.0, 6.0)]
    dump()

    print("== Leg D ==", flush=True)
    summary["leg_d_force_forms"] = [
        xval.leg_b_droplet(64, 14.0, 600, force_form="mu_grad_phi"),
        xval.leg_b_droplet(64, 14.0, 400, force_form="phi_grad_mu"),
    ]

    path = os.path.join(xval.OUTDIR, "summary_ci.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())