"""Diagnostics for the two CKS1998 gates that failed in the quick profile.

E2 (Sod, relative L2 < 3 %): is the failure a resolution shortfall or a
limiter-stage artefact?  Sweep degree x resolution x limiter_stage and report
the per-field relative L2.

E3 (no new extrema <= 1 %): where does the residual overshoot live?  Report the
extrema of the cell averages and of the polynomial trace per refinement level,
plus the effect of also limiting the initial projection.

Run: OMP_NUM_THREADS=1 python validation/repro_cks1998/diagnose_sod_tvd.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmarks.sod_exact import cell_averages  # noqa: E402
from tensordg.advection_1d import l1_error, solve_advection  # noqa: E402
from tensordg.euler_1d import GAMMA  # noqa: E402
from tensordg.euler_dg_1d import solve_euler_dg_1d  # noqa: E402


def square_wave(x):
    xm = torch.remainder(x, 1.0)
    return ((xm > 0.25) & (xm < 0.75)).to(x.dtype)


def sod_sweep():
    out = []
    for degree, ncells, stage, tvb in [
        (2, 200, "final", 1.0), (2, 400, "final", 1.0), (2, 800, "final", 1.0),
        (2, 400, "every", 1.0), (2, 400, "final", 0.0),
        (1, 400, "final", 1.0), (1, 800, "final", 1.0),
        (0, 800, "final", 1.0), (0, 1600, "final", 1.0),
    ]:
        torch.set_num_threads(1)
        _, coeff, meta = solve_euler_dg_1d(
            (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=ncells, degree=degree,
            final_time=0.2, cfl=0.12, limiter="tvb", tvb_constant=tvb,
            limiter_stage=stage,
        )
        avg = coeff[:, 0]
        rho = avg[:, 0].numpy()
        vel = (avg[:, 1] / avg[:, 0]).numpy()
        pre = ((GAMMA - 1.0) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / avg[:, 0])).numpy()
        _, ref = cell_averages(ncells, 0.2)
        rel = {k: float(np.sqrt(np.mean((got - ref[k]) ** 2)) /
                        np.sqrt(np.mean(ref[k] ** 2)))
               for k, got in (("density", rho), ("velocity", vel), ("pressure", pre))}
        row = {"degree": degree, "ncells": ncells, "stage": stage, "M": tvb,
               "steps": meta["steps"], **{f"rel_{k}": v for k, v in rel.items()},
               "worst": max(rel.values())}
        print(json.dumps(row), flush=True)
        out.append(row)
    return out


def tvd_sweep():
    out = []
    for degree, ncells in [(1, 40), (1, 80), (1, 160), (1, 320),
                           (2, 80), (2, 160), (3, 80)]:
        torch.set_num_threads(1)
        for limiter in ("tvb", None):
            c, coeff, _ = solve_advection(
                ncells=ncells, degree=degree, final_time=1.0, cfl=0.2,
                initial=square_wave, limiter=limiter, tvb_constant=0.0,
                time_integrator="ssp_rk3",
            )
            avg = coeff[:, 0]
            xi = torch.linspace(-1, 1, 41, dtype=coeff.dtype)
            basis = torch.stack([
                torch.as_tensor(np.polynomial.legendre.legval(xi.numpy(),
                                                              np.eye(degree + 1)[k]))
                for k in range(degree + 1)], dim=1)
            vals = coeff @ basis.T
            row = {"degree": degree, "ncells": ncells, "limiter": limiter or "none",
                   "avg_max": float(avg.max()), "avg_min": float(avg.min()),
                   "pt_max": float(vals.max()), "pt_min": float(vals.min()),
                   "argmax_cell": int(avg.argmax()),
                   "overshoot_pct": 100 * max(0.0, float(vals.max()) - 1.0,
                                              -float(vals.min())),
                   "l1": l1_error(c, coeff, 1.0, exact=square_wave,
                                  jumps=(0.25, 0.75))}
            print(json.dumps(row), flush=True)
            out.append(row)
    return out


if __name__ == "__main__":
    torch.set_num_threads(1)
    payload = {"sod": sod_sweep(), "tvd": tvd_sweep()}
    Path(REPO / "runs" / "cks1998").mkdir(parents=True, exist_ok=True)
    (REPO / "runs" / "cks1998" / "diagnostics.json").write_text(
        json.dumps(payload, indent=2))
    print("done")