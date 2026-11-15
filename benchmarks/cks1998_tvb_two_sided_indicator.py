"""CKS1998 square wave: one-sided vs two-sided TVB indicator (mission m21413).

Measures, for the CKS1998 §5 square-wave test (``u_t + u_x = 0``, periodic,
``u0 = 1`` on ``[0.25, 0.75]`` and 0 elsewhere, SSP-RK3, ``M = 0``):

* the pointwise (nodal) overshoot of the approximate solution above the exact
  data range ``[0, 1]`` -- the quantity the ``REPRO-DG-CKS1998`` gate
  "TVD 界无振荡, 无超调 > 1 %" is evaluated on;
* the cell-average overshoot (the quantity the TVD theory actually controls);
* the L1 error of the cell averages and its mesh-refinement order.

``--mode one_sided`` restores the pre-m21413 indicator
(``deviation = |d_left - minmod|``) by monkey-patching
``tensordg.limiters.troubled_from_differences``, so the baseline is produced by
the same driver, the same solver and the same metrics as the shipped
two-sided indicator.

Usage
-----
    python benchmarks/cks1998_tvb_two_sided_indicator.py --out artifacts/tvb_indicator.json
    python benchmarks/cks1998_tvb_two_sided_indicator.py --modes two_sided --degrees 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensordg.limiters as limiters_mod  # noqa: E402
from tensordg.advection_1d import solve_advection  # noqa: E402
from tensordg.tvb import troubled_from_differences as two_sided  # noqa: E402

SHIPPED = two_sided


def _one_sided_from_differences(d_left, d_right, spacing, tvb_constant):
    """Pre-m21413 indicator: deviation measured on ``d_left`` only."""
    a_left, a_right = d_left.abs(), d_right.abs()
    mm = 0.5 * (torch.sign(d_left) + torch.sign(d_right)) * torch.minimum(a_left, a_right)
    deviation = (d_left - mm).abs().max(dim=-1).values
    threshold = tvb_constant * spacing * spacing
    scale = torch.maximum(a_left, a_right).max(dim=-1).values
    troubled = deviation > threshold + 1e-12 * scale.clamp_min(1.0)
    return troubled, mm


MODES = {"two_sided": SHIPPED, "one_sided": _one_sided_from_differences}


def square_wave(x):
    return ((x >= 0.25) & (x <= 0.75)).to(x.dtype)


def nodal_values(coeff: torch.Tensor, npts: int = 41) -> np.ndarray:
    nmodes = coeff.shape[1]
    xi = torch.linspace(-1.0, 1.0, npts, dtype=coeff.dtype)
    basis = torch.stack(
        [torch.as_tensor(np.polynomial.legendre.legval(xi.numpy(), np.eye(nmodes)[k]))
         for k in range(nmodes)], dim=1)
    return (coeff @ basis.T).numpy()


def run_case(degree: int, ncells: int, tvb_constant: float = 0.0) -> dict:
    t0 = time.time()
    centres, coeff, meta = solve_advection(
        ncells=ncells, degree=degree, final_time=1.0, cfl=0.2,
        initial=square_wave, limiter="tvb", tvb_constant=tvb_constant,
        time_integrator="ssp_rk3",
    )
    values = nodal_values(coeff)
    avg = coeff[:, 0].numpy()
    # u0 advects by speed 1 over t = 1: the exact cell averages are u0(x - 1).
    x = centres.numpy()
    exact = np.where(((x - 1.0) % 1.0 >= 0.25) & ((x - 1.0) % 1.0 <= 0.75), 1.0, 0.0)
    return {
        "degree": degree,
        "ncells": ncells,
        "h": 1.0 / ncells,
        "tvb_constant": tvb_constant,
        "pointwise_max": float(values.max()),
        "pointwise_min": float(values.min()),
        "cell_average_max": float(avg.max()),
        "cell_average_min": float(avg.min()),
        "overshoot_pointwise_pct": 100.0 * max(0.0, float(values.max()) - 1.0, -float(values.min())),
        "overshoot_cell_average_pct": 100.0 * max(0.0, float(avg.max()) - 1.0, -float(avg.min())),
        "l1_cell_average": float(np.abs(avg - exact).mean()),
        "steps": int(meta["steps"]),
        "seconds": round(time.time() - t0, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", nargs="+", default=["two_sided", "one_sided"], choices=sorted(MODES))
    ap.add_argument("--degrees", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--cells", nargs="+", type=int, default=[50, 100, 200, 400])
    ap.add_argument("--tvb-constant", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    payload = {"case": "CKS1998 square wave, limited DG, M=%g" % args.tvb_constant,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": {}}
    for mode in args.modes:
        limiters_mod.troubled_from_differences = MODES[mode]
        rows = []
        for degree in args.degrees:
            for ncells in args.cells:
                row = run_case(degree, ncells, args.tvb_constant)
                row["mode"] = mode
                rows.append(row)
                print(f"{mode:9s} P{degree} N={ncells:4d} "
                      f"pt_over={row['overshoot_pointwise_pct']:7.4f}% "
                      f"avg_over={row['overshoot_cell_average_pct']:7.4f}% "
                      f"L1={row['l1_cell_average']:.6f} ({row['seconds']}s)", flush=True)
        # refinement order of the L1 error per degree
        for degree in args.degrees:
            series = [r for r in rows if r["degree"] == degree]
            series.sort(key=lambda r: r["ncells"])
            errs = [r["l1_cell_average"] for r in series]
            orders = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
            for row in series:
                row["l1_order_pairwise"] = orders
            payload.setdefault("orders", {})[f"{mode}_p{degree}"] = orders
            print(f"{mode:9s} P{degree} L1 orders {['%.4f' % o for o in orders]}", flush=True)
        payload["runs"][mode] = rows
    limiters_mod.troubled_from_differences = SHIPPED

    payload["gate"] = {"name": "REPRO-DG-CKS1998 TVD no-overshoot > 1%",
                       "threshold_pct": 1.0,
                       "evaluated_on": "pointwise (nodal) envelope of u_h"}
    payload["gate_pass"] = {
        mode: all(r["overshoot_pointwise_pct"] <= 1.0 for r in rows)
        for mode, rows in payload["runs"].items()
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())