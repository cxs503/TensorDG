"""TVB-constant sensitivity sweep for the TensorDG shock limiters.

The development report for the TVB/WENO limiters (m274) flagged two open
issues: the limiter increased the shock-tube error and the recommended fix was
"problem-specific tuning of M".  This script answers that question with data:
it sweeps the TVB constant ``M`` (threshold ``M * dx^2``, Cockburn & Shu 1989)
for both limiters, records

* the relative L2 error of rho/u/p against the exact Riemann solution,
* the cell-average L1 error, the total variation and the overshoot,
* the mean fraction of cells flagged as troubled (limiter activity),

so that the error can be traded against limiter activity instead of guessed.

Outputs ``artifacts/limiter_validation/tvb_sweep.json`` and a figure.

    PYTHONPATH=. python benchmarks/limiter_tvb_constant_sweep.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

from benchmarks.riemann_dg_error import riemann_dg_errors
from tensordg import limiters as limiters_module
from tensordg.euler_dg_1d import solve_euler_dg_1d

CASES = {
    "sod": {"left": (1.0, 0.0, 1.0), "right": (0.125, 0.0, 0.1), "time": 0.2},
    "lax": {"left": (0.445, 0.698, 3.528), "right": (0.5, 0.0, 0.571), "time": 0.13},
}


class _ActivityRecorder:
    """Wrapper around the troubled-cell indicator that records limiter activity."""

    def __init__(self, original):
        self.original = original
        self.reset()

    def reset(self):
        self.calls = 0
        self.flagged = 0
        self.cells = 0

    def __call__(self, coeff, dx, tvb_constant=0.0):
        mask = self.original(coeff, dx, tvb_constant=tvb_constant)
        self.calls += 1
        self.flagged += int(mask.sum())
        self.cells += int(mask.numel())
        return mask

    @property
    def mean_flagged_fraction(self):
        return self.flagged / self.cells if self.cells else 0.0


def _total_variation(values):
    return float(np.abs(np.diff(values)).sum())


def run_config(case, limiter, tvb_constant, recorder, ncells, cfl, degree):
    spec = CASES[case]
    left, right, final_time = spec["left"], spec["right"], spec["time"]
    recorder.reset()
    x, coeff, meta = solve_euler_dg_1d(
        left, right, ncells=ncells, degree=degree, final_time=final_time,
        cfl=cfl, limiter=limiter, tvb_constant=tvb_constant,
    )
    errors = riemann_dg_errors(x, coeff, final_time, left, right)
    rho = coeff.detach().cpu().numpy()[:, 0, 0]
    dx = 1.0 / ncells
    return {
        "limiter": limiter or "none",
        "tvb_constant": tvb_constant,
        "steps": meta["steps"],
        "errors_percent": {key: errors[key]["relative_l2_percent"] for key in errors},
        "density_total_variation": _total_variation(rho),
        "density_overshoot": float(np.max(rho) - max(left[0], right[0])),
        "mean_flagged_fraction": recorder.mean_flagged_fraction,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--output-dir", default="artifacts/limiter_validation")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    recorder = _ActivityRecorder(limiters_module._troubled_cell_indicator)
    original = limiters_module._troubled_cell_indicator
    limiters_module._troubled_cell_indicator = recorder

    constants = [0.0, 1.0, 10.0, 100.0, 1000.0]
    plan = [("sod", None, 1.0), ("sod", "tvb", None), ("sod", "weno", None),
            ("lax", None, 1.0), ("lax", "tvb", None), ("lax", "weno", None)]

    results = {}
    try:
        for case, limiter, fixed in plan:
            key = f"{case}_{limiter or 'none'}"
            results[key] = []
            values = [fixed] if fixed is not None else constants
            for M in values:
                record = run_config(case, limiter, M, recorder, args.cells, args.cfl, args.degree)
                results[key].append(record)
                print(f"{key:12s} M={M:<7g} rho={record['errors_percent']['density']:7.3f}% "
                      f"u={record['errors_percent']['velocity']:7.3f}% "
                      f"flagged={100*record['mean_flagged_fraction']:5.1f}%", flush=True)
    finally:
        limiters_module._troubled_cell_indicator = original

    with open(out / "tvb_sweep.json", "w") as handle:
        json.dump({"config": vars(args), "results": results}, handle, indent=2, default=str)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    styles = {"tvb": "tab:blue", "weno": "tab:green"}
    markers = {"tvb": "o", "weno": "s"}
    for case in CASES:
        ax = axes[0] if case == "sod" else axes[1]
        baseline = results[f"{case}_none"][0]
        ax.axhline(baseline["errors_percent"]["density"], color="tab:red", ls="--",
                   label="no limiter (density)")
        for limiter in ("tvb", "weno"):
            records = results[f"{case}_{limiter}"]
            Ms = np.maximum([r["tvb_constant"] for r in records], 1e-2)
            ax.plot(Ms, [r["errors_percent"]["density"] for r in records],
                    styles[limiter] + markers[limiter] + "-", label=f"{limiter} density")
        ax.set_xscale("symlog", linthresh=1e-1)
        ax.set_xlabel("TVB constant M (threshold M·dx²)")
        ax.set_ylabel("relative L2 error of rho [%]")
        ax.set_title(f"{case.upper()} ({args.cells} cells, P{args.degree})")
        ax.axhline(3.0, color="k", ls=":", lw=1, label="3% tolerance")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out / "tvb_sweep.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for limiter in ("tvb", "weno"):
        records = results[f"sod_{limiter}"]
        Ms = np.maximum([r["tvb_constant"] for r in records], 1e-2)
        ax.semilogx(Ms, [100 * r["mean_flagged_fraction"] for r in records],
                    styles[limiter] + markers[limiter] + "-", label=limiter)
    ax.set_xlabel("TVB constant M")
    ax.set_ylabel("mean flagged cells per sub-step [%]")
    ax.set_title("Limiter activity, SOD (%d cells, P%d)" % (args.cells, args.degree))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out / "tvb_activity.png", dpi=160)
    plt.close(fig)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()