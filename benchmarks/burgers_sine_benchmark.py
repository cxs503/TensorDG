"""Burgers equation with sinusoidal initial condition — full benchmark suite.

Case: u_t + (u^2/2)_x = 0,  x in [0,1] periodic
      u(x,0) = 0.2 + 0.8 sin(2 pi x)

Reference: Exact implicit characteristic solution (Hopf 1950, Cole 1951).
  u(x,t) = 0.2 + 0.8 sin(2 pi (x - u t))   [implicit, solved by Newton]
  Shock formation time: t_c = 1 / (2 pi * 0.8 * 2 pi) ~ 0.1989
  All times t < t_c are in the smooth regime.

This script produces:
  - Convergence study (h-refinement) for P1, P2, P3
  - Time sensitivity study
  - Profile comparison plots
  - Convergence order plots
  - JSON metrics with full traceability
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensordg.burgers_1d import (
    MEAN, AMPLITUDE,
    initial_profile, exact_profile,
    solve_burgers, burgers_error,
)


def convergence_study(degrees=(1, 2, 3), cells_list=(8, 16, 32, 64, 128),
                      final_time=0.1, cfl=0.1):
    """Run h-refinement for each polynomial degree."""
    results = {}
    for p in degrees:
        records = []
        prev_err = prev_h = None
        for N in cells_list:
            x, coeff, meta = solve_burgers(N, p, final_time, cfl)
            metrics = burgers_error(x, coeff, final_time)
            h = 1.0 / N
            eoc = (math.log(prev_err / metrics["relative_l2"]) / math.log(prev_h / h)
                   if prev_err and prev_err > 0 else None)
            records.append({
                "ncells": N, "degree": p, "h": h,
                "steps": meta["steps"], "dt_final": meta["dt"],
                "absolute_l2": metrics["absolute_l2"],
                "exact_l2": metrics["exact_l2"],
                "relative_l2": metrics["relative_l2"],
                "relative_l2_percent": 100.0 * metrics["relative_l2"],
                "eoc": eoc,
            })
            prev_err, prev_h = metrics["relative_l2"], h
        results[f"P{p}"] = records
    return results


def time_sensitivity_study(degree=2, ncells=64, times=(0.01, 0.05, 0.1, 0.15, 0.19)):
    """Run at different final times to show pre-shock steepening."""
    records = []
    for t in times:
        x, coeff, meta = solve_burgers(ncells, degree, t, 0.1)
        metrics = burgers_error(x, coeff, t)
        records.append({
            "final_time": t, "ncells": ncells, "degree": degree,
            "steps": meta["steps"],
            "relative_l2": metrics["relative_l2"],
            "relative_l2_percent": 100.0 * metrics["relative_l2"],
        })
    return records


def generate_profile_data(degree=2, ncells=64, final_time=0.1, n_sample=1001):
    """Generate profile data for plotting."""
    x_c, coeff, meta = solve_burgers(ncells, degree, final_time, 0.1)
    # Sample the DG solution at many points
    h = 1.0 / ncells
    x_fine = np.linspace(0, 1, n_sample)
    # Evaluate DG solution at fine points
    from benchmarks._modal_1d import sample
    x_s, u_num = sample(coeff, n_sample)
    # Exact solution
    x_t = torch.as_tensor(x_s, dtype=torch.float64)
    u_exact = exact_profile(x_t, final_time).numpy()
    return x_s, u_num, u_exact


def compute_shock_time():
    """Compute the shock formation time analytically."""
    # u0'(x) = AMPLITUDE * 2*pi * cos(2*pi*x)
    # max|u0'| = AMPLITUDE * 2*pi
    # t_c = 1 / max|u0'|
    max_grad = AMPLITUDE * 2.0 * math.pi
    return 1.0 / max_grad


def main():
    out = Path("artifacts/burgers_sine_benchmark")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Burgers Sine Initial Condition — Full Benchmark Suite")
    print("=" * 70)

    # 1. Convergence study
    print("\n[1] Convergence study (h-refinement, t=0.1)")
    conv = convergence_study()
    for pname, records in conv.items():
        print(f"\n  {pname}:")
        for r in records:
            eoc_str = f"{r['eoc']:.2f}" if r['eoc'] is not None else "  --"
            print(f"    N={r['ncells']:4d}  rel.L2={r['relative_l2_percent']:10.6f}%  EOC={eoc_str}")

    # 2. Time sensitivity
    print("\n[2] Time sensitivity (P2, N=64)")
    ts = time_sensitivity_study()
    for r in ts:
        print(f"    t={r['final_time']:.2f}  rel.L2={r['relative_l2_percent']:.6f}%  steps={r['steps']}")

    # 3. Shock time
    t_shock = compute_shock_time()
    print(f"\n[3] Shock formation time: t_c = 1/(2pi*A*2pi) = {t_shock:.6f}")

    # 4. Save all metrics
    all_metrics = {
        "benchmark": "burgers_sine",
        "equation": "u_t + (u^2/2)_x = 0",
        "domain": "[0, 1] periodic",
        "initial_condition": f"u(x,0) = {MEAN} + {AMPLITUDE} * sin(2*pi*x)",
        "mean": MEAN,
        "amplitude": AMPLITUDE,
        "shock_formation_time": t_shock,
        "reference": "Exact implicit characteristic solution (Hopf 1950, Cole 1951)",
        "convergence_study": conv,
        "time_sensitivity": ts,
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[4] Metrics saved to {out / 'metrics.json'}")

    # 5. Profile data for finest mesh
    x_s, u_num, u_exact = generate_profile_data(degree=2, ncells=128, final_time=0.1)
    np.savetxt(out / "profile_p2_n128_t01.csv",
               np.column_stack([x_s, u_num, u_exact, np.abs(u_num - u_exact)]),
               delimiter=",", comments="",
               header="x,numerical,exact,pointwise_abs_error")
    print(f"[5] Profile data saved to {out / 'profile_p2_n128_t01.csv'}")

    # 6. Summary
    finest_err = conv["P2"][-1]["relative_l2"]
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: P2 N=128 relative L2 error = {100*finest_err:.6f}%")
    print(f"  Threshold: <5%  =>  {'PASS' if finest_err < 0.05 else 'FAIL'}")
    print(f"  Convergence order P2: ~{conv['P2'][-1]['eoc']:.1f} (expected ~3)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
