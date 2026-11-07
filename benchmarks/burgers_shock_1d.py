"""Burgers shock-capturing verification study (IN16-K7).

Three acceptance questions, each with its own machine-checkable metric:

1. **Design order in the smooth region is preserved.**  Periodic sine data well
   before breaking time, P1/P2/P3, both interface fluxes; the observed order
   must stay at or above ``p + 0.9``.
2. **The captured shock sits at the Rankine--Hugoniot location and the error
   converges under refinement.**  Riemann shock data, three independent shock
   locators (level crossing, wave-front centroid, mass/RH inversion).
3. **The Gibbs oscillation is bounded by shock capturing.**  Same shock data
   with and without the TVB limiter; the overshoot outside the exact solution
   range is compared on the same meshes.

Writes ``metrics.json``, profile CSV and four figures.  Exit code is non-zero if
any acceptance criterion fails.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tensordg.burgers_1d import burgers_error, initial_profile as smooth_profile
from tensordg.burgers_shock_1d import (
    RiemannData,
    exact_solution,
    gibbs_overshoot,
    solve_burgers_shock,
    total_variation,
)
from tensordg.burgers_shock_1d import _operators

ORDER_MARGIN = 0.9          # required observed order >= p + ORDER_MARGIN
GIBBS_BOUND = 0.07          # absolute overshoot bound for the limited scheme
GIBBS_REDUCTION = 3.0       # limited overshoot must be this much smaller
RH_TOLERANCE = 1e-8         # mass-based shock position tolerance


def smooth_order_study(cfl: float) -> dict:
    """Design-order study in the smooth (pre-shock) regime."""
    cases = {
        1: (32, 64, 128, 256),
        2: (32, 64, 128, 256),
        3: (32, 64, 128, 256),
    }
    records = []
    for degree, meshes in cases.items():
        for flux in ("lax_friedrichs", "roe"):
            errors = []
            for cells in meshes:
                x, coeff, meta = solve_burgers_shock(
                    ncells=cells, degree=degree, final_time=0.1, cfl=cfl,
                    flux=flux, periodic=True, initial=smooth_profile,
                )
                errors.append(burgers_error(x, coeff, 0.1)["relative_l2"])
            rates = [math.log2(errors[i] / errors[i + 1])
                     for i in range(len(errors) - 1)]
            finest = [math.log2(errors[-2] / errors[-1])][0]
            mean_rate = float(np.mean(rates[-2:]))
            records.append({
                "degree": degree, "flux": flux, "cells": list(meshes),
                "relative_l2": errors, "pairwise_orders": rates,
                "finest_pair_order": finest, "mean_order_finest_two": mean_rate,
                "required_order": degree + ORDER_MARGIN,
                "passed": min(finest, mean_rate) >= degree + ORDER_MARGIN,
            })
            print(f"  P{degree} {flux:14s} orders={['%.2f' % r for r in rates]} "
                  f"required>={degree + ORDER_MARGIN:.2f}")
    return {"cases": records,
            "passed": all(record["passed"] for record in records)}


def shock_study(meshes, degree: int, final_time: float, cfl: float, data) -> dict:
    """Shock position and Gibbs oscillation before/after limiting."""
    records = []
    profiles = {}
    for limiter in (None, "tvb"):
        for flux in ("lax_friedrichs", "roe"):
            for cells in meshes:
                x, coeff, meta = solve_burgers_shock(
                    ncells=cells, degree=degree, final_time=final_time, cfl=cfl,
                    flux=flux, limiter=limiter, data=data,
                )
                ops = _operators(degree, torch.device("cpu"), torch.float64)
                row = {
                    "limiter": limiter, "flux": flux, "cells": cells,
                    "steps": meta["steps"],
                    "shock_position": meta["shock_position"],
                    "shock_position_exact": meta["shock_position_exact"],
                    "level_crossing_error": meta["shock_position_error"],
                    "centroid_error": meta["shock_position_moment_error"],
                    "rankine_hugoniot_error": meta["shock_position_from_mass_error"],
                    "conservation_defect": meta["conservation_defect"],
                    "gibbs_overshoot": gibbs_overshoot(coeff, ops[2],
                                                       min(data.u_left, data.u_right),
                                                       max(data.u_left, data.u_right)),
                    "total_variation": total_variation(coeff, ops[2], meta["dx"]),
                }
                row["total_variation_exact"] = abs(data.u_left - data.u_right)
                # profile error against the exact solution's cell averages
                exact_avg = _exact_cell_averages(coeff.shape[0], meta["dx"], data,
                                                 final_time)
                deviation = (coeff[:, 0] - exact_avg).abs()
                jump = abs(data.u_left - data.u_right)
                row["l1_error"] = float(deviation.mean().item())
                row["linf_error"] = float(deviation.max().item())
                row["l1_error_percent_of_jump"] = 100.0 * row["l1_error"] / jump
                row["relative_l2_profile"] = float(
                    torch.sqrt((deviation ** 2).mean()).item()
                    / torch.sqrt((exact_avg ** 2).mean()).item())
                row["linf_error_percent_of_jump"] = 100.0 * row["linf_error"] / jump
                records.append(row)
                profiles[(limiter, flux, cells)] = (x, coeff)
                print(f"  limiter={str(limiter):5s} {flux:14s} n={cells:4d} "
                      f"xs_err={row['level_crossing_error']:.2e} "
                      f"RH_err={row['rankine_hugoniot_error']:.2e} "
                      f"gibbs={row['gibbs_overshoot']:.4f} "
                      f"L1={row['l1_error_percent_of_jump']:.3f}%")
    # acceptance: RH exactness, error decay, Gibbs bound and reduction
    rh_ok = all(r["rankine_hugoniot_error"] < RH_TOLERANCE for r in records)
    unlimited = [r for r in records if r["limiter"] is None]
    limited = [r for r in records if r["limiter"] == "tvb"]
    unlimited_max = max(r["gibbs_overshoot"] for r in unlimited)
    limited_max = max(r["gibbs_overshoot"] for r in limited)
    gibbs_ok = bool(limited_max < GIBBS_BOUND and
                    unlimited_max > GIBBS_REDUCTION * limited_max)
    decay = {}
    for flux in ("lax_friedrichs", "roe"):
        for limiter in (None, "tvb"):
            series = sorted(
                [r for r in records if r["flux"] == flux and r["limiter"] == limiter],
                key=lambda r: r["cells"])
            coarse, fine = series[0]["centroid_error"], series[-1]["centroid_error"]
            order = (math.log(coarse / fine) / math.log(series[-1]["cells"] / series[0]["cells"])
                     if fine > 0.0 else float("inf"))
            decay[f"{flux}|{limiter}"] = {
                "cells": [series[0]["cells"], series[-1]["cells"]],
                "centroid_error_coarse": coarse, "centroid_error_fine": fine,
                "observed_order": order, "passed": fine < coarse,
            }
    return {"records": records, "profiles": profiles,
            "rankine_hugoniot_passed": rh_ok,
            "gibbs_limited_max": limited_max, "gibbs_unlimited_max": unlimited_max,
            "gibbs_passed": gibbs_ok, "decay": decay,
            "decay_passed": all(item["passed"] for item in decay.values()),
            "passed": bool(rh_ok and gibbs_ok and all(i["passed"] for i in decay.values()))}


def _exact_cell_averages(ncells, dx, data, final_time):
    """Exact entropy-solution cell averages (128-point sub-sampling per cell)."""
    sub = 128
    offsets = (torch.arange(sub, dtype=torch.float64) + 0.5) / sub
    left = torch.arange(ncells, dtype=torch.float64) * dx
    points = left[:, None] + offsets[None, :] * dx
    values = exact_solution(points.flatten(), final_time, data)
    return values.reshape(ncells, sub).mean(dim=1)


def make_figures(out: Path, shock: dict, order: dict, data, degree, final_time):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cells = max(r["cells"] for r in shock["records"])
    for limiter, style in ((None, "-"), ("tvb", "--")):
        x, coeff = shock["profiles"][(limiter, "lax_friedrichs", cells)]
        ops = _operators(degree, torch.device("cpu"), torch.float64)
        nodes = np.polynomial.legendre.leggauss(max(2 * degree + 3, 4))[0]
        values = torch.einsum("cmv,qm->cqv", coeff.unsqueeze(-1), ops[2]).squeeze(-1).numpy()
        dx = 1.0 / cells
        xq = (np.arange(cells)[:, None] + 0.5 * (nodes[None, :] + 1.0)) * dx
        axes[0].plot(xq.flatten(), values.flatten(), style, lw=1.2,
                     label="unlimited P%d" % degree if limiter is None else "TVB-limited P%d" % degree)
    grid = np.linspace(0.0, 1.0, 4001)
    exact = exact_solution(torch.as_tensor(grid), final_time, data).numpy()
    axes[0].plot(grid, exact, "k", lw=1.0, label="exact (Rankine--Hugoniot)")
    axes[0].set_xlim(data.shock_location(final_time) - 0.12, data.shock_location(final_time) + 0.12)
    axes[0].set_ylim(-0.3, 1.3)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].set_title("Burgers shock, %d cells, t=%g" % (cells, final_time))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for limiter, marker, label in ((None, "o", "unlimited"), ("tvb", "s", "TVB-limited")):
        series = sorted([r for r in shock["records"]
                         if r["limiter"] == limiter and r["flux"] == "lax_friedrichs"],
                        key=lambda r: r["cells"])
        axes[1].loglog([r["cells"] for r in series],
                       [max(r["gibbs_overshoot"], 1e-6) for r in series],
                       marker + "-", label=label)
    axes[1].set_xlabel("cells")
    axes[1].set_ylabel("Gibbs overshoot (|u| outside exact range)")
    axes[1].set_title("Oscillation before/after shock capturing")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig1_profiles_gibbs.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for flux, marker in (("lax_friedrichs", "o"), ("roe", "s")):
        series = sorted([r for r in shock["records"]
                         if r["limiter"] == "tvb" and r["flux"] == flux],
                        key=lambda r: r["cells"])
        cells_axis = np.array([r["cells"] for r in series], dtype=float)
        for key, style in (("level_crossing_error", ":"),
                           ("centroid_error", "-"),
                           ("rankine_hugoniot_error", "--")):
            axes[0].loglog(cells_axis, [max(r[key], 1e-16) for r in series],
                           marker + style, label="%s / %s" % (flux, key.split("_")[0]))
    axes[0].set_xlabel("cells")
    axes[0].set_ylabel("shock position error")
    axes[0].set_title("Shock location vs analytic (TVB-limited)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")

    for case in order["cases"]:
        if case["flux"] != "lax_friedrichs":
            continue
        cells_axis = np.array(case["cells"], dtype=float)
        axes[1].loglog(cells_axis, case["relative_l2"], "o-",
                       label="P%d (order %.2f)" % (case["degree"],
                                                   case["mean_order_finest_two"]))
    axes[1].set_xlabel("cells")
    axes[1].set_ylabel("relative L2 error")
    axes[1].set_title("Smooth-regime design order (pre-shock)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig2_shock_order.png", dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=0.2)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--u-left", type=float, default=1.0)
    parser.add_argument("--u-right", type=float, default=0.0)
    parser.add_argument("--shock-x0", type=float, default=0.1)
    parser.add_argument("--output-dir", default="artifacts/burgers_shock_1d")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = RiemannData(args.u_left, args.u_right, args.shock_x0)

    print("[1/3] smooth-regime design order")
    order = smooth_order_study(cfl=0.1)
    print("[2/3] shock position and Gibbs oscillation")
    shock = shock_study(args.cells, args.degree, args.final_time, args.cfl, data)
    print("[3/3] figures")
    make_figures(out, shock, order, data, args.degree, args.final_time)

    profiles = shock.pop("profiles")
    cells = max(r["cells"] for r in shock["records"])
    for limiter in (None, "tvb"):
        x, coeff = profiles[(limiter, "lax_friedrichs", cells)]
        exact = exact_solution(torch.as_tensor(x, dtype=coeff.dtype), args.final_time, data)
        name = "limited" if limiter else "unlimited"
        np.savetxt(out / f"profile_{name}_{cells}.csv",
                   np.column_stack((x.numpy(), coeff[:, 0].numpy(), exact.numpy())),
                   delimiter=",", header="x,u_cell_average,exact",
                   comments="", fmt="%.10g")

    metrics = {
        "benchmark": "burgers_shock_1d",
        "equation": "u_t + (u^2/2)_x = 0",
        "riemann": {"u_left": data.u_left, "u_right": data.u_right, "x0": data.x0,
                    "shock_speed_exact": data.shock_speed,
                    "shock_position_exact": data.shock_location(args.final_time),
                    "final_time": args.final_time},
        "degree": args.degree, "cfl": args.cfl,
        "smooth_order": order["cases"],
        "smooth_order_passed": order["passed"],
        "shock_records": shock["records"],
        "rankine_hugoniot_passed": shock["rankine_hugoniot_passed"],
        "gibbs": {"limited_max": shock["gibbs_limited_max"],
                  "unlimited_max": shock["gibbs_unlimited_max"],
                  "bound": GIBBS_BOUND, "required_reduction": GIBBS_REDUCTION,
                  "passed": shock["gibbs_passed"]},
        "shock_decay": shock["decay"],
        "criteria": {
            "smooth_order_min": ORDER_MARGIN,
            "rankine_hugoniot_tolerance": RH_TOLERANCE,
            "gibbs_absolute_bound": GIBBS_BOUND,
            "gibbs_minimum_reduction": GIBBS_REDUCTION,
        },
        "limiter_higher_order_status": "roadmap (TVB reduces troubled cells to "
                                       "first order; WENO-type limited DG is not "
                                       "implemented)",
        "passed": bool(order["passed"] and shock["passed"]),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps({key: metrics[key] for key in
                      ("smooth_order_passed", "rankine_hugoniot_passed",
                       "gibbs", "passed")}, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())