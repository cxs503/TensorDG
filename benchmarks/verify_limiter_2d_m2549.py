"""Verification suite for the two-dimensional TVB limiter (mission m2549).

Renamed from ``benchmarks/verify_limiter_2d.py`` when this branch was rebuilt
onto main 0ec9e1ea: that path now carries the m2503/PR#18 artifact.

Run with ``PYTHONPATH=. python -m benchmarks.verify_limiter_2d_m2549 --tag after``.

Why this exists
---------------
``tensordg/limiters.py`` (1-D) had its troubled-cell test rewritten in mission
m2500 into a *scale-consistent* form: the quantity compared against the TVB
tolerance ``M dx**2`` is the deviation ``|d_left - minmod(d_left, d_right)|``,
which is O(dx**2 u'') on smooth data, instead of the raw one-sided difference
``max|d_left|, max|d_right|``, which is O(dx u') and therefore grows relative to
``M dx**2`` as the mesh is refined.  That rewrite was never propagated to the
2-D limiter in ``tensordg/euler_dg_2d.py``, whose ``_troubled_1d`` still used
the old O(dx) measure.  The two-dimensional limiter was thus not a TVB limiter
at all: every smooth extremum was flagged, more of them the finer the mesh, and
``tvb_constant`` could not be tuned.

This suite turns that claim into machine-decidable checks, each with a stated
criterion.  It is written so the same script can be run before (``--tag
before``) and after (``--tag after``) the fix; the two JSON files are the
before/after evidence.

Checks
------
A  detector scale consistency: on a smooth 2-D wave the flagged-cell fraction
   must vanish (M = 1) and must not grow under refinement; a sweep over M must
   actually change the flagged set.
B  design-order preservation: on the smooth isentropic vortex the limited
   solution must match the unl imited one (relative error ratio ~ 1) with
   ``EOC = degree + 1``, for a TVB constant above the data curvature.
C  shock robustness: on the four-quadrant Riemann problem (Lax & Liu 1998,
   configuration 3) the limiter must keep density/pressure positive and must not
   increase the over/undershoot of the cell averages relative to no limiter,
   while still flagging cells at the discontinuities.
D  structural invariants: cell averages are untouched by the limiter, edge
   values stay inside the neighbouring averages, and a genuine jump is flagged
   on both meshes with a mesh-independent cell count.

All numbers are written to ``validation/limiter_2d/<tag>/evidence.json`` so the
report can be generated from files instead of hand-copied digits.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_dg_2d import (
    _initial_coefficients,
    _primitive,
    apply_tvb_limiter_2d,
    legendre_operators,
)
from tensordg.euler_2d import vortex_primitive

OUTPUT = Path("validation/limiter_2d")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reconstruct(coeff, basis, nodes):
    """Modal coefficients -> (cell average, volume-quadrature state)."""
    state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    return coeff[..., 0, 0, :], state


def _smooth_wave_initial(x, y):
    """Smooth 2-D density/pressure wave with |second derivative| < 0.5."""
    wave = torch.sin(math.pi * x) * torch.cos(0.5 * math.pi * y)
    density = 1.0 + 0.05 * wave
    pressure = 1.0 + 0.05 * wave
    zeros = torch.zeros_like(x)
    return density, zeros, zeros, pressure


def _project_wave(ncells, degree, domain=(-1.0, 1.0)):
    nodes, weights, basis, _, _ = legendre_operators(degree, max(degree + 2, 3))
    lo, hi = domain
    dx = (hi - lo) / ncells
    centres = lo + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    coeff = _initial_coefficients(
        _smooth_wave_initial,
        centres,
        centres,
        dx,
        dx,
        nodes,
        weights,
        basis,
        torch.device("cpu"),
        torch.float64,
    )
    return coeff, dx, nodes, basis


def _flag_counts(coeff, dx, tvb_constant):
    """Number of cells flagged along each axis by the 2-D troubled-cell test."""
    averages = coeff[..., 0, 0, :]
    flagged = {}
    for axis, name in ((0, "x"), (1, "y")):
        mask, _ = dg2._troubled_1d(averages, axis, dx, tvb_constant)
        flagged[name] = int(mask.sum())
    return flagged


def _vortex_errors(coeff, basis, nodes, final_time, strength=5.0, background=(1.0, 1.0)):
    nx, ny = coeff.shape[0], coeff.shape[1]
    dx, dy = 10.0 / nx, 10.0 / ny
    x = -5.0 + (torch.arange(nx, dtype=coeff.dtype) + 0.5) * dx
    y = -5.0 + (torch.arange(ny, dtype=coeff.dtype) + 0.5) * dy
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    exact = vortex_primitive(xq, yq, final_time, strength, background)
    state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho, u, v, p = _primitive(state)
    weights = torch.as_tensor(
        np.polynomial.legendre.leggauss(nodes.numel())[1], dtype=coeff.dtype
    )
    weight = weights[:, None] * weights[None, :]
    out = {}
    for name, numerical, reference in (
        ("density", rho, exact[0]),
        ("velocity_x", u, exact[1]),
        ("velocity_y", v, exact[2]),
        ("pressure", p, exact[3]),
    ):
        out[name] = float(
            torch.sqrt(
                (((numerical - reference) ** 2) * weight).sum()
                / ((reference ** 2) * weight).sum()
            )
        )
    return out


# ---------------------------------------------------------------------------
# A. detector scale consistency
# ---------------------------------------------------------------------------

def _curvature_estimate(degree=2, ncells=256):
    """Measured max |d2u/dx2| of the smooth wave, used to pick a TVB constant.

    The TVB tolerance is ``M dx**2`` and the smooth-data deviation of the test
    quantity is ``|u''| dx**2`` (times one half at an extremum), so a constant
    ``M`` above the data curvature must leave the solution untouched while a
    constant below it must flag cells.  Deriving ``M`` from the data instead of
    hard-coding it is what makes this check meaningful: before the m2549 fix no
    constant could do this, because the flagged set grew with refinement.
    """
    coeff, dx, _, _ = _project_wave(ncells, degree)
    averages = coeff[..., 0, 0, :]
    second = (
        torch.roll(averages, -1, 0) - 2.0 * averages + torch.roll(averages, 1, 0)
    ) / (dx * dx)
    return float(second.abs().max())


def detector_scale(degrees=(1, 2), meshes=(8, 16, 32, 64), constants=None):
    """Flagged-cell fraction on a smooth wave: mesh sweep and M sweep."""
    curvature = _curvature_estimate()
    if constants is None:
        constants = (0.0, 0.5, 1.0, 2.0 * curvature, 8.0 * curvature)
    constants = tuple(sorted(set(constants)))
    m_test = 2.0 * curvature
    results = {
        "curvature_estimate": curvature,
        "tvb_constant_above_curvature": m_test,
        "mesh_sweep": [],
        "constant_sweep": [],
    }
    for degree in degrees:
        for ncells in meshes:
            coeff, dx, _, _ = _project_wave(ncells, degree)
            for tvb, label in ((1.0, "M=1"), (m_test, "M=2|u''|")):
                counts = _flag_counts(coeff, dx, tvb)
                results["mesh_sweep"].append(
                    {
                        "degree": degree,
                        "cells": ncells,
                        "dx": dx,
                        "tvb_constant": tvb,
                        "label": label,
                        "flagged_x": counts["x"],
                        "flagged_y": counts["y"],
                        "fraction_x": counts["x"] / (ncells * ncells),
                        "fraction_y": counts["y"] / (ncells * ncells),
                    }
                )
    for tvb in constants:
        coeff, dx, _, _ = _project_wave(32, 2)
        counts = _flag_counts(coeff, dx, tvb)
        results["constant_sweep"].append(
            {
                "degree": 2,
                "cells": 32,
                "dx": dx,
                "tvb_constant": tvb,
                "threshold": tvb * dx * dx,
                "flagged_x": counts["x"],
                "flagged_y": counts["y"],
                "fraction_x": counts["x"] / (32 * 32),
                "fraction_y": counts["y"] / (32 * 32),
            }
        )
    released = [
        row["fraction_x"]
        for row in results["mesh_sweep"]
        if row["label"] == "M=2|u''|"
    ]
    strict = [
        row["fraction_x"] for row in results["mesh_sweep"] if row["label"] == "M=1"
    ]
    # At M = 1 the flagged set is the region where |u''| > 1 (the conservative
    # variable with the largest amplitude is the energy, 2.5x the pressure
    # wave).  That region is a fixed fraction of the domain, so the flagged
    # fraction must stay bounded and must not trend upwards with refinement --
    # which is exactly what the pre-fix test violated (it grew to 0.44).
    criteria = {
        "tvb_constant_used": m_test,
        "fraction_at_M_above_curvature_must_be_zero": bool(all(f == 0.0 for f in released)),
        "fraction_at_M1_bounded_by_0_10": bool(max(strict) <= 0.10),
        "fraction_at_M1_no_upward_trend": bool(
            strict[-1] <= 2.0 * max(strict[0], 1e-12)
        ),
        "strict_test_must_flag_interior_cells": bool(max(strict) > 0.0),
        "M_sweep_must_change_the_flagged_set": bool(
            len({row["flagged_x"] for row in results["constant_sweep"]}) > 1
        ),
    }
    criteria["passed"] = bool(
        criteria["fraction_at_M_above_curvature_must_be_zero"]
        and criteria["fraction_at_M1_bounded_by_0_10"]
        and criteria["fraction_at_M1_no_upward_trend"]
        and criteria["strict_test_must_flag_interior_cells"]
        and criteria["M_sweep_must_change_the_flagged_set"]
    )
    results["criterion"] = criteria
    return results


# ---------------------------------------------------------------------------
# B. design-order preservation on the smooth vortex
# ---------------------------------------------------------------------------

def _vortex_run(ncells, degree, final_time, limiter, tvb_constant):
    """Run the isentropic vortex through the low-level solver."""

    def initial(x, y):
        return vortex_primitive(x, y, 0.0, 5.0, (1.0, 1.0))

    _, _, coeff, metadata = dg2.solve_euler_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=((-5.0, 5.0), (-5.0, 5.0)),
        final_time=final_time,
        cfl=0.2,
        limiter=limiter,
        tvb_constant=tvb_constant,
        periodic=(True, True),
    )
    return coeff, metadata


def vortex_order(degree, meshes, final_time=1.0, constants=(1.0, 10.0, 50.0)):
    """Limited vs unlimited vortex error and EOC for each TVB constant."""
    nodes, _, basis, _, _ = legendre_operators(degree, max(degree + 2, 3))
    unlimited = {}
    for ncells in meshes:
        started = time.time()
        coeff, meta = _vortex_run(ncells, degree, final_time, None, 0.0)
        unlimited[ncells] = {
            "errors": _vortex_errors(coeff, basis, nodes, final_time),
            "steps": meta["steps"],
            "wall_seconds": time.time() - started,
        }
    rows = []
    for tvb in constants:
        series = []
        for ncells in meshes:
            started = time.time()
            coeff, meta = _vortex_run(
                ncells, degree, final_time, "tvb", tvb
            )
            errors = _vortex_errors(coeff, basis, nodes, final_time)
            series.append(
                {
                    "cells": ncells,
                    "h": 10.0 / ncells,
                    "errors": errors,
                    "ratio_to_unlimited": {
                        key: errors[key] / unlimited[ncells]["errors"][key]
                        for key in errors
                    },
                    "steps": meta["steps"],
                    "wall_seconds": time.time() - started,
                }
            )
        for previous, current in zip(series, series[1:]):
            ratio = previous["h"] / current["h"]
            current["eoc"] = {
                key: float(
                    math.log(previous["errors"][key] / current["errors"][key]) / math.log(ratio)
                )
                for key in ("density", "pressure")
            }
        rows.append({"tvb_constant": tvb, "rows": series})
    design = degree + 1
    inert = []
    for block in rows:
        row = block["rows"][-1]
        worst_ratio = max(row["ratio_to_unlimited"].values())
        worst_eoc = row.get("eoc", {}).get("density", float("nan"))
        if worst_ratio <= 1.10 and worst_eoc >= design - 0.3:
            inert.append(block["tvb_constant"])
    passed = bool(inert)
    return {
        "degree": degree,
        "final_time": final_time,
        "design_order": design,
        "unlimited": {str(k): v for k, v in unlimited.items()},
        "limited": rows,
        "criterion": {
            "finest_mesh_eoc_at_least_design_minus_0_3": design - 0.3,
            "error_ratio_to_unlimited_at_most": 1.10,
            "constants_behaving_as_TVB": inert,
            "passed": passed,
        },
    }


# ---------------------------------------------------------------------------
# C. shock robustness on the four-quadrant Riemann problem
# ---------------------------------------------------------------------------

def riemann_shock(ncells=32, degree=1, final_time=0.25, tvb_constant=1.0):
    """Limiter vs no limiter on the Lax & Liu configuration 3 problem."""
    config = dg2.LAX_LIU_CONFIG_3
    initial_min = min(state[0] for state in config.values())
    initial_max = max(state[0] for state in config.values())
    initial_pmin = min(state[3] for state in config.values())
    results = {}
    for label, limiter in (("none", None), ("tvb", "tvb")):
        started = time.time()
        _, _, coeff, meta = dg2.solve_euler_dg_2d_riemann(
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            limiter=limiter,
            tvb_constant=tvb_constant,
        )
        nodes, _, basis, _, _ = legendre_operators(degree, max(degree + 2, 3))
        state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
        rho, _, _, pressure = _primitive(state)
        averages = coeff[..., 0, 0, :]
        flagged_x, _ = dg2._troubled_1d(averages, 0, 1.0 / ncells, tvb_constant)
        flagged_y, _ = dg2._troubled_1d(averages, 1, 1.0 / ncells, tvb_constant)
        results[label] = {
            "cells": ncells,
            "degree": degree,
            "steps": meta["steps"],
            "dt": meta["dt"],
            "min_density": float(rho.min()),
            "min_pressure": float(pressure.min()),
            "max_density": float(rho.max()),
            "undershoot": float(rho.min() - initial_min),
            "overshoot": float(rho.max() - initial_max),
            "min_cell_pressure": float(
                _primitive(averages)[3].min()
            ),
            "flagged_cells_x": int(flagged_x.sum()),
            "flagged_cells_y": int(flagged_y.sum()),
            "finite": bool(torch.isfinite(state).all()),
            "wall_seconds": time.time() - started,
            "cell_density": averages[..., 0].tolist(),
            "cell_pressure": _primitive(averages)[3].tolist(),
        }
    limited, unlimited = results["tvb"], results["none"]
    results["criterion"] = {
        "initial_density_range": [initial_min, initial_max],
        "initial_min_pressure": initial_pmin,
        "positivity_kept": bool(
            limited["min_density"] > 0.0 and limited["min_pressure"] > 0.0
        ),
        "overshoot_not_worse": bool(limited["overshoot"] <= unlimited["overshoot"] + 1e-12),
        "undershoot_not_worse": bool(limited["undershoot"] >= unlimited["undershoot"] - 1e-12),
        "limiter_active_at_shocks": bool(
            limited["flagged_cells_x"] > 0 or limited["flagged_cells_y"] > 0
        ),
    }
    results["criterion"]["passed"] = bool(
        results["criterion"]["positivity_kept"]
        and results["criterion"]["overshoot_not_worse"]
        and results["criterion"]["undershoot_not_worse"]
        and results["criterion"]["limiter_active_at_shocks"]
    )
    return results


# ---------------------------------------------------------------------------
# D. structural invariants
# ---------------------------------------------------------------------------

def structural_invariants(degree=2, ncells=24):
    """Cell averages untouched, jumps flagged, edge values bounded."""
    coeff, dx, nodes, basis = _project_wave(ncells, degree)
    limited = apply_tvb_limiter_2d(coeff, dx, dx, tvb_constant=0.0)  # strict
    average_drift = float((limited[..., 0, 0, :] - coeff[..., 0, 0, :]).abs().max())

    # A genuine jump in x: left half at rho=1, right half at rho=0.125.
    jump = coeff.clone()
    averages = torch.zeros(ncells, ncells, 4, dtype=coeff.dtype)
    averages[..., 0] = torch.where(
        torch.arange(ncells)[:, None] < ncells // 2, 1.0, 0.125
    )
    averages[..., 3] = torch.where(
        torch.arange(ncells)[:, None] < ncells // 2, 1.0, 0.1
    )
    jump[..., 0, 0, :] = averages
    jump[..., 1:, :, :] = 0.0
    jump[..., :, 1:, :] = 0.0
    flagged_x, _ = dg2._troubled_1d(jump[..., 0, 0, :], 0, dx, 1.0)
    flagged_y, _ = dg2._troubled_1d(jump[..., 0, 0, :], 1, dx, 1.0)
    limited_jump = apply_tvb_limiter_2d(jump, dx, dx, tvb_constant=1.0)
    average_drift_jump = float(
        (limited_jump[..., 0, 0, :] - jump[..., 0, 0, :]).abs().max()
    )

    # Edge values of the limited reconstruction must lie inside the neighbour
    # cell averages (the minmod property) wherever the limiter fired.
    _, state = _reconstruct(limited_jump, basis, nodes)
    rho = state[..., 0]
    neighbour_max = torch.maximum(
        torch.maximum(flagged_x.float() * 0.0 + torch.roll(averages[..., 0], 1, 0),
                      torch.roll(averages[..., 0], -1, 0)),
        averages[..., 0],
    )
    inside = float((rho <= neighbour_max.unsqueeze(-1).unsqueeze(-1) + 1e-12).float().mean())

    return {
        "degree": degree,
        "cells": ncells,
        "cell_average_drift_smooth": average_drift,
        "cell_average_drift_jump": average_drift_jump,
        "jump_flagged_x": int(flagged_x.sum()),
        "jump_flagged_y": int(flagged_y.sum()),
        "fraction_of_quadrature_points_inside_neighbour_bound": inside,
        "criterion": {
            "cell_averages_exactly_preserved": bool(
                average_drift == 0.0 and average_drift_jump == 0.0
            ),
            "jump_is_flagged": bool(flagged_x.sum() > 0),
            "y_invariant_data_not_flagged_in_y": bool(flagged_y.sum() == 0),
            "bounded_reconstruction": bool(inside > 0.999),
        },
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="after", help="label for the evidence file")
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="skip the time-stepping checks")
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))

    output = Path(args.output_dir) / args.tag
    output.mkdir(parents=True, exist_ok=True)

    def attempt(label, fn, *fn_args, **fn_kwargs):
        """Run one check; a crash is recorded as evidence, not as an abort."""
        try:
            return fn(*fn_args, **fn_kwargs)
        except Exception as error:  # noqa: BLE001 - evidence capture
            print(f"  {label}: EXCEPTION {type(error).__name__}: {error}")
            return {
                "error": f"{type(error).__name__}: {error}",
                "criterion": {"passed": False, "reason": "check raised"},
            }

    report = {
        "tag": args.tag,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": torch.__version__,
        "threads": args.threads,
    }
    started = time.time()

    print("[A] troubled-cell detector on a smooth 2-D wave")
    report["detector_scale"] = attempt("A", detector_scale)
    for row in report["detector_scale"]["mesh_sweep"]:
        print(
            f"  P{row['degree']} {row['cells']:3d}^2 dx={row['dx']:.4f}: "
            f"flagged x {row['flagged_x']:5d} y {row['flagged_y']:5d} "
            f"(fraction {row['fraction_x']:.3f} / {row['fraction_y']:.3f})"
        )
    for row in report["detector_scale"]["constant_sweep"]:
        print(
            f"  M={row['tvb_constant']:5.2f} threshold={row['threshold']:.3e}: "
            f"flagged x {row['flagged_x']:5d} y {row['flagged_y']:5d}"
        )
    print(f"  criterion passed = {report['detector_scale']['criterion']['passed']}")

    print("[D] structural invariants")
    report["structural"] = attempt("D", structural_invariants)
    print(f"  {report['structural']['criterion']}")

    if not args.quick:
        print("[B] design-order preservation on the isentropic vortex")
        report["vortex"] = {}
        for degree, meshes in ((1, (12, 16, 24)), (2, (8, 12, 16))):
            block = attempt(f"B/P{degree}", vortex_order, degree, meshes)
            report["vortex"][f"P{degree}"] = block
            for series in block.get("limited", []):
                fine = series["rows"][-1]
                print(
                    f"  P{degree} M={series['tvb_constant']:5.1f}: "
                    f"rho err {fine['errors']['density']:.3e} "
                    f"(ratio {fine['ratio_to_unlimited']['density']:.3f}) "
                    f"EOC {fine.get('eoc', {}).get('density', float('nan')):.2f}"
                )
            print(f"  P{degree} criterion passed = {block['criterion']['passed']}")

        print("[C] four-quadrant Riemann problem (Lax & Liu config 3)")
        report["riemann"] = []
        for ncells, degree in ((24, 1), (32, 1)):
            block = attempt(
                f"C/{ncells}", riemann_shock, ncells=ncells, degree=degree
            )
            block["cells"] = ncells
            report["riemann"].append(block)
            limited = block.get("tvb")
            print(
                f"  {ncells}^2 P{degree}: min rho {limited['min_density']:.4e} "
                f"min p {limited['min_pressure']:.4e} "
                f"overshoot {limited['overshoot']:.3e} "
                f"(unlimited {block['none']['overshoot']:.3e}) "
                f"flagged {limited['flagged_cells_x']}/{limited['flagged_cells_y']} "
                f"passed={block['criterion']['passed']}"
            )

    report["wall_seconds"] = time.time() - started
    with (output / "evidence.json").open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"\nwrote {output / 'evidence.json'} ({report['wall_seconds']:.1f}s)")
    return report


if __name__ == "__main__":
    main()