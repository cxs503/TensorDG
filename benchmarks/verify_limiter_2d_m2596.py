"""Verification of the two-dimensional TVB shock limiter (m2596).

Renamed from ``benchmarks/verify_limiter_2d.py`` when this branch was rebuilt
onto main 0ec9e1ea: that path now carries the m2503/PR#18 artifact.

The 2-D limited path of :mod:`tensordg.euler_dg_2d` had two defects that this
benchmark pins down with numbers and that the accompanying patch repairs:

D1  *mesh-dependent troubled-cell detector.*  ``_troubled_1d`` gated the TVB
    test on ``max(|delta_left|, |delta_right|) > M dx^2`` -- an ``O(dx u')``
    quantity compared against an ``O(dx^2)`` tolerance.  The effective
    tolerance therefore scales like ``O(dx/M)`` and refinement flags *more*
    smooth cells, not fewer: on the smooth isentropic vortex 10-14 % of the
    cells were limited at 16x16/24x24 and the P1 error grew with refinement
    (EOC = -0.26).  The detector now uses the scale-consistent deviation
    ``|delta_left - minmod(delta_left, delta_right)|`` of
    ``limiters._troubled_cell_indicator`` (the m274/m2496 fix, never ported to
    2-D).

D2  *degree >= 2 crash.*  The higher-mode blocks were masked with a 3-D/4-D
    broadcast (``mask.unsqueeze(2)``/``mask.unsqueeze(1)``) against 5-D mode
    blocks, so ``limiter="tvb"`` raised ``RuntimeError`` as soon as one cell of
    a P>=2 state was flagged.

Verification sections
---------------------
A  detector parity: the 2-D axis detector and the 1-D indicator must return
   the same troubled-cell mask on the same data (smooth periodic wave and a
   Sod jump), i.e. one policy, two implementations;
B  design order: the smooth isentropic vortex must keep ``EOC = p + 1`` with
   the limiter enabled (the limiter must be *inert* on resolved smooth data)
   and the unlimited path must stay unchanged;
C  shock behaviour: extruded Sod tube -- positivity, overshoot and error vs
   the exact Riemann solution with the limiter on/off, so the accuracy fix is
   not bought by disarming the limiter;
D  genuinely 2-D shock: Lax & Liu (1998) configuration-3 four-quadrant Riemann
   problem -- positivity and mesh self-convergence with the limiter on/off.

Usage::

    python -m benchmarks.verify_limiter_2d_m2596 --output-dir validation/euler_dg_2d
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
from tensordg.limiters import _troubled_cell_indicator

from benchmarks.sod_exact import cell_averages as sod_cell_averages
from benchmarks.verify_euler_dg_2d import _vortex_error

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "validation" / "euler_dg_2d"

#: TVB constant that is inert on the resolved smooth vortex (see the
#: sweep in section B2); the pre-fix detector could not be tuned at all.
TUNED_TVB_CONSTANT = 10.0
FIGURES = ROOT / "docs" / "figures"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _LimitCounter:
    """Count how many cells the 2-D limiter actually modifies."""

    def __init__(self):
        self.calls = 0
        self.flagged = 0
        self.cells = 0
        self._original = None

    def __enter__(self):
        self._original = dg2.apply_tvb_limiter_2d
        original = self._original
        counter = self

        def wrapped(coeff, dx, dy, tvb_constant=1.0):
            limited = original(coeff, dx, dy, tvb_constant)
            flat = (limited - coeff).abs().reshape(
                coeff.shape[0], coeff.shape[1], -1
            ).max(dim=-1).values
            counter.calls += 1
            counter.flagged += int((flat > 0).sum())
            counter.cells += coeff.shape[0] * coeff.shape[1]
            return limited

        dg2.apply_tvb_limiter_2d = wrapped
        return self

    def __exit__(self, *exc):
        dg2.apply_tvb_limiter_2d = self._original
        return False

    @property
    def fraction(self):
        return self.flagged / self.cells if self.cells else 0.0


def _averages_from_profile(values: torch.Tensor, nx: int, ny: int, axis: int):
    """Broadcast a 1-D cell-average profile into a y- or x-invariant 2-D field."""
    if axis == 0:
        return values.view(nx, 1, 1).expand(nx, ny, 1).clone()
    return values.view(1, ny, 1).expand(nx, ny, 1).clone()


def _eoc(rows, field="density"):
    out = []
    for previous, current in zip(rows, rows[1:]):
        ratio = previous["h"] / current["h"]
        out.append(
            float(math.log(previous["errors"][field] / current["errors"][field])
                  / math.log(ratio))
        )
    return out


# ---------------------------------------------------------------------------
# A. detector parity (2-D axis detector vs 1-D indicator)
# ---------------------------------------------------------------------------

def detector_parity(ncells=64, tvb_constant=1.0):
    """The 2-D axis detector must reproduce the 1-D indicator cell by cell."""
    results = {}

    # (i) smooth periodic wave: no cell may be flagged by either detector
    x = (np.arange(ncells) + 0.5) / ncells
    smooth = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
    # (ii) Sod jump: a handful of cells around the diaphragm must be flagged
    jump = np.where(x < 0.5, 1.0, 0.125)

    for name, profile in (("smooth_periodic", smooth), ("sod_jump", jump)):
        values = torch.as_tensor(profile, dtype=torch.float64)
        averages = _averages_from_profile(values, ncells, 4, axis=0)
        dx = 1.0 / ncells
        mask_2d, slope_2d = dg2._troubled_1d(averages, 0, dx, tvb_constant)
        coeff_1d = torch.zeros(ncells, 2, 1, dtype=torch.float64)
        coeff_1d[:, 0, 0] = values
        mask_1d = _troubled_cell_indicator(coeff_1d, dx, tvb_constant)

        # the 1-D indicator freezes the outermost cells (no full stencil); the
        # 2-D axis detector wraps periodically.  Compare the interior only.
        interior = slice(1, ncells - 1)
        mask_2d_c = mask_2d[:, 0][interior]
        mask_1d_c = mask_1d[interior]
        results[name] = {
            "cells": ncells,
            "tvb_constant": tvb_constant,
            "flagged_2d_interior": int(mask_2d_c.sum()),
            "flagged_1d_interior": int(mask_1d_c.sum()),
            "masks_identical_interior": bool(torch.equal(mask_2d_c, mask_1d_c)),
        }

    return results


def detector_parity_scan(ncells_sequence=(20, 40, 80, 160), tvb_constant=10.0):
    """Mesh independence: the same smooth wave must be flagged 0 cells at every h."""
    rows = []
    for ncells in ncells_sequence:
        x = (np.arange(ncells) + 0.5) / ncells
        profile = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
        values = torch.as_tensor(profile, dtype=torch.float64)
        averages = _averages_from_profile(values, ncells, 2, axis=0)
        dx = 1.0 / ncells
        mask_2d, _ = dg2._troubled_1d(averages, 0, dx, tvb_constant)
        rows.append(
            {
                "cells": ncells,
                "dx": dx,
                "flagged_2d": int(mask_2d.sum()),
            }
        )
    return {"tvb_constant": tvb_constant, "rows": rows}


# ---------------------------------------------------------------------------
# B. design order on the smooth isentropic vortex
# ---------------------------------------------------------------------------

def vortex_convergence(degree, meshes, limiter=None, tvb_constant=1.0,
                       final_time=1.0, cfl=0.2):
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    rows = []
    for ncells in meshes:
        started = time.time()
        with _LimitCounter() as counter:
            _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
                ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
                limiter=limiter, tvb_constant=tvb_constant,
            )
        errors = _vortex_error(coeff, basis, nodes, ((-5.0, 5.0), (-5.0, 5.0)),
                              final_time)
        rows.append(
            {
                "cells": ncells,
                "h": 10.0 / ncells,
                "steps": metadata["steps"],
                "errors": errors,
                "flagged_fraction": counter.fraction,
                "wall_seconds": time.time() - started,
            }
        )
        print(
            f"    P{degree} limiter={limiter} M={tvb_constant} {ncells:3d}^2: "
            f"rho {errors['density']:.4e} p {errors['pressure']:.4e} "
            f"flagged {counter.fraction:.2e} ({rows[-1]['wall_seconds']:.1f}s)",
            flush=True,
        )
    for row, eoc in zip(rows[1:], _eoc(rows)):
        row["eoc_density"] = eoc
    return {
        "degree": degree,
        "limiter": limiter or "none",
        "tvb_constant": tvb_constant,
        "final_time": final_time,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# B2. TVB constant sweep: the fixed detector must be *tunable*
# ---------------------------------------------------------------------------

def tvb_sweep(degree=2, ncells=16, constants=(1.0, 3.0, 10.0, 100.0),
              final_time=1.0, cfl=0.2):
    """L2 error and flagged fraction as a function of the TVB constant M.

    The historic O(dx) detector made M unusable: raising it never disarmed the
    limiter on smooth data (it only postponed the onset from dx > 1/M to
    dx > M).  With the scale-consistent measure a finite M exists for which the
    limiter is inert and the error collapses onto the unlimited value.
    """
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    _, _, coeff, _ = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells, degree=degree, final_time=final_time, cfl=cfl)
    unlimited = _vortex_error(coeff, basis, nodes, ((-5.0, 5.0), (-5.0, 5.0)),
                              final_time)
    rows = [{"tvb_constant": None, "errors": unlimited, "flagged_fraction": 0.0}]
    for constant in constants:
        with _LimitCounter() as counter:
            _, _, coeff, _ = dg2.solve_euler_dg_2d_vortex(
                ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
                limiter="tvb", tvb_constant=constant)
        errors = _vortex_error(coeff, basis, nodes, ((-5.0, 5.0), (-5.0, 5.0)),
                               final_time)
        rows.append({"tvb_constant": constant, "errors": errors,
                     "flagged_fraction": counter.fraction})
        print(f"    M={constant:6g}: rho {errors['density']:.4e} "
              f"flagged {counter.fraction:.2e} "
              f"(unlimited {unlimited['density']:.4e})", flush=True)
    return {"degree": degree, "cells": ncells, "final_time": final_time,
            "rows": rows}


# ---------------------------------------------------------------------------
# C. extruded Sod tube: shock capture with the limiter
# ---------------------------------------------------------------------------

def extruded_sod(degree=2, ncells=200, final_time=0.2, cfl=0.2,
                 limiter=None, tvb_constant=1.0):
    _, reference = sod_cell_averages(ncells, final_time)
    _, _, coeff, metadata = dg2.solve_euler_dg_2d_sod(
        ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
        limiter=limiter, tvb_constant=tvb_constant,
    )
    rho = coeff[:, 0, 0, 0, 0]
    momentum_x = coeff[:, 0, 0, 0, 1]
    energy = coeff[:, 0, 0, 0, 3]
    pressure = (dg2.GAMMA - 1.0) * (energy - 0.5 * momentum_x ** 2 / rho)
    numeric = rho.detach().cpu().numpy()
    exact = np.asarray(reference["density"])
    velocity = (momentum_x / rho).detach().cpu().numpy()
    velocity_exact = np.asarray(reference["velocity"])

    def relative_l2(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    # overshoot: excess above the largest exact plateau value (0.42632 behind
    # the shock for Sod); a robust oscillation measure at fixed resolution.
    plateau = float(np.max(exact))
    return {
        "degree": degree,
        "cells": ncells,
        "final_time": final_time,
        "limiter": limiter or "none",
        "tvb_constant": tvb_constant,
        "steps": metadata["steps"],
        "density_relative_l2": relative_l2(numeric, exact),
        "velocity_relative_l2": relative_l2(velocity, velocity_exact),
        "max_density": float(numeric.max()),
        "overshoot_above_plateau": float(max(0.0, numeric.max() - plateau)),
        "min_density": float(numeric.min()),
        "min_pressure": float(pressure.min()),
        "density": numeric.tolist(),
        "density_exact": exact.tolist(),
    }


# ---------------------------------------------------------------------------
# D. four-quadrant Riemann problem (Lax & Liu 1998, config 3)
# ---------------------------------------------------------------------------

def four_quadrant(ncells, degree=1, final_time=0.25, cfl=0.2,
                  limiter=None, tvb_constant=1.0):
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    out = {}
    for mesh in ncells:
        with _LimitCounter() as counter:
            _, _, coeff, metadata = dg2.solve_euler_dg_2d_riemann(
                ncells=mesh, degree=degree, final_time=final_time, cfl=cfl,
                limiter=limiter, tvb_constant=tvb_constant,
            )
        state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
        rho = state[..., 0]
        pressure = (dg2.GAMMA - 1.0) * (
            state[..., 3] - 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / rho
        )
        out[str(mesh)] = {
            "cells": mesh,
            "steps": metadata["steps"],
            "min_density": float(rho.min()),
            "min_pressure": float(pressure.min()),
            "finite": bool(torch.isfinite(state).all()),
            "flagged_fraction": counter.fraction,
            "cell_density": coeff[..., 0, 0, 0].detach().cpu().reshape(-1).tolist(),
        }
    return {"degree": degree, "limiter": limiter or "none",
            "tvb_constant": tvb_constant, "meshes": out}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(report, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # left: vortex L2 error vs h, unlimited vs tuned limiter
    ax = axes[0]
    styles = {("none", 1): "+-", ("tvb", 1): "v--", ("none", 2): "+-",
              ("tvb", 2): "v--", ("none", 3): "+-", ("tvb", 3): "v--"}
    for block in report["vortex"]:
        degree = block["degree"]
        limiter = block["limiter"]
        h = [row["h"] for row in block["rows"]]
        err = [row["errors"]["density"] for row in block["rows"]]
        label = (f"P{degree}, no limiter" if limiter == "none"
                 else f"P{degree}, TVB M={block['tvb_constant']:g}")
        ax.loglog(h, err, styles.get((limiter, degree), "-o"), label=label,
                  ms=4)
        if limiter == "none":
            order = degree + 1
            ref = [err[0] * (h[0] / hh) ** order for hh in h]
            ax.loglog(h, ref, ":", color="grey", lw=0.8)
            ax.annotate(f"$p+1={order}$", (h[-1], ref[-1]), fontsize=7,
                        color="grey")
    ax.set_xlabel("h = 10/N")
    ax.set_ylabel(r"relative $L^2$ density error")
    ax.set_title("smooth vortex: design order kept by TVB")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    # middle: tunability of the TVB constant
    ax = axes[1]
    sweep = report["tvb_sweep"][0]
    constants, errors = [], []
    unlimited = sweep["rows"][0]["errors"]["density"]
    for row in sweep["rows"][1:]:
        constants.append(row["tvb_constant"])
        errors.append(row["errors"]["density"])
    ax.semilogx(constants, errors, "o-", label="TVB limited")
    ax.axhline(unlimited, color="k", ls="--", lw=1.0,
               label="no limiter")
    ax.set_xlabel("TVB constant M")
    ax.set_ylabel(r"relative $L^2$ density error")
    ax.set_title(f"tunability (P{sweep['degree']}, {sweep['cells']}$^2$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    # right: Sod density profiles, exact vs limited/unlimited
    ax = axes[2]
    sod = report["sod"]
    ax.plot(sod["none"]["density_exact"], "k-", lw=1.2, label="exact Riemann")
    ax.plot(sod["none"]["density"], "C0--", lw=1.0,
            label=f"no limiter ({sod['none']['density_relative_l2']*100:.2f}%)")
    for key, colour in (("tvb_M1", "C3"), ("tvb_M10", "C2")):
        if key in sod:
            ax.plot(sod[key]["density"], colour, lw=0.9,
                    label=f"M={sod[key]['tvb_constant']:g} "
                          f"({sod[key]['density_relative_l2']*100:.2f}%)")
    ax.set_xlabel("cell index")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title(f"2-D Sod, P{sod['none']['degree']}, {sod['none']['cells']} cells "
                 "(rel-L2 vs exact)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    path = FIGURES / "tensordg-m2596-limiter-2d.png"
    fig.savefig(path, dpi=180)
    fig.savefig(output_dir / "limiter_2d.png", dpi=180)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true",
                        help="coarser mesh sequences (used by the regression test)")
    parser.add_argument("--json", default="limiter_2d.json")
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.quick:
        vortex_meshes = {1: (8, 12, 16), 2: (8, 12, 16), 3: (8, 12)}
    else:
        vortex_meshes = {1: (8, 16, 24), 2: (8, 16, 24), 3: (8, 12, 16)}

    started = time.time()
    report = {}

    print("[A] detector parity (2-D axis detector vs 1-D indicator)")
    report["detector_parity"] = detector_parity()
    report["detector_parity_scan"] = detector_parity_scan()
    for name, block in report["detector_parity"].items():
        print(f"  {name}: 2-D {block['flagged_2d_interior']} vs 1-D "
              f"{block['flagged_1d_interior']} flagged, identical="
              f"{block['masks_identical_interior']}")
    print("  mesh scan (M=10): " + ", ".join(
        f"{row['cells']}->{row['flagged_2d']}" for row in
        report["detector_parity_scan"]["rows"]))

    print("[B] design order on the smooth isentropic vortex")
    blocks = []
    for degree in (1, 2, 3):
        meshes = vortex_meshes[degree]
        blocks.append(vortex_convergence(degree, meshes, limiter=None))
        blocks.append(vortex_convergence(degree, meshes, limiter="tvb",
                                        tvb_constant=TUNED_TVB_CONSTANT))
    report["vortex"] = blocks

    print("[B2] TVB constant sweep (tunability of the fixed detector)")
    report["tvb_sweep"] = [tvb_sweep(degree=2, ncells=16)]

    print("[C] extruded Sod tube (shock capture)")
    sod = {
        "none": extruded_sod(limiter=None),
        "tvb_M1": extruded_sod(limiter="tvb", tvb_constant=1.0),
        "tvb_M10": extruded_sod(limiter="tvb", tvb_constant=10.0),
    }
    for name, block in sod.items():
        print(f"  {name:8s}: rho rel-L2 {block['density_relative_l2']*100:.3f}% "
              f"overshoot {block['overshoot_above_plateau']:.4f} "
              f"min p {block['min_pressure']:.3e}")
    report["sod"] = sod

    print("[D] four-quadrant Riemann problem (Lax & Liu config 3)")
    meshes = (32, 48) if not args.quick else (24, 32)
    report["four_quadrant"] = {
        "none": four_quadrant(meshes, limiter=None),
        "tvb_M1": four_quadrant(meshes, limiter="tvb", tvb_constant=1.0),
        "tvb_M10": four_quadrant(meshes, limiter="tvb",
                                 tvb_constant=TUNED_TVB_CONSTANT),
    }
    fine = str(meshes[-1])
    for name in report["four_quadrant"]:
        block = report["four_quadrant"][name]["meshes"][fine]
        print(f"  {name:5s} {fine}^2: min rho {block['min_density']:.3e} "
              f"min p {block['min_pressure']:.3e} finite={block['finite']}")

    figure = make_figures(report, output)
    report["figure"] = str(figure.relative_to(ROOT))
    report["environment"] = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "wall_seconds": time.time() - started,
        "quick": args.quick,
    }
    target = output / args.json
    target.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {target} and {figure} in {report['environment']['wall_seconds']:.1f}s")
    return report


if __name__ == "__main__":
    main()