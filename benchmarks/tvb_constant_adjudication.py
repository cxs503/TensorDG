"""Adjudicate the ``tvb_constant`` "units bug" claim on the smooth 2-D vortex.

Motivation
----------
``tests/test_euler_dg_2d_time_split.py`` carried an ``xfail(strict=True)`` that
states, with numbers, that the *default* ``tvb_constant=1.0`` fires on the smooth
isentropic vortex and multiplies the density error by ~34 (8x8 cells) and ~136
(12x12 cells).  Its recorded diagnosis is

    "Raising the constant to 10 restores the unlimited error exactly, so the
     threshold scale (or the units it is compared against) is off, not the
     limiter machinery."

That diagnosis contradicts the repository's own settled analysis of the same
detector (``docs/limiter_2d_verification.md`` section 5 item 3 and
``docs/limiter_policy_verification.md`` section 6): after mission m2500/m2549 the
TVB test compares a *dimensionally consistent* quantity against ``M h**2``, so
``M`` is an **absolute bound on the modulus of the second derivative** of the
cell averages.  With ``M`` smaller than that bound a detector *must* fire -- that
is the definition of the Cockburn--Shu criterion, not a scale slip.

This benchmark decides between the two readings with measurements that do not
depend on which one is believed:

* **A** -- the analytic ``max |d2(component)/dx2|`` of the conserved variables of
  the initial vortex, from ``torch.autograd`` (no finite differences, no solver).
  Its convergent value is the ``M`` a detector needs to stay quiet.
* **B** -- the troubled-cell mask measured on the *projected* initial state
  (what the solver actually sees -- ``dg2._initial_coefficients``) against the
  mask computed by the **same shipped routine** ``dg2._troubled_1d`` applied to
  the **analytic point values**.  If the detector is a faithful implementation of
  ``|deviation| > M h**2`` the two masks agree cell by cell; a units/scale slip
  would make them disagree.
* **C** -- the flagged *fraction* as a function of the mesh at ``M = 1``.  A
  measure that is ``O(h**3)`` against an ``O(h**2)`` threshold would flag an
  ever larger fraction as ``h -> 0`` (fraction -> 1); the correct criterion flags
  the fixed region ``{|u''| > M}``, so the fraction saturates.
* **D** -- the long-time vortex error for ``M`` in a sweep (the number the xfail
  quotes), the smallest ``M`` that is *bit-identical* to the unlimited run, and
  the growth exponent of the limited/unlimited error ratio.

Run::

    PYTHONPATH=. OMP_NUM_THREADS=2 python -m benchmarks.tvb_constant_adjudication

Writes ``validation/euler_dg_2d/tvb_constant_adjudication.json`` and
``docs/figures/tvb_constant_adjudication.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tensordg.euler_dg_2d as dg2  # noqa: E402
from benchmarks.euler_dg_2d_time_split import FINAL_TIME, run_case  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "euler_dg_2d" / "tvb_constant_adjudication.json"
FIG = ROOT / "docs" / "figures" / "tvb_constant_adjudication.png"

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)
COMPONENTS = ("density", "momentum_x", "momentum_y", "energy")
MESHES = (8, 12, 16, 32, 64)          # mask statistics
M_SWEEP = (1.0, 2.0, 10.0)  # error sweep
RUN_MESHES = (8, 12)                  # long-time error (the numbers the xfail quotes)


# ---------------------------------------------------------------------------
# geometry / analytic state
# ---------------------------------------------------------------------------
def cell_centres(n: int):
    """Cell centres of the ``n x n`` uniform mesh on :data:`DOMAIN`."""
    (x0, x1), (y0, y1) = DOMAIN
    dx = (x1 - x0) / n
    dy = (y1 - y0) / n
    xc = torch.linspace(x0 + dx / 2.0, x1 - dx / 2.0, n, dtype=torch.float64)
    yc = torch.linspace(y0 + dy / 2.0, y1 - dy / 2.0, n, dtype=torch.float64)
    return (xc, yc), dx, dy


def meshgrid(n: int):
    """Cell centres as a 2-D meshgrid (point values for the analytic state)."""
    (xc, yc), dx, dy = cell_centres(n)
    return torch.meshgrid(xc, yc, indexing="ij"), dx, dy


def conserved_point_values(x, y):
    """Exact conserved variables of the vortex at ``t = 0``."""
    rho, u, v, p = dg2.vortex_primitive(x, y, 0.0, STRENGTH, BACKGROUND)
    return dg2._conservative(rho, u, v, p)


def projected_cell_averages(n: int, degree: int = 3, quadrature: int = 5):
    """The cell averages the solver itself starts from (L2 projection)."""
    (x, y), dx, dy = cell_centres(n)
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, quadrature)
    initial = lambda a, b: dg2.vortex_primitive(a, b, 0.0, STRENGTH, BACKGROUND)  # noqa: E731
    coeff = dg2._initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis,
                                      torch.device("cpu"), torch.float64)
    return coeff[..., 0, 0, :], dx, dy


def analytic_second_derivative_bound(n_sample: int = 128):
    """``max |d2 f / dx2|`` and ``|d2 f / dy2|`` of the conserved vortex.

    Computed by forward-mode-free ``torch.autograd`` on the analytic primitive
    formula, so there is no finite-difference truncation in the number.
    """
    (x, y), _, _ = meshgrid(n_sample)
    out = {}
    rho, u, v, p = dg2.vortex_primitive(x, y, 0.0, STRENGTH, BACKGROUND)
    values = (rho, u, v, p)
    for name, value in zip(COMPONENTS, values):
        row = {}
        for axis, tag in ((0, "d2_dx2"), (1, "d2_dy2")):
            a = x.clone().requires_grad_(True)
            b = y.clone().requires_grad_(True)
            r = dg2.vortex_primitive(a, b, 0.0, STRENGTH, BACKGROUND)
            idx = ("density", "momentum_x", "momentum_y", "energy").index(name)
            q = r[idx]
            var = a if axis == 0 else b
            g, = torch.autograd.grad(q.sum(), var, create_graph=True)
            d2, = torch.autograd.grad(g.sum(), var)
            row[tag] = float(d2.abs().max())
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# detector masks
# ---------------------------------------------------------------------------
def shipped_masks(averages, dx, dy, constant):
    """Mask from the shipped detector, per axis (``dg2._troubled_1d``)."""
    masks = {}
    for axis, spacing, tag in ((0, dx, "x"), (1, dy, "y")):
        mask, _ = dg2._troubled_1d(averages, axis, spacing, constant, True)
        masks[tag] = mask
    return masks


def mask_record(n, constant):
    """Measured (projected averages) vs analytic-point-value masks on mesh ``n``."""
    averages, dx, dy = projected_cell_averages(n)
    (x, y), _, _ = meshgrid(n)
    analytic = conserved_point_values(x, y)
    measured = shipped_masks(averages, dx, dy, constant)
    predicted = shipped_masks(analytic, dx, dy, constant)
    record = {"cells": n, "h": dx, "tvb_constant": constant}
    for tag in ("x", "y"):
        m, p = measured[tag], predicted[tag]
        record[f"flagged_{tag}"] = int(m.sum())
        record[f"predicted_{tag}"] = int(p.sum())
        record[f"mismatch_{tag}"] = int((m ^ p).sum())
        record[f"fraction_{tag}"] = float(m.float().mean())
    union = measured["x"] | measured["y"]
    record["flagged_any"] = int(union.sum())
    record["fraction_any"] = float(union.float().mean())
    return record, measured, predicted, averages


def detector_table():
    """Section A + B of the report: derivative bound and mask agreement."""
    bound = analytic_second_derivative_bound()
    flat = max(v for row in bound.values() for v in row.values())
    table = []
    for constant in (1.0, 2.0, 4.0):
        for n in MESHES:
            record, _, _, _ = mask_record(n, constant)
            table.append(record)
    return {"derivative_bound": bound, "max_abs_second_derivative": flat, "masks": table}


def sampled_bound_table():
    """``max |d2|`` sampled on the cell centres of each mesh (what the detector sees)."""
    rows = []
    for n in MESHES:
        (x, y), _, _ = meshgrid(n)
        row = {"cells": n}
        best = 0.0
        for name, index in zip(COMPONENTS, range(4)):
            for axis, var in ((0, x), (1, y)):
                a = x.clone().requires_grad_(True)
                b = y.clone().requires_grad_(True)
                q = dg2.vortex_primitive(a, b, 0.0, STRENGTH, BACKGROUND)[index]
                target = a if axis == 0 else b
                g, = torch.autograd.grad(q.sum(), target, create_graph=True)
                d2, = torch.autograd.grad(g.sum(), target)
                row[f"max_d2_{name}_d{'xy'[axis]}"] = float(d2.abs().max())
                best = max(best, float(d2.abs().max()))
        row["max_abs_second_derivative"] = best
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# long-time error sweep (the numbers the retired xfail quotes)
# ---------------------------------------------------------------------------
def error_sweep(log=print):
    rows = []
    for n in RUN_MESHES:
        started = time.time()
        unlimited, _ = run_case(3, n)
        base = unlimited["density"]
        row = {"cells": n, "h": 10.0 / n, "unlimited": base, "tvb": {}}
        for constant in M_SWEEP:
            limited, _ = run_case(3, n, limiter="tvb", limiter_constant=constant)
            value = limited["density"]
            row["tvb"][str(constant)] = {
                "error": value,
                "ratio": value / base,
                "bit_identical_to_unlimited": bool(value == base),
            }
            log(f"  P3 {n:3d}^2 M={constant:<5g} rho_err={value:.6e} ratio={value / base:.3f}")
        identical = [float(k) for k, v in row["tvb"].items() if v["bit_identical_to_unlimited"]]
        row["smallest_inert_constant"] = min(identical) if identical else None
        row["wall_seconds"] = round(time.time() - started, 1)
        rows.append(row)
    ratios = []
    for row in rows:
        if "1.0" in row["tvb"]:
            ratios.append(row["tvb"]["1.0"]["ratio"])
    exponent = None
    if len(ratios) == 2 and min(ratios) > 0:
        exponent = float(math.log(ratios[0] / ratios[1]) / math.log(RUN_MESHES[1] / RUN_MESHES[0]))
    return {"meshes": rows, "ratio_growth_exponent_M1": exponent}


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure(path: Path, record, masks, error_rows):
    measured, predicted = masks["x"], masks["x"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    extent = (-5.0, 5.0, -5.0, 5.0)
    ax.imshow(measured["y"].T, origin="lower", extent=extent, cmap="Greys",
              alpha=0.75, interpolation="nearest")
    ax.contour(predicted["y"].T.astype(float), levels=[0.5], colors="crimson",
               linewidths=1.2, extent=extent, origin="lower")
    ax.set_title(f"flagged cells (y-axis, M=1, 16$^2$)\nred: analytic criterion")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[1]
    rows = record["masks"]
    for constant in (1.0, 2.0, 4.0):
        cells = [r["cells"] for r in rows if r["tvb_constant"] == constant]
        frac = [100.0 * r["fraction_any"] for r in rows if r["tvb_constant"] == constant]
        ax.semilogx(cells, frac, "o-", label=f"M={constant:g}")
    ax.axhline(100.0, color="grey", ls=":", lw=1.0)
    ax.set_xlabel("cells per direction")
    ax.set_ylabel("flagged cell fraction [%]")
    ax.set_title("flagged fraction vs mesh (falsifiable)")
    ax.legend()

    ax = axes[2]
    for row in error_rows["meshes"]:
        constants = sorted(float(k) for k in row["tvb"])
        ratios = [row["tvb"][f"{c:g}"]["ratio"] for c in constants]
        ax.semilogx(constants, ratios, "o-", label=f"{row['cells']}$^2$ cells")
    ax.axhline(1.0, color="grey", ls=":")
    ax.set_xlabel("TVB constant M")
    ax.set_ylabel("rho error / unlimited error")
    ax.set_title("M sweep, P3, t=1")
    ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the long-time error sweep (mask evidence only)")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))

    started = time.time()
    detector = detector_table()
    sampled = sampled_bound_table()
    for row in detector["masks"]:
        print(f"  M={row['tvb_constant']:<4g} {row['cells']:3d}^2 "
              f"flagged_any={row['flagged_any']:3d} ({100 * row['fraction_any']:5.1f}%) "
              f"mismatch(x/y)={row['mismatch_x']}/{row['mismatch_y']}")
    print(f"  analytic max|d2| = {detector['max_abs_second_derivative']:.4f}")

    errors = None
    if not args.quick:
        print("long-time error sweep:")
        errors = error_sweep()

    record = {
        "benchmark": "tvb_constant_adjudication",
        "problem": "periodic smooth isentropic vortex (exact 2-D Euler solution)",
        "domain": [list(DOMAIN[0]), list(DOMAIN[1])],
        "vortex_strength": STRENGTH,
        "background_velocity": list(BACKGROUND),
        "final_time": FINAL_TIME,
        "provenance": {
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "wall_seconds": None,
        },
        "detector": detector,
        "sampled_derivative_bound": sampled,
        "error_sweep": errors,
        "claim_under_test": (
            "default tvb_constant=1.0 fires on the smooth vortex because the "
            "threshold scale/units are off"
        ),
    }
    if errors is not None:
        record["verdict"] = {
            "analytic_bound_exceeds_M1": detector["max_abs_second_derivative"] > 1.0,
            "mask_mismatch_total": sum(r["mismatch_x"] + r["mismatch_y"]
                                       for r in detector["masks"] if r["tvb_constant"] == 1.0),
            "ratio_growth_exponent_M1": errors["ratio_growth_exponent_M1"],
            "smallest_inert_constant": {str(r["cells"]): r["smallest_inert_constant"]
                                        for r in errors["meshes"]},
        }
    record["provenance"]["wall_seconds"] = round(time.time() - started, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")

    if not args.no_figure:
        _, measured, predicted, _ = mask_record(16, 1.0)
        figure(FIG, record, {"x": measured, "y": predicted},
               errors if errors is not None else {"meshes": []})
        print(f"wrote {FIG.relative_to(ROOT)}")
    return record


if __name__ == "__main__":
    main()