"""Accuracy ladder of the isothermal-wall heat-flux estimators (m16493).

What this script measures
-------------------------
Four ways of reporting the wall-normal conductive heat flux ``q_n = -kappa dT/dn``
of the *same* DG state, on the same exactly-solvable steady conduction state:

===============  ==============================================================
``solver``       :func:`tensordg.ns_dg_2d.wall_heat_flux` -- the flux the
                 strong-form wall closure actually used (design order ``p``).
``fd2``          ``wall_heat_flux`` with the opt-in two-point wall stencil
                 (``wall_gradient="fd2"``, i.e. wall value + 2 cell centres
                 inside the operator).
``extrapolated`` :func:`tensordg.wall_flux_extrapolation.wall_heat_flux_extrapolated`
                 -- Lagrange extrapolation of the interior *face* fluxes.
``stencil``      :func:`tensordg.wall_flux_stencil.wall_heat_flux_stencil` -- new
                 one-sided stencil on the cell-centre temperatures (``points``
                 cell centres + the imposed wall value).
===============  ==============================================================

Two independent analytic families are run, so the axis handling of the new
estimator is not inferred from rotation equivalence alone:

* ``y`` -- walls at ``ymin``/``ymax`` (``T = 1.0`` / ``0.5``), periodic in x;
* ``x`` -- walls at ``xmin``/``xmax``, periodic in y (same temperature pair).

Reference value: ``q = kappa (T_hot - T_cold) / L = 0.246479`` (Fourier's law,
``L = 1``, same positive sign at both walls in the operator's convention).

Predeclared decision and thresholds (fixed before the numbers were inspected,
see ``docs/reports/tensordg-ns-wall-flux-stencil.md`` §1.2):

* ``D1`` at P2 the new stencil reaches order ``>= 2.75`` **and** a finest-grid
  error at least 100x below the solver flux -> recommended estimator for ``p=2``;
* ``D2`` at P1 the stencil stays at order ``<= 1.3`` (the ``O(h**2)`` centre bias
  divided by the one-sided ``O(1/h)`` weight) while the face extrapolation stays
  ``>= 1.9`` -> the extrapolation remains the recommended P1 estimator;
* ``D3`` the ``x`` family reproduces the ``y`` family to within 5 % on the
  finest mesh -> the axis handling is verified on both axes, independently;
* ``D4`` ``stencil(points=2)`` reproduces the operator's own opt-in ``fd2`` wall
  flux (same stencil, independent code path) to round-off -> the sign/normal
  conventions of the new estimator are validated, not assumed;
* ``D5`` (honest negative, recorded not to oversell the ladder) at P3 the stencil
  does **not** beat the solver flux order; the estimator is recommended for
  ``p = 2`` only.

Every reported number is written to JSON by this script and re-read from the
artifact, so nothing in the report is transcribed by hand.

Run::

    PYTHONPATH=. python -m benchmarks.verify_ns_wall_flux_stencil

writes ``validation/ns_heat_conduction/wall_flux_stencil.json`` (evidence,
tracked) and ``runs/m16493_ns_wall_flux_stencil/`` (run log + manifest + figure).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch

import tensordg.ns_dg_2d as ns
from benchmarks.verify_ns_wall_heat_flux import (
    DEFAULT_CELLS,
    EXACT_WALL_FLUX,
    KAPPA,
    MU,
    NX,
    PRANDTL,
    PRESSURE,
    T_COLD,
    T_HOT,
    _boundary,
    _projected_state,
    observed_order,
)
from tensordg.wall_flux_extrapolation import wall_heat_flux_extrapolated_summary
from tensordg.wall_flux_stencil import (
    stencil_weights,
    wall_heat_flux_stencil_summary,
)

DEFAULT_OUT = Path("validation/ns_heat_conduction/wall_flux_stencil.json")
DEFAULT_RUNS = Path("runs/m16493_ns_wall_flux_stencil")
DEFAULT_FIGURE = Path("docs/figures/ns_wall_flux_stencil_order.png")

DEGREES = (1, 2, 3)
POINTS = (1, 2, 3)


def _steady_x(x, y):
    """Exact steady conduction between ``xmin`` (hot) and ``xmax`` (cold)."""
    temperature = T_HOT + (T_COLD - T_HOT) * (x + 0.0 * y)
    return (PRESSURE / temperature, torch.zeros_like(x), torch.zeros_like(y),
            PRESSURE * torch.ones_like(x))


def _boundary_x():
    return {"xmin": {"velocity": (0.0, 0.0), "temperature": T_HOT},
            "xmax": {"velocity": (0.0, 0.0), "temperature": T_COLD},
            "ymin": None, "ymax": None}


def _projected_state_x(ncells, degree):
    """L2 projection of the x-direction conduction state on ``ncells x NX``."""
    nx, ny = ncells, NX
    dx, dy = 1.0 / nx, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(_steady_x, x, y, dx, dy, nodes, weights,
                                     basis, torch.device("cpu"), torch.float64)
    return coeff, (nodes, weights, basis, derivative, signs), (dx, dy)


FAMILIES = {
    "y": {"walls": ("ymin", "ymax"), "periodic": (True, False),
          "project": _projected_state, "boundary": lambda: _boundary("face")},
    "x": {"walls": ("xmin", "xmax"), "periodic": (False, True),
          "project": _projected_state_x, "boundary": _boundary_x},
}


def _estimators(coeff, degree, ops, dx, dy, boundary, periodic):
    """Every estimator applied to one DG state."""
    nodes, weights, basis, derivative, signs = ops
    penalty = 2.0 * (degree + 1.0) ** 2
    result = {}
    for label, gradient in (("solver", "face"), ("fd2", "fd2")):
        result[label] = ns.wall_heat_flux_summary(
            coeff, basis, derivative, signs, weights, dx, dy, boundary, MU,
            KAPPA, penalty, gradient, periodic=periodic)
    result["extrapolated"] = wall_heat_flux_extrapolated_summary(
        coeff, basis, derivative, signs, weights, dx, dy, boundary, MU, KAPPA,
        penalty, "face", None, periodic, 2)
    for points in POINTS:
        result[f"stencil{points}"] = wall_heat_flux_stencil_summary(
            coeff, dx, dy, boundary, KAPPA, points, periodic)
    return result


def centre_data_order(ncells, degree, family):
    """Observed order of the cell-centre temperature bias of the same states."""
    spec = FAMILIES[family]
    errors = []
    for count in ncells:
        coeff, _, (dx, dy) = spec["project"](count, degree)
        temperature = ns._cell_centre_primitives(coeff)[2]
        axis_size, spacing = (count, dx) if family == "x" else (count, dy)
        centres = (torch.arange(axis_size, dtype=torch.float64) + 0.5) * spacing
        exact = T_HOT + (T_COLD - T_HOT) * centres
        values = temperature[0] if family == "y" else temperature[:, 0]
        errors.append(float((values - exact).abs().max()))
    return {"errors": errors, "order": observed_order(errors, ncells)}


def run(cells=DEFAULT_CELLS, degrees=DEGREES):
    cases = []
    for family, spec in FAMILIES.items():
        for degree in degrees:
            rows = {}
            for count in cells:
                coeff, ops, (dx, dy) = spec["project"](count, degree)
                rows[count] = _estimators(coeff, degree, ops, dx, dy,
                                          spec["boundary"](), spec["periodic"])
            record = {"family": family, "degree": degree, "cells": list(cells),
                      "walls": list(spec["walls"]), "estimators": {},
                      "ref": EXACT_WALL_FLUX}
            for label in rows[cells[0]]:
                entry = {}
                for wall in spec["walls"]:
                    values = [rows[count][label][wall] for count in cells]
                    errors = [abs(value - EXACT_WALL_FLUX) for value in values]
                    entry[wall] = {
                        "values": values,
                        "errors": errors,
                        "rel_err_pct": [100.0 * err / EXACT_WALL_FLUX for err in errors],
                        "order": observed_order(errors, cells),
                    }
                record["estimators"][label] = entry
            record["centre_data"] = centre_data_order(cells, degree, family)
            cases.append(record)
    return cases


def verdict(cases):
    """Predeclared decisions D1-D5 evaluated on the measured numbers."""
    def entry(family, degree, label, wall):
        for case in cases:
            if case["family"] == family and case["degree"] == degree:
                return case["estimators"][label][wall]
        raise KeyError((family, degree, label))

    y_walls = FAMILIES["y"]["walls"]
    d1 = {}
    for degree in (2,):
        for wall in y_walls:
            stencil = entry("y", degree, "stencil1", wall)
            solver = entry("y", degree, "solver", wall)
            d1[f"P{degree}_{wall}"] = {
                "order": stencil["order"],
                "error_gain_over_solver": solver["errors"][-1] / max(stencil["errors"][-1], 1e-300),
                "pass": bool(stencil["order"] >= 2.75
                             and solver["errors"][-1] >= 100.0 * stencil["errors"][-1]),
            }
    d2 = {}
    for wall in y_walls:
        stencil = entry("y", 1, "stencil2", wall)
        extrapolated = entry("y", 1, "extrapolated", wall)
        d2[f"P1_{wall}"] = {
            "stencil_order": stencil["order"],
            "extrapolated_order": extrapolated["order"],
            "pass": bool(stencil["order"] <= 1.3 and extrapolated["order"] >= 1.9),
        }
    d3 = {}
    for degree in DEGREES:
        for label in ("stencil1", "stencil2", "extrapolated"):
            y_err = entry("y", degree, label, "ymin")["errors"][-1]
            x_err = entry("x", degree, label, "xmin")["errors"][-1]
            d3[f"P{degree}_{label}"] = {
                "y_err": y_err, "x_err": x_err,
                "pass": bool(abs(x_err - y_err) <= 0.05 * max(y_err, 1e-16)),
            }
    d4 = {}
    for degree in DEGREES:
        worst = 0.0
        for wall in y_walls:
            stencil = entry("y", degree, "stencil2", wall)["values"]
            fd2 = entry("y", degree, "fd2", wall)["values"]
            worst = max(worst, max(abs(a - b) for a, b in zip(stencil, fd2)))
        d4[f"P{degree}"] = {"max_abs_diff_vs_fd2": worst, "pass": bool(worst <= 1e-12)}
    d5 = {}
    for wall in y_walls:
        stencil = entry("y", 3, "stencil1", wall)["order"]
        solver = entry("y", 3, "solver", wall)["order"]
        d5[f"P3_{wall}"] = {"stencil_order": stencil, "solver_order": solver,
                            "stencil_not_better": bool(stencil <= solver + 0.25)}
    return {
        "D1_P2_stencil_order_and_error_gain": d1,
        "D2_P1_stencil_floor_and_extrapolation_headroom": d2,
        "D3_x_family_matches_y_family_within_5pct": d3,
        "D4_stencil2_reproduces_operator_fd2": d4,
        "D5_P3_stencil_does_not_beat_solver": d5,
        "all_pass": (all(item["pass"] for item in d1.values())
                     and all(item["pass"] for item in d2.values())
                     and all(item["pass"] for item in d3.values())
                     and all(item["pass"] for item in d4.values())
                     and all(item["stencil_not_better"] for item in d5.values())),
    }


def _meta(command):
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--", "tensordg", "benchmarks",
             "tests", "docs", "tools", "README.md", "pyproject.toml"],
            capture_output=True, text=True, check=False).stdout.strip())
    except Exception:  # pragma: no cover - git is always present in CI
        sha, dirty = "", True
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": sha,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "host": platform.node(),
        "command": command,
        "mu": MU,
        "prandtl": PRANDTL,
        "kappa": KAPPA,
        "temperature_walls": [T_HOT, T_COLD],
        "ref_flux": EXACT_WALL_FLUX,
        "stencil_weights": {str(points): stencil_weights(points) for points in POINTS},
        "thresholds": {
            "D1_stencil_order_at_least": 2.75,
            "D1_error_gain_over_solver_at_least": 100.0,
            "D2_stencil_floor_order_at_most": 1.3,
            "D2_extrapolated_order_at_least": 1.9,
            "D3_family_agreement_rel": 0.05,
            "D4_stencil2_vs_fd2_abs": 1e-12,
        },
    }


def figure(cases, path):
    """Four panels: error ladder, order summary, state profile, centre bias."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    colours = {"solver": "tab:gray", "fd2": "tab:orange",
               "extrapolated": "tab:blue", "stencil1": "tab:red",
               "stencil2": "tab:green", "stencil3": "tab:purple"}
    for case in cases:
        if case["family"] != "y":
            continue
        cells = case["cells"]
        for label, entry in case["estimators"].items():
            errors = entry["ymin"]["errors"]
            axes[0, 0].loglog(cells, errors, "o-", color=colours[label],
                              label=f"P{case['degree']} {label}")
    axes[0, 0].set_xlabel("cells across the gap (ny)")
    axes[0, 0].set_ylabel(r"$|q_{wall} - \kappa/2|$")
    axes[0, 0].set_title("hot wall, y-walls family: error vs mesh")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=6, ncol=2)

    labels, values, colours_bar = [], [], []
    for case in cases:
        if case["family"] != "y":
            continue
        for label in ("solver", "fd2", "extrapolated", "stencil1", "stencil2"):
            labels.append(f"P{case['degree']} {label}")
            values.append(case["estimators"][label]["ymin"]["order"])
            colours_bar.append(colours[label])
    axes[0, 1].bar(range(len(values)), values, color=colours_bar)
    axes[0, 1].set_xticks(range(len(values)))
    axes[0, 1].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[0, 1].set_ylabel("observed order (log-log fit)")
    axes[0, 1].plot([-0.5, len(values) - 0.5], [2.75, 2.75], "k--", lw=0.8,
                    label="D1 threshold 2.75")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].set_title("observed convergence order")

    # the field the estimators read: cell-centre temperature vs the exact line
    cells = 16
    for degree, colour in ((1, "tab:red"), (2, "tab:green")):
        coeff, _, (dx, dy) = _projected_state(cells, degree)
        temperature = ns._cell_centre_primitives(coeff)[2]
        y = (torch.arange(cells, dtype=torch.float64) + 0.5) * dy
        exact = T_HOT + (T_COLD - T_HOT) * y
        axes[1, 0].plot(y, temperature[0], "o", ms=4, color=colour,
                        label=f"P{degree} cell centres")
        axes[1, 1].semilogy(y, (temperature[0] - exact).abs(), "o-", ms=3,
                            color=colour, label=f"P{degree}")
    axes[1, 0].plot(y, exact, "k-", lw=1.0, label="exact T(y)")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].set_ylabel("T at cell centres")
    axes[1, 0].set_title(f"measured state, ny={cells}")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 1].set_xlabel("y")
    axes[1, 1].set_ylabel(r"$|T_h - T|$ at cell centres")
    axes[1, 1].set_title("centre-value bias (the stencil's ceiling)")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=list(DEFAULT_CELLS))
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEGREES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    start = time.time()
    command = "PYTHONPATH=. python -m benchmarks.verify_ns_wall_flux_stencil"
    cases = run(tuple(args.cells), tuple(args.degrees))
    summary = {
        "meta": _meta(command),
        "verdict": verdict(cases),
        "cases": cases,
        "runtime_s": time.time() - start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out}")

    for case in cases:
        data = case["centre_data"]
        print(f"[{case['family']}] P{case['degree']} centre-bias order "
              f"{data['order']:.2f}")
        for label, entry in case["estimators"].items():
            print(f"  {label:12s} " + " ".join(
                f"{wall}: err={entry[wall]['errors'][-1]:.3e} "
                f"order={entry[wall]['order']:+.2f}"
                for wall in case["walls"]))
    print(json.dumps(summary["verdict"], indent=2))

    manifest = {"meta": summary["meta"], "runtime_s": summary["runtime_s"],
                "artifacts": []}
    args.runs.mkdir(parents=True, exist_ok=True)
    if not args.no_figure:
        figure(cases, args.figure)
        manifest["artifacts"].append(str(args.figure))
    manifest["artifacts"].append(str(args.out))
    for artifact in manifest["artifacts"]:
        payload = Path(artifact).read_bytes()
        manifest.setdefault("sha256", {})[artifact] = hashlib.sha256(payload).hexdigest()
    (args.runs / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.figure} and {args.runs}/manifest.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())