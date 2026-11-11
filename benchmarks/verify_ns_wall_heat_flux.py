"""Wall heat-flux benchmark for the isothermal channel of ``tensordg.ns_dg_2d``.

What this script measures
-------------------------
``tests/test_ns_heat_conduction.py`` carries two ``xfail(strict=True)`` pins that
record an O(1) energy residual of the *exact* steady conduction state on the wall
row, and their historical documentation attributed that to an *inconsistent*
isothermal wall energy closure.  This benchmark measures the quantities that
decide the question, and nothing else:

* ``wall_flux``      : the total energy flux the wall closure puts into the
  strong-form divergence, read back through the public API
  :func:`tensordg.ns_dg_2d.wall_heat_flux`, compared with the exact
  ``-kappa dT/dn = kappa/2``;
* ``interior_flux``  : the same quantity on the neighbouring interior face;
* ``row_residual``   : the wall-row energy residual (the pinned quantity) and
  the volume-weighted global residual ``||R||`` of the whole discrete operator;
* ``solver_case``    : the same wall flux as reported by the *solver* itself
  (``solve_navier_stokes_dg_2d`` -> ``metadata["wall_heat_flux"]``), i.e. the
  integration path, on a short transient from the exact steady state.

If the wall flux converges to the exact value, the closure is *consistent* and
the stagnation of ``row_residual`` is a measure-``O(h)`` boundary effect whose
strong-form residual is ``O(h) / h`` -- not a solution-accuracy defect.  If it
does not converge, the closure itself is wrong.

Every number in the report is written by this script (``ref``/``got``/``err``)
and re-read from the artifact; nothing is transcribed by hand.

Run::

    PYTHONPATH=. python -m benchmarks.verify_ns_wall_heat_flux            # gates
    PYTHONPATH=. python -m benchmarks.verify_ns_wall_heat_flux --full     # + solver case

writes ``validation/ns_heat_conduction/wall_flux_order.json`` (evidence, tracked)
and ``runs/m14554_ns_wall_flux/`` (raw run log + artifact manifest + figure).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

import tensordg.ns_dg_2d as ns

MU = 0.1
PRANDTL = 0.71
KAPPA = MU * ns.CP / PRANDTL
T_HOT, T_COLD, PRESSURE = 1.0, 0.5, 1.0
EXACT_WALL_FLUX = KAPPA * (T_HOT - T_COLD)  # unit gap, -kappa dT/dn
NX = 2
DEFAULT_CELLS = (8, 16, 32, 64)
DEFAULT_OUT = Path("validation/ns_heat_conduction/wall_flux_order.json")
DEFAULT_RUNS = Path("runs/m14554_ns_wall_flux")


def _steady(x, y):
    """Exact steady conduction state (satisfies the compressible NS exactly)."""
    reference = x * y + 0.0 * x + 0.0 * y
    temperature = T_HOT + (T_COLD - T_HOT) * (y + 0.0 * x)
    return (PRESSURE / temperature, torch.zeros_like(reference),
            torch.zeros_like(reference), PRESSURE * torch.ones_like(reference))


def _boundary(wall_gradient):
    return {"xmin": None, "xmax": None,
            "ymin": {"velocity": (0.0, 0.0), "temperature": T_HOT},
            "ymax": {"velocity": (0.0, 0.0), "temperature": T_COLD}}


def _projected_state(ny, degree):
    """L2 projection of the exact state on the ``NX x ny`` tensor-product mesh."""
    dx, dy = 1.0 / NX, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(NX, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(_steady, x, y, dx, dy, nodes, weights, basis,
                                     torch.device("cpu"), torch.float64)
    return coeff, (nodes, weights, basis, derivative, signs), (dx, dy)


def _case(ny, degree, wall_gradient):
    """One mesh case: public-API wall flux, interior flux and residuals."""
    coeff, ops, (dx, dy) = _projected_state(ny, degree)
    nodes, weights, basis, derivative, signs = ops
    penalty = 2.0 * (degree + 1.0) ** 2
    boundary = _boundary(wall_gradient)

    # (1) the public post-processing API -- the same call the solver makes
    flux = ns.wall_heat_flux(coeff, basis, derivative, signs, weights, dx, dy,
                             boundary, MU, KAPPA, penalty, wall_gradient)
    wall_flux = float(flux["ymin"].mean())
    wall_flux_top = float(flux["ymax"].mean())

    # (2) the neighbouring interior face, straight from the operator
    right_y, _, _, _ = ns._axis_fluxes_ns(
        coeff, basis, derivative, signs, weights, dx, dy, False,
        boundary["ymin"], boundary["ymax"], MU, KAPPA, penalty, 1, wall_gradient)
    # right_y[:, j] is the face to the *right* of row j, projected as int F* phi_k
    interior_flux = float(right_y[:, 0, 0, 3].mean()) / 2.0

    # (3) residuals of the same operator, energy component
    rhs = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                     (True, False), boundary, MU, KAPPA, penalty, wall_gradient)
    modal = rhs[..., 3]
    modes = torch.arange(degree + 1, dtype=torch.float64)
    mass = 1.0 / ((2.0 * modes + 1.0) * (2.0 * modes + 1.0))
    l2 = float((modal.pow(2) * mass.view(1, 1, *mass.shape, 1)).sum().sqrt()) \
        * math.sqrt(dx * dy / 4.0)
    return {
        "ny": ny,
        "degree": degree,
        "wall_gradient": wall_gradient,
        "wall_flux": wall_flux,
        "wall_flux_top": wall_flux_top,
        "interior_flux": interior_flux,
        "row_residual": float(modal[:, 0].abs().max()),
        "row_residual_inner": float(modal[:, 1:-1].abs().max()),
        "l2_residual": l2,
    }


def observed_order(values, cells):
    """Log-log least-squares slope of ``values`` against the mesh size."""
    xs = [math.log(n) for n in cells]
    ys = [math.log(v) for v in values]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def solver_case(degree, ny, final_time):
    """Integration path: the flux the *solver* reports through its metadata.

    Starts from the exact steady state so the analytic reference applies, and
    advances a short transient (``final_time``) with the shipped defaults; the
    returned number comes from ``metadata["wall_heat_flux"]``, i.e. from
    ``solve_navier_stokes_dg_2d`` -> :func:`tensordg.ns_dg_2d.wall_heat_flux`.
    """
    start = time.time()
    _, _, _, metadata = ns.solve_navier_stokes_dg_2d(
        _steady, ncells=(NX, ny), degree=degree, final_time=final_time, cfl=0.3,
        mu=MU, prandtl=PRANDTL, periodic=(True, False),
        boundary=_boundary("face"), dtype=torch.float64)
    reported = metadata["wall_heat_flux"]
    row = {
        "degree": degree,
        "ny": ny,
        "final_time": final_time,
        "steps": metadata["steps"],
        "dt": metadata["dt"],
        "ref": EXACT_WALL_FLUX,
        "got_ymin": reported["ymin"],
        "got_ymax": reported["ymax"],
        "rel_err_ymin_pct": 100.0 * (reported["ymin"] - EXACT_WALL_FLUX) / EXACT_WALL_FLUX,
        "rel_err_ymax_pct": 100.0 * (reported["ymax"] - EXACT_WALL_FLUX) / EXACT_WALL_FLUX,
        "runtime_s": time.time() - start,
    }
    return row


def _meta():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip()
        # only the source tree counts: the run itself rewrites runs/ and
        # validation/, so those must not mark the provenance as dirty
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
        "numpy_backend": "torch.float64",
        "host": platform.node(),
        "command": "PYTHONPATH=. python -m benchmarks.verify_ns_wall_heat_flux",
        "nx": NX,
        "cells": list(DEFAULT_CELLS),
        "mu": MU,
        "prandtl": PRANDTL,
        "kappa": KAPPA,
        "ref_flux": EXACT_WALL_FLUX,
    }


def _figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [case["ny"] for case in summary["cases"] if case["degree"] == 1
             and case["wall_gradient"] == "face"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.axhline(EXACT_WALL_FLUX, color="k", ls="--", lw=1.0,
               label=r"exact $\kappa (T_h-T_c)/L$")
    for degree in (1, 2):
        for gradient, style in (("face", "o-"), ("taylor", "s-")):
            rows = [c for c in summary["cases"]
                    if c["degree"] == degree and c["wall_gradient"] == gradient]
            ax.plot([c["ny"] for c in rows], [c["wall_flux"] for c in rows], style,
                    label=f"wall P{degree} {gradient}")
            ax.plot([c["ny"] for c in rows], [c["interior_flux"] for c in rows],
                    style, alpha=0.35)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cells in y")
    ax.set_ylabel("energy flux  [$-\\kappa \\partial_y T$]")
    ax.set_title("wall (solid markers) vs interior (faint) flux")
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for degree in (1, 2):
        for gradient, style in (("face", "o-"), ("taylor", "s-")):
            rows = [c for c in summary["cases"]
                    if c["degree"] == degree and c["wall_gradient"] == gradient]
            ax.plot([c["ny"] for c in rows],
                    [abs(c["wall_flux_error"]) for c in rows], style,
                    label=f"|wall err| P{degree} {gradient}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("cells in y")
    ax.set_ylabel("|wall flux - exact|")
    ax.set_title("wall flux error vs mesh")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def build(cells=DEFAULT_CELLS, degrees=(1, 2), gradients=("taylor", "face"),
          full=False, solver_final_time=0.01):
    summary = {"kappa": KAPPA, "exact_wall_flux": EXACT_WALL_FLUX,
               "prandtl": PRANDTL, "mu": MU,
               "temperatures": {"hot": T_HOT, "cold": T_COLD},
               "pressure": PRESSURE, "cells": list(cells)}
    rows = []
    for degree in degrees:
        for wall_gradient in gradients:
            cases = [_case(ny, degree, wall_gradient) for ny in cells]
            for case in cases:
                case["ref"] = EXACT_WALL_FLUX
                case["wall_flux_error"] = case["wall_flux"] - EXACT_WALL_FLUX
                case["rel_err_pct"] = 100.0 * case["wall_flux_error"] / EXACT_WALL_FLUX
                case["interior_flux_error"] = case["interior_flux"] - EXACT_WALL_FLUX
            orders = {
                "wall_flux_error": observed_order(
                    [abs(c["wall_flux_error"]) for c in cases], cells),
                "interior_flux_error": observed_order(
                    [abs(c["interior_flux_error"]) for c in cases], cells),
                "row_residual": observed_order([c["row_residual"] for c in cases], cells),
                "row_residual_inner": observed_order(
                    [c["row_residual_inner"] for c in cases], cells),
                "l2_residual": observed_order([c["l2_residual"] for c in cases], cells),
            }
            summary.setdefault("orders", {})[f"p{degree}_{wall_gradient}"] = orders
            rows.extend(cases)
    summary["cases"] = rows
    summary["benchmark"] = [  # ref vs got, one line per (degree, gradient, finest mesh)
        {"degree": c["degree"], "wall_gradient": c["wall_gradient"], "ny": c["ny"],
         "ref": c["ref"], "got": c["wall_flux"],
         "abs_err": abs(c["wall_flux_error"]), "rel_err_pct": c["rel_err_pct"],
         "observed_order": summary["orders"][f"p{c['degree']}_{c['wall_gradient']}"]["wall_flux_error"]}
        for c in rows if c["ny"] == max(cells)]
    if full:
        summary["solver_integration"] = [
            solver_case(degree, 16, solver_final_time) for degree in (1, 2)]
    summary["meta"] = _meta()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=list(DEFAULT_CELLS))
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--gradients", nargs="+", default=["taylor", "face"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--full", action="store_true",
                        help="also run the solver integration case")
    parser.add_argument("--solver-final-time", type=float, default=0.01)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="allow a reduced sweep to overwrite the canonical "
                             "evidence file (default: refuse)")
    args = parser.parse_args(argv)

    reduced = (tuple(args.cells) != DEFAULT_CELLS or tuple(args.degrees) != (1, 2)
               or tuple(args.gradients) != ("taylor", "face") or not args.full)
    if args.out == DEFAULT_OUT and reduced and not args.force:
        parser.error("refusing to overwrite the canonical evidence "
                     f"{DEFAULT_OUT} with a reduced sweep (cells={args.cells}, "
                     f"degrees={args.degrees}, gradients={args.gradients}, "
                     f"full={args.full}); pass --force or a different --out")

    summary = build(cells=tuple(args.cells), degrees=tuple(args.degrees),
                    gradients=tuple(args.gradients), full=args.full,
                    solver_final_time=args.solver_final_time)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    args.runs.mkdir(parents=True, exist_ok=True)

    print(f"kappa = {KAPPA:.6f}   exact wall energy flux = {EXACT_WALL_FLUX:.6f}")
    header = (f"{'deg':>3} {'wg':>6} {'ny':>4} {'wall_flux':>10} {'err':>10}"
              f" {'rel%':>7} {'interior':>10} {'err':>10} {'row_res':>9} {'l2_res':>10}")
    print(header)
    for case in summary["cases"]:
        print(f"{case['degree']:>3} {case['wall_gradient']:>6} {case['ny']:>4}"
              f" {case['wall_flux']:10.6f} {case['wall_flux_error']:+10.2e}"
              f" {case['rel_err_pct']:+7.3f}"
              f" {case['interior_flux']:10.6f} {case['interior_flux_error']:+10.2e}"
              f" {case['row_residual']:9.4f} {case['l2_residual']:10.3e}")
    print("\nobserved orders (log-log slope over "
          f"{list(args.cells)}):")
    for key, orders in summary["orders"].items():
        print(f"  {key}: " + "  ".join(f"{name}={value:+.2f}"
                                       for name, value in orders.items()))
    for row in summary.get("solver_integration", []):
        print(f"solver P{row['degree']} ny={row['ny']} t={row['final_time']}: "
              f"ref={row['ref']:.6f} got(ymin)={row['got_ymin']:.6f} got(ymax)="
              f"{row['got_ymax']:.6f} rel_err={row['rel_err_ymin_pct']:+.3f}%/"
              f"{row['rel_err_ymax_pct']:+.3f}% steps={row['steps']}")

    artifacts = {"benchmark": str(args.out)}
    if not args.no_figure:
        figure = _figure(summary, args.runs / "ns_wall_flux_order.png")
        artifacts["figure"] = str(figure)
    # run.log is the captured stdout of this very process (tee-owned), so the
    # manifest hashes the artifacts this process writes: evidence JSON + figure
    digests = {Path(path).as_posix(): hashlib.sha256(Path(path).read_bytes()).hexdigest()
               for path in artifacts.values() if Path(path).is_file()}
    manifest = {"artifacts": artifacts, "sha256": digests, "meta": summary["meta"]}
    (args.runs / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out}\nwrote {args.runs}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())