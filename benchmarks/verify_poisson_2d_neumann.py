"""Convergence + accuracy verification of the per-side Neumann SIPG Poisson path.

Runs the three manufactured cases of ``tensordg/poisson_2d_bc.py`` (pure
Neumann, mixed zero-flux, mixed inhomogeneous flux), a negative control in
which the flux load is dropped, and writes

    validation/poisson_neumann.json                 (all numbers, programmatic)
    validation/poisson_neumann_convergence.png      (figure for the report)

Usage
-----
    PYTHONPATH=. python -m benchmarks.verify_poisson_2d_neumann [--cells 8 16 32]
    PYTHONPATH=. python -m benchmarks.verify_poisson_2d_neumann --figure-only
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensordg.poisson_2d import poisson_error_2d, solve_poisson_2d  # noqa: E402
from tensordg.poisson_2d_bc import (  # noqa: E402
    l2_error_2d,
    solve_poisson_2d_bc,
)

PI = math.pi
NEUMANN_ALL = {side: "neumann" for side in ("left", "right", "bottom", "top")}
MIXED_ZERO_FLUX = {"left": "neumann", "bottom": "neumann"}

CASES = {
    "A_pure_neumann": {
        "exact": lambda x, y: torch.cos(PI * x) * torch.cos(PI * y),
        "forcing": lambda x, y: 2.0 * PI ** 2 * torch.cos(PI * x) * torch.cos(PI * y),
        "bc": NEUMANN_ALL,
        "flux": None,
        "remove_constant": True,
        "note": "du/dn = 0 on all four sides; operator singular, rank-one gauge",
    },
    "B_mixed_zero_flux": {
        "exact": lambda x, y: torch.cos(0.5 * PI * x) * torch.cos(0.5 * PI * y),
        "forcing": lambda x, y: 0.5 * PI ** 2
        * torch.cos(0.5 * PI * x) * torch.cos(0.5 * PI * y),
        "bc": MIXED_ZERO_FLUX,
        "flux": None,
        "remove_constant": False,
        "note": "Neumann on left/bottom, Dirichlet on right/top",
    },
    "C_mixed_inhomogeneous_flux": {
        "exact": lambda x, y: torch.sin(PI * x) * torch.sin(PI * y),
        "forcing": lambda x, y: 2.0 * PI ** 2 * torch.sin(PI * x) * torch.sin(PI * y),
        "bc": {"left": "neumann"},
        "flux": {"left": lambda x, y: -PI * torch.sin(PI * y)},
        "remove_constant": False,
        "note": "du/dn = -pi sin(pi y) on the left edge, Dirichlet elsewhere",
    },
}


def observed(errors):
    return [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


def _modal_values(coeff, degree, samples):
    nodes = torch.linspace(-1.0, 1.0, samples)
    basis = []
    for k in range(degree + 1):
        poly = np.zeros(k + 1)
        poly[-1] = 1.0
        basis.append(np.polynomial.legendre.legval(nodes.numpy(), poly))
    basis = torch.as_tensor(np.stack(basis, axis=1))
    return torch.einsum("xykl,qk,rl->xyqr", coeff, basis, basis)


def make_figure(results, outdir):
    """Plot the convergence study (p = 1, p = 2) and two solution panels."""
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))

    ax = axes[0, 0]
    for key, value in results["cases"].items():
        if not key.startswith("deg1"):
            continue
        cells = [row["cells"] for row in value["rows"]]
        errors = [row["relative_l2"] for row in value["rows"]]
        ax.loglog(cells, errors, "o-", label=key.split("/")[1])
    cells = [row["cells"] for row in results["cases"]["deg1/A_pure_neumann"]["rows"]]
    ax.loglog(cells, [cells[0] / n ** 2 for n in cells], "k--", lw=1,
              label=r"$h^{2}$ reference")
    ax.axhline(0.03, color="r", ls=":", label="3 % gate")
    ax.set_xlabel("cells per direction")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_title("SIPG convergence with Neumann data (p = 1)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for key, value in results["cases"].items():
        if not key.startswith("deg2"):
            continue
        cells = [row["cells"] for row in value["rows"]]
        errors = [row["relative_l2"] for row in value["rows"]]
        ax.loglog(cells, errors, "s-", label=key.split("/")[1])
    ax.loglog(cells, [cells[0] / n ** 3 for n in cells], "k--", lw=1,
              label=r"$h^{3}$ reference")
    ax.axhline(0.03, color="r", ls=":", label="3 % gate")
    ax.set_xlabel("cells per direction")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_title("p = 2: third order for all three Neumann setups")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    case = CASES["A_pure_neumann"]
    _, _, coeff, _ = solve_poisson_2d_bc(
        ncells=32, degree=1, bc=case["bc"], forcing=case["forcing"],
    )
    values = _modal_values(coeff, 1, 9).numpy()
    ax = axes[1, 0]
    field = values.transpose(0, 2, 1, 3).reshape(32 * 9, 32 * 9)
    image = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    ax.set_title(r"case A: $u_h$, pure Neumann, p = 1, $32^2$ cells")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    case = CASES["C_mixed_inhomogeneous_flux"]
    _, _, coeff, _ = solve_poisson_2d_bc(
        ncells=16, degree=2, bc=case["bc"], forcing=case["forcing"],
        neumann_flux=case["flux"],
    )
    values = _modal_values(coeff, 2, 7)
    strip = values[0]                       # first element column, x in [0, 1/16]
    ynodes = 0.5 / 16 + 0.5 / 16 * torch.linspace(-1.0, 1.0, 7)
    for iy in (0, 8, 15):
        ax.plot(ynodes.numpy(), strip[iy, 3, :].numpy(), "o", ms=4,
                label=f"$u_h$, y-element {iy}")
    yy = np.linspace(0.0, 1.0, 400)
    ax.plot(yy, np.sin(PI * 0.5 / 16) * np.sin(PI * yy), "k-", lw=1.2,
            label=r"exact $\sin(\pi x)\sin(\pi y)$ at $x=h/2$")
    ax.set_xlabel("y")
    ax.set_ylabel("u")
    ax.set_title("case C: inhomogeneous Neumann, p = 2, $16^2$ cells")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(outdir / "poisson_neumann_convergence.png", dpi=72)
    plt.close(fig)
    print(f"figure -> {outdir/'poisson_neumann_convergence.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--figure-only", action="store_true")
    parser.add_argument("--max-dofs", type=int, default=9216)
    args = parser.parse_args()

    outdir = pathlib.Path("validation")
    outdir.mkdir(exist_ok=True)

    if args.figure_only:
        make_figure(json.loads((outdir / "poisson_neumann.json").read_text()), outdir)
        return

    results = {"cases": {}, "degrees": args.degrees, "cells": args.cells}
    started = time.time()

    for degree in args.degrees:
        cells = [n for n in args.cells if (degree + 1) ** 2 * n * n <= args.max_dofs]
        for name, case in CASES.items():
            errors, entry = [], []
            for n in cells:
                t0 = time.time()
                x, y, coeff, meta = solve_poisson_2d_bc(
                    ncells=n, degree=degree, bc=case["bc"],
                    forcing=case["forcing"], neumann_flux=case["flux"],
                )
                elapsed = time.time() - t0
                error = l2_error_2d(
                    x, y, coeff, case["exact"],
                    remove_constant=case["remove_constant"],
                )
                errors.append(error["relative_l2"])
                entry.append({
                    "cells": n, "dofs": meta["dofs"],
                    "relative_l2": error["relative_l2"],
                    "absolute_l2": error["absolute_l2"],
                    "max_abs_error": error["max_abs_error"],
                    "constant_offset": error["constant_offset"],
                    "matrix_symmetry_error": meta["matrix_symmetry_error"],
                    "nullspace_residual": meta["nullspace_residual"],
                    "linear_residual": meta["linear_residual"],
                    "seconds": elapsed,
                })
            results["cases"][f"deg{degree}/{name}"] = {
                "note": case["note"], "bc": case["bc"],
                "rows": entry, "orders": observed(errors),
                "theory_order": degree + 1,
            }

    # Negative control: drop the inhomogeneous flux load of case C.
    control = []
    for n in (8, 16):
        x, y, coeff, _ = solve_poisson_2d_bc(
            ncells=n, degree=1, bc={"left": "neumann"},
            forcing=CASES["C_mixed_inhomogeneous_flux"]["forcing"],
        )
        control.append(l2_error_2d(
            x, y, coeff, CASES["C_mixed_inhomogeneous_flux"]["exact"],
        )["relative_l2"])
    results["negative_control_dropped_flux_deg1"] = {
        "cells": [8, 16], "relative_l2": control,
        "note": "same problem as case C but with the Neumann load omitted",
    }

    # Regression: the all-Dirichlet path must reproduce the shipped solver.
    reference = []
    for degree in args.degrees:
        for n in (8, 16):
            x_ref, y_ref, coeff_ref, _ = solve_poisson_2d(ncells=n, degree=degree)
            _, _, coeff, _ = solve_poisson_2d_bc(
                ncells=n, degree=degree,
                forcing=CASES["C_mixed_inhomogeneous_flux"]["forcing"],
            )
            reference.append({
                "degree": degree, "cells": n,
                "max_abs_difference": float(torch.max(torch.abs(coeff - coeff_ref)).item()),
                "reference_relative_l2": poisson_error_2d(
                    x_ref, y_ref, coeff_ref,
                )["relative_l2"],
            })
    results["dirichlet_path_regression"] = reference
    results["elapsed_seconds"] = time.time() - started

    (outdir / "poisson_neumann.json").write_text(json.dumps(results, indent=2))

    for key, value in results["cases"].items():
        print(f"{key:36s} theory {value['theory_order']}   "
              f"orders {[round(o, 3) for o in value['orders']]}")
        for row in value["rows"]:
            print(f"    n={row['cells']:3d} dofs={row['dofs']:5d} "
                  f"relL2={row['relative_l2']:.4e} maxabs={row['max_abs_error']:.3e}")
    print("negative control (flux dropped):", control)
    print("dirichlet regression max|diff|:",
          [item["max_abs_difference"] for item in reference])
    print(f"elapsed {results['elapsed_seconds']:.1f} s -> {outdir/'poisson_neumann.json'}")

    make_figure(results, outdir)


if __name__ == "__main__":
    main()