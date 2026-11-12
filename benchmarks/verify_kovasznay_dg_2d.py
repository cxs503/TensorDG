"""Verification driver for the steady Kovasznay Navier--Stokes case.

Case card: ``CASE-DG-KOVASZNAY-NS`` -- steady Navier--Stokes, ``Re = 40``,
DG degree ``3`` on a ``24 x 24`` mesh, gate = relative ``L2`` velocity error
against Kovasznay's (1948) exact solution below ``1 %``.

No number in the report is typed by hand: this script is the only producer of
the tables, it writes one JSON per run into ``validation/kovasznay/`` and the
figures into ``docs/figures/``.

Usage::

    PYTHONPATH=. python -m benchmarks.verify_kovasznay_dg_2d --job primary
    PYTHONPATH=. python -m benchmarks.verify_kovasznay_dg_2d --job mach120,mach030
    PYTHONPATH=. python -m benchmarks.verify_kovasznay_dg_2d --job summary

``--job`` accepts any comma-separated list of run names from :data:`RUNS`.
Each run is independent, so a mesh/Mach sweep can be sharded over cluster
jobs; ``--job summary`` aggregates whatever JSON files exist.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from tensordg.kovasznay import (
    CP,
    DEFAULT_REYNOLDS,
    DEFAULT_WAVENUMBER,
    decay_rate,
    default_domain,
    kovasznay_exact,
    kovasznay_residual,
    solve_kovasznay_dg_2d,
    velocity_l2_error,
)

torch.set_num_threads(int(os.environ.get("KOVASZNAY_THREADS", "1")))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "validation", "kovasznay")
FIGS = os.path.join(ROOT, "docs", "figures")

#: Gate of the case card: relative L2 velocity error.
GATE = 0.01

#: Every run: Re = 40 Kovasznay vortex array on x in [-0.5, 1], y in [-0.5, 1.5].
RUNS = {
    # name: (degree, ncells, sound_speed, final_time, chunks, extra kwargs)
    "primary": (3, 24, 60.0, 0.08, 8, {}),
    "mach120": (3, 24, 120.0, 0.2, 10, {}),
    "mach030": (3, 24, 30.0, 0.2, 10, {}),
    "p1_12": (1, 12, 60.0, 0.2, 10, {}),
    "p2_12": (2, 12, 60.0, 0.2, 10, {}),
    "p3_12": (3, 12, 60.0, 0.2, 10, {}),
    "p1_24": (1, 24, 60.0, 0.2, 10, {}),
    "p2_24": (2, 24, 60.0, 0.2, 10, {}),
    "bc_extrap": (3, 24, 60.0, 0.2, 10, {"outlet": "extrapolate"}),
    "bc_adjoint": (3, 24, 60.0, 0.2, 10, {"wall_adjoint": True}),
    "ic_uniform": (1, 12, 30.0, 1.0, 20, {"initial_mode": "uniform"}),
    "ic_exact_ref": (1, 12, 30.0, 1.0, 20, {}),
    "wavenumber3": (3, 24, 60.0, 0.2, 10, {"wavenumber": 3.0}),
}


def _case(name):
    degree, ncells, sound, final_time, chunks, extra = RUNS[name]
    wavenumber = extra.get("wavenumber", DEFAULT_WAVENUMBER)
    domain = default_domain(wavenumber)
    started = time.time()
    x, y, coeff, metadata = solve_kovasznay_dg_2d(
        ncells=ncells,
        degree=degree,
        reynolds=DEFAULT_REYNOLDS,
        wavenumber=wavenumber,
        sound_speed=sound,
        final_time=final_time,
        chunks=chunks,
        cfl=0.5,
        **extra,
    )
    wall = time.time() - started
    info = metadata["kovasznay"]
    pressure_offset = info["pressure_offset"]
    final = metadata["history"][-1]
    record = {
        "run": name,
        "config": {
            "reynolds": DEFAULT_REYNOLDS,
            "wavenumber": wavenumber,
            "decay_rate": decay_rate(DEFAULT_REYNOLDS, wavenumber),
            "degree": degree,
            "ncells": list(metadata["kovasznay"]["ncells"]),
            "domain": [list(interval) for interval in domain],
            "sound_speed": sound,
            "mach": info["mach"],
            "mu": info["mu"],
            "kappa": info["mu"] * CP / 0.71,
            "prandtl": 0.71,
            "final_time": final_time,
            "chunks": chunks,
            "dofs": metadata["dofs"],
            "outlet": info["outlet"],
            "initial_mode": info["initial_mode"],
            "wall_adjoint": bool(extra.get("wall_adjoint", False)),
        },
        "error": {key: final[key] for key in ("u", "v", "velocity")},
        "pressure_dynamic_relative": final.get("pressure_dynamic"),
        "steady_residual": metadata["steady_residual"],
        "steps": sum(entry["steps"] for entry in metadata["history"]),
        "wall_seconds": wall,
        "history": metadata["history"],
        "gate": GATE,
        "gate_passed": final["velocity"] < GATE,
    }
    if name == "primary":
        # store the nodal field for the figures: re-running the march inside
        # the summary job would double the cost of the case
        from tensordg.kovasznay import _quadrature_grid

        grid = _quadrature_grid((ncells, ncells), degree, domain)
        nodal = torch.einsum("xyklv,ak,bl->xyabv", coeff, grid["basis"], grid["basis"])
        rho = nodal[..., 0]
        exact = kovasznay_exact(grid["x"], grid["y"], DEFAULT_REYNOLDS, wavenumber)
        np.savez(
            os.path.join(OUT, "fields_primary.npz"),
            x=grid["x"].numpy(), y=grid["y"].numpy(),
            u=(nodal[..., 1] / rho).numpy(), v=(nodal[..., 2] / rho).numpy(),
            u_exact=exact["u"].numpy(), v_exact=exact["v"].numpy(),
        )
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as handle:
        json.dump(record, handle, indent=1)
    print(
        f"{name:14s} p={degree} n={ncells:2d} c={sound:5.1f} M={info['mach']:.4f} "
        f"|u|rel={final['u']:.3e} |v|rel={final['v']:.3e} "
        f"vec={final['velocity']:.3e} residual={metadata['steady_residual']:.2e} "
        f"steps={record['steps']} {wall:.0f}s -> {path}"
    )
    return record


def _load(name):
    path = os.path.join(OUT, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _nodal_fields(record_name):
    """Nodal velocity field of the primary run, as saved by ``_case``."""
    archive = np.load(os.path.join(OUT, f"fields_{record_name}.npz"))
    return {key: archive[key] for key in archive.files}


def _order_table():
    """Observed orders of accuracy from the mesh/degree ladder."""
    rows = []
    for degree, meshes in ((1, (12, 24)), (2, (12, 24)), (3, (12, 24))):
        errors = []
        for mesh in meshes:
            record = _load(f"p{degree}_{mesh}")
            if record is not None:
                errors.append((mesh, record["error"]["velocity"]))
        if len(errors) == 2:
            (h1, e1), (h2, e2) = errors
            observed = math.log(e1 / e2) / math.log(h1 / h2)
            rows.append({"degree": degree, "errors": errors, "observed_order": observed})
    return rows


def _summary():
    os.makedirs(FIGS, exist_ok=True)
    summary = {
        "case": "CASE-DG-KOVASZNAY-NS",
        "reference": "Kovasznay 1948 closed form (verified by kovasznay_residual)",
        "gate": GATE,
        "analytic_residual": kovasznay_residual(),
        "rows": {},
    }
    for name in RUNS:
        record = _load(name)
        if record is not None:
            summary["rows"][name] = {
                "degree": record["config"]["degree"],
                "ncells": record["config"]["ncells"],
                "sound_speed": record["config"]["sound_speed"],
                "mach": record["config"]["mach"],
                "outlet": record["config"]["outlet"],
                "initial_mode": record["config"]["initial_mode"],
                "wall_adjoint": record["config"]["wall_adjoint"],
                "velocity": record["error"]["velocity"],
                "u": record["error"]["u"],
                "v": record["error"]["v"],
                "steady_residual": record["steady_residual"],
                "gate_passed": record["gate_passed"],
                "steps": record["steps"],
                "wall_seconds": record["wall_seconds"],
            }
    summary["order_table"] = _order_table()
    with open(os.path.join(OUT, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    try:
        _figures(summary)
    except Exception as exc:  # a figure must never take the numbers down
        print("figure generation failed:", repr(exc))
    print(json.dumps(summary["rows"], indent=1))
    print(json.dumps(summary["order_table"], indent=1))


def _figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # ---- figure 1: field comparison + pointwise error (primary case) --------
    if _load("primary") is not None:
        fields = _nodal_fields("primary")
        nx, ny = fields["u"].shape[0], fields["u"].shape[1]
        a, b = fields["u"].shape[2], fields["u"].shape[3]

        def flat(array):
            """(nx, ny, a, b) quadrature grid -> plottable (nx*a, ny*b)."""
            return array.transpose(0, 2, 1, 3).reshape(nx * a, ny * b)

        fields = {key: flat(value) for key, value in fields.items()}
        x, y = fields["x"], fields["y"]
        du = fields["u"] - fields["u_exact"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        levels = np.linspace(-0.7, 2.7, 25)
        contour = axes[0].contourf(x, y, fields["u"], levels=levels, cmap="viridis")
        axes[0].set_title("DG P3 24x24: $u$")
        axes[1].contourf(x, y, fields["u_exact"], levels=levels, cmap="viridis")
        axes[1].set_title("Kovasznay 1948: $u$")
        error_levels = np.linspace(-1, 1, 21) * max(abs(du).max(), 1e-12)
        filled = axes[2].contourf(x, y, du, levels=error_levels, cmap="RdBu_r")
        axes[2].set_title("$u_{DG} - u_{exact}$ (max %.2e)" % abs(du).max())
        for axis in axes:
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_aspect("equal")
        fig.colorbar(contour, ax=axes[:2], shrink=0.85)
        fig.colorbar(filled, ax=axes[2], shrink=0.85)
        fig.suptitle("Steady Kovasznay flow, Re = 40 (compressible NS DG/BR1, low Mach)")
        fig.savefig(os.path.join(FIGS, "kovasznay_fields.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

        # ---- figure 2: steady-state evidence, error and residual vs time ----
        history = _load("primary")["history"]
        times = [entry["time"] for entry in history]
        fig, axis = plt.subplots(1, 2, figsize=(11, 4))
        axis[0].semilogy(times, [entry["velocity"] for entry in history], "o-",
                         label="relative $L_2$ velocity error")
        axis[0].set_xlabel("time")
        axis[0].set_ylabel("error")
        axis[0].grid(True, which="both", alpha=0.3)
        axis[0].legend()
        axis2 = axis[1]
        axis2.semilogy(times, [entry["residual"] for entry in history], "s-",
                       color="tab:red")
        axis2.set_xlabel("time")
        axis2.set_ylabel("relative steady residual")
        axis2.grid(True, which="both", alpha=0.3)
        fig.suptitle("Explicit march converges to the discrete steady state")
        fig.savefig(os.path.join(FIGS, "kovasznay_steady_state.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- figure 3: mesh/degree ladder, Mach sweep ---------------------------
    ladder = summary["order_table"]
    mach_rows = [
        (row["mach"], row["velocity"], name)
        for name, row in summary["rows"].items()
        if name in ("mach030", "primary", "mach120")
    ]
    if ladder or mach_rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for entry in ladder:
            meshes = [mesh for mesh, _ in entry["errors"]]
            errors = [err for _, err in entry["errors"]]
            axes[0].loglog(meshes, errors, "o-",
                           label="P%d (order %.2f)" % (entry["degree"],
                                                      entry["observed_order"]))
        axes[0].set_xlabel("cells per direction")
        axes[0].set_ylabel("relative $L_2$ velocity error")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()
        axes[0].set_title("Mesh ladder (Re = 40, $M \\to 0$ embedding)")
        if mach_rows:
            mach_rows.sort()
            machs = [row[0] for row in mach_rows]
            axes[1].loglog([m * m for m in machs], [row[1] for row in mach_rows],
                           "o-", label="measured")
            reference = [row[1] for row in mach_rows][len(mach_rows) // 2]
            axes[1].loglog([m * m for m in machs],
                           [reference * (m / machs[len(machs) // 2]) ** 2
                            for m in machs], "k--", label="$\\propto M^2$")
            axes[1].set_xlabel("$M^2$ (compressible embedding)")
            axes[1].set_ylabel("relative $L_2$ velocity error")
            axes[1].grid(True, which="both", alpha=0.3)
            axes[1].legend()
            axes[1].set_title("Mach-number sensitivity")
        fig.savefig(os.path.join(FIGS, "kovasznay_convergence.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
    print("figures written to", FIGS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default="primary",
                        help="comma separated run names, or 'summary'/'all'")
    args = parser.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    names = [part.strip() for part in args.job.split(",") if part.strip()]
    if names == ["all"]:
        names = list(RUNS) + ["summary"]
    for name in names:
        if name == "summary":
            _summary()
        elif name in RUNS:
            _case(name)
        else:
            parser.error(f"unknown job {name!r}; known: {sorted(RUNS)} + summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())