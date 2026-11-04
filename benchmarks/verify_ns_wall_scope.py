"""Scope audit of the ``ns_dg_2d`` Dirichlet-wall limitation (m2589 claim).

m2589 (``docs/reports/tensordg-ns-wall-verification.md``) reported three
defects for the strong-form BR1/SIPG wall path: (i) the wall heat flux is
first order in ``h``, (ii) the exact analytic steady state leaves an ``O(1)``
residual that does not vanish under refinement, (iii) the wall-bounded
transient diverges.  All three were measured **only at degree 1**.

This script re-measures the same three criteria at degree 1, 2 and 3, and adds
two discriminants the earlier round did not have:

* whether the ``O(1)`` residual is an independent inconsistency or merely the
  flux error divided by ``h`` (the residual is a divergence, so a flux error
  ``eps`` produces a residual of size ``eps / h``);
* whether the transient failure depends on the polynomial degree, i.e. whether
  raising ``p`` changes anything (a representability defect would) or not (an
  instability of the wall closure would not).

Every number is written to ``validation/ns_dg_2d_wall/scope_audit.json`` and
the figure is generated from that JSON (zero hand-copied digits).

Usage
-----
    PYTHONPATH=. python -m benchmarks.verify_ns_wall_scope
    PYTHONPATH=. python -m benchmarks.verify_ns_wall_scope --skip-transient
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import tensordg.ns_dg_2d as ns
from tools.diag_ns_wall import (
    ANALYTIC_E_FLUX_BOTTOM,
    ANALYTIC_MX_FLUX,
    KAPPA,
    MU,
    march,
    build_state,
    wall_fluxes,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "ns_dg_2d_wall"
FIG = ROOT / "docs" / "figures"
DEGREES = (1, 2, 3)
REFINEMENTS = (8, 16, 32, 64)


# ---------------------------------------------------------------------------
# 1. residual of the exact analytic steady state
# ---------------------------------------------------------------------------
def residual_table(refinements=(8, 16, 32), degrees=DEGREES):
    rows = []
    for degree in degrees:
        for ny in refinements:
            d = build_state((2, ny), degree)
            penalty = 2.0 * (degree + 1.0) ** 2
            rhs = ns._rhs_ns(
                d["coeff"], d["basis"], d["derivative"], d["weights"], d["signs"],
                d["dx"], d["dy"], (True, False), d["boundary"], MU, KAPPA, penalty,
            )
            rows.append(dict(
                degree=degree, ny=ny, h=d["dy"],
                rhs_max=float(rhs.abs().max()),
                rhs_energy=float(rhs[..., 3].abs().max()),
                wall_row_max=float(rhs[:, [0, -1]].abs().max()),
            ))
    # observed order of the residual per degree
    for degree in degrees:
        series = [r for r in rows if r["degree"] == degree]
        for previous, current in zip(series, series[1:]):
            ratio = previous["rhs_max"] / current["rhs_max"]
            current["order"] = float(torch.log2(torch.tensor(ratio)))
    return rows


# ---------------------------------------------------------------------------
# 2. wall heat flux error and its observed order
# ---------------------------------------------------------------------------
def flux_table(refinements=REFINEMENTS, degrees=DEGREES):
    rows = []
    for degree in degrees:
        series = []
        for ny in refinements:
            f = wall_fluxes((2, ny), degree)
            error_energy = abs(f["energy_bottom"] - ANALYTIC_E_FLUX_BOTTOM)
            error_mx = abs(f["mx_top"] - ANALYTIC_MX_FLUX)
            row = dict(degree=degree, ny=ny, h=1.0 / ny,
                       energy_bottom=f["energy_bottom"],
                       energy_error=error_energy,
                       energy_relative_percent=100.0 * error_energy
                       / abs(ANALYTIC_E_FLUX_BOTTOM),
                       mx_error=error_mx)
            series.append(row)
        for previous, current in zip(series, series[1:]):
            current["order_energy"] = float(torch.log2(
                torch.tensor(previous["energy_error"] / current["energy_error"])))
            current["order_mx"] = float(torch.log2(
                torch.tensor(previous["mx_error"] / current["mx_error"])))
        rows.extend(series)
    return rows


# ---------------------------------------------------------------------------
# 3. is the O(1) residual an independent inconsistency?
# ---------------------------------------------------------------------------
def residual_versus_flux(residual_rows, flux_rows):
    """residual is a divergence: a flux error ``eps`` gives residual ~ eps/h.

    If the identity holds, the 'residual does not vanish' finding is a
    restatement of the flux order, not a separate defect.
    """
    table = []
    for residual in residual_rows:
        if residual["degree"] not in (1, 2):
            continue
        match = [f for f in flux_rows
                 if f["degree"] == residual["degree"] and f["ny"] == residual["ny"]]
        if not match:
            continue
        flux_error = match[0]["energy_error"]
        table.append(dict(
            degree=residual["degree"], ny=residual["ny"], h=residual["h"],
            residual_energy=residual["rhs_energy"],
            flux_error_energy=flux_error,
            residual_times_h=residual["rhs_energy"] * residual["h"],
            ratio_residual_h_over_flux_error=(residual["rhs_energy"] * residual["h"]
                                              / flux_error),
        ))
    return table


# ---------------------------------------------------------------------------
# 4. transient behaviour: steady-state march and impulsive start
# ---------------------------------------------------------------------------
def march_table(degrees=DEGREES, max_steps=120, ncells=(2, 8)):
    rows = []
    for degree in degrees:
        started = time.time()
        history, failure = march(ncells=ncells, degree=degree, max_steps=max_steps)
        rows.append(dict(
            degree=degree, ncells=list(ncells), max_steps=max_steps,
            survived=failure is None, failure=failure,
            wall_seconds=round(time.time() - started, 2),
        ))
    return rows


def impulse_table(degrees=DEGREES, ncells=(2, 8), final_time=0.1,
                  cfl_viscous=1.0):
    rows = []
    for degree in degrees:
        started = time.time()
        try:
            _, coeff, meta = ns.solve_couette_dg_2d(
                ncells=ncells, degree=degree, mu=MU, final_time=final_time,
                cfl_viscous=cfl_viscous,
            )
            rows.append(dict(degree=degree, final_time=final_time,
                             reached=True, finite=bool(torch.isfinite(coeff).all()),
                             cfl_viscous=cfl_viscous,
                             steps=meta.get("steps"),
                             wall_seconds=round(time.time() - started, 2)))
        except Exception as exc:  # noqa: BLE001 - the failure mode is the datum
            rows.append(dict(degree=degree, final_time=final_time,
                             reached=False, cfl_viscous=cfl_viscous,
                             error=type(exc).__name__,
                             message=str(exc).splitlines()[0][:120],
                             wall_seconds=round(time.time() - started, 2)))
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(residual_rows, flux_rows, march_rows, impulse_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    colors = {1: "tab:red", 2: "tab:blue", 3: "tab:green"}

    ax = axes[0]
    for degree in DEGREES:
        series = [r for r in flux_rows if r["degree"] == degree]
        if not series:
            continue
        ax.loglog([r["h"] for r in series], [max(r["energy_error"], 1e-17) for r in series],
                  "o-", color=colors[degree], label=f"P{degree} measured")
        h = [r["h"] for r in series]
        ref = series[0]["energy_error"]
        ax.loglog(h, [ref * (x / h[0]) ** degree for x in h], "--",
                  color=colors[degree], alpha=0.6, label=f"$\\propto h^{degree}$")
    ax.set_xlabel("h = 1/ny")
    ax.set_ylabel("|wall heat flux error|")
    ax.set_title("(a) wall flux order is p, not 1\n(P1 first order, P2 second, P3 exact)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for degree in DEGREES:
        series = [r for r in residual_rows if r["degree"] == degree]
        if not series:
            continue
        ax.loglog([r["h"] for r in series], [r["rhs_max"] for r in series],
                  "o-", color=colors[degree], label=f"P{degree}")
    ax.set_xlabel("h = 1/ny")
    ax.set_ylabel("max |RHS(exact steady state)|")
    ax.set_title("(b) exact-state residual: flat at P1,\ndecaying at P2/P3")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    labels, values, tint = [], [], []
    for row in impulse_rows:
        labels.append(f"impulse P{row['degree']}")
        values.append(1.0 if row["reached"] else 0.0)
        tint.append(colors[row["degree"]])
    for row in march_rows:
        labels.append(f"steady march P{row['degree']}")
        values.append(1.0 if row["survived"] else 0.0)
        tint.append(colors[row["degree"]])
    position = range(len(labels))
    ax.barh(list(position), values, color=tint, alpha=0.75)
    ax.set_yticks(list(position))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("1 = completed / 0 = non-physical")
    ax.set_title("(c) transient: impulse fails at every p,\nsteady state only at P1")
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("TensorDG ns_dg_2d Dirichlet wall: order + stability scope audit (m2786)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.mkdir(parents=True, exist_ok=True)
    target = FIG / "ns_dg_2d_wall_scope.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-transient", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    torch.set_num_threads(2)
    residual_rows = residual_table()
    flux_rows = flux_table()
    identity = residual_versus_flux(residual_rows, flux_rows)
    if args.skip_transient:
        march_rows, impulse_rows = [], []
    else:
        march_rows = march_table()
        impulse_rows = impulse_table()

    payload = dict(
        mission="m2786",
        question="is the ns_dg_2d wall defect an order defect, an inconsistency, "
                 "or a transient instability?",
        constants=dict(mu=MU, kappa=KAPPA, analytic_energy_flux=ANALYTIC_E_FLUX_BOTTOM,
                       analytic_mx_flux=ANALYTIC_MX_FLUX),
        residual=residual_rows,
        flux=flux_rows,
        residual_versus_flux=identity,
        steady_march=march_rows,
        impulsive_start=impulse_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "scope_audit.json"
    json_path.write_text(json.dumps(payload, indent=2))
    figure = make_figure(residual_rows, flux_rows, march_rows, impulse_rows)

    print(f"wrote {json_path}")
    print(f"wrote {figure}")
    for row in flux_rows:
        if row["ny"] in (8, 64):
            print(f"P{row['degree']} ny={row['ny']:3d} energy_err="
                  f"{row['energy_error']:.3e} ({row['energy_relative_percent']:.3f}%)"
                  f" order={row.get('order_energy')}")
    for row in residual_rows:
        print(f"P{row['degree']} ny={row['ny']:3d} residual={row['rhs_max']:.3e}"
              f" order={row.get('order')}")
    for row in march_rows:
        print(f"steady march P{row['degree']}: survived={row['survived']}")
    for row in impulse_rows:
        print(f"impulse P{row['degree']}: reached={row['reached']} "
              f"{row.get('error', '')}")


if __name__ == "__main__":
    main()